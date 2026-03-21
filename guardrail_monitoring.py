"""
Guardrail Monitoring & Evaluation Dashboard
============================================
Real-time monitoring and visualization of guardrail performance
- Request/response logging
- Performance metrics tracking
- Failure analysis
- A/B testing support
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


class GuardrailMonitor:
    """
    Monitor and track guardrail system performance
    """
    
    def __init__(self, log_file: str = "guardrail_logs.parquet"):
        self.log_file = log_file
        self.logs = []
        self.load_logs()
    
    def load_logs(self):
        """Load existing logs"""
        try:
            self.logs_df = pd.read_parquet(self.log_file)
            print(f"✓ Loaded {len(self.logs_df)} log entries")
        except FileNotFoundError:
            self.logs_df = pd.DataFrame()
            print("No existing logs found")
    
    def calculate_metrics(self) -> Dict:
        """Calculate key performance metrics"""
        if self.logs_df.empty:
            return {}
        
        metrics = {}
        
        # Basic stats
        metrics['total_requests'] = len(self.logs_df)
        
        # Input validation metrics
        if 'guardrails' in self.logs_df.columns:
            input_valid_count = sum(
                log.get('input', {}).get('valid', True) 
                for log in self.logs_df['guardrails']
                if isinstance(log, dict)
            )
            metrics['input_pass_rate'] = input_valid_count / metrics['total_requests']
            metrics['input_block_rate'] = 1 - metrics['input_pass_rate']
            
            # Output validation metrics
            output_valid_count = sum(
                log.get('output', {}).get('valid', True)
                for log in self.logs_df['guardrails']
                if isinstance(log, dict) and 'output' in log
            )
            metrics['output_pass_rate'] = output_valid_count / metrics['total_requests']
            metrics['output_fail_rate'] = 1 - metrics['output_pass_rate']
        
        # Response metrics
        if 'response' in self.logs_df.columns:
            blocked_count = sum(
                str(resp).startswith('[BLOCKED]') 
                for resp in self.logs_df['response']
            )
            metrics['blocked_responses'] = blocked_count
            metrics['block_rate'] = blocked_count / metrics['total_requests']
        
        # RAG usage
        if 'metadata' in self.logs_df.columns:
            rag_used_count = sum(
                meta.get('rag_used', False)
                for meta in self.logs_df['metadata']
                if isinstance(meta, dict)
            )
            metrics['rag_usage_rate'] = rag_used_count / metrics['total_requests']
        
        return metrics
    
    def get_failure_analysis(self) -> pd.DataFrame:
        """Analyze failed guardrail checks"""
        if self.logs_df.empty:
            return pd.DataFrame()
        
        failures = []
        
        for idx, row in self.logs_df.iterrows():
            guardrails = row.get('guardrails', {})
            
            # Input failures
            if isinstance(guardrails, dict) and 'input' in guardrails:
                input_checks = guardrails['input'].get('checks', {})
                for check_name, check_result in input_checks.items():
                    if isinstance(check_result, dict) and not check_result.get('passed', True):
                        failures.append({
                            'timestamp': row.get('timestamp'),
                            'stage': 'input',
                            'check': check_name,
                            'reason': check_result.get('reason', 'Unknown'),
                            'prompt': row.get('prompt', '')[:100]
                        })
            
            # Output failures
            if isinstance(guardrails, dict) and 'output' in guardrails:
                output_checks = guardrails['output'].get('checks', {})
                for check_name, check_result in output_checks.items():
                    if isinstance(check_result, dict) and not check_result.get('passed', True):
                        failures.append({
                            'timestamp': row.get('timestamp'),
                            'stage': 'output',
                            'check': check_name,
                            'score': check_result.get('score', 'N/A'),
                            'response': str(row.get('response', ''))[:100]
                        })
        
        return pd.DataFrame(failures)
    
    def get_check_statistics(self) -> pd.DataFrame:
        """Get statistics for each guardrail check"""
        if self.logs_df.empty:
            return pd.DataFrame()
        
        check_stats = defaultdict(lambda: {'passed': 0, 'failed': 0, 'scores': []})
        
        for _, row in self.logs_df.iterrows():
            guardrails = row.get('guardrails', {})
            
            # Process input checks
            if isinstance(guardrails, dict) and 'input' in guardrails:
                for check_name, check_result in guardrails['input'].get('checks', {}).items():
                    if isinstance(check_result, dict):
                        if check_result.get('passed', True):
                            check_stats[f'input_{check_name}']['passed'] += 1
                        else:
                            check_stats[f'input_{check_name}']['failed'] += 1
                        
                        if 'score' in check_result:
                            check_stats[f'input_{check_name}']['scores'].append(check_result['score'])
            
            # Process output checks
            if isinstance(guardrails, dict) and 'output' in guardrails:
                for check_name, check_result in guardrails['output'].get('checks', {}).items():
                    if isinstance(check_result, dict):
                        if check_result.get('passed', True):
                            check_stats[f'output_{check_name}']['passed'] += 1
                        else:
                            check_stats[f'output_{check_name}']['failed'] += 1
                        
                        if 'score' in check_result:
                            check_stats[f'output_{check_name}']['scores'].append(check_result['score'])
        
        # Convert to DataFrame
        stats_list = []
        for check_name, stats in check_stats.items():
            total = stats['passed'] + stats['failed']
            stats_list.append({
                'check': check_name,
                'passed': stats['passed'],
                'failed': stats['failed'],
                'total': total,
                'pass_rate': stats['passed'] / total if total > 0 else 0,
                'avg_score': np.mean(stats['scores']) if stats['scores'] else None,
                'std_score': np.std(stats['scores']) if stats['scores'] else None
            })
        
        return pd.DataFrame(stats_list).sort_values('pass_rate')
    
    def plot_metrics_over_time(self, window: str = '1H'):
        """Plot key metrics over time"""
        if self.logs_df.empty:
            print("No data to plot")
            return
        
        # Convert timestamp to datetime
        self.logs_df['timestamp'] = pd.to_datetime(self.logs_df['timestamp'])
        self.logs_df = self.logs_df.sort_values('timestamp')
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Guardrail System Performance Over Time', fontsize=16)
        
        # 1. Request volume over time
        request_counts = self.logs_df.set_index('timestamp').resample(window).size()
        axes[0, 0].plot(request_counts.index, request_counts.values, marker='o')
        axes[0, 0].set_title('Request Volume')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Number of Requests')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Block rate over time
        def calculate_block_rate(group):
            blocked = sum(str(r).startswith('[BLOCKED]') for r in group['response'])
            return blocked / len(group) if len(group) > 0 else 0
        
        block_rates = self.logs_df.set_index('timestamp').resample(window).apply(calculate_block_rate)
        axes[0, 1].plot(block_rates.index, block_rates.values, marker='o', color='red')
        axes[0, 1].set_title('Block Rate')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Block Rate')
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. RAG usage rate
        def calculate_rag_rate(group):
            rag_used = sum(m.get('rag_used', False) for m in group['metadata'] if isinstance(m, dict))
            return rag_used / len(group) if len(group) > 0 else 0
        
        rag_rates = self.logs_df.set_index('timestamp').resample(window).apply(calculate_rag_rate)
        axes[1, 0].plot(rag_rates.index, rag_rates.values, marker='o', color='green')
        axes[1, 0].set_title('RAG Usage Rate')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('RAG Usage Rate')
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Check failure distribution
        failures_df = self.get_failure_analysis()
        if not failures_df.empty:
            check_counts = failures_df['check'].value_counts()
            axes[1, 1].barh(check_counts.index, check_counts.values)
            axes[1, 1].set_title('Guardrail Check Failures')
            axes[1, 1].set_xlabel('Number of Failures')
        else:
            axes[1, 1].text(0.5, 0.5, 'No failures recorded', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.savefig('guardrail_metrics.png', dpi=150, bbox_inches='tight')
        print("✓ Metrics plot saved to guardrail_metrics.png")
    
    def generate_report(self) -> str:
        """Generate comprehensive monitoring report"""
        report = []
        report.append("="*80)
        report.append("GUARDRAIL SYSTEM MONITORING REPORT")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Overall metrics
        metrics = self.calculate_metrics()
        report.append("OVERALL METRICS")
        report.append("-"*80)
        for key, value in metrics.items():
            if isinstance(value, float):
                report.append(f"{key:.<50} {value:.2%}")
            else:
                report.append(f"{key:.<50} {value}")
        report.append("")
        
        # Check statistics
        check_stats = self.get_check_statistics()
        if not check_stats.empty:
            report.append("CHECK STATISTICS")
            report.append("-"*80)
            report.append(check_stats.to_string())
            report.append("")
        
        # Failure analysis
        failures = self.get_failure_analysis()
        if not failures.empty:
            report.append("TOP FAILURE REASONS")
            report.append("-"*80)
            
            # Input failures
            input_failures = failures[failures['stage'] == 'input']
            if not input_failures.empty:
                report.append("\nInput Validation Failures:")
                reason_counts = input_failures['reason'].value_counts().head(5)
                for reason, count in reason_counts.items():
                    report.append(f"  - {reason}: {count}")
            
            # Output failures
            output_failures = failures[failures['stage'] == 'output']
            if not output_failures.empty:
                report.append("\nOutput Validation Failures:")
                check_counts = output_failures['check'].value_counts().head(5)
                for check, count in check_counts.items():
                    report.append(f"  - {check}: {count}")
        
        report.append("")
        report.append("="*80)
        
        return "\n".join(report)
    
    def save_report(self, filename: str = "guardrail_report.txt"):
        """Save report to file"""
        report = self.generate_report()
        with open(filename, 'w') as f:
            f.write(report)
        print(f"✓ Report saved to {filename}")
        return report


class ABTestingFramework:
    """
    A/B testing framework for comparing guardrail configurations
    """
    
    def __init__(self):
        self.experiments = {}
        self.results = {}
    
    def create_experiment(self, name: str, config_a: Dict, config_b: Dict):
        """Create new A/B test experiment"""
        self.experiments[name] = {
            'config_a': config_a,
            'config_b': config_b,
            'results_a': [],
            'results_b': []
        }
        print(f"✓ Created experiment: {name}")
    
    def log_result(self, experiment_name: str, variant: str, result: Dict):
        """Log result for specific variant"""
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment {experiment_name} not found")
        
        if variant not in ['a', 'b']:
            raise ValueError("Variant must be 'a' or 'b'")
        
        self.experiments[experiment_name][f'results_{variant}'].append(result)
    
    def analyze_experiment(self, experiment_name: str) -> Dict:
        """Analyze experiment results"""
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment {experiment_name} not found")
        
        exp = self.experiments[experiment_name]
        results_a = pd.DataFrame(exp['results_a'])
        results_b = pd.DataFrame(exp['results_b'])
        
        analysis = {
            'experiment': experiment_name,
            'sample_size_a': len(results_a),
            'sample_size_b': len(results_b)
        }
        
        # Compare key metrics
        for metric in ['latency', 'block_rate', 'confidence_score']:
            if metric in results_a.columns and metric in results_b.columns:
                analysis[f'{metric}_a_mean'] = results_a[metric].mean()
                analysis[f'{metric}_b_mean'] = results_b[metric].mean()
                analysis[f'{metric}_improvement'] = (
                    (results_b[metric].mean() - results_a[metric].mean()) / 
                    results_a[metric].mean() * 100
                )
        
        return analysis
    
    def plot_comparison(self, experiment_name: str):
        """Plot comparison between variants"""
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment {experiment_name} not found")
        
        exp = self.experiments[experiment_name]
        results_a = pd.DataFrame(exp['results_a'])
        results_b = pd.DataFrame(exp['results_b'])
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'A/B Test Results: {experiment_name}')
        
        # Plot metrics
        metrics = ['latency', 'block_rate', 'confidence_score']
        for idx, metric in enumerate(metrics):
            if metric in results_a.columns and metric in results_b.columns:
                axes[idx].boxplot([results_a[metric], results_b[metric]], 
                                 labels=['Variant A', 'Variant B'])
                axes[idx].set_title(metric.replace('_', ' ').title())
                axes[idx].set_ylabel(metric)
        
        plt.tight_layout()
        plt.savefig(f'ab_test_{experiment_name}.png', dpi=150)
        print(f"✓ Comparison plot saved to ab_test_{experiment_name}.png")


# Example Usage
if __name__ == "__main__":
    
    # Initialize monitor
    monitor = GuardrailMonitor("guardrail_logs.parquet")
    
    # Calculate metrics
    metrics = monitor.calculate_metrics()
    print("\nKey Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2%}")
        else:
            print(f"  {key}: {value}")
    
    # Get check statistics
    print("\nCheck Statistics:")
    print(monitor.get_check_statistics())
    
    # Generate and save report
    report = monitor.generate_report()
    print(report)
    monitor.save_report()
    
    # Plot metrics
    monitor.plot_metrics_over_time(window='1H')
    
    # A/B Testing example
    ab_test = ABTestingFramework()
    ab_test.create_experiment(
        name="confidence_threshold_test",
        config_a={'confidence_threshold': 0.7},
        config_b={'confidence_threshold': 0.8}
    )
    
    print("\n✓ Monitoring setup complete!")
