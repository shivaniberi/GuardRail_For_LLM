"""
runners/run_eval.py
-------------------
Entrypoint to run the full LLM-as-a-Judge evaluation pipeline.

Usage:
    # Full run
    python runners/run_eval.py

    # Smoke test (first 20 cases only)
    python runners/run_eval.py --max-cases 20

    # Extra debate rounds
    python runners/run_eval.py --rounds 2
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import argparse
from evaluation.eval_runner import run_eval
from evaluation.analysis import weak_spot_report

DEFAULT_OUTPUT = "logs/guardrail_logs_qwen2.5.parquet"


def parse_args():
    parser = argparse.ArgumentParser(description="Run guardrail LLM-as-a-Judge evaluation")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Output parquet path (default: logs/guardrail_logs_qwen2.5.parquet)")
    parser.add_argument("--max-cases", type=int, default=None,
                        help="Cap number of test cases (omit for full run)")
    parser.add_argument("--rounds", type=int, default=1,
                        help="Debate rounds in ChainOfDebateJudge (default: 1)")
    parser.add_argument("--model", default="qwen2.5:7b",
                        help="Ollama model tag for judge (default: qwen2.5:7b)")
    parser.add_argument("--skip-analysis", action="store_true",
                        help="Skip weak spot report after eval")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"Starting evaluation → {args.output}")
    print(f"  model={args.model}  rounds={args.rounds}  max_cases={args.max_cases}\n")

    run_eval(
        output_path=args.output,
        max_cases=args.max_cases,
        judge_rounds=args.rounds,
        judge_model=args.model,
    )

    if not args.skip_analysis:
        print("\n--- Weak spot report ---")
        weak_spot_report(args.output)
