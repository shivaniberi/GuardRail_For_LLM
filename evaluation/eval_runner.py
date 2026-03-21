"""
eval_runner.py
--------------
Loads evaluation datasets, runs each prompt through the existing guardrail
pipeline, then scores the results with ChainOfDebateJudge.

Output: a Parquet file at `output_path` (default: logs/guardrail_logs_qwen2.5.parquet)
"""

import os
import pandas as pd
from datasets import load_dataset

# Adjust this import to match the actual path inside your repo
from core.guardrail import Phi3GuardrailSystem, GuardrailConfig
from evaluation.judge import ChainOfDebateJudge


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------

def _safety_cases() -> list[dict]:
    """Binary block/pass cases from a safety-prompts dataset."""
    cases = []
    try:
        ds = load_dataset("allenai/safety-prompts", split="test")
        for row in ds:
            cases.append(
                {
                    "input": row["prompt"],
                    "expected": "BLOCK",
                    "category": "safety",
                    "context": [],
                }
            )
    except Exception as e:
        print(f"[WARN] Could not load safety-prompts: {e}")
    return cases


def _truthfulqa_cases() -> list[dict]:
    """Hallucination detection cases from TruthfulQA."""
    cases = []
    try:
        ds = load_dataset("truthful_qa", "generation", split="validation")
        for row in ds:
            cases.append(
                {
                    "input": row["question"],
                    "expected": row["best_answer"],
                    "category": "hallucination",
                    "context": [],
                }
            )
    except Exception as e:
        print(f"[WARN] Could not load truthful_qa: {e}")
    return cases


def _winobias_cases() -> list[dict]:
    """Gender-bias fairness cases from WinoBias."""
    cases = []
    try:
        ds = load_dataset("wino_bias", split="test")
        for row in ds:
            cases.append(
                {
                    "input": row["sentence"],
                    "expected": str(row.get("label", "")),
                    "category": "bias",
                    "context": [],
                }
            )
    except Exception as e:
        print(f"[WARN] Could not load wino_bias: {e}")
    return cases


def build_test_cases() -> list[dict]:
    """Combine all evaluation datasets into a single list of test cases."""
    cases = []
    cases.extend(_safety_cases())
    cases.extend(_truthfulqa_cases())
    cases.extend(_winobias_cases())
    print(f"[INFO] Total test cases: {len(cases)}")
    return cases


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_eval(
    output_path: str = "logs/guardrail_logs_qwen2.5.parquet",
    max_cases: int | None = None,
    judge_rounds: int = 1,
    judge_model: str = "qwen2.5:7b",
) -> pd.DataFrame:
    """
    Run the full evaluation pipeline.

    Args:
        output_path:  Where to save the resulting Parquet file.
        max_cases:    Cap the number of test cases (useful for quick smoke tests).
        judge_rounds: Number of debate rounds in ChainOfDebateJudge.
        judge_model:  Ollama model tag for the judge.

    Returns:
        DataFrame of all results (also saved to output_path).
    """
    # --- Setup ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    config = GuardrailConfig(
        confidence_threshold=0.7,
        toxicity_threshold=0.5,
        enable_rag=True,
    )
    system = Phi3GuardrailSystem(config)
    judge = ChainOfDebateJudge(model=judge_model, rounds=judge_rounds)

    test_cases = build_test_cases()
    if max_cases is not None:
        test_cases = test_cases[:max_cases]

    records = []
    for i, tc in enumerate(test_cases):
        print(f"[{i+1}/{len(test_cases)}] category={tc['category']} | {tc['input'][:60]!r}")

        # --- Run through your guardrail pipeline ---
        try:
            guarded = system.generate_with_guardrails(tc["input"])
        except Exception as e:
            print(f"  [ERROR] Guardrail failed: {e}")
            continue

        bundle = {
            "input": tc["input"],
            "raw_response": guarded.get("raw_response", ""),
            "guarded_response": guarded.get("response", ""),
            "retrieval_context": guarded.get("context", []),
            "expected": tc["expected"],
            "category": tc["category"],
        }

        # --- Run judge ---
        try:
            verdict = judge.judge(bundle)
        except Exception as e:
            print(f"  [ERROR] Judge failed: {e}")
            continue

        records.append(
            {
                **bundle,
                **verdict.get("final_scores", {}),
                "verdict": verdict.get("verdict", "UNKNOWN"),
                "judge_reasoning": verdict.get("reasoning", ""),
                "confidence": verdict.get("confidence", 0.0),
            }
        )

    df = pd.DataFrame(records)
    df.to_parquet(output_path, index=False)
    print(f"\n[DONE] Saved {len(df)} records → {output_path}")
    return df
