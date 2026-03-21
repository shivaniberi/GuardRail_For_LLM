"""
analysis.py
-----------
Weak spot analysis for guardrail evaluation results.

Usage:
    from evaluation.analysis import weak_spot_report
    weak_spot_report("logs/guardrail_logs_qwen2.5.parquet")

Or run directly:
    python -m evaluation.analysis logs/guardrail_logs_qwen2.5.parquet
"""

import sys
import pandas as pd

SCORE_COLS = ["safety", "grounding", "hallucination", "bias"]


def weak_spot_report(parquet_path: str, top_n: int = 10) -> None:
    """
    Print a structured weak spot report from a guardrail evaluation log.

    Args:
        parquet_path: Path to the .parquet file produced by eval_runner.
        top_n:        Number of worst cases to surface per axis.
    """
    df = pd.read_parquet(parquet_path)
    available_scores = [c for c in SCORE_COLS if c in df.columns]

    _section("1. Dataset summary")
    print(f"Total test cases : {len(df)}")
    print(f"Categories       : {df['category'].value_counts().to_dict()}")
    verdict_counts = df["verdict"].value_counts()
    print(f"Verdicts         : {verdict_counts.to_dict()}")
    fail_rate = (df["verdict"] == "FAIL").mean()
    print(f"Overall FAIL rate: {fail_rate:.1%}")

    _section("2. Score distributions (all cases)")
    print(df[available_scores].describe().round(3).to_string())

    _section("3. Mean scores by category")
    print(
        df.groupby("category")[available_scores]
        .mean()
        .round(3)
        .to_string()
    )

    _section("4. FAIL rate by category")
    fail_by_cat = (
        df.groupby("category")
        .apply(lambda g: (g["verdict"] == "FAIL").mean())
        .rename("fail_rate")
        .round(3)
    )
    print(fail_by_cat.to_string())

    _section(f"5. Worst {top_n} cases — lowest safety score")
    cols = ["input", "category", "verdict", "safety", "judge_reasoning"]
    worst_safety = df.nsmallest(top_n, "safety")[
        [c for c in cols if c in df.columns]
    ]
    _print_cases(worst_safety)

    if "hallucination" in df.columns:
        _section(f"6. Worst {top_n} cases — highest hallucination score")
        worst_halluc = df.nlargest(top_n, "hallucination")[
            [c for c in cols if c in df.columns]
        ]
        _print_cases(worst_halluc)

    if "bias" in df.columns:
        _section(f"7. Worst {top_n} cases — highest bias score")
        worst_bias = df.nlargest(top_n, "bias")[
            [c for c in cols if c in df.columns]
        ]
        _print_cases(worst_bias)

    _section("8. Low-confidence verdicts (confidence < 0.6)")
    if "confidence" in df.columns:
        low_conf = df[df["confidence"] < 0.6].sort_values("confidence")
        print(f"Count: {len(low_conf)}")
        if not low_conf.empty:
            _print_cases(
                low_conf[["input", "category", "verdict", "confidence"]].head(top_n)
            )
    else:
        print("No confidence column found.")

    _section("Summary of weak spots")
    if available_scores:
        means = df[available_scores].mean()
        worst_axis = means.idxmin() if "safety" in means else means.idxmax()
        print(f"Lowest mean score axis : {worst_axis} ({means[worst_axis]:.3f})")
    worst_cat = fail_by_cat.idxmax()
    print(f"Highest FAIL rate category : {worst_cat} ({fail_by_cat[worst_cat]:.1%})")
    print(f"Recommended action: Review {worst_cat} threshold and training data.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _print_cases(df: pd.DataFrame) -> None:
    pd.set_option("display.max_colwidth", 80)
    pd.set_option("display.width", 120)
    print(df.to_string(index=False))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "logs/guardrail_logs_qwen2.5.parquet"
    weak_spot_report(path)
