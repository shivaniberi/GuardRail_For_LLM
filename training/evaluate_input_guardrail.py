"""
Evaluate the ML INPUT safety classifier (MLInputGuardrail) on a test/val dataset.

What this file does:
- Loads your saved joblib bundle (input_safety_all7.joblib), which is a dict containing:
    - 'embed_model_name'  (SentenceTransformer model name)
    - 'classifier'        (sklearn classifier trained on embeddings)
- Loads a dataset from local parquet OR s3://... parquet
- Encodes text using the same SentenceTransformer used during training
- Runs predictions (and probabilities if available)
- Prints and saves:
  - confusion matrix
  - precision / recall / F1 (classification report)
  - ROC-AUC and PR-AUC (if probabilities available)
- Saves plots ("photos") as PNGs in an output folder

Outputs (in --out_dir/<timestamp>/):
- report.txt
- report.json
- confusion_matrix.png
- roc_curve.png (if proba available)
- pr_curve.png  (if proba available)
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

DEFAULT_TEXT_COL_CANDIDATES = ["prompt", "input", "input_text", "text", "QUESTION", "query"]
DEFAULT_LABEL_COL_CANDIDATES = ["label", "LABEL", "is_unsafe", "unsafe", "HATESPEECH", "target", "y"]


def _ensure_out_dir(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _detect_columns(df: pd.DataFrame, text_col: Optional[str], label_col: Optional[str]) -> Tuple[str, str]:
    if text_col and text_col in df.columns:
        chosen_text = text_col
    else:
        chosen_text = next((c for c in DEFAULT_TEXT_COL_CANDIDATES if c in df.columns), None)
        if chosen_text is None:
            raise ValueError(
                f"Could not auto-detect text column. Provide --text_col. "
                f"Available columns: {list(df.columns)}"
            )

    if label_col and label_col in df.columns:
        chosen_label = label_col
    else:
        chosen_label = next((c for c in DEFAULT_LABEL_COL_CANDIDATES if c in df.columns), None)
        if chosen_label is None:
            raise ValueError(
                f"Could not auto-detect label column. Provide --label_col. "
                f"Available columns: {list(df.columns)}"
            )

    return chosen_text, chosen_label


def _binarize_labels(y_raw: pd.Series, positive_label: Optional[str], positive_int: Optional[int]) -> np.ndarray:
    """
    Convert labels to {0,1}.
    - If labels already numeric 0/1 -> use as-is
    - Else map:
        - if --positive_label given: label == positive_label -> 1 else 0
        - elif --positive_int given: label == positive_int -> 1 else 0
        - else try common mappings: {"unsafe","harmful","yes","true",1,"toxic"} as positive
    """
    if pd.api.types.is_numeric_dtype(y_raw):
        vals = y_raw.fillna(0).astype(int).to_numpy()
        uniq = set(np.unique(vals).tolist())
        if uniq.issubset({0, 1}):
            return vals
        if positive_int is not None:
            return (vals == int(positive_int)).astype(int)
        return (vals != 0).astype(int)

    y_str = y_raw.astype(str).fillna("").str.strip().str.lower()

    if positive_label is not None:
        pl = str(positive_label).strip().lower()
        return (y_str == pl).astype(int).to_numpy()

    common_pos = {"unsafe", "harmful", "yes", "true", "1", "toxic"}
    return y_str.isin(common_pos).astype(int).to_numpy()


def _save_text(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _plot_confusion_matrix(cm: np.ndarray, out_path: str, title: str = "Confusion Matrix") -> None:
    fig = plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_roc(y_true: np.ndarray, y_score: np.ndarray, out_path: str) -> Dict[str, float]:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig = plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (AUC={roc_auc:.4f})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

    return {"roc_auc": float(roc_auc)}


def _plot_pr(y_true: np.ndarray, y_score: np.ndarray, out_path: str) -> Dict[str, float]:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    fig = plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (AP={ap:.4f})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

    return {"average_precision": float(ap)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="training/input_safety_all7.joblib")
    parser.add_argument("--data_path", required=True, help="Local parquet path or s3://... parquet")
    parser.add_argument("--text_col", default=None, help="Text column name (optional)")
    parser.add_argument("--label_col", default=None, help="Label column name (optional)")
    parser.add_argument("--positive_label", default=None, help="If label is string, which value means UNSAFE?")
    parser.add_argument("--positive_int", type=int, default=None, help="If label is int, which value means UNSAFE?")
    parser.add_argument("--out_dir", default="eval_outputs/input_guardrail")
    parser.add_argument("--batch_size", type=int, default=64, help="Embedding batch size")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = _ensure_out_dir(os.path.join(args.out_dir, ts))

    # -----------------------
    # Load model bundle
    # -----------------------
    bundle = joblib.load(args.model_path)
    if not isinstance(bundle, dict) or "classifier" not in bundle or "embed_model_name" not in bundle:
        raise ValueError(
            "Expected a dict with keys: 'embed_model_name' and 'classifier'. "
            f"Got: {type(bundle)} keys={list(bundle.keys()) if isinstance(bundle, dict) else None}"
        )

    embed_model_name = bundle["embed_model_name"]
    clf = bundle["classifier"]

    print(f"Loaded model bundle from: {args.model_path}")
    print(f"Embedding model: {embed_model_name}")
    print(f"Classifier type: {type(clf)}")

    embedder = SentenceTransformer(embed_model_name, device="cpu")

    # -----------------------
    # Load data
    # -----------------------
    df = pd.read_parquet(args.data_path)
    text_col, label_col = _detect_columns(df, args.text_col, args.label_col)

    X_text = df[text_col].astype(str).fillna("").tolist()
    y_true = _binarize_labels(df[label_col], args.positive_label, args.positive_int)

    # -----------------------
    # Embed -> Predict
    # -----------------------
    X_emb = embedder.encode(
        X_text,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=args.batch_size,
        normalize_embeddings=True,
    ).astype(np.float32)

    y_pred = clf.predict(X_emb)

    # Normalize y_pred to {0,1}
    if not np.issubdtype(np.array(y_pred).dtype, np.number):
        y_pred = pd.Series(y_pred).astype(str).str.strip().str.lower()
        y_pred = y_pred.isin({"unsafe", "harmful", "yes", "true", "1", "toxic"}).astype(int).to_numpy()
    else:
        y_pred = np.array(y_pred).astype(int)
        if set(np.unique(y_pred).tolist()) - {0, 1}:
            y_pred = (y_pred != 0).astype(int)

    # -----------------------
    # Reports + Plots
    # -----------------------
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report_str = classification_report(y_true, y_pred, digits=4)

    _save_text(os.path.join(out_dir, "report.txt"), report_str)
    _plot_confusion_matrix(cm, os.path.join(out_dir, "confusion_matrix.png"))

    results: Dict[str, Any] = {
        "model_path": args.model_path,
        "data_path": args.data_path,
        "text_col": text_col,
        "label_col": label_col,
        "positive_label": args.positive_label,
        "positive_int": args.positive_int,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_str,
    }

    # Probabilities -> ROC/PR
    y_score = None
    if hasattr(clf, "predict_proba"):
        try:
            proba = clf.predict_proba(X_emb)
            if proba is not None and len(proba.shape) == 2 and proba.shape[1] >= 2:
                y_score = proba[:, 1]
        except Exception:
            y_score = None

    if y_score is not None:
        results.update(_plot_roc(y_true, y_score, os.path.join(out_dir, "roc_curve.png")))
        results.update(_plot_pr(y_true, y_score, os.path.join(out_dir, "pr_curve.png")))
    else:
        results["note"] = "Classifier has no usable predict_proba(); ROC/PR plots skipped."

    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== Evaluation complete ===")
    print("Saved to:", out_dir)
    print("\n=== Classification Report ===")
    print(report_str)
    print("\n=== Confusion Matrix (rows=actual, cols=pred) ===")
    print(cm)


if __name__ == "__main__":
    main()