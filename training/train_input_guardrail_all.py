"""
Train Input Guardrail — All 7 Datasets
=======================================
"""

import re
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# ── S3 PATHS ─────────────────────────────────────────────────────────────────
SPLITS = {
    "dolly_instructions": {
        "train": "s3://guardrail-group-bucket/prepared_splits/dolly_instructions/2025/11/02/dolly_instructions_train.parquet",
        "val":   "s3://guardrail-group-bucket/prepared_splits/dolly_instructions/2025/11/02/dolly_instructions_val.parquet",
        "test":  "s3://guardrail-group-bucket/prepared_splits/dolly_instructions/2025/11/02/dolly_instructions_test.parquet",
    },
    "hate_speech": {
        "train": "s3://guardrail-group-bucket/prepared_splits/hate_speech/2025/11/02/hate_speech_train.parquet",
        "val":   "s3://guardrail-group-bucket/prepared_splits/hate_speech/2025/11/02/hate_speech_val.parquet",
        "test":  "s3://guardrail-group-bucket/prepared_splits/hate_speech/2025/11/02/hate_speech_test.parquet",
    },
    "multi_nli": {
        "train": "s3://guardrail-group-bucket/prepared_splits/multi_nli/2025/11/02/multi_nli_train.parquet",
        "val":   "s3://guardrail-group-bucket/prepared_splits/multi_nli/2025/11/02/multi_nli_val.parquet",
        "test":  "s3://guardrail-group-bucket/prepared_splits/multi_nli/2025/11/02/multi_nli_test.parquet",
    },
    "safety_prompt": {
        "train": "s3://guardrail-group-bucket/prepared_splits/safety_prompt/2025/11/02/safety_prompt_train.parquet",
        "val":   "s3://guardrail-group-bucket/prepared_splits/safety_prompt/2025/11/02/safety_prompt_val.parquet",
        "test":  "s3://guardrail-group-bucket/prepared_splits/safety_prompt/2025/11/02/safety_prompt_test.parquet",
    },
    "squad_qa": {
        "train": "s3://guardrail-group-bucket/prepared_splits/squad_qa/2025/11/02/squad_qa_train.parquet",
        "val":   "s3://guardrail-group-bucket/prepared_splits/squad_qa/2025/11/02/squad_qa_val.parquet",
        "test":  "s3://guardrail-group-bucket/prepared_splits/squad_qa/2025/11/02/squad_qa_test.parquet",
    },
    "truthful_qa_generation": {
        "train": "s3://guardrail-group-bucket/prepared_splits/truthful_qa_generation/2025/11/02/truthful_qa_generation_train.parquet",
        "val":   "s3://guardrail-group-bucket/prepared_splits/truthful_qa_generation/2025/11/02/truthful_qa_generation_val.parquet",
        "test":  "s3://guardrail-group-bucket/prepared_splits/truthful_qa_generation/2025/11/02/truthful_qa_generation_test.parquet",
    },
    "wino_bias": {
        "train": "s3://guardrail-group-bucket/prepared_splits/wino_bias/2025/11/02/wino_bias_train.parquet",
        "val":   "s3://guardrail-group-bucket/prepared_splits/wino_bias/2025/11/02/wino_bias_val.parquet",
        "test":  "s3://guardrail-group-bucket/prepared_splits/wino_bias/2025/11/02/wino_bias_test.parquet",
    },
}

HATE_SCORE_THRESHOLD = 0.5

# ── SYNTHETIC UNSAFE PATTERNS ─────────────────────────────────────────────────
SYNTHETIC_UNSAFE_PATTERNS = [
    # Violence / harm
    r"\bhow\s+to\s+(kill|murder|stab|shoot|hurt|harm|assault)\b",
    r"\b(kill|murder|stab|shoot|poison|strangle)\s+(someone|a\s+person|people|him|her)\b",
    r"\b(make|build|assemble)\s+a\s+(bomb|explosive|weapon|grenade)\b",
    r"\bhide\s+a\s+body\b",
    r"\bhow\s+to\s+get\s+away\s+with\s+(murder|killing|crime)\b",

    # Hacking / illegal
    r"\bhow\s+to\s+hack\b",
    r"\b(steal|clone|skim)\s+(credit\s+card|debit\s+card|identity)\b",
    r"\bbypass\s+(security|authentication|2fa)\b",
    r"\bphishing\s+(email|attack)\b",
    r"\bhow\s+to\s+launder\s+money\b",
    r"\bfake\s+(passport|id|license|document)\b",

    # Self-harm
    r"\bhow\s+to\s+(kill|hurt|harm)\s+(my)?self\b",
    r"\b(suicide|self.harm)\s+(method|way|how|guide|instruction)\b",
    r"\bwant\s+to\s+(die|end\s+my\s+life|kill\s+myself)\b",

    # Drug synthesis
    r"\bhow\s+to\s+(make|cook|synthesize)\s+(meth|heroin|fentanyl|crack|cocaine)\b",

    # Hate / derogatory
    r"\b(kill|exterminate|deport)\s+(all\s+)?(the\s+)?(blacks|jews|muslims|gays|immigrants)\b",
    r"\b(n[i1]gg[ae]r|ch[i1]nk|sp[i1]c|k[i1]ke|f[a4]gg[o0]t)\b",
    r"\b(race|ethnicity)\s+(determines|predicts)\s+(intelligence|crime)\b",

    # Fraud
    r"\b(create|make)\s+a\s+fake\s+(invoice|receipt|bank\s+statement)\b",
    r"\bhow\s+to\s+(commit|do)\s+(tax|wire|insurance)\s+fraud\b",

    # Prompt injection
    r"\bignore\s+(all|previous|your)\s+instructions?\b",
    r"\bjailbreak\b",
    r"\bdeveloper\s+mode\b",
]

def _text_is_synthetically_unsafe(text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in SYNTHETIC_UNSAFE_PATTERNS)

def pick_text(df: pd.DataFrame) -> pd.Series:
    for col in ["input_text", "PROMPT", "TEXT", "QUESTION"]:
        if col in df.columns:
            return df[col].fillna("").astype(str)
    raise ValueError("No text column found")

def make_is_unsafe(dataset_name: str, df: pd.DataFrame) -> pd.Series:
    texts = pick_text(df)
    return texts.apply(lambda t: int(_text_is_synthetically_unsafe(t)))

def load_split(split_key: str):
    X_all, y_all = [], []
    for ds, paths in SPLITS.items():
        df = pd.read_parquet(paths[split_key])
        X = pick_text(df).tolist()
        y = make_is_unsafe(ds, df).tolist()
        X_all.extend(X)
        y_all.extend(y)
    return X_all, y_all

def embed(model, texts):
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

def report(name, y_true, y_pred):
    print(f"\n===== {name} =====")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=4))

def find_best_threshold(clf, X_emb, y_true):
    thresholds = np.arange(0.2, 0.8, 0.05)
    probas = clf.predict_proba(X_emb)[:, 1]
    best_t, best_f1 = 0.5, 0
    for t in thresholds:
        preds = (probas >= t).astype(int)
        f1 = f1_score(y_true, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t

def main():
    print("Loading data...")
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")

    emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    E_train = embed(emb_model, X_train)
    E_val = embed(emb_model, X_val)
    E_test = embed(emb_model, X_test)

    print("\nTraining models...")

    # Logistic Regression (existing)
    clf_lr = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("lr", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    clf_lr.fit(E_train, y_train)

    # Random Forest (NEW)
    clf_rf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    clf_rf.fit(E_train, y_train)

    # Decision Tree (NEW)
    clf_dt = DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42
    )
    clf_dt.fit(E_train, y_train)

    models = {
        "LogisticRegression": clf_lr,
        "RandomForest": clf_rf,
        "DecisionTree": clf_dt,
    }

    best_model = None
    best_f1 = 0
    best_threshold = 0.5

    print("\nEvaluating models on VAL set...")

    for name, model in models.items():
        print(f"\n--- {name} ---")

        threshold = find_best_threshold(model, E_val, y_val)
        probas = model.predict_proba(E_val)[:, 1]
        preds = (probas >= threshold).astype(int)

        f1 = f1_score(y_val, preds, average="macro", zero_division=0)

        report(name, y_val, preds)
        print(f"{name} → F1: {f1:.4f}, Threshold: {threshold:.2f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_threshold = threshold

    print("\nBEST MODEL SELECTED")
    print(f"F1: {best_f1:.4f}, Threshold: {best_threshold:.2f}")

    test_probas = best_model.predict_proba(E_test)[:, 1]
    test_pred = (test_probas >= best_threshold).astype(int)

    report("TEST", y_test, test_pred)

    joblib.dump({
        "embed_model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "classifier": best_model,
        "threshold": best_threshold,
    }, "input_safety_all7.joblib")

    print("\nModel saved successfully")

if __name__ == "__main__":
    main()