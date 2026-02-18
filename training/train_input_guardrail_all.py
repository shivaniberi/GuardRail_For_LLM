import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# --------- S3 PATHS (train/val/test for all 7 datasets) ----------
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

# --------- LABELING RULES (unified is_unsafe) ----------
# We only have explicit safety labels in:
# - safety_prompt: IS_SAFE (safe=1, unsafe=0)
# - hate_speech: HATESPEECH / VIOLENCE / HATE_SPEECH_SCORE
#
# For datasets with no explicit safety label (dolly, squad_qa, truthful_qa_generation, multi_nli, wino_bias),
# we label them as SAFE (is_unsafe = 0) by default.
#
# This is intentional so you can "train on all datasets" while still learning unsafe patterns from the labeled sets.
HATE_SCORE_THRESHOLD = 0.5  # adjust if you want stricter/looser unsafe labeling

def pick_text(df: pd.DataFrame) -> pd.Series:
    # Your splits consistently include input_text_trunc; prefer that
    if "input_text_trunc" in df.columns and df["input_text_trunc"].notna().any():
        return df["input_text_trunc"].fillna("").astype(str)
    if "input_text" in df.columns and df["input_text"].notna().any():
        return df["input_text"].fillna("").astype(str)
    # fallbacks
    if "PROMPT" in df.columns and df["PROMPT"].notna().any():
        return df["PROMPT"].fillna("").astype(str)
    if "TEXT" in df.columns and df["TEXT"].notna().any():
        return df["TEXT"].fillna("").astype(str)
    instr = df["INSTRUCTION"].fillna("").astype(str) if "INSTRUCTION" in df.columns else None
    ctx   = df["CONTEXT"].fillna("").astype(str) if "CONTEXT" in df.columns else None
    if instr is not None and ctx is not None:
        return (instr + "\n" + ctx).astype(str)
    if "QUESTION" in df.columns and df["QUESTION"].notna().any():
        return df["QUESTION"].fillna("").astype(str)
    raise ValueError("No usable text column found.")

def make_is_unsafe(dataset_name: str, df: pd.DataFrame) -> pd.Series:
    if dataset_name == "safety_prompt":
        # IS_SAFE: 1 means safe, 0 means unsafe
        y_safe = df["IS_SAFE"]
        if y_safe.dtype == bool:
            y_safe = y_safe.astype(int)
        y_safe = y_safe.fillna(1).astype(int)
        return (1 - y_safe).astype(int)

    if dataset_name == "hate_speech":
        hatespeech = df["HATESPEECH"].fillna(0).astype(int) if "HATESPEECH" in df.columns else 0
        violence   = df["VIOLENCE"].fillna(0).astype(int) if "VIOLENCE" in df.columns else 0
        score      = df["HATE_SPEECH_SCORE"].fillna(0.0).astype(float) if "HATE_SPEECH_SCORE" in df.columns else 0.0
        unsafe = (hatespeech == 1) | (violence == 1) | (score >= HATE_SCORE_THRESHOLD)
        return unsafe.astype(int)

    # default: treat as SAFE (no explicit safety label available)
    return pd.Series([0] * len(df), index=df.index, dtype=int)

def load_split(split_key: str):
    X_all, y_all, src_all = [], [], []
    for ds, paths in SPLITS.items():
        p = paths[split_key]
        df = pd.read_parquet(p)
        X = pick_text(df).tolist()
        y = make_is_unsafe(ds, df).tolist()
        X_all.extend(X)
        y_all.extend(y)
        src_all.extend([ds] * len(X))
    return X_all, y_all, src_all

def embed(model, texts):
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

def report(name, y_true, y_pred):
    print(f"\n===== {name} =====")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=4))

def main():
    print("Loading ALL datasets (TRAIN)...")
    X_train, y_train, src_train = load_split("train")

    print("Loading ALL datasets (VAL)...")
    X_val, y_val, src_val = load_split("val")

    print("Loading ALL datasets (TEST)...")
    X_test, y_test, src_test = load_split("test")

    print("\nCounts (unsafe=1):")
    print("TRAIN:", sum(y_train), "/", len(y_train))
    print("VAL:  ", sum(y_val), "/", len(y_val))
    print("TEST: ", sum(y_test), "/", len(y_test))

    print("\nLoading embedding model (CPU)...")
    emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    print("Embedding TRAIN...")
    E_train = embed(emb_model, X_train)

    print("Training LogisticRegression (binary unsafe vs safe)...")
    clf = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])
    clf.fit(E_train, y_train)

    print("Embedding VAL...")
    E_val = embed(emb_model, X_val)
    val_pred = clf.predict(E_val)
    report("VAL (All datasets merged)", y_val, val_pred)

    print("Embedding TEST...")
    E_test = embed(emb_model, X_test)
    test_pred = clf.predict(E_test)
    report("TEST (All datasets merged)", y_test, test_pred)

    out_file = "input_safety_all7.joblib"
    joblib.dump(
        {"embed_model_name": "sentence-transformers/all-MiniLM-L6-v2", "classifier": clf},
        out_file
    )
    print("\nSaved trained guardrail classifier to:", out_file)

if __name__ == "__main__":
    main()
