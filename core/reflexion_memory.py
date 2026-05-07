"""
Reflexion Memory
================
Stores human feedback (thumbs up/down + corrections) and retrieves
similar past corrections using vector similarity search.

When a user corrects a wrong answer, that correction is saved.
Next time a similar question is asked, the correction is injected
into the prompt so the LLM gives the right answer.

Storage:
- If S3_FEEDBACK_PATH env var is set → saves/reads from S3 (shared across local + cloud)
- Otherwise → falls back to local logs/human_feedback.parquet
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from io import BytesIO

_MODEL = None
_LOCAL_FEEDBACK_PATH = Path("logs/human_feedback.parquet")
_SIMILARITY_THRESHOLD = 0.85

# Set S3_FEEDBACK_PATH in .env to share feedback across local + cloud
# Example: S3_FEEDBACK_PATH=s3://guardrail-group-bucket/feedback/human_feedback.parquet
_S3_FEEDBACK_PATH = os.getenv("S3_FEEDBACK_PATH", "")


def _get_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _MODEL


def _embed(text: str) -> list:
    return _get_model().encode(text, convert_to_numpy=True).tolist()


def _using_s3() -> bool:
    return bool(_S3_FEEDBACK_PATH and _S3_FEEDBACK_PATH.startswith("s3://"))


def _parse_s3_path(s3_path: str):
    path = s3_path.replace("s3://", "")
    bucket, _, key = path.partition("/")
    return bucket, key


def _read_df() -> pd.DataFrame:
    """Read feedback dataframe from S3 or local."""
    if _using_s3():
        try:
            import boto3
            bucket, key = _parse_s3_path(_S3_FEEDBACK_PATH)
            s3 = boto3.client("s3")
            obj = s3.get_object(Bucket=bucket, Key=key)
            return pd.read_parquet(BytesIO(obj["Body"].read()))
        except Exception as e:
            print(f"[ReflexionMemory] S3 read info: {e}")
            return pd.DataFrame()
    else:
        if not _LOCAL_FEEDBACK_PATH.exists():
            return pd.DataFrame()
        try:
            return pd.read_parquet(_LOCAL_FEEDBACK_PATH)
        except Exception:
            return pd.DataFrame()


def _write_df(df: pd.DataFrame):
    """Write feedback dataframe to S3 or local."""
    if _using_s3():
        try:
            import boto3
            bucket, key = _parse_s3_path(_S3_FEEDBACK_PATH)
            s3 = boto3.client("s3")
            buffer = BytesIO()
            df.to_parquet(buffer, index=False)
            buffer.seek(0)
            s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
            print(f"[ReflexionMemory] Saved to S3: {_S3_FEEDBACK_PATH}")
        except Exception as e:
            print(f"[ReflexionMemory] S3 write error: {e} — falling back to local")
            _write_local(df)
    else:
        _write_local(df)


def _write_local(df: pd.DataFrame):
    _LOCAL_FEEDBACK_PATH.parent.mkdir(exist_ok=True)
    df.to_parquet(_LOCAL_FEEDBACK_PATH, index=False)


def save_feedback(prompt: str, response: str, rating: int, correction: str = ""):
    """
    Save a piece of human feedback to S3 or local.

    Args:
        prompt:     The original user question
        response:   The system's answer that was rated
        rating:     1 = thumbs up, -1 = thumbs down
        correction: The correct answer provided by the user (optional)
    """
    entry = {
        "timestamp":  datetime.now().isoformat(),
        "prompt":     prompt,
        "response":   response,
        "rating":     rating,
        "correction": correction,
        "embedding":  json.dumps(_embed(prompt)),
    }

    df = _read_df()
    if df.empty:
        df = pd.DataFrame([entry])
    else:
        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)

    _write_df(df)
    storage = f"S3 ({_S3_FEEDBACK_PATH})" if _using_s3() else "local"
    print(f"[ReflexionMemory] Saved feedback to {storage} — rating: {rating}, correction: '{correction}'")


def get_correction(prompt: str) -> str:
    """
    Check if a human has previously corrected a similar question.
    Returns the correction string if found, else empty string.
    """
    df = _read_df()

    if df.empty:
        return ""

    corrected = df[
        (df["correction"].notna()) &
        (df["correction"].str.strip() != "")
    ]

    if corrected.empty:
        return ""

    query_emb = np.array(_embed(prompt))
    best_score = 0.0
    best_correction = ""

    for _, row in corrected.iterrows():
        try:
            stored_emb = np.array(json.loads(row["embedding"]))
            score = float(
                np.dot(query_emb, stored_emb) /
                (np.linalg.norm(query_emb) * np.linalg.norm(stored_emb) + 1e-9)
            )
            if score > best_score:
                best_score = score
                best_correction = row["correction"]
        except Exception:
            continue

    if best_score >= _SIMILARITY_THRESHOLD:
        print(f"[ReflexionMemory] Found correction (similarity: {best_score:.3f}): '{best_correction}'")
        return best_correction

    return ""


def get_all_feedback() -> pd.DataFrame:
    """Return all stored feedback as a DataFrame. Useful for retraining."""
    return _read_df()