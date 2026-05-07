"""
Reflexion Memory
================
Stores human feedback (thumbs up/down + corrections) and retrieves
similar past corrections using vector similarity search.

When a user corrects a wrong answer, that correction is saved.
Next time a similar question is asked, the correction is injected
into the prompt so the LLM gives the right answer.
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

_MODEL = None
_FEEDBACK_PATH = Path("logs/human_feedback.parquet")
_SIMILARITY_THRESHOLD = 0.85  # how similar a question must be to reuse a correction


def _get_model():
    """Load embedding model once and reuse."""
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _MODEL


def _embed(text: str) -> list:
    """Convert text to embedding vector."""
    return _get_model().encode(text, convert_to_numpy=True).tolist()


def save_feedback(prompt: str, response: str, rating: int, correction: str = ""):
    """
    Save a piece of human feedback to logs/human_feedback.parquet.

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

    _FEEDBACK_PATH.parent.mkdir(exist_ok=True)

    if _FEEDBACK_PATH.exists():
        df = pd.read_parquet(_FEEDBACK_PATH)
        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    else:
        df = pd.DataFrame([entry])

    df.to_parquet(_FEEDBACK_PATH, index=False)
    print(f"[ReflexionMemory] Saved feedback — rating: {rating}, correction: '{correction}'")


def get_correction(prompt: str) -> str:
    """
    Check if a human has previously corrected a similar question.

    Returns the correction string if a similar past question was corrected,
    otherwise returns empty string.

    Args:
        prompt: The current user question

    Returns:
        Correction string if found, else ""
    """
    if not _FEEDBACK_PATH.exists():
        return ""

    try:
        df = pd.read_parquet(_FEEDBACK_PATH)
    except Exception:
        return ""

    # Only look at rows that have a correction provided
    corrected = df[
        (df["correction"].notna()) &
        (df["correction"].str.strip() != "")
    ]

    if corrected.empty:
        return ""

    # Embed the current question
    query_emb = np.array(_embed(prompt))

    best_score = 0.0
    best_correction = ""

    for _, row in corrected.iterrows():
        try:
            stored_emb = np.array(json.loads(row["embedding"]))
            # Cosine similarity
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
    if not _FEEDBACK_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(_FEEDBACK_PATH)
