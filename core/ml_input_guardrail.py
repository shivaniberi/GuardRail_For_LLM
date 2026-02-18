import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

class MLInputGuardrail:
    """
    Trained binary classifier:
      predicts is_unsafe (1) vs safe (0) from user prompt.
    Uses:
      - SentenceTransformer embeddings
      - LogisticRegression pipeline saved in joblib
    """

    def __init__(self, model_path: str = "input_safety_all7.joblib", threshold: float = 0.5):
        bundle = joblib.load(model_path)
        self.embed_model_name = bundle["embed_model_name"]
        self.clf = bundle["classifier"]
        self.threshold = threshold
        self.embedder = SentenceTransformer(self.embed_model_name, device="cpu")

    def predict_proba_unsafe(self, text: str) -> float:
        emb = self.embedder.encode([text], convert_to_numpy=True)
        proba = self.clf.predict_proba(emb)[0]
        # class 1 = unsafe
        return float(proba[1])

    def validate(self, prompt: str) -> dict:
        p_unsafe = self.predict_proba_unsafe(prompt)
        is_unsafe = p_unsafe >= self.threshold
        return {
            "valid": not is_unsafe,
            "unsafe_probability": p_unsafe,
            "threshold": self.threshold,
            "block_category": "ml_unsafe" if is_unsafe else None,
        }
