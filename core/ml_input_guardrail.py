"""
MLInputGuardrail — Trained binary safety classifier
====================================================

"""

import os
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))


class MLInputGuardrail:
    """
    Trained binary classifier:
      predicts is_unsafe (1) vs safe (0) from user prompt.
    Uses:
      - SentenceTransformer embeddings (shared instance preferred)
      - LogisticRegression pipeline saved in joblib
    """

    def __init__(
        self,
        model_path: str = os.path.join(_HERE, "input_safety_all7.joblib"),
        threshold: float = 0.5,
        embedder=None,  # NEW: accept pre-loaded SentenceTransformer to avoid duplicate load
    ):
        self.threshold = threshold
        self._available = False

        # ── Load classifier bundle ────────────────────────────────────────────
        try:
            import joblib
            bundle = joblib.load(model_path)
            if not isinstance(bundle, dict) or "classifier" not in bundle:
                raise ValueError(f"Unexpected bundle format in {model_path}: {list(bundle.keys())}")
            self.embed_model_name = bundle["embed_model_name"]
            self.clf = bundle["classifier"]
            self._available = True
        except FileNotFoundError:
            logger.error(
                f"ML guardrail model not found at: {model_path}\n"
                f"  → Run train_input_guardrail_all.py to generate it.\n"
                f"  → ML guardrail will be DISABLED until the model is present."
            )
            self.clf = None
            self.embed_model_name = None
        except Exception as e:
            logger.error(f"Failed to load ML guardrail model: {e}")
            self.clf = None
            self.embed_model_name = None

        # ── Embedder: use shared instance if provided, else load own ─────────
        if embedder is not None:
            # Use the passed-in shared instance — avoids loading a duplicate model
            self.embedder = embedder
        elif self._available and self.embed_model_name:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedder = SentenceTransformer(self.embed_model_name, device="cpu")
            except Exception as e:
                logger.error(f"Failed to load ML guardrail embedder: {e}")
                self.embedder = None
                self._available = False
        else:
            self.embedder = None

    def predict_proba_unsafe(self, text: str) -> float:
        """Returns probability that the input is unsafe (class 1)."""
        if not self._available or self.embedder is None or self.clf is None:
            return 0.0

        emb   = self.embedder.encode([text], convert_to_numpy=True)
        proba = self.clf.predict_proba(emb)[0]
        # class order: [safe=0, unsafe=1]
        return float(proba[1]) if len(proba) > 1 else float(proba[0])

    # Uncertain zone: ML score is between these bounds → escalate to LlamaGuard
    UNCERTAIN_LOW  = 0.20
    UNCERTAIN_HIGH = 0.50

    def validate(self, prompt: str) -> dict:
        if not self._available:
            # ML model unavailable — go straight to LlamaGuard
            return self._llamaguard_validate(prompt, ml_prob=None)

        p_unsafe = self.predict_proba_unsafe(prompt)

        # 1. Clearly unsafe — block immediately, no LlamaGuard needed
        if p_unsafe >= self.threshold:
            return {
                "valid": False,
                "unsafe_probability": round(p_unsafe, 6),
                "threshold": self.threshold,
                "block_category": "ml_unsafe",
                "classifier": "ml",
            }

        # 2. Uncertain zone — ML not confident, escalate to LlamaGuard
        if p_unsafe >= self.UNCERTAIN_LOW:
            return self._llamaguard_validate(prompt, ml_prob=p_unsafe)

        # 3. Clearly safe — pass through
        return {
            "valid": True,
            "unsafe_probability": round(p_unsafe, 6),
            "threshold": self.threshold,
            "block_category": None,
            "classifier": "ml",
        }

    def _llamaguard_validate(self, prompt: str, ml_prob) -> dict:
        """Escalate to LlamaGuard 3 on Groq for intent-aware classification."""
        try:
            from .ollama_client import llamaguard_check
            lg = llamaguard_check(prompt)
            is_unsafe = not lg["safe"]
            p = ml_prob if ml_prob is not None else (0.9 if is_unsafe else 0.0)
            return {
                "valid": not is_unsafe,
                "unsafe_probability": round(p, 6),
                "threshold": self.threshold,
                "block_category": lg["block_category"] if is_unsafe else None,
                "classifier": "llamaguard",
                "llamaguard_category": lg.get("category"),
            }
        except Exception as e:
            logger.error(f"LlamaGuard fallback failed: {e}")
            # If LlamaGuard also fails, default to safe (avoid blocking legitimate requests)
            return {
                "valid": True,
                "unsafe_probability": round(ml_prob, 6) if ml_prob else 0.0,
                "threshold": self.threshold,
                "block_category": None,
                "classifier": "fallback_safe",
            }