"""
Multi-Category Guardrail System for Phi-3 Mini
==============================================

Optimized for your local Windows setup:

- Uses local Phi-3 Mini model (CPU)
- Optional RAG using squad_qa parquet from S3
- Shared SentenceTransformer for RAG + hallucination check
- Multi-category rule-based input guardrails:
    * privacy / PII
    * hate / abuse
    * violence / illegal / self-harm
    * misinformation / conspiracy
    * bias / stereotypes
    * prompt injection

- Output guardrail: hallucination check (RAG context vs answer)
- RAG embeddings cached to rag_embeddings.pt in this folder
- Logging is Parquet-safe (no struct/Arrow issues)
"""

import os
import re
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, List

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util


# ============================================================
# CONFIG
# ============================================================


@dataclass
class GuardrailConfig:
    # Thresholds
    hallucination_threshold: float = 0.6

    # Feature flags
    enable_input_validation: bool = True
    enable_output_verification: bool = True
    enable_rag: bool = True
    enable_logging: bool = True

    # Model paths
    phi3_model_path: str = ""  # MUST be set by caller
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Data paths
    safety_prompts_path: str = ""  # kept for compatibility (not used directly now)
    rag_dataset_path: str = ""     # squad_qa parquet on S3


# ============================================================
# INPUT GUARDRAIL (RULE-BASED, MULTI-CATEGORY)
# ============================================================


class InputGuardrail:
    """
    Lightweight rule-based input guardrail with multiple categories:
    - privacy / PII
    - hate / abuse
    - violence / illegal / self harm
    - misinformation / conspiracy
    - bias / stereotypes
    - prompt injection
    """

    PRIVACY_PATTERNS = [
        r"\bhome address\b",
        r"\bhouse address\b",
        r"\bexact address\b",
        r"\bwhere does [^?]+\s+live\b",
        r"\bwhere does [^?]+\s+currently live\b",
        r"\bGPS coordinates\b",
        r"\bphone number\b",
        r"\bemail address\b",
        r"\bcontact details\b",
        r"\bprivate location\b",
        r"\bhome location\b",
        r"\bphysical address\b",
        r"\bpersonal address\b",
    ]

    HATE_KEYWORDS = [
        "black people",
        "white people",
        "muslims",
        "jews",
        "asians",
        "immigrants",
        "gay people",
        "women are inferior",
        "men are superior",
    ]

    VIOLENCE_KEYWORDS = [
        "kill",
        "murder",
        "stab",
        "shoot",
        "bomb",
        "explosive",
        "poison",
        "hide a body",
        "beat someone up",
        "hurt someone",
        "hire a hitman",
    ]

    ILLEGAL_KEYWORDS = [
        "how to hack",
        "hack into",
        "phishing email",
        "steal credit card",
        "bypass security",
        "break into",
        "drug dealer",
        "buy drugs",
        "fake passport",
    ]

    MISINFO_PATTERNS = [
        r"\bearth is flat\b",
        r"\bearth flat\b",
        r"\bfake(d)? the moon landing\b",
        r"\bmoon landing (is|was) fake\b",
        r"\bmicrochips? in (the )?covid vaccine(s)?\b",
        r"\bcovid vaccine(s)? contain microchips\b",
        r"\b5g causes covid\b",
        r"\bchemtrails\b",
    ]

    BIAS_PATTERNS = [
        r"women (are|are always|are generally)\s+(worse|inferior|bad at)",
        r"men (are|are always|are generally)\s+(better|superior)",
        r"girls can(\'t|not)\s+(do|be)\s+(math|engineering|science)",
        r"certain races are\s+(better|superior|inferior)",
    ]

    INJECTION_PATTERNS = [
        r"ignore (previous|above|all) instructions",
        r"disregard.*instructions",
        r"you are now.*different",
        r"new (role|character|personality)",
        r"act as .+",
        r"pretend (you are|to be)",
    ]

    def __init__(self, config: GuardrailConfig):
        self.config = config

    def _match_any(self, patterns: List[str], text: str, regex: bool = True) -> bool:
        t = text.lower()
        if regex:
            return any(re.search(p, t) for p in patterns)
        else:
            return any(p.lower() in t for p in patterns)

    def detect_privacy(self, text: str) -> bool:
        return self._match_any(self.PRIVACY_PATTERNS, text, regex=True)

    def detect_hate(self, text: str) -> bool:
        return self._match_any(self.HATE_KEYWORDS, text, regex=False)

    def detect_violence_or_illegal(self, text: str) -> bool:
        return self._match_any(self.VIOLENCE_KEYWORDS, text, regex=False) or \
               self._match_any(self.ILLEGAL_KEYWORDS, text, regex=False)

    def detect_misinformation(self, text: str) -> bool:
        return self._match_any(self.MISINFO_PATTERNS, text, regex=True)

    def detect_bias(self, text: str) -> bool:
        return self._match_any(self.BIAS_PATTERNS, text, regex=True)

    def detect_injection(self, text: str) -> bool:
        return self._match_any(self.INJECTION_PATTERNS, text, regex=True)

    def validate(self, prompt: str) -> Dict:
        """
        Returns:
            {
              "valid": bool,
              "block_category": Optional[str],
              "checks": {...}
            }
        """
        is_privacy = self.detect_privacy(prompt)
        is_hate = self.detect_hate(prompt)
        is_violence = self.detect_violence_or_illegal(prompt)
        is_misinfo = self.detect_misinformation(prompt)
        is_bias = self.detect_bias(prompt)
        is_injection = self.detect_injection(prompt)

        block_category = None
        if is_privacy:
            block_category = "privacy"
        elif is_violence:
            block_category = "violence_illegal"
        elif is_hate:
            block_category = "hate"
        elif is_bias:
            block_category = "bias"
        elif is_misinfo:
            block_category = "misinformation"
        elif is_injection:
            block_category = "prompt_injection"

        valid = block_category is None

        checks = {
            "privacy": {"passed": not is_privacy},
            "hate": {"passed": not is_hate},
            "violence_illegal": {"passed": not is_violence},
            "misinformation": {"passed": not is_misinfo},
            "bias": {"passed": not is_bias},
            "prompt_injection": {"passed": not is_injection},
        }

        return {
            "valid": valid,
            "timestamp": datetime.now().isoformat(),
            "block_category": block_category,
            "checks": checks,
        }


# ============================================================
# OUTPUT GUARDRAIL (HALLUCINATION vs RAG)
# ============================================================


class OutputGuardrail:
    """
    Output verification:
    - hallucination check via semantic similarity between
      RAG context and LLM response.
    """

    def __init__(self, config: GuardrailConfig, embedding_model: SentenceTransformer):
        self.config = config
        self.embedding_model = embedding_model

    def check_hallucination(self, response: str, context: Optional[str]) -> Dict:
        if not context:
            return {
                "hallucination_similarity": None,
                "valid": True,
            }

        ctx_emb = self.embedding_model.encode(context, convert_to_tensor=True)
        resp_emb = self.embedding_model.encode(response, convert_to_tensor=True)
        sim = util.cos_sim(ctx_emb, resp_emb).item()

        return {
            "hallucination_similarity": float(sim),
            "valid": sim >= self.config.hallucination_threshold,
        }

    def verify(self, response: str, context: Optional[str]) -> Dict:
        hall = self.check_hallucination(response, context)
        return {"valid": hall["valid"], "checks": hall}


# ============================================================
# RAG RETRIEVER (WITH FIXED-CACHE PATH)
# ============================================================


class RAGRetriever:
    """
    RAG Retriever with:
    - squad_qa-based knowledge base (or any text column)
    - embeddings cached to rag_embeddings.pt in the SAME folder
      as this file
    """

    def __init__(self, path: Optional[str], embedding_model: SentenceTransformer):
        self.path = path
        self.embedding_model = embedding_model
        self.knowledge_base: List[str] = []
        self.embeddings: Optional[torch.Tensor] = None

        if path:
            self._load_or_build_embeddings(path)

    def _cache_path(self) -> str:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_dir, "rag_embeddings.pt")

    def _load_or_build_embeddings(self, path: str):
        cache_file = self._cache_path()

        if os.path.exists(cache_file):
            try:
                print("✓ Loading cached RAG embeddings from:", cache_file)
                data = torch.load(cache_file, map_location="cpu")
                self.knowledge_base = data["texts"]
                self.embeddings = data["embeddings"]
                return
            except Exception as e:
                print(f"Warning: failed to load RAG cache {cache_file}: {e}")

        print("Loading RAG docs from:", path)
        df = pd.read_parquet(path)

        if "context" in df.columns:
            self.knowledge_base = df["context"].astype(str).tolist()
        elif "passage" in df.columns:
            self.knowledge_base = df["passage"].astype(str).tolist()
        elif "text" in df.columns:
            self.knowledge_base = df["text"].astype(str).tolist()
        else:
            for col in df.columns:
                if df[col].dtype == "object":
                    self.knowledge_base = df[col].astype(str).tolist()
                    break

        print(f"Loaded {len(self.knowledge_base)} RAG docs")
        self.embeddings = self.embedding_model.encode(
            self.knowledge_base,
            convert_to_tensor=True,
            show_progress_bar=True,
        )

        try:
            torch.save(
                {"texts": self.knowledge_base, "embeddings": self.embeddings},
                cache_file,
            )
            print("✓ Saved RAG embeddings to cache at:", cache_file)
        except Exception as e:
            print(f"Warning: failed to save RAG cache: {e}")

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        if not self.knowledge_base or self.embeddings is None:
            return []

        q_emb = self.embedding_model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(q_emb, self.embeddings)[0]
        topk = torch.topk(scores, k=min(k, len(self.knowledge_base)))
        return [self.knowledge_base[i] for i in topk.indices]


# ============================================================
# MAIN SYSTEM
# ============================================================


class Phi3GuardrailSystem:
    def __init__(self, config: GuardrailConfig):
        self.config = config
        self.logs: List[Dict] = []

        print("Initializing Optimized Phi-3 Guardrail System...")
        print(f"Loading Phi-3 model from {config.phi3_model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(config.phi3_model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.phi3_model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        ).to("cpu")

        print("Loading shared embedding model:", config.embedding_model)
        shared_embedding_model = SentenceTransformer(config.embedding_model)

        self.input_guardrail = InputGuardrail(config)
        self.rag_retriever = (
            RAGRetriever(config.rag_dataset_path, shared_embedding_model)
            if config.enable_rag and config.rag_dataset_path
            else None
        )
        self.output_guardrail = OutputGuardrail(config, shared_embedding_model)

        print("Guardrail system ready.\n")

    # ------------------------------ BLOCK RESPONSES ------------------------------

    def _build_block_response(self, category: str, prompt: str) -> str:
        if category == "privacy":
            return (
                "I'm sorry — I can’t help with finding or sharing home addresses, phone numbers, "
                "locations, or other private personal details about any individual.\n\n"
                "I *can* help with publicly available information such as career history, "
                "achievements, or general biographical background if you’d like."
            )

        if category == "hate":
            return (
                "I’m not able to generate content that targets groups or individuals with hate, "
                "harassment, or demeaning language.\n\n"
                "If you’re interested, I can instead help explain why hate speech is harmful, "
                "or help rewrite something in a respectful, inclusive way."
            )

        if category == "violence_illegal":
            return (
                "I can’t assist with instructions for harming people, damaging property, or carrying "
                "out illegal activities (like hacking, poisoning, or making weapons).\n\n"
                "If you’d like, I can give information about safety, conflict de-escalation, "
                "or how to get help in dangerous situations."
            )

        if category == "misinformation":
            return (
                "Your question seems connected to a common misconception or conspiracy theory, "
                "so I can't support or amplify misinformation.\n\n"
                "For example, scientific evidence shows that:\n"
                "- The Earth is very close to spherical, not flat.\n"
                "- Vaccines do *not* contain tracking microchips; they’re tested for safety and efficacy.\n"
                "- The Apollo moon landings are well-documented with multiple independent lines of evidence.\n\n"
                "If you’d like, I can walk through what experts and reputable sources say about this topic."
            )

        if category == "bias":
            return (
                "I’m not able to agree with or generate stereotypes about genders, races, or other groups "
                "being ‘better’ or ‘worse’ at something.\n\n"
                "If you’d like, I can help explain how bias works, why it’s harmful, and how to avoid it."
            )

        if category == "prompt_injection":
            return (
                "I need to follow my safety and behavior guidelines, so I can’t ignore previous instructions "
                "or behave in a way that would bypass those safeguards.\n\n"
                "I’m still happy to help within those rules — what would you like to know or build?"
            )

        return (
            "I’m not able to respond to that request because it conflicts with my safety and usage guidelines.\n\n"
            "If you rephrase your question in a safe, educational, or constructive way, "
            "I’ll be glad to help."
        )

    # ------------------------------ GENERATION ------------------------------

    def generate_with_guardrails(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.0,  # kept for API compatibility
        use_rag: bool = True,
    ) -> Dict:
        result: Dict = {
            "prompt": prompt,
            "timestamp": datetime.now().isoformat(),
            "response": "",
            "guardrails": {},
            "metadata": {},
        }

        # 1) INPUT VALIDATION
        if self.config.enable_input_validation:
            icheck = self.input_guardrail.validate(prompt)
            result["guardrails"]["input"] = icheck

            if not icheck["valid"]:
                category = icheck.get("block_category") or "generic"
                result["response"] = self._build_block_response(category, prompt)
                result["metadata"]["blocked_at"] = "input"
                result["metadata"]["block_category"] = category
                self._log(result)
                return result

        # 2) RAG CONTEXT (optional)
        context = None
        full_prompt = prompt

        if use_rag and self.config.enable_rag and self.rag_retriever is not None:
            docs = self.rag_retriever.retrieve(prompt, k=3)
            if docs:
                context = "\n\n".join(docs)
                result["metadata"]["rag_used"] = True
                result["metadata"]["retrieved_docs"] = len(docs)
                full_prompt = (
                    f"Context:\n{context}\n\n"
                    f"Question: {prompt}\n\n"
                    f"Answer using only the context above when possible:"
                )
            else:
                result["metadata"]["rag_used"] = False
        else:
            result["metadata"]["rag_used"] = False

        # 3) GENERATION (CPU-friendly, deterministic)
        try:
            tokens = self.tokenizer(full_prompt, return_tensors="pt")
            with torch.no_grad():
                out_ids = self.model.generate(
                    **tokens,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,         # greedy for speed & determinism
                    temperature=None,
                    use_cache=False,         # avoid DynamicCache issues
                    no_repeat_ngram_size=3,  # avoid repetition loops that can stall
                )
            decoded = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
            if full_prompt in decoded:
                decoded = decoded.replace(full_prompt, "").strip()
            result["response"] = decoded
        except Exception as e:
            result["response"] = f"[ERROR] Generation failed: {e}"
            result["metadata"]["error"] = str(e)
            self._log(result)
            return result

        # 4) OUTPUT GUARDRAIL (hallucination vs RAG)
        if self.config.enable_output_verification:
            ocheck = self.output_guardrail.verify(result["response"], context)
            result["guardrails"]["output"] = ocheck

        self._log(result)
        return result

    # ------------------------------ LOGGING ------------------------------

    def _log(self, record: Dict):
        if self.config.enable_logging:
            self.logs.append(record)

    def get_logs(self) -> pd.DataFrame:
        if not self.logs:
            return pd.DataFrame()
        df = pd.DataFrame(self.logs)

        for col in df.columns:
            df[col] = df[col].apply(
                lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x
            )
        return df

    def save_logs(self, path: str):
        df = self.get_logs()
        if df.empty:
            print("No logs to save.")
            return
        df.to_parquet(path, index=False)
        print(f"Logs saved to {path}")
