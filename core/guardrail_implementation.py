"""
GuardRail_For_LLM — Multi-Category Guardrail System (Ollama-backed) + Dual Knowledge Bases (RAG)
===============================================================================================

What this code does (brief):
- Runs an Ollama LLM for responses (qwen/llama/mistral/phi3 via your ollama_client.py)
- Applies INPUT guardrails:
  1) Rule-based checks (privacy, hate, violence/illegal, misinformation, bias, prompt injection)
  2) ML classifier check (MLInputGuardrail)
- If unsafe => returns a firm refusal message (NO “rephrase and I’ll answer”)
- If safe => retrieves context from TWO knowledge bases (Parquet, local or S3):
  1) primary KB: GuardrailConfig.rag_dataset_path
  2) wiki KB: GuardrailConfig.wiki_rag_dataset_path
  and merges them into one RAG context
- Generates an answer using RAG context (but DOES NOT force “only use context” — reduces wrong CEO answers)
- Applies OUTPUT guardrail:
  - semantic similarity check between merged context and answer (hallucination check)
- Caches embeddings separately per KB (so primary/wiki caches don’t overwrite)

Important behavior changes vs your old code:
- Fixed refusal text: no more “If you rephrase…” anywhere.
- Generation prompt: no longer forces the model to ONLY use context.
  (Your CEO failures happened because context retrieved “YouTube CEO Neal Mohan” and the model obeyed it.)
"""

import os
import re
import json
import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, List, Tuple

import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# ML-based trained gatekeeper
from .ml_input_guardrail import MLInputGuardrail

# Ollama adapter (multi-model)
from .ollama_client import ollama_generate


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
    phi3_model_path: str = ""  # kept for compatibility (not used when using Ollama)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Data paths (RAG knowledge bases)
    safety_prompts_path: str = ""  # kept for compatibility (not used directly now)

    # Existing KB (your current one)
    rag_dataset_path: str = "s3://guardrail-group-bucket/processed/train.parquet"

    # Wikipedia KB (Airflow DAG uploaded this) — update if your S3 key differs
    wiki_rag_dataset_path: str = (
        "s3://guardrail-group-bucket/knowledge_base/wikipedia/latest/wikipedia_en_articles.parquet"
    )

    # Retrieval knobs
    rag_k_primary: int = 3
    rag_k_wiki: int = 3
    max_context_chars: int = 8000  # cap to avoid huge prompts

    # Ollama model key (must be one of your SUPPORTED_MODELS keys)
    ollama_model_name: str = "qwen0.5"

    # ML guardrail model config
    ml_guardrail_model_path: str = "input_safety_all7.joblib"
    ml_guardrail_threshold: float = 0.5

    # Whether to also call a raw non-RAG LLM response for comparison/debug
    always_return_raw_llm: bool = True

    # If True, we DO NOT force the model to rely solely on context.
    # This prevents “wrong-but-in-context” answers (e.g., YouTube CEO returned for Google CEO).
    allow_model_knowledge_if_context_insufficient: bool = True


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
        r"\bgps coordinates\b",
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
        "ugly",
        "stupid",
        "idiots",
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
        "rob a bank",
        "bank robbery",
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
        r"women (are|are always|are generally)\s+(late|slow|emotional)",
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
        t = (text or "").lower()
        if regex:
            return any(re.search(p, t) for p in patterns)
        return any(p.lower() in t for p in patterns)

    def detect_privacy(self, text: str) -> bool:
        return self._match_any(self.PRIVACY_PATTERNS, text, regex=True)

    def detect_hate(self, text: str) -> bool:
        return self._match_any(self.HATE_KEYWORDS, text, regex=False)

    def detect_violence_or_illegal(self, text: str) -> bool:
        return (
            self._match_any(self.VIOLENCE_KEYWORDS, text, regex=False)
            or self._match_any(self.ILLEGAL_KEYWORDS, text, regex=False)
        )

    def detect_misinformation(self, text: str) -> bool:
        return self._match_any(self.MISINFO_PATTERNS, text, regex=True)

    def detect_bias(self, text: str) -> bool:
        return self._match_any(self.BIAS_PATTERNS, text, regex=True)

    def detect_injection(self, text: str) -> bool:
        return self._match_any(self.INJECTION_PATTERNS, text, regex=True)

    def validate(self, prompt: str) -> Dict:
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
    - hallucination check via semantic similarity between merged RAG context and response
    """

    def __init__(self, config: GuardrailConfig, embedding_model: SentenceTransformer):
        self.config = config
        self.embedding_model = embedding_model

    def check_hallucination(self, response: str, context: Optional[str]) -> Dict:
        if not context:
            return {"hallucination_similarity": None, "valid": True}

        ctx_emb = self.embedding_model.encode(context, convert_to_tensor=True).to("cpu")
        resp_emb = self.embedding_model.encode(response, convert_to_tensor=True).to("cpu")
        sim = util.cos_sim(ctx_emb, resp_emb).item()

        return {
            "hallucination_similarity": float(sim),
            "valid": sim >= self.config.hallucination_threshold,
        }

    def verify(self, response: str, context: Optional[str]) -> Dict:
        hall = self.check_hallucination(response, context)
        return {"valid": hall["valid"], "checks": hall}


# ============================================================
# RAG RETRIEVER (PER-KB CACHE FILE)
# ============================================================

class RAGRetriever:
    """
    Generic RAG Retriever:
    - Loads texts from a parquet (local or s3://...)
    - Embeds and caches to a per-dataset .pt file next to this code
    """

    def __init__(self, name: str, path: Optional[str], embedding_model: SentenceTransformer):
        self.name = name
        self.path = path
        self.embedding_model = embedding_model
        self.knowledge_base: List[str] = []
        self.embeddings: Optional[torch.Tensor] = None

        if path:
            self._load_or_build_embeddings(path)

    def _cache_path(self, dataset_path: str) -> str:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        h = hashlib.md5(dataset_path.encode("utf-8")).hexdigest()[:10]
        return os.path.join(base_dir, f"rag_embeddings_{self.name}_{h}.pt")

    def _pick_texts_from_df(self, df: pd.DataFrame) -> List[str]:
        # Prefer wiki-style schema
        if "title" in df.columns and "text" in df.columns:
            titles = df["title"].astype(str).fillna("").tolist()
            texts = df["text"].astype(str).fillna("").tolist()
            out: List[str] = []
            for t, x in zip(titles, texts):
                t = t.strip()
                x = x.strip()
                if not x:
                    continue
                out.append(f"TITLE: {t}\n\n{x}" if t else x)
            return out

        # Prefer your existing schemas
        for col in ["CONTEXT", "context", "TEXT", "text", "passage"]:
            if col in df.columns:
                return df[col].astype(str).fillna("").tolist()

        # Fallback: first object column
        for col in df.columns:
            if df[col].dtype == "object":
                return df[col].astype(str).fillna("").tolist()

        return []

    def _load_or_build_embeddings(self, path: str):
        cache_file = self._cache_path(path)

        if os.path.exists(cache_file):
            try:
                print(f"✓ Loading cached RAG embeddings ({self.name}) from: {cache_file}")
                data = torch.load(cache_file, map_location="cpu")
                self.knowledge_base = data["texts"]
                self.embeddings = data["embeddings"].to("cpu")
                return
            except Exception as e:
                print(f"Warning: failed to load RAG cache {cache_file}: {e}")

        print(f"Loading RAG docs ({self.name}) from: {path}")
        df = pd.read_parquet(path)

        kb = self._pick_texts_from_df(df)
        self.knowledge_base = [t for t in kb if isinstance(t, str) and t.strip()]

        print(f"Loaded {len(self.knowledge_base)} RAG docs ({self.name})")

        self.embeddings = self.embedding_model.encode(
            self.knowledge_base,
            convert_to_tensor=True,
            show_progress_bar=True,
        ).to("cpu")

        try:
            torch.save({"texts": self.knowledge_base, "embeddings": self.embeddings}, cache_file)
            print(f"✓ Saved RAG embeddings ({self.name}) to cache at: {cache_file}")
        except Exception as e:
            print(f"Warning: failed to save RAG cache ({self.name}): {e}")

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        if not self.knowledge_base or self.embeddings is None:
            return []
        q_emb = self.embedding_model.encode(query, convert_to_tensor=True).to("cpu")
        scores = util.cos_sim(q_emb, self.embeddings.to("cpu"))[0]
        topk = torch.topk(scores, k=min(k, len(self.knowledge_base)))
        return [self.knowledge_base[i] for i in topk.indices]


def _truncate_context(text: str, max_chars: int) -> str:
    if not text:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[...TRUNCATED...]"


# ============================================================
# MAIN SYSTEM
# ============================================================

class GuardrailSystem:
    def __init__(self, config: GuardrailConfig):
        self.config = config
        self.logs: List[Dict] = []

        print("Initializing Guardrail System (Ollama-backed)...")
        print("Using Ollama model key:", config.ollama_model_name)

        # Keep these for compatibility (unused when using Ollama)
        self.tokenizer = None
        self.model = None

        print("Loading shared embedding model:", config.embedding_model)
        shared_embedding_model = SentenceTransformer(config.embedding_model, device="cpu")

        self.input_guardrail = InputGuardrail(config)
        self.ml_input_guardrail = MLInputGuardrail(
            model_path=config.ml_guardrail_model_path,
            threshold=config.ml_guardrail_threshold
        )

        self.rag_retriever_primary = (
            RAGRetriever("primary", config.rag_dataset_path, shared_embedding_model)
            if config.enable_rag and config.rag_dataset_path
            else None
        )

        self.rag_retriever_wiki = (
            RAGRetriever("wiki", config.wiki_rag_dataset_path, shared_embedding_model)
            if config.enable_rag and config.wiki_rag_dataset_path
            else None
        )

        self.output_guardrail = OutputGuardrail(config, shared_embedding_model)
        print("Guardrail system ready.\n")

    # ------------------------------ BLOCK RESPONSES ------------------------------

    def _build_block_response(self, category: str, prompt: str) -> str:
        # NOTE: these are intentionally firm. No “rephrase and I’ll help”.
        if category == "privacy":
            return (
                "I can’t help with requests for private personal information such as a home address, "
                "phone number, email, or precise location of a private individual."
            )

        if category == "hate":
            return "I can’t help with hateful or harassing content."

        if category == "violence_illegal":
            return "I can’t help with wrongdoing, including violence or illegal activities."

        if category == "misinformation":
            return "I can’t help with promoting misinformation or conspiracy claims."

        if category == "bias":
            return "I can’t help with stereotyping or demeaning generalizations about people or groups."

        if category == "prompt_injection":
            return "I can’t help with requests to bypass safety or system instructions."

        if category == "ml_unsafe":
            return "I can’t help with that request."

        return "I can’t help with that request."

    # ------------------------------ RAG HELPERS ------------------------------

    def _retrieve_from_both_kbs(self, prompt: str) -> Tuple[List[str], List[str]]:
        primary_docs: List[str] = []
        wiki_docs: List[str] = []

        if self.rag_retriever_primary is not None and self.config.rag_k_primary > 0:
            primary_docs = self.rag_retriever_primary.retrieve(prompt, k=self.config.rag_k_primary)

        if self.rag_retriever_wiki is not None and self.config.rag_k_wiki > 0:
            wiki_docs = self.rag_retriever_wiki.retrieve(prompt, k=self.config.rag_k_wiki)

        return primary_docs, wiki_docs

    def _build_rag_prompt(self, prompt: str, context: str) -> str:
        """
        Critical fix:
        - Do NOT force “Answer only from context”.
        That instruction caused your wrong “Google CEO = Neal Mohan” answers,
        because your retrieved context mentioned YouTube’s CEO.
        """
        if self.config.allow_model_knowledge_if_context_insufficient:
            return (
                "You are a helpful assistant.\n"
                "Use the provided context when it is relevant and correct.\n"
                "If the context is irrelevant or does not answer the question, answer using your general knowledge.\n"
                "If you are unsure, say you are unsure.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {prompt}\n"
                "Answer:"
            )

        # Legacy strict mode (not recommended for your use case)
        return (
            f"Context:\n{context}\n\n"
            f"Question: {prompt}\n\n"
            "Answer using only the context above when possible. If the context does not contain the answer, say you don't know."
        )

    # ------------------------------ GENERATION ------------------------------

    def generate_with_guardrails(
        self,
        prompt: str,
        max_new_tokens: int = 256,
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

        # 0) Raw (no RAG) output for comparison/debug
        if self.config.always_return_raw_llm:
            try:
                raw_generic = ollama_generate(
                    prompt=prompt,
                    model_name=self.config.ollama_model_name,
                    system="You are a helpful assistant.",
                )
                result["metadata"]["raw_llm_response"] = raw_generic
            except Exception as e:
                result["metadata"]["raw_llm_response"] = f"[ERROR] Raw LLM failed: {e}"

        # 1) INPUT validation (rule-based + ML)
        if self.config.enable_input_validation:
            icheck = self.input_guardrail.validate(prompt)
            mlcheck = self.ml_input_guardrail.validate(prompt)
            result["guardrails"]["input"] = {"rule_based": icheck, "ml_based": mlcheck}

            if (not icheck["valid"]) or (not mlcheck["valid"]):
                category = icheck.get("block_category") or mlcheck.get("block_category") or "generic"
                result["response"] = self._build_block_response(category, prompt)
                result["metadata"]["blocked_at"] = "input"
                result["metadata"]["block_category"] = category
                self._log(result)
                return result

        # 2) RAG context (merged from both KBs)
        context = None
        full_prompt = prompt
        kb_sources: List[str] = []
        retrieved_total: int = 0

        if use_rag and self.config.enable_rag:
            primary_docs, wiki_docs = self._retrieve_from_both_kbs(prompt)

            # metadata
            result["metadata"]["rag_primary_docs"] = len(primary_docs)
            result["metadata"]["rag_wiki_docs"] = len(wiki_docs)
            retrieved_total = len(primary_docs) + len(wiki_docs)

            all_docs: List[str] = []
            if primary_docs:
                kb_sources.append("primary")
                all_docs.append("### PRIMARY_KB")
                all_docs.extend(primary_docs)

            if wiki_docs:
                kb_sources.append("wiki")
                all_docs.append("### WIKIPEDIA_KB")
                all_docs.extend(wiki_docs)

            if all_docs:
                context = "\n\n".join(all_docs)
                context = _truncate_context(context, self.config.max_context_chars)
                result["metadata"]["rag_used"] = True

                # build RAG-friendly prompt (non-strict by default)
                full_prompt = self._build_rag_prompt(prompt, context)
            else:
                result["metadata"]["rag_used"] = False
        else:
            result["metadata"]["rag_used"] = False

        # 3) Generation (with RAG prompt if enabled)
        try:
            guarded_llm = ollama_generate(
                prompt=full_prompt,
                model_name=self.config.ollama_model_name,
                system="You are a helpful assistant.",
            )
            result["metadata"]["guarded_llm_response"] = guarded_llm
            result["response"] = guarded_llm
        except Exception as e:
            result["response"] = f"[ERROR] Generation failed: {e}"
            result["metadata"]["error"] = str(e)
            self._log(result)
            return result

        # 4) OUTPUT guardrail (hallucination vs merged context)
        if self.config.enable_output_verification:
            result["guardrails"]["output"] = self.output_guardrail.verify(result["response"], context)

        # Helpful RAG metadata (fix your “None” fields)
        result["metadata"]["retrieved_docs_total"] = retrieved_total
        result["metadata"]["kb_sources"] = kb_sources

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
            df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)
        return df

    def save_logs(self, path: str):
        df = self.get_logs()
        if df.empty:
            print("No logs to save.")
            return
        df.to_parquet(path, index=False)
        print(f"Logs saved to {path}")
