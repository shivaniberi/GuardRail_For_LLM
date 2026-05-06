"""
Multi-Agent Collaboration: Chain-of-Debate Orchestrator
======================================================

Purpose:
- Provide a lightweight multi-agent orchestration layer that integrates with
  the existing GuardrailSystem in this repo.
- Implements a Chain-of-Debate pattern: multiple specialist LLM agents propose
  arguments, critics rebut, and a Judge agent produces a final adjudicated
  answer. Final output is passed through the existing guardrails (safety +
  factual verification).

Design Goals (MVP):
- Reuse ollama_generate and GuardrailSystem components (RAGRetriever, input/output
  guardrails, shared embedding model) to avoid duplicating heavy resources.
- Keep the interface simple: Orchestrator.debate(question, ... ) -> structured result.
- Be async-friendly so specialist agents can be called concurrently.

Usage example (see bottom of file):
  from core.guardrail_implementation import GuardrailConfig, GuardrailSystem
  from core.multi_agent import ChainOfDebateOrchestrator

  cfg = GuardrailConfig()
  gs = GuardrailSystem(cfg)
  orchestrator = ChainOfDebateOrchestrator(gs)
  res = orchestrator.debate("Who is the CEO of Google?", num_agents=3, rounds=2)

Notes:
- This module is intentionally lightweight and intended as a starting point.
  Extend it with broker/queue, worker processes, advanced adjudication, and
  RL-based verifiers as needed.
"""

import asyncio
import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from .ollama_client import ollama_generate
from .guardrail_implementation import GuardrailSystem, GuardrailConfig


def _extract_json_from_text(text: str) -> dict:
    """Attempt to extract the first JSON object from arbitrary text.
    - Strips common markdown/code fences (``` or ```json) before searching.
    - Returns a dict on success or None on failure.
    """
    if not text:
        return None
    cleaned = text.strip()
    # Remove surrounding ``` fences if present
    if cleaned.startswith("```") and cleaned.endswith("```"):
        parts = cleaned.split("```")
        cleaned = " ".join(p for p in parts if p.strip() and not p.strip().lower().startswith("json"))
    # Try to find a JSON object in the cleaned text
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not m:
        return None
    candidate = m.group(0)
    try:
        return json.loads(candidate)
    except Exception:
        return None


class AgentBase:
    """Simple agent interface. Subclass for specialized agents."""

    def __init__(self, agent_id: str, system_prompt: str = "You are a helpful assistant."):
        self.agent_id = agent_id
        self.system_prompt = system_prompt

    async def run(self, instruction: str, context: Optional[str] = None, **kwargs) -> Dict:
        """
        Run the agent on an instruction. Returns a dict with fields:
          - agent_id, content, elapsed
        """
        start = time.time()
        prompt = instruction
        if context:
            prompt = f"Context:\n{context}\n\n{instruction}"

        # FIX: use get_running_loop() instead of get_event_loop() to work inside FastAPI
        loop = asyncio.get_running_loop()
        # FIX: default model changed from phi3 to qwen0.5
        model = kwargs.get("model_name") or "qwen0.5"
        timeout = kwargs.get("timeout", 120)

        try:
            content = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: ollama_generate(prompt=prompt, system=self.system_prompt, model_name=model),
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return {"agent_id": self.agent_id, "content": f"[ERROR] agent timeout after {timeout}s", "elapsed": time.time() - start}
        except Exception as e:
            return {"agent_id": self.agent_id, "content": f"[ERROR] agent failed: {e}", "elapsed": time.time() - start}

        return {"agent_id": self.agent_id, "content": content.strip(), "elapsed": time.time() - start}


class LLMAgent(AgentBase):
    """Generic LLM-backed specialist agent. Currently thin wrapper around AgentBase."""
    pass


class ChainOfDebateOrchestrator:
    """
    Orchestrator that runs a chain-of-debate with multiple LLM agents and a final judge.

    Workflow (MVP):
      1. Input validation via GuardrailSystem.input_guardrail
      2. Optionally fetch RAG context via GuardrailSystem
      3. Spawn N proponents (agents) that each propose an answer + reasoning
      4. Spawn N critics (can be same agent type with different system prompt) that
         rebut other agents' proposals for a number of rounds
      5. Call Judge agent to evaluate proposals + rebuttals and produce final answer
      6. Run final output through GuardrailSystem.output_guardrail.verify
      7. Return structured record with provenance and guardrail verdicts
    """

    def __init__(self, guardrail_system: GuardrailSystem):
        self.gs = guardrail_system

    def _build_agent_prompts(self, role: str) -> List[str]:
        """Return system prompts for agents depending on role."""
        if role == "proponent":
            return [
                "You are an advocate. Provide a concise answer and supporting reasons, and cite any assumptions.",
                "You are an analytic proponent. Give a direct answer, then a short chain-of-thought style justification.",
                "You are a careful explainer. State your final answer first, then provide the top 2 reasons in bullet points.",
            ]
        if role == "critic":
            return [
                "You are a critic. Identify possible errors, missing evidence, or hallucinations in the provided answers. Be brief and factual.",
                "You are a skeptic. Provide counter-arguments and point out contradictions or unsupported claims.",
            ]
        if role == "judge":
            return [
                "You are an impartial judge. Given multiple proposals and rebuttals, pick the most supported answer, explain why, and cite any remaining uncertainties. Output JSON: {\"final_answer\":..., \"rationale\":..., \"confidence\":0-1}",
            ]
        return ["You are a helpful assistant. Answer succinctly."]

    async def _call_agents(self, agents: List[AgentBase], instruction: str, context: Optional[str] = None, model_name: Optional[str] = None) -> List[Dict]:
        coros = [agent.run(instruction, context=context, model_name=model_name) for agent in agents]
        results = await asyncio.gather(*coros, return_exceptions=False)
        return results

    def _retrieve_context(self, query: str) -> Tuple[Optional[str], Dict[str, Any]]:
        """Use GuardrailSystem's retrievers to produce context and metadata."""
        try:
            primary_docs, wiki_docs = self.gs._retrieve_from_both_kbs(query)
            all_docs = []
            metadata = {"primary_count": len(primary_docs), "wiki_count": len(wiki_docs)}
            if primary_docs:
                all_docs.append("### PRIMARY_KB")
                all_docs.extend(primary_docs)
            if wiki_docs:
                all_docs.append("### WIKIPEDIA_KB")
                all_docs.extend(wiki_docs)
            context = None
            if all_docs:
                context = "\n\n".join(all_docs)
            return context, metadata
        except Exception as e:
            return None, {"error": str(e)}

    # FIX: debate() is now async to work properly inside FastAPI's event loop
    async def debate(self, question: str, num_agents: int = 3, rounds: int = 2, model_name: str = "qwen0.5") -> Dict:
        """
        Run a chain-of-debate on `question`.

        Returns a structured dict:
          {
            "question": str,
            "input_validation": {...},
            "context_metadata": {...},
            "rounds": [ {"proposals": [...], "rebuttals": [...]}, ... ],
            "judge": {...},
            "final_verdict": {...}
          }
        """
        # STEP 0 — Input validation
        ig_result = self.gs.input_guardrail.validate(question) if self.gs and self.gs.input_guardrail else {"valid": True}
        if not ig_result.get("valid", True):
            return {"question": question, "verdict": "blocked_input", "input_validation": ig_result}

        # STEP 1 — Retrieve context
        context, ctx_meta = self._retrieve_context(question) if self.gs else (None, {})

        # STEP 2 — Create proponent agents
        proponent_prompts = self._build_agent_prompts("proponent")
        proponents: List[AgentBase] = []
        for i in range(num_agents):
            prompt = proponent_prompts[i % len(proponent_prompts)]
            proponents.append(LLMAgent(agent_id=f"proponent_{i+1}", system_prompt=prompt))

        # STEP 3 — Initial proposals (FIX: await directly, no new event loop)
        proposals = await self._call_agents(
            proponents,
            f"Question: {question}\nProvide answer and reasoning.",
            context=context,
            model_name=model_name,
        )

        rounds_data: List[Dict] = []
        current_statements = [p["content"] for p in proposals]

        # STEP 4 — Rounds of rebuttal
        for r in range(rounds):
            critic_prompts = self._build_agent_prompts("critic")
            critics: List[AgentBase] = []
            for i in range(num_agents):
                prom = critic_prompts[i % len(critic_prompts)]
                critics.append(LLMAgent(agent_id=f"critic_{r+1}_{i+1}", system_prompt=prom))

            instruction = "Given the following proposals, provide concise critiques for each.\n\n"
            for idx, stmt in enumerate(current_statements, start=1):
                instruction += f"Proposal {idx}: {stmt}\n\n"
            instruction += "For each proposal, list up to 2 potential errors, unsupported claims, or required evidence. Be concise."

            rebuttals = await self._call_agents(critics, instruction, context=context, model_name=model_name)

            rounds_data.append({
                "round": r + 1,
                "proposals": proposals,
                "rebuttals": rebuttals,
            })

            synth_instruction = "Review the rebuttals below and revise/defend your original short answer in 1-2 sentences.\n\n"
            synth_instruction += "Rebuttals:\n"
            for rb in rebuttals:
                synth_instruction += f"- {rb['agent_id']}: {rb['content']}\n"

            proponents_defend = await self._call_agents(proponents, synth_instruction, context=context, model_name=model_name)
            proposals = proponents_defend
            current_statements = [p["content"] for p in proposals]

        # STEP 5 — Judge
        judge_prompt = self._build_agent_prompts("judge")[0]
        judge_agent = LLMAgent(agent_id="judge", system_prompt=judge_prompt)

        judge_instruction = "You are the judge. Review the following proposals and rebuttals and produce a final answer. Output strict JSON with keys: final_answer, rationale, confidence (0-1).\n\n"
        judge_instruction += "Proposals:\n"
        for idx, p in enumerate(proposals, start=1):
            judge_instruction += f"Proposal {idx} ({p['agent_id']}): {p['content']}\n"
        judge_instruction += "\nAll rebuttals:\n"
        for rd in rounds_data:
            for rb in rd["rebuttals"]:
                judge_instruction += f"{rb['agent_id']}: {rb['content']}\n"

        judge_result = await judge_agent.run(judge_instruction, context=context, model_name=model_name)

        final_answer = judge_result.get("content", "")
        parsed_judge = {"raw": final_answer}

        extracted = _extract_json_from_text(final_answer)
        if extracted:
            parsed_judge = {**parsed_judge, **extracted}
        else:
            parsed_judge["parse_error"] = True
            parsed_judge["rationale_fallback"] = final_answer

        # If judge referenced a proposal by number, resolve it
        try:
            candidate = (parsed_judge.get("final_answer") or parsed_judge.get("raw", ""))
            if isinstance(candidate, str):
                m = re.match(r"^\s*proposal\s*(\d+)\s*$", candidate.strip(), re.IGNORECASE)
                if m:
                    idx = int(m.group(1))
                    if 1 <= idx <= len(proposals):
                        chosen = proposals[idx - 1].get("content", "")
                        parsed_judge["chosen_proposal"] = {"index": idx, "agent_id": proposals[idx - 1].get("agent_id")}
                        parsed_judge["final_answer"] = chosen
                else:
                    m2 = re.search(r"proposal\s*(\d+)", parsed_judge.get("raw", ""), re.IGNORECASE)
                    if m2:
                        idx = int(m2.group(1))
                        if 1 <= idx <= len(proposals):
                            chosen = proposals[idx - 1].get("content", "")
                            parsed_judge["chosen_proposal"] = {"index": idx, "agent_id": proposals[idx - 1].get("agent_id")}
                            parsed_judge["final_answer"] = chosen
        except Exception:
            pass

        # STEP 6 — Output guardrail verification
        output_verification = self.gs.output_guardrail.verify(
            query=question,
            response=parsed_judge.get("final_answer", parsed_judge.get("raw", "")),
            context=context,
        )

        return {
            "question": question,
            "input_validation": ig_result,
            "context_metadata": ctx_meta,
            "context_used": bool(context),
            "rounds": rounds_data,
            "judge": parsed_judge,
            "output_verification": output_verification,
        }


# FIX: run_debate_sync now uses asyncio.run() since debate() is now async
def run_debate_sync(guardrail_system: GuardrailSystem, question: str, num_agents: int = 3, rounds: int = 2, model_name: str = "qwen0.5") -> Dict:
    orchestrator = ChainOfDebateOrchestrator(guardrail_system)
    return asyncio.run(orchestrator.debate(question, num_agents=num_agents, rounds=rounds, model_name=model_name))


if __name__ == "__main__":
    print("Chain-of-Debate multi-agent orchestrator demo. This will initialize GuardrailSystem (may be slow).")
    cfg = GuardrailConfig()
    gs = GuardrailSystem(cfg)
    orchestrator = ChainOfDebateOrchestrator(gs)
    q = "Who is the current CEO of Google?"
    print("Running debate for question:", q)
    res = asyncio.run(orchestrator.debate(q, num_agents=2, rounds=1))
    print(json.dumps(res, indent=2))