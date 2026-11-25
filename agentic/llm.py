"""LLM access. `FakeLLM` makes every example runnable and testable offline."""
from __future__ import annotations

import os
import re


class FakeLLM:
    """Deterministic stand-in. Understands the tiny prompts used in /examples."""

    def invoke(self, prompt: str) -> str:
        p = prompt.lower()
        if p.startswith("classify:"):
            q = p[len("classify:"):]
            if re.search(r"\d\s*[-+*/]\s*\d", q):
                return "math"
            return "refund" if "refund" in q else "general"
        if p.startswith("tool-decision:"):
            m = re.search(r"(\d[\d\s.+\-*/()%]*\d)", p)
            obs = re.search(r"observation:[ \t]*(\S[^\n]*)", p)
            if obs:
                return f"FINAL {obs.group(1).strip()}"
            return f"CALL calculator {m.group(1).strip()}" if m else "FINAL no tool needed"
        if p.startswith("draft:"):
            return "Hi, we are sorry about the trouble. Your refund has been issued."
        return "OK"


class AnthropicLLM:
    def __init__(self, model: str | None = None) -> None:
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model or os.getenv("AGENT_MODEL", "claude-sonnet-5-5")

    def invoke(self, prompt: str) -> str:
        r = self.client.messages.create(model=self.model, max_tokens=500,
                                        messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")


def get_llm():
    return AnthropicLLM() if os.getenv("ANTHROPIC_API_KEY") else FakeLLM()
