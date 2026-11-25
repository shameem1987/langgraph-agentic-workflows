"""Pattern 2: TOOL LOOP (ReAct-style). Model decides -> tool runs -> observation -> repeat.
A hard step limit prevents runaway loops.

Run: python examples/02_tool_loop.py "What is 15*4+2?"
"""
import sys
from typing import TypedDict

from langgraph.graph import END, StateGraph

from agentic.llm import get_llm
from agentic.tools import TOOLS

llm = get_llm()
MAX_STEPS = 5


class State(TypedDict, total=False):
    question: str
    scratch: str
    steps: int
    answer: str
    pending: str


def decide(s: State) -> State:
    prompt = (f"tool-decision: {s['question']}\nobservation: {s.get('scratch', '')}\n"
              "Reply 'CALL <tool> <input>' or 'FINAL <answer>'. Tools: " + ", ".join(TOOLS))
    out = llm.invoke(prompt).strip()
    steps = s.get("steps", 0) + 1
    if out.startswith("CALL") and steps <= MAX_STEPS:
        return {"pending": out, "steps": steps}
    final = out[6:] if out.startswith("FINAL") else s.get("scratch", out)
    return {"answer": final, "pending": "", "steps": steps}


def act(s: State) -> State:
    _, tool, arg = s["pending"].split(" ", 2)
    result = TOOLS[tool](arg) if tool in TOOLS else f"error: unknown tool {tool}"
    return {"scratch": result}


g = StateGraph(State)
g.add_node("decide", decide)
g.add_node("act", act)
g.set_entry_point("decide")
g.add_conditional_edges("decide", lambda s: "act" if s.get("pending") else END, {"act": "act", END: END})
g.add_edge("act", "decide")
app = g.compile()

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What is 15*4+2?"
    out = app.invoke({"question": q})
    print(out["answer"], f"(steps={out['steps']})")
