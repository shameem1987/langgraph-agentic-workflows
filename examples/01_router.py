"""Pattern 1: ROUTER. Classify the request, then branch to a specialist node.

Run: python examples/01_router.py "refund my order"
"""
import sys
from typing import TypedDict

from langgraph.graph import END, StateGraph

from agentic.llm import get_llm
from agentic.tools import calculator

llm = get_llm()


class State(TypedDict, total=False):
    question: str
    route: str
    answer: str


def classify(s: State) -> State:
    label = llm.invoke(f"classify: {s['question']}").strip().lower()
    return {"route": label if label in {"math", "refund", "general"} else "general"}


def math_node(s: State) -> State:
    return {"answer": calculator(s["question"].split("?")[0].split("is")[-1].strip())}


def refund_node(s: State) -> State:
    return {"answer": llm.invoke(f"draft: refund reply for {s['question']}")}


def general_node(s: State) -> State:
    return {"answer": llm.invoke(s["question"])}


g = StateGraph(State)
for name, fn in [("classify", classify), ("math", math_node), ("refund", refund_node), ("general", general_node)]:
    g.add_node(name, fn)
g.set_entry_point("classify")
g.add_conditional_edges("classify", lambda s: s["route"],
                        {"math": "math", "refund": "refund", "general": "general"})
for n in ("math", "refund", "general"):
    g.add_edge(n, END)
app = g.compile()

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What is 12*7?"
    print(app.invoke({"question": q})["answer"])
