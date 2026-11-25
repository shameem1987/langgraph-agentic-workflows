"""Pattern 3: HUMAN-IN-THE-LOOP. The graph pauses BEFORE the risky step (sending the reply)
and only continues after a person approves or edits the draft.

Run: python examples/03_human_in_loop.py
"""
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from agentic.llm import get_llm

llm = get_llm()


class State(TypedDict, total=False):
    ticket: str
    draft: str
    sent: bool


def draft(s: State) -> State:
    return {"draft": llm.invoke(f"draft: reply to ticket: {s['ticket']}")}


def send(s: State) -> State:
    print(f"\n>>> SENT: {s['draft']}")
    return {"sent": True}


g = StateGraph(State)
g.add_node("draft", draft)
g.add_node("send", send)
g.set_entry_point("draft")
g.add_edge("draft", "send")
g.add_edge("send", END)
app = g.compile(checkpointer=MemorySaver(), interrupt_before=["send"])

if __name__ == "__main__":
    cfg = {"configurable": {"thread_id": "ticket-42"}}
    app.invoke({"ticket": "Customer wants a refund"}, cfg)          # runs until the pause
    state = app.get_state(cfg)
    print("DRAFT FOR REVIEW:\n", state.values["draft"])
    choice = input("\n[a]pprove / [e]dit / [r]eject? ").strip().lower()
    if choice == "r":
        print("Rejected. Nothing sent.")
    else:
        if choice == "e":
            app.update_state(cfg, {"draft": input("New draft: ")})
        app.invoke(None, cfg)                                         # resume
