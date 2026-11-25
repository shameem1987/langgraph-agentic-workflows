# langgraph-agentic-workflows

Three reference patterns for building agent workflows with [LangGraph](https://langchain-ai.github.io/langgraph/), each small enough to read in one sitting:

| Example | Pattern | Why it matters |
|---|---|---|
| `examples/01_router.py` | **Router**: classify, then branch to a specialist | Cheap, predictable multi-skill agents |
| `examples/02_tool_loop.py` | **Tool loop**: decide, act, observe, repeat (with a step limit) | Agents that use tools safely |
| `examples/03_human_in_loop.py` | **Human-in-the-loop**: pause before a risky step, resume on approval | Control over irreversible actions |

## Quick start
```bash
pip install -r requirements.txt
python examples/01_router.py "What is 12*7?"
python examples/02_tool_loop.py "What is 15*4+2?"
python examples/03_human_in_loop.py
```
Runs **offline** with a deterministic fake LLM. Set `ANTHROPIC_API_KEY` (and optionally `AGENT_MODEL`) to use a real model.

## Safety choices demonstrated
- Calculator tool uses an AST allow-list, never `eval`
- Hard step limit stops runaway loops
- Approval gate before side effects; reject path sends nothing

## Tests
```bash
pytest -q       # tools and fake-LLM logic; graph examples need `pip install langgraph`
```
