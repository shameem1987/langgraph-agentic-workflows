from agentic.llm import FakeLLM
from agentic.tools import calculator, word_count


def test_calculator_basic_and_precedence():
    assert calculator("15*4+2") == "62"
    assert calculator("2**3") == "8"
    assert calculator("-(3+4)") == "-7"


def test_calculator_rejects_code_and_bad_input():
    assert calculator("__import__('os').system('echo hi')").startswith("error")
    assert calculator("1/0").startswith("error")
    assert calculator("2**1000").startswith("error")
    assert calculator("abc").startswith("error")


def test_word_count():
    assert word_count("a b  c") == "3"


def test_fake_llm_routing_and_tool_decision():
    llm = FakeLLM()
    assert llm.invoke("classify: what is 12*7?") == "math"
    assert llm.invoke("classify: I want a refund") == "refund"
    assert llm.invoke("classify: hello") == "general"
    first = llm.invoke("tool-decision: What is 15*4+2?\nobservation: ")
    assert first == "CALL calculator 15*4+2"
    assert llm.invoke("tool-decision: What is 15*4+2?\nobservation: 62\nReply...") == "FINAL 62"
