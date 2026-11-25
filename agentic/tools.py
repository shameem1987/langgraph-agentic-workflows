"""Small, safe tools for the examples. No eval(), no shell."""
from __future__ import annotations

import ast
import operator as op

_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
        ast.Pow: op.pow, ast.USub: op.neg, ast.Mod: op.mod}


def calculator(expression: str) -> str:
    """Evaluate +, -, *, /, **, % arithmetic only."""
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            if isinstance(node.op, ast.Pow) and abs(ev(node.right)) > 100:
                raise ValueError("exponent too large")
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("unsupported expression")
    try:
        return str(ev(ast.parse(expression, mode="eval").body))
    except (ValueError, SyntaxError, ZeroDivisionError) as e:
        return f"error: {e}"


def word_count(text: str) -> str:
    return str(len(text.split()))


TOOLS = {"calculator": calculator, "word_count": word_count}
