"""
Tests for the calculator tool, including attempts to make it run code.
"""

import math

import pytest

from agent_loop.tools import ToolError
from agent_loop.tools.calculator import calculate, evaluate, make_calculator_tool


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 3 * 4", 14),                # operator precedence
        ("(2 + 3) * 4", 20),
        ("2**32 / 7", 2**32 / 7),
        ("7 // 2", 3),
        ("7 % 3", 1),
        ("-5 + +2", -3),
        ("sqrt(16)", 4.0),
        ("round(2.567, 2)", 2.57),
        ("max(3, 9, 4)", 9),
        ("2 * pi", 2 * math.pi),
        ("(120 + 45.5 + 300) * 1.14", (120 + 45.5 + 300) * 1.14),
    ],
)
def test_arithmetic(expression: str, expected: float) -> None:
    assert evaluate(expression) == pytest.approx(expected)


def test_result_text_echoes_the_expression() -> None:
    assert calculate("6 * 7") == "6 * 7 = 42"


@pytest.mark.parametrize(
    "attack",
    [
        "__import__('os').system('echo hacked')",   # import + call
        "(1).__class__.__bases__",                  # attribute access
        "open('/etc/passwd').read()",               # function not on the allow-list
        "[x for x in range(10)]",                   # comprehension
        "lambda: 1",                                # lambda
        "'abc' * 3",                                # strings aren't numbers
        "x = 5",                                    # statements aren't expressions
    ],
)
def test_anything_but_math_is_rejected(attack: str) -> None:
    with pytest.raises(ToolError):
        evaluate(attack)


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("1 / 0", "Division by zero"),
        ("9**9**9", "too large"),                        # would have ~370 million digits
        ("((10**100)**10)**10", "digits"),               # small exponents, huge result
        ("2 ^ 8", r"Use \*\*"),                          # helpful hint for a common mistake
        ("2,340 * 2", "Commas"),                         # thousands separator mistake
        ("sqrt(-1)", "sqrt"),                            # math domain error
        ("(-8) ** (1/3)", "complex"),
        ("1" * 301, "too long"),
    ],
)
def test_errors_explain_what_went_wrong(expression: str, message: str) -> None:
    with pytest.raises(ToolError, match=message):
        evaluate(expression)


def test_the_tool_wraps_the_function() -> None:
    tool = make_calculator_tool()
    assert tool.name == "calculator"
    assert tool.execute({"expression": "1 + 1"}) == "1 + 1 = 2"
