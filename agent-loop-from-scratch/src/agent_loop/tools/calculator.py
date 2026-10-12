"""
tools/calculator.py: exact arithmetic, evaluated SAFELY.

WHY GIVE A LANGUAGE MODEL A CALCULATOR?
    LLMs predict text. They don't compute, so they can confidently get
    arithmetic wrong, especially with long numbers. A tool gives an exact
    answer, every time. This is the classic first tool in every agent tutorial.

WHY NOT JUST `eval(expression)`?
    `eval` runs ANY Python code. The expression comes from a model, which may
    be steered by a malicious web page or file it read earlier (prompt
    injection). With eval, this "math" would delete your files:
        __import__('os').system('rm -rf ~')
    So we never execute the text. Instead we:
      1. PARSE it into a syntax tree with Python's `ast` module (parsing only
         reads the text; it runs nothing), and
      2. WALK the tree ourselves, allowing only numbers, + - * / // % **,
         parentheses, and a short list of math functions and constants.
    Anything else (names, attributes, imports, strings...) is rejected. This
    is called an ALLOW-LIST: we list what's permitted, rather than trying
    to list every dangerous thing (a deny-list always misses something).

EXAMPLE: the syntax tree of "2 + 3 * 4"
        BinOp(+)
        ├── Constant(2)
        └── BinOp(*)
            ├── Constant(3)
            └── Constant(4)
    `_evaluate_node` visits it recursively: 3*4 = 12, then 2+12 = 14.
"""

import ast
import math
import operator
from collections.abc import Callable
from typing import Any

from agent_loop.tools.base import Tool, ToolError

# --- What's allowed ----------------------------------------------------------

# Binary operators: `a <op> b`. Each AST operator class maps to a function.
BINARY_OPERATORS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,            # a + b
    ast.Sub: operator.sub,            # a - b
    ast.Mult: operator.mul,           # a * b
    ast.Div: operator.truediv,        # a / b   (always a float)
    ast.FloorDiv: operator.floordiv,  # a // b  (rounds down)
    ast.Mod: operator.mod,            # a % b   (remainder)
    ast.Pow: operator.pow,            # a ** b
}

# Unary operators: `-a` and `+a`.
UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Functions the expression may call, e.g. sqrt(2).
FUNCTIONS: dict[str, Callable[..., Any]] = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "floor": math.floor,
    "ceil": math.ceil,
    "log": math.log,      # natural log; log(x, base) also works
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "sin": math.sin,      # angles in radians
    "cos": math.cos,
    "tan": math.tan,
}

# Named constants, e.g. 2 * pi.
CONSTANTS: dict[str, float] = {"pi": math.pi, "e": math.e, "tau": math.tau}

# --- Safety limits -------------------------------------------------------------
# Even pure math can freeze a computer: 9**9**9 has about 370 million digits.
# These limits stop such "denial of service" expressions.
MAX_EXPRESSION_LENGTH = 300
MAX_EXPONENT = 1000
MAX_RESULT_DIGITS = 1000

# log10(2): converts a number of BITS into a number of decimal DIGITS.
DIGITS_PER_BIT = math.log10(2)


def _digit_count(value: int) -> float:
    """Roughly how many decimal digits an integer has, WITHOUT converting it to text.

    (`len(str(n))` would be exact, but Python refuses to convert integers
    with more than 4300 digits to text. It's a safety limit of its own.)
    """
    return abs(value).bit_length() * DIGITS_PER_BIT


def _check_power_is_reasonable(base: Any, exponent: Any) -> None:
    """Refuse powers whose result would be gigantic, BEFORE computing them.

    Each step of ((10**1000)**1000)**1000 has a small exponent, but the result
    would have a billion digits, and computing it would freeze the program. So
    we estimate the size of the result first: digits(base) × exponent.
    """
    if abs(exponent) > MAX_EXPONENT:
        raise ToolError(f"Exponent {exponent} is too large (limit {MAX_EXPONENT}).")
    if isinstance(base, int) and isinstance(exponent, int) and exponent > 0:
        estimated_digits = max(1.0, _digit_count(base)) * exponent
        if estimated_digits > MAX_RESULT_DIGITS:
            raise ToolError(
                f"The result would have about {estimated_digits:,.0f} digits (limit {MAX_RESULT_DIGITS})."
            )


def evaluate(expression: str) -> int | float:
    """Safely evaluate an arithmetic expression and return the number."""
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ToolError(f"Expression too long ({len(expression)} chars; limit {MAX_EXPRESSION_LENGTH}).")

    try:
        # mode="eval" means "a single expression" (not statements like `x = 1`).
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError:
        raise ToolError(
            f"Could not parse {expression!r}. Write a single Python-style arithmetic "
            "expression, e.g. '(120 + 45.5) * 1.14'."
        ) from None

    result = _evaluate_node(tree.body)

    # Final checks on the result.
    if isinstance(result, complex):
        raise ToolError("The result is a complex number (e.g. an odd root of a negative number); not supported.")
    if isinstance(result, float) and not math.isfinite(result):
        raise ToolError("The result is too large (infinite) or undefined (NaN).")
    if isinstance(result, int) and _digit_count(result) > MAX_RESULT_DIGITS:
        raise ToolError(f"The result has more than {MAX_RESULT_DIGITS} digits.")
    return result


def _evaluate_node(node: ast.AST) -> Any:
    """Recursively evaluate one node of the syntax tree, allowing only safe nodes.

    `match` compares the node's TYPE and FIELDS against each pattern in turn.
    `ast.BinOp(left=left, op=op, right=right)` matches a BinOp node and binds
    its three fields to variables, all in one line.
    """
    match node:
        # A number literal like 42 or 3.14. (bool is excluded: True is an int in Python.)
        case ast.Constant(value=value) if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value

        # A named constant like pi.
        case ast.Name(id=name) if name in CONSTANTS:
            return CONSTANTS[name]

        # `left <op> right`
        case ast.BinOp(left=left, op=op, right=right) if type(op) in BINARY_OPERATORS:
            left_value = _evaluate_node(left)
            right_value = _evaluate_node(right)
            if isinstance(op, ast.Pow):
                _check_power_is_reasonable(left_value, right_value)
            try:
                return BINARY_OPERATORS[type(op)](left_value, right_value)
            except ZeroDivisionError:
                raise ToolError("Division by zero.") from None
            except OverflowError:
                raise ToolError("The result is too large.") from None

        # `^` is XOR in Python, but people (and models) often mean "power".
        case ast.BinOp(op=ast.BitXor()):
            raise ToolError("'^' is not a power operator here. Use ** instead, e.g. 2**8.")

        # `-x` or `+x`
        case ast.UnaryOp(op=op, operand=operand) if type(op) in UNARY_OPERATORS:
            return UNARY_OPERATORS[type(op)](_evaluate_node(operand))

        # `sqrt(2)`: only plain calls to allowed functions, no keyword arguments.
        case ast.Call(func=ast.Name(id=name), args=args, keywords=[]) if name in FUNCTIONS:
            values = [_evaluate_node(arg) for arg in args]
            try:
                return FUNCTIONS[name](*values)
            except (ValueError, TypeError, OverflowError) as exc:
                raise ToolError(f"{name}() failed: {exc}.") from None

        # "2,340" parses as a TUPLE (2, 340): a classic mistake with thousands separators.
        case ast.Tuple():
            raise ToolError("Commas aren't allowed: write 2340, not 2,340 (and one expression per call).")

        # Anything else: names, attributes, strings, comprehensions, lambdas...
        case _:
            allowed = ", ".join(FUNCTIONS)
            raise ToolError(
                f"Unsupported element in the expression: {type(node).__name__}. Only numbers, "
                f"+ - * / // % **, parentheses, the constants pi/e/tau and these functions are "
                f"allowed: {allowed}."
            )


def calculate(expression: str) -> str:
    """The tool's function: evaluate and return a short, unambiguous result text."""
    result = evaluate(expression)
    # Echoing the expression helps the model connect the result to its request.
    return f"{expression} = {result!r}"


def make_calculator_tool() -> Tool:
    """Build the calculator Tool (name + description + schema + function)."""
    return Tool(
        name="calculator",
        # The description is a PROMPT: it tells the model what the tool does
        # AND when to use it. "Never do arithmetic in your head" noticeably
        # increases how often models reach for the tool.
        description=(
            "Evaluate an arithmetic expression and return the exact result. Use this for ANY "
            "calculation the answer needs, even simple ones; never do arithmetic in your head. "
            "Supports + - * / // % ** and parentheses, the constants pi, e, tau, and the functions "
            "sqrt, abs, round, min, max, floor, ceil, log, log10, log2, exp, sin, cos, tan (radians). "
            "One expression per call."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Python-style arithmetic, e.g. '(120 + 45.5) * 1.14' or 'sqrt(2) * 10'. Use ** for powers.",
                },
            },
            "required": ["expression"],
            # No other keys allowed. Required by strict mode, and good hygiene anyway.
            "additionalProperties": False,
        },
        function=calculate,
    )
