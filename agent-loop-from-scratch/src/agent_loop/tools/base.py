"""
tools/base.py: what a "tool" is, in plain Python.

A TOOL HAS TWO HALVES
    1. The DESCRIPTION the model reads (sent in the request's "tools" list):
           {"name": "calculator",
            "description": "Evaluate an arithmetic expression...",
            "input_schema": { ...JSON Schema... },
            "strict": true}
    2. The CODE that runs on YOUR machine: a normal Python function.

    The model never runs anything. When it wants a tool, it writes a
    `tool_use` block, in effect a structured request:
           {"type": "tool_use", "id": "toolu_01...", "name": "calculator",
            "input": {"expression": "2**32 / 7"}}
    and stops with stop_reason "tool_use". Your program decides whether and
    how to run it (Phase 4 closes that loop).

WHAT IS A JSON SCHEMA?
    A standard way to describe the shape of JSON data: which keys exist, their
    types, and which are required. The model reads it to know how to call the
    tool. With `"strict": true` the API also GUARANTEES that the model's input
    matches the schema.

    Tested live on Haiku 4.5: strict mode rejects some JSON Schema keywords,
    e.g. `"minimum": 0` → 400 "property 'minimum' is not supported". So
    schemas describe SHAPE (types, required keys), and rules like "must be
    at least 1" are checked in our Python code instead.

WHY VALIDATE AGAIN IF THE API GUARANTEES IT?
    Defense in depth. A future change might turn strict mode off, a test or
    another code path might call the tool directly, and the check is cheap.
    Tool inputs come from a language model, so treat them as untrusted, just
    like user input in a web form.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# The API requires tool names to look like this: letters, digits, _ or -,
# 1 to 64 characters.
TOOL_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


class ToolError(Exception):
    """An EXPECTED tool failure: bad input, file not found, page unreachable...

    The message goes back to the MODEL (in Phase 4, as a tool_result with
    "is_error": true), so write it FOR the model: say what went wrong and how
    to fix it. The model reads it and usually retries correctly.
        Bad:  "Error 2"
        Good: "No such file: 'price.csv'. Call read_file with path '.' to list files."
    """


class ToolInputError(ToolError):
    """The tool input doesn't match the tool's JSON Schema."""


# -----------------------------------------------------------------------------
# A tiny JSON Schema validator
#
# Real projects use a library (`jsonschema`, or pydantic models). We write the
# small subset we need, so you can see there's no magic: it's just checking
# keys and types. It supports: type "object" with properties, "required",
# "additionalProperties": false, simple types, and "enum".
# -----------------------------------------------------------------------------

def _matches_type(value: Any, json_type: str) -> bool:
    """Does a Python value match a JSON Schema type name?

    Watch out for a Python quirk: `bool` is a subclass of `int`, so
    `isinstance(True, int)` is True! JSON treats booleans and numbers as
    different types, so we exclude bools explicitly.
    """
    if json_type == "string":
        return isinstance(value, str)
    if json_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if json_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if json_type == "boolean":
        return isinstance(value, bool)
    if json_type == "object":
        return isinstance(value, dict)
    if json_type == "array":
        return isinstance(value, list)
    if json_type == "null":
        return value is None
    return False  # an unknown type name: treat as a mismatch


def validate_input(schema: dict[str, Any], data: Any) -> None:
    """Check `data` against an object schema. Raises ToolInputError if invalid."""
    if not isinstance(data, dict):
        raise ToolInputError("The tool input must be a JSON object.")

    properties: dict[str, Any] = schema.get("properties", {})

    missing = [name for name in schema.get("required", []) if name not in data]
    if missing:
        raise ToolInputError(f"Missing required input field(s): {', '.join(missing)}.")

    if schema.get("additionalProperties") is False:
        unknown = [name for name in data if name not in properties]
        if unknown:
            raise ToolInputError(
                f"Unknown input field(s): {', '.join(unknown)}. "
                f"Allowed: {', '.join(properties) or '(none)'}."
            )

    for name, value in data.items():
        spec = properties.get(name)
        if spec is None:
            continue
        expected_type = spec.get("type")
        if expected_type and not _matches_type(value, expected_type):
            raise ToolInputError(
                f"Input field '{name}' must be of type {expected_type}, "
                f"got {type(value).__name__} ({value!r})."
            )
        if "enum" in spec and value not in spec["enum"]:
            raise ToolInputError(f"Input field '{name}' must be one of {spec['enum']}, got {value!r}.")


# -----------------------------------------------------------------------------
# The Tool type
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Tool:
    """One tool: what the model sees, plus the Python function that does the work.

    The function receives the model's input as keyword arguments, so the
    JSON input {"path": "notes.md", "start_line": 5} becomes the Python call
    read_file(path="notes.md", start_line=5). It returns plain text (the
    tool result), or raises ToolError.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    function: Callable[..., str]
    # Ask the API to guarantee the model's input matches input_schema.
    strict: bool = True

    def __post_init__(self) -> None:
        # Catch mistakes when the tool is DEFINED, not when the API rejects it later.
        if not TOOL_NAME_PATTERN.match(self.name):
            raise ValueError(f"Invalid tool name {self.name!r}: use letters, digits, _ or -, max 64 chars.")
        if self.input_schema.get("type") != "object":
            raise ValueError(f"Tool {self.name!r}: input_schema must have \"type\": \"object\".")

    def definition(self) -> dict[str, Any]:
        """The JSON the model sees: one entry of the request's "tools" list."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "strict": self.strict,
        }

    def execute(self, tool_input: dict[str, Any]) -> str:
        """Validate the input, then run the function.

        Raises:
            ToolInputError: the input doesn't match the schema.
            ToolError: the tool failed in an expected way.
            Any other exception means a BUG in the tool. Phase 4 decides how
            to report those to the model.
        """
        validate_input(self.input_schema, tool_input)
        return self.function(**tool_input)
