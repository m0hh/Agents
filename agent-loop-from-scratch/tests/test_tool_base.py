"""
Tests for tools/base.py (the Tool type and the mini JSON Schema validator)
and for the tool registry.
"""

import pytest

from agent_loop.tools import Tool, ToolInputError, ToolRegistry, default_registry, validate_input

SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "start_line": {"type": "integer"},
        "mode": {"type": "string", "enum": ["fast", "slow"]},
    },
    "required": ["path"],
    "additionalProperties": False,
}


def test_valid_input_passes() -> None:
    validate_input(SCHEMA, {"path": "a.txt", "start_line": 3, "mode": "fast"})


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({}, "Missing required"),
        ({"path": "a", "colour": "red"}, "Unknown input field"),
        ({"path": 42}, "must be of type string"),
        ({"path": "a", "start_line": "3"}, "must be of type integer"),
        # Python quirk: True is an int! JSON booleans must still be rejected for "integer".
        ({"path": "a", "start_line": True}, "must be of type integer"),
        ({"path": "a", "mode": "medium"}, "must be one of"),
        (["not", "an", "object"], "must be a JSON object"),
    ],
)
def test_invalid_inputs_are_rejected_with_helpful_messages(data, message: str) -> None:
    with pytest.raises(ToolInputError, match=message):
        validate_input(SCHEMA, data)


def test_execute_validates_then_passes_input_as_keyword_arguments() -> None:
    def greet(path: str, start_line: int = 1, mode: str = "fast") -> str:
        return f"{path}:{start_line}:{mode}"

    tool = Tool(name="greet", description="test", input_schema=SCHEMA, function=greet)

    assert tool.execute({"path": "x", "start_line": 7}) == "x:7:fast"
    with pytest.raises(ToolInputError):
        tool.execute({"start_line": 7})  # validation runs BEFORE the function


def test_definition_has_the_shape_the_api_expects() -> None:
    tool = Tool(name="t", description="d", input_schema=SCHEMA, function=lambda **_: "")
    assert tool.definition() == {"name": "t", "description": "d", "input_schema": SCHEMA, "strict": True}


@pytest.mark.parametrize("bad_name", ["", "has space", "x" * 65, "dots.not.allowed"])
def test_invalid_tool_names_are_rejected_when_defined(bad_name: str) -> None:
    with pytest.raises(ValueError):
        Tool(name=bad_name, description="d", input_schema=SCHEMA, function=lambda **_: "")


def test_registry_refuses_duplicate_names() -> None:
    tool = Tool(name="same", description="d", input_schema=SCHEMA, function=lambda **_: "")
    with pytest.raises(ValueError, match="Duplicate"):
        ToolRegistry([tool, tool])


def test_default_registry_contents(tmp_path) -> None:
    registry = default_registry(tmp_path)
    assert registry.names == ["calculator", "read_file", "fetch_url"]
    assert registry.get("calculator") is not None
    assert registry.get("does_not_exist") is None


def _walk_keys(value):
    """Yield every dict key found anywhere inside a nested JSON-like value."""
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def test_all_default_schemas_are_strict_mode_compatible(tmp_path) -> None:
    # Encodes what we learned live: strict mode rejects keywords like "minimum"
    # (400 error), and requires "additionalProperties": false on objects.
    unsupported = {"minimum", "maximum", "minLength", "maxLength", "exclusiveMinimum", "exclusiveMaximum"}
    for definition in default_registry(tmp_path).definitions():
        schema = definition["input_schema"]
        assert definition["strict"] is True
        assert schema["additionalProperties"] is False
        assert not unsupported & set(_walk_keys(schema)), definition["name"]
        # Every tool should tell the model WHEN to use it, not just what it does.
        assert "Use this" in definition["description"], definition["name"]
