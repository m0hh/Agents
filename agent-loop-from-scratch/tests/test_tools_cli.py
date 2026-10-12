"""
Tests for tools_cli.py: `tools list`, `tools call` and `tools ask`.
"""

import io
from pathlib import Path

from rich.console import Console

from agent_loop.config import Settings
from agent_loop.tools import default_registry
from agent_loop.tools_cli import run_ask, run_call, run_list


def plain_console() -> tuple[Console, io.StringIO]:
    buffer = io.StringIO()
    return Console(file=buffer, width=200, force_terminal=False, color_system=None), buffer


def test_list_shows_every_definition(tmp_path: Path) -> None:
    out, buffer = plain_console()
    run_list(default_registry(tmp_path), out)
    text = buffer.getvalue()
    for name in ("calculator", "read_file", "fetch_url"):
        assert f'"name": "{name}"' in text


def test_call_runs_the_tool_and_shows_the_tool_result_block(tmp_path: Path) -> None:
    out, buffer = plain_console()
    exit_code = run_call(default_registry(tmp_path), out, "calculator", '{"expression": "6 * 7"}')
    assert exit_code == 0
    assert "6 * 7 = 42" in buffer.getvalue()
    assert '"type": "tool_result"' in buffer.getvalue()


def test_call_reports_tool_errors_as_is_error(tmp_path: Path) -> None:
    out, buffer = plain_console()
    exit_code = run_call(default_registry(tmp_path), out, "read_file", '{"path": "../outside.txt"}')
    assert exit_code == 1
    assert "outside the workspace" in buffer.getvalue()
    assert '"is_error": true' in buffer.getvalue()


def test_call_explains_bad_json_and_unknown_tools(tmp_path: Path) -> None:
    out, buffer = plain_console()
    assert run_call(default_registry(tmp_path), out, "calculator", "{not json}") == 1
    assert run_call(default_registry(tmp_path), out, "teleport", "{}") == 1
    assert "isn't valid JSON" in buffer.getvalue()
    assert "No tool named 'teleport'" in buffer.getvalue()


def test_ask_sends_the_tools_and_shows_the_tool_use_block(fake_api, settings: Settings, tmp_path: Path) -> None:
    registry = default_registry(tmp_path)
    fake_api.reply(
        [{"type": "tool_use", "id": "toolu_123", "name": "calculator", "input": {"expression": "2**32 / 7"}}],
        stop_reason="tool_use",
    )
    out, buffer = plain_console()

    run_ask(settings, registry, fake_api.client(settings), out, "What is 2**32 / 7?")

    request = fake_api.requests[0]
    assert request["tools"] == registry.definitions()  # all definitions were sent
    text = buffer.getvalue()
    assert "toolu_123" in text
    assert "WAITING for a tool_result" in text
    assert "uv run tools call calculator" in text  # the copy-paste hint


def test_ask_without_tools_sends_no_tools(fake_api, settings: Settings, tmp_path: Path) -> None:
    fake_api.reply_text("About 613566756.57")
    out, _ = plain_console()

    run_ask(settings, default_registry(tmp_path), fake_api.client(settings), out, "2**32/7?", use_tools=False)

    assert "tools" not in fake_api.requests[0]
