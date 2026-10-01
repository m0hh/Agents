"""
Tests for hello.py: building the request body and explaining the response.

To check what gets PRINTED, we swap the module's `console` for one that
writes into a string buffer (io.StringIO) instead of the terminal, then look
for the text we expect. pytest's `monkeypatch` fixture restores the original
automatically after each test.
"""

import io

import pytest
from rich.console import Console

from agent_loop import hello
from agent_loop.config import Settings

FAKE_KEY = "sk-ant-test-0000000000abcd"


@pytest.fixture
def captured_output(monkeypatch: pytest.MonkeyPatch) -> io.StringIO:
    """Redirect hello.py's console into a buffer we can read."""
    buffer = io.StringIO()
    # force_terminal=False + color_system=None → plain text, no ANSI style codes
    # (even if the environment sets FORCE_COLOR), so our string checks are simple.
    plain_console = Console(file=buffer, width=200, force_terminal=False, color_system=None)
    monkeypatch.setattr(hello, "console", plain_console)
    return buffer


def test_minimal_payload_has_only_the_three_required_fields() -> None:
    payload = hello.build_payload(Settings(api_key=FAKE_KEY), "Hi", system=None)

    assert payload == {
        "model": "claude-haiku-4-5",
        "max_tokens": 16_000,
        "messages": [{"role": "user", "content": "Hi"}],
    }


def test_payload_with_system_prompt_and_thinking() -> None:
    settings = Settings(api_key=FAKE_KEY, thinking_budget=2048)
    payload = hello.build_payload(settings, "Hi", system="Be brief.")

    # The system prompt is a TOP-LEVEL field, not a message in the list.
    assert payload["system"] == "Be brief."
    assert payload["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert payload["messages"] == [{"role": "user", "content": "Hi"}]


def test_cost_estimate_matches_snapshot_model_names() -> None:
    usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
    # $1 per 1M input + $5 per 1M output = $6
    assert hello.estimate_cost("claude-haiku-4-5-20251001", usage) == pytest.approx(6.0)
    assert hello.estimate_cost("unknown-model", usage) is None


def test_explain_response_describes_every_block_type(captured_output: io.StringIO) -> None:
    body = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-4-5-20251001",
        "content": [
            {"type": "thinking", "thinking": "Let me consider [this] carefully.", "signature": "abc123"},
            {"type": "text", "text": "The answer is 42."},
        ],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 20, "output_tokens": 30},
    }

    hello.explain_response(Settings(api_key=FAKE_KEY), body)
    output = captured_output.getvalue()

    assert "thinking" in output
    assert "[this]" in output  # square brackets survive (not treated as rich markup)
    assert "signature: 6 chars" in output
    assert "The answer is 42." in output
    assert "end_turn" in output
    assert "$0.000170" in output  # (20 * $1 + 30 * $5) / 1M


def test_bad_thinking_budget_on_the_command_line_is_a_config_error(
    captured_output: io.StringIO, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Make load_settings() see a fake key without touching the real .env / environment.
    monkeypatch.setattr(hello, "load_settings", lambda: Settings(api_key=FAKE_KEY))

    exit_code = hello.main(["Hi", "--thinking", "500"])

    assert exit_code == 2
    assert "Config error" in captured_output.getvalue()
