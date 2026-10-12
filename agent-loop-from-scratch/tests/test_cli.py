"""
Tests for cli.py: drive the chat REPL with scripted keyboard input.

`ChatREPL` takes a `read_line` function. Instead of the keyboard, we give it a
function that returns pre-written lines one by one (and raises EOFError,
like Ctrl-D, when they run out). Output goes to a string buffer we inspect.
"""

import io
from collections.abc import Callable

from rich.console import Console

from agent_loop.agent import Agent
from agent_loop.cli import ChatREPL
from agent_loop.config import Settings


def scripted_input(*lines: str) -> Callable[[], str]:
    """Return a read_line function that 'types' these lines, then presses Ctrl-D."""
    remaining = list(lines)

    def read_line() -> str:
        if not remaining:
            raise EOFError
        return remaining.pop(0)

    return read_line


def run_chat(agent: Agent, *lines: str) -> str:
    """Run the REPL with scripted input and return everything it printed (plain text)."""
    buffer = io.StringIO()
    out = Console(file=buffer, width=150, force_terminal=False, color_system=None)
    ChatREPL(agent, out, read_line=scripted_input(*lines)).run()
    return buffer.getvalue()


def test_a_short_chat_with_commands(fake_api, settings: Settings) -> None:
    fake_api.reply_text("Hello **Sam**!", input_tokens=30, output_tokens=8)
    fake_api.reply_text("You are Sam.", input_tokens=52, output_tokens=6)
    agent = Agent(settings, fake_api.client(settings))

    output = run_chat(agent, "Hi, I'm Sam", "What's my name?", "/history", "/stats", "/exit")

    assert "Hello Sam!" in output  # Markdown was rendered (no ** left)
    assert "You are Sam." in output
    assert "in 52 tok (+22)" in output  # the footer shows the input growth
    assert '"role": "assistant"' in output  # /history printed the raw JSON
    assert "Token usage per turn" in output  # /stats printed the table
    assert "Bye!" in output


def test_an_api_error_is_shown_and_the_chat_continues(fake_api, settings: Settings) -> None:
    fake_api.reply_error(529, "overloaded_error", "Overloaded")
    fake_api.reply_text("Recovered!")
    agent = Agent(settings, fake_api.client(settings))

    output = run_chat(agent, "first try", "second try")

    assert "API error 529 overloaded_error" in output
    assert "Nothing was added to the history" in output
    assert "Recovered!" in output
    assert len(agent.conversation) == 2  # only the successful exchange


def test_reset_and_unknown_commands(fake_api, settings: Settings) -> None:
    fake_api.reply_text("hi")
    agent = Agent(settings, fake_api.client(settings))

    output = run_chat(agent, "hello", "/reset", "/history", "/nonsense")

    assert "Started a new conversation" in output
    assert "The history is empty" in output
    assert "Unknown command '/nonsense'" in output


def test_cut_off_answers_are_flagged(fake_api, settings: Settings) -> None:
    fake_api.reply_text("Once upon a", stop_reason="max_tokens")
    agent = Agent(settings, fake_api.client(settings))

    output = run_chat(agent, "Tell me a story")

    assert "The answer was cut off" in output
