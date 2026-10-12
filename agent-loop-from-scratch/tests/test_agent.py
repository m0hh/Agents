"""
Tests for agent.py: does each turn send the right history, and is the history
always left in a valid state?

These tests use the `fake_api` and `settings` fixtures from conftest.py.
"""

import dataclasses

import pytest

from agent_loop.agent import DEFAULT_SYSTEM_PROMPT, Agent
from agent_loop.api_client import ClaudeHTTPError
from agent_loop.config import Settings


def test_first_turn_sends_the_system_prompt_and_one_user_message(fake_api, settings: Settings) -> None:
    fake_api.reply_text("Hello, Sam!")
    agent = Agent(settings, fake_api.client(settings))

    result = agent.run_turn("Hi, I'm Sam.")

    assert fake_api.requests[0] == {
        "model": "claude-haiku-4-5",
        "max_tokens": 16_000,
        "system": DEFAULT_SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": "Hi, I'm Sam."}],
    }
    assert result.text == "Hello, Sam!"
    assert result.committed is True
    assert result.messages_sent == 1
    assert len(agent.conversation) == 2  # user + assistant


def test_second_turn_resends_the_whole_history(fake_api, settings: Settings) -> None:
    # THE key behaviour of Phase 2: memory = resending everything.
    first_answer = [
        {"type": "thinking", "thinking": "They told me their name.", "signature": "sig123"},
        {"type": "text", "text": "Nice to meet you, Sam!"},
    ]
    fake_api.reply(first_answer, input_tokens=20)
    fake_api.reply_text("Your name is Sam.", input_tokens=45)
    agent = Agent(settings, fake_api.client(settings))

    agent.run_turn("Hi, I'm Sam.")
    second = agent.run_turn("What's my name?")

    assert fake_api.requests[1]["messages"] == [
        {"role": "user", "content": "Hi, I'm Sam."},
        # The assistant content comes back VERBATIM, thinking block and signature included.
        {"role": "assistant", "content": first_answer},
        {"role": "user", "content": "What's my name?"},
    ]
    assert second.messages_sent == 3
    assert [t.input_tokens for t in agent.turns] == [20, 45]  # input grows each turn


def test_a_failed_call_leaves_the_history_unchanged(fake_api, settings: Settings) -> None:
    fake_api.reply_text("First answer")
    fake_api.reply_error(529, "overloaded_error", "Overloaded")
    fake_api.reply_text("Second answer")
    agent = Agent(settings, fake_api.client(settings))

    agent.run_turn("one")
    with pytest.raises(ClaudeHTTPError):
        agent.run_turn("two (fails)")

    # The failed user message was never committed...
    assert len(agent.conversation) == 2
    # ...so retrying produces a perfectly alternating history.
    agent.run_turn("two (retry)")
    roles = [m["role"] for m in agent.conversation.messages]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert "two (fails)" not in str(agent.conversation.messages)


def test_an_empty_answer_is_not_committed(fake_api, settings: Settings) -> None:
    fake_api.reply([], stop_reason="refusal")
    agent = Agent(settings, fake_api.client(settings))

    result = agent.run_turn("something")

    assert result.committed is False
    assert len(agent.conversation) == 0
    assert len(agent.turns) == 1  # it still cost money, so it's still counted


def test_a_cut_off_answer_is_still_committed(fake_api, settings: Settings) -> None:
    fake_api.reply_text("Once upon a ti", stop_reason="max_tokens")
    agent = Agent(settings, fake_api.client(settings))

    result = agent.run_turn("Tell me a story")

    # A truncated answer is still a valid message; the user can say "continue".
    assert result.stop_reason == "max_tokens"
    assert result.committed is True


def test_thinking_setting_adds_the_thinking_field(fake_api, settings: Settings) -> None:
    thinking_settings = dataclasses.replace(settings, thinking_budget=2048)
    fake_api.reply_text("ok")
    agent = Agent(thinking_settings, fake_api.client(thinking_settings))

    agent.run_turn("hi")

    assert fake_api.requests[0]["thinking"] == {"type": "enabled", "budget_tokens": 2048}


def test_reset_starts_a_new_empty_conversation(fake_api, settings: Settings) -> None:
    fake_api.reply_text("first")
    fake_api.reply_text("second")
    agent = Agent(settings, fake_api.client(settings))
    agent.run_turn("one")

    agent.reset()
    agent.run_turn("fresh start")

    assert fake_api.requests[1]["messages"] == [{"role": "user", "content": "fresh start"}]
    assert len(agent.turns) == 1
