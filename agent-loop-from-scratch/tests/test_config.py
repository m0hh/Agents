"""
Tests for config.py.

HOW THESE TESTS STAY INDEPENDENT OF YOUR MACHINE
    `load_settings(environ=...)` accepts a plain dict. When we pass one, it
    reads ONLY that dict: it doesn't load `.env` and doesn't look at your real
    environment variables. So the tests behave the same on every computer,
    with or without a real API key.

Run all tests with:   uv run pytest
"""

import dataclasses

import pytest

from agent_loop.config import DEFAULT_MODEL, ConfigError, Settings, load_settings

# A fake key: tests never talk to the real API, so it only needs to be non-empty.
FAKE_KEY = "sk-ant-test-0000000000abcd"


def test_defaults_are_applied_when_only_the_key_is_set() -> None:
    settings = load_settings(environ={"ANTHROPIC_API_KEY": FAKE_KEY})

    assert settings.api_key == FAKE_KEY
    assert settings.workspace_id is None
    assert settings.model == DEFAULT_MODEL == "claude-haiku-4-5"
    assert settings.max_tokens == 16_000
    assert settings.thinking_budget == 0
    assert settings.thinking_enabled is False


def test_values_are_read_from_the_environment() -> None:
    settings = load_settings(
        environ={
            "ANTHROPIC_API_KEY": FAKE_KEY,
            "ANTHROPIC_WORKSPACE_ID": "wrkspc_test",
            "AGENT_MODEL": "some-other-model",
            "AGENT_MAX_TOKENS": "8_000",  # underscores are allowed
            "AGENT_THINKING_BUDGET": "2048",
            "AGENT_TIMEOUT_SECONDS": "7.5",
        }
    )

    assert settings.workspace_id == "wrkspc_test"
    assert settings.model == "some-other-model"
    assert settings.max_tokens == 8000
    assert settings.thinking_budget == 2048
    assert settings.thinking_enabled is True
    assert settings.timeout_seconds == 7.5


def test_missing_api_key_fails_with_a_clear_message() -> None:
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        load_settings(environ={})


def test_non_numeric_value_names_the_bad_variable() -> None:
    with pytest.raises(ConfigError, match="AGENT_MAX_TOKENS"):
        load_settings(environ={"ANTHROPIC_API_KEY": FAKE_KEY, "AGENT_MAX_TOKENS": "lots"})


@pytest.mark.parametrize(
    ("budget", "max_tokens", "expected_error"),
    [
        (500, 16_000, ">= 1024"),       # below the API minimum
        (4000, 4000, "smaller than"),   # must leave room for the answer
    ],
)
def test_invalid_thinking_budgets_are_rejected(budget: int, max_tokens: int, expected_error: str) -> None:
    with pytest.raises(ConfigError, match=expected_error):
        Settings(api_key=FAKE_KEY, thinking_budget=budget, max_tokens=max_tokens)


def test_the_api_key_never_appears_in_repr() -> None:
    # If someone does print(settings) or logs it, the secret must not leak.
    settings = Settings(api_key=FAKE_KEY)
    assert FAKE_KEY not in repr(settings)


def test_settings_are_immutable() -> None:
    settings = Settings(api_key=FAKE_KEY)
    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.model = "changed"  # type: ignore[misc]
