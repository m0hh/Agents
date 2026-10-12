"""
Tests for conversation.py: the rules of the message history.
"""

import json

import pytest

from agent_loop.conversation import Conversation, ConversationError


def test_a_valid_exchange_produces_the_api_message_format() -> None:
    conversation = Conversation()
    conversation.append_user("Hi")
    conversation.append_assistant([{"type": "text", "text": "Hello!"}])

    assert conversation.messages == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": [{"type": "text", "text": "Hello!"}]},
    ]
    assert len(conversation) == 2
    assert conversation.next_role == "user"


def test_the_first_message_must_come_from_the_user() -> None:
    with pytest.raises(ConversationError, match="must have role 'user'"):
        Conversation().append_assistant([{"type": "text", "text": "Hello!"}])


def test_two_user_messages_in_a_row_are_rejected() -> None:
    # The real API would silently MERGE these. We refuse, so bugs stay visible.
    conversation = Conversation()
    conversation.append_user("first")
    with pytest.raises(ConversationError, match="must have role 'assistant'"):
        conversation.append_user("second")


@pytest.mark.parametrize("empty", ["", "   ", []])
def test_empty_user_content_is_rejected(empty: str | list) -> None:
    with pytest.raises(ConversationError):
        Conversation().append_user(empty)


def test_empty_assistant_content_is_rejected() -> None:
    conversation = Conversation()
    conversation.append_user("Hi")
    with pytest.raises(ConversationError):
        conversation.append_assistant([])


def test_changing_the_returned_list_cannot_change_the_history() -> None:
    conversation = Conversation()
    conversation.append_user("Hi")

    snapshot = conversation.messages
    snapshot.pop()  # mutate the copy...
    snapshot.append({"role": "user", "content": "sneaky"})

    assert conversation.messages == [{"role": "user", "content": "Hi"}]  # ...history untouched


def test_changing_the_original_content_after_appending_cannot_change_the_history() -> None:
    conversation = Conversation()
    conversation.append_user("Hi")
    blocks = [{"type": "text", "text": "original"}]
    conversation.append_assistant(blocks)

    blocks[0]["text"] = "edited later"  # the caller changes its own dict

    assert conversation.messages[1]["content"][0]["text"] == "original"


def test_to_json_is_exactly_the_message_list() -> None:
    conversation = Conversation()
    conversation.append_user("Hi")
    assert json.loads(conversation.to_json()) == conversation.messages
