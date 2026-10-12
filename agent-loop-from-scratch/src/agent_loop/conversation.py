"""
conversation.py: the conversation history, which is the agent's only memory.

THE BIG IDEA OF PHASE 2
    The API is stateless: it forgets everything after each request. A chatbot
    "remembers" only because the program keeps a list of every message and
    sends the WHOLE list again on every request:

        turn 1 sends: [user₁]
        turn 2 sends: [user₁, assistant₁, user₂]
        turn 3 sends: [user₁, assistant₁, user₂, assistant₂, user₃]

    That list is what this class holds. Framework "memory" classes and
    LangGraph checkpointers are, at their core, this list plus a place to
    store it.

    Side effect: each request is bigger than the last, so input_tokens (and
    cost) grow every turn. Watch the footer in the chat to see it happen.

TWO RULES THIS CLASS ENFORCES
    1. Roles alternate: user, assistant, user, assistant... starting with user.
       Fun fact, tested live: the API is more forgiving than this. It silently
       MERGES two consecutive user messages into one, and accepts an empty
       assistant message. We're stricter on purpose: a silent merge hides
       bugs in our code, and in Phase 4 the strict order becomes essential
       (each tool_result must come right after its tool_use).
       See experiments/message_rules.py to probe the API's rules yourself.
    2. Append-only: there is no method to edit or delete a message.
       * Prompt caching (Phase 8) only works while the start of the history
         stays byte-for-byte identical.
       * Thinking blocks carry signatures that must come back unchanged.
       * An append-only log is easy to reason about and to debug.
       Starting over (/reset) means creating a NEW Conversation, not editing
       this one.
"""

import copy
import json
from typing import Any, Literal

# A type alias: a "Role" is one of these two exact strings.
Role = Literal["user", "assistant"]

# Message content is either a plain string (shortcut for one text block) or a
# list of typed blocks such as {"type": "text", "text": "..."}.
Content = str | list[dict[str, Any]]


class ConversationError(Exception):
    """Raised when an append would break the conversation's rules."""


class Conversation:
    """An append-only, validated list of messages in the Messages API format."""

    def __init__(self) -> None:
        # The leading underscore means "private: use the methods below, don't
        # touch this list directly". Python doesn't enforce it, but every
        # Python developer respects it.
        self._messages: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ reading

    @property
    def messages(self) -> list[dict[str, Any]]:
        """A COPY of the history, ready to put in a request body.

        Why a deep copy? If we returned our internal list, a caller could do
        `conversation.messages.pop()` or edit a block in place and silently
        break the append-only rule. Returning a copy ("defensive copying")
        makes that impossible. The cost is small: a chat history is just a
        few kilobytes of dicts and strings.
        """
        return copy.deepcopy(self._messages)

    def __len__(self) -> int:
        """`len(conversation)` gives the number of messages."""
        return len(self._messages)

    @property
    def next_role(self) -> Role:
        """Whose turn it is to add a message, based on alternation."""
        if not self._messages or self._messages[-1]["role"] == "assistant":
            return "user"
        return "assistant"

    def to_json(self, indent: int = 2) -> str:
        """The history as pretty JSON, exactly what the API receives."""
        return json.dumps(self._messages, indent=indent, ensure_ascii=False)

    # ------------------------------------------------------------------ appending

    def append_user(self, content: Content) -> None:
        """Add a user message (a string, or a list of content blocks)."""
        self._append("user", content)

    def append_assistant(self, content: list[dict[str, Any]]) -> None:
        """Add an assistant message.

        `content` must be the response's `content` list EXACTLY as the API
        returned it: every block, in order, unchanged. Don't extract only the
        text. Thinking blocks must come back with their signatures intact, and
        from Phase 4 on, tool_use blocks are what tool results refer to.
        """
        self._append("assistant", content)

    def _append(self, role: Role, content: Content) -> None:
        """Validate, then append one message. The only place the list grows."""
        # Rule: no empty messages. (The API rejects empty user content with a 400.)
        if isinstance(content, str):
            if not content.strip():
                raise ConversationError(f"A {role} message can't be empty.")
        elif not content:
            raise ConversationError(f"A {role} message needs at least one content block.")

        # Rule: strict alternation, starting with "user".
        if role != self.next_role:
            raise ConversationError(
                f"The next message must have role {self.next_role!r}, not {role!r}. "
                "Roles must alternate: user, assistant, user, ..."
            )

        # Store a deep copy, so changes the caller makes to `content` later
        # can't reach into our history.
        self._messages.append({"role": role, "content": copy.deepcopy(content)})
