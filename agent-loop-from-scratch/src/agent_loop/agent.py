"""
agent.py: the "brain" that connects the conversation, the request builder and
the API client.

PHASE 2: A LOOP OF LENGTH ONE
    Right now one user message means exactly one API call:

        user text ──► [history + user text] ──► POST /v1/messages ──► answer

    In Phase 4 this same `run_turn` method grows a `while` loop: the model
    will be able to ask for tools, we'll run them and call the API again,
    possibly many times per user message. Compare this file before and after
    Phase 4 to see exactly what "an agent loop" adds to a chatbot.

SEPARATING LOGIC FROM THE USER INTERFACE
    This module never prints anything and never reads the keyboard. It takes
    text in and returns data (`TurnResult`). The terminal UI lives in cli.py.
    Benefits:
      * we can test the logic without a terminal (tests/test_agent.py);
      * later you could put a FastAPI or React front-end on this same class.

COMMIT ONLY ON SUCCESS
    The new user message is NOT added to the history before the API call.
    We send `history + [new message]`, and only when a good answer comes back
    do we append BOTH the user message and the answer. If the call fails (or
    you press Ctrl-C), the history is exactly as it was: still valid, and
    nothing needs deleting. Databases call this a "transaction": all or nothing.
"""

from dataclasses import dataclass
from typing import Any

from agent_loop.api_client import MessagesClient
from agent_loop.config import Settings
from agent_loop.conversation import Conversation
from agent_loop.payload import build_request

# The default system prompt. It's sent with EVERY request (the API has no
# memory, remember?), so it costs input tokens on every turn. Keep it short.
#
# Keep it CONSTANT too: no current time, no random IDs. Phase 8's prompt
# caching only works while the start of the request stays identical.
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant inside a command-line chat app. "
    "Answer clearly and concisely. Use Markdown only when it helps (lists, code)."
)


@dataclass(frozen=True)
class TurnResult:
    """Everything that happened in one user → assistant exchange."""

    user_text: str
    # The assistant's content blocks, exactly as the API returned them.
    content: list[dict[str, Any]]
    # Why the model stopped ("end_turn", "max_tokens", ...).
    stop_reason: str | None
    # The response's `usage` object (token counts).
    usage: dict[str, Any]
    # The exact model version that answered, e.g. "claude-haiku-4-5-20251001".
    model: str
    duration_seconds: float
    request_id: str | None
    # How many messages the request contained (history + the new one).
    messages_sent: int
    # Whether this exchange was added to the conversation history.
    committed: bool

    @property
    def text(self) -> str:
        """All text blocks joined: the visible answer."""
        return "\n".join(b.get("text", "") for b in self.content if b.get("type") == "text")

    @property
    def input_tokens(self) -> int:
        return int(self.usage.get("input_tokens", 0))

    @property
    def output_tokens(self) -> int:
        return int(self.usage.get("output_tokens", 0))


class Agent:
    """Holds one conversation and talks to the model on its behalf."""

    def __init__(
        self,
        settings: Settings,
        client: MessagesClient,
        system: str | None = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self.settings = settings
        self.client = client
        # The system prompt is fixed for the conversation's whole lifetime.
        # Changing it mid-conversation would change the start of every request
        # (bad for prompt caching), so there's deliberately no setter.
        self.system = system
        self.conversation = Conversation()
        # Every API call we made, including ones that weren't committed.
        # They all cost money, so they all count for the stats.
        self.turns: list[TurnResult] = []

    def reset(self) -> None:
        """Start a brand-new conversation.

        We don't clear or edit the old Conversation. We replace it with a new,
        empty one, and the old object is simply dropped. That's how
        "append-only" and "start over" fit together.
        """
        self.conversation = Conversation()
        self.turns = []

    def run_turn(self, user_text: str) -> TurnResult:
        """Send one user message (plus the whole history) and return the result.

        Raises:
            ClaudeHTTPError / ClaudeConnectionError: the call failed. The
                history is unchanged, so the caller can just let the user retry.
            KeyboardInterrupt: Ctrl-C while waiting. The history is unchanged too.
        """
        # 1) The request contains the FULL history + the new user message.
        #    `conversation.messages` returns a copy, so adding to it here
        #    doesn't touch the real history.
        user_message = {"role": "user", "content": user_text}
        request_messages = self.conversation.messages + [user_message]
        payload = build_request(self.settings, request_messages, system=self.system)

        # 2) Send it. If this raises, we haven't modified anything yet.
        response = self.client.create_message(payload)
        body = response.body
        content: list[dict[str, Any]] = body.get("content", [])

        # 3) Commit both messages only if the answer has content. An empty
        #    answer (possible with stop_reason "refusal") would leave the
        #    history in a strange state, so we keep that exchange out.
        committed = bool(content)
        if committed:
            self.conversation.append_user(user_text)
            self.conversation.append_assistant(content)

        result = TurnResult(
            user_text=user_text,
            content=content,
            stop_reason=body.get("stop_reason"),
            usage=body.get("usage", {}),
            model=body.get("model", self.settings.model),
            duration_seconds=response.duration_seconds,
            request_id=response.request_id,
            messages_sent=len(request_messages),
            committed=committed,
        )
        self.turns.append(result)
        return result
