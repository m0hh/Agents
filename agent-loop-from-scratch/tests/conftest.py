"""
conftest.py: shared test helpers ("fixtures") that pytest loads automatically.

Any test function that has a parameter named like a fixture below (e.g.
`fake_api`) automatically receives a fresh instance of it. No import needed.

THE FAKE MESSAGES API
    `FakeMessagesAPI` stands in for Anthropic's server:
      * you queue up the responses it should give, in order;
      * it records every request body it receives, so tests can check
        EXACTLY what our code sent (e.g. "did turn 2 include the history?").
    It plugs into our MessagesClient through httpx2.MockTransport, the same
    fake-network trick as in Phase 1. Phase 4 uses it to script whole
    tool-calling conversations.
"""

import json
from collections import deque
from typing import Any

import httpx2
import pytest

from agent_loop.api_client import MessagesClient
from agent_loop.config import Settings

FAKE_KEY = "sk-ant-test-0000000000abcd"


class FakeMessagesAPI:
    """A scripted stand-in for POST /v1/messages."""

    def __init__(self) -> None:
        # deque = a list that's fast to pop from the FRONT (first queued, first served).
        self._responses: deque[httpx2.Response] = deque()
        # Every request body received, as parsed JSON, in order.
        self.requests: list[dict[str, Any]] = []

    # --- queueing responses -----------------------------------------------------

    def reply(
        self,
        content: list[dict[str, Any]],
        *,
        stop_reason: str = "end_turn",
        input_tokens: int = 10,
        output_tokens: int = 5,
    ) -> None:
        """Queue a successful response with these content blocks."""
        body = {
            "id": f"msg_fake_{len(self._responses) + len(self.requests)}",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5-20251001",
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        }
        self._responses.append(httpx2.Response(200, json=body, headers={"request-id": "req_fake"}))

    def reply_text(self, text: str, **kwargs: Any) -> None:
        """Queue a successful response containing a single text block."""
        self.reply([{"type": "text", "text": text}], **kwargs)

    def reply_error(self, status: int, error_type: str, message: str) -> None:
        """Queue an API error response in the real error-body shape."""
        body = {"type": "error", "error": {"type": error_type, "message": message}, "request_id": "req_err"}
        self._responses.append(httpx2.Response(status, json=body))

    # --- the fake network -------------------------------------------------------

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        """Called by MockTransport for every request our code sends."""
        self.requests.append(json.loads(request.content))
        if not self._responses:
            raise AssertionError("The code made more API calls than the test queued responses for.")
        return self._responses.popleft()

    def client(self, settings: Settings) -> MessagesClient:
        """A MessagesClient whose network is this fake."""
        return MessagesClient(settings, transport=httpx2.MockTransport(self.handle))


@pytest.fixture
def settings() -> Settings:
    """Default settings with a fake key (never sent anywhere real)."""
    return Settings(api_key=FAKE_KEY)


@pytest.fixture
def fake_api() -> FakeMessagesAPI:
    """A fresh fake API for each test."""
    return FakeMessagesAPI()
