"""
Tests for api_client.py, using a FAKE network.

THE TRICK: httpx2.MockTransport
    A "transport" is the part of an HTTP client that actually sends bytes over
    the network. `httpx2.MockTransport(handler)` replaces it with a Python
    function: every request is handed to `handler`, and whatever Response the
    handler returns is what the client "receives".

    That lets us:
      * inspect the exact request our code built (URL, headers, JSON body),
      * pretend the server returned any response we like, errors included,
      * run the tests offline, instantly, and for free.

    Later phases use the same trick to test the whole agent loop.
"""

import json

import httpx2
import pytest

from agent_loop.api_client import (
    ClaudeConnectionError,
    ClaudeHTTPError,
    MessagesClient,
    mask_secret,
)
from agent_loop.config import Settings

FAKE_KEY = "sk-ant-test-0000000000abcd"

# A realistic successful response body, shaped like the real API's.
FAKE_MESSAGE = {
    "id": "msg_test_123",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-4-5",
    "content": [{"type": "text", "text": "Hello from a fake Claude!"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 12, "output_tokens": 7},
}


def make_client(handler) -> MessagesClient:
    """Build a MessagesClient whose network is replaced by `handler`."""
    settings = Settings(api_key=FAKE_KEY)
    return MessagesClient(settings, transport=httpx2.MockTransport(handler))


def test_sends_the_right_request_and_returns_the_parsed_body() -> None:
    captured: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured.append(request)  # remember what our code sent
        return httpx2.Response(200, json=FAKE_MESSAGE, headers={"request-id": "req_abc"})

    payload = {
        "model": "claude-haiku-4-5",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Hi"}],
    }
    with make_client(handler) as client:
        response = client.create_message(payload)

    # --- the request we built --------------------------------------------------
    request = captured[0]
    assert request.method == "POST"
    assert str(request.url) == "https://api.anthropic.com/v1/messages"
    assert request.headers["x-api-key"] == FAKE_KEY
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert request.headers["content-type"] == "application/json"
    assert json.loads(request.content) == payload  # body is exactly our payload

    # --- the result we returned ------------------------------------------------
    assert response.status_code == 200
    assert response.body == FAKE_MESSAGE
    assert response.request_id == "req_abc"
    assert response.duration_seconds >= 0


def test_api_error_becomes_a_claude_http_error_with_all_details() -> None:
    error_body = {
        "type": "error",
        "error": {"type": "authentication_error", "message": "invalid x-api-key"},
        "request_id": "req_err_1",
    }

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(401, json=error_body)

    with make_client(handler) as client, pytest.raises(ClaudeHTTPError) as caught:
        client.create_message({"model": "x", "max_tokens": 1, "messages": []})

    error = caught.value
    assert error.status_code == 401
    assert error.error_type == "authentication_error"
    assert error.message == "invalid x-api-key"
    assert error.request_id == "req_err_1"


def test_non_json_error_body_is_handled_gracefully() -> None:
    # Something between us and the API (a proxy, a load balancer) may answer with HTML.
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(502, text="<html>Bad Gateway</html>")

    with make_client(handler) as client, pytest.raises(ClaudeHTTPError) as caught:
        client.create_message({"model": "x", "max_tokens": 1, "messages": []})

    assert caught.value.status_code == 502
    assert caught.value.error_type == "http_error"
    assert "Bad Gateway" in caught.value.message


def test_network_failure_becomes_a_claude_connection_error() -> None:
    # A handler that RAISES simulates a network failure: no response at all.
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("connection refused", request=request)

    with make_client(handler) as client, pytest.raises(ClaudeConnectionError):
        client.create_message({"model": "x", "max_tokens": 1, "messages": []})


def test_timeout_becomes_a_claude_connection_error_that_mentions_the_timeout() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("too slow", request=request)

    with make_client(handler) as client, pytest.raises(ClaudeConnectionError, match="Timed out"):
        client.create_message({"model": "x", "max_tokens": 1, "messages": []})


def test_workspace_header_is_sent_only_when_configured() -> None:
    seen_headers: list[httpx2.Headers] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen_headers.append(request.headers)
        return httpx2.Response(200, json=FAKE_MESSAGE)

    payload = {"model": "x", "max_tokens": 1, "messages": []}
    # 1) no workspace configured → no header
    with make_client(handler) as client:
        client.create_message(payload)
    # 2) workspace configured → header present
    settings = Settings(api_key=FAKE_KEY, workspace_id="wrkspc_test")
    with MessagesClient(settings, transport=httpx2.MockTransport(handler)) as client:
        client.create_message(payload)

    assert "anthropic-workspace-id" not in seen_headers[0]
    assert seen_headers[1]["anthropic-workspace-id"] == "wrkspc_test"


def test_mask_secret_keeps_only_the_last_characters() -> None:
    masked = mask_secret(FAKE_KEY)
    assert masked == "sk-ant-…abcd"
    assert FAKE_KEY not in masked
