"""
api_client.py: talks to the Claude Messages API with plain HTTP.

THIS IS WHAT AN LLM "SDK" REALLY IS
    Strip away the nice classes and every call to Claude is a single HTTP
    request:

        POST https://api.anthropic.com/v1/messages
        x-api-key: <your key>
        anthropic-version: 2023-06-01
        content-type: application/json

        { "model": "...", "max_tokens": ..., "messages": [...] }

    The server answers with JSON: either a *message* (success) or an
    *error* object. This module does exactly that and nothing more:
      * builds the headers,
      * sends the JSON body,
      * turns an error response into a Python exception,
      * returns the parsed JSON plus the useful HTTP details.

    It deliberately does NOT retry yet. We add retries in Phase 6, so you
    first see failures exactly as they happen.

THE API IS STATELESS
    The server remembers nothing between requests. If you want the model to
    "remember" earlier messages, you must send them again every time. That
    single fact explains most of how agents work (and why they get
    expensive), and we explore it in Phase 2.
"""

import time
from dataclasses import dataclass
from typing import Any

import httpx2

from agent_loop.config import ANTHROPIC_VERSION, Settings

# The path of the Messages endpoint, relative to the base URL.
MESSAGES_PATH = "/v1/messages"


# -----------------------------------------------------------------------------
# Errors
#
# We wrap every failure in OUR OWN exception types. Code that calls this module
# (the CLI, later the agent loop) then only needs to know about these classes,
# not about httpx2 internals. That boundary is what lets us swap the HTTP
# library later without touching the rest of the app.
#
#   ClaudeAPIError                 ← base class: "something went wrong talking to Claude"
#   ├── ClaudeHTTPError            ← the server ANSWERED, but with an error status (4xx/5xx)
#   └── ClaudeConnectionError      ← we got NO answer (network down, DNS, timeout...)
#
# Phase 6 uses the difference between the two: some errors are worth retrying
# (overloaded server, timeouts) and some are not (a bad API key).
# -----------------------------------------------------------------------------

class ClaudeAPIError(Exception):
    """Base class for every error raised by this module."""


class ClaudeHTTPError(ClaudeAPIError):
    """The API responded, but with a non-2xx status code.

    The Claude API always describes errors with the same JSON shape:

        {
          "type": "error",
          "error": {"type": "authentication_error", "message": "invalid x-api-key"},
          "request_id": "req_011C..."
        }

    We copy those fields onto the exception so callers can react to them
    (e.g. `if err.error_type == "rate_limit_error": ...`).
    """

    def __init__(
        self,
        status_code: int,
        error_type: str,
        message: str,
        request_id: str | None,
        body: Any,
    ) -> None:
        self.status_code = status_code  # e.g. 401
        self.error_type = error_type    # e.g. "authentication_error"
        self.message = message          # human-readable explanation from the API
        self.request_id = request_id    # include this if you ever contact support
        self.body = body                # the full parsed error body, for debugging
        super().__init__(f"{status_code} {error_type}: {message}")


class ClaudeConnectionError(ClaudeAPIError):
    """No HTTP response at all: network failure, DNS failure, or timeout."""


# -----------------------------------------------------------------------------
# The successful result
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class APIResponse:
    """Everything useful about one successful API call.

    `body` is the model's message (the parsed JSON). We also keep a few
    HTTP-level details that the body doesn't include:
      * `headers`, e.g. `request-id` and the rate-limit headers,
      * `duration_seconds`, how long the call took (LLM latency matters!).
    """

    status_code: int
    headers: dict[str, str]
    body: dict[str, Any]
    duration_seconds: float

    @property
    def request_id(self) -> str | None:
        """The unique ID Anthropic assigns to every request (header `request-id`)."""
        return self.headers.get("request-id")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def build_headers(settings: Settings) -> dict[str, str]:
    """The headers every Messages API request needs.

    Always sent:
    * x-api-key: who you are (and who gets billed).
    * anthropic-version: which version of the API's JSON format you expect.
    * content-type: "the body I'm sending is JSON".

    Sometimes sent:
    * anthropic-workspace-id: which workspace the request belongs to. Only
      needed when your API key is not scoped to a single workspace.
    """
    headers = {
        "x-api-key": settings.api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    if settings.workspace_id:
        headers["anthropic-workspace-id"] = settings.workspace_id
    return headers


def mask_secret(secret: str, visible: int = 4) -> str:
    """Hide a secret for display, keeping only the last few characters.

    "sk-ant-api03-abcdef...wxyz" → "sk-ant-…wxyz"
    Showing the end of a key is a common way to tell keys apart without
    leaking them.
    """
    if len(secret) <= visible:
        return "…"
    prefix = "sk-ant-" if secret.startswith("sk-ant-") else ""
    return f"{prefix}…{secret[-visible:]}"


def _parse_error(response: httpx2.Response) -> ClaudeHTTPError:
    """Turn a non-2xx HTTP response into a ClaudeHTTPError.

    Usually the body is the JSON error object described above. But sometimes
    a proxy or load balancer in between answers with HTML or plain text, so
    we must not assume the body is valid JSON.
    """
    try:
        body: Any = response.json()
    except ValueError:  # the body was not JSON
        body = response.text

    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        error = body["error"]
        return ClaudeHTTPError(
            status_code=response.status_code,
            error_type=str(error.get("type", "unknown_error")),
            message=str(error.get("message", "")),
            request_id=body.get("request_id") or response.headers.get("request-id"),
            body=body,
        )

    # Not the standard error shape. Keep the first 500 characters as the message.
    return ClaudeHTTPError(
        status_code=response.status_code,
        error_type="http_error",
        message=str(body)[:500],
        request_id=response.headers.get("request-id"),
        body=body,
    )


# -----------------------------------------------------------------------------
# The client
# -----------------------------------------------------------------------------

class MessagesClient:
    """A tiny client for the `POST /v1/messages` endpoint.

    Use it as a context manager so the network connection is always closed:

        with MessagesClient(settings) as client:
            response = client.create_message({"model": ..., "max_tokens": ..., "messages": [...]})
            print(response.body["content"])

    Why keep ONE `httpx2.Client` instead of calling `httpx2.post(...)` each
    time? A Client reuses the TCP + TLS connection between requests
    ("connection pooling"). An agent makes many calls in a row, and skipping a
    new TLS handshake each time saves real latency.
    """

    def __init__(
        self,
        settings: Settings,
        transport: httpx2.BaseTransport | None = None,
    ) -> None:
        """Create the underlying HTTP client.

        Args:
            settings: the validated app settings (key, base URL, timeout).
            transport: normally None (use the real network). Tests pass an
                `httpx2.MockTransport`, a fake network that returns
                pre-written responses. That's how our tests run without
                spending money or needing internet access.
        """
        self._settings = settings
        self._http = httpx2.Client(
            # Every request path (like "/v1/messages") is joined onto this.
            base_url=settings.base_url,
            # Sent with EVERY request made through this client.
            headers=build_headers(settings),
            # Timeout(overall, connect=...):
            #   * connect: how long to wait to OPEN the connection. If the host
            #     is unreachable we want to know fast, so 10 seconds.
            #   * the other phases (read/write/pool) use the overall value. The
            #     "read" timeout matters most: it's the longest we wait for the
            #     next piece of data from the server. In a NON-streaming call
            #     the server sends nothing until the WHOLE answer is generated,
            #     so this effectively limits how long generation may take.
            #     (Streaming, in Phase 9, sends data continuously and avoids that.)
            timeout=httpx2.Timeout(settings.timeout_seconds, connect=10.0),
            transport=transport,
        )

    # --- context-manager support: `with MessagesClient(...) as client:` -------
    def __enter__(self) -> "MessagesClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the pooled network connections."""
        self._http.close()

    # --- the one real method -------------------------------------------------
    def create_message(self, payload: dict[str, Any]) -> APIResponse:
        """Send one request to the Messages API and return the parsed response.

        Args:
            payload: the JSON body. At minimum: `model`, `max_tokens`, `messages`.

        Returns:
            An `APIResponse` whose `.body` is the message JSON, e.g.
            {"id": "msg_...", "type": "message", "role": "assistant",
             "content": [...], "stop_reason": "end_turn", "usage": {...}}

        Raises:
            ClaudeHTTPError: the server answered with an error (4xx/5xx).
            ClaudeConnectionError: no answer (network problem or timeout).
        """
        # perf_counter() is a high-resolution clock meant for measuring
        # durations (unlike time.time(), it never jumps if the system clock changes).
        started = time.perf_counter()
        try:
            # `json=payload` makes httpx2 serialize the dict to JSON for us.
            response = self._http.post(MESSAGES_PATH, json=payload)
        except httpx2.TimeoutException as exc:
            # TimeoutException is a subclass of TransportError, so it must be
            # caught FIRST to get its own, clearer message.
            raise ClaudeConnectionError(
                f"Timed out after {self._settings.timeout_seconds:.0f}s waiting for "
                f"{self._settings.base_url} ({type(exc).__name__})."
            ) from exc
        except httpx2.TransportError as exc:
            # TransportError covers everything that stops us from getting a
            # response: DNS failure, refused connection, dropped connection...
            raise ClaudeConnectionError(
                f"Could not reach {self._settings.base_url}: {exc!r}"
            ) from exc
        duration = time.perf_counter() - started

        # `is_success` means status 200-299. Anything else is an API error.
        if not response.is_success:
            raise _parse_error(response)

        return APIResponse(
            status_code=response.status_code,
            # HTTP header names are case-insensitive, so we lowercase them
            # to make lookups simple and predictable.
            headers={name.lower(): value for name, value in response.headers.items()},
            body=response.json(),
            duration_seconds=duration,
        )
