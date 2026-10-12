"""
payload.py: builds the JSON body for POST /v1/messages.

In Phase 1, hello.py built its own body. Now two programs need one (hello.py
and the chat), so the logic lives here and both share it. This is the
"single source of truth" principle: when the request format changes (Phase 3
adds `tools`), we change ONE function.

This module only builds a Python dict. It doesn't send anything. Keeping
"what to send" (here) separate from "how to send it" (api_client.py) makes
both halves simple and easy to test.
"""

from typing import Any

from agent_loop.config import Settings


def build_request(
    settings: Settings,
    messages: list[dict[str, Any]],
    *,
    system: str | None = None,
) -> dict[str, Any]:
    """Build the request body for the Messages API.

    The three REQUIRED fields:
      * model:      which model answers.
      * max_tokens: the most tokens the model may WRITE (a safety ceiling).
      * messages:   the WHOLE conversation so far. The API has no memory, so
                    this list is the only context the model gets.

    OPTIONAL fields we use:
      * system:     instructions that frame the whole conversation (persona,
                    rules, format). It's a separate top-level field, not a
                    message in the list.
      * thinking:   extended thinking with a token budget (when enabled).

    The `*` in the signature means `system` must be passed by name
    (`system="..."`), which keeps call sites readable.
    """
    payload: dict[str, Any] = {
        "model": settings.model,
        "max_tokens": settings.max_tokens,
        "messages": messages,
    }
    if system:
        payload["system"] = system
    if settings.thinking_enabled:
        # "enabled" + budget_tokens is how Haiku 4.5 does extended thinking.
        # (Newer, bigger models use {"type": "adaptive"} instead. Request
        # fields can differ between model families, so read each model's docs.)
        payload["thinking"] = {"type": "enabled", "budget_tokens": settings.thinking_budget}
    return payload
