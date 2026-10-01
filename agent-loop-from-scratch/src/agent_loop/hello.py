"""
hello.py (Phase 1): send ONE message with raw HTTP and take the response apart.

Run it:
    uv run hello
    uv run hello "Explain recursion in one sentence."
    uv run hello "Is 1001 a prime number?" --thinking 2048
    uv run hello "Hi" --system "You are a pirate. Stay in character."
    uv run hello "Write a long poem about the sea" --max-tokens 50

The output has three parts:
    ① THE REQUEST:  the exact URL, headers and JSON body we send
    ② THE RESPONSE: the raw HTTP status, headers and JSON body we get back
    ③ READING IT:   every field of the response explained

No agent, no tools, no memory yet. Just one request and one response,
so you can see the "atom" everything else is built from.
"""

import argparse
import dataclasses
import json
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agent_loop.api_client import (
    MESSAGES_PATH,
    APIResponse,
    ClaudeConnectionError,
    ClaudeHTTPError,
    MessagesClient,
    build_headers,
    mask_secret,
)
from agent_loop.config import ConfigError, Settings, load_settings

# One shared Console object prints everything (colors, tables, JSON).
console = Console()

DEFAULT_PROMPT = "Hello! In two short sentences: who are you, and what is an 'agent loop'?"

# -----------------------------------------------------------------------------
# Reference data used to explain the response
# -----------------------------------------------------------------------------

# Why did the model stop writing? The API always tells you in `stop_reason`.
# In Phase 5 the agent loop makes decisions based on this value.
STOP_REASON_MEANINGS: dict[str, str] = {
    "end_turn": "The model finished its answer naturally.",
    "max_tokens": "The answer was CUT OFF because it reached max_tokens.",
    "stop_sequence": "The model wrote one of your custom stop_sequences.",
    "tool_use": "The model wants you to run a tool (Phase 3+).",
    "pause_turn": "A long server-side tool run was paused; send the request again to continue.",
    "refusal": "A safety system declined to continue this response.",
}

# Price per 1 MILLION tokens in US dollars: (input, output).
# Output tokens cost 5x more than input tokens because generating text is far
# more work for the model than reading it.
# (Phase 7 turns this into a proper cost-tracking module.)
PRICES_PER_MILLION: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.00, 5.00),
}


# -----------------------------------------------------------------------------
# Step 1: build the request body
# -----------------------------------------------------------------------------

def build_payload(settings: Settings, prompt: str, system: str | None) -> dict[str, Any]:
    """Build the JSON body for POST /v1/messages.

    The three REQUIRED fields:
      * model:      which model answers.
      * max_tokens: the most tokens the model may WRITE (a safety ceiling).
      * messages:   the conversation so far, a list of {"role", "content"}.
                    Here it's just one user message. The API has no memory,
                    so this list is the ONLY context the model gets.

    OPTIONAL fields we use:
      * system:     instructions that frame the whole conversation (persona,
                    rules, format). It's a separate top-level field, not a message.
      * thinking:   turn on extended thinking with a token budget.
    """
    payload: dict[str, Any] = {
        "model": settings.model,
        "max_tokens": settings.max_tokens,
        "messages": [
            # `content` can be a plain string (shortcut) OR a list of typed
            # blocks like [{"type": "text", "text": "..."}]. The string form
            # is converted by the API into a single text block.
            {"role": "user", "content": prompt},
        ],
    }
    if system:
        payload["system"] = system
    if settings.thinking_enabled:
        # "enabled" + budget_tokens is how Haiku 4.5 does extended thinking.
        # (Newer, bigger models use {"type": "adaptive"} instead. Request
        # fields can differ between model families, so read each model's docs.)
        payload["thinking"] = {"type": "enabled", "budget_tokens": settings.thinking_budget}
    return payload


# -----------------------------------------------------------------------------
# Step 2: show what we send
# -----------------------------------------------------------------------------

def show_request(settings: Settings, payload: dict[str, Any]) -> None:
    """Print the exact HTTP request, with the API key masked."""
    console.rule("[bold cyan]① THE REQUEST[/] (what we send)")
    console.print(f"[bold]POST[/] {settings.base_url}{MESSAGES_PATH}\n")

    headers = build_headers(settings)
    headers["x-api-key"] = mask_secret(headers["x-api-key"])  # never print a real key!
    table = Table(title="Headers", show_header=True, header_style="bold")
    table.add_column("Header")
    table.add_column("Value")
    for name, value in headers.items():
        table.add_row(name, value)
    console.print(table)

    console.print("[bold]Body (JSON):[/]")
    console.print_json(data=payload)


# -----------------------------------------------------------------------------
# Step 3: show what we got back
# -----------------------------------------------------------------------------

def show_raw_response(response: APIResponse) -> None:
    """Print the HTTP-level details and the raw JSON body."""
    console.rule("[bold green]② THE RESPONSE[/] (what comes back)")
    console.print(
        f"[bold]HTTP {response.status_code}[/] in {response.duration_seconds:.2f}s   "
        f"request-id: {response.request_id or '-'}"
    )

    # Rate-limit headers tell you how much quota you have left. We'll use
    # them in Phase 6 (retries). Only headers containing "ratelimit" are shown.
    rate_limit_headers = {k: v for k, v in response.headers.items() if "ratelimit" in k}
    if rate_limit_headers:
        table = Table(title="Rate-limit headers", header_style="bold")
        table.add_column("Header")
        table.add_column("Value")
        for name, value in sorted(rate_limit_headers.items()):
            table.add_row(name, value)
        console.print(table)

    console.print("[bold]Body (JSON):[/]")
    console.print_json(data=response.body)


def _preview(value: str, limit: int = 160) -> str:
    """Shorten long text to one readable line for the tables."""
    one_line = " ".join(value.split())
    return one_line if len(one_line) <= limit else one_line[: limit - 1] + "…"


def describe_block(block: dict[str, Any]) -> str:
    """Summarize one content block in a single line, based on its `type`.

    The response `content` is a LIST of typed blocks, not a string. Good
    code always branches on block["type"] and never assumes that
    content[0] is the text. For example, when thinking is on, the first
    block is the thinking, not the answer.
    """
    block_type = block.get("type")
    if block_type == "text":
        return _preview(block.get("text", ""))
    if block_type == "thinking":
        # The model's reasoning. The `signature` is a cryptographic seal that
        # proves the thinking came from the API unmodified. That's why, in
        # later phases, we must send thinking blocks back EXACTLY as received.
        signature = block.get("signature", "")
        return f"{_preview(block.get('thinking', ''))}  [signature: {len(signature)} chars]"
    if block_type == "redacted_thinking":
        return "(encrypted reasoning; must be passed back unchanged)"
    if block_type == "tool_use":
        return f"{block.get('name')}({json.dumps(block.get('input', {}))})"
    # Unknown/new block types: show the raw JSON. Never crash on something new!
    return _preview(json.dumps(block))


def estimate_cost(model: str, usage: dict[str, Any]) -> float | None:
    """Rough dollar cost of this call, or None if we don't know the model's price.

    The response may report the model's full snapshot name (for example
    "claude-haiku-4-5-<date>"), so we match by prefix.
    """
    for known_model, (input_price, output_price) in PRICES_PER_MILLION.items():
        if model.startswith(known_model):
            return (
                usage.get("input_tokens", 0) * input_price
                + usage.get("output_tokens", 0) * output_price
            ) / 1_000_000
    return None


def explain_response(settings: Settings, body: dict[str, Any]) -> None:
    """Walk through the important fields of the message JSON."""
    console.rule("[bold magenta]③ READING THE RESPONSE[/]")

    # --- identity fields ------------------------------------------------------
    console.print(f"[bold]id[/]          {body.get('id')}   (unique ID of this message)")
    console.print(f"[bold]type / role[/] {body.get('type')} / {body.get('role')}   (always an assistant message)")
    console.print(
        f"[bold]model[/]       {body.get('model')}   "
        f"(you asked for {settings.model!r}; this is the model that actually answered)"
    )

    # --- content blocks -------------------------------------------------------
    blocks: list[dict[str, Any]] = body.get("content", [])
    table = Table(title=f"content: a LIST of {len(blocks)} typed block(s)", header_style="bold")
    table.add_column("#", justify="right")
    table.add_column("type")
    table.add_column("what's inside")
    for index, block in enumerate(blocks):
        # Text(...) prints the string literally. Plain strings would be
        # parsed as rich markup, and model text containing [brackets] would break.
        table.add_row(str(index), str(block.get("type")), Text(describe_block(block)))
    console.print(table)

    # --- stop_reason ----------------------------------------------------------
    stop_reason = body.get("stop_reason")
    meaning = STOP_REASON_MEANINGS.get(stop_reason or "", "Unknown value: handle it defensively.")
    style = "green" if stop_reason == "end_turn" else "yellow"
    console.print(f"[bold]stop_reason[/] [{style}]{stop_reason}[/]: {meaning}")

    # --- usage (tokens = money) ----------------------------------------------
    usage: dict[str, Any] = body.get("usage", {})
    usage_table = Table(title="usage: what this call consumed", header_style="bold")
    usage_table.add_column("field")
    usage_table.add_column("value", justify="right")
    # Show every simple (number/string) usage field the API returned, whatever it is.
    for name, value in usage.items():
        if isinstance(value, (int, float, str)) or value is None:
            usage_table.add_row(name, str(value))
    console.print(usage_table)
    console.print(
        "  • [bold]input_tokens[/]: everything we SENT (system + messages, plus tool definitions later).\n"
        "  • [bold]output_tokens[/]: everything the model WROTE, [italic]including thinking[/].\n"
        "  • [bold]cache_*[/] fields: prompt caching (Phase 8). Zero for now."
    )
    cost = estimate_cost(body.get("model") or settings.model, usage)
    if cost is not None:
        console.print(f"  ≈ [bold]${cost:.6f}[/] for this call")

    # --- the answer itself ----------------------------------------------------
    # Join ALL text blocks (there can be more than one).
    answer = "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    console.print(Panel(Text(answer or "(no text blocks)"), title="The answer (all text blocks joined)"))


# -----------------------------------------------------------------------------
# Error display
# -----------------------------------------------------------------------------

# What usually causes each API error type, and what to do about it.
ERROR_HINTS: dict[str, str] = {
    "authentication_error": "Your API key is missing, wrong, or revoked. Check ANTHROPIC_API_KEY.",
    "permission_error": "Your key isn't allowed to do this (organization/workspace settings).",
    "not_found_error": "Usually a typo in the model name. Check AGENT_MODEL / --model.",
    "invalid_request_error": "The JSON body is wrong. The message says which field.",
    "rate_limit_error": "Too many requests or tokens per minute. Wait and retry (Phase 6).",
    "overloaded_error": "The API is temporarily overloaded. Retry with backoff (Phase 6).",
    "api_error": "A temporary server-side problem. Retry with backoff (Phase 6).",
    "billing_error": "A billing/payment problem on your account.",
}


def show_http_error(error: ClaudeHTTPError) -> None:
    """Print an API error the way you'd want to debug it."""
    console.rule("[bold red]② THE RESPONSE: an error[/]")
    console.print(f"[bold red]HTTP {error.status_code}[/]  type: [bold]{error.error_type}[/]")
    console.print(f"message:    {error.message}")
    console.print(f"request_id: {error.request_id or '-'}")
    if isinstance(error.body, dict):
        console.print("[bold]Raw error body:[/]")
        console.print_json(data=error.body)
    hint = ERROR_HINTS.get(error.error_type)
    # A more specific hint for one common setup problem: an API key that
    # isn't tied to a workspace needs the workspace named in a header.
    if "anthropic-workspace-id" in error.message:
        hint = (
            "Your key isn't scoped to a workspace. Add ANTHROPIC_WORKSPACE_ID=wrkspc_... "
            "to .env (find the ID in the Claude Console under Settings → Workspaces), "
            "or create an API key inside a workspace instead."
        )
    if hint:
        console.print(f"[yellow]Hint:[/] {hint}")


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------

def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Define the command-line options. `argparse` is Python's built-in CLI parser."""
    parser = argparse.ArgumentParser(
        prog="hello",
        description="Phase 1: send one message to Claude over raw HTTP and dissect the response.",
    )
    parser.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT, help="what to say to the model")
    parser.add_argument("--system", help="a system prompt (instructions/persona)")
    parser.add_argument("--model", help="override AGENT_MODEL")
    parser.add_argument("--max-tokens", type=int, help="override AGENT_MAX_TOKENS")
    parser.add_argument(
        "--thinking",
        type=int,
        metavar="BUDGET",
        help="turn on extended thinking with this token budget (>= 1024)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns an exit code: 0 = success, 1 = API failure, 2 = bad config."""
    args = parse_args(argv)

    # 1) Load and validate settings, then apply command-line overrides.
    #    dataclasses.replace() creates a NEW Settings object (ours are frozen),
    #    which re-runs the validation in __post_init__.
    try:
        settings = load_settings()
        overrides: dict[str, Any] = {}
        if args.model:
            overrides["model"] = args.model
        if args.max_tokens is not None:
            overrides["max_tokens"] = args.max_tokens
        if args.thinking is not None:
            overrides["thinking_budget"] = args.thinking
        settings = dataclasses.replace(settings, **overrides)
    except ConfigError as exc:
        console.print(f"[bold red]Config error:[/] {exc}")
        return 2

    # 2) Build and show the request.
    payload = build_payload(settings, args.prompt, args.system)
    show_request(settings, payload)

    # 3) Send it. The `with` block guarantees the connection is closed afterwards.
    with MessagesClient(settings) as client:
        try:
            with console.status("Waiting for the model…"):
                response = client.create_message(payload)
        except ClaudeHTTPError as exc:
            show_http_error(exc)
            return 1
        except ClaudeConnectionError as exc:
            console.print(f"[bold red]Connection problem:[/] {exc}")
            return 1

    # 4) Show and explain the response.
    show_raw_response(response)
    explain_response(settings, response.body)
    return 0


# Lets you also run the file directly: `uv run python -m agent_loop.hello`
if __name__ == "__main__":
    raise SystemExit(main())
