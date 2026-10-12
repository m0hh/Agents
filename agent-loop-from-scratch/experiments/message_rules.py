"""
experiments/message_rules.py: what does the API ACTUALLY accept?

Documentation and tutorials don't always match the live API, so engineers
probe it with tiny, cheap requests. This script sends a few deliberately
"wrong" conversations and prints how the API reacts.

Run it (it makes 5 tiny real API calls, costing well under $0.001 in total):
    uv run python experiments/message_rules.py

What we found when we ran it (2026-10-01, claude-haiku-4-5):
    * two user messages in a row → 200 OK: the API silently MERGES them
    * first message from the assistant → 200 OK
    * an empty assistant message in the middle → 200 OK
    * an empty user message → 400 "user messages must have non-empty content"
    * a normal alternating conversation → 200 OK (the control case)

LESSON: the API is forgiving, but your code shouldn't be. Our Conversation
class (src/agent_loop/conversation.py) enforces strict alternation and
non-empty messages anyway. A silent merge would hide bugs in our code, and in
Phase 4 the strict order is required (tool_result right after tool_use).
"""

from typing import Any

from rich.console import Console
from rich.table import Table

from agent_loop.api_client import ClaudeHTTPError, MessagesClient
from agent_loop.config import load_settings

# Each case: a name and a deliberately unusual `messages` list.
CASES: dict[str, list[dict[str, Any]]] = {
    "normal alternation (control)": [
        {"role": "user", "content": "My name is Sam."},
        {"role": "assistant", "content": "Nice to meet you, Sam!"},
        {"role": "user", "content": "What is my name? Answer in 3 words."},
    ],
    "two user messages in a row": [
        {"role": "user", "content": "My name is Sam."},
        {"role": "user", "content": "What is my name? Answer in 3 words."},
    ],
    "first message from the assistant": [
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "Say ok."},
    ],
    "empty assistant message in the middle": [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": []},
        {"role": "user", "content": "Say ok."},
    ],
    "empty user message": [
        {"role": "user", "content": ""},
    ],
}


def main() -> None:
    console = Console()
    settings = load_settings()
    table = Table(title=f"How {settings.model} handles unusual conversations", header_style="bold")
    table.add_column("case")
    table.add_column("result")
    table.add_column("details", overflow="fold")

    with MessagesClient(settings) as client:
        for name, messages in CASES.items():
            # max_tokens=50 keeps every probe tiny and cheap.
            payload = {"model": settings.model, "max_tokens": 50, "messages": messages}
            try:
                response = client.create_message(payload)
                answer = " ".join(
                    block.get("text", "") for block in response.body["content"] if block["type"] == "text"
                )
                tokens = response.body["usage"]["input_tokens"]
                table.add_row(name, "[green]200 OK[/]", f"answer: {answer!r} · input_tokens={tokens}")
            except ClaudeHTTPError as exc:
                table.add_row(name, f"[red]{exc.status_code}[/]", f"{exc.error_type}: {exc.message}")

    console.print(table)


if __name__ == "__main__":
    main()
