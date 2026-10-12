"""
cli.py (Phase 2): an interactive chat in the terminal, a REPL.

REPL = Read-Eval-Print Loop: read a line, act on it, print the result, repeat.
The Python shell itself is a REPL.

Run it:
    uv run agent
    uv run agent --thinking 2048
    uv run agent --system "You are a pirate. Stay in character."

Inside the chat, type /help for commands. The most educational ones:
    /history  the raw message list (exactly what is sent on every turn)
    /stats    tokens per turn: watch the input grow as the conversation does

This file only does input and output. All the logic is in agent.py, which
returns plain data that we display here.
"""

import argparse
import json
from collections.abc import Callable
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agent_loop.agent import DEFAULT_SYSTEM_PROMPT, Agent, TurnResult
from agent_loop.api_client import ClaudeConnectionError, ClaudeHTTPError, MessagesClient
from agent_loop.cli_args import add_settings_arguments, apply_overrides
from agent_loop.config import ConfigError, load_settings
from agent_loop.usage import estimate_cost

# Importing `readline` has a useful side effect: Python's input() gains
# arrow-key editing and ↑/↓ history of what you typed. It's not available on
# Windows, which is fine; input() just works without those extras there.
try:
    import readline  # noqa: F401
except ImportError:
    pass

console = Console()

# The chat commands and what they do. Used by /help and the welcome banner.
COMMANDS: dict[str, str] = {
    "/help": "show this list",
    "/history": "print the raw messages list (exactly what the API receives)",
    "/stats": "tokens per turn: see how the input grows",
    "/system": "show the system prompt",
    "/reset": "start a new, empty conversation",
    "/exit": "quit (also /quit, Ctrl-D, or Ctrl-C at the prompt)",
}


# -----------------------------------------------------------------------------
# Rendering helpers (pure functions: data in, rich objects out)
# -----------------------------------------------------------------------------

def render_content(out: Console, content: list[dict[str, Any]]) -> None:
    """Print the assistant's content blocks, each according to its type.

    Always branch on block["type"]: a response can contain several blocks
    of different kinds, in any order.
    """
    for block in content:
        block_type = block.get("type")
        if block_type == "thinking":
            # The model's reasoning, shown dimmed so it doesn't look like the answer.
            out.print(
                Panel(
                    Text(block.get("thinking", ""), style="dim italic"),
                    title="💭 thinking",
                    title_align="left",
                    border_style="dim",
                )
            )
        elif block_type == "redacted_thinking":
            out.print(Text("💭 (redacted thinking: encrypted, passed back unchanged)", style="dim"))
        elif block_type == "text":
            # Models often answer in Markdown; rich renders it nicely in the terminal.
            out.print(Markdown(block.get("text", "")))
        else:
            # A block type we don't know yet. Show it rather than crash.
            out.print(Text(f"[{block_type} block] {json.dumps(block)[:300]}", style="dim"))


def session_cost(turns: list[TurnResult]) -> float:
    """Total estimated dollars for every API call in this conversation."""
    return sum(estimate_cost(t.model, t.usage) or 0.0 for t in turns)


def format_footer(turns: list[TurnResult]) -> Text:
    """A one-line summary of the latest turn, shown under each answer.

    `+N` next to the input tokens is the GROWTH since the previous turn: the
    previous answer plus your new message, which are now part of the history
    that gets re-sent every time.
    """
    result = turns[-1]
    previous = turns[-2] if len(turns) > 1 else None
    growth = f" (+{result.input_tokens - previous.input_tokens:,})" if previous else ""
    turn_cost = estimate_cost(result.model, result.usage)
    cost_text = f" · ${turn_cost:.4f} turn / ${session_cost(turns):.4f} total" if turn_cost is not None else ""
    return Text(
        f"[turn {len(turns)} · sent {result.messages_sent} msgs · "
        f"in {result.input_tokens:,} tok{growth} · out {result.output_tokens:,} tok · "
        f"{result.duration_seconds:.1f}s{cost_text}]",
        style="dim",
    )


def build_stats_table(turns: list[TurnResult]) -> Table:
    """A per-turn table of token usage, with a bar chart of input size."""
    table = Table(title="Token usage per turn", header_style="bold")
    table.add_column("turn", justify="right")
    table.add_column("msgs sent", justify="right")
    table.add_column("input tok", justify="right")
    table.add_column("output tok", justify="right")
    table.add_column("cost $", justify="right")
    table.add_column("input size")

    largest = max((t.input_tokens for t in turns), default=1) or 1
    for number, turn in enumerate(turns, start=1):
        # A bar up to 30 characters long, proportional to this turn's input.
        bar = "█" * max(1, round(30 * turn.input_tokens / largest))
        cost = estimate_cost(turn.model, turn.usage)
        table.add_row(
            str(number),
            str(turn.messages_sent),
            f"{turn.input_tokens:,}",
            f"{turn.output_tokens:,}",
            f"{cost:.4f}" if cost is not None else "?",
            bar,
        )

    total_in = sum(t.input_tokens for t in turns)
    total_out = sum(t.output_tokens for t in turns)
    table.add_section()
    table.add_row("total", "", f"{total_in:,}", f"{total_out:,}", f"{session_cost(turns):.4f}", "")
    return table


# -----------------------------------------------------------------------------
# The REPL
# -----------------------------------------------------------------------------

class ChatREPL:
    """The read → act → print loop of the chat.

    `read_line` is how we get the user's next line. Normally that's the
    keyboard, but tests pass a function that returns pre-written lines. That's
    dependency injection again, the same trick as the fake network in Phase 1.
    """

    def __init__(
        self,
        agent: Agent,
        out: Console,
        read_line: Callable[[], str] | None = None,
    ) -> None:
        self.agent = agent
        self.out = out
        self.read_line = read_line or (lambda: self.out.input("[bold green]You ›[/] "))

    def run(self) -> None:
        """Loop until the user quits."""
        self.print_welcome()
        while True:
            try:
                line = self.read_line().strip()
            except (EOFError, KeyboardInterrupt):
                # EOFError = Ctrl-D (end of input); KeyboardInterrupt = Ctrl-C.
                self.out.print("\n[dim]Bye![/]")
                return

            if not line:
                continue
            if line.startswith("/"):
                keep_going = self.handle_command(line)
                if not keep_going:
                    self.out.print("[dim]Bye![/]")
                    return
                continue
            self.send(line)

    # ------------------------------------------------------------------ chatting

    def send(self, text: str) -> None:
        """Send one message to the model and show the answer."""
        try:
            # `status` shows a spinner while we wait for the HTTP response.
            with self.out.status("Waiting for Claude…"):
                result = self.agent.run_turn(text)
        except KeyboardInterrupt:
            # Thanks to "commit only on success" in agent.py, cancelling is safe.
            self.out.print("[yellow]Cancelled.[/] Nothing was added to the history.")
            return
        except ClaudeHTTPError as exc:
            self.out.print(f"[bold red]API error {exc.status_code} {exc.error_type}:[/] {exc.message}")
            self.out.print("[dim]Nothing was added to the history; you can try again.[/]")
            return
        except ClaudeConnectionError as exc:
            self.out.print(f"[bold red]Connection problem:[/] {exc}")
            self.out.print("[dim]Nothing was added to the history; you can try again.[/]")
            return

        self.out.print("[bold magenta]Claude ›[/]")
        render_content(self.out, result.content)
        self.print_stop_reason_notes(result)
        self.out.print(format_footer(self.agent.turns))

    def print_stop_reason_notes(self, result: TurnResult) -> None:
        """Explain anything unusual about how the answer ended."""
        if result.stop_reason == "max_tokens":
            self.out.print(
                f"[yellow]⚠ The answer was cut off: it reached max_tokens "
                f"({self.agent.settings.max_tokens}).[/] Say 'continue', or raise --max-tokens."
            )
        elif result.stop_reason == "refusal":
            self.out.print("[red]The model declined to continue this answer (stop_reason: refusal).[/]")
        elif result.stop_reason not in ("end_turn", None):
            self.out.print(f"[dim]stop_reason: {result.stop_reason}[/]")
        if not result.committed:
            self.out.print("[yellow]The answer was empty, so this exchange was not added to the history.[/]")

    # ------------------------------------------------------------------ commands

    def handle_command(self, line: str) -> bool:
        """Run a /command. Returns False when the chat should end."""
        command = line.split()[0].lower()
        # `match` (Python 3.10+) compares `command` against each `case` in order.
        # `case "/exit" | "/quit"` matches either string.
        match command:
            case "/exit" | "/quit":
                return False
            case "/help":
                self.print_help()
            case "/history":
                self.print_history()
            case "/stats":
                self.print_stats()
            case "/system":
                self.out.print(Panel(Text(self.agent.system or "(no system prompt)"), title="System prompt"))
            case "/reset":
                self.agent.reset()
                self.out.print("[green]Started a new conversation.[/] (The old one was dropped, not edited.)")
            case _:
                # `_` matches anything that didn't match above.
                self.out.print(f"[yellow]Unknown command {command!r}.[/] Type /help.")
        return True

    def print_welcome(self) -> None:
        settings = self.agent.settings
        thinking = f"on, budget {settings.thinking_budget:,} tokens" if settings.thinking_enabled else "off"
        lines = [
            f"model: [bold]{settings.model}[/]   thinking: {thinking}   max_tokens: {settings.max_tokens:,}",
            f"system prompt: [italic]{self.agent.system or '(none)'}[/]",
            "Type a message, or /help for commands.",
        ]
        self.out.print(Panel("\n".join(lines), title="Agent Loop · Phase 2: chat with memory", border_style="cyan"))

    def print_help(self) -> None:
        table = Table(show_header=False, box=None)
        for name, description in COMMANDS.items():
            table.add_row(f"[bold]{name}[/]", description)
        self.out.print(table)

    def print_history(self) -> None:
        conversation = self.agent.conversation
        if not len(conversation):
            self.out.print("[dim]The history is empty.[/]")
            return
        self.out.print(
            f"[bold]{len(conversation)} messages.[/] This whole list (plus your next message) "
            "is sent with EVERY request:"
        )
        self.out.print_json(conversation.to_json())

    def print_stats(self) -> None:
        turns = self.agent.turns
        if not turns:
            self.out.print("[dim]No turns yet.[/]")
            return
        self.out.print(build_stats_table(turns))
        self.out.print(
            "[dim]Each turn re-sends everything before it, so 'input tok' keeps growing. "
            "The TOTAL input you pay for is the sum of that column, and it grows much "
            "faster than the conversation itself (roughly quadratically).[/]"
        )


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="agent",
        description="Phase 2: chat with Claude in the terminal; the conversation history is resent every turn.",
    )
    parser.add_argument("--system", help="replace the default system prompt")
    add_settings_arguments(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point for `uv run agent`. Returns an exit code (0 = OK, 2 = bad config)."""
    args = parse_args(argv)
    try:
        settings = apply_overrides(load_settings(), args)
    except ConfigError as exc:
        console.print(f"[bold red]Config error:[/] {exc}")
        return 2

    with MessagesClient(settings) as client:
        agent = Agent(settings, client, system=args.system or DEFAULT_SYSTEM_PROMPT)
        ChatREPL(agent, console).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
