"""
tools_cli.py (Phase 3): explore the tools by hand, and watch the model ASK for them.

    uv run tools list                                   # what the model sees
    uv run tools call calculator '{"expression": "2**32 / 7"}'
    uv run tools call read_file '{"path": "."}'         # list the workspace
    uv run tools call read_file '{"path": "../.env"}'   # blocked: outside the sandbox
    uv run tools ask "What is 2**32 / 7?"               # the model requests a tool
    uv run tools ask "What is 2**32 / 7?" --no-tools    # compare: no tools attached

`call` runs a tool LOCALLY, exactly as the agent will in Phase 4: same
validation, same result text. No model, no API key, no cost.

`ask` sends ONE question to the model with the tools attached and shows
the `tool_use` block it answers with. It does NOT run the tool. Running it
and sending the result back is precisely what Phase 4's agent loop adds.
"""

import argparse
import json
import shlex
from typing import Any

from rich.console import Console
from rich.json import JSON
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from agent_loop.agent import DEFAULT_SYSTEM_PROMPT
from agent_loop.api_client import ClaudeConnectionError, ClaudeHTTPError, MessagesClient
from agent_loop.cli_args import add_settings_arguments, apply_overrides
from agent_loop.config import ConfigError, Settings, load_settings
from agent_loop.payload import build_request
from agent_loop.tools import ToolError, ToolRegistry, default_registry
from agent_loop.usage import estimate_cost

console = Console()


# -----------------------------------------------------------------------------
# `tools list`
# -----------------------------------------------------------------------------

def run_list(registry: ToolRegistry, out: Console) -> int:
    """Print every tool definition exactly as it is sent to the model."""
    for tool in registry:
        out.print(Panel(JSON(json.dumps(tool.definition())), title=f"[bold]{tool.name}[/]", title_align="left"))
    out.print(
        f"[dim]These {len(registry)} definitions travel with EVERY request that includes tools, "
        "so they cost input tokens every time, even when no tool is used.[/]"
    )
    return 0


# -----------------------------------------------------------------------------
# `tools call NAME 'JSON'`
# -----------------------------------------------------------------------------

def run_call(registry: ToolRegistry, out: Console, name: str, raw_input: str) -> int:
    """Run one tool locally and show the tool_result block Phase 4 would send back."""
    tool = registry.get(name)
    if tool is None:
        out.print(f"[red]No tool named {name!r}.[/] Available: {', '.join(registry.names)}")
        return 1

    # The model's tool input is JSON, so here you type JSON too.
    try:
        tool_input: Any = json.loads(raw_input)
    except json.JSONDecodeError as exc:
        out.print(f"[red]That input isn't valid JSON:[/] {exc}")
        out.print('[dim]Example: uv run tools call calculator \'{"expression": "1 + 1"}\'[/]')
        return 1

    # ToolError = an EXPECTED failure. Its message is written for the model
    # and becomes a tool_result with is_error: true.
    try:
        result, is_error = tool.execute(tool_input), False
    except ToolError as exc:
        result, is_error = str(exc), True

    style = "red" if is_error else "green"
    out.print(Panel(Text(result), title=f"{name} → {'ERROR' if is_error else 'result'}", border_style=style))

    # Show the exact block the agent will send back to the model in Phase 4.
    tool_result: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": "toolu_… (the id from the model's tool_use block)",
        "content": result,
    }
    if is_error:
        tool_result["is_error"] = True
    out.print("[bold]In Phase 4, the agent sends this back to the model:[/]")
    out.print(JSON(json.dumps(tool_result)))
    return 1 if is_error else 0


# -----------------------------------------------------------------------------
# `tools ask "question"`
# -----------------------------------------------------------------------------

def run_ask(
    settings: Settings,
    registry: ToolRegistry,
    client: MessagesClient,
    out: Console,
    question: str,
    use_tools: bool = True,
) -> int:
    """Ask the model one question with the tools attached and show what it requests."""
    tools = registry.definitions() if use_tools else None
    payload = build_request(
        settings,
        [{"role": "user", "content": question}],
        system=DEFAULT_SYSTEM_PROMPT,
        tools=tools,
    )
    if tools:
        out.print(f"Sending your question with [bold]{len(tools)} tool definitions[/]: {', '.join(registry.names)}")
    else:
        out.print("Sending your question with [bold]no tools[/].")

    try:
        with out.status("Waiting for Claude…"):
            response = client.create_message(payload)
    except (ClaudeHTTPError, ClaudeConnectionError) as exc:
        out.print(f"[bold red]API call failed:[/] {exc}")
        return 1

    body = response.body
    for block in body.get("content", []):
        block_type = block.get("type")
        if block_type == "tool_use":
            # The heart of tool use: a structured request for OUR code to run something.
            hint = f"uv run tools call {block['name']} {shlex.quote(json.dumps(block['input']))}"
            out.print(
                Panel(
                    JSON(json.dumps(block)),
                    title="🔧 tool_use: the model asks YOU to run a tool",
                    title_align="left",
                    border_style="yellow",
                )
            )
            out.print(f"[dim]Run it yourself to see the result:[/] {hint}")
        elif block_type == "text":
            out.print(Markdown(block.get("text", "")))
        elif block_type == "thinking":
            out.print(Panel(Text(block.get("thinking", ""), style="dim italic"), title="💭 thinking", border_style="dim"))
        else:
            out.print(Text(json.dumps(block), style="dim"))

    stop_reason = body.get("stop_reason")
    if stop_reason == "tool_use":
        out.print(
            "[bold yellow]stop_reason: tool_use[/]: the model has stopped and is WAITING for a "
            "tool_result. Nothing happens until our code runs the tool and sends the result back. "
            "That's Phase 4's agent loop."
        )
    else:
        out.print(f"[bold]stop_reason: {stop_reason}[/]: the model answered without needing a tool.")

    usage = body.get("usage", {})
    cost = estimate_cost(body.get("model", settings.model), usage)
    cost_text = f" · ≈ ${cost:.5f}" if cost is not None else ""
    note = " The tool definitions are part of the input tokens." if tools else ""
    out.print(
        f"[dim]usage: {usage.get('input_tokens', 0):,} input tokens · {usage.get('output_tokens', 0):,} "
        f"output tokens{cost_text}.{note}[/]"
    )
    return 0


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def local_registry() -> ToolRegistry:
    """The tool registry for `list` and `call`, which need no API key.

    We still respect AGENT_WORKSPACE_DIR when the settings load. If they
    can't load (e.g. no API key yet), fall back to the default workspace,
    because running a tool locally doesn't need the API.
    """
    try:
        return default_registry(load_settings().workspace_dir)
    except ConfigError:
        return default_registry()


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="tools", description="Phase 3: explore the agent's tools.")
      # Sub-commands, like `git commit` / `git push`: `tools list`, `tools call`, `tools ask`.
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("list", help="show the tool definitions the model sees")

    call = commands.add_parser("call", help="run a tool locally with a JSON input")
    call.add_argument("name", help="tool name, e.g. calculator")
    call.add_argument("input", nargs="?", default="{}", help='tool input as JSON, e.g. \'{"expression": "1+1"}\'')

    ask = commands.add_parser("ask", help="ask the model a question with the tools attached")
    ask.add_argument("question")
    ask.add_argument("--no-tools", action="store_true", help="send WITHOUT tools (compare the input tokens)")
    add_settings_arguments(ask)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.command == "list":
        return run_list(local_registry(), console)
    if args.command == "call":
        return run_call(local_registry(), console, args.name, args.input)

    # `ask` talks to the API, so it needs valid settings (API key etc.).
    try:
        settings = apply_overrides(load_settings(), args)
    except ConfigError as exc:
        console.print(f"[bold red]Config error:[/] {exc}")
        return 2
    with MessagesClient(settings) as client:
        return run_ask(settings, default_registry(settings.workspace_dir), client, console, args.question, not args.no_tools)


if __name__ == "__main__":
    raise SystemExit(main())
