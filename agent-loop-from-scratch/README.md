# Agent Loop From Scratch

An AI agent built with **no agent framework**: plain HTTP to the Claude Messages API, hand-written tools, and our own loop, stop conditions, retries and token accounting.

👉 The full roadmap, the concepts and the API cheat sheet are in **[PROJECT_PLAN.md](PROJECT_PLAN.md)**.

## Setup

```bash
cd agent-loop-from-scratch

# 1. Install the dependencies into .venv (uv reads pyproject.toml + uv.lock)
uv sync

# 2. Create your .env from the template and add your API key
cp .env.example .env
#    then edit .env:  ANTHROPIC_API_KEY=sk-ant-...
#    If the API says your key "is not scoped to a workspace", also set
#    ANTHROPIC_WORKSPACE_ID=wrkspc_...  (Claude Console → Settings → Workspaces)
```

## Run

| Phase | Command | What it does |
|-------|---------|--------------|
| 1 | `uv run hello` | Sends one message over raw HTTP and explains the request and response |
| 1 | `uv run hello "Is 1001 prime?" --thinking 2048` | The same, with extended thinking turned on |
| 2 | `uv run agent` | Interactive chat with memory. Try `/history` and `/stats` |
| 2 | `uv run agent --thinking 2048` | The same, with thinking shown dimmed above each answer |
| 2 | `uv run python experiments/message_rules.py` | Probes which "invalid" conversations the real API accepts |
| 3 | `uv run tools list` | Shows the tool definitions (JSON Schemas) the model sees |
| 3 | `uv run tools call read_file '{"path": "."}'` | Runs a tool locally (no model, no API key needed) |
| 3 | `uv run tools ask "What is 2**32 / 7?"` | Shows the `tool_use` block the model answers with (doesn't run it) |

Add `--help` to any command to see all its options.

## Tests

```bash
uv run pytest        # free and offline: the API is replaced by a fake network
```

## Configuration (`.env`)

| Variable | Default | Meaning |
|----------|---------|---------|
| `ANTHROPIC_API_KEY` | (required) | Your secret API key |
| `ANTHROPIC_WORKSPACE_ID` | (unset) | Only needed if your key isn't scoped to a workspace |
| `AGENT_MODEL` | `claude-haiku-4-5` | Which model to call |
| `AGENT_MAX_TOKENS` | `16000` | Max tokens the model may write per response |
| `AGENT_THINKING_BUDGET` | `0` (off) | Extended-thinking budget (≥ 1024 and < max tokens) |
| `AGENT_TIMEOUT_SECONDS` | `120` | How long to wait for the API |
| `AGENT_BASE_URL` | `https://api.anthropic.com` | API root URL |
| `AGENT_WORKSPACE_DIR` | `./workspace` | The only folder `read_file` may read |
