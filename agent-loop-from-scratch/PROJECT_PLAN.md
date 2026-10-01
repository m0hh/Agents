# Agent Loop From Scratch: Project Plan

> A command-line AI assistant built with **no agent framework**. It sends plain HTTP requests to the Claude Messages API, defines its own tools, runs those tools itself, sends the results back, and decides when to stop.
>
> The assistant itself is not the main point. The main point is to understand every part that LangChain, LangGraph and SDK "tool runners" normally hide from you.

---

## Progress tracker

| # | Phase | Status |
|---|-------|--------|
| 1 | Hello, raw API: your first HTTP request to the model | ✅ |
| 2 | Conversation memory: a chat REPL with no tools | ⬜ |
| 3 | Tools: JSON schemas and implementations | ⬜ |
| 4 | The agent loop: run tools and feed results back | ⬜ |
| 5 | Stop conditions and guardrails | ⬜ |
| 6 | Reliability: retries, timeouts and errors | ⬜ |
| 7 | Measuring tokens, cost and context growth | ⬜ |
| 8 | Controlling growth: prompt caching and context management | ⬜ |
| 9 | Streaming by hand (Server-Sent Events) | ⬜ |
| 10 | Capstone: rebuild it with the frameworks and compare | ⬜ |

When you're ready for the next phase, say **"implement next phase"**.

---

## 1. Why this project?

When you write something like `create_agent(model, tools)` in a framework, a lot happens that you never see:

1. It converts your Python functions into **JSON Schemas** the model can read.
2. It builds the provider's exact **request JSON** (roles, content blocks, tool definitions).
3. It reads **tool calls** out of the response.
4. It **runs your tools**, catches their exceptions and formats their results.
5. It matches each result to the call that asked for it, **by ID**.
6. It decides **when to stop** looping.
7. It **retries** failed HTTP calls with backoff.
8. It **counts tokens** and tracks cost.
9. It **manages context** as the conversation grows.

In this project you write each of these yourself. Afterwards, any agent framework will read like code you have already written.

---

## 2. What we're building

A terminal assistant with three tools:

| Tool | What it does | What it teaches |
|------|--------------|-----------------|
| `calculator` | Evaluates math like `(1250 * 0.14) + 99` | Simple schema; why you **never** use `eval()` on model output |
| `read_file` | Reads a text file from a sandbox folder (`./workspace`) | Path-traversal safety; keeping tool output small |
| `fetch_url` | Downloads a web page and converts it to plain text | Timeouts, size limits, untrusted content (prompt injection) |

Target experience (by Phase 7):

```text
$ uv run agent
You › Read prices.csv and give me the total with 14% VAT.
  🔧 read_file(path="prices.csv")                     ✓ 412 chars
  🔧 calculator(expression="(120 + 45.5 + 300) * 1.14")  ✓ 530.67
Assistant › The subtotal is 465.50; with 14% VAT the total is 530.67.
  [3 API calls · in 6,210 tok · out 1,044 tok · $0.011 · context 2.9K / 50K]
```

---

## 3. Tech stack

| Piece | Choice | Why |
|-------|--------|-----|
| Language | Python 3.12+ (this machine has 3.14) | Your main AI-engineering language; matches the FastAPI/LangGraph work later |
| Project manager | [`uv`](https://docs.astral.sh/uv/) | The current community standard for Python projects: fast installs, lockfile, `uv run` |
| HTTP client | `httpx2` | The maintained, API-compatible fork of `httpx` (the original hasn't been released since Dec 2024). The official Anthropic SDK uses it internally, so we do by hand what the SDK does for you. |
| Model API | Claude Messages API: `POST https://api.anthropic.com/v1/messages` | One endpoint handles chat, tools, thinking, caching and streaming |
| Model | `claude-haiku-4-5` (latest Haiku), configurable with `AGENT_MODEL` | Fast and cheap: $1 / $5 per 1M input/output tokens, 200K-token context window, 64K max output |
| HTML → text | `beautifulsoup4` | The most widely used HTML parser, used by `fetch_url` |
| Terminal UI | `rich` | Colors, panels and the token/cost tables |
| Config | `python-dotenv` | Loads the API key from `.env` |
| Tests | `pytest` and `httpx2.MockTransport` | Tests the whole loop against fake API responses, so tests cost nothing |

**Deliberately not used:** the `anthropic` SDK, LangChain, LangGraph, or any "tool runner". They appear only in Phase 10, for comparison.

> **Note on FastAPI + React:** your usual stack is FastAPI + React + LangGraph. This project is CLI-only on purpose, because the goal is to understand the loop itself. Later projects will use these fundamentals inside FastAPI/LangGraph apps.

---

## 4. The big picture: the agent loop

This is the whole project in one diagram. Each phase builds one piece of it.

```text
  user types a question
          │
          ▼
  messages.append({"role": "user", "content": "..."})
          │
          ▼
  ┌──────────────────────────────────────────────────────────┐
  │  POST /v1/messages   { model, system, tools, messages }   │  ◄── retries (Phase 6)
  └──────────────────────────────────────────────────────────┘
          │
          ▼
  messages.append({"role": "assistant", "content": response.content})   (copied exactly)
          │
          ▼
  what is response.stop_reason?
     │
     ├── "tool_use"  ──► run EVERY tool_use block locally (Phase 4)
     │                    messages.append({"role": "user", "content": [all tool_result blocks]})
     │                    check guardrails: iterations? cost? context? Ctrl-C? (Phase 5)
     │                    └──► loop back to POST
     │
     ├── "end_turn"  ──► print the answer, STOP ✅
     ├── "max_tokens" ─► output was cut off; handle it (Phase 5)
     ├── "refusal"   ──► a safety classifier declined; don't run tools (Phase 5)
     └── anything else ► stop safely and log it (Phase 5)
```

---

## 5. Cheat sheet: the Messages API wire format

You'll come back to this section often.

### 5.1 Request

```http
POST https://api.anthropic.com/v1/messages
x-api-key: <your key>
anthropic-version: 2023-06-01
content-type: application/json
anthropic-workspace-id: wrkspc_...     ← only if your key isn't scoped to a workspace
```

```json
{
  "model": "claude-haiku-4-5",
  "max_tokens": 16000,
  "system": "You are a helpful command-line assistant...",
  "thinking": { "type": "enabled", "budget_tokens": 2048 },
  "tools": [
    {
      "name": "calculator",
      "description": "Evaluate an arithmetic expression. Use this whenever the user needs an exact numeric result.",
      "input_schema": {
        "type": "object",
        "properties": {
          "expression": { "type": "string", "description": "e.g. (1250 * 0.14) + 99" }
        },
        "required": ["expression"],
        "additionalProperties": false
      },
      "strict": true
    }
  ],
  "messages": [
    { "role": "user", "content": "What is 17% of 2,340?" }
  ]
}
```

### 5.2 Response: the model wants to use a tool

```json
{
  "id": "msg_01...",
  "type": "message",
  "role": "assistant",
  "model": "claude-haiku-4-5",
  "content": [
    { "type": "thinking", "thinking": "The user wants 17% of 2340...", "signature": "EqQB..." },
    { "type": "tool_use", "id": "toolu_01A...", "name": "calculator", "input": { "expression": "0.17 * 2340" } }
  ],
  "stop_reason": "tool_use",
  "usage": {
    "input_tokens": 812,
    "output_tokens": 95,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0
  }
}
```

### 5.3 The next request's `messages`: send the result back

```json
[
  { "role": "user", "content": "What is 17% of 2,340?" },
  { "role": "assistant", "content": [ "...the response content above, copied exactly..." ] },
  { "role": "user", "content": [
      { "type": "tool_result", "tool_use_id": "toolu_01A...", "content": "397.8" }
  ]}
]
```

### 5.4 Content block types

| Block `type` | Who sends it | Meaning |
|--------------|--------------|---------|
| `text` | user or assistant | Normal text |
| `thinking` | assistant | The model's reasoning. Signed; you must send it back **unchanged** |
| `tool_use` | assistant | "Please run tool `name` with this `input`." Has a unique `id` |
| `tool_result` | user (that's you, the program) | Output of a tool; `tool_use_id` points to the call. Add `is_error: true` if the tool failed |

### 5.5 The five rules of the message list

1. **Roles alternate**: `user`, `assistant`, `user`, ... The first message is always `user`.
2. **Copy the assistant's `content` array back exactly**, including `thinking` blocks. Don't extract only the text.
3. **Every `tool_use` gets exactly one `tool_result`** with the same ID, in the very next `user` message.
4. **All tool results from one turn go in one `user` message.** If you split them, the model learns to stop making parallel calls.
5. **History is append-only.** Never edit or delete earlier messages, and never change the system prompt or tool list mid-conversation. Editing breaks the prompt cache (Phase 8). Some newer models also reject an edited history that contains thinking blocks with a `400` error, so the habit protects you when you switch models.

### 5.6 `stop_reason` values

| Value | Meaning | Our reaction |
|-------|---------|--------------|
| `end_turn` | Model finished its answer | Stop, print the answer |
| `tool_use` | Model wants tools run | Run them, send results, loop |
| `max_tokens` | Hit the `max_tokens` limit; the output is cut off | Don't run a half-written tool call; retry bigger or report it |
| `stop_sequence` | Hit one of your `stop_sequences` | Stop (we won't use custom sequences much) |
| `refusal` | Safety classifier declined (still HTTP **200**) | Report it with `stop_details`; never run that turn's tools |
| `pause_turn` | Only happens with *server-side* tools (e.g. Anthropic's hosted web search) | We use client tools only, but we'll handle it defensively |

### 5.7 HTTP errors

| Code | Type | Retry? |
|------|------|--------|
| 400 | `invalid_request_error`: your JSON is wrong | ❌ Fix the code |
| 401 | `authentication_error`: bad or missing key | ❌ |
| 402 | `billing_error` | ❌ |
| 403 | `permission_error` | ❌ |
| 404 | `not_found_error`: e.g. a typo in the model name | ❌ |
| 413 | `request_too_large` | ❌ Shrink the request |
| 429 | `rate_limit_error` | ✅ Wait for the `retry-after` header |
| 500 | `api_error` | ✅ Backoff |
| 529 | `overloaded_error` | ✅ Backoff |
| - | Network error or timeout | ✅ Backoff |

Error body shape: `{"type": "error", "error": {"type": "...", "message": "..."}, "request_id": "req_..."}`

### 5.8 Model facts that affect our code (`claude-haiku-4-5`)

- **Size limits:** 200K-token context window; up to 64K output tokens per response.
- **Thinking is off by default.** Turn it on with `"thinking": {"type": "enabled", "budget_tokens": N}`, where N ≥ 1024 and N < `max_tokens` (the thinking and the answer share `max_tokens`). Our setting: `AGENT_THINKING_BUDGET` (0 = off).
- Thinking tokens are **billed as output tokens**.
- `output_config.effort` is **not supported** on Haiku 4.5 (400). Bigger models use effort and "adaptive" thinking instead of a budget. Request fields differ between model families.
- With thinking on, don't send `temperature` / `top_k`, and only `tool_choice` `auto` (the default) or `none` work. We always use `auto` and steer the model through the prompt and tool descriptions.
- Pricing per 1M tokens: input **$1**, output **$5**, cache read **$0.10**, 5-minute cache write **$1.25**.
- Prompt caching only starts once the cacheable prefix reaches **4096 tokens** on Haiku 4.5. Shorter prompts are silently not cached (Phase 8).

---

## 6. Final project structure

```text
agent-loop-from-scratch/
├── PROJECT_PLAN.md          ← this file
├── README.md                ← how to install & run (Phase 1)
├── pyproject.toml           ← uv project + dependencies
├── .env.example             ← copy to .env and add your API key
├── src/agent_loop/
│   ├── config.py            ← settings from environment variables        (P1)
│   ├── api_client.py        ← raw HTTP to /v1/messages                    (P1, P6, P9)
│   ├── conversation.py      ← the append-only message list                (P2)
│   ├── tools/
│   │   ├── base.py          ← the Tool definition + ToolError             (P3)
│   │   ├── calculator.py    ← safe math via Python's `ast` module         (P3)
│   │   ├── file_reader.py   ← sandboxed read_file                         (P3)
│   │   ├── web_fetch.py     ← fetch_url with limits                       (P3)
│   │   └── __init__.py      ← tool registry                               (P3)
│   ├── agent.py             ← THE LOOP                                    (P4, P5)
│   ├── usage.py             ← token + cost accounting                     (P7)
│   ├── tracing.py           ← JSONL log of every request/response         (P7)
│   ├── context.py           ← context budget + compaction                 (P8)
│   ├── sse.py               ← hand-written streaming parser               (P9)
│   └── cli.py               ← the terminal UI (REPL)                      (P2 onward)
├── workspace/               ← the only folder read_file may read
├── traces/                  ← raw request/response logs (git-ignored)
├── tests/                   ← pytest, using fake API responses
└── comparisons/             ← the same agent built with SDK / LangGraph   (P10)
```

---

## 7. The phases

Each phase has the same parts:
- **Goal**: one sentence.
- **You'll learn**: the concepts.
- **We'll build**: the files.
- **Try it**: experiments to run yourself. This is where most of the learning happens.
- **Done when**: how you know the phase is finished.
- **What frameworks hide here**: what this phase looks like inside a framework.
- **Check yourself**: questions you should be able to answer afterwards.

---

### Phase 1: Hello, raw API

**Goal:** Send one message with plain HTTP and understand every field in the request and the response.

**You'll learn:**
- The endpoint and the required headers (`x-api-key`, `anthropic-version`, `content-type`), plus `anthropic-workspace-id` for keys that aren't scoped to a workspace.
- The required body fields (`model`, `max_tokens`, `messages`) and useful optional ones (`system`, `thinking`).
- The response's parts: `id`, `content` (a **list of typed blocks**, not a string), `stop_reason`, `usage`.
- The API is **stateless**: it remembers nothing between calls.
- How to keep secrets out of git (`.env`, `.gitignore`).

**We'll build:**
- The project scaffold: `uv init`, `pyproject.toml`, `.env.example`, `.gitignore`, `README.md`.
- `config.py`: a frozen `Settings` dataclass loaded from environment variables and validated in one place.
- `api_client.py`: `MessagesClient.create_message(payload) -> APIResponse` using `httpx2`, with our own error types. No retries yet.
- `hello.py`: sends one message and prints the **raw** request and response, then explains every field.
- Tests that use a fake network (`httpx2.MockTransport`), so they're free and work offline.

**Try it:**
- Run it and read the response JSON field by field.
- Run with `--thinking 2048` and compare the content blocks and `usage.output_tokens` with a run without thinking.
- Use a wrong API key and read the 401 error body. Use a misspelled model name and read the 404.
- Use `--max-tokens 20` on a long question and watch `stop_reason` become `max_tokens`.

**Done when:** you can explain every field of the response JSON without looking it up.

**What frameworks hide here:** `ChatAnthropic(...).invoke("hi")` builds this exact request and hides most of the response (usage, stop reason, thinking blocks) inside wrapper objects.

**Check yourself:** Why is `content` a list? Where do the thinking tokens show up in `usage`?

---

### Phase 2: Conversation memory (chat REPL, no tools yet)

**Goal:** A multi-turn chat that shows "memory" is just a list you send again every time.

**You'll learn:**
- How the `messages` list is built, and the role-alternation rule.
- Why you append the assistant's full `content` array, not just its text.
- Because the API is stateless, **every turn resends the whole history**, so `input_tokens` grows every turn. This is your first view of context growth.
- How to render content blocks by `type` (show text normally and thinking summaries dimmed).

**We'll build:**
- `conversation.py`: a `Conversation` class holding `messages`, with only *append* methods (append-only by design).
- `cli.py`: a REPL built with `rich`. Commands: `/exit`, `/reset`, `/history` (prints the raw JSON list).
- A footer after each turn showing input and output tokens.

**Try it:**
- Chat for 6 turns and watch `input_tokens` rise each turn.
- Type `/history` and read the raw array that is sent to the API.
- Send two `user` messages in a row on purpose and read the 400 error.

**Done when:** the REPL remembers earlier turns and you can explain why tokens grow.

**What frameworks hide here:** "Memory" classes and LangGraph checkpointers are, at their core, this list plus storage.

**Check yourself:** If turn 1 costs 500 input tokens and each turn adds about 300, roughly what does turn 10's input cost?

---

### Phase 3: Tools, their JSON schemas and implementations

**Goal:** Define three tools as plain Python functions with hand-written JSON Schemas, and watch the model *ask* for a tool. We don't run the tool yet.

**You'll learn:**
- The parts of a tool definition: `name`, `description`, `input_schema`.
- **Descriptions are prompts.** Say *when* to use the tool, not only what it does.
- `strict: true` + `additionalProperties: false`: the API guarantees the model's input matches your schema.
- **Client tools vs server tools.** Anthropic also hosts server-side tools, such as its own web fetch. Ours run on *your* machine, and you control them completely.
- **Tools are an attack surface.** Tool inputs come from the model, so treat them as untrusted:
  - `calculator` parses with Python's `ast` module and allows only math operations. Never `eval()`.
  - `read_file` resolves the path and refuses anything outside `./workspace`, blocking `../../etc/passwd`. It also caps file size.
  - `fetch_url` accepts only `http`/`https`, uses a timeout and a max download size, blocks `localhost` and private IPs (SSRF), and converts HTML to text. Fetched pages can contain prompt injection, so their text is data, never instructions.
- **Keep tool outputs small.** Every character you return is resent on every later loop iteration.

**We'll build:**
- `tools/base.py`: a `Tool` dataclass (`name`, `description`, `input_schema`, `run`) and a `ToolError` exception.
- `tools/calculator.py`, `tools/file_reader.py`, `tools/web_fetch.py`.
- `tools/__init__.py`: a registry with `tool_definitions()` (the list sent to the API) and `get_tool(name)`.
- `workspace/` with sample files (e.g. `prices.csv`, `notes.md`).
- `tests/test_tools.py`: unit tests, including the attacks (path traversal, `__import__` in the calculator).
- A one-shot script that sends a question with the tools attached and prints the `tool_use` block that comes back.

**Try it:**
- Ask "What is 2**32 / 7?" and read the `tool_use` block: `id`, `name`, `input`.
- Ask "What's the capital of France?" and see the model answer without any tool.
- Make the calculator's description vague and see whether the model still chooses it.

**Done when:** tests pass, and you can read a `tool_use` block and say what the model wants.

**What frameworks hide here:** `@tool` decorators read your function's type hints and docstring and generate this JSON Schema automatically.

**Check yourself:** Why is `eval(model_input)` dangerous even on a laptop? Why does a 50KB tool result hurt you *five* iterations later?

---

### Phase 4: The agent loop

**Goal:** Close the loop: run the tool calls, feed the results back, and repeat until the model is done.

**You'll learn:**
- The core cycle: *call → check `stop_reason` → run tools → send `tool_result`s → call again*.
- Matching `tool_result.tool_use_id` to `tool_use.id`.
- **Parallel tool calls:** one response may contain several `tool_use` blocks. Run them all and return all results in one `user` message.
- **Errors become results.** For an unknown tool, invalid input or an exception, return `is_error: true` with a helpful message, and the model usually corrects itself.
- Validate tool inputs before running them, even with `strict: true`.

**We'll build:**
- `agent.py`: `Agent.run(user_text)` containing the `while` loop, plus `execute_tool_calls(blocks) -> list[tool_result]`.
- Simple **event callbacks** (`on_tool_call`, `on_tool_result`, `on_text`) so the CLI can print each step. A small version of framework callbacks.
- `cli.py` now uses the Agent.
- `tests/test_agent.py`: uses `httpx2.MockTransport` to return scripted responses (a `tool_use`, then `end_turn`) and checks the exact JSON the agent sends.

**Try it:**
- Multi-step: "Read prices.csv and compute the total with 14% VAT."
- Parallel: "What are 3^7 and 7^3?" Watch for two `tool_use` blocks in one response.
- Error recovery: "Read secrets.txt" (it doesn't exist) and watch the model react to `is_error`.

**Done when:** the agent answers multi-step questions, and the tests check the exact request payloads.

**What frameworks hide here:** LangGraph's `ToolNode` plus the "should I continue?" conditional edge; LangChain's `AgentExecutor`.

**Check yourself:** What happens to the *next* request if you forget to send a `tool_result` for one `tool_use`?

---

### Phase 5: Stop conditions and guardrails

**Goal:** The loop always stops for the right reason, never runs forever, and never leaves the history in an invalid state.

**You'll learn:**
- The correct reaction to every `stop_reason` (table 5.6), including:
  - `max_tokens` during a tool call means the tool input may be cut off. Don't run it; retry once with a larger `max_tokens`, or report it.
  - `refusal`: the model declined to continue (still HTTP 200!). Report it clearly and don't run that turn's tools.
- **Our own stop conditions**, which the API doesn't know about:
  - max loop iterations
  - max tool calls per task
  - max cost (in dollars) per task
  - wall-clock timeout
  - loop detection (the same tool with the same input 3 times in a row)
  - Ctrl-C from the user
- **The history invariant:** if we stop after the model asked for tools, the history ends in a `tool_use` with no answer. The fix is to append `tool_result`s marked `is_error: "Cancelled by user"`. We add to the history instead of deleting from it.
- Returning a structured `RunResult(stop_reason, final_text, iterations, cost)` instead of just a string.

**We'll build:** guardrail settings in `config.py`, the stop logic in `agent.py`, a `RunResult` type, and a test for every stop path.

**Try it:**
- Set `MAX_ITERATIONS=2` and give a 4-step task.
- Set `MAX_TOKENS=300` and watch `max_tokens` appear, since thinking uses up the budget.
- Press Ctrl-C during a slow `fetch_url`, then keep chatting. The conversation must still work.

**Done when:** every row of table 5.6 and every guardrail has a passing test.

**What frameworks hide here:** LangGraph's `recursion_limit` and AgentExecutor's `max_iterations`.

**Check yourself:** Why is a *cost* budget more useful than an *iteration* budget for a real product?

---

### Phase 6: Reliability, with retries, timeouts and errors

**Goal:** Survive a flaky network and a busy API without hiding real bugs.

**You'll learn:**
- Parsing the error body and turning it into Python exceptions.
- **Retryable** (429, 500, 529, network errors, timeouts) vs **not retryable** (400, 401, 403, 404, 413). Retrying a 400 just repeats your bug.
- **Exponential backoff with jitter** (e.g. waits of 1s, 2s, 4s... each randomized), capped by max attempts and max total wait.
- Respecting the `retry-after` header on 429.
- Timeouts: *connect* vs *read*. Thinking makes responses slow, so the read timeout must be generous. Streaming (Phase 9) is the long-term fix.
- The rate-limit headers the API returns on every response.
- **Three levels of "retry" in an agent:**
  1. transport level: resend the HTTP request (this phase)
  2. semantic level: the model retries after an `is_error` tool result (Phase 4)
  3. task level: re-ask after a cut-off response (Phase 5)
- **Idempotency:** retrying a model call is safe. Retrying a tool that *sends an email* is not. Our tools are read-only on purpose.

**We'll build:** an exception hierarchy in `api_client.py` (`RetryableAPIError`, `NonRetryableAPIError`, ...), `send_with_retries()`, retry logging, and tests where the fake API returns `529, 529, 200` or a `400`. The sleep function is injectable, so tests don't actually wait.

**Try it:**
- Set a 1-second read timeout and watch timeouts and retries happen.
- Point the base URL at a non-existent host to see connection errors.

**Done when:** tests prove it retries 529 and then succeeds, never retries 400, and respects `retry-after`.

**What frameworks hide here:** the official SDK retries 408/409/429/5xx twice by default with backoff (`max_retries=2`), and you never see it happen.

**Check yourself:** Why add *random* jitter to the backoff instead of waiting exactly 1s, 2s, 4s?

---

### Phase 7: Measuring tokens, cost and context growth

**Goal:** See exactly where tokens and money go during an agent run.

**You'll learn:**
- All the `usage` fields: `input_tokens`, `output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`.
- Converting usage into dollars for `claude-haiku-4-5`.
- **Why agent cost grows roughly quadratically.** Iteration *k* resends everything from iterations *1..k−1*, so the *total* input across a task is a sum of growing numbers.
- Hidden thinking tokens are still billed as output.
- `POST /v1/messages/count_tokens` measures a request *before* you send it. It is free to call (rate-limited). Don't estimate Claude tokens with OpenAI's `tiktoken`, which uses a different tokenizer.
- The context window (200K tokens on Haiku 4.5) vs a **budget** you set yourself (e.g. 50K). Most real apps set a budget far below the maximum.
- **Thinking is a cost lever:** the same task with thinking off vs a 2K vs an 8K budget.

**We'll build:**
- `usage.py`: `UsageRecord`, `UsageTracker` and a price table.
- A `/stats` command: a per-iteration table (input, output, cumulative $, context size).
- `tracing.py`: writes every raw request and response to `traces/<session>.jsonl`.
- A `/tokens` command that calls `count_tokens` on the current history.
- `experiments/growth.py`: runs a fixed multi-step task and prints how the tokens grow.

**Try it:**
- Run the growth experiment with thinking off, then at 2K and 8K budgets, and compare cost *and* answer quality.
- Open a trace file and follow a full tool round trip in raw JSON.

**Done when:** you can roughly predict the next request's input tokens and explain the cost curve.

**What frameworks hide here:** LangSmith and usage callbacks sum these numbers for you, so you rarely look at them.

**Check yourself:** Two agents give the same answer: one in 3 iterations with big tool outputs, one in 6 with small ones. How do you decide which is cheaper?

---

### Phase 8: Controlling growth with prompt caching and context management

**Goal:** Make long sessions cheaper and keep them within a context budget, *without* breaking the append-only rule.

**You'll learn:**
- **Prompt caching:** the API can cache the start of your request (the *prefix*: tools → system → messages, in that order). Cached tokens cost **$0.10 per 1M instead of $1**, which is 90% cheaper. Topics:
  - top-level `"cache_control": {"type": "ephemeral"}` (automatic placement)
  - the minimum cacheable size (4096 tokens on Haiku 4.5, so caching only kicks in once a conversation grows) and the 5-minute lifetime
  - confirming it works with `usage.cache_read_input_tokens`
  - **silent cache breakers:** a timestamp in the system prompt, a changed tool list, reordered JSON keys
- **Context strategies and their trade-offs:**
  1. **Prevent:** cap tool output size when you *insert* it (started in Phase 3).
  2. **Compact:** when `count_tokens` exceeds the budget, ask the model to summarize the conversation, then start a *new* history that begins with the summary. The whole history is replaced, so no stale thinking blocks are sent again.
  3. **What not to do:** deleting old tool results in place. It breaks the cache prefix and invalidates thinking-block signatures (400 errors on newer accounts).
  4. **API-native options:** server-side compaction and context editing (beta features). These are the "framework" version of what you just built.

**We'll build:** a caching switch in the request builder, `context.py` with a `ContextManager` (pre-flight token count, budget, `compact()`), a `/compact` command, and a context gauge in the footer (`context 12.3K / 50K`).

**Try it:**
- Run the same long session with caching off vs on, and compare the dollar cost.
- Set `CONTEXT_BUDGET=8000` to force compaction, then check that the agent still remembers the important facts.

**Done when:** you can see cache hits in the stats, and compaction lets a session continue past the budget.

**What frameworks hide here:** LangChain's message trimming and summarization middleware, and LangGraph's checkpointer.

**Check yourself:** Why does putting `datetime.now()` in the system prompt silently destroy your cache hit rate?

---

### Phase 9: Streaming by hand (Server-Sent Events)

**Goal:** Print the answer token by token by parsing the raw event stream yourself.

**You'll learn:**
- `"stream": true` and the SSE wire format (`event:` / `data:` lines).
- The event sequence: `message_start` → `content_block_start` → `content_block_delta` → `content_block_stop` → `message_delta` (with `stop_reason` + final `usage`) → `message_stop`, plus `ping` and `error`.
- The delta types: `text_delta`, `thinking_delta`, `signature_delta`, `input_json_delta`. Tool inputs arrive as **JSON fragments** that you join together and parse when the block ends.
- Why streaming prevents timeouts on long responses, and how to handle an error that arrives *mid-stream*.

**We'll build:** `sse.py` (the parser) and `stream_message()` in `api_client.py`, which returns the **same final message shape** as the non-streaming call, so `agent.py` needs almost no changes. That is a lesson in good abstraction.

**Done when:** streaming and non-streaming runs produce identical conversation histories, and a test proves it.

**What frameworks hide here:** `.stream()`, `astream_events()`, and the SDK's `get_final_message()`.

---

### Phase 10: Capstone, meeting the frameworks

**Goal:** Rebuild the same agent in a few lines with (a) the official `anthropic` SDK's tool runner and (b) LangChain/LangGraph, then map every hidden piece back to your own code.

**You'll learn:** what each framework gives you, what it costs you in control and visibility, and when to use which.

**We'll build:**
- `comparisons/sdk_tool_runner.py`
- `comparisons/langgraph_agent.py` (we'll check the latest LangChain/LangGraph API at implementation time)
- `COMPARISON.md`: a mapping table like the one below, filled in with real file and line references.

| Concept | Your code | Anthropic SDK | LangChain / LangGraph |
|---------|-----------|---------------|-----------------------|
| Request building | `api_client.py` | `client.messages.create` | chat model class |
| Tool schema | hand-written JSON Schema | `@beta_tool` decorator | `@tool` decorator |
| The loop | `agent.py` while-loop | tool runner | agent graph / `ToolNode` |
| Stop conditions | `RunResult` + guardrails | runner iteration limit | `recursion_limit` |
| Retries | `send_with_retries` | `max_retries` | client config |
| Token accounting | `usage.py` | `response.usage` | callbacks / LangSmith |
| Context control | `context.py` | compaction helpers | trim / summarize middleware |

**Done when:** you can open any agent framework's source and point to where its loop, tool schemas, stop conditions, retries and token accounting live.

---

## 8. How we'll work through the phases

1. You say **"implement next phase"**.
2. I write the code with **detailed comments above everything**, and explain in chat what it does and why.
3. You read the code top to bottom, run the **Try it** experiments, and answer the **Check yourself** questions. Ask me about anything unclear.
4. Commit after each phase, e.g. `git commit -m "agent-loop: phase 3 - tools"`, so you can compare how the code changed.

---

## 9. Cost and safety notes

- **Get an API key** from the Claude Console and put it in `.env`, which is git-ignored. Never commit it.
- **Set a monthly spend limit** in the Console before you start.
- **Rough costs at `claude-haiku-4-5` prices:** a short chat turn costs about $0.001-$0.003. A 10-iteration agent run whose history grows to about 20K tokens costs roughly $0.10-$0.20. The whole project should cost a dollar or two. Phase 7 teaches you to calculate this exactly instead of guessing.
- **Tests never call the real API.** They use fake responses, so `pytest` is free.
- **All tools are read-only**, so a confused model can't damage anything. We'll discuss what changes when tools have side effects, such as writing files or sending messages.
- You can change `AGENT_MODEL` in `.env`. Other model families accept slightly different parameters (e.g. bigger models use "adaptive" thinking and `effort` instead of a thinking budget), and seeing that is a lesson in itself.

---

## 10. Glossary

| Term | Meaning |
|------|---------|
| **Agent** | An LLM in a loop that can call tools and decides for itself when it's done |
| **Agent loop** | call model → run requested tools → send results → repeat |
| **Tool / function calling** | The model outputs a structured request (`tool_use`) and *your code* runs it |
| **JSON Schema** | A standard way to describe the shape of JSON data; it tells the model what inputs a tool takes |
| **Token** | The unit models read and write, roughly ¾ of an English word. It is also the unit of billing |
| **Context window** | The maximum tokens one request can hold (input + output) |
| **`stop_reason`** | The API's explanation for why the model stopped writing |
| **Stateless API** | The server remembers nothing; you send the full history every time |
| **Prompt caching** | Reusing an already processed request prefix at a large discount |
| **Compaction** | Replacing a long history with a summary to save context |
| **Exponential backoff + jitter** | Waiting longer after each failed attempt, with randomness so clients don't retry at the same moment |
| **Idempotent** | Safe to repeat: doing it twice has the same effect as doing it once |
| **SSE** | Server-Sent Events, an HTTP response that streams many small events |
| **Prompt injection** | Text in a tool result (e.g. a web page) that tries to give the model new instructions |
