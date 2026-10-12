"""
config.py: every setting the app needs, loaded from environment variables.

WHY A CONFIG MODULE?
    Hard-coding values like the model name or the API key inside the code
    is a classic mistake:
      * secrets end up committed to git,
      * changing the model means editing code.
    Instead we read everything from ENVIRONMENT VARIABLES, and we let a `.env`
    file provide them during development. This is the "12-factor app" style
    used by almost every production service (FastAPI apps included).

WHAT YOU SHOULD NOTICE IN THIS FILE
    1. `Settings` is a *frozen dataclass*: once created it can't be changed,
       so no part of the program can silently modify the config.
    2. The API key is excluded from `repr()`, so printing or logging the
       settings never leaks your secret.
    3. All validation happens in ONE place (`__post_init__`), when the
       settings are created. Bad config fails immediately with a clear
       message, instead of failing later inside an HTTP call.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# The latest Haiku model: fast and cheap ($1 per 1M input tokens, $5 per 1M
# output tokens), with a 200K-token context window and up to 64K output tokens.
# We use the ALIAS "claude-haiku-4-5" (no date suffix) as the default.
DEFAULT_MODEL = "claude-haiku-4-5"

# The root URL of the Claude API. Every endpoint lives under it, e.g.
# https://api.anthropic.com/v1/messages
DEFAULT_BASE_URL = "https://api.anthropic.com"

# The API version header value. Anthropic versions the API with a date, and
# "2023-06-01" is the current stable version of the Messages API. New
# features are added as optional fields or "beta" headers, so this value
# rarely changes.
ANTHROPIC_VERSION = "2023-06-01"

# Extended thinking needs a budget of at least 1024 tokens (an API rule).
MIN_THINKING_BUDGET = 1024

# Default number of seconds to wait for a response. LLM calls are SLOW compared
# with normal web APIs: a long answer can take a minute or more to generate.
# (Note: httpx2's own default is only 5 seconds, far too short for us.)
DEFAULT_TIMEOUT_SECONDS = 120.0

# Default ceiling on how many tokens the model may WRITE in one response.
# It's a *ceiling*, not a target: you pay only for the tokens actually
# generated. It must also leave room for thinking, if thinking is on.
DEFAULT_MAX_TOKENS = 16_000

# The `.env` file sits in the project root (two folders above this file):
#   agent-loop-from-scratch/        ← parents[2]  (.env lives here)
#   └── src/                        ← parents[1]
#       └── agent_loop/             ← parents[0]
#           └── config.py           ← __file__
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"

# The ONLY folder the read_file tool may read from (Phase 3). The model can
# ask to read any path it likes; the tool refuses anything outside this
# "sandbox" folder.
DEFAULT_WORKSPACE_DIR = PROJECT_ROOT / "workspace"


# -----------------------------------------------------------------------------
# Errors
# -----------------------------------------------------------------------------

class ConfigError(Exception):
    """Raised when a setting is missing or invalid.

    A dedicated exception type lets the CLI catch *config* problems and print
    a friendly hint, without accidentally hiding unrelated bugs.
    """


# -----------------------------------------------------------------------------
# The Settings object
# -----------------------------------------------------------------------------

# `frozen=True` makes instances immutable (assigning an attribute raises an
# error). To get a modified copy, use `dataclasses.replace(settings, model=...)`.
@dataclass(frozen=True)
class Settings:
    """All runtime settings of the app, validated when created."""

    # The secret key sent in the `x-api-key` header.
    # `repr=False` hides it from repr(), print() and log output.
    api_key: str = field(repr=False)

    # Which Claude *workspace* the requests belong to (IDs look like
    # "wrkspc_01..."). A Claude organization can have several workspaces, each
    # with its own limits and billing. A key created INSIDE a workspace
    # already knows its workspace, so this stays None. A key that is NOT
    # scoped to a workspace must name one on every request, through the
    # `anthropic-workspace-id` header; otherwise the API answers 400.
    workspace_id: str | None = None

    # Which model to call (e.g. "claude-haiku-4-5").
    model: str = DEFAULT_MODEL

    # Max tokens the model may generate per response (thinking + visible text).
    max_tokens: int = DEFAULT_MAX_TOKENS

    # Extended thinking budget in tokens. 0 means thinking is OFF.
    # When on, the model "thinks" (reasons privately) before answering.
    # Thinking tokens are billed as OUTPUT tokens and count toward max_tokens.
    thinking_budget: int = 0

    # Where to send requests (the API root, without "/v1/...").
    base_url: str = DEFAULT_BASE_URL

    # Seconds to wait for the server before giving up on a request.
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    # The sandbox folder for the read_file tool.
    workspace_dir: Path = DEFAULT_WORKSPACE_DIR

    # `__post_init__` runs automatically right after the dataclass sets its
    # fields. It's the standard place to validate a dataclass.
    def __post_init__(self) -> None:
        if not self.api_key:
            raise ConfigError(
                "ANTHROPIC_API_KEY is not set. Put it in the .env file or "
                "export it in your shell."
            )
        if not self.model:
            raise ConfigError("The model name must not be empty.")
        if self.max_tokens < 1:
            raise ConfigError(f"max_tokens must be >= 1 (got {self.max_tokens}).")
        if self.timeout_seconds <= 0:
            raise ConfigError(f"timeout_seconds must be > 0 (got {self.timeout_seconds}).")

        # Thinking has two API rules, and we check both here so you get a
        # clear message instead of a 400 error from the server:
        #   1. the budget must be at least 1024 tokens;
        #   2. the budget must be SMALLER than max_tokens, because the thinking
        #      and the final answer share the max_tokens allowance.
        if self.thinking_budget != 0:
            if self.thinking_budget < MIN_THINKING_BUDGET:
                raise ConfigError(
                    f"thinking_budget must be 0 (off) or >= {MIN_THINKING_BUDGET} "
                    f"(got {self.thinking_budget})."
                )
            if self.thinking_budget >= self.max_tokens:
                raise ConfigError(
                    f"thinking_budget ({self.thinking_budget}) must be smaller than "
                    f"max_tokens ({self.max_tokens})."
                )

    @property
    def thinking_enabled(self) -> bool:
        """True when extended thinking should be requested."""
        return self.thinking_budget > 0


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------

def _read_int(environ: Mapping[str, str], name: str, default: int) -> int:
    """Read an integer environment variable, with a clear error if it's not a number.

    Environment variables are ALWAYS strings, so every number has to be
    converted explicitly. `int("abc")` would raise a confusing ValueError,
    so we turn it into a ConfigError that names the variable.
    """
    raw = environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw.replace("_", ""))  # allow "16_000" as well as "16000"
    except ValueError:
        raise ConfigError(f"{name} must be a whole number (got {raw!r}).") from None


def _read_float(environ: Mapping[str, str], name: str, default: float) -> float:
    """Same as `_read_int`, but for decimal numbers such as a timeout of 7.5s."""
    raw = environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a number (got {raw!r}).") from None


def load_settings(
    environ: Mapping[str, str] | None = None,
    env_file: Path | None = DEFAULT_ENV_FILE,
) -> Settings:
    """Build a validated `Settings` object from environment variables.

    Args:
        environ: where to read variables from. `None` (the normal case) means
            "the real process environment" (`os.environ`). Tests pass a plain
            dict instead, so they don't depend on your machine's environment.
            Passing the source in as a parameter like this is called
            *dependency injection*, and it makes code easy to test.
        env_file: the `.env` file to load first (only used when `environ` is
            None). Pass `None` to skip loading a file.

    Variables read:
        ANTHROPIC_API_KEY      (required) the secret API key
        ANTHROPIC_WORKSPACE_ID (optional) only needed if your key isn't scoped to a workspace
        AGENT_MODEL            model name            (default: claude-haiku-4-5)
        AGENT_MAX_TOKENS       output token ceiling  (default: 16000)
        AGENT_THINKING_BUDGET  thinking tokens, 0=off (default: 0)
        AGENT_TIMEOUT_SECONDS  request timeout        (default: 120)
        AGENT_BASE_URL         API root URL           (default: https://api.anthropic.com)
        AGENT_WORKSPACE_DIR    read_file's sandbox    (default: <project>/workspace)

    Why the AGENT_ prefix? Other tools on your machine (e.g. Claude Code or the
    official SDK) may read variables such as ANTHROPIC_MODEL or
    ANTHROPIC_BASE_URL. Using our own prefix means we never pick up their
    settings by accident. ANTHROPIC_API_KEY and ANTHROPIC_WORKSPACE_ID keep
    their standard names because they describe your Anthropic account, and
    that's the name the official tools use too.
    """
    if environ is None:
        # load_dotenv() copies KEY=VALUE lines from the file into os.environ.
        # By default it does NOT override variables that already exist, so a
        # real `export ANTHROPIC_API_KEY=...` in your shell wins over `.env`.
        # If the file doesn't exist, it quietly does nothing.
        if env_file is not None:
            load_dotenv(env_file)
        environ = os.environ

    return Settings(
        api_key=environ.get("ANTHROPIC_API_KEY", "").strip(),
        # `or None` turns an empty string into None ("not set").
        workspace_id=environ.get("ANTHROPIC_WORKSPACE_ID", "").strip() or None,
        model=environ.get("AGENT_MODEL", "").strip() or DEFAULT_MODEL,
        max_tokens=_read_int(environ, "AGENT_MAX_TOKENS", DEFAULT_MAX_TOKENS),
        thinking_budget=_read_int(environ, "AGENT_THINKING_BUDGET", 0),
        base_url=environ.get("AGENT_BASE_URL", "").strip() or DEFAULT_BASE_URL,
        timeout_seconds=_read_float(environ, "AGENT_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS),
        workspace_dir=Path(environ.get("AGENT_WORKSPACE_DIR", "").strip() or DEFAULT_WORKSPACE_DIR),
    )
