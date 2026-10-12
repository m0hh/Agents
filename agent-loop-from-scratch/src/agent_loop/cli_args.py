"""
cli_args.py: command-line options shared by every program in this project.

Both `uv run hello` and `uv run agent` accept --model, --max-tokens and
--thinking to override the .env settings for a single run. Defining them
once here keeps the two programs consistent.

Precedence (highest wins):  command-line flag  >  real env variable  >  .env file  >  default
"""

import argparse
import dataclasses
from typing import Any

from agent_loop.config import Settings


def add_settings_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the shared --model / --max-tokens / --thinking options to a parser."""
    parser.add_argument("--model", help="override AGENT_MODEL")
    parser.add_argument("--max-tokens", type=int, help="override AGENT_MAX_TOKENS")
    parser.add_argument(
        "--thinking",
        type=int,
        metavar="BUDGET",
        help="extended thinking token budget (>= 1024; 0 = off). Overrides AGENT_THINKING_BUDGET",
    )


def apply_overrides(settings: Settings, args: argparse.Namespace) -> Settings:
    """Return a copy of `settings` with any command-line overrides applied.

    Settings are frozen (immutable), so we can't assign to them.
    `dataclasses.replace()` builds a NEW Settings object, which also re-runs
    the validation in `Settings.__post_init__`. A bad flag like
    `--thinking 500` therefore raises ConfigError, just like a bad .env value.
    """
    overrides: dict[str, Any] = {}
    if args.model:
        overrides["model"] = args.model
    if args.max_tokens is not None:
        overrides["max_tokens"] = args.max_tokens
    if args.thinking is not None:
        overrides["thinking_budget"] = args.thinking
    return dataclasses.replace(settings, **overrides)
