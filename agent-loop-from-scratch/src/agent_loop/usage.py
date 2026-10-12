"""
usage.py: turning token counts into money.

Every response includes a `usage` object, for example:
    {"input_tokens": 84, "output_tokens": 286, "cache_read_input_tokens": 0, ...}

Tokens are what you pay for, and input and output tokens have different
prices. This small module holds the price table and the cost formula, so the
whole app agrees on one number. (Phase 7 grows it into full cost tracking,
including prompt-cache prices.)
"""

from typing import Any

# Price per 1 MILLION tokens in US dollars: (input, output).
# Output tokens cost 5x more than input tokens because generating text is far
# more work for the model than reading it.
PRICES_PER_MILLION: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.00, 5.00),
}


def estimate_cost(model: str, usage: dict[str, Any]) -> float | None:
    """Dollar cost of one API call, or None if we don't know the model's price.

    The response names the model's full snapshot version (e.g.
    "claude-haiku-4-5-20251001") even when you asked for the short alias
    "claude-haiku-4-5", so we match by prefix.
    """
    for known_model, (input_price, output_price) in PRICES_PER_MILLION.items():
        if model.startswith(known_model):
            return (
                usage.get("input_tokens", 0) * input_price
                + usage.get("output_tokens", 0) * output_price
            ) / 1_000_000
    return None
