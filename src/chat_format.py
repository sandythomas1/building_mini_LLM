"""Plain-text chat template shared by data preparation and sampling.

A conversation is rendered as alternating role-prefixed lines:

    Customer: My order hasn't arrived.
    Agent: I'm sorry about that. Could you share your order number?

Training and inference must use the exact same template, so both import it from here.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

ROLE_LABELS = {"customer": "Customer", "agent": "Agent"}
AGENT_PREFIX = f"{ROLE_LABELS['agent']}:"
# Start of any following turn, from either role.
NEXT_TURN = re.compile("\n(?:" + "|".join(map(re.escape, ROLE_LABELS.values())) + "):")

Turn = tuple[str, str]  # (role, text)


def format_turn(role: str, text: str) -> str:
    try:
        label = ROLE_LABELS[role.strip().lower()]
    except KeyError:
        raise ValueError(f"Unknown role {role!r}; expected one of {sorted(ROLE_LABELS)}") from None
    return f"{label}: {' '.join(text.split())}\n"


def format_conversation(turns: Iterable[Turn]) -> str:
    """Render turns as a single training document. Empty turns are dropped."""
    return "".join(format_turn(role, text) for role, text in turns if text and text.strip())


def build_prompt(history: Iterable[Turn], customer_message: str) -> str:
    """Prompt that asks the model to write the next agent reply."""
    return format_conversation([*history, ("customer", customer_message)]) + AGENT_PREFIX
