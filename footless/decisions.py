"""Typed decision semantics: prompt wording, label alphabets, and the math.

A decision question is answered from the model's next-token scores at the
answer position, never from generated text. This module holds the policy the
typed decision routes share: how a question renders into one user message,
which labels stand for which options, which option names stay unambiguous,
and how label log-probabilities become probabilities, masses, and
confidences. It is pure policy: no model names, no hardware, no I/O.

The wording is versioned. `PROMPT_FORMAT_VERSION` names the wording and the
labels this module renders; any change to either needs a new version, so
callers pinning a version fail loudly instead of scoring different text.
"""

from __future__ import annotations

import json
import math
import string
import unicodedata
from dataclasses import dataclass
from typing import Any

PROMPT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class QuestionView:
    """A question as the renderer and scorer see it."""

    # 'choice', 'score', or 'yes_no'
    kind: str
    # None or blank when the question has no text of its own
    question: Any
    # Option names, level indices, or yes and no, in candidate order
    names: list[str]
    # Option descriptions, levels, or the yes and no descriptions
    details: list[Any]


def render_text(value) -> str:
    """The text a value takes in the prompt: itself, or compact JSON."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def render_question(text: str, view: QuestionView, labels: list[str]) -> str:
    """Prompt wording of PROMPT_FORMAT_VERSION.

    `text` is the request's input or state, rendered first; the question
    lines follow after a blank line. A question without its own text drops
    the question line and a yes or no question keeps its lead in.
    """
    question_text = "" if _blank(view.question) else render_text(view.question)
    if view.kind == "choice":
        lines = [f"Question: {question_text}"] if question_text else []
        for label, name, description in zip(labels, view.names, view.details):
            detail = render_text(description)
            lines.append(
                f"{label}: {name} - {detail}" if detail else f"{label}: {name}"
            )
        lines.append("Answer with the letter of one option only.")
    elif view.kind == "score":
        lines = [f"Question: {question_text}"] if question_text else []
        lines += [
            f"{label}: {render_text(level)}"
            for label, level in zip(labels, view.details)
        ]
        lines.append("Answer with the number of one level only.")
    else:
        lines = [
            f"Is the following true? {question_text}"
            if question_text
            else "Is the following true?"
        ]
        for label, description in zip(labels, view.details):
            detail = render_text(description)
            if detail:
                lines.append(f"{label}: {detail}")
        lines.append("Answer with yes or no only.")
    return "\n".join([text, "", *lines])


def default_labels(view: QuestionView) -> list[str]:
    """Single-token labels in candidate order: A to Z, level indices, or yes and no."""
    if view.kind == "choice":
        return list(string.ascii_uppercase[: len(view.names)])
    return list(view.names)


def pair_labels() -> list[str]:
    """Two-letter labels for options beyond 26, in one fixed order."""
    return [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]


def check_option_names(names) -> None:
    """Refuse option names that would make the rendered option lines ambiguous."""
    seen = set()
    for name in names:
        key = name.strip().casefold()
        if not key:
            raise ValueError("option names must be nonempty")
        # Each option is rendered as one prompt line.
        if any(unicodedata.category(c) in ("Cc", "Zl", "Zp") for c in name):
            raise ValueError(
                f"option name {name!r} must not contain control or line break "
                "characters"
            )
        if key in seen:
            raise ValueError(f"option name {name!r} repeats another option")
        seen.add(key)


def label_mass(logprobs: list[float]) -> float:
    """Full-vocabulary probability of all answer labels at the answer position."""
    return math.fsum(math.exp(logprob) for logprob in logprobs)


def probabilities(logprobs: list[float], temperature: float = 1.0) -> list[float]:
    """Softmax over the labels only, centered so small temperatures do not overflow.

    Temperature scales the option probabilities only, not `label_mass`.
    """
    maximum = max(logprobs)
    weights = [math.exp((score - maximum) / temperature) for score in logprobs]
    denominator = sum(weights)
    return [weight / denominator for weight in weights]


def choice_confidence(q: list[float]) -> float:
    """How far the top option stands above a uniform guess, from 0 to 1."""
    n = len(q)
    if n == 1:
        return 1.0
    return min(1.0, max(0.0, (n * max(q) - 1) / (n - 1)))


def score_confidence(q: list[float]) -> float:
    """One minus the spread around the top level relative to a uniform spread, floored at 0."""
    n = len(q)
    if n == 1:
        return 1.0
    top = q.index(max(q))
    spread = math.fsum(p * abs(i - top) for i, p in enumerate(q))
    uniform_spread = math.fsum(abs(i - (n - 1) / 2) for i in range(n)) / n
    return max(0.0, 1 - spread / uniform_spread)


def _blank(value) -> bool:
    """A question without text of its own renders without its question line."""
    return not (value.strip() if isinstance(value, str) else value)
