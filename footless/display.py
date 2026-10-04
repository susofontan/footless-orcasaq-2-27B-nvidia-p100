"""Terminal presentation: streaming text, muted thinking, closing rates."""

from __future__ import annotations

GRAY = "\033[90m"
RESET = "\033[39m"


def _tps(tokens: int, seconds: float) -> str:
    return f"{tokens / seconds:.1f}" if seconds > 0 else "-"


def _secs(seconds: float) -> str:
    """Seconds to one decimal, or a dash for a turn that produced nothing."""
    return f"{seconds:.1f}" if seconds > 0 else "-"


# The last turn: how much each phase produced, how fast, and the two numbers a
# reader actually feels — how long the answer took to START, and how long the
# whole thing took.
#
# The rate is the phase's own mean, `tokens / seconds`. It is NOT the mean of
# the per-step rates, and the difference matters: a model that verifies a draft
# hands some tokens out on steps that do no forward at all
# (`StepResult.forwarded`, CONTRACT.md), so the mean of the steps understates
# what the turn delivered. On one 57-token turn, 33 forwards in 4364 ms is
# 7.56 t/s a step and 56 tokens over the same 4364 ms is 12.83 t/s delivered;
# 7.56 x (1 + 23/33) = 12.83 exactly, the 23 being the free ones.
_HEADER = "  " + " " * 8 + f"{'tokens':>6}" + f"{'mean t/s':>11}"


def _stats(timing) -> str:
    """The last turn's two phases, plus when it started and how long it took."""
    rows = [_HEADER]
    for name, tokens, seconds in (
        ("prefill", timing.prefill_tokens, timing.prefill_seconds),
        ("decode", timing.decode_tokens, timing.decode_seconds),
    ):
        rows.append(f"  {name:<8}{tokens:>6}{_tps(tokens, seconds):>11}")
    rows.append(f"  {'first':<8}{_secs(timing.first_token_seconds):>6} s"
                f"    {'generated':<12}{_secs(timing.total_seconds):>7} s")
    return "\n".join(rows)


class StreamDisplay:
    """Renders one generation.

    The text streams as it is generated and nothing else ever shares its
    rows: when the response ends its per-phase rates print once, right
    below it, as a small table — the tokens of each phase, the phase's
    own mean rate, and how long the turn took to start and in total.

    EVERY token paints, and each paint flushes. The two go together and
    neither is optional. A terminal buffers lines, so a paint without a
    flush sits inside the program and the text arrives in one lump at the
    end; that was the bug a paint interval was added for, and it is fixed by
    the flush. What the interval then did was harm: it is a rate limit with
    no timer behind it, so a token arriving before the interval elapsed had
    nothing to wake it, and it waited for the NEXT token to be painted --
    which is why the output read as chunks rather than as a stream. There
    was no cost to removing it either: one write and one flush per token at
    the rate a decoder produces them is nothing, and a model that verifies a
    draft hands two tokens out microseconds apart, which is still one flush.

    Thinking markers are ROLES: the first id of thinking_markers opens a
    block, the second closes one. A template may leave a block open in the
    prompt, in which case generation starts inside one (open_at_start).
    """

    def __init__(self, out, decode, thinking_markers, open_at_start: bool = False) -> None:
        self.out = out
        self.decode = decode
        ids = list(thinking_markers)
        self._open_id = ids[0] if ids else None
        self._close_id = ids[1] if len(ids) > 1 else None
        self._ids: list[int] = []
        self._text = ""
        self._thinking = open_at_start
        self._span_open = False

    def token(self, token_id: int, timing) -> None:
        if token_id == self._open_id or token_id == self._close_id:
            self.flush(timing)  # pending text belongs to the current span
            if token_id == self._open_id:
                self._thinking = True
                self.out.write("\n" + GRAY)
                self._span_open = True
            else:
                self._thinking = False
                self.out.write((RESET if self._span_open else "") + "\n")
                self._span_open = False
        else:
            self._ids.append(token_id)
            self.flush(timing)   # every token paints; see the note below

    def flush(self, timing) -> None:
        """Paint the accumulated text."""
        piece = self.decode(self._ids)[len(self._text) :] if self._ids else ""
        if self._thinking and not self._span_open:
            self.out.write(GRAY)
            self._span_open = True
        if piece:
            self._text += piece
            self.out.write(piece)
        self.out.flush()  # a terminal buffers lines: a paint must say so

    def close(self, timing) -> None:
        self.flush(timing)
        if self._span_open:
            self.out.write(RESET)
            self._span_open = False
            self._thinking = False
        self.out.write(f"\n{GRAY}{_stats(timing)}{RESET}\n")
        self.out.flush()
