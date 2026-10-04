"""One generation's output on its way to a client: spans, stop rules, deltas.

The engine hands the front end one sampled token at a time (`on_step`). This
assembles them into what a client sees: the thinking block and the answer as
separate texts — the markers that delimit a block are generated like any token
but never rendered as text (CONTRACT.md) — the request's stop strings applied
to the detokenized stream, and text handed out only once no later token can
take it back. A stop string may still end the text mid-token; what has already
been streamed cannot be recalled, so what is handed out early is exactly the
part no stop string can claim.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LogprobRow:
    """One sampled token's logprobs, as the step reported them."""

    token_id: int
    text: str
    start: int  # character offsets in the rendered text
    end: int
    logprob: float | None  # of the sampled token itself; None when unreported
    top: list[tuple[int, str, float]]  # (token_id, text, logprob)


class _Span:
    """One contiguous run of tokens of one kind."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.ids: list[int] = []
        self.text = ""


class Output:
    """One request's generated tokens, split and trimmed for its response.

    `thinking_markers` are the ids that open and close a thinking block (the
    first opens, the second closes); `open_at_start` means the rendered prompt
    left a block open, so generation begins inside one. `stop` strings match
    the detokenized stream — thinking and answer text alike — and trim it at
    the match, or just past it with `include_stop_str_in_output`.
    """

    def __init__(
        self,
        decode,
        thinking_markers: list[int] = (),
        open_at_start: bool = False,
        stop=(),
        include_stop_str_in_output: bool = False,
        with_logprobs: bool = False,
    ) -> None:
        ids = list(thinking_markers)
        self.decode = decode
        self._open_id = ids[0] if ids else None
        self._close_id = ids[1] if len(ids) > 1 else None
        self._stop = [s for s in stop if s]
        self._include_stop = include_stop_str_in_output
        self._with_logprobs = with_logprobs
        self._spans: list[_Span] = [_Span("reasoning" if open_at_start else "content")]
        self._pending: list[int] = []  # tokens that end inside a character
        self._full = ""
        self._emitted = 0
        self._limit: int | None = None  # where a stop string trimmed the text
        self._rows: list[LogprobRow] = []
        self._rows_done = 0
        self.tokens: list[int] = []
        self.reasoning_tokens = 0
        self.stop_reason: str | int | None = None

    def push(self, token: int, top_logprobs=None) -> bool:
        """One sampled token joins the output; True when a stop string fired."""
        self.tokens.append(token)
        if token == self._open_id or token == self._close_id:
            self._settle()
            self._spans.append(
                _Span("reasoning" if token == self._open_id else "content")
            )
            return False
        if self._spans[-1].kind == "reasoning":
            self.reasoning_tokens += 1
        span = self._spans[-1]
        span.ids.append(token)
        # a character may span tokens (an emoji, an accent the vocabulary
        # splits): its first bytes alone decode to U+FFFD, so the text waits
        # for the rest -- a UTF-8 character is at most 4 bytes, so 4 tokens
        self._pending.append(token)
        text = self.decode(self._pending)
        if text.endswith("\ufffd") and len(self._pending) < 4:
            text = ""
        else:
            self._pending = []
        span.text += text
        start = len(self._full)
        self._full += text
        if self._with_logprobs:
            top = []
            for top_id, top_logprob in top_logprobs or ():
                top.append((top_id, self.decode([top_id]), top_logprob))
            own = next((lp for tid, _, lp in top if tid == token), None)
            self._rows.append(
                LogprobRow(token, self.decode([token]), start, len(self._full), own, top)
            )
        match = self._match()
        if match is not None:
            index, length, string = match
            self._limit = index + (length if self._include_stop else 0)
            self.stop_reason = string
            return True
        return False

    def stop(self, reason: str | int | None = None) -> None:
        """Generation ends here: a stop token, or the model's own stop rules."""
        if reason is not None:
            self.stop_reason = reason

    def take(self) -> tuple[list[tuple[str, str]], list[LogprobRow]]:
        """The deltas since the last take that no later token can take back.

        Text comes out as `(kind, text)` pieces in stream order — a thinking
        piece and an answer piece can leave in the same call — together with
        the logprob rows of the tokens whose text is now complete.
        """
        return self._flush(final=False)

    def close(self) -> tuple[list[tuple[str, str]], list[LogprobRow]]:
        """The rest of the output: generation is over."""
        self._settle()
        return self._flush(final=True)

    @property
    def text(self) -> str:
        """The rendered stream — thinking and answer alike — as one text."""
        return self._full[: self._kept()]

    @property
    def reasoning(self) -> str:
        """The thinking blocks' text, without their markers."""
        return self._kind_text("reasoning")

    @property
    def content(self) -> str:
        """The answer text, outside every thinking block."""
        return self._kind_text("content")

    # -- internals ---------------------------------------------------------

    def _settle(self) -> None:
        """Bytes still waiting for the rest of a character join the text as
        they are: nothing more will come to complete them."""
        if self._pending:
            text = self.decode(self._pending)
            self._pending = []
            self._spans[-1].text += text
            self._full += text

    def _kept(self) -> int:
        return self._limit if self._limit is not None else len(self._full)

    def _kind_text(self, kind: str) -> str:
        out, pos, kept = "", 0, self._kept()
        for span in self._spans:
            start, end = pos, pos + len(span.text)
            pos = end
            if span.kind == kind:
                out += span.text[: max(0, min(kept - start, end - start))]
        return out

    def _match(self) -> tuple[int, int, str] | None:
        """The earliest stop string in the stream, as (index, length, string)."""
        best = None
        for string in self._stop:
            index = self._full.find(string)
            if index >= 0 and (best is None or index < best[0]):
                best = (index, len(string), string)
        return best

    def _risk(self) -> int:
        """Where the text a future stop string could still claim begins.

        Everything from here on is held back: the tail of the stream may be
        the head of a stop string that only a later token completes, and a
        streamed character cannot be recalled when it turns out to belong to
        one.
        """
        risk = len(self._full)
        for string in self._stop:
            for head in range(min(len(string) - 1, len(self._full)), 0, -1):
                if self._full.endswith(string[:head]):
                    risk = min(risk, len(self._full) - head)
                    break
        return risk

    def _flush(self, final: bool) -> tuple[list[tuple[str, str]], list[LogprobRow]]:
        limit = self._kept()
        if not final:
            limit = min(limit, self._risk())
        pieces: list[tuple[str, str]] = []
        pos = 0
        for span in self._spans:
            start, end = pos, pos + len(span.text)
            pos = end
            a = max(start, self._emitted) - start
            b = max(min(end, limit), start) - start
            if b > a:
                pieces.append((span.kind, span.text[a:b]))
        self._emitted = max(self._emitted, limit)
        return pieces, self._take_rows(limit, final)

    def _take_rows(self, limit: int, final: bool) -> list[LogprobRow]:
        rows = []
        while self._rows_done < len(self._rows):
            row = self._rows[self._rows_done]
            if row.end > limit and not (final and row.start < limit):
                break
            rows.append(row)
            self._rows_done += 1
        return rows


class ToolCalls:
    """The answer text with the tool calls it makes carved out, as it streams.

    `parse(text, final) -> (content, calls)` is the package's own reading of
    its call format (the `tool_calls` verb, the request's tools bound). Content
    leaves as soon as the package says no call can claim it, and a call leaves
    whole once the package reads it complete. As in vLLM, content that is all
    whitespace waits for what follows it: it is dropped when a call does, kept
    otherwise. `limit` keeps only the first calls (`parallel_tool_calls`).
    """

    def __init__(self, parse, limit: int | None = None) -> None:
        self._parse = parse
        self._limit = limit
        self._text = ""
        self._sent = 0  # content characters handed out (or dropped)
        self.content = ""  # the content as last read
        self.calls: list[dict] = []  # the calls as last read

    def push(self, text: str) -> tuple[str, list[tuple[int, dict]]]:
        """More answer text: the content and the (index, call)s now safe."""
        self._text += text
        return self._read(final=False)

    def close(self) -> tuple[str, list[tuple[int, dict]]]:
        """The answer is over: whatever is left."""
        return self._read(final=True)

    def _read(self, final: bool) -> tuple[str, list[tuple[int, dict]]]:
        content, calls = self._parse(self._text, final)
        if self._limit is not None:
            calls = calls[: self._limit]
        known = len(self.calls)
        self.content, self.calls = content, list(calls)
        fresh = content[self._sent:]
        if not content.strip():  # nothing but whitespace so far
            if calls:
                self._sent = len(content)  # dropped before a call
                fresh = ""
            elif not final:
                fresh = ""  # held
        self._sent += len(fresh)
        return fresh, list(enumerate(self.calls))[known:]
