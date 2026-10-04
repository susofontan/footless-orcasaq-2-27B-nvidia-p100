"""The engine: policy and bookkeeping only. It knows nothing about models."""

from __future__ import annotations

import importlib.util
import random
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from pathlib import Path

from .manifest import Manifest, read as read_manifest, verify, verify_backends
from .sdk import (
    LABEL_LOGPROBS,
    TOOL_CALLS,
    RUNTIME_METHODS,
    BudgetRefused,
    Claim,
    ContractError,
    Facts,
    Runtime,
    SamplingSpec,
    State,
    StepItem,
    StepResult,
)

CHUNK = 4096  # prefill is fed in chunks; only the last one samples
DEFAULT_BUDGET = 1 << 34
DEFAULT_MAX_TOKENS = 16384

# The repetition guard: how long a repeating tail may run before it is a loop.
# A model that collapses repeats one span of tokens verbatim until the budget
# runs out -- 16k tokens of "Let me structure the response..." -- and nothing
# in the model's own stop rules ever fires, because the loop emits no stop
# token. The threshold counts the REPEATED part, the copies past the first:
# counting the whole run instead hid a short cycle inside a long answer (a
# few tokens of code repeated) behind a hundred copies of it, and counting
# nothing would fire on style -- three identical lines, a refrain.
LOOP_PERIOD_MAX = 256  # longer spans than this repeating are not covered
LOOP_MIN_REPEATS = 3  # copies of the span that end the output
# Tokens in the copies past the first, at least. 24 cut valid code: a map row
# `[1,1,1,...` at 33 tokens, a zeros array, a table's identical rows -- a
# truncated file the model then took for a broken write. A runaway never ends,
# so a high bar only costs it ~500 more tokens; the loop matrix's runaways
# still fire (_build/LOOPS_2026-10-01.md, 2026-10-02 replay).
LOOP_MIN_REPEATED_TOKENS = 512
LOOP_NOISY_PERIOD_MIN = 16  # below this the exact rule is the cheaper answer
# copies may differ by at most a third of their tokens and still be the same
# span: a runaway that re-names its variables between copies is still a
# runaway, and exact-only matching let "the same code block with different
# identifiers" run to the token cap all afternoon. Measured against the case
# that motivated this: 3 varying identifiers in a 20-token block (15%) -- an
# eighth (12.5%) missed it.
LOOP_MISMATCH_DEN = 3
# the fuzzy match is looser, so it demands one MORE copy than the exact rule:
# laxer evidence, stricter quorum. A real runaway hands over the 4th copy
# within one span anyway; a refrain of three does not.
LOOP_NOISY_MIN_REPEATS = 4


# What the repetition guard runs (`generate_tokens(loop_guard=...)`): nothing,
# the exact rule alone (the default), or all three passes. The exact rule cut
# every verbatim runaway of a 54-generation sampling matrix and nothing in 40k
# tokens of real prose and code; the fuzzy passes are the ones that cut healthy
# structured answers (models/OrcaSAQ-2-27B/.../_build/LOOPS_2026-10-01.md).
LOOP_GUARD_MODES = ("off", "exact", "full")


def loop_cut(tokens: list[int]) -> int | None:
    """Where a repeating tail says to keep the output, or None when it flows.

    Returns the length to keep -- the prefix plus ONE copy of the repeated
    span -- when the output ends with the same span of 1..LOOP_PERIOD_MAX
    tokens repeated LOOP_MIN_REPEATS times or more, those repeats past the
    first holding LOOP_MIN_REPEATED_TOKENS tokens or more. The repeats must
    reach the end of the output: a mid-text refrain is style, a runaway tail is
    the bug this exists for. Copies may be EXACT (the fast rule, 3 copies) or
    differ by up to a third of their tokens (the noisy rule, 4 copies,
    adjacent copy compared to adjacent copy so drift cannot accumulate past
    the bar).
    """
    return loop_cut_exact(tokens) or _noisy_loop_cut(tokens) or _rewrite_cut(tokens)


def loop_cut_exact(tokens: list[int]) -> int | None:
    """`loop_cut`'s first pass alone: the tail is the same span, verbatim,
    LOOP_MIN_REPEATS times or more."""
    n = len(tokens)
    for period in range(1, min(LOOP_PERIOD_MAX, n // LOOP_MIN_REPEATS) + 1):
        if tokens[n - 1] != tokens[n - 1 - period]:
            continue  # the tail cannot be copies of this period
        copies, at = 1, n - period
        while at - period >= 0 and tokens[at - period : at] == tokens[at : at + period]:
            copies += 1
            at -= period
        if copies >= LOOP_MIN_REPEATS \
                and (copies - 1) * period >= LOOP_MIN_REPEATED_TOKENS:
            return n - (copies - 1) * period
    return None


def _noisy_loop_cut(tokens: list[int]) -> int | None:
    """loop_cut's second pass: the same span with small variations between
    copies. Adjacent copies are compared to adjacent copies (not all against
    the last), so a drifting loop is caught at its beginning rather than only
    at the moment the drift happens to fit the bar. The mismatch bar aborts
    early, so ordinary prose costs a few compares per period and dies."""
    n = len(tokens)
    for period in range(LOOP_NOISY_PERIOD_MIN,
                        min(LOOP_PERIOD_MAX, n // LOOP_MIN_REPEATS) + 1):
        allow = max(period // LOOP_MISMATCH_DEN, 1)
        copies, at = 1, n - period
        while at - period >= 0:
            bad = 0
            for i in range(period):
                if tokens[at - period + i] != tokens[at + i]:
                    bad += 1
                    if bad > allow:
                        break
            if bad > allow:
                break
            copies += 1
            at -= period
        if copies >= LOOP_NOISY_MIN_REPEATS \
                and (copies - 1) * period >= LOOP_MIN_REPEATED_TOKENS:
            return n - (copies - 1) * period
    return None


# The rewrite rule's reach: the loop the two rules above cannot see. A model
# that REWRITES a long block (same implementation, renamed variables) produces
# periods of hundreds of tokens with only TWO copies before the budget dies
# mid-expression -- measured on the case that motivated this: key code lines
# appearing twice at 1200 tokens (~600-token period), no guard message, output
# cut off inside a variable name.
LOOP_REWRITE_PERIOD_MIN = 128
LOOP_REWRITE_PERIOD_MAX = 2048


def _rewrite_cut(tokens: list[int]) -> int | None:
    """loop_cut's third pass: the last two long blocks are the same block.

    Tail-anchored like the others: two near-duplicate spans of 128..2048 tokens
    closing the output mean the second one is a rewrite, and the output keeps
    the first. Runs every 8 tokens and pre-samples 16 positions per period --
    the per-token path must not pay for 2000 full span compares, and a runaway
    hands over another copy every period, so the delay is one span's worth of
    noise. The bar is the noisy rule's looser cousin: a quarter of the tokens
    may differ (renamed identifiers), and the pre-sample is only a cheap
    rejection of obviously-different spans."""
    n = len(tokens)
    if n % 8:
        return None
    for period in range(LOOP_REWRITE_PERIOD_MIN,
                       min(LOOP_REWRITE_PERIOD_MAX, n // 2) + 1):
        allow = period // 4
        bad = 0
        for s in range(16):  # cheap pre-sample: reject obvious non-matches
            i = s * period // 16
            if tokens[n - period + i] != tokens[n - 2 * period + i]:
                bad += 1
        if bad > 6:
            continue
        bad = 0
        for i in range(period):
            if tokens[n - period + i] != tokens[n - 2 * period + i]:
                bad += 1
                if bad > allow:
                    break
        if bad <= allow:
            return n - period
    return None


def request_spec(sampling: SamplingSpec | None) -> SamplingSpec:
    """The request's sampling spec — one per-request draw stream.

    A `seed` identifies one per-request draw stream, advanced once per
    sampled token, so the spec is copied per request even when the caller
    pins every field: a shared spec object must not carry one request's
    stream into the next. An unpinned seed (None, the default) is drawn
    here from fresh entropy — the engine owns randomness, the model
    executes the spec where its logits live.
    """
    base = sampling if sampling is not None else SamplingSpec()
    seed = base.seed
    if seed is None:
        seed = random.SystemRandom().randrange(1 << 62)
    return replace(base, seed=seed)


class Ledger:
    """Byte accounting: the budget is the engine's to enforce."""

    def __init__(self, budget: int, evict) -> None:
        self.budget = budget
        self.used = 0
        self._evict = evict

    def claim(self, nbytes: int) -> Claim:
        if nbytes < 0:
            raise ContractError(f"negative claim: {nbytes}")
        while self.used + nbytes > self.budget:
            if not self._evict():
                raise BudgetRefused(
                    f"budget {self.budget}B cannot admit {nbytes}B "
                    f"({self.used}B used, nothing left to evict)"
                )
        self.used += nbytes
        return Claim(nbytes)

    def release(self, claim: Claim) -> None:
        self.used -= claim.nbytes

    def available(self) -> int:
        return self.budget - self.used

    def set_budget(self, budget: int) -> None:
        """Re-base the budget on the device's truth. Optional on the
        Accounting protocol: a runtime whose device is smaller than the
        declared budget clamps it here, so states and cached clones are
        admitted against what the card actually has rather than meeting
        cuMemAlloc mid-generation. Never GROWS past the opener's declaration."""
        self.budget = min(self.budget, max(budget, 0))


class HostBuffers:
    def __init__(self, ledger: Ledger) -> None:
        self._ledger = ledger

    def host(self, nbytes: int):
        from .sdk import Buffer

        return Buffer(bytearray(nbytes), self._ledger.claim(nbytes))

    def free(self, buffer) -> None:
        self._ledger.release(buffer.claim)


class StepContext:
    """The step context service, checked at step boundaries."""

    def cancelled(self) -> bool:
        return False


class EngineServices:
    def __init__(self, ledger: Ledger) -> None:
        self.accounting = ledger
        self.buffers = HostBuffers(ledger)
        self.log = self._log

    @staticmethod
    def _log(level: str, message: str) -> None:
        print(f"[{level}] {message}", file=sys.stderr)


class PrefixCache:
    """Prefix reuse: which cached state to keep, key it, evict it."""

    def __init__(self, runtime: Runtime, available=None) -> None:
        self._runtime = runtime
        self._entries: "OrderedDict[tuple[int, ...], State]" = OrderedDict()
        # the ledger's free bytes (set by Engine.open): a copy that would not
        # fit beside the entry hands the entry itself over instead
        self.available = available

    def lookup(self, tokens: list[int]) -> tuple[State, int]:
        key = tuple(tokens)
        best: tuple[int, ...] | None = None
        best_len = 0
        for entry in self._entries:
            common = 0
            for a, b in zip(entry, key):
                if a != b:
                    break
                common += 1
            # a partial match truncates the clone, and a truncate may queue the
            # kept ids for re-forwarding in the next step (CONTRACT.md,
            # `truncate_state`) -- invisible here, which is why the rates book
            # `StepResult.forwarded_tokens` instead of what the request fed.
            if common > best_len:
                best, best_len = entry, common
        if __import__("os").environ.get("FOOTLESS_DEBUG"):
            import sys as _s
            print(f"[dbg cach] entries={len(self._entries)} best_len={best_len} "
                  f"entry_len={len(best) if best else 0} key_len={len(key)}",
                  file=_s.stderr)
        if best is None:
            return self._runtime.new_state(), 0
        # out of the cache while it is copied: making room for the copy may
        # evict entries, and never this one
        entry = self._entries.pop(best)
        fits = self.available is None or \
            self._runtime.state_nbytes(entry) <= self.available()
        state = None
        if fits:
            try:
                state = self._runtime.clone_state(entry)
            except BudgetRefused:
                state = None
        if state is None:
            # no room for a copy (a long conversation's state is most of the
            # card): the request takes the entry itself and continues it --
            # a fresh state would re-forward the whole prefix
            self._runtime.truncate_state(entry, best_len)
            return entry, best_len
        self._entries[best] = entry
        self._runtime.truncate_state(state, best_len)
        return state, best_len

    def store(self, tokens: list[int], state: State) -> None:
        key = tuple(tokens)
        old = self._entries.pop(key, None)
        if old is not None:
            self._runtime.free_state(old)
        self._entries[key] = state

    def evict_lru(self) -> bool:
        if not self._entries:
            return False
        _, state = self._entries.popitem(last=False)
        self._runtime.free_state(state)
        return True

    def free_all(self) -> None:
        while self.evict_lru():
            pass


@dataclass
class Rates:
    """One phase's per-step speeds, in tokens per second.

    A prefill sample is one chunk, a decode sample is one step — for the
    models here, one token. A step is a sample only if it RAN: one that
    produced nothing did not run, and neither did one that returned a token it
    had already computed (`StepResult.forwarded`). Both are real tokens and
    both are real wall time — they are in the phase's token and second totals,
    which is what the `speed t/s` column reports — but neither is a sample of
    how fast a forward runs, and a microsecond sample drags the mean and the
    max three orders of magnitude off.
    """

    values: list[float] = field(default_factory=list)

    def add(self, tokens: int, seconds: float) -> None:
        if tokens > 0 and seconds > 0:
            self.values.append(tokens / seconds)

    @property
    def minimum(self) -> float | None:
        return min(self.values) if self.values else None

    @property
    def mean(self) -> float | None:
        return sum(self.values) / len(self.values) if self.values else None

    @property
    def maximum(self) -> float | None:
        return max(self.values) if self.values else None


@dataclass
class Timing:
    """Per-phase progress, filled live while a generation runs."""

    prefill_tokens: int = 0
    prefill_seconds: float = 0.0
    decode_tokens: int = 0
    decode_seconds: float = 0.0
    # How long the answer took to START, and how long the whole request took.
    # `first_token_seconds` is set when the first token joins the output, so it
    # is a prefix of `total_seconds` and covers the prefill plus one step: the
    # two numbers a reader actually feels, and the reason a chat that answers
    # in a sentence can be slower than one that answers in a page.
    first_token_seconds: float = 0.0
    total_seconds: float = 0.0
    prefill_rates: Rates = field(default_factory=Rates)
    decode_rates: Rates = field(default_factory=Rates)


@dataclass
class GenerationResult:
    text: str
    tokens: list[int]
    prompt_tokens_fed: int
    finished: bool
    timing: Timing = field(default_factory=Timing)
    # Did the repetition guard cut the output? `tokens` then holds the prefix
    # plus one copy of the repeated span, while `timing` counts everything the
    # model actually generated before the cut.
    looped: bool = False
    # What the model ABSORBED this request, cut or not: the tail the state
    # holds and the cache is keyed on. `tokens` above is what the output shows
    # (the guard's trimmed view); a caller that extends the conversation as a
    # token stream -- the chat loop, so the prefix cache stays an exact hit --
    # must extend it with THESE, or every turn after a cut silently re-runs the
    # whole history (the key keeps the trimmed tail, the stream would not).
    absorbed_tokens: list[int] = field(default_factory=list)


class Engine:
    def __init__(
        self,
        model: Manifest,
        runtime: Runtime,
        facts: Facts,
        services: EngineServices,
        cache: PrefixCache,
    ) -> None:
        self.model = model
        self.runtime = runtime
        self.facts = facts
        self.services = services
        self.cache = cache

    @classmethod
    def open(
        cls,
        model_dir: str | Path,
        backend: str | None = None,
        budget: int = DEFAULT_BUDGET,
    ) -> "Engine":
        model_dir = Path(model_dir)
        footless_dir = model_dir / "footless"
        if not footless_dir.is_dir():
            raise ContractError(f"not a model package: {model_dir} (no footless/ folder)")
        model = read_manifest(footless_dir)
        verify_backends(model, footless_dir)
        if backend is None:
            if len(model.backends) != 1:
                raise ContractError(f"pick a backend, manifest has: {model.backends}")
            backend = model.backends[0]
        if backend not in model.backends:
            raise ContractError(f"backend {backend!r} not in manifest: {model.backends}")
        runtime = load_runtime(footless_dir / backend / "runtime.py")
        cache = PrefixCache(runtime)
        services = EngineServices(Ledger(budget, cache.evict_lru))
        cache.available = services.accounting.available
        facts = runtime.open(str(model_dir), services)
        verify(model, facts)
        return cls(model, runtime, facts, services, cache)

    def generate(
        self, prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS, on_token=None,
        sampling: SamplingSpec | None = None, loop_guard: bool | str = "exact",
    ) -> GenerationResult:
        return self.generate_tokens(
            self.runtime.encode(prompt), max_tokens, on_token, sampling,
            loop_guard=loop_guard,
        )

    def speculative(self, enabled: bool = True) -> None:
        """Turn speculative decoding on, if this package offers it.

        The engine does not know how a package speculates, or whether it
        speculates at all — it asks, and the package's own `capabilities`
        answer (rule 4). A package that does not declare `speculative` is
        refused here, with the reason, rather than quietly generating
        without it.

        `set_speculative` is an OPTIONAL runtime verb (CONTRACT.md, Runtime):
        it is deliberately not in `RUNTIME_METHODS`, so a package that does not
        implement one is not broken by this engine, and a package that declares
        the capability without the verb is a contradiction we report rather
        than a silent no-op.
        """
        if not enabled:
            return
        if "speculative" not in self.facts.capabilities:
            raise ContractError(
                "speculative decoding was asked for, but this package declares "
                f"no speculative capability (it declares: "
                f"{self.facts.capabilities or 'nothing'})"
            )
        verb = getattr(self.runtime, "set_speculative", None)
        if not callable(verb):
            raise ContractError(
                "this package declares the speculative capability but its "
                "runtime exposes no set_speculative verb"
            )
        verb(True)

    def label_logprobs(self, input_tokens: list[int],
                       labels: list[str]) -> list[tuple[int, float]]:
        """Score `labels` at the answer position after `input_tokens`, if this
        package offers it.

        The engine does not know how a package resolves or scores a label, or
        whether it scores labels at all — it asks, and the package's own
        `capabilities` answer (rule 4). A package that does not declare
        `label_logprobs` is refused here, with the reason, rather than quietly
        scoring without it.

        `label_logprobs` is an OPTIONAL runtime verb (CONTRACT.md, Optional
        verbs): it is deliberately not in `RUNTIME_METHODS`, so a package that
        does not implement one is not broken by this engine, and a package that
        declares the capability without the verb is a contradiction we report
        rather than a silent no-op.

        A `ValueError` from the verb — a label that is not exactly one token
        at the answer position, labels colliding on one token id, prompt text
        that does not round-trip through decode/encode — propagates unchanged:
        the caller's input at fault, which the serving layer maps to a client
        error.
        """
        if LABEL_LOGPROBS not in self.facts.capabilities:
            raise ContractError(
                "label scoring was asked for, but this package declares no "
                "label_logprobs capability (it declares: "
                f"{self.facts.capabilities or 'nothing'})"
            )
        verb = getattr(self.runtime, "label_logprobs", None)
        if not callable(verb):
            raise ContractError(
                "this package declares the label_logprobs capability but its "
                "runtime exposes no label_logprobs verb"
            )
        return verb(input_tokens, labels)

    def label_logprobs_batch(self, items: list, ctx: StepContext | None = None,
                             shared: int = 0) -> list[list[tuple[int, float]]] | None:
        """`label_logprobs` for several (input_tokens, labels) pairs, in order.

        Through the package's optional `label_logprobs_batch` when it has one
        (it may share the work the prompts have in common), else one
        `label_logprobs` call per pair -- the same gate either way, and a
        cancelled `ctx` stops between pairs (None is returned). A `ValueError`
        carries `.item`, the index of the pair at fault.

        `shared` names a head of that many tokens the caller expects later
        requests to repeat (a decision's template and input text), clamped to
        what every pair really shares short of its last token. With the batch
        verb the head goes through the prefix cache: looked up, the rest of it
        forwarded by `step`, the pairs scored from that state, and the state
        kept as the head's entry -- a repeated head is not forwarded again.
        """
        if LABEL_LOGPROBS not in self.facts.capabilities:
            self.label_logprobs([], [])        # raises the capability's ContractError
        batch = getattr(self.runtime, "label_logprobs_batch", None)
        if callable(batch):
            if shared > 0 and items:
                first = items[0][0]
                shared = min(shared, *(len(t) - 1 for t, _ in items))
                for t, _ in items[1:]:
                    n = 0
                    while n < shared and t[n] == first[n]:
                        n += 1
                    shared = n
            if shared > 0 and items:
                return self._label_from_cache(batch, items, list(first[:shared]), ctx)
            return batch(items)
        out = []
        for k, (input_tokens, labels) in enumerate(items):
            if ctx is not None and ctx.cancelled():
                return None
            try:
                out.append(self.label_logprobs(input_tokens, labels))
            except ValueError as exc:
                exc.item = k
                raise
        return out

    def _label_from_cache(self, batch, items: list, head: list[int],
                          ctx: StepContext | None):
        """`label_logprobs_batch` from the cached state of `head`."""
        ctx = ctx if ctx is not None else StepContext()
        state, matched = self.cache.lookup(head)
        try:
            rest = head[matched:]
            for i in range(0, len(rest), CHUNK):
                self.runtime.step([StepItem(state, rest[i:i + CHUNK], None)], ctx)
                if ctx.cancelled():
                    self.runtime.free_state(state)
                    return None
            if not rest:
                # settles what a partial match queued (a replay), forwards nothing otherwise
                self.runtime.step([StepItem(state, [], None)], ctx)
            result = batch(items, state=state)
        except ValueError:
            self.cache.store(head, state)  # the caller's labels at fault; the head is good
            raise
        except BaseException:
            self.runtime.free_state(state)
            raise
        # the verb leaves the state as it found it: the head's entry
        self.cache.store(head, state)
        return result

    def chat_tokens(self, messages: list[dict], thinking: bool = True,
                    level: str | None = None, tools: list[dict] | None = None) -> list[str]:
        """The conversation as prompt tokens, via the package's own template.

        `level` is one of the package's `thinking_levels`; None leaves the
        package its own default. `tools` are function tools in OpenAI's
        shape, for a package that declares `tool_calls`. Each is passed only
        when set, so a runtime written before the parameter keeps serving
        requests that do not use it.
        """
        kwargs = {}
        if level is not None:
            kwargs["level"] = level
        if tools:
            self._tool_gate()
            kwargs["tools"] = tools
        return self.runtime.chat(messages, thinking, **kwargs)

    def tool_calls(self, text: str, tools: list[dict],
                   final: bool) -> tuple[str, list[dict]]:
        """The answer `text` split into its content and the calls it makes.

        The OPTIONAL verb behind `tool_calls` (sdk.TOOL_CALLS): how a package
        writes a call is its template's business, so it reads them back too.
        """
        self._tool_gate()
        return self.runtime.tool_calls(text, tools, final)

    def _tool_gate(self) -> None:
        if TOOL_CALLS not in self.facts.capabilities:
            raise ContractError(
                "function tools were asked for, but this package declares no "
                f"tool_calls capability (it declares: {self.facts.capabilities or 'nothing'})")
        if not callable(getattr(self.runtime, "tool_calls", None)):
            raise ContractError(
                "this package declares the tool_calls capability but its "
                "runtime exposes no tool_calls verb")

    def decode(self, tokens: list[int]) -> str:
        return self.runtime.decode(tokens)

    def generate_tokens(
        self, tokens: list[int], max_tokens: int = DEFAULT_MAX_TOKENS,
        on_token=None, sampling: SamplingSpec | None = None,
        on_step=None, ctx: StepContext | None = None, loop_guard: bool | str = "exact",
    ) -> GenerationResult:
        """One request: prefill `tokens`, then decode up to `max_tokens` tokens.

        `on_token` is called with every token that joins the output. `on_step`
        is called after every step with that step's `StepResult` and the timing
        so far — including steps that sample nothing and stop tokens the
        engine's own rules keep out of the output — and returning `True` from it
        ends the generation: the caller's stop policy on top of the engine's.
        `ctx` is the step context the model checks at step boundaries; a
        cancelled one ends the request at the next step.

        `loop_guard` picks the engine's repetition rule (LOOP_GUARD_MODES):
        output that ends in one span repeated past the threshold ends the
        request and is cut back to a single copy, reported as
        `GenerationResult.looped`. The default, "exact", acts on VERBATIM
        copies only (`loop_cut_exact`); "full" adds the fuzzy passes, which
        read tokens alone and cannot tell a runaway from a table of
        iterations -- a false positive there costs the rest of a healthy
        answer, so they run only when asked. "off" lets a collapse run to
        `max_tokens`. `True` / `False` mean "full" / "off".
        """
        mode = ("full" if loop_guard else "off") if isinstance(loop_guard, bool) \
            else loop_guard
        if mode not in LOOP_GUARD_MODES:
            raise ValueError(f"loop_guard {loop_guard!r}: one of {LOOP_GUARD_MODES}")
        rule = {"exact": loop_cut_exact, "full": loop_cut}.get(mode)
        runtime = self.runtime
        if len(tokens) > self.facts.max_context:
            raise ContractError(
                f"prompt is {len(tokens)} tokens, max_context is {self.facts.max_context}"
            )
        # a prompt whose cache alone cannot fit the whole budget is refused now,
        # not after the minutes of prefill it takes to reach the limit (the
        # state's fixed part is the model's and not counted: this is the floor)
        need = self.facts.cache_bytes_per_token * len(tokens)
        budget = getattr(self.services.accounting, "budget", None)
        if budget is not None and need > budget:
            raise BudgetRefused(
                f"prompt is {len(tokens)} tokens: its cache needs {need} B, more than "
                f"the whole budget of {budget} B")
        state, matched = self.cache.lookup(tokens)
        fed = tokens[matched:]
        ctx = ctx if ctx is not None else StepContext()
        # what a request does not pin runs under the package's own defaults
        # (Facts.default_sampling -- its generation config), and only where a
        # package declares none under the spec's own (CONTRACT.md)
        spec = request_spec(sampling if sampling is not None
                            else self.facts.default_sampling)
        timing = Timing()
        t_start = time.perf_counter()   # the request, not any one step
        generated: list[int] = []
        finished = False
        cut: int | None = None  # where the repetition guard trimmed the output

        def absorb(step_result):
            nonlocal finished, cut
            before = len(generated)
            stop = self._absorb(step_result, generated)
            finished = step_result.finished or stop
            if len(generated) > before:
                if cut is None and rule is not None:
                    keep = rule(generated)
                    if keep is not None:
                        # the loop's tail is trimmed away at the end; what has
                        # already streamed cannot be recalled, and `timing`
                        # still counts everything the model really produced
                        cut, finished = keep, True
                if not timing.first_token_seconds:
                    # the first token to join the output, measured from the
                    # start of the request and nothing else
                    timing.first_token_seconds = time.perf_counter() - t_start
                if on_token is not None:
                    on_token(generated[-1], timing)
            if on_step is not None and on_step(step_result, timing):
                finished = True

        consumed = 0
        try:
            chunks = [fed[i : i + CHUNK] for i in range(0, len(fed), CHUNK)]
            for index, chunk in enumerate(chunks):
                last = index == len(chunks) - 1
                # a step with no sampling spec samples nothing: max_tokens = 0 is
                # exactly zero output, and chunks before the last one never sample
                sample = spec if last and max_tokens > 0 else None
                # only the step is timed: notify and bookkeeping wait outside, so
                # a slow front end cannot be read as a slow model
                started = time.perf_counter()
                result = runtime.step([StepItem(state, chunk, sample)], ctx)[0]
                elapsed = time.perf_counter() - started
                absorb(result)
                # book the tokens that ran, not the tokens the request fed: a
                # truncate can queue the kept prefix for re-forwarding inside this
                # step, and that work must not be read as the model slowing down
                ran = result.forwarded_tokens or len(chunk)
                timing.prefill_rates.add(ran, elapsed)
                timing.prefill_seconds += elapsed
                timing.prefill_tokens += ran
                consumed += len(chunk)
                if finished or (max_tokens > 0 and len(generated) >= max_tokens):
                    break

            while not finished and len(generated) < max_tokens:
                before = len(generated)
                started = time.perf_counter()
                result = runtime.step([StepItem(state, [], spec)], ctx)[0]
                elapsed = time.perf_counter() - started
                absorb(result)
                produced = len(generated) - before
                if produced:  # a step that sampled no output is not a decode speed
                    timing.decode_tokens += produced
                    timing.decode_seconds += elapsed
                    # ...and neither is a step that returned a token it had already
                    # computed. A model that verifies a draft hands the second token
                    # out on the next step with no forward at all; that step takes
                    # microseconds, so as a rate sample it reads in the tens of
                    # thousands and takes the phase's mean and max with it. The
                    # token and the time still count -- the aggregate is right --
                    # but it is not a sample of how fast a forward runs.
                    if result.forwarded:
                        timing.decode_rates.add(produced, elapsed)

        except BaseException:
            # a refused allocation, a cancelled or failed step: the state is
            # this request's alone, and nothing else would ever free it
            runtime.free_state(state)
            raise
        timing.total_seconds = time.perf_counter() - t_start
        result_tokens = generated if cut is None else generated[:cut]
        text = runtime.decode(result_tokens)
        # the cache entry is keyed on what the state actually saw — including
        # tokens the guard later trimmed, which the state absorbed all the
        # same. An early end may also have left part of the prompt unfed.
        if __import__("os").environ.get("FOOTLESS_DEBUG"):
            import sys as _s
            print(f"[dbg eng] matched={matched} consumed={consumed} "
                  f"req={len(tokens)} gen={len(generated)} result_eq_gen="
                  f"{result_tokens == generated} cut={cut}", file=_s.stderr)
        # the request is done with its state: the cache keeps it as is (a
        # copy would need the room of a second one, which a long
        # conversation's state does not leave)
        self.cache.store(tokens[: matched + consumed] + generated, state)
        return GenerationResult(text, result_tokens, consumed, finished, timing,
                                looped=cut is not None, absorbed_tokens=generated)

    def close(self) -> None:
        self.cache.free_all()
        self.runtime.close()

    def _absorb(self, result: StepResult, generated: list[int]) -> bool:
        if result.token is None:
            return False
        if result.token in self.facts.stop_tokens:
            return True
        generated.append(result.token)
        return False


def load_runtime(path: Path) -> Runtime:
    module_name = f"footless_runtime_{abs(hash(path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ContractError(f"cannot load runtime: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    create = getattr(module, "create", None)
    if not callable(create):
        raise ContractError(f"{path} does not define create()")
    runtime = create()
    missing = [name for name in RUNTIME_METHODS if not callable(getattr(runtime, name, None))]
    if missing:
        raise ContractError(f"{path} runtime is missing methods: {missing}")
    return runtime
