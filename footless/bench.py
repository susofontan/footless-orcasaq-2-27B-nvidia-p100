"""Speed benchmark over the contract: prefill and decode rates per frontier.

The shape mirrors ds4's ds4-bench: a frontier sweep over one real prompt, a
greedy decode per frontier, one CSV row per frontier. Each frontier measures
only the prompt interval added since the previous one; the first decode cycle
forwards the last prompt token (the contract samples as part of a forward), so
it carries a full cycle like ds4's argmax+eval first step. Between frontiers
the state is truncated back, which is ds4's snapshot restore.
"""

from __future__ import annotations

import sys
import time

from .engine import CHUNK
from .sdk import ContractError, SamplingSpec, StepItem

GREEDY = SamplingSpec(temperature=0.0)
HEADER = (
    "ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,"
    "gen_first_ms,gen_steady_tokens,gen_steady_tps,kvcache_bytes"
)


class _Ctx:
    def cancelled(self) -> bool:
        return False


def frontiers(start: int, maximum: int, incr: int, mul: float) -> list[int]:
    out = []
    f = start
    while f <= maximum:
        out.append(f)
        f = int(f * mul) if mul > 1.0 else f + incr
    return out


def _tps(tokens: int, seconds: float) -> float:
    return tokens / seconds if seconds > 0 and tokens > 0 else 0.0


def run(engine, tokens: list[int], sweep: list[int], gen_tokens: int,
        show_output: bool, out) -> None:
    runtime = engine.runtime
    ctx = _Ctx()
    state = runtime.new_state()
    print(
        f"bench: model={engine.model.name} cache_bytes_per_token="
        f"{engine.facts.cache_bytes_per_token} max_context={engine.facts.max_context}",
        file=sys.stderr,
    )
    print(HEADER, file=out)
    try:
        previous = 0
        for f in sweep:
            if f > len(tokens):
                raise ContractError(
                    f"frontier {f} exceeds the prompt ({len(tokens)} tokens)"
                )
            # prefill window: pure forwards of the new prompt interval; with a
            # decode to run, the last prompt token waits for the first cycle
            feed = tokens[previous : f - 1] if gen_tokens else tokens[previous:f]
            started = time.perf_counter()
            for i in range(0, len(feed), CHUNK):
                chunk = feed[i : i + CHUNK]
                if chunk:
                    runtime.step([StepItem(state, chunk, None)], ctx)
            prefill_seconds = time.perf_counter() - started
            snap = runtime.clone_state(state)

            first_seconds = 0.0
            steady_seconds = 0.0
            generated: list[int] = []
            for i in range(gen_tokens):
                item = [tokens[f - 1]] if i == 0 else []
                started = time.perf_counter()
                result = runtime.step([StepItem(state, item, GREEDY)], ctx)[0]
                elapsed = time.perf_counter() - started
                if result.token is not None:
                    generated.append(result.token)
                if i == 0:
                    first_seconds = elapsed
                else:
                    steady_seconds += elapsed

            kv_bytes = runtime.state_nbytes(state)
            # rewind by snapshot: truncate_state would zero the recurrence and
            # make the next row re-forward the whole kept prefix, a re-prefill
            # this sweep does not mean to measure. A clone plus the one prompt
            # token the first cycle forwarded lands the frontier exactly as a
            # fresh feed of tokens[:f] would.
            runtime.free_state(state)
            state = snap
            if gen_tokens:
                runtime.step([StepItem(state, [tokens[f - 1]], None)], ctx)
            if show_output:
                print(runtime.decode(generated), file=sys.stderr)

            steady = max(gen_tokens - 1, 0)
            row = (
                f"{f},{len(feed)},{_tps(len(feed), prefill_seconds):.2f},"
                f"{gen_tokens},{_tps(gen_tokens, first_seconds + steady_seconds):.2f},"
                f"{first_seconds * 1000:.3f},{steady},"
                f"{_tps(steady, steady_seconds):.2f},{kv_bytes}"
            )
            print(row, file=out, flush=True)
            previous = f
    finally:
        runtime.free_state(state)
