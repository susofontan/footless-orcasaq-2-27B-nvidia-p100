"""The contract as Python. Model runtimes are built against this module.

Normative prose lives in CONTRACT.md; this file is that surface, nothing more.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, TypeAlias

State: TypeAlias = Any  # opaque to the engine: it moves it and counts its bytes


class ContractError(Exception):
    """A contract violation. A hard error, never a fixup (rule 3)."""


class BudgetRefused(ContractError):
    """A claim the budget could not admit after eviction (rule 4)."""


@dataclass(frozen=True)
class Facts:
    """Load-time truths, compared against the manifest on open."""

    cache_bytes_per_token: int
    max_batch: int
    max_context: int
    capabilities: list[str]
    stop_tokens: list[int]
    thinking_markers: list = field(default_factory=list)  # [open, close] by id
    thinking_open_at_start: bool = False  # the thinking mode starts inside a block
    thinking_levels: list = field(default_factory=list)  # levels `chat` accepts
    # The package's own sampling defaults -- its generation config. A request
    # that pins nothing runs under these; a field it pins goes over them, and
    # only a package that declares none falls back to the spec's own defaults.
    default_sampling: "SamplingSpec | None" = None


@dataclass(frozen=True)
class SamplingSpec:
    """Produced by the engine's sampling policy, executed by the model.

    A `seed` identifies one per-request draw stream, advanced once per
    sampled token. `None` means unpinned: the engine gives the request a
    fresh seed, so two unpinned requests never share a stream.
    """

    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    min_p: float = 0.0
    logprobs: int = 0  # 0 = off
    seed: int | None = None  # None = unpinned: the engine seeds per request
    # Anti-repetition, with vLLM's semantics verbatim (its sampling_params.py
    # and layers/utils.py apply_penalties). All three act on the RAW logits,
    # before the temperature divide and the top-k/top-p cuts, in the order
    # repetition -> frequency -> presence, and they count for greedy too.
    # Neutral values leave every path untouched.
    repetition_penalty: float = 1.0  # HF scale on tokens seen in prompt+output:
    #   seen: logit > 0 ? logit / p : logit * p   (> 1 discourages repeats)
    presence_penalty: float = 0.0  # minus this, once, for tokens seen in OUTPUT
    frequency_penalty: float = 0.0  # minus this x times-seen, OUTPUT tokens only


@dataclass
class StepItem:
    state: State
    input_tokens: list[int]  # empty = decode; positions are never passed
    sampling: SamplingSpec | None = None  # None = prefill chunk, samples nothing


@dataclass
class StepResult:
    token: int | None
    finished: bool
    top_logprobs: list[tuple[int, float]] | None = None  # capability "logprobs"
    # Did this step run the model? A step that returns a token it had already
    # computed -- a speculative round's second token, handed out from what the
    # previous step verified -- did not forward, and the engine must not book it
    # as a decode speed: it takes microseconds, so the rate it implies is three
    # orders of magnitude off and it poisons the phase's min/mean/max. The
    # token and the wall time still count; the per-step sample does not.
    # Defaults True, so a runtime that never heard of this is unaffected.
    forwarded: bool = True
    # How many tokens this step actually ran through the model. It exists for
    # `truncate_state`: a truncate that cannot rewind for free queues the kept
    # ids as a replay, and the NEXT step's prefill silently runs the replay
    # beside what the request fed. Rates must count the work that ran, and the
    # engine cannot see a replay cross the boundary. 0 means "as many as were
    # fed", exact for any runtime whose truncate never queues a replay.
    forwarded_tokens: int = 0


class StepContext(Protocol):
    def cancelled(self) -> bool: ...


class Claim:
    """A claimed span of the memory budget, held by the runtime."""

    def __init__(self, nbytes: int) -> None:
        self.nbytes = nbytes


class Buffer:
    """Accounted host bytes."""

    def __init__(self, data: bytearray, claim: Claim) -> None:
        self.data = data
        self.claim = claim


class Accounting(Protocol):
    def claim(self, nbytes: int) -> Claim: ...  # raises BudgetRefused when refused

    def release(self, claim: Claim) -> None: ...

    def available(self) -> int: ...


class Buffers(Protocol):
    def host(self, nbytes: int) -> Buffer: ...

    def free(self, buffer: Buffer) -> None: ...


class Services(Protocol):
    """What the engine offers the model; optional, a runtime may use none."""

    log: Callable[[str, str], None]  # (level, message)
    accounting: Accounting
    buffers: Buffers


class Runtime(Protocol):
    """What a <backend>/runtime.py provides; create() -> Runtime."""

    # lifecycle — binds weights and tokenizer where they already are (rule 6)
    def open(self, model_dir: str, services: Services) -> Facts: ...

    def close(self) -> None: ...

    # cache state (rule 2)
    def new_state(self) -> State: ...

    def clone_state(self, state: State) -> State: ...

    def truncate_state(self, state: State, n_tokens: int) -> None: ...

    def state_nbytes(self, state: State) -> int: ...

    def export_state(self, state: State) -> bytes: ...

    def import_state(self, blob: bytes) -> State: ...

    def free_state(self, state: State) -> None: ...

    # text
    def encode(self, text: str) -> list[int]: ...

    def decode(self, tokens: list[int]) -> str: ...

    # conversation to prompt tokens, via the package's own template
    def chat(self, messages: list[dict], thinking: bool = True,
             level: str | None = None) -> list[int]: ...

    # one step over a batch of requests
    def step(self, batch: list[StepItem], ctx: StepContext) -> list[StepResult]: ...


RUNTIME_METHODS = (
    "open",
    "close",
    "new_state",
    "clone_state",
    "truncate_state",
    "state_nbytes",
    "export_state",
    "import_state",
    "free_state",
    "encode",
    "decode",
    "chat",
    "step",
)


# The one spelling of the `label_logprobs` capability. Behind it is the
# OPTIONAL verb of the same name (CONTRACT.md, Optional verbs), deliberately
# not in RUNTIME_METHODS:
#
#     label_logprobs(input_tokens: list[int], labels: list[str])
#         -> list[tuple[int, float]]
#
# Each label resolves to the single token it adds at the answer position
# after `input_tokens`' text, and comes back as (token_id, full-vocabulary
# next-token log-probability), in the labels' order. One prefill, no
# sampling, no penalties, no temperature. Raises ValueError naming the
# offending label when it is not exactly one token at the answer position,
# when labels collide on one token id, or when the prompt text does not
# round-trip through decode/encode.
LABEL_LOGPROBS = "label_logprobs"


# The one spelling of the `tool_calls` capability. Behind it, `chat` renders
# the request's function tools (`tools=`, OpenAI's shape) into the prompt, and
# the OPTIONAL verb of the same name reads the calls back out of the answer:
#
#     tool_calls(text: str, tools: list[dict], final: bool)
#         -> tuple[str, list[dict]]
#
# `text` is the answer so far (outside any thinking block); the result is its
# plain content and the calls it makes, each {"name": str, "arguments": str}
# (JSON text), in order. With `final` false the text may still grow: content
# stops short of anything that may yet open a call, and only calls already
# complete are listed — each result extends the previous one.
TOOL_CALLS = "tool_calls"
