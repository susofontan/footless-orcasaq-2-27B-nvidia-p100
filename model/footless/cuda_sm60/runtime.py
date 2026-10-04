"""OrcaSAQ-2-27B on the Tesla P100 (sm_60): the footless runtime.

A Qwen3.5 dense hybrid — 64 layers, 48 gated-delta-net (linear attention) and 16
full attention — quantized to exl3 trellis weights. Text only: the checkpoint has
no vision tower. The checkpoint also carries an MTP drafter (39 tensors, 213
MB); it is NOT loaded until the engine asks for speculative decoding with
`set_speculative` (CONTRACT.md, optional verbs), so a run that never speculates
never pays for it.

What is where:

  bridge.py        the CUDA driver bridge (ctypes -> libcuda), nvcc -> cubin at open
  kernels.cu       the architecture kernels (norms, rope, attention, conv, GDN, argmax)
  kernels_exl3.cu  the fused trellis-decode GEMV and the two Hadamard rotations
  tokenizer.py     BPE and the chat template, transcribed from the package's own files
  _build/          the harnesses: census, exl3 decode oracle, nvcc build, benchmarks

The engine's contract is ../../../../CONTRACT.md. Two clauses shape this file:

  * Cached state is opaque to the engine. `_State` is ours; the engine moves it
    and counts its bytes.
  * Positions are the model's business. `pos0` is derived from the state alone
    (see `_step_one`), never passed in.

Weights are uploaded once and never moved: one device buffer per safetensors
shard holding the file's bytes verbatim, every tensor addressed as
`shard_base + data_offset`. The trellis stays packed — the GEMV decodes it in
registers — because 12.06 GB of trellis fits a 16 GB card but its fp16 expansion
(54 GB) does not.

One algebraic shortcut is taken, and it is exact:

  * The reference scales the GDN's L2-normalised q by `128**-0.5` before reading
    the state out. `gdn_scan` applies that same scale to its output instead —
    the read-out is linear in q, so it is the same arithmetic with q kept at full
    fp16 precision. (Scaling the gated norm's *gamma* by it is NOT the same:
    `rsqrt(mean(o^2)+eps)` is not homogeneous when eps != 0, so a gamma fold
    scales the whole branch instead of the read-out. That mistake cost this port
    a long detour; see _build/NOTES.md.)
  * `l2norm_scaled` is applied to the whole conv'd q|k|v row at once (the kernel
    indexes a token's slab as `(t*rows + r)*pitch`, so `rows=80, pitch=128`
    walks the 10240-wide row as head-sized rows). Only the q and k rows are read
    back out; the v rows it also normalises are ignored, and v is taken raw from
    the conv output.
"""

from __future__ import annotations

import heapq
import importlib.util
import json
import math
import os
import random
import struct
import sys
import time
import tomllib
from pathlib import Path

try:                      # the sampler's fast path; the fallback is exact too
    import numpy as _np
except ImportError:       # a runtime without numpy still samples, just slower
    _np = None

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# The model, as constants. Every one is checked against the checkpoint in
# `_check_geometry`; a contradiction is a hard error, never a fixup.
# ---------------------------------------------------------------------------

PREFIX = "model.language_model."
N_LAYERS = 64
HIDDEN = 5120
INTER = 17408
VOCAB = 248320  # embedding rows; the tokenizer's real vocabulary is 248077
N_HEADS = 24  # full-attention q heads
N_KV = 4
HEAD_DIM = 256
ROPE_DIM = 64  # partial rotary: the first 64 of 256 dims rotate
ROPE_THETA = 1e7
ATTN_SCALE = HEAD_DIM ** -0.5

GDN_LAYERS = 48
GDN_K_HEADS = 16
GDN_V_HEADS = 48
GDN_K = 128
GDN_V = 128
GDN_GROUP = GDN_V_HEADS // GDN_K_HEADS  # 3 v-heads per k-head
CONV_WIDTH = 4
QKV_ROWS = 10240
# (channels, row offset) of the three depthwise conv segments inside the qkv row
CONV_SEGS = ((2048, 0), (2048, 2048), (6144, 4096))
# The GDN's L2 scale for q. ARCHITECTURE.md documents the reference as scaling
# q by this after its L2 norm; measurement says otherwise — folding it into the
# GDN output norm's gamma (which is the algebraically equivalent form) put the
# perplexity at 144 000 against 867 without it, on the Italian fixture. See
# _build/NOTES.md. Kept named because the question is worth re-asking against a
# reference implementation, not re-deriving.
Q_SCALE = GDN_K ** -0.5
# --- two values an experiment once bent through environment overrides -------
# `ORCA_GAMMA_SCALE` folded the q scale into the gated norm's gamma,
# `ORCA_GDN_EPS` grew the gated norm's eps. Both overrides are GONE: each one
# moves the model away from the checkpoint -- the eps "improves" the Italian
# fixture's perplexity by suppressing near-zero gated rows instead of
# normalising them (27 at 1e-4 against 709 at the config's 1e-6, on a fixture
# whose numbers do not even rank against the WikiText-2 gate) -- and an
# exported variable changes every later run with no trace in the code. The
# measured curves and the open question live in _build/NOTES.md; the shipped
# values are the checkpoint's own.
_GAMMA_SCALE = 1.0    # scales linear_attn.norm.weight when != 1.0 (never shipped)
GDN_NORM_EPS = 1e-6

NORM_EPS = 1e-6
L2_EPS = 1e-6
# The full-attention KV cache format, the manifest's `[runtime] kv_cache`, set
# by `open` (_kv_format). "int8" (KV8): per position, layer and k|v, 4 x 256
# int8 and an fp16 scale per 32-dim block (kernels.cu's kv_store_q8) -- 1088 B
# against fp16's 2048. Measured against fp16 (_build/OPTIMIZE log, "KV cache
# in int8"): PPL equal to 4 digits at 2k / 8k / 16k tokens. Module globals: one
# runtime a process, and an A/B harness may flip them between states.
KV_FORMATS = {"int8": True, "fp16": False}
KV8 = True


def _kv_row(kv8: bool | None = None) -> int:
    """Bytes a position takes in one layer's K (or V) buffer, at KV8 unless named."""
    return 4 * 256 + 4 * (256 // 32) * 2 if (KV8 if kv8 is None else kv8) else 4 * 256 * 2


CACHE_BYTES_PER_TOKEN = 16 * 2 * _kv_row()  # 16 full-attn layers, k and v: 34816 (fp16: 65536)
MAX_BATCH = 1
# what one sequence fits in int8 KV (speculation on: ~77k, see log); in fp16 the
# budget refuses past ~58k (~50k speculating) before this does
MAX_CONTEXT = 102400


def _dev_bytes(nbytes: int) -> int:
    """What one cuMemAlloc of `nbytes` takes from the card. Measured on the
    P100 (driver 2026-10): above 1 MiB the driver rounds to 2 MiB, below it to
    256 KiB -- a 4.25 MiB KV layer takes 6 MiB, the 10 KB hidden row 256 KiB.
    Counting the requested size left ~50 MiB a state off the ledger, and a
    cache of six states hit CUDA_ERROR_OUT_OF_MEMORY with the ledger at 97%."""
    gran = (2 << 20) if nbytes > (1 << 20) else (256 << 10)
    return -(-nbytes // gran) * gran


def _kv_format(model_dir: Path) -> bool:
    """KV8 from the manifest's `[runtime]` table, checked against the
    manifest's own cache_bytes_per_token (the engine compares it with Facts
    too, but cannot name the setting that disagrees)."""
    raw = tomllib.loads((model_dir / "footless" / "manifest").read_text())
    table = raw.get("runtime", {})
    unknown = sorted(set(table) - {"kv_cache"})
    if unknown:
        raise _contract_error(f"manifest [runtime] has unknown keys: {unknown}")
    fmt = table.get("kv_cache", "int8")
    if fmt not in KV_FORMATS:
        raise _contract_error(f"manifest [runtime] kv_cache = {fmt!r}: one of "
                              f"{', '.join(map(repr, KV_FORMATS))}")
    per_token = 16 * 2 * _kv_row(KV_FORMATS[fmt])
    if raw.get("cache_bytes_per_token") != per_token:
        raise _contract_error(
            f"manifest [runtime] kv_cache = {fmt!r} takes cache_bytes_per_token = "
            f"{per_token}; the manifest says {raw.get('cache_bytes_per_token')}")
    return KV_FORMATS[fmt]

STOP_TOKENS = [248044, 248046]  # endoftext, im_end
THINKING_MARKERS = [248068, 248069]

# GDN recurrent state: 48 layers * 48 v-heads * 128 * 128 fp32 (144 MiB).
GDN_S_BYTES = GDN_LAYERS * GDN_V_HEADS * GDN_V * GDN_K * 4
# GDN conv state: 48 layers * 10240 channels * 3 frames fp16, doubled for the
# ping-pong the conv kernel requires (st_in and st_out must be distinct).
GDN_CONV_BYTES = GDN_LAYERS * 3 * QKV_ROWS * 2

# The engine chunks prefill at 4096; we sub-chunk again to bound the activation
# scratch (a 4096-token chunk would want ~400 MB of it). Larger is faster.
#
# The scratch is ~439 KB a token (sum the T-proportional arenas below), so this
# constant is also the arena size and a bigger one is a device-memory decision
# as much as a speed one. ORCA_MAX_CHUNK overrides it for the sweep in
# _build/bench_chunk_pair.py. 1024 since the prefill GEMM (xp_gemm128): its
# 128-token tiles leave a 5120-wide output at 160 blocks on 112 slots at 512
# (1.4 waves) and 320 at 1024 (2.9); measured +5-7% on a 4096-token prompt,
# 2048 no better and 430 MB less state budget (_build/OPTIMIZE_2026-10-01.md).
MAX_CHUNK = int(__import__("os").environ.get("ORCA_MAX_CHUNK", "1024"))
AMAX_BLOCKS = 64  # the argmax grid; keys/idxs are this long
# The device sampler's candidate extraction (kernels_exl3.cu): CAND_BLOCKS
# blocks histogram the row into CAND_BINS buckets of its fp16 sortable keys,
# and the collect pass returns at most CAND_CAP (value, index) pairs -- an
# overflow falls back to the host path, which is exact by construction.
CAND_BLOCKS = 32
CAND_BINS = 512            # == kernels_exl3.cu's CAND_BINS / CAND_SHIFT
CAND_SHIFT = 7             # bucket = key >> CAND_SHIFT (512 << 7 == 65536)
CAND_CAP = 16384
# decode attention's KV split: up to ATT_SPLIT blocks per (head, token), one per
# ATT_SPLIT_ROWS cached rows. 8 / 512 left a 512-token context on 24 blocks
# walking every row one warp-reduction at a time (109 us a layer).
ATT_SPLIT = 32
ATT_SPLIT_ROWS = 64
# the GQA form of the decode attention (kernels.cu's attention_split_gqa): one
# block per kv head instead of per q head, so it wants more splits per row count
ATT_GQA = int(__import__("os").environ.get("ORCA_ATT_GQA", "1"))
ATT_GQA_ROWS = int(__import__("os").environ.get("ORCA_ATT_GQA_ROWS", "32"))
# the tiled decode attention over the int8 cache (kernels.cu's attention_dec_q8,
# a block a token): a token's kv split is DEC_TMIN tiles of 64 rows at least, DEC_SMAX
# splits at most (4 kv heads x 28 = two blocks an SM on 56 SMs)
ATT_DEC = int(__import__("os").environ.get("ORCA_ATT_DEC", "1"))
DEC_TMIN = 1
DEC_SMAX = 28
# the GEMM-shaped (FlashAttention-2) prefill attention (kernels.cu's attention_prefill_fa)
APF_FA = int(__import__("os").environ.get("ORCA_APF_FA", "1"))
# prefill attention as kernels.cu's attention_prefill_fh (GQA rows, kv split,
# P V in fp16); 0 = attention_prefill_fa
APF_FH = int(__import__("os").environ.get("ORCA_APF_FH", "1"))
# prefill GDN scan with the state's rows over 4 blocks a head (gdn_scan_rows); 0 = gdn_scan
GDN_ROWS = int(__import__("os").environ.get("ORCA_GDN_ROWS", "1"))
# in_proj_a/b as kernels_exl3.cu's tiled gemm_ab_f32 from this T up (else the GEMV)
GAB_MIN_T = int(__import__("os").environ.get("ORCA_GAB_MIN_T", "384"))  # measured crossover ~300
# decode's GDN scan with the q/k L2 norm and beta/decay folded in (gdn_scan_f)
GDN_FUSED = int(__import__("os").environ.get("ORCA_GDN_FUSED", "1"))
# the fused decode / verify scan on gdn_scan_rows' rows (kernels.cu's gdn_scan_rows_f)
GDN_ROWS_F = int(__import__("os").environ.get("ORCA_GDN_ROWS_F", "1"))
# the verify's per-row conv calls as one launch (kernels.cu's conv1d_causal_rows)
CONV_ROWS = int(__import__("os").environ.get("ORCA_CONV_ROWS", "1"))
# a decode step's conv inside gdn_scan_rows_f (gdn_conv1)
GDN_CONV_F = int(__import__("os").environ.get("ORCA_GDN_CONV_F", "1"))
# the lm_head through the fused entry (pre-rotation + GEMV + post-rotation)
HEAD_FUSED = int(__import__("os").environ.get("ORCA_HEAD_FUSED", "1"))
# the drafter's one-layer KV cache in the package's int8 format (attention_dec_q8
# and attention_prefill_fh_q8 read it); 0 keeps it fp16 whatever the format
DRAFT_KV8 = int(__import__("os").environ.get("ORCA_DRAFT_KV8", "1"))
# the no-top-k sampler on arrays (`_sample_np`); 0: the list path
SAMPLE_NP = int(__import__("os").environ.get("ORCA_SAMPLE_NP", "1"))
# a decode attention block's small launches folded (attn_prep_q8, the gated merge)
ATT_PREP = int(__import__("os").environ.get("ORCA_ATT_PREP", "1"))
MIN_VOCAB_ID = 248077  # the tokenizer ends here; rows above are embedding padding

# The exl3 `mul1` codebook multiplier. Every module in this checkpoint carries
# the same one; `_verify_mul1` refuses the load if any disagrees.
MUL1 = 0x83DCD12D

# ---------------------------------------------------------------------------
# MTP speculative decoding. OFF by default; ORCA_MTP=1 turns it on.
#
# The drafter is the checkpoint's own `mtp.*` head (39 tensors, 213 MB, in the
# last shard): a 4-bit `fc [5120 <- 10240]`, ONE full_attention decoder layer,
# `mtp.norm`, and the target's SHARED embedding table and SHARED 6-bit lm_head.
# The normative reference is _reference/vllm/.../qwen3_5_mtp.py and
# _reference/ARCHITECTURE.md's MTP section; the draft is
#
#     cat([norm(embed(x_{p+1})), norm(h_p)])   # EMBEDDING HALF FIRST
#         -> fc -> the layer -> mtp.norm -> the shared head
#
# i.e. the drafter's row at position p is built from the TARGET's hidden at p and
# the token at p+1, stores its KV at p, attends over its own 0..p, and predicts
# the token at p+2. That is vLLM's convention and it is OFF BY ONE from the
# obvious reading -- `_reference/.../llm_base_proposer.py:set_inputs_first_pass`
# SHIFTS the target's token ids and leaves the positions and hidden states
# unshifted. Getting it wrong does not break anything; it makes the drafter
# predict the token it was just handed, and the acceptance rate collapses to the
# lag-by-one pattern. See `_mtp_draft`.
#
# The one thing Round 11 got wrong, and the whole of this round: at T = 2 the
# verify batch took `_full_attn`'s attention_prefill branch, whose grid is
# (N_HEADS, ceil(T/8)) = 24 blocks, where a decode step takes attention_split,
# whose grid is (N_HEADS, T, ATT_SPLIT) = 192. attention_split is ALREADY
# row-general -- it computes `seqlen = pos0 + blockIdx.y + 1` and reads
# `q + blockIdx.y * n_q_heads * q_pitch` -- so a T = 2 verify launched with pos0
# at the first row's position and grid (N_HEADS, 2, ATT_SPLIT) is EXACTLY the
# decode path, row for row, and `kv_store` has already written both rows. The
# fix is therefore this one constant and the pacc/pm arenas it sizes: no kernel
# edit at all. See MTP_SPLIT_MAX.
# ---------------------------------------------------------------------------
MTP_SPEC = int(__import__("os").environ.get("ORCA_MTP", "0"))   # env override; see set_speculative
# Drafts per round: 1 verifies [p, d] (T = 2); 2 also chains a second draft
# off the drafter's own hidden (vLLM's construction) and verifies [p, d1, d2]
# in one T = 3 forward (m = 3 group), accepting 0, 1 or 2 drafts.
MTP_K = int(__import__("os").environ.get("ORCA_MTP_K", "2"))
# The largest T the SPLIT attention path takes. 1 is the shipped behaviour, byte
# for byte; 4 is what lets a T = 2 verify run the DECODE attention (Round 11's
# fix) without a kernel change. Above it the prefill kernel's tiled queries are
# the right shape anyway, and its partial arena would be T times bigger.
# The arena sizing depth: how many attention/logit rows the buffers can hold.
# It is ALWAYS 4, even with speculation off, because the switch that asks for
# speculation arrives after `open` and the buffers are allocated there. What the
# switch actually changes is how many of those rows are USED -- see
# `set_speculative` and `self.split_max`.
MTP_SPLIT_MAX = int(__import__("os").environ.get("MTP_SPLIT_MAX", "4"))
# A verify's GEMVs take the T = 1 k-split (`_gemv_shape` at the T = 1 cap) and
# the T = 2 token group, so the partials are the same COUNT and in the same
# order -- the verify is bit-identical to two sequential T = 1 forwards -- while
# the group keeps the trellis read to ONE pass. The split cap falls as 1/T, so
# taking T = 2's own value would HALVE the blocks for the same bytes; pinning it
# to T = 1 is both the exact choice and the faster one. 0 takes T = 2's own.
MTP_SHAPE_T1 = 1
# The warm-up chunk for the drafter's own KV cache, and the width of the two
# halves of the drafter's `fc` input.
MTP_WARM_CHUNK = 128
# Logit rows the head's two buffers hold. One in the shipped build (and
# with MTP off), MTP_SPLIT_MAX when the verify wants a sample at every
# position of its batch.
LOGIT_ROWS = MTP_SPLIT_MAX


class _Exl3Ab(__import__("ctypes").Structure):
    """kernels_exl3.cu's Exl3Ab: the fp16 a / b rows an exl3_gemv_w4a1fpn launch adds."""
    _fields_ = [(n, __import__("ctypes").c_void_p) for n in ("wa", "wb", "ya", "yb")] + \
               [("out", __import__("ctypes").c_int)]


class _Exl3Fm(__import__("ctypes").Structure):
    """kernels_exl3.cu's Exl3Fm: one module of an exl3_gemv_w4a1fpn launch."""
    _fields_ = [(n, __import__("ctypes").c_void_p) for n in
                ("suh", "trellis", "svh", "y", "part", "cnt")] + \
               [(n, __import__("ctypes").c_int) for n in ("out", "bits_x2", "S")]


def _load(name: str, path: Path):
    """Import a module that sits beside this file. The engine loads runtime.py by
    path with no package context, so neighbours come in the same way."""
    spec = importlib.util.spec_from_file_location(f"footless_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_bridge = _load("orca_bridge", HERE / "bridge.py")
_tokenizer = _load("orca_tokenizer", HERE / "tokenizer.py")

_ST_DTYPES = {"F16": ("<f2", 2), "F32": ("<f4", 4), "I8": ("i1", 1),
              "I16": ("<i2", 2), "I32": ("<i4", 4), "BF16": ("<u2", 2)}


def _contract_error(msg: str):
    from footless.sdk import ContractError

    return ContractError(msg)


def bf16_to_float(u16: int) -> float:
    """bfloat16 -> float, by the top-half-of-fp32 definition."""
    return struct.unpack("<f", struct.pack("<I", u16 << 16))[0]


# ---------------------------------------------------------------------------
# The fp16 sortable-key bijection the device sampler works in (kernels_exl3.cu's
# exl3_key16 and its inverse). Ascending key == ascending VALUE, so "above a
# threshold" is one integer comparison on the device, and a histogram of the
# keys buckets the row by value without any floating-point compare. -0.0 and
# +0.0 share a key: they are equal values and the sampler's order is
# (value desc, index desc), so a bucket edge landing between them must not be
# able to split them.
# ---------------------------------------------------------------------------

def _key_value(key: int) -> float:
    """The fp16 value a sortable key names.

    The key space is not the whole 16-bit range: +0.0 is 0x8000 and the most
    negative half (0xFBFF) is 0x400, so every key a real value produces is
    >= 0x400, and anything below that is a padding key -- there is no such
    value, and it sorts below every fp16 value, which is what a bucket edge
    falling there should do.
    """
    if key < 0x400:
        return -65504.0
    if key < 0x8000:
        u = 0xFFFF - key
    else:
        u = key - 0x8000
    if u > 0xFBFF:
        return -65504.0                    # -inf / NaN in the value space
    return struct.unpack("<e", struct.pack("<H", u))[0]


def _half_above(x: float) -> float:
    """The smallest fp16 strictly greater than x -- so `v >= it` == `v > x`.

    For an fp16 value v the host path's `v > floor` is therefore a `>=` against
    this, exactly, whatever the floor's rounding.
    """
    if x != x:                             # NaN floor: nothing is above it
        return float("inf")
    if x >= 65504.0:
        return float("inf")
    if x < -65504.0:
        return -65504.0                    # every fp16 value is above it
    u = struct.unpack("<H", struct.pack("<e", x))[0]   # x to the nearest half
    h = struct.unpack("<e", struct.pack("<H", u))[0]
    if h > x:
        return h
    # x rounded DOWN to u, so the answer is the next fp16 in VALUE order. One
    # key up is exactly that, for either sign (the key is value-ascending), and
    # -0.0 shares +0.0's key -- the device folds them the same way.
    if u == 0x8000:
        u = 0x0000
    k = (0xFFFF - u) if (u & 0x8000) else (u | 0x8000)
    return _key_value(k + 1)


# ---------------------------------------------------------------------------
# The label scorer's host normalizer, in the sampler's dual-path discipline
# (`_candidates`: a numpy path and the same computation in struct/math below
# it -- _build/check_sampler.py checks the pair against each other on real
# rows, and tests/test_label_logprobs.py does the same here). Both forms are
# the reference's `compute_row_log_normalizer` shape: per row (max,
# logsumexp - max) with fp32 accumulation, so a log-prob is
# (logit - max) - log_sum and rows with a common offset stay exact. The
# normalizer sees the WHOLE row (248 077 halves -- the padded tail past
# MIN_VOCAB_ID is embedding padding and is never read); only the labels are
# gathered.
# ---------------------------------------------------------------------------

def _row_logprobs_np(raw: bytes, ids: list[int]) -> list[float]:
    """The label log-probs of one fp16 row, numpy form (fp32 accumulation)."""
    flat = _np.frombuffer(raw, dtype="<f2").astype(_np.float32)
    mx = float(flat.max())
    if mx == mx and abs(mx) != float("inf"):
        log_sum = float(_np.log(_np.exp(flat - flat.max()).sum(dtype=_np.float32)))
    else:
        log_sum = 0.0        # the reference's `row_max.isinf() -> 0` guard
    return [(float(flat[i]) - mx) - log_sum for i in ids]


def _row_logprobs_py(raw: bytes, ids: list[int]) -> list[float]:
    """The same computation with no numpy: struct unpack to doubles, and
    `math.fsum` for the exp sum (exact, so this form is the reference's
    fp32 accumulation without its rounding -- the two agree to ~1e-6 and
    the tests bound that)."""
    values = struct.unpack(f"<{MIN_VOCAB_ID}e", raw)
    mx = max(values)
    if mx == mx and abs(mx) != float("inf"):
        log_sum = math.log(math.fsum(math.exp(v - mx) for v in values))
    else:
        log_sum = 0.0
    return [(values[i] - mx) - log_sum for i in ids]


def _row_logprobs(raw: bytes, ids: list[int]) -> list[float]:
    return _row_logprobs_np(raw, ids) if _np is not None else _row_logprobs_py(raw, ids)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class _State:
    """One sequence's cache. Opaque to the engine.

    `tokens` are the positions the KV and the recurrences already cover;
    `pending` is a sampled token not yet forwarded; `replay` is a prefix the
    recurrent state no longer matches (see `truncate_state`).
    """

    __slots__ = ("tokens", "fed", "pending", "replay", "s", "conv", "phase",
                 "kk", "kv", "lcap", "cap", "claim", "services", "logits", "has_logits",
                 "mtp")

    def __init__(self) -> None:
        self.tokens: list[int] = []
        self.fed = 0
        self.pending: int | None = None
        self.replay: list[int] | None = None
        self.s = None  # GDN recurrent state, fp32 [48][48,128,128]
        self.conv = None  # two conv-state arenas, ping-ponged per step
        self.phase = 0
        # KV cache: one buffer per full-attention layer (k and v), fp16
        # [lcap[l], 4, 256] -- per layer so a growth needs one layer's room
        # beside the cache, not a second cache. `cap` is what every layer holds.
        self.kk = None
        self.kv = None
        self.lcap = None
        self.cap = 0
        self.claim = None
        self.services = None
        # The logits row of the last forwarded position. The engine hands a
        # decode step with no input tokens when the whole prompt was a prefix
        # hit (`engine.py` finds `fed` empty, so no prefill step runs at all),
        # and the token to generate is the one predicted at the last cached
        # position — which is this row. Re-forwarding that token instead would
        # advance the recurrences a second time, so the row is kept.
        self.logits = None
        self.has_logits = False
        # The drafter's own bookkeeping. Always present and always empty until
        # `set_speculative(True)`; see `_MtpState`.
        self.mtp = _MtpState()

    def logical(self) -> int:
        return len(self.tokens) + (1 if self.pending is not None else 0)


class _MtpState:
    """One sequence's MTP drafter state. An optimisation: a stale one must cost
    SPEED and never correctness, so every path that changes what the drafter has
    seen invalidates it (`mtp_len` back to 0) rather than trying to repair it.

      out      verified tokens the drafter produced that the engine has not been
               handed out yet. The engine's decode loop feeds `input_tokens=[]`
               every step and gets one token back, so a round that verified two
               tokens hands the second out on the next step with no device work.
      draft    the drafter's token for the position the NEXT round will verify
               -- i.e. the token at (last forwarded position) + 2. Produced at
               the END of the step that forwarded the position before it, which
               is the only moment its two inputs (the target's h_{p-1} and the
               token at p) both exist.
      draft2   (MTP_K = 2) the chained second draft: the token after `draft`,
               from a drafter row fed `draft` and the drafter's own hidden.
      rb_tag   the verify a level-2 round came from (see `_mtp_unforward`).
      spec     the request's sampling spec, for drafting under its penalties.
      mtp_len  the drafter's KV cache is valid for positions 0..mtp_len-1.
      hprev    the target's FINAL NORMED HIDDEN at position len(tokens)-1, the
               row `pre_fc_norm_hidden` needs. Saved per step because the
               drafter runs at the position AFTER the one just forwarded.
      htail    the same, carried across prefill sub-chunks so the drafter's row
               at a chunk's first position has the previous chunk's last hidden.
      dk, dv   the drafter's own one-layer KV cache (4096 B a position).
    """

    __slots__ = ("out", "draft", "draft2", "rb_tag", "spec", "mtp_len", "hprev",
                 "htail", "dk", "dv", "cap")

    def __init__(self) -> None:
        self.out: list[int] = []
        self.draft: int | None = None
        self.draft2: int | None = None
        self.rb_tag = -1
        self.spec = None
        self.mtp_len = 0
        self.hprev = None
        self.htail = None
        self.dk = None
        self.dv = None
        self.cap = 0


# ---------------------------------------------------------------------------
# The runtime
# ---------------------------------------------------------------------------

def default_sampling(model_dir) -> "SamplingSpec":
    """The checkpoint's own sampling defaults, from `generation_config.json`.

    The engine runs a request that pins nothing under these (CONTRACT.md).
    A field the file does not name keeps the SDK's own default; an explicit
    `do_sample: false` is greedy here, which is `temperature = 0` (the
    sampler's greedy path). `logprobs` and `seed` are never taken from the
    file: one is a request's reporting choice, the other the engine's to draw.
    """
    from dataclasses import replace

    from footless.sdk import SamplingSpec

    casts = {"temperature": float, "top_k": int, "top_p": float, "min_p": float,
             "repetition_penalty": float, "presence_penalty": float,
             "frequency_penalty": float}
    path = Path(model_dir) / "generation_config.json"
    gen = json.loads(path.read_text()) if path.is_file() else {}
    pins = {name: casts[name](gen[name]) for name in casts if name in gen}
    if gen.get("do_sample") is False:
        pins["temperature"] = 0.0
    return replace(SamplingSpec(), **pins)


class CudaRuntime:
    def open(self, model_dir: str, services):
        from footless.sdk import Facts

        t0 = time.time()
        self.dir = Path(model_dir)
        self.services = services
        self.log = services.log
        self.align_off: dict[str, int] = {}
        self._aux_bufs: list = []  # every device allocation we own
        # speculative-decode bookkeeping, read by the _build/ harnesses
        self.shard_ptr: dict[str, tuple[int, dict, int]] = {}
        self._headers: dict[str, dict] = {}
        self.weight_bytes = 0
        self.target_bytes = 0
        self.mtp_rounds = self.mtp_accept = self.mtp_reject = 0
        self.mtp_accept2 = 0     # rounds that accepted both drafts (MTP_K = 2)
        self.verify_tag = 0      # bumps on every verify forward (snap_s owner)
        self.segs = None         # (start, len) per sequence of a segmented forward
        self.lab_s = self.lab_conv = None   # its scratch (label_logprobs_batch)
        # Speculation is OFF until the engine asks (CONTRACT.md, optional
        # verbs). `split_max` is how many of the MTP_SPLIT_MAX buffer rows are
        # actually used: 1 is the shipped decode's worth, 4 is the verify's.
        self.mtp_on = bool(MTP_SPEC)
        self.split_max = MTP_SPLIT_MAX if self.mtp_on else 1
        self.mtp_loaded = False
        self.snap_s = None
        self.snap_c: tuple = ()
        self.tok = _tokenizer.Tokenizer(self.dir)
        self.aux: dict[str, tuple[int, int]] = {}  # name -> (ptr, nbytes)
        global KV8, CACHE_BYTES_PER_TOKEN
        KV8 = _kv_format(self.dir)
        CACHE_BYTES_PER_TOKEN = 16 * 2 * _kv_row()
        if KV8 and not (ATT_GQA and APF_FA):
            raise RuntimeError("the int8 KV cache (KV8) is read by attention_split_gqa_q8 and "
                               "attention_prefill_fa_q8 only: ORCA_ATT_GQA and ORCA_APF_FA must be 1")
        self.log("info", f"kv cache: {'int8' if KV8 else 'fp16'}")

        self._read_index()
        self._check_geometry()
        self._open_device()
        self._load_plan()
        self._upload()
        self._allocate(model_dir)
        self._verify_mul1()
        if self.mtp_on:
            # the env override asks for speculation before the engine does
            self.set_speculative(True)

        self.log("info", f"open: {time.time() - t0:.1f}s, "
                         f"weights {self.weight_bytes / 1e9:.2f} GB resident")
        return Facts(
            cache_bytes_per_token=CACHE_BYTES_PER_TOKEN,
            max_batch=MAX_BATCH,
            max_context=MAX_CONTEXT,
            # must cover the manifest's list (rule 3: a disagreement is a hard
            # error). `label_logprobs` is the second optional verb
            # (CONTRACT.md, optional verbs): label log-probs at the answer
            # position, one prefill, no sampling -- see the verb below.
            # `tool_calls`: the template renders function tools and the verb
            # reads the calls back (tokenizer.tool_calls).
            capabilities=["speculative", "label_logprobs", "tool_calls"],
            stop_tokens=list(STOP_TOKENS),
            thinking_markers=list(THINKING_MARKERS),
            thinking_open_at_start=True,
            # the levels the template takes; the manifest declares the same list
            # and the engine makes a disagreement a hard error
            thinking_levels=list(_tokenizer.THINKING_LEVELS),
            default_sampling=default_sampling(self.dir),
        )

    # -- open-time plumbing -------------------------------------------------

    def _read_index(self) -> None:
        self.config = json.loads((self.dir / "config.json").read_text())
        tc = self.config.get("text_config", self.config)
        self.tcfg = tc
        types = tc["layer_types"]
        if len(types) != N_LAYERS:
            raise _contract_error(f"layer_types has {len(types)} entries, want {N_LAYERS}")
        self.full_ord = {i: k for k, i in enumerate(j for j, t in enumerate(types)
                                                    if t == "full_attention")}
        self.gdn_ord = {i: k for k, i in enumerate(j for j, t in enumerate(types)
                                                   if t != "full_attention")}
        if len(self.full_ord) != 16 or len(self.gdn_ord) != 48:
            raise _contract_error(
                f"the cache layout assumes 16/48 layers, got {len(self.full_ord)}"
                f"/{len(self.gdn_ord)}")
        self.gdn_layers = sorted(self.gdn_ord)
        self.index = json.loads((self.dir / "model.safetensors.index.json").read_text())
        self.quant = json.loads((self.dir / "quantization_config.json").read_text())

    def _check_geometry(self) -> None:
        """The checkpoint must agree with this file's constants."""
        tc = self.tcfg
        want = {
            "hidden_size": HIDDEN, "num_hidden_layers": N_LAYERS,
            "intermediate_size": INTER, "vocab_size": VOCAB,
            "num_attention_heads": N_HEADS, "num_key_value_heads": N_KV,
            "head_dim": HEAD_DIM, "linear_num_key_heads": GDN_K_HEADS,
            "linear_num_value_heads": GDN_V_HEADS, "linear_key_head_dim": GDN_K,
            "linear_value_head_dim": GDN_V, "linear_conv_kernel_dim": CONV_WIDTH,
        }
        for key, value in want.items():
            if tc.get(key) != value:
                raise _contract_error(f"config {key} = {tc.get(key)}, runtime wants {value}")
        hidden = round(HEAD_DIM * tc["partial_rotary_factor"])
        if hidden != ROPE_DIM:
            raise _contract_error(f"partial rotary gives {hidden} dims, want {ROPE_DIM}")
        # float: config.json writes it as the integer 10000000, and the bridge
        # packs a Python int as an integer -- rope_partial's `float theta` then
        # read those bits as 1.4e-38 and scrambled 62 of the 64 rotary dims at
        # every position past 0 (2026-10-03: exact copy failed within 7 words)
        self.rope_theta = float((tc.get("rope_parameters") or {}).get("rope_theta", ROPE_THETA))

    def _open_device(self) -> None:
        self.dev = _bridge.Device(0)
        self.log("info", f"device {self.dev.name} sm_{self.dev.cc[0]}{self.dev.cc[1]}, "
                         f"{self.dev.total_memory / 2**30:.2f} GiB")
        self.dev.compile(HERE / "kernels.cu")
        self.dev.compile(HERE / "kernels_exl3.cu")
        self.mod = self.dev.load_module(HERE / "kernels.cubin", "kernels")
        self.mod_x = self.dev.load_module(HERE / "kernels_exl3.cubin", "kernels_exl3")
        self.k = {n: self.mod.kernel(n) for n in KERNEL_NAMES}
        self.kx = {n: self.mod_x.kernel(n) for n in KERNEL_X_NAMES}

    def _load_plan(self) -> None:
        """Which tensors we read, and how.

        The 39 `mtp.*` drafter tensors are PLANNED here and uploaded later, by
        `_load_mtp`, on the first `set_speculative(True)` -- so a run that never
        speculates never pays for them (CONTRACT.md, optional verbs). Their
        bitrate is NOT in
        quantization_config.json (that manifest stops at `lm_head`, the last
        target module), so it comes from the trellis shape the way the kernel
        does: `bits_x2 = shape[2] // 8`. Every `mtp.*` module is 4-bit
        (shape[2] == 64), which is the claim _reference/ARCHITECTURE.md makes
        and `_check_mtp_bits` below proves from the checkpoint.
        """
        # `roles` maps a tensor to its UPLOAD key, which a group prefixes
        # ("target:...", "mtp:...") so `_tensor` finds either buffer the same
        # way; `shard_file` keeps the plain file name, which `_raw` needs.
        self.roles: dict[str, str] = {}
        self.shard_file: dict[str, str] = {}
        storage = self.quant["tensor_storage"]
        self.bits: dict[str, int] = {}  # module -> bits_x2 code (the kernel's)
        self.mtp_modules: list[str] = []
        for name, shard in self.index["weight_map"].items():
            self.roles[name] = shard
            self.shard_file[name] = shard
        for module, entry in storage.items():
            bits = entry.get("bits_per_weight")
            if bits is None:
                continue
            # the kernel takes 2*bits: the constant differs only in scale, and
            # the trellis word count is bits*256/32 either way.
            self.bits[module] = {"2": 4, "3": 6, "3.5": 7, "4": 8, "6": 12}[str(bits)]

    def _header(self, shard: str) -> dict:
        """A checkpoint file's safetensors header, read once and kept.

        The dtype and shape of a tensor are properties of the FILE, not of the
        buffer it will land in, so they must be answerable for the drafter's
        tensors before `_load_mtp` has uploaded anything. One header is a few
        hundred KB of JSON, read once per shard, for the process.
        """
        got = self._headers.get(shard)
        if got is None:
            with open(self.dir / shard, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                got = (json.loads(fh.read(n)), 8 + n)
            self._headers[shard] = got
        return got[0]

    def _upload(self) -> None:
        """The target's weights. The drafter's are `_load_mtp`'s, on demand."""
        self._upload_group("target", lambda n: not n.startswith("mtp."))

    def _load_mtp(self) -> None:
        """Upload the drafter's 39 tensors. Idempotent, and claimed lazily.

        This is why the switch is a verb and not an `open` argument: a run that
        never speculates must not reserve 212 MB of device memory, and a run
        that does must not pay for them twice.
        """
        if self.mtp_loaded:
            return
        self._upload_group("mtp", lambda n: n.startswith("mtp."))
        self.mtp_loaded = True
        self._check_mtp()      # which announces the drafter, and checks it

    def _upload_group(self, group: str, keep) -> None:
        """One device buffer per shard of a group, each tensor at an aligned offset.

        The file's own layout packs tensors contiguously, which leaves 884 of
        them at offsets that are only 4-byte aligned — and the kernels read the
        trellis and the `suh`/`svh` scales with vector loads, so they fault on
        those. Laying the tensors out ourselves costs nothing but the padding
        (a few KB per shard).

        A group gets its own `shard_ptr` key, so `_tensor` reaches both groups
        through the same `roles -> shard_ptr` lookup and nothing downstream has
        to know the drafter is a second buffer.
        """
        ALIGN = 16
        names = [n for n in self.roles if keep(n)]
        shards = sorted({self.index["weight_map"][n] for n in names})
        for shard in shards:
            path = self.dir / shard
            header = self._header(shard)
            body = self._headers[shard][1]
            with open(path, "rb") as fh:
                plan = []
                cursor = 0
                for name, info in header.items():
                    if name == "__metadata__" or not keep(name):
                        continue
                    lo, hi = info["data_offsets"]
                    cursor = (cursor + ALIGN - 1) // ALIGN * ALIGN
                    plan.append((name, lo, cursor, hi - lo))
                    cursor += hi - lo
                buf = self.dev.alloc(max(cursor, 1))
                for name, lo, off, nbytes in plan:
                    fh.seek(body + lo)
                    done = 0
                    while done < nbytes:
                        block = fh.read(min(8 << 20, nbytes - done))
                        if not block:
                            break
                        self.dev.htod(buf, block, offset=off + done)
                        done += len(block)
                    self.align_off[name] = off
            self._aux_bufs.append(buf)
            key = group + ":" + shard
            self.shard_ptr[key] = (buf.ptr, header, buf.nbytes)
            self.weight_bytes += buf.nbytes
            if group == "target":
                self.target_bytes += buf.nbytes
            for name in names:
                if self.index["weight_map"][name] == shard:
                    self.roles[name] = key
            self.log("info", f"uploaded {shard} ({buf.nbytes / 1e9:.3f} GB, "
                             f"{len(plan)} tensors)")

    def _tensor(self, name: str):
        """(ptr, nbytes, dtype, shape) of a checkpoint tensor, wherever it lives."""
        cached = self.aux.get(name)
        if cached is not None:
            return cached[0], cached[1], "AUX", None
        shard = self.roles[name]
        base, header, _ = self.shard_ptr[shard]
        info = header[name]
        return (base + self.align_off[name], info["data_offsets"][1] - info["data_offsets"][0],
                info["dtype"], info["shape"])

    def ptr(self, name: str) -> int:
        return self._tensor(name)[0]

    def _raw(self, name: str) -> bytes:
        """A tensor's bytes, read from the checkpoint on the host."""
        shard = self.shard_file[name]
        with open(self.dir / shard, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
            lo, hi = header[name]["data_offsets"]
            fh.seek(8 + n + lo)
            return fh.read(hi - lo)

    def _convert(self, name: str) -> tuple:
        """The tensors whose stored dtype is not the dtype a kernel wants.

        Norm gammas are fp32 (`rmsnorm1p`/`rmsnorm_gated` take `const float*`),
        as are `A_log` and `dt_bias`; the conv weight goes to fp16 flattened
        [dim, width]; the int8 embedding keeps its bytes but its per-row scales
        become fp16. Everything else is used as stored.
        """
        raw = self._raw(name)
        assert self.roles_dtype(name) == "BF16", f"{name} is not a conversion target"
        count = len(raw) // 2
        words = struct.unpack(f"<{count}H", raw)
        floats = [struct.unpack("<f", struct.pack("<I", w << 16))[0] for w in words]
        # fp16 targets: the conv taps and the embedding's per-row scales.
        # Everything else here is a gamma or a decay constant, which the kernels
        # take as `const float*`.
        if name.endswith("linear_attn.norm.weight") and _GAMMA_SCALE != 1.0:
            floats = [v * _GAMMA_SCALE for v in floats]
        if name.endswith(("conv1d.weight", "embed_tokens.scales")):
            return self._retype(name, struct.pack(f"<{count}e", *floats))
        return self._retype(name, struct.pack(f"<{count}f", *floats))

    def roles_dtype(self, name: str) -> str:
        shard = self.roles[name]
        return self._header(self.shard_file[name])[name]["dtype"]

    def _retype(self, name: str, raw: bytes) -> tuple:
        """Put a host-side conversion in the aux arena and address it there."""
        buf = self.dev.alloc(len(raw))
        self.dev.htod(buf, raw)
        self._aux_bufs.append(buf)
        self.aux[name] = (buf.ptr, len(raw))
        return buf.ptr, len(raw), "AUX", None

    def _verify_mul1(self) -> None:
        """The one codebook constant every module must carry."""
        for module, entry in self.quant["tensor_storage"].items():
            mult = entry.get("mul1_multiplier")
            if mult is not None and mult != MUL1:
                raise _contract_error(f"{module}: mul1 multiplier {mult:#x}, want {MUL1:#x}")

    def _allocate(self, model_dir: str) -> None:
        # host-side conversions first: they need the shard headers, which are up
        for name in list(self.roles):
            if name.endswith(("conv1d.weight", "A_log", "dt_bias", "q_norm.weight",
                              "k_norm.weight", "embed_tokens.scales")) \
                    or name.endswith(".norm.weight") or name.endswith("layernorm.weight"):
                if self.roles_dtype(name) == "BF16":
                    self._convert(name)
        T = MAX_CHUNK
        def arena(nbytes, label):
            buf = self.dev.alloc(nbytes)
            self._aux_bufs.append(buf)
            return buf
        self.w = {
            "ids": arena(T * 4, "ids"),
            "x": arena(T * HIDDEN * 2, "x"),           # the residual stream
            "h": arena(T * HIDDEN * 2, "h"),           # normed copy
            # sized by the widest GEMV *input*, which is down_proj's 17408
            "xr": arena(T * INTER * 2, "xr"),          # suh-scaled, hadamard'd
            # the same, fp32 and lane-ordered, for the m = 16 GEMV
            # (allocated only for the EXL3_PRE_X32 path it exists for)
            "xr32": arena(T * INTER * 4 if EXL3_PRE_X32 else 16, "xr32"),
            "z": arena(T * INTER * 2, "z"),            # raw GEMV output
            "p": arena(T * INTER * 2, "p"),            # post-hadamard GEMV output
            "qkv": arena(T * QKV_ROWS * 2, "qkv"),     # in_proj_qkv
            "c": arena(T * QKV_ROWS * 2, "c"),         # conv output
            "ln": arena(T * QKV_ROWS * 2, "ln"),       # l2norm output
            "zn": arena(T * 6144 * 2, "zn"),           # in_proj_z
            "y": arena(T * 6144 * 2, "y"),             # gdn_scan output
            "yn": arena(T * 6144 * 2, "yn"),           # gated-norm output
            "ab": arena(T * 48 * 4 * 2, "ab"),         # a, b (fp32)
            "bd": arena(T * 48 * 4 * 2, "bd"),         # beta, decay (fp32)
            "qg": arena(T * 12288 * 2, "qg"),          # q|gate
            "k": arena(T * 1024 * 2, "k"),             # k
            "v": arena(T * 1024 * 2, "v"),             # v
            "att": arena(T * 6144 * 2, "att"),         # attention output
            "mlp": arena(T * INTER * 2, "mlp"),        # gate
            "up": arena(T * INTER * 2, "up"),          # up
            "sm": arena(T * INTER * 2, "sm"),          # silu(gate)*up
            # fp32 partials for a split GEMV (S slices x T x out floats)
            "pt": arena(max(EXL3_PT_BYTES, XP_PT_BYTES), "pt"),
            # one row in the shipped build; MTP_SPLIT_MAX rows when the
            # speculative verify asks the shared head for a sample at every
            # position of its batch.
            "logits": arena(LOGIT_ROWS * VOCAB * 2, "logits"),
            "logits2": arena(LOGIT_ROWS * VOCAB * 2, "logits2"),
            "amax": arena(AMAX_BLOCKS * 4 + AMAX_BLOCKS * 4 + 8, "amax"),
            # the device sampler: per-block key histograms, the summed one (its
            # slot 0 is the row's largest key) and the candidate buffer
            # (count u32, then CAND_CAP f32 values, then CAND_CAP u32 indices)
            "chist": arena(CAND_BLOCKS * CAND_BINS * 4, "chist"),
            "csum": arena((CAND_BINS + 1) * 4 + CAND_BLOCKS * 4, "csum"),
            "cand": arena(8 + 8 * CAND_CAP, "cand"),
            # attention_split's partials are per (TOKEN, head, split), so the
            # arena is MTP_SPLIT_MAX rows deep. At MTP_SPLIT_MAX == 1 that is
            # the shipped size, byte for byte; the MTP build widens it so a
            # T = 2 verify can take the split path (Round 11's fix).
            "pacc": arena(MTP_SPLIT_MAX * ATT_SPLIT * N_HEADS * HEAD_DIM * 4, "pacc"),
            "pm": arena(MTP_SPLIT_MAX * ATT_SPLIT * N_HEADS * 4 * 2, "pm"),
        }
        # the drafter's `fc` input: cat([norm(embed), norm(hidden)]) per row,
        # 10240 wide, and the shared lm_head's rows for a T = 2 verify. Small,
        # and allocated here because the switch arrives after open().
        self.w["mcat"] = arena(MTP_WARM_CHUNK * 2 * HIDDEN * 2, "mcat")
        # The target's final normed hidden, PRESERVED. The drafter reuses the
        # target's `h` arena for its own normed rows -- that is what keeps it
        # from needing a second set of ~440 KB-a-token arenas -- so the rows
        # the drafter's `pre_fc_norm_hidden` needs are copied out once per
        # forward, before the drafter's own layer overwrites them.
        self.w["hkeep"] = arena(MAX_CHUNK * HIDDEN * 2, "hkeep")
        self.amax_key = self.w["amax"].ptr
        self.amax_idx = self.w["amax"].ptr + AMAX_BLOCKS * 4
        self.amax_out = self.w["amax"].ptr + AMAX_BLOCKS * 8
        self.cand_perm = self.w["csum"].ptr + (CAND_BINS + 1) * 4
        self.pacc = self.w["pacc"].ptr
        self.pm = self.w["pm"].ptr
        # attention_split's (m, d) partials are per (TOKEN, head, split) like
        # pacc's, so the two halves of the arena are MTP_SPLIT_MAX rows each --
        # one row's worth in the shipped build, which is all a T = 1 decode ever
        # wrote. At T = 2 a one-row offset would make row 1's m/d land on row
        # 0's accumulator.
        self.pd = self.w["pm"].ptr + MTP_SPLIT_MAX * ATT_SPLIT * N_HEADS * 4
        self.zero_block = self.dev.alloc(1 << 20)
        self.dev.htod(self.zero_block, bytes(1 << 20))
        # the fused GEMV's last-block counters: zero once, each launch's last
        # block resets its own (kernels_exl3.cu's exl3_post_tail)
        self.gcnt = self.dev.alloc(GCNT_SLOTS * 4)
        self._aux_bufs.append(self.gcnt)
        self._zero(self.gcnt)
        self.free_bytes = self.dev.free_memory()[0]
        self.log("info", f"free device memory after the load plan: "
                         f"{self.free_bytes / 2**30:.2f} GiB")
        # Hand the ledger the device truth: what the load left minus a safety
        # margin is what states and cached clones may occupy. The engine's
        # default budget (16 GiB) is larger than this whole card; left as-is it
        # admits clones until cuMemAlloc itself fails -- a 1 GiB KV clone mid-
        # generation took a chat down that way. Optional Accounting verb.
        margin = int(__import__("os").environ.get("ORCA_MEM_MARGIN",
                                                 str(256 << 20)))
        set_budget = getattr(self.services.accounting, "set_budget", None)
        self.kv_budget = None
        if set_budget is not None:
            kv_budget = max(self.free_bytes - margin, 0)
            set_budget(kv_budget)
            self.kv_budget = kv_budget
            self.log("info", f"kv/state budget: {kv_budget / 2**30:.2f} GiB "
                             f"(free minus {margin >> 20} MiB safety margin)")

    # -- scratch addressing -------------------------------------------------

    def _p(self, name: str) -> int:
        return self.w[name].ptr

    def _view(self, ptr: int, nbytes: int):
        """A DeviceBuffer over an address inside one of our arenas."""
        return _bridge.DeviceBuffer(ptr, nbytes, self.dev)

    def _salloc(self, nbytes: int):
        """State memory, refused like a budget claim (BudgetRefused) so the
        engine can evict and retry around it: a CudaError traceback mid-request
        cannot be recovered from, and the card's real free memory is what the
        ledger budget stands in for."""
        from footless.sdk import BudgetRefused
        try:
            return self.dev.alloc(nbytes)
        except Exception as e:
            raise BudgetRefused(
                f"device cannot admit {nbytes}B of state memory: {e}") from e

    # -- state verbs --------------------------------------------------------

    def new_state(self):
        state = _State()
        state.services = self.services
        # claimed BEFORE the device is asked: the claim is what lets the engine
        # evict cached states to make room, and a refusal anywhere below
        # frees what was already taken (free_state takes a half-built state)
        state.claim = self.services.accounting.claim(self._claim_bytes(state))
        try:
            state.s = self._salloc(GDN_S_BYTES)
            state.conv = (self._salloc(GDN_CONV_BYTES), None)
            state.conv = (state.conv[0], self._salloc(GDN_CONV_BYTES))
            state.logits = self._salloc(VOCAB * 2)
            m = state.mtp
            m.hprev = self._salloc(HIDDEN * 2)
            m.htail = self._salloc(HIDDEN * 2)
        except BaseException:
            self.free_state(state)
            raise
        self._zero(state.s)
        self._zero(state.conv[0])
        self._zero(state.conv[1])
        return state

    def clone_state(self, state):
        state2 = _State()
        state2.services = self.services
        state2.claim = self.services.accounting.claim(self._claim_bytes(state))
        try:
            self._clone_into(state, state2)
        except BaseException:
            self.free_state(state2)      # a refused buffer leaves nothing behind
            raise
        self._reclaim(state2)
        # a clone is what a cache keeps: no forwarded-but-unhanded token in it
        self._mtp_unforward(state2)
        return state2

    def _clone_into(self, state, state2) -> None:
        state2.tokens = list(state.tokens)
        state2.fed = state.fed
        state2.pending = state.pending
        state2.replay = None if state.replay is None else list(state.replay)
        state2.phase = state.phase
        state2.s = self._salloc(GDN_S_BYTES)
        state2.conv = (self._salloc(GDN_CONV_BYTES), None)
        state2.conv = (state2.conv[0], self._salloc(GDN_CONV_BYTES))
        state2.logits = self._salloc(VOCAB * 2)
        state2.has_logits = state.has_logits
        if state.has_logits:
            self.dev.dtod(state2.logits, state.logits, VOCAB * 2)
        self.dev.dtod(state2.s, state.s, GDN_S_BYTES)
        self.dev.dtod(state2.conv[0], state.conv[0], GDN_CONV_BYTES)
        self.dev.dtod(state2.conv[1], state.conv[1], GDN_CONV_BYTES)
        if state.kk is not None:
            n = N_LAYERS // 4
            state2.kk, state2.kv, state2.lcap = [None] * n, [None] * n, [0] * n
            for li in range(n):
                span = state.lcap[li] * _kv_row()
                state2.kk[li] = self._salloc(span)
                state2.kv[li] = self._salloc(span)
                state2.lcap[li] = state.lcap[li]
                self.dev.dtod(state2.kk[li], state.kk[li], span)
                self.dev.dtod(state2.kv[li], state.kv[li], span)
            state2.cap = state.cap
        m2 = state2.mtp
        m2.hprev = self._salloc(HIDDEN * 2)
        m2.htail = self._salloc(HIDDEN * 2)
        if self.mtp_on and state.mtp is not None:
            m = state.mtp
            self.dev.dtod(m2.hprev, m.hprev, HIDDEN * 2)
            self.dev.dtod(m2.htail, m.htail, HIDDEN * 2)
            m2.mtp_len = m.mtp_len
            m2.out = list(m.out)
            m2.draft = m.draft
            m2.draft2 = m.draft2
            m2.rb_tag = m.rb_tag
            self._mtp_grow(state2, state.cap)
            # _mtp_grow hands the CLONE fresh buffers, and it copies from the
            # clone's own (empty) drafter cache, so the drafter's KV has to
            # come across from the SOURCE explicitly. Without this the clone
            # claims `mtp_len` positions it cannot see and drafts from zeros:
            # measured, the clone's drafter KV read back all zeros while the
            # source had 7168 non-zero halves. That is invisible on a fresh
            # state and poisons every PREFIX-CACHE HIT, which is most turns of
            # a chat -- it made --mtp 1.45x SLOWER than plain decode there.
            if m.dk is not None and m2.dk is not None and m.mtp_len:
                n = min(m.cap, m2.cap)
                q8 = self._draft_q8()
                self._kv_copy(m2.dk, m2.cap, m.dk, m.cap, n, q8)
                self._kv_copy(m2.dv, m2.cap, m.dv, m.cap, n, q8)

    def _mtp_grow(self, state, cap: int) -> None:
        """(Re)allocate the DRAFTER's own one-layer KV cache to `cap` positions.

        Position-addressable like the target's, so growing copies; the drafter
        cache is invalidated either way, because a copy is only as good as the
        `mtp_len` that describes it.
        """
        m = state.mtp
        if cap <= m.cap and m.dk is not None:
            return
        q8 = self._draft_q8()
        span = cap * _kv_row(q8)
        # k, then v: one buffer's room beside the cache, not two. A refusal
        # between them leaves a larger k, harmless (fp16 rows sit at the same
        # offsets at any capacity; an int8 buffer's scales move with it, so its
        # rows are copied region by region and `cap` stays the old one until
        # both are done) and `cap` still the smaller.
        keep = m.cap if (m.cap and m.mtp_len) else 0
        acc = self.services.accounting
        try:
            for which in ("dk", "dv"):
                old = getattr(m, which)
                if old is not None and old.nbytes >= span and not q8:
                    continue
                # the new buffer sits beside the old one until the copy is done:
                # claimed for that moment, like _grow's layers
                claim = acc.claim(_dev_bytes(span))
                try:
                    new = self._salloc(span)
                    if old is not None:
                        if keep:
                            self._kv_copy(new, cap, old, m.cap, min(keep, cap), q8)
                        old.free()
                    setattr(m, which, new)
                finally:
                    acc.release(claim)
        except BaseException:
            if q8:
                m.mtp_len = 0       # k laid out for `cap`, v for the old one: unusable
            raise
        m.cap = cap

    def _claim_bytes(self, state) -> int:
        """What `state` holds on the device: the fixed GDN state, the logits
        row, the drafter's hidden rows, every layer's K and V, and the
        drafter's one-layer KV -- each buffer as the device rounds it
        (_dev_bytes), since that is what fills the card."""
        n = (_dev_bytes(GDN_S_BYTES) + 2 * _dev_bytes(GDN_CONV_BYTES)
             + _dev_bytes(VOCAB * 2) + 2 * _dev_bytes(HIDDEN * 2))
        if state.lcap is not None:
            n += sum(2 * _dev_bytes(c * _kv_row()) for c in state.lcap)
        m = state.mtp
        if m is not None:
            n += sum(_dev_bytes(b.nbytes) for b in (m.dk, m.dv) if b is not None)
        return n

    def _reclaim(self, state) -> None:
        """One claim for what the state holds now (the device already holds it,
        and the ledger held at least as much a moment ago)."""
        acc = self.services.accounting
        if state.claim is not None:
            acc.release(state.claim)
            state.claim = None
        state.claim = acc.claim(self._claim_bytes(state))

    def _kvl(self, state, ord_: int):
        """(k pointer, v pointer, capacity) of full-attention layer `ord_`."""
        return state.kk[ord_].ptr, state.kv[ord_].ptr, state.lcap[ord_]

    def _kv_copy(self, dst, dst_cap: int, src, src_cap: int, rows: int,
                 q8: bool | None = None) -> None:
        """The first `rows` positions of one layer's K (or V) buffer into
        another of a different capacity. int8 buffers hold two regions whose
        offsets scale with the capacity: the values, then the block scales."""
        q8 = KV8 if q8 is None else q8
        if not q8:
            self.dev.dtod(dst, src, nbytes=rows * _kv_row(False))
            return
        vals = N_KV * HEAD_DIM
        self.dev.dtod(dst, src, nbytes=rows * vals)
        self.dev.dtod(dst, src, nbytes=rows * (_kv_row(True) - vals),
                      dst_off=dst_cap * vals, src_off=src_cap * vals)

    @staticmethod
    def _draft_q8() -> bool:
        """Is the drafter's KV cache int8 (the package's format, DRAFT_KV8)?"""
        return bool(KV8 and DRAFT_KV8)

    def _kv_rows(self, buf, cap: int, rows: int) -> bytes:
        """The first `rows` positions of one layer's buffer, as a buffer of
        exactly `rows` positions would hold them (export's layout)."""
        if not KV8:
            return self.dev.dtoh(buf, rows * _kv_row())
        if rows == cap:
            return self.dev.dtoh(buf, cap * _kv_row())
        vals = N_KV * HEAD_DIM
        return (self.dev.dtoh(buf, rows * vals)
                + self.dev.dtoh(self._view(buf.ptr + cap * vals, rows * (_kv_row() - vals)),
                                rows * (_kv_row() - vals)))

    @staticmethod
    def _next_cap(need: int) -> int:
        """The KV capacity for `need` positions: doubling from 32 up to
        KV_GROW_STEP, then whole steps of it (a long context over-reserves
        less than a step), capped at MAX_CONTEXT. A fixed ladder, so an
        exported capacity is one a fresh state reproduces."""
        c = 32
        while c < need and c < KV_GROW_STEP:
            c *= 2
        if c < need:
            c = -(-need // KV_GROW_STEP) * KV_GROW_STEP
        return min(c, MAX_CONTEXT)

    def truncate_state(self, state, n_tokens: int) -> None:
        """Keep the first n_tokens, behaving exactly as a fresh state fed them.

        The KV cache is position-addressable and shrinks for free. The GDN
        recurrence is not: dropping its tail would need the tail re-run, so a
        truncate below the cached count zeroes the recurrent state and records
        the kept ids as a replay. The next step re-forwards them from position
        0, as one prefill. Correct by construction; costs one re-prefill.
        """
        if n_tokens < 0:
            raise _contract_error(f"truncate to a negative count {n_tokens}")
        # what is kept is the next request's prompt, all of it: a state from
        # the cache once ended a request, and `fed` still split it there --
        # the old answer counted as output, and the presence penalty fell on
        # the code the model had just written when it wrote it again
        if n_tokens >= state.logical():
            state.fed = len(state.tokens)
            return
        if n_tokens == len(state.tokens) - 1 and self._mtp_unforward(state):
            pass                    # the level-2 round's extra row, undone for free
        if n_tokens == len(state.tokens):
            state.pending = None
            state.fed = n_tokens
            return
        # Truncating to zero leaves the state a fresh one: the zeroed recurrence
        # already matches the empty prefix, so there is no replay to record and
        # `replay` stays None (`[]` would claim a prefix needs re-forwarding
        # when none does -- both are falsy to `_step_one`, so this is a
        # consistency fix, not a behavioural one).
        state.replay = list(state.tokens[:n_tokens]) or None
        state.tokens = state.tokens[:n_tokens]
        state.fed = n_tokens
        state.pending = None
        state.has_logits = False  # the carried row described a longer prefix
        if self.mtp_on and state.mtp is not None:
            state.mtp.mtp_len = 0   # the drafter's cache described the old tail
            state.mtp.out = []
            state.mtp.draft = None
            state.mtp.draft2 = None
        self._zero(state.s)
        self._zero(state.conv[0])
        self._zero(state.conv[1])

    def state_nbytes(self, state) -> int:
        return GDN_S_BYTES + 2 * GDN_CONV_BYTES + state.cap * CACHE_BYTES_PER_TOKEN

    def export_state(self, state) -> bytes:
        head = json.dumps({"tokens": state.tokens, "fed": state.fed,
                           "pending": state.pending, "replay": state.replay,
                           "cap": state.cap, "phase": state.phase}).encode()
        out = bytearray(struct.pack("<I", len(head)) + head)
        out += self.dev.dtoh(state.s, GDN_S_BYTES)
        out += self.dev.dtoh(state.conv[0], GDN_CONV_BYTES)
        out += self.dev.dtoh(state.conv[1], GDN_CONV_BYTES)
        out += struct.pack("<I", 1 if state.has_logits else 0)
        if state.has_logits:
            out += self.dev.dtoh(state.logits, VOCAB * 2)
        if state.cap:
            # [layer][cap][4][256], the layout of the one-buffer cache this
            # format was defined over
            for bufs in (state.kk, state.kv):
                for buf, lcap in zip(bufs, state.lcap):
                    out += self._kv_rows(buf, lcap, state.cap)
        return bytes(out)

    def import_state(self, blob: bytes):
        n = struct.unpack("<I", blob[:4])[0]
        head = json.loads(blob[4:4 + n])
        off = 4 + n
        state = self.new_state()
        state.tokens = list(head["tokens"])
        state.fed = head["fed"]
        state.pending = head["pending"]
        state.replay = head["replay"]
        state.phase = head["phase"]
        self.dev.htod(state.s, blob[off:off + GDN_S_BYTES])
        off += GDN_S_BYTES
        self.dev.htod(state.conv[0], blob[off:off + GDN_CONV_BYTES])
        off += GDN_CONV_BYTES
        self.dev.htod(state.conv[1], blob[off:off + GDN_CONV_BYTES])
        off += GDN_CONV_BYTES
        state.has_logits = struct.unpack("<I", blob[off:off + 4])[0] == 1
        off += 4
        if self.mtp_on and state.mtp is not None:
            state.mtp.mtp_len = 0    # the blob carries no drafter cache
        if state.has_logits:
            self.dev.htod(state.logits, blob[off:off + VOCAB * 2])
            off += VOCAB * 2
        if head["cap"]:
            # `_grow` must be asked for the capacity, not told it: it returns
            # immediately when `need <= state.cap`, so assigning `state.cap`
            # first left the KV buffers unallocated (`state.kk is None`) AND the
            # claim at its cap-0 size -- every import of a grown state raised
            # there, and the accounting never grew to cover the rows it does
            # write. Ask first, then check what it reserved: the export ladder
            # only ever emits a capacity `_grow` reproduces exactly, so a
            # mismatch is a foreign blob whose rows would be read at the wrong
            # stride. That is a hard error, never a guess.
            self._grow(state, head["cap"], reserve_only=True)
            if state.cap != head["cap"]:
                raise _contract_error(
                    f"blob declares cap {head['cap']}, which reserves "
                    f"{state.cap} positions")
            row = head["cap"] * _kv_row()          # cap == every lcap: a fresh grow
            for bufs in (state.kk, state.kv):
                for buf in bufs:
                    self.dev.htod(buf, blob[off:off + row])
                    off += row
        return state

    def free_state(self, state) -> None:
        if state.claim is not None:
            self.services.accounting.release(state.claim)
            state.claim = None
        for buf in (state.s, state.logits, *(state.kk or ()), *(state.kv or ())):
            if buf is not None:
                buf.free()
        state.lcap = None
        state.cap = 0
        if state.conv is not None:
            for buf in state.conv:
                if buf is not None:
                    buf.free()
        state.s = state.kk = state.kv = state.conv = state.logits = None
        state.tokens = []
        if state.mtp is not None:       # hprev/htail exist with speculation off too
            m = state.mtp
            for buf in (m.dk, m.dv, m.hprev, m.htail):
                if buf is not None:
                    buf.free()
            m.dk = m.dv = m.hprev = m.htail = None
            m.cap = 0

    def _zero(self, buf) -> None:
        """Zero a device buffer. The bridge does not bind `cuMemsetD8_v2`, so the
        fill is a chain of device-to-device copies from a zeroed block."""
        off = 0
        while off < buf.nbytes:
            n = min(self.zero_block.nbytes, buf.nbytes - off)
            self.dev.dtod(buf, self.zero_block, nbytes=n, dst_off=off)
            off += n

    def _grow(self, state, need: int, reserve_only: bool = False) -> None:
        """Grow the KV cache to hold `need` positions (`_next_cap`), one layer
        at a time: each layer's new K and V are claimed and allocated beside
        its old ones, the rows copied, the old freed, so the room a growth
        needs is one layer's -- not a second cache. A refusal part-way leaves
        a consistent state: the grown layers hold more, `cap` (the minimum)
        is unchanged, and the claim is what the device holds."""
        if need <= state.cap:
            return
        if need > MAX_CONTEXT:
            raise _contract_error(f"context {need} exceeds max_context {MAX_CONTEXT}")
        cap = self._next_cap(need)
        n = N_LAYERS // 4
        if state.kk is None:
            state.kk, state.kv, state.lcap = [None] * n, [None] * n, [0] * n
        acc = self.services.accounting
        span = cap * _kv_row()
        for li in range(n):
            old = state.lcap[li]
            if old >= cap:
                continue
            claim = acc.claim(2 * _dev_bytes(span))
            k = v = None
            try:
                k = self._salloc(span)
                v = self._salloc(span)
            except BaseException:
                if k is not None:
                    k.free()
                acc.release(claim)
                raise
            if old and not reserve_only:
                for dst, src in ((k, state.kk[li]), (v, state.kv[li])):
                    self._kv_copy(dst, cap, src, old, old)
            for buf in (state.kk[li], state.kv[li]):
                if buf is not None:
                    buf.free()
            state.kk[li], state.kv[li], state.lcap[li] = k, v, cap
            acc.release(claim)
            self._reclaim(state)
        state.cap = min(state.lcap)
        if self.mtp_on and state.mtp is not None:
            self._mtp_grow(state, cap)
            self._reclaim(state)

    # -- text ---------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self.tok.decode(tokens)

    def chat(self, messages: list[dict], thinking: bool = True,
             level: str | None = None, tools: list | None = None) -> list[int]:
        """`level` is one of the package's `thinking_levels`; None leaves the
        template its own default. `tools` go into the template's system turn."""
        return self.tok.chat(messages, thinking, tools=tools, reasoning_effort=level)

    def tool_calls(self, text: str, tools: list, final: bool) -> tuple[str, list[dict]]:
        """The answer's calls, read back in the template's own format (the
        optional verb behind `tool_calls`; tokenizer.tool_calls)."""
        return _tokenizer.tool_calls(text, tools, final)

    def close(self) -> None:
        """Release everything we allocated. The engine frees its states first
        (PrefixCache.free_all), so what is left is the weights and the scratch."""
        if getattr(self, "dev", None) is None:
            return
        for buf in getattr(self, "_aux_bufs", []):
            try:
                buf.free()
            except Exception:
                pass
        self._aux_bufs = []
        self.dev.close()

    # -- label_logprobs (optional verb) -------------------------------------
    #
    # Behind the capability of its own name (CONTRACT.md, optional verbs): the
    # engine hands the RENDERED prompt ids and the label strings and gets
    # (token_id, full-vocabulary next-token log-probability) per label, in
    # order. One prefill, no sampling, no penalties, no temperature -- the raw
    # model distribution at the answer position, which is what the decision
    # routes' math wants (PLAN_DECISIONS.md §4.1/§4.5). The state is a
    # temporary one (a `new_state` claim on the ledger, freed after) and
    # nothing shared is left touched: no `hkeep` copy, no drafter rows, no
    # pending/replay -- `self.mtp_on` decides nothing here.

    def label_logprobs(self, input_tokens: list[int], labels: list[str]) -> list[tuple[int, float]]:
        """Resolve each label to the single token it adds at the answer position after
        input_tokens' text, and return (token_id, full-vocabulary next-token log-probability)
        for each, in order. One prefill, no sampling, no penalties, no temperature.
        Raises ValueError naming the offending label when it is not exactly one token at the
        answer position, when labels collide on one token id, or when the prompt text does not
        round-trip through decode/encode."""
        if not input_tokens:
            # no position to score from: a forward of zero tokens computes
            # nothing this model can condition on (there is no bos)
            raise ValueError("input_tokens is empty: there is no answer position")
        ids = self._label_ids(input_tokens, labels)
        if not ids:
            return []            # nothing to score; every gate above still ran
        raw = self._label_row(input_tokens)
        return list(zip(ids, _row_logprobs(raw, ids)))

    def label_logprobs_batch(self, items: list, state=None) -> list[list[tuple[int, float]]]:
        """`label_logprobs` for several (input_tokens, labels) at once, the
        result of each what `label_logprobs` returns for it alone (the same
        model, a different split of the same forward: equal to fp16 rounding).

        The prompts of one decision request share their head (template and
        state). That prefix is forwarded ONCE; the tails then run together as
        one segmented forward (`self.segs`): every projection and norm over all
        their tokens in one pass over the weights, and per tail only what is
        position-dependent -- rope, K/V and attention over prefix + itself, the
        conv and the GDN scan from copies of the prefix's state. A label error
        raises ValueError with `.item` = the index of the offending item.

        `state`, when given, is the head already forwarded (by `step`, e.g. a
        cached prefix): its tokens are a prefix of every item's, nothing is
        pending or queued for replay, and the tails run from it. The state is
        left as found -- the segments read its recurrence and conv history and
        write only KV rows past its tokens, which nothing reads -- so the
        caller may keep it (a prefix cache entry).
        """
        if state is not None:
            if state.s is None:
                raise _contract_error("state was freed")
            if state.pending is not None or state.replay:
                raise _contract_error("label_logprobs_batch: the state has a "
                                      "token pending or a replay queued")
            n0 = len(state.tokens)
            for input_tokens, _ in items:
                if len(input_tokens) <= n0 or list(input_tokens[:n0]) != state.tokens:
                    raise _contract_error("label_logprobs_batch: the state's tokens "
                                          "are not a proper prefix of every item")
        ids_all = []
        for k, (input_tokens, labels) in enumerate(items):
            if not input_tokens:
                exc = ValueError("input_tokens is empty: there is no answer position")
                exc.item = k
                raise exc
            try:
                ids_all.append(self._label_ids(input_tokens, labels))
            except ValueError as exc:
                exc.item = k
                raise
        seqs = [list(t) for t, _ in items]
        if state is not None:
            rows = self._label_tails(state, seqs, len(state.tokens)) if seqs else []
        else:
            rows = self._label_rows_batch(seqs) if len(seqs) > 1 else \
                [self._label_row(seqs[0])] if seqs else []
        return [list(zip(ids, _row_logprobs(raw, ids))) if ids else []
                for ids, raw in zip(ids_all, rows)]

    def _label_rows_batch(self, seqs: list[list[int]]) -> list[bytes]:
        """The answer-position rows of several prompts: the shared prefix once,
        then the tails as segmented forwards of up to MAX_CHUNK tokens."""
        P = len(seqs[0])
        for q in seqs[1:]:
            n = min(P, len(q))
            j = 0
            while j < n and q[j] == seqs[0][j]:
                j += 1
            P = j
        P = min(P, min(len(q) for q in seqs) - 1)    # every tail keeps its answer token
        if P < LABEL_PREFIX_MIN:
            return [self._label_row(q) for q in seqs]
        state = self.new_state()
        try:
            self._grow(state, max(len(q) for q in seqs))
            for start in range(0, P, MAX_CHUNK):
                chunk = seqs[0][start:min(P, start + MAX_CHUNK)]
                self._forward(state, chunk, start, want_logits=False)
                state.phase ^= 1
            state.tokens = list(seqs[0][:P])
            state.fed = P
            return self._label_tails(state, seqs, P)
        finally:
            self.free_state(state)

    def _label_tails(self, state, seqs: list[list[int]], P: int) -> list[bytes]:
        """The answer-position rows of `seqs` whose first P tokens `state`
        holds: the tails as segmented forwards from it (`_label_groups`)."""
        if self.lab_s is None:
            self.lab_s = self.dev.alloc(GDN_V_HEADS * GDN_V * GDN_K * 4)
            self.lab_conv = self.dev.alloc(GDN_CONV_BYTES)
            self._aux_bufs += [self.lab_s, self.lab_conv]
        out: list = [None] * len(seqs)
        self._grow(state, max(len(q) for q in seqs))
        for group in self._label_groups([len(q) - P for q in seqs]):
            tokens, segs = [], []
            for k in group:
                segs.append((len(tokens), len(seqs[k]) - P))
                tokens += seqs[k][P:]
            self.segs = segs
            try:
                self._forward(state, tokens, P, want_logits=False)
            finally:
                self.segs = None
            # each segment's last row to the front of `h` (sources sit at
            # or after their destination, so in order nothing is clobbered)
            h = self._p("h")
            for r, (s0, L) in enumerate(segs):
                src = s0 + L - 1
                if src != r:
                    self.dev.dtod(self._view(h + r * HIDDEN * 2, HIDDEN * 2),
                                  self._view(h + src * HIDDEN * 2, HIDDEN * 2),
                                  HIDDEN * 2)
            for r0 in range(0, len(group), LOGIT_ROWS):
                n = min(LOGIT_ROWS, len(group) - r0)
                buf = self._head(h + r0 * HIDDEN * 2, n)
                self.dev.sync()
                for r in range(n):
                    out[group[r0 + r]] = self.dev.dtoh(
                        self._view(buf.ptr + r * VOCAB * 2, VOCAB * 2), MIN_VOCAB_ID * 2)
        return out

    @staticmethod
    def _forward_ms(T: int) -> float:
        for top, ms in FORWARD_MS:
            if T <= top:
                return ms
        return FORWARD_MS[-1][1] + 5.0 * (T - FORWARD_MS[-1][0])

    def _label_groups(self, tails: list[int]) -> list[list[int]]:
        """The tails' indices split into segmented forwards of the least
        estimated time (FORWARD_MS steps with the tile count, so e.g. 55 + 45
        and 29 in two forwards beat 129 in one). Exact over subsets up to 12
        tails, longest-first packing beyond; no group past MAX_CHUNK tokens."""
        n = len(tails)
        if n > 12:
            groups, cur, size = [], [], 0
            for k in sorted(range(n), key=lambda k: -tails[k]):
                if cur and size + tails[k] > MAX_CHUNK:
                    groups.append(cur)
                    cur, size = [], 0
                cur.append(k)
                size += tails[k]
            return groups + ([cur] if cur else [])
        full = (1 << n) - 1
        size = [0] * (full + 1)
        for m in range(1, full + 1):
            low = m & -m
            size[m] = size[m ^ low] + tails[low.bit_length() - 1]
        best = [0.0] + [float("inf")] * full
        pick = [0] * (full + 1)
        for m in range(1, full + 1):
            low = m & -m                     # the lowest tail goes in this group
            rest = m ^ low
            sub = rest
            while True:
                g = sub | low
                if size[g] <= MAX_CHUNK:
                    c = best[m ^ g] + self._forward_ms(size[g])
                    if c < best[m]:
                        best[m], pick[m] = c, g
                if sub == 0:
                    break
                sub = (sub - 1) & rest
        groups, m = [], full
        while m:
            g = pick[m]
            groups.append([k for k in range(n) if g >> k & 1])
            m ^= g
        return groups

    def _label_ids(self, input_tokens: list[int], labels: list[str]) -> list[int]:
        """Each label's one token at the answer position, or a ValueError.

        The reference's `label_context` / `label_token_id` / `_encode_labels`
        (serving_decisions.py 456-511) against this tokenizer. First the
        round-trip gate: a prompt whose text does not re-encode to its own ids
        cannot be checked OR scored honestly, so it is refused before any
        forward runs (the reference refuses lossy tokenizers the same way).
        Then each label must add exactly ONE id to the context -- and no two
        labels may share one.
        """
        text = self.decode(input_tokens)
        again = self.encode(text)
        if again != list(input_tokens):
            raise ValueError(
                f"the prompt text does not round-trip through decode/encode: "
                f"{len(input_tokens)} ids decode to text that re-encodes to "
                f"{len(again)} ids")
        ctx_text, ctx_ids = self._label_context(text, list(input_tokens))
        out: list[int] = []
        for label in labels:
            ids = self.encode(ctx_text + label)
            if len(ids) != len(ctx_ids) + 1 or ids[:-1] != ctx_ids:
                raise ValueError(
                    f"label {label!r} is not one distinct token at the answer position")
            token_id = ids[-1]
            if token_id in out:
                raise ValueError(
                    f"label {label!r} is not one distinct token at the answer "
                    f"position: token {token_id} is already an earlier label's")
            out.append(token_id)
        return out

    def _label_context(self, text: str, text_ids: list[int]) -> tuple[str, list[int]]:
        """(text, ids) the labels are checked against -- the SUFFIX after the
        last added token when that suffix tokenizes on its own, else the whole
        prompt.

        Added tokens are split out before tokenization, so the text after the
        last one encodes independently of everything before it -- and when the
        prompt ends with an added token, each label starts a fresh segment,
        which is how the model continues after it. The thinking-off chat prompt
        ends `">"\\n\\n` (the close marker then im_end's `>` then two newlines),
        so the suffix here is `"\\n\\n"` and every label is checked against two
        newlines instead of the whole prompt: O(1) per label against O(prompt),
        which is what makes 255 pair labels affordable.
        """
        added = self.tok.added_tokens
        last = None
        for i in range(len(text_ids) - 1, -1, -1):
            if text_ids[i] in added:
                last = i
                break
        if last is not None:
            token = added[text_ids[last]]
            start = text.rfind(token)
            if start >= 0:
                suffix = text[start + len(token):]
                if self.encode(suffix) == text_ids[last + 1:]:
                    return suffix, text_ids[last + 1:]
        return text, text_ids

    def _label_row(self, input_tokens: list[int]) -> bytes:
        """The answer-position logits row for `input_tokens`: MIN_VOCAB_ID fp16
        halves on the host, the row the sampler reads (the padded tail past the
        tokenizer's vocabulary is embedding padding and is never part of a
        distribution).

        ONE prefill on a temporary state, and `_step_one`'s prefill loop
        exactly -- `_grow`, MAX_CHUNK sub-chunks, `want_logits` on the last one
        only, the conv ping-pong per sub-chunk -- minus the drafter (no
        `_mtp_warm`, no `hkeep` copy: that buffer is the drafter's). No sample
        runs, so no penalty stack and no temperature touch the row: the raw
        model distribution a `step` at this position would sample from. The
        readback is one `dtoh` after a device sync (`step` syncs for its
        caller; this verb runs outside `step` and must not read a half-written
        row). The state is freed whatever happens -- its claim is
        `new_state`'s own, and a leak is the next request refused.
        """
        state = self.new_state()
        try:
            self._grow(state, len(input_tokens))
            row = None
            for start in range(0, len(input_tokens), MAX_CHUNK):
                chunk = input_tokens[start:start + MAX_CHUNK]
                want = start + len(chunk) == len(input_tokens)
                row = self._forward(state, chunk, start, want_logits=want)
                state.phase ^= 1
            state.tokens = list(input_tokens)
            state.fed = len(input_tokens)
            self.dev.sync()
            return self.dev.dtoh(row, MIN_VOCAB_ID * 2)
        finally:
            self.free_state(state)

    # -- the step ----------------------------------------------------------

    def step(self, batch, ctx):
        from footless.sdk import ContractError, StepResult

        if len(batch) > MAX_BATCH:
            raise ContractError(f"batch of {len(batch)} exceeds max_batch {MAX_BATCH}")
        out = []
        for item in batch:
            if ctx.cancelled():
                out.append(StepResult(None, True))
                continue
            out.append(self._step_one(item))
        # step() is synchronous to its caller: the engine times it as one, and
        # an undrained queue reports the host's enqueue rate (209 t/s here
        # against 105.6 real)
        self.dev.sync()
        return out

    def _step_one(self, item):
        from footless.sdk import ContractError, StepResult

        state = item.state
        if state.s is None:
            raise ContractError("state was freed")
        if self.mtp_on and state.mtp is not None and state.mtp.out:
            # a verified token the engine has not been handed out yet. The
            # engine's decode loop feeds input_tokens=[] and takes one token per
            # step, so a round that verified two hands the second out here, with
            # no device work at all -- and the state is already advanced past it.
            #
            # `forwarded=False` is the whole reason that flag exists. This step
            # takes about five MICROSECONDS, so the engine booking it as a
            # decode speed would report ~200 000 t/s and drag the turn's mean
            # and max with it (measured: mean 68 701 t/s beside a real 15.0).
            # The token and the time still count toward the aggregate, which is
            # right; it is only not a sample of how fast a forward runs.
            if not item.input_tokens:
                return StepResult(state.mtp.out.pop(0), False, forwarded=False)
            # A real prompt feed with a queued tail: the queue and the pending
            # token belong to the PREVIOUS request's stream (that request ended
            # mid-queue). The new request's stream is the conversation now, so
            # the stale tail is dropped rather than spliced into it -- and the
            # prompt is certainly not dropped behind it, which is what the
            # unconditional handout did: a whole prefill vanished behind one
            # queued token (it booked as 21 tokens at 47 683 tok/s).
            self._mtp_unforward(state)
            state.mtp.out = []
            state.mtp.draft = None
            state.mtp.draft2 = None
            state.pending = None
        inputs = list(item.input_tokens)
        carried = state.pending is not None
        if carried:
            inputs.insert(0, state.pending)
            state.pending = None
        replay, state.replay = state.replay, None
        if replay:
            inputs = list(replay) + inputs
        # `state.fed` is the prompt|output split of `state.tokens`, and a
        # carried token lands on one side or the other by WHAT FED IT: on a
        # prompt feed (the chat stream's request boundary) the carried tail is
        # the previous request's output and becomes THIS request's prompt --
        # vLLM's split for a new request -- while a decode step's carried
        # token is this request's own output. Counting only `input_tokens` put
        # the split inside the previous answer from turn two of a chat on, so
        # the frequency/presence counts penalized the new question's vocabulary
        # and let the answer's own repeats through.
        if item.input_tokens:
            # a prompt feed starts a request: all that came before is prompt
            state.fed = len(state.tokens)
        fed_add = len(item.input_tokens) + (1 if carried and item.input_tokens else 0)
        if not inputs and item.sampling is not None and not state.has_logits:
            raise ContractError(
                "nothing to forward: empty step with no pending token and no "
                "logits carried by the state")

        pos0 = len(state.tokens) - (len(replay) if replay else 0)
        if pos0 + len(inputs) > MAX_CONTEXT:
            raise ContractError(
                f"state would reach {pos0 + len(inputs)} tokens, max_context {MAX_CONTEXT}")

        logits = state.logits if (not inputs and state.has_logits) else None
        lastT = 0        # rows in the LAST sub-chunk's `h`, for the drafter
        m = state.mtp if self.mtp_on else None
        if m is not None and item.sampling is not None:
            m.spec = item.sampling

        # Is this step a speculative round? Decided BEFORE the forward, because
        # the round's T = 2 forward IS the step's forward: running the shipped
        # single-token forward first and the round after it would feed position
        # P twice and leave the recurrence two positions ahead of `tokens`.
        want_round = (m is not None and item.sampling is not None and inputs
                      and len(inputs) == 1 and not replay
                      and m.draft is not None and m.mtp_len >= pos0
                      and m.cap >= pos0 + 2 and pos0 + 2 <= MAX_CONTEXT)

        if inputs and not want_round:
            self._grow(state, pos0 + len(inputs))
            for start in range(0, len(inputs), MAX_CHUNK):
                chunk = inputs[start:start + MAX_CHUNK]
                last = start + len(chunk) == len(inputs)
                want = last and item.sampling is not None
                last_T = len(chunk)
                lastT = last_T
                row = self._forward(state, chunk, pos0 + start, want_logits=want)
                if self.mtp_on:
                    # Preserve the target's final normed hidden for this chunk,
                    # THEN warm the drafter over it: the drafter's own layer
                    # writes the same `h` arena, and the post-step draft reads
                    # this copy. `hkeep` therefore holds the LAST chunk's rows,
                    # which is what the post-step draft's h0_row indexes into.
                    self.dev.dtod(self._view(self._p("hkeep"), last_T * HIDDEN * 2),
                                  self._view(self._p("h"), last_T * HIDDEN * 2),
                                  last_T * HIDDEN * 2)
                    self._mtp_warm(state, chunk, pos0 + start)
                # the conv state ping-pongs per SUB-CHUNK, not per step: a
                # 4096-token prefill arrives as 8 chunks of 512 and each one's
                # history is the one the previous chunk wrote. Toggling once per
                # step leaves chunks 2..8 convolving against the pre-step state,
                # i.e. against nothing for a fresh sequence.
                state.phase ^= 1
                logits = None
                if want and row is not None:
                    self.dev.dtod(state.logits, row, VOCAB * 2)
                    state.has_logits = True
                    logits = state.logits
            state.tokens = state.tokens[:pos0] + inputs
            state.fed += fed_add

        token = None
        finished = False
        if item.sampling is not None:
            if want_round:
                outcome = self._mtp_round(state, item, inputs, pos0, fed_add)
                if outcome:
                    # accepted: the round verified two and queued the second;
                    # rejected: it recovered on its own bookkeeping and queued
                    # the shipped token. Either way one waits in m.out and it
                    # is this step's answer.
                    token = m.out.pop(0)
                else:
                    # Declined mid-flight (_mtp_round returned 0): no forward
                    # happened in the round, so this step runs the shipped
                    # single-token path.
                    self._shipped_decode_step(state, item, inputs, pos0, fed_add)
                    token = self._sample(state.logits, item.sampling,
                                         *self._pen_ctx(state))
                    self._mtp_next_draft(state, pos0, self._p("hkeep"), 0, [token])
            else:
                if logits is None:
                    raise ContractError(
                        "no logits to sample: the step forwarded nothing")
                token = self._sample(logits, item.sampling,
                                     *self._pen_ctx(state))
                if m is not None and inputs:
                    # No round this step, so queue the draft the NEXT one needs.
                    # Under the drafter's contract that is the row at the LAST
                    # forwarded position, E: it reads the target's hidden at E
                    # (the LAST row of the LAST sub-chunk's `h` -- a multi-chunk
                    # prefill's `h` holds only its own final chunk) and the token
                    # at E + 1, which is the token just sampled.
                    self._mtp_next_draft(state, pos0 + len(inputs) - 1,
                                         self._p("hkeep"), lastT - 1, [token])
            finished = token in STOP_TOKENS
            # after an accepted round the state has already consumed what it
            # returned, and the queued token is the next forward's input
            # the DEEPEST queued token is the next forward's input (with
            # MTP_K = 2 a round can queue two)
            state.pending = m.out[-1] if (m is not None and m.out) else token
        return StepResult(token, finished, forwarded_tokens=len(inputs))

    def _shipped_decode_step(self, state, item, inputs, pos0, fed_add) -> None:
        """The shipped single-token step: one T = 1 forward at `pos0`, the
        sampled row kept on the state, the tokens/conv-phase bookkeeping, and the
        target's final hidden preserved for the drafter.

        This is what a step looks like with speculation OFF, and it is what a
        REJECTED round falls back to, so the rejected stream is the shipped one
        rather than a re-derivation of it.
        """
        self._grow(state, pos0 + len(inputs))
        row = self._forward(state, inputs, pos0, want_logits=True)
        self.dev.dtod(self._view(self._p("hkeep"), HIDDEN * 2),
                      self._view(self._p("h"), HIDDEN * 2), HIDDEN * 2)
        state.phase ^= 1
        self.dev.dtod(state.logits, row, VOCAB * 2)
        state.has_logits = True
        state.tokens = state.tokens[:pos0] + inputs
        state.fed += fed_add

    # -- the speculative round ----------------------------------------------

    def _mtp_round(self, state, item, inputs, pos0, fed_add) -> int:
        """One K = 1 speculative round in place of the single-token sample.

        `pos0 = P` is the position of the step's own token `p`, and `d =
        state.mtp.draft` is the drafter's token for `P + 1` (already in hand: it
        was produced at the end of the previous step, the only moment its inputs
        -- the target's h_{P-1} and the token at P -- both existed).

        ONE target forward of `[p, d]` at P, T = 2. Row r's logits are position
        P + r, so they sample the token at P + r + 1:

          row 0  ->  t0, the token at P + 1      (the position d claims)
          row 1  ->  t1, the token at P + 2

        ACCEPT iff t0 == d. Then the state covers 0..P+1 with (p, d == t0), t0
        and t1 are both the baseline's tokens, and the forward at P + 1 that the
        baseline would have done never happens: this step returns t0, t1 waits
        in the queue and is ALSO the next forward's input. TWO tokens for ONE
        target forward.

        REJECT: the round undoes row 1 and nothing else. Row 0 of the verify IS
        the shipped forward (the paragraph below), so the fallback keeps it:
        the GDN recurrence falls back to the mid-state `gdn_scan` checkpoints
        after row 0, the conv mid-state is already live (row 0 rotated the ring
        in place and the phase never flipped), and the token is the first draw
        over row 0's kept logits -- the same draw, and the same value, the
        shipped re-forward would make. Only t1's draw is thrown away. So a
        rejection is bit-exact by construction AND costs no second forward.

        Row 0 of the verify is, in fact, BIT-IDENTICAL to that T = 1 forward:
        the verify takes the T = 1 k-split with the T = 2 token group, so its
        fp32 partials are the same COUNT in the same order, `gdn_scan` walks t
        in order, `conv1d_causal` derives E[i] from the same frames, and the
        attention IS the decode path row for row (MTP_SPLIT_MAX).
        `_build/bench_mtp_exact.py` measures that -- 0 of 248077 logits differ --
        and `shape_T=2` is in the same table for contrast, because dropping the
        T = 1 k-split is what breaks it (231348 of 248077 differ).

        Returns 0 (not applicable), 1 (accepted), 2 (rejected).
        """
        m = state.mtp
        spec = item.sampling
        if spec is None or not inputs or pos0 + 2 > MAX_CONTEXT \
                or m.cap < pos0 + 2 or m.mtp_len < pos0 or m.draft is None:
            return 0
        if m.draft2 is not None and pos0 + 3 <= MAX_CONTEXT and m.cap >= pos0 + 3:
            return self._mtp_round2(state, item, inputs, pos0, fed_add)
        d = m.draft
        rng = self._rng(spec.seed)
        st0 = rng.getstate()
        row = self._forward(state, [inputs[0], d], pos0, want_logits=True,
                            all_rows=True, shape_T=MTP_SHAPE_T1, verify=True)
        self.mtp_rounds += 1
        p_ids, o_ids = self._pen_ctx(state)
        # the step's own token `p` was taken out of `pending` and is not yet in
        # `state.tokens`, so the split above cannot see it: add it by hand or
        # the penalty stack scores these rows against a context missing the
        # token right before them. The shipped step's sample DOES see it (it
        # scores after absorbing), so without this an accepted round emits a
        # token the baseline would not have -- penalties on, streams no longer
        # the baseline's.
        out0 = o_ids + [inputs[0]]
        t0 = self._sample(self._view(row.ptr, VOCAB * 2), spec, p_ids, out0)
        # vLLM's rejection sampler scores the later draft rows against an
        # output that already includes the earlier ones of the round
        t1 = self._sample(self._view(row.ptr + VOCAB * 2, VOCAB * 2), spec,
                          p_ids, out0 + [t0])
        h = self._p("hkeep")        # row r below is position pos0 + r
        # preserve the verify's two hidden rows -- the drafter is about to write
        # over the arena they are in
        self.dev.dtod(self._view(h, HIDDEN * 2 * 2),
                      self._view(self._p("h"), HIDDEN * 2 * 2), HIDDEN * 2 * 2)
        if t0 == d:
            self.mtp_accept += 1
            state.tokens = state.tokens[:pos0] + [inputs[0], d]
            state.fed += fed_add
            state.phase ^= 1      # the verify wrote conv[phase ^ 1]
            # Two drafter rows, one T = 2 call. The row at pos0 is the drafter's
            # GAP -- nothing has written its KV yet, and the row at pos0+1 cannot
            # attend over a hole -- so it is filled here from the verify's row 0
            # and the accepted token; the row at pos0+1, from row 1 and t1, is
            # the one whose PREDICTION the next round (at pos0+2) verifies.
            self._mtp_next_draft(state, pos0, h, 0, [d, t1])
            m.out = [t0, t1]
            return 1
        # Reject. The recurrence falls back to its post-row-0 checkpoint (the
        # conv mid-state is already live: row 0 rotated the ring in place and
        # the phase never flipped), and the first draw is re-made over row 0's
        # kept logits -- t0's value again, one draw off the baseline's stream.
        # What is undone is row 1's absorption and t1's draw. The drafter's row
        # at pos0 stands (built from h_{pos0-1} and p, both kept); the row at
        # pos0+1 is rewound and the one re-draft below rebuilds from pos0, as
        # the shipped recovery would have.
        self.dev.dtod(state.s, self.snap_s, GDN_S_BYTES)
        rng.setstate(st0)     # BOTH draws go: the re-sample makes the first
        self.mtp_reject += 1
        state.tokens = state.tokens[:pos0] + [inputs[0]]
        state.fed += fed_add
        token = self._sample(self._view(row.ptr, VOCAB * 2), spec,
                             *self._pen_ctx(state))
        self.dev.dtod(state.logits, self._view(row.ptr, VOCAB * 2), VOCAB * 2)
        state.has_logits = True
        m.out = [token]
        m.draft = m.draft2 = None
        m.mtp_len = min(m.mtp_len, pos0)
        self._mtp_next_draft(state, pos0, self._p("hkeep"), 0, [token])
        return 2

    def _mtp_round2(self, state, item, inputs, pos0, fed_add) -> int:
        """The MTP_K = 2 round: ONE T = 3 forward of [p, d1, d2] at P, rows r =
        0..2 sampling t_r (the token at P + r + 1) in the baseline's order and
        penalty context, then the longest accepted prefix j:

          j = 2  t0 == d1 and t1 == d2: t0, t1, t2 out; everything kept. The
                 conv state after row 2 is in the spare buffer (`_gdn`) and is
                 adopted into conv[phase]; conv[phase ^ 1] keeps the state after
                 row 1 for `_mtp_unforward`.
          j = 1  t0 == d1 only: t0, t1 out; the recurrence falls back to its
                 checkpoint after row 1, the conv state after row 1 is the other
                 phase buffer, t2's draw is undone.
          j = 0  the K = 1 reject: checkpoint after row 0, t0 re-drawn.

        A stop token caps j where it appears, so a finished stream never leaves
        a token forwarded past its end. Every kept row is bit-identical to the
        baseline's T = 1 forward at that position (the T = 1 k-split, a token
        group of 3, the decode attention row by row), so the streams are the
        baseline's at every level. Returns 1 (accepted 1 or 2) or 2 (rejected).
        """
        m = state.mtp
        spec = item.sampling
        p, d1, d2 = inputs[0], m.draft, m.draft2
        self._grow(state, pos0 + 4)
        rng = self._rng(spec.seed)
        st0 = rng.getstate()
        row = self._forward(state, [p, d1, d2], pos0, want_logits=True,
                            all_rows=True, shape_T=MTP_SHAPE_T1, verify=True)
        self.mtp_rounds += 1
        p_ids, o_ids = self._pen_ctx(state)
        out0 = o_ids + [p]
        t0 = self._sample(self._view(row.ptr, VOCAB * 2), spec, p_ids, out0)
        t1 = self._sample(self._view(row.ptr + VOCAB * 2, VOCAB * 2), spec,
                          p_ids, out0 + [t0])
        st2 = rng.getstate()
        t2 = self._sample(self._view(row.ptr + 2 * VOCAB * 2, VOCAB * 2), spec,
                          p_ids, out0 + [t0, t1])
        h = self._p("hkeep")
        self.dev.dtod(self._view(h, HIDDEN * 2 * 3),
                      self._view(self._p("h"), HIDDEN * 2 * 3), HIDDEN * 2 * 3)
        j = 0 if (t0 != d1 or t0 in STOP_TOKENS) else \
            1 if (t1 != d2 or t1 in STOP_TOKENS) else 2
        if j == 2:
            self.mtp_accept += 1
            self.mtp_accept2 += 1
            state.tokens = state.tokens[:pos0] + [p, d1, d2]
            state.fed += fed_add
            self.dev.dtod(state.conv[state.phase], self.snap_c[0], GDN_CONV_BYTES)
            m.rb_tag = self.verify_tag
            self._mtp_next_draft(state, pos0, h, 0, [d1, d2, t2])
            m.out = [t0, t1, t2]
            return 1
        if j == 1:
            self.mtp_accept += 1
            self.dev.dtod(state.s, self._view(self.snap_s.ptr + GDN_S_BYTES, GDN_S_BYTES),
                          GDN_S_BYTES)
            rng.setstate(st2)
            state.tokens = state.tokens[:pos0] + [p, d1]
            state.fed += fed_add
            state.phase ^= 1
            m.mtp_len = min(m.mtp_len, pos0)
            self._mtp_next_draft(state, pos0, h, 0, [d1, t1])
            m.out = [t0, t1]
            return 1
        self.dev.dtod(state.s, self.snap_s, GDN_S_BYTES)
        rng.setstate(st0)
        self.mtp_reject += 1
        state.tokens = state.tokens[:pos0] + [p]
        state.fed += fed_add
        token = self._sample(self._view(row.ptr, VOCAB * 2), spec,
                             *self._pen_ctx(state))
        self.dev.dtod(state.logits, self._view(row.ptr, VOCAB * 2), VOCAB * 2)
        state.has_logits = True
        m.out = [token]
        m.draft = m.draft2 = None
        m.mtp_len = min(m.mtp_len, pos0)
        self._mtp_next_draft(state, pos0, h, 0, [token])
        return 2

    def _mtp_unforward(self, state) -> bool:
        """Undo the one row a level-2 round forwarded ahead of the engine.

        After j = 2 the state covers [.., p, d1, d2] while the engine has been
        handed only t0: d2 (= t1) is forwarded but unhanded as long as two
        tokens wait in the queue. A cache entry or a new prompt must not see it,
        and re-forwarding the whole prefix (`truncate_state`'s replay) would be
        the price. While no verify has run since (`verify_tag`), the round's
        checkpoint after row 1 IS the state without it: restore it, take the
        conv state after row 1 (the other phase buffer), and make d2 the pending
        token -- the same logical sequence, one row less forwarded.
        """
        m = state.mtp
        if m is None or len(m.out) < 2 or m.rb_tag != self.verify_tag \
                or self.snap_s is None or not state.tokens:
            return False
        self.dev.dtod(state.s, self._view(self.snap_s.ptr + GDN_S_BYTES, GDN_S_BYTES),
                      GDN_S_BYTES)
        state.phase ^= 1
        state.pending = state.tokens[-1]
        state.tokens = state.tokens[:-1]
        state.fed = min(state.fed, len(state.tokens))
        m.out = []
        m.draft = m.draft2 = None
        m.mtp_len = min(m.mtp_len, len(state.tokens))
        m.rb_tag = -1
        return True

    # -- the forward pass ---------------------------------------------------

    def _launch(self, fn, grid, block, args, shared=0):
        self.dev.launch(fn, grid, block, args, shared=shared)

    def _gemv_shape(self, module: str, out_dim: int, T: int) -> tuple[int, int]:
        """(WT, S) for the wide GEMV, or (0, 0) for the one-tile kernel.

        WT is fixed (EXL3_WT); S is the k-split that brings the block count up to
        EXL3_BLOCKS, capped so the fp32 partials stay under EXL3_PT_SHARE of the
        trellis this call reads and fit the arena. Both are measured numbers, not
        derived ones -- see the constants' note at the head of this file.
        """
        if T > EXL3_WIDE_T or T <= 0:
            return 0, 0
        if T == 1 and DEC_SPLIT_ON:
            s = DEC_SPLIT.get(module.rsplit(".", 1)[-1])
            if s:
                return EXL3_WT, s
        in_dim, _ = self._dims(module)
        kin, nto = in_dim // 16, out_dim // 16
        bx = -(-nto // (8 * EXL3_WT))                 # blocks before the split
        cap = int(EXL3_PT_SHARE * kin * self.bits[module] / (4.0 * T))
        s = 1
        while s * 2 <= EXL3_BLOCKS // max(bx, 1) and s * 2 <= min(cap, kin):
            s *= 2
        if s * T * out_dim * 4 > EXL3_PT_BYTES:
            return 0, 0
        return EXL3_WT, s

    def _gemv_m(self, out_dim: int, T: int) -> tuple[int, str]:
        """(tokens per block, entry name) for the prefill GEMV at this T.

        m = 8 is the general entry's largest token group and is what prefill has
        run since the fused GEMV landed. `exl3_gemv_pre` adds m = 16, which
        divides the number of full passes over a module's trellis by two; the
        walk's cost is per-warp k-step LATENCY and not bandwidth, so that pass
        count is what a prefill chunk's GEMV time is made of. Measured paired in
        one process at T = 512 (_build/bench_prefill_m.py), m = 16 against
        m = 8: gate_proj 1.229, down_proj 1.121, in_proj_qkv 1.155, k_proj 1.279
        -- and bit-identical output at every m and every T (the group size only
        changes WHICH block computes a (token, column) pair).

        m = 16 halves the block count, so it is only taken when the launch still
        fills the card -- two conditions, both from a measurement:

          * at least two token groups (T >= 2 * m). At T = 16 the prefill chunk
            has one y-block and the whole model is block-starved: measured
            paired in one process, the 16-token chunk is 0.82x with m = 16
            (_build/bench_prefill_pair.py, the coolest of four reps), because
            the only parallelism a 16-token chunk has is the module's x grid.
            From T = 32 up, every pairing favours m = 16.
          * at least EXL3_PRE_MIN_BLOCKS blocks in the grid. A module whose
            output axis is short (k_proj and v_proj, 8 blocks wide) holds 64
            blocks at T = 128 and measures 0.88-0.90x there.

        m = 32 is taken on top of that, and only WITH the shared-memory stage
        (EXL3_SM_BYTES > 0): the decode is per-warp-k-step and does not depend on
        the group size, so doubling the group halves the decode per token, but
        WITHOUT the stage the 2*m accumulators do not fit beside the walk's
        registers and m = 32 is a large loss at every module (measured today at
        T = 512, gate_proj: 87.8 ms against m = 16's 21.7 = 5.2x SLOWER). With
        the stage freeing the per-token x addressing it fits, and it measures
        (paired, one process, T = 512, against the staged m = 16):

          gate_proj 1.28x  up_proj 1.28x  down_proj 1.26x  in_proj_qkv 1.21x
          in_proj_z 1.26x  out_proj 1.26x  k_proj 1.03x    v_proj 1.10x

        k_proj and v_proj are 8 x-blocks wide, so m = 32 leaves them 128 blocks
        -- 1.1 waves of the 112 resident -- and they barely gain. They keep
        m = 16 behind EXL3_PRE_M32_MIN_BLOCKS; everything else takes m = 32.

        Modules that fail every condition keep m = 8. m = 64 exists in the kernel
        behind -D and is still a measured loss (2*64 accumulators do not fit at
        any occupancy); see the entry's note.
        """
        for m_pre, min_blocks in ((EXL3_M_PRE, EXL3_PRE_M32_MIN_BLOCKS),
                                  (EXL3_M_PRE16, EXL3_PRE_MIN_BLOCKS)):
            if T >= 2 * m_pre and \
                    (out_dim // 128) * -(-T // m_pre) >= min_blocks:
                if EXL3_PRE_X32:
                    # the fp32-input entry is the measured dead end it is, and it
                    # is instantiated to m = 16 only -- asking it for 32 would
                    # write nothing (a silent geometry guard, the failure the
                    # dispatch's `default: break` makes possible).
                    return min(m_pre, EXL3_M_PRE16), "exl3_gemv_pre32"
                return m_pre, "exl3_gemv_pre"
        if T == 3:
            # one exact group of 3 (the K = 2 verify): m = 2 would walk the
            # trellis twice and m = 4 carry a dead token. The wide entries take
            # it; the one-tile entry rounds it up to 4 (see _gemv).
            return 3, "exl3_gemv"
        m = 1
        while m * 2 <= T and m < 8:
            m *= 2
        return m, "exl3_gemv"

    def _gemv(self, x_ptr, module, out_ptr, T, shape_T: int = 0, x_up: int = 0):
        """One quantized projection: suh-rotate, trellis GEMV, svh-rotate.

        Three launches (four when the k axis is split), none of which materialises
        the weight. `m` groups tokens per block so the decode (the expensive half)
        is amortised across a prefill chunk; the general kernel accepts 1, 2, 4
        or 8 and the prefill entry 1..16 (`_gemv_m` picks between them).
        A small T takes the wide entry instead -- WT tiles per warp and S k-slices
        -- which is what gives the walk its memory-level parallelism; see
        EXL3_WT's note for the measured table behind both constants.

        `shape_T` overrides only the CHOICE of WT and S, never the launch's T or
        m. It exists for the speculative verify (see MTP_SHAPE_T1): at T = 2 the
        k-split cap is `EXL3_PT_SHARE * kin * bits / (4*T)`, i.e. HALF the T = 1
        split, so taking T = 2's own shape would sum half as many fp32 partials
        in a different order and halve the block count for the same bytes. Pinning
        S to the T = 1 value keeps the block count, keeps the partial count (and
        so the summation order) the T = 1 path's, and lets `m = 2` carry both
        tokens in one trellis pass -- exact AND faster.
        """
        in_dim, out_dim = self._dims(module)
        m, entry = self._gemv_m(out_dim, T)
        if EXL3_FUSED_PRE and EXL3_AFFINE and EXL3_FUSED_POST:
            # the wide path in ONE launch: pre-rotation, GEMV, split reduce and
            # post-rotation (kernels_exl3.cu's exl3_gemv_w4a1fp / w4afp)
            st = shape_T or T
            wt, s = self._gemv_shape(module, out_dim, st)
            if wt == 4 and out_dim % 512 == 0 and in_dim % 128 == 0:
                while st != T and s > 1 and s * T * out_dim * 4 > EXL3_PT_BYTES:
                    s //= 2
                bx = -(-(out_dim // 16) // 32)
                ty = (T + m - 1) // m
                if bx * ty <= GCNT_SLOTS:
                    kmax = -(-(in_dim // 16) // s)
                    name = "exl3_gemv_w4a1fp" if m == 1 else "exl3_gemv_w4afp"
                    if m == 3 and T == 3 and M3_ENTRY:
                        name = "exl3_gemv_w4a3fp3" if in_dim >= M3_OCC3_MIN_IN else "exl3_gemv_w4a3fp"
                    self._launch(self.kx[name],
                                 (bx, ty, s), (256,),
                                 [x_ptr, x_up, self.ptr(module + ".suh"),
                                  self.ptr(module + ".trellis"),
                                  self._p("pt"), self.ptr(module + ".svh"), out_ptr,
                                  self.gcnt.ptr, in_dim, out_dim, T, self.bits[module], m],
                                 shared=m * kmax * 32)
                    return
        # had128_* indexes rows by blockIdx.y and takes `rows` as the buffer's
        # row count, so a chunk of T tokens is one row each: grid.y = T. The
        # single-load entry takes the fp32 lane-ordered rotation instead -- same
        # values, same order, one buffer per layout.
        single = entry == "exl3_gemv_pre32"
        if x_up and single:
            # not the fused path: materialise silu(gate) * up first
            self._launch(self.k["silu_mul"], ((T * in_dim + 511) // 512,), (256,),
                         [x_ptr, x_up, self._p("sm"), T * in_dim])
            x_ptr, x_up = self._p("sm"), 0
        xbuf = self._p("xr32") if single else self._p("xr")
        nblocks = (in_dim + 1023) // 1024
        if x_up:
            # silu(gate) * up rides in the pre-rotation (had128_pre_silu: the
            # silu_mul values, bit for bit)
            self._launch(self.kx["had128_pre_silu"], (nblocks, T), (256,),
                         [x_ptr, x_up, self.ptr(module + ".suh"), xbuf, in_dim, T])
        else:
            self._launch(self.kx["had128_pre_lc" if single else "had128_pre"],
                         (nblocks, T), (256,),
                         [x_ptr, self.ptr(module + ".suh"), xbuf, in_dim, T])
        st = shape_T or T
        wt, s = self._gemv_shape(module, out_dim, st)
        xp = None if wt or single else self._xp_shape(in_dim, out_dim, T)
        if xp:
            name, bm, s = xp
            self._launch(self.kx[name], (out_dim // 128, (T + bm - 1) // bm, s),
                         (2 * bm if name.startswith("xh_") else 256,),
                         [xbuf, self.ptr(module + ".trellis"), self._p("z"),
                          self._p("pt"), in_dim, out_dim, T, self.bits[module]])
            if s > 1:
                # the split's reduce rides in the post-rotation: the same sum
                # order and rounding as exl3_sk_reduce + had128_post
                npost = (out_dim + 1023) // 1024
                self._launch(self.kx["had128_post_sk"], (npost, T), (256,),
                             [self._p("pt"), s, self.ptr(module + ".svh"), out_ptr,
                              out_dim, T])
                return
        elif wt:
            if st != T:
                # S came from a SMALLER T, so the partials (S*T*out floats) may
                # not fit the arena; give S back until they do. This branch is
                # unreachable whenever shape_T is unset, so the shipped shape
                # choice -- including its own "does not fit -> no split" answer --
                # is untouched.
                while s > 1 and s * T * out_dim * 4 > EXL3_PT_BYTES:
                    s //= 2
            nto = out_dim // 16
            bx = -(-nto // (8 * wt))
            if EXL3_AFFINE and EXL3_FUSED_POST and wt == 4 and out_dim % 512 == 0 \
                    and bx * ((T + m - 1) // m) <= GCNT_SLOTS:
                # GEMV + split reduce + post-rotation, one launch
                self._launch(self.kx["exl3_gemv_w4a1f" if m == 1 else "exl3_gemv_w4af"],
                             (bx, (T + m - 1) // m, s), (256,),
                             [self._p("xr"), self.ptr(module + ".trellis"), self._p("pt"),
                              self.ptr(module + ".svh"), out_ptr, self.gcnt.ptr,
                              in_dim, out_dim, T, self.bits[module], m])
                return
            self._launch(self.kx[self._wide(wt, m)], (bx, (T + m - 1) // m, s),
                         (256,),
                         [self._p("xr"), self.ptr(module + ".trellis"),
                          self._p("z"), self._p("pt"), in_dim, out_dim, T,
                          self.bits[module], m])
            if s > 1:
                # the split's reduce rides in the post-rotation (had128_post_sk)
                npost = (out_dim + 1023) // 1024
                self._launch(self.kx["had128_post_sk"], (npost, T), (256,),
                             [self._p("pt"), s, self.ptr(module + ".svh"), out_ptr,
                              out_dim, T])
                return
        else:
            m = 4 if m == 3 else m          # the one-tile entry has no m = 3
            self._launch(self.kx[entry], (out_dim // 128, (T + m - 1) // m),
                         (256,),
                         [xbuf, self.ptr(module + ".trellis"),
                          self._p("z"), in_dim, out_dim, T, self.bits[module], m])
        npost = (out_dim + 1023) // 1024
        self._launch(self.kx["had128_post"], (npost, T), (256,),
                     [self._p("z"), self.ptr(module + ".svh"), out_ptr, out_dim, T])

    def _gemv_many(self, x_ptr, items, T: int, shape_T: int = 0, ab=None) -> bool:
        """`_gemv` of several modules that read the same input, `items` =
        [(module, out_ptr)]: at T = 1 (and the verify's 3) one exl3_gemv_w4a1fpn
        (w4a3fpn) launch whose blocks
        do exactly what each module's own exl3_gemv_w4a1fp launch does (the same
        k-split, partials and rotations: the same bits); otherwise one `_gemv`
        each. `ab` = (wa, wb, ya, yb, out): gemv_ab_f32's rows ride in the same
        launch; returns whether they did."""
        fm = []
        if (T == 1 or (T == 3 and M3_ENTRY)) and GEMV_N and EXL3_FUSED_PRE and EXL3_AFFINE \
                and EXL3_FUSED_POST:
            in0 = self._dims(items[0][0])[0]
            pt, cnt, kmax = 0, 0, 0
            for module, out_ptr in items:
                in_dim, out_dim = self._dims(module)
                st = shape_T or T
                wt, s = self._gemv_shape(module, out_dim, st)
                while st != T and s > 1 and s * T * out_dim * 4 > EXL3_PT_BYTES:
                    s //= 2                     # as _gemv gives a smaller T's split back
                bx = -(-(out_dim // 16) // 32)
                if in_dim != in0 or wt != 4 or out_dim % 512 or in_dim % 128 \
                        or cnt + bx > GCNT_SLOTS:
                    fm = []
                    break
                fm.append(_Exl3Fm(self.ptr(module + ".suh"), self.ptr(module + ".trellis"),
                                  self.ptr(module + ".svh"), out_ptr, self._p("pt") + pt,
                                  self.gcnt.ptr + 4 * cnt, out_dim, self.bits[module], s))
                pt += s * T * out_dim * 4
                cnt += bx
                kmax = max(kmax, -(-(in_dim // 16) // s))
            if pt > EXL3_PT_BYTES:
                fm = []
        if not fm:
            for module, out_ptr in items:
                self._gemv(x_ptr, module, out_ptr, T, shape_T)
            return False
        nb = sum(-(-(f.out // 16) // 32) * f.S for f in fm)
        abs_ = _Exl3Ab(*ab) if ab is not None else _Exl3Ab(0, 0, 0, 0, 0)
        self._launch(self.kx["exl3_gemv_w4a1fpn" if T == 1 else "exl3_gemv_w4a3fpn"],
                     (nb + abs_.out * T,), (256,),
                     [x_ptr, in0, len(fm)] + fm + [fm[-1]] * (3 - len(fm)) + [abs_],
                     shared=T * kmax * 32)
        return ab is not None

    @staticmethod
    def _wide(wt: int, m: int = 0) -> str:
        """The wide (T <= 4) GEMV entry: the affine-decode one when enabled."""
        if EXL3_AFFINE and wt == 4:
            return "exl3_gemv_w4a1" if (m == 1 and EXL3_W4A1) else "exl3_gemv_w4a"
        return "exl3_gemv_w%d" % wt

    def _xp_shape(self, in_dim: int, out_dim: int, T: int):
        """(entry, BM, S) of the prefill GEMM for this call, or None.

        In fp16 math (XP_HALF) the entry and its k-split come from XP_COST's
        measured cost model, the cheapest within the partials' arena; in fp32
        see XP_MIN_T's note for the measured ranges.
        """
        if XP_HALF and in_dim % 32 == 0 and T >= XS_MIN_T:
            best = None
            for name, bm, occ, a, q, tmin, tmax in XP_COST:
                if not tmin <= T <= tmax:
                    continue
                for s in range(1, min(XS_SPLIT_MAX, in_dim // 32) + 1):
                    if s > 1 and s * T * out_dim * 4 > XP_PT_BYTES:
                        break
                    nb = (out_dim // 128) * -(-T // bm) * s
                    full, m = divmod(nb, XP_SMS * occ)
                    units = full * occ + (max(-(-m // XP_SMS), min(q, occ)) if m else 0)
                    t = a * bm * in_dim / s * units
                    if s > 1:
                        t += XP_SPLIT_COST * s * T * out_dim
                    if best is None or t < best[0]:
                        best = (t, name + ("s" if s > 1 else ""), bm, s)
                    # the `t` entry: the last token tile's padding warps skip
                    # their products (XP_TAIL: the measured cost of such a
                    # tile, c0 + (1 - c0) * live)
                    c0, pen = XP_TAIL.get(name, (None, 1.0))
                    if c0 is not None and T % bm:
                        nt = -(-T // bm)
                        L = T - (nt - 1) * bm
                        live = (-(-min(L, bm // 2) // 8) + -(-max(L - bm // 2, 0) // 8)) * 8 / bm
                        tt = (t - (XP_SPLIT_COST * s * T * out_dim if s > 1 else 0)) \
                            * pen * (nt - 1 + c0 + (1 - c0) * live) / nt \
                            + (XP_SPLIT_COST * s * T * out_dim if s > 1 else 0)
                        if tt < best[0]:
                            best = (tt, name + ("s" if s > 1 else "") + "t", bm, s)
            return best[1:]
        if T >= XP_MIN_T:
            nb = (out_dim // 128) * -(-T // 128)
            if nb >= XP_SPLIT_BLOCKS:
                return "xp_gemm128", 128, 1
            s = max(1, min(4, XP_SLOTS // nb))
            while s > 1 and s * T * out_dim * 4 > XP_PT_BYTES:
                s -= 1
            return ("xp_gemm128s", 128, s) if s > 1 else ("xp_gemm128", 128, 1)
        if XP64_T[0] <= T < XP64_T[1]:
            return "xp_gemm64", 64, 1
        return None

    def _dims(self, module: str) -> tuple[int, int]:
        """(in, out) of a quantized projection, from its trellis shape."""
        shape = self._tensor(module + ".trellis")[3]
        return shape[0] * 16, shape[1] * 16

    def set_speculative(self, enabled: bool = True) -> None:
        """The engine's optional verb: draft ahead, and verify the draft here.

        Everything this costs is paid here and not before, which is the point:
        `open` has already returned, so a run that never asks for speculation
        never uploads the drafter's 212 MB and never claims the 150 MB rewind
        buffer. The switch is idempotent, and turning it back off restores the
        shipped decode path exactly -- `split_max` goes to 1, which is the
        `T == 1` test the shipped runtime used.
        """
        enabled = bool(enabled)
        if enabled == self.mtp_on and (not enabled or self.snap_s is not None):
            return
        if not enabled:
            self.mtp_on = False
            self.split_max = 1
            return
        # The GDN recurrence cannot be rewound, only copied: a T = 2 verify
        # advances it two positions in one call, so a REJECTED round has to put
        # it back. One buffer for the runtime (max_batch is 1, so one step is
        # ever in flight), claimed once rather than per state.
        free0 = self.dev.free_memory()[0]
        if self.snap_s is None:
            # one checkpoint slot per row a verify can fall back to
            self.snap_s = self.dev.alloc(GDN_S_BYTES * max(1, MTP_K))
            self.snap_c = (self.dev.alloc(GDN_CONV_BYTES),
                           self.dev.alloc(GDN_CONV_BYTES))
            self._aux_bufs += [self.snap_s] + list(self.snap_c)
        self.mtp_on = True
        self.split_max = MTP_SPLIT_MAX
        self._load_mtp()
        if self.kv_budget is not None:
            # the drafter and the rollback buffers come out of what states
            # may occupy (the budget was the device's free memory at open)
            self.kv_budget -= max(free0 - self.dev.free_memory()[0], 0)
            self.services.accounting.set_budget(max(self.kv_budget, 0))
            self.log("info", f"kv/state budget: {self.kv_budget / 2**30:.2f} GiB "
                             f"(speculation's buffers taken out)")
        self.log("info", f"speculative decoding on: drafting ahead, "
                         f"split_max {self.split_max}")

    def _check_mtp(self) -> None:
        """The drafter's modules, their bitrate, and their geometry.

        The bitrate is read off the trellis shape (`bits_x2 = shape[2] // 8`)
        because quantization_config.json stops at `lm_head`; getting it wrong
        makes the drafter's output noise, which looks exactly like a drafter
        that does not work, so it is asserted here rather than discovered by a
        low acceptance rate.
        """
        self.mtp_modules = sorted(
            {n.rsplit(".", 1)[0] for n in self.roles
             if n.startswith("mtp.") and n.endswith(".trellis")})
        for module in self.mtp_modules:
            shape = self._tensor(module + ".trellis")[3]
            self.bits[module] = shape[2] // 8
            if self.bits[module] != 8:
                raise _contract_error(
                    f"{module}: trellis bits_x2 {self.bits[module]}, the drafter "
                    f"is 4-bit (shape {shape})")
        for name, want in (("mtp.fc", (10240, 5120)),
                           ("mtp.layers.0.self_attn.q_proj", (HIDDEN, 2 * N_HEADS * HEAD_DIM)),
                           ("mtp.layers.0.self_attn.k_proj", (HIDDEN, N_KV * HEAD_DIM)),
                           ("mtp.layers.0.self_attn.v_proj", (HIDDEN, N_KV * HEAD_DIM)),
                           ("mtp.layers.0.self_attn.o_proj", (N_HEADS * HEAD_DIM, HIDDEN)),
                           ("mtp.layers.0.mlp.gate_proj", (HIDDEN, INTER)),
                           ("mtp.layers.0.mlp.up_proj", (HIDDEN, INTER)),
                           ("mtp.layers.0.mlp.down_proj", (INTER, HIDDEN))):
            if name + ".trellis" not in self.roles:
                raise _contract_error(f"the drafter is missing {name}")
            got = self._dims(name)
            if got != want:
                raise _contract_error(f"{name} is {got}, the drafter wants {want}")
        self.mtp_bytes = sum(self._tensor(m + ".trellis")[1]
                             for m in self.mtp_modules)
        self.log("info", f"MTP drafter: {len(self.mtp_modules)} modules, "
                         f"{self.mtp_bytes / 1e6:.1f} MB, 4-bit")

    def _forward(self, state, tokens: list[int], pos0: int,
                 want_logits: bool, all_rows: bool = False,
                 shape_T: int = 0, verify: bool = False) -> int | None:
        """The target's forward. Returns a pointer to the last row's logits, or
        (with `all_rows`) to a [T, VOCAB] fp16 block whose row r is position
        pos0 + r -- which is what a speculative verify needs, since it has to
        sample at both of its positions.

        `shape_T` is threaded into every `_gemv` (see its note).
        """
        T = len(tokens)
        if verify:
            self.verify_tag += 1        # the checkpoints in snap_s are now this one's
        ids = struct.pack(f"<{T}i", *tokens)
        self.dev.htod(self.w["ids"], ids)
        self._launch(self.k["embed_gather"], (8, T), (256,),
                     [self.ptr(f"{PREFIX}embed_tokens.qweight"),
                      self.ptr(f"{PREFIX}embed_tokens.scales"), self._p("ids"),
                      self._p("x"), HIDDEN])
        # Every residual add is fused with the norm that reads its result
        # (`_add_rms`); only layer 0's input norm stands alone.
        self._rms(self._p("x"), self.ptr(f"{PREFIX}layers.0.input_layernorm.weight"),
                  self._p("h"), HIDDEN, 1, HIDDEN, T)
        for i in range(N_LAYERS):
            p = f"{PREFIX}layers.{i}."
            if i in self.full_ord:
                self._full_attn(state, i, T, pos0, shape_T)
            else:
                self._gdn(state, i, T, pos0, shape_T, verify=verify)
            self._add_rms(self.ptr(p + "post_attention_layernorm.weight"), T)
            self._mlp(p, T, shape_T)
            nxt = (f"{PREFIX}layers.{i + 1}.input_layernorm.weight"
                   if i + 1 < N_LAYERS else f"{PREFIX}norm.weight")
            self._add_rms(self.ptr(nxt), T)
        if not want_logits:
            return None
        # The head. TWO forms, and the shipped one is the default:
        #   the LAST row only (T_head = 1, `all_rows` false) -- what a prefill
        #     and every ordinary decode step do, and what `logits`/`logits2` are
        #     sized for;
        #   EVERY row (T_head = T, `all_rows` true) -- the speculative verify's,
        #     which has to sample at BOTH of its positions. One pass over the
        #     shared 6-bit trellis covers them, with m = T_head tokens per block:
        #     the group size only decides WHICH block computes a (token, column)
        #     pair, so it is bit-identical, and at T_head = 2 it is worth the
        #     954 MB a second pass would have cost.
        Tr = T if all_rows else 1
        last = self._p("h") if all_rows else self._p("h") + (T - 1) * HIDDEN * 2
        return self._head(last, Tr)

    def _head(self, last: int, Tr: int):
        """The lm_head over `Tr` (<= LOGIT_ROWS) final-normed rows at `last`:
        the `logits2` buffer, row r the r-th row's logits."""
        if HEAD_FUSED and EXL3_FUSED_PRE and EXL3_AFFINE and EXL3_FUSED_POST and Tr <= 4:
            # pre-rotation, GEMV and post-rotation in one launch, one token group
            # (the same values as the three launches below: 0.94x their time)
            name = {1: "exl3_gemv_w4a1fp", 3: "exl3_gemv_w4a3fp"}.get(Tr, "exl3_gemv_w4afp")
            self._launch(self.kx[name], (VOCAB // 512, 1, 1), (256,),
                         [last, 0, self.ptr("lm_head.suh"), self.ptr("lm_head.trellis"),
                          self._p("pt"), self.ptr("lm_head.svh"), self._p("logits2"),
                          self.gcnt.ptr, HIDDEN, VOCAB, Tr, self.bits["lm_head"], Tr],
                         shared=Tr * (HIDDEN // 16) * 32)
            return self.w["logits2"]
        self._launch(self.kx["had128_pre"], ((HIDDEN // 128 + 7) // 8, Tr), (256,),
                     [last, self.ptr("lm_head.suh"), self._p("xr"), HIDDEN, Tr])
        # the sampled row goes through the same shape choice as a projection
        wt, s = self._gemv_shape("lm_head", VOCAB, 1)
        mh = min(4, Tr) if wt else min(2, Tr)    # one group: one pass over the head
        out = self._p("logits")
        n = Tr * VOCAB
        if wt:
            self._launch(self.kx[self._wide(wt, mh)],
                         (-(-(VOCAB // 16) // (8 * wt)), (Tr + mh - 1) // mh, s), (256,),
                         [self._p("xr"), self.ptr("lm_head.trellis"),
                          out, self._p("pt"), HIDDEN, VOCAB, Tr,
                          self.bits["lm_head"], mh])
            if s > 1:
                self._launch(self.kx["exl3_sk_reduce"],
                             (min(2048, (n + 255) // 256),), (256,),
                             [self._p("pt"), out, n, s])
        else:
            self._launch(self.kx["exl3_gemv"], (VOCAB // 128, (Tr + mh - 1) // mh),
                         (256,),
                         [self._p("xr"), self.ptr("lm_head.trellis"),
                          out, HIDDEN, VOCAB, Tr, self.bits["lm_head"], mh])
        self._launch(self.kx["had128_post"], ((VOCAB // 128 + 7) // 8, Tr), (256,),
                     [out, self.ptr("lm_head.svh"), self._p("logits2"), VOCAB, Tr])
        return self.w["logits2"]

    def _rms(self, x, w, y, n, rows, pitch, T):
        self._launch(self.k["rmsnorm1p"], (rows, T), (256,), [x, w, y, n, rows, pitch, NORM_EPS])

    def _add_rms(self, w, T: int) -> None:
        """x += p, then h = rmsnorm1p(x, w) in one launch (kernels.cu's
        add_rmsnorm1p: the same values up to the fp32 order of the sum of
        squares)."""
        self._launch(self.k["add_rmsnorm1p"], (T,), (HIDDEN // 8,),
                     [self._p("x"), self._p("p"), w, self._p("h"), HIDDEN, NORM_EPS])

    def _add(self, T: int) -> None:
        """x += p over the whole chunk: the residual add, twice a layer.

        This was launched as ONE block, so 128 of a 512-token chunk's launches
        ran the whole 5.2 MB read + 5.2 MB write on a single SM: measured 2536
        us a call, 325 ms a chunk = 2.8% of its time, at 4.1 GB/s. `add_inplace`
        is a grid-stride loop, so the grid is the caller's to choose and no
        kernel change is needed; ~8 half2 a thread is enough in flight to stream
        it (measured 24 ms a chunk, 14x). The values are elementwise and
        grid-independent, so this is bit-neutral by construction.
        """
        n = T * HIDDEN
        self._launch(self.k["add_inplace"], (-(-(n >> 1) // (256 * EXL3_ADD_PER_THREAD)),),
                     (256,), [self._p("x"), self._p("p"), n])

    def _mlp(self, p: str, T: int, shape_T: int = 0) -> None:
        self._gemv_many(self._p("h"), [(p + "mlp.gate_proj", self._p("mlp")),
                                        (p + "mlp.up_proj", self._p("up"))], T, shape_T)
        # silu(gate) * up rides in down_proj's fused prologue (or is
        # materialised by _gemv when the call takes another path)
        self._gemv(self._p("mlp"), p + "mlp.down_proj", self._p("p"), T, shape_T,
                   x_up=self._p("up"))

    def _full_attn(self, state, i: int, T: int, pos0: int, shape_T: int = 0) -> None:
        p = f"{PREFIX}layers.{i}."
        a = p + "self_attn."
        ord_ = self.full_ord[i]
        kp, vp, lcap = self._kvl(state, ord_)
        self._gemv_many(self._p("h"), [(a + "q_proj", self._p("qg")), (a + "k_proj", self._p("k")),
                                        (a + "v_proj", self._p("v"))], T, shape_T)
        if self.segs is None and ATT_PREP and ATT_DEC and KV8 and T <= self.split_max:
            # decode / verify: q/k norms, rope and the int8 store in one launch,
            # the attention, and the gate in its merge (the same values)
            self._attn_prep(self._p("qg"), a, kp, vp, pos0, lcap, T)
            self._attn_dec(self._p("qg"), kp, vp, self._p("att"), pos0, lcap, T, gate=True)
            self._gemv(self._p("att"), a + "o_proj", self._p("p"), T, shape_T)
            return
        # per-head q/k norms, Gemma-style, 256 wide: q inside the q|gate buffer
        self._rms(self._p("qg"), self.ptr(a + "q_norm.weight"), self._p("qg"),
                  HEAD_DIM, N_HEADS, 2 * HEAD_DIM, T)
        self._rms(self._p("k"), self.ptr(a + "k_norm.weight"), self._p("k"),
                  HEAD_DIM, N_KV, HEAD_DIM, T)
        if self.segs is not None:
            self._full_attn_segs(state, ord_, pos0)
            self._launch(self.k["attn_gate"], ((N_HEADS * HEAD_DIM + 255) // 256, T), (256,),
                         [self._p("att"), self._p("qg"), N_HEADS, HEAD_DIM, HEAD_DIM,
                          2 * HEAD_DIM, HEAD_DIM])
            self._gemv(self._p("att"), a + "o_proj", self._p("p"), T, shape_T)
            return
        self._launch(self.k["rope_partial"],
                     ((N_HEADS + N_KV + 7) // 8, T), (256,),
                     [self._p("qg"), self._p("k"), N_HEADS, N_KV, HEAD_DIM, ROPE_DIM,
                      2 * HEAD_DIM, HEAD_DIM, pos0, self.rope_theta])
        self._launch(self.k["kv_store_q8" if KV8 else "kv_store"], (N_KV, T), (256,),
                     [self._p("k"), self._p("v"), kp, vp,
                      0, pos0, N_KV, HEAD_DIM, lcap])
        # split_max is 1 with speculation off, which makes this the same
        # branch it always was. The MTP build raises it so a T = 2 verify runs
        # the DECODE attention: `attention_split` indexes the token by
        # blockIdx.y and derives its own seqlen as `pos0 + t + 1`, so launched
        # with pos0 at the batch's first position and grid (N_HEADS, T, ATT_SPLIT)
        # row t is bit-identical to the T = 1 decode at position pos0 + t -- and
        # `kv_store` above has already written every row of the batch. The
        # prefill kernel instead tiles 8 queries per block, so T = 2 gives it
        # ceil(2/8) = 1 y-block: 24 blocks against the decode path's 192, on a
        # 56-SM part, for a latency-bound kernel. NO KERNEL EDIT: the split path
        # is already row-general (see MTP_SPLIT_MAX's note).
        if T <= self.split_max and ATT_DEC and KV8:
            self._attn_dec(self._p("qg"), kp, vp, self._p("att"), pos0, lcap, T)
        elif T <= self.split_max:
            if ATT_GQA:
                # all 6 q heads of a kv head per block: k/v rows read once
                self._launch(self.k["attention_split_gqa_q8" if KV8 else "attention_split_gqa"],
                             (N_KV, T, ATT_SPLIT), (256,),
                             [self._p("qg"), kp, vp, self.pacc,
                              self.pm, self.pd,
                              0, N_HEADS, N_KV, HEAD_DIM, 2 * HEAD_DIM, pos0,
                              lcap, ATTN_SCALE, ATT_GQA_ROWS])
            else:
                self._launch(self.k["attention_split"], (N_HEADS, T, ATT_SPLIT), (256,),
                             [self._p("qg"), kp, vp, self.pacc,
                              self.pm, self.pd,
                              0, N_HEADS, N_KV, HEAD_DIM, 2 * HEAD_DIM, pos0, lcap,
                              ATTN_SCALE, ATT_SPLIT_ROWS])
            self._launch(self.k["attention_merge"], (N_HEADS, T, ATT_SPLIT), (256,),
                         [self.pacc, self.pm, self.pd, self._p("att"), N_HEADS, HEAD_DIM,
                          HEAD_DIM])
        elif APF_FH:
            self._attn_fh(self._p("qg"), kp, vp, self._p("att"), pos0, lcap, T)
        elif APF_FA:
            self._launch(self.k["attention_prefill_fa_q8" if KV8 else "attention_prefill_fa"],
                         (N_HEADS, (T + 63) // 64), (256,),
                         [self._p("qg"), kp, vp, self._p("att"), 0,
                          N_HEADS, N_KV, HEAD_DIM, 2 * HEAD_DIM, HEAD_DIM, pos0,
                          lcap, ATTN_SCALE, T])
        else:
            self._launch(self.k["attention_prefill"], (N_HEADS, (T + 7) // 8), (256,),
                         [self._p("qg"), kp, vp, self._p("att"), 0,
                          N_HEADS, N_KV, HEAD_DIM, 2 * HEAD_DIM, HEAD_DIM, pos0,
                          lcap, ATTN_SCALE, T])
        self._launch(self.k["attn_gate"], ((N_HEADS * HEAD_DIM + 255) // 256, T), (256,),
                     [self._p("att"), self._p("qg"), N_HEADS, HEAD_DIM, HEAD_DIM,
                      2 * HEAD_DIM, HEAD_DIM])
        self._gemv(self._p("att"), a + "o_proj", self._p("p"), T, shape_T)

    def _full_attn_segs(self, state, ord_: int, pos0: int) -> None:
        """`_full_attn`'s position-dependent half for a segmented forward
        (`label_logprobs_batch`): each segment is its own sequence continuing
        the SAME prefix at `pos0`. Per segment: rope at pos0.., its K/V into
        positions pos0.. (the previous segment's rows there are dead: its
        attention for this layer has already run), and its queries attended
        causally over the prefix and itself."""
        qrow, krow, arow = N_HEADS * 2 * HEAD_DIM * 2, N_KV * HEAD_DIM * 2, N_HEADS * HEAD_DIM * 2
        kp, vp, lcap = self._kvl(state, ord_)
        for s0, L in self.segs:
            qg, k, v = (self._p("qg") + s0 * qrow, self._p("k") + s0 * krow,
                        self._p("v") + s0 * krow)
            self._launch(self.k["rope_partial"], ((N_HEADS + N_KV + 7) // 8, L), (256,),
                         [qg, k, N_HEADS, N_KV, HEAD_DIM, ROPE_DIM, 2 * HEAD_DIM, HEAD_DIM,
                          pos0, self.rope_theta])
            self._launch(self.k["kv_store_q8" if KV8 else "kv_store"], (N_KV, L), (256,),
                         [k, v, kp, vp, 0, pos0, N_KV, HEAD_DIM, lcap])
            if APF_FH:
                self._attn_fh(qg, kp, vp, self._p("att") + s0 * arow, pos0, lcap, L)
                continue
            self._launch(self.k[("attention_prefill_fa_q8" if KV8 else "attention_prefill_fa")
                                if APF_FA else "attention_prefill"],
                         (N_HEADS, (L + 63) // 64 if APF_FA else (L + 7) // 8), (256,),
                         [qg, kp, vp, self._p("att") + s0 * arow, 0,
                          N_HEADS, N_KV, HEAD_DIM, 2 * HEAD_DIM, HEAD_DIM, pos0,
                          lcap, ATTN_SCALE, L])

    def _attn_dec(self, qg: int, kp: int, vp: int, z: int, pos0: int, lcap: int,
                  T: int, gate: bool = False) -> None:
        """Decode attention for T tokens at pos0.. over a layer's int8 cache
        (attention_dec_q8 + attention_dec_merge); each token's row is the T = 1
        call's at its position. `gate`: attn_gate rides in the merge."""
        nkt = -(-(pos0 + T) // 64)                     # the last token's tiles
        S = min(DEC_SMAX, -(-nkt // DEC_TMIN))        # its split is the widest
        self._launch(self.k["attention_dec_q8"], (N_KV, T, S), (256,),
                     [qg, kp, vp, self.pacc, self.pm, self.pd, N_HEADS, N_KV,
                      2 * HEAD_DIM, pos0, lcap, ATTN_SCALE * LOG2E, DEC_TMIN, DEC_SMAX])
        self._launch(self.k["attention_dec_merge_gate" if gate else "attention_dec_merge"],
                     (T, N_HEADS), (HEAD_DIM,),
                     [self.pacc, self.pm, self.pd, z, N_HEADS, HEAD_DIM, pos0,
                      DEC_TMIN, DEC_SMAX] + ([qg, 2 * HEAD_DIM, HEAD_DIM] if gate else []))

    def _attn_prep(self, qg: int, a: str, kp: int, vp: int, pos0: int, lcap: int,
                   T: int) -> None:
        """q_norm, k_norm, rope and kv_store_q8 for T tokens as one launch
        (kernels.cu's attn_prep_q8), `a` the attention module's prefix."""
        self._launch(self.k["attn_prep_q8"], (N_HEADS + N_KV, T), (HEAD_DIM,),
                     [qg, self._p("k"), self._p("v"), self.ptr(a + "q_norm.weight"),
                      self.ptr(a + "k_norm.weight"), kp, vp, N_HEADS, N_KV, HEAD_DIM,
                      2 * HEAD_DIM, pos0, lcap, ROPE_DIM, self.rope_theta, NORM_EPS])

    @staticmethod
    def _fh_split(T: int, pos0: int) -> int:
        """attention_prefill_fh's kv split: the count whose blocks fill whole
        waves of the card (one block an SM) for the fewest tiles a block, one
        tile of per-block overhead each; capped by the `pt` arena it borrows."""
        b0 = N_KV * -(-T * (N_HEADS // N_KV) // FH_ROWS)
        nkt = -(-(pos0 + T) // FH_ROWS)
        cap = min(FH_SPLIT_MAX, nkt, XP_PT_BYTES // (T * N_HEADS * (HEAD_DIM * 4 + 8)))
        return min(range(1, max(cap, 1) + 1),
                   key=lambda S: -(-b0 * S // XP_SMS) * (-(-nkt // S) + 1))

    def _attn_fh(self, qg: int, kp: int, vp: int, z: int, pos0: int, lcap: int, T: int,
                 q8: bool | None = None) -> None:
        """Causal prefill attention over a layer's cache (kernels.cu's
        attention_prefill_fh): T queries at pos0.., z [T, N_HEADS, HEAD_DIM];
        the cache in the package's KV format unless `q8` says otherwise."""
        q8 = KV8 if q8 is None else q8
        S = self._fh_split(T, pos0)
        pacc = self._p("pt")
        pm = pacc + T * N_HEADS * S * HEAD_DIM * 4
        pd = pm + T * N_HEADS * S * 4
        self._launch(self.k["attention_prefill_fh_q8" if q8 else "attention_prefill_fh"],
                     (N_KV, -(-T * (N_HEADS // N_KV) // FH_ROWS), S), (256,),
                     [qg, kp, vp, z, pacc, pm, pd, N_HEADS, N_KV, HEAD_DIM, 2 * HEAD_DIM,
                      HEAD_DIM, pos0, lcap, ATTN_SCALE * LOG2E, T])
        if S > 1:
            self._launch(self.k["attention_merge_fh"], (T, N_HEADS), (HEAD_DIM,),
                         [pacc, pm, pd, z, N_HEADS, HEAD_DIM, HEAD_DIM, S])

    def _gdn(self, state, i: int, T: int, pos0: int, shape_T: int = 0,
             verify: bool = False) -> None:
        p = f"{PREFIX}layers.{i}."
        a = p + "linear_attn."
        g = self.gdn_ord[i]
        s_in = state.conv[state.phase]
        s_out = state.conv[state.phase ^ 1]
        # the two unquantized fp16 projections, fp32 out (`gdn_scalars` wants
        # fp32), ride in the qkv / z launch at decode shapes
        ab = (self.ptr(a + "in_proj_a.weight"), self.ptr(a + "in_proj_b.weight"),
              self._p("ab"), self._p("ab") + T * 48 * 4, 48) if GEMV_AB else None
        done = self._gemv_many(self._p("h"), [(a + "in_proj_qkv", self._p("qkv")),
                                              (a + "in_proj_z", self._p("zn"))], T, shape_T,
                               ab=ab)
        # otherwise their own launch; a prefill chunk as one tiled GEMM
        # (weights read once a 32 tokens)
        if done:
            pass
        elif T >= GAB_MIN_T:
            self._launch(self.kx["gemm_ab_f32"], (-(-T // 32), 2), (256,),
                         [self._p("h"), self.ptr(a + "in_proj_a.weight"),
                          self.ptr(a + "in_proj_b.weight"), self._p("ab"),
                          self._p("ab") + T * 48 * 4, HIDDEN, 48, T])
        else:
            self._launch(self.kx["gemv_ab_f32"], (2 * 48, T), (128,),
                         [self._p("h"), self.ptr(a + "in_proj_a.weight"),
                          self.ptr(a + "in_proj_b.weight"), self._p("ab"),
                          self._p("ab") + T * 48 * 4, HIDDEN, 48])
        # depthwise causal conv over the 10240-wide q|k|v row; a decode step's
        # rides in the scan (gdn_scan_rows_f's gdn_conv1)
        conv_w = self.ptr(a + "conv1d.weight")
        conv_in_scan = (T == 1 and not verify and self.segs is None and GDN_CONV_F
                        and GDN_ROWS_F and GDN_FUSED)
        base = g * 3 * QKV_ROWS * 2
        # The three CONV_SEGS are contiguous in every operand (rows, weights at
        # d * CONV_WIDTH, state at d * (CONV_WIDTH - 1)), so the whole 10240-wide
        # row is ONE launch -- each channel is independent, the values are the
        # per-segment calls' exactly.
        if self.segs is not None:
            # each segment continues the prefix's conv state (read only); the
            # state it would leave goes to scratch
            for s0, L in self.segs:
                self._launch(self.k["conv1d_causal"], ((QKV_ROWS + 255) // 256, L), (256,),
                             [self._p("qkv") + s0 * QKV_ROWS * 2, conv_w, s_in.ptr + base,
                              self._p("c") + s0 * QKV_ROWS * 2, self.lab_conv.ptr + base,
                              QKV_ROWS, L, pos0, QKV_ROWS, CONV_WIDTH])
        elif verify:
            # A verify wants the state after every row but the last too (a
            # rejection keeps one of them), so the rows go as T = 1 calls: row 0
            # rotates the ring IN PLACE -- the mid-state lands where the live
            # state already is, so a rejection keeps it by flipping nothing --
            # row 1 lands in the other phase buffer, exactly where one T = 2
            # call would leave it, and a third row (MTP_K = 2) goes to the spare
            # `snap_c[0]`, which the round adopts only if it keeps that row. The
            # rows are independent (row t's conv window takes its newest frame
            # from the qkv buffer, not from the ring), so this is the same math
            # in the same order.
            dsts = (s_in, s_out, self.snap_c[0])
            if CONV_ROWS and T <= 3:
                # the same rows in one launch (kernels.cu's conv1d_causal_rows)
                self._launch(self.k["conv1d_causal_rows"], ((QKV_ROWS + 255) // 256,), (256,),
                             [self._p("qkv"), conv_w, s_in.ptr + base, self._p("c")]
                             + [b.ptr + base for b in dsts] + [QKV_ROWS, T, pos0, QKV_ROWS])
            else:
                src = s_in
                for t, dst in enumerate(dsts[:T]):
                    self._launch(self.k["conv1d_causal"], ((QKV_ROWS + 255) // 256, 1), (256,),
                                 [self._p("qkv") + t * QKV_ROWS * 2, conv_w, src.ptr + base,
                                  self._p("c") + t * QKV_ROWS * 2, dst.ptr + base, QKV_ROWS, 1,
                                  pos0 + t, QKV_ROWS, CONV_WIDTH])
                    src = dst
        elif not conv_in_scan:
            self._launch(self.k["conv1d_causal"], ((QKV_ROWS + 255) // 256, T), (256,),
                         [self._p("qkv"), conv_w, s_in.ptr + base, self._p("c"),
                          s_out.ptr + base, QKV_ROWS, T, pos0, QKV_ROWS, CONV_WIDTH])
        ck = (self.snap_s.ptr + g * GDN_V_HEADS * GDN_V * GDN_K * 4) if verify else 0
        if T <= 3 and GDN_FUSED and self.segs is None:
            # decode / verify: the q/k L2 norm and beta/decay inside the scan
            # (kernels.cu's gdn_scan_f, the same values as the three launches)
            if GDN_ROWS_F:
                # the same, its rows over 4x the blocks (kernels.cu's gdn_scan_rows_f)
                self._launch(self.k["gdn_scan_rows_f"], (GDN_V_HEADS, GDN_V // 32), (256,),
                             [self._p("c"), self._p("c") + 2048 * 2, self._p("c") + 4096 * 2,
                              state.s.ptr + g * GDN_V_HEADS * GDN_V * GDN_K * 4,
                              self._p("y"), T, GDN_K_HEADS, GDN_V_HEADS, GDN_K, GDN_V,
                              QKV_ROWS, QKV_ROWS, GDN_V_HEADS * GDN_V, Q_SCALE, GDN_GROUP,
                              ck, GDN_S_BYTES // 4, self._p("ab"), self._p("ab") + T * 48 * 4,
                              self.ptr(a + "A_log"), self.ptr(a + "dt_bias"), 48, L2_EPS]
                             + ([self._p("qkv"), conv_w, s_in.ptr + base, s_out.ptr + base, pos0]
                                if conv_in_scan else [0, 0, 0, 0, 0]))
            else:
                self._launch(self.k["gdn_scan_f"], (GDN_V_HEADS,), (256,),
                         [self._p("c"), self._p("c") + 2048 * 2,
                          self._p("c") + 4096 * 2, 0, 0,
                          state.s.ptr + g * GDN_V_HEADS * GDN_V * GDN_K * 4,
                          self._p("y"), T, GDN_K_HEADS, GDN_V_HEADS, GDN_K, GDN_V,
                          QKV_ROWS, QKV_ROWS, GDN_V_HEADS * GDN_V, Q_SCALE, GDN_GROUP,
                          ck, GDN_S_BYTES // 4, self._p("ab"), self._p("ab") + T * 48 * 4,
                          self.ptr(a + "A_log"), self.ptr(a + "dt_bias"), 48, L2_EPS])
            self._launch(self.k["rmsnorm_gated"], (GDN_V_HEADS, T), (128,),
                         [self._p("y"), self._p("zn"), self.ptr(a + "norm.weight"),
                          self._p("yn"), GDN_V, GDN_V_HEADS, GDN_V, GDN_NORM_EPS])
            self._gemv(self._p("yn"), a + "out_proj", self._p("p"), T, shape_T)
            return
        # the L2 norm runs over the whole 10240-wide row as 80 head-sized rows;
        # only the q (0..15) and k (16..31) rows are read back (see the header).
        self._launch(self.k["l2norm_scaled"], (80, T), (128,),
                     [self._p("c"), self._p("ln"), GDN_K, 80, GDN_K, L2_EPS, 1.0])
        self._launch(self.k["gdn_scalars"], (1, T), (256,),
                     [self._p("ab"), self._p("ab") + T * 48 * 4,
                      self.ptr(a + "A_log"), self.ptr(a + "dt_bias"),
                      self._p("bd"), self._p("bd") + T * 48 * 4, T, GDN_V_HEADS,
                      48, 48])
        if self.segs is not None:
            # each segment scans from a COPY of the prefix's recurrent state
            layer_s = GDN_V_HEADS * GDN_V * GDN_K * 4
            for s0, L in self.segs:
                self.dev.dtod(self.lab_s, self._view(state.s.ptr + g * layer_s, layer_s),
                              layer_s)
                self._launch(self.k["gdn_scan_rows" if GDN_ROWS else "gdn_scan"],
                             (GDN_V_HEADS, GDN_V // 32) if GDN_ROWS else (GDN_V_HEADS,), (256,),
                             [self._p("ln") + s0 * QKV_ROWS * 2,
                              self._p("ln") + 16 * GDN_K * 2 + s0 * QKV_ROWS * 2,
                              self._p("c") + 4096 * 2 + s0 * QKV_ROWS * 2,
                              self._p("bd") + s0 * 48 * 4,
                              self._p("bd") + T * 48 * 4 + s0 * 48 * 4,
                              self.lab_s.ptr, self._p("y") + s0 * GDN_V_HEADS * GDN_V * 2,
                              L, GDN_K_HEADS, GDN_V_HEADS, GDN_K, GDN_V,
                              QKV_ROWS, QKV_ROWS, GDN_V_HEADS * GDN_V, Q_SCALE, GDN_GROUP]
                             + ([] if GDN_ROWS else [0, 0]))
        elif GDN_ROWS and not ck:
            self._launch(self.k["gdn_scan_rows"], (GDN_V_HEADS, GDN_V // 32), (256,),
                         [self._p("ln"), self._p("ln") + 16 * GDN_K * 2,
                          self._p("c") + 4096 * 2, self._p("bd"), self._p("bd") + T * 48 * 4,
                          state.s.ptr + g * GDN_V_HEADS * GDN_V * GDN_K * 4,
                          self._p("y"), T, GDN_K_HEADS, GDN_V_HEADS, GDN_K, GDN_V,
                          QKV_ROWS, QKV_ROWS, GDN_V_HEADS * GDN_V, Q_SCALE, GDN_GROUP])
        else:
            self._launch(self.k["gdn_scan"], (GDN_V_HEADS,), (256,),
                         [self._p("ln"), self._p("ln") + 16 * GDN_K * 2,
                          self._p("c") + 4096 * 2, self._p("bd"), self._p("bd") + T * 48 * 4,
                          state.s.ptr + g * GDN_V_HEADS * GDN_V * GDN_K * 4,
                          self._p("y"), T, GDN_K_HEADS, GDN_V_HEADS, GDN_K, GDN_V,
                          QKV_ROWS, QKV_ROWS, GDN_V_HEADS * GDN_V, Q_SCALE, GDN_GROUP, ck,
                          GDN_S_BYTES // 4])
        self._launch(self.k["rmsnorm_gated"], (GDN_V_HEADS, T), (128,),
                     [self._p("y"), self._p("zn"), self.ptr(a + "norm.weight"),
                      self._p("yn"), GDN_V, GDN_V_HEADS, GDN_V, GDN_NORM_EPS])
        self._gemv(self._p("yn"), a + "out_proj", self._p("p"), T, shape_T)

    # -- the MTP drafter ----------------------------------------------------

    MTP = "mtp."
    MTP_A = "mtp.layers.0."
    MTP_SA = "mtp.layers.0.self_attn."

    def _mtp_cat(self, tokens: list[int], pos0: int, hbase: int, h0_row: int,
                 T: int) -> int:
        """Build the drafter's `fc` input: a [T, 10240] block whose row t is

            cat([norm(embed(x_{pos0+t})), norm(h_{pos0+t-1})])   -- EMB half FIRST

        `hbase` is the target's final-normed-hidden buffer at position
        pos0 + h0_row, read with a row stride of HIDDEN, so the rows the drafter
        wants (h_{pos0-1} .. h_{pos0+T-2}, i.e. the chunk's own h shifted back by
        one) are addressed one row at a time.

        The block is 10240 wide per row and NO shipped kernel writes a packed
        [T, 5120] source into a 10240-strided destination -- rmsnorm1p uses one
        `pitch` for both, embed_gather strides by its own `hidden`. So it is
        built ONE ROW AT A TIME, three launches each, and all three are
        weight-free (the only reads are the embedding row and the target's
        hidden row). At MTP_WARM_CHUNK = 128 that is 384 launches, ~1.5 ms of
        launch overhead against a prefill chunk that reads 12 GB, and it lets
        the `fc` GEMV and the whole layer run as ONE batched pass over the
        drafter's 212 MB -- the reason the warm-up is chunked at all.

        `h0_row` is the offset of the h row for the chunk's FIRST drafter
        position within `hbase`; for a prefill sub-chunk that is -1 (the previous
        sub-chunk's last hidden, handed in through `htail`), and for a decode
        position it is 0 (the target's h at pos-1, saved in `hprev`).
        """
        if T > MTP_WARM_CHUNK:
            raise _contract_error(f"{T} drafter rows exceed the mcat arena "
                                  f"({MTP_WARM_CHUNK})")
        cat = self._p("mcat")
        self.dev.htod(self.w["ids"], struct.pack(f"<{T}i", *tokens))
        for t in range(T):
            src = hbase + (h0_row + t) * HIDDEN * 2
            dst = cat + t * 2 * HIDDEN * 2
            # the embedding half: gather straight into the strided row
            self._launch(self.k["embed_gather"], (8, 1), (256,),
                         [self.ptr(f"{PREFIX}embed_tokens.qweight"),
                          self.ptr(f"{PREFIX}embed_tokens.scales"),
                          self._p("ids") + t * 4, dst, HIDDEN])
            self._launch(self.k["rmsnorm1p"], (1, 1), (256,),
                         [dst, self.ptr(self.MTP + "pre_fc_norm_embedding.weight"),
                          dst, HIDDEN, 1, HIDDEN, NORM_EPS])
            self._launch(self.k["rmsnorm1p"], (1, 1), (256,),
                         [src, self.ptr(self.MTP + "pre_fc_norm_hidden.weight"),
                          dst + HIDDEN * 2, HIDDEN, 1, HIDDEN, NORM_EPS])
        return cat

    def _mtp_layer(self, state, T: int, pos0: int) -> None:
        """The drafter's ONE full_attention layer, over a [T, 5120] residual in
        `x`, with its own KV pair. The target's kernels, the target's arenas, the
        target's geometry -- the layer is a full_attention layer, so there is no
        recurrence to snapshot and no conv state to ping-pong.

        The residual stream is entered at the layer's input (the `fc` output has
        no residual of its own -- vLLM passes residual = None for it), so this
        is the standard `_full_attn` / `_add` / `_mlp` / `_add` body with the
        target's weight names replaced.
        """
        a = self.MTP_SA
        m = state.mtp
        self._rms(self._p("x"), self.ptr(self.MTP + "layers.0.input_layernorm.weight"),
                  self._p("h"), HIDDEN, 1, HIDDEN, T)
        self._gemv_many(self._p("h"), [(a + "q_proj", self._p("qg")), (a + "k_proj", self._p("k")),
                                        (a + "v_proj", self._p("v"))], T)
        if self._draft_q8() and ATT_PREP and ATT_DEC and T <= self.split_max:
            # the target's decode block: prep, attention, gated merge
            self._attn_prep(self._p("qg"), a, m.dk.ptr, m.dv.ptr, pos0, m.cap, T)
            self._attn_dec(self._p("qg"), m.dk.ptr, m.dv.ptr, self._p("att"), pos0, m.cap, T,
                           gate=True)
        else:
            self._mtp_attn(state, T, pos0)
        self._gemv(self._p("att"), a + "o_proj", self._p("p"), T)
        self._add(T)
        self._rms(self._p("x"), self.ptr(self.MTP + "layers.0.post_attention_layernorm.weight"),
                  self._p("h"), HIDDEN, 1, HIDDEN, T)
        self._mlp(self.MTP + "layers.0.", T)
        self._add(T)

    def _mtp_attn(self, state, T: int, pos0: int) -> None:
        """The drafter layer's attention block between its projections and
        o_proj, launch by launch (norms, rope, the KV store, attention, gate)."""
        a = self.MTP_SA
        m = state.mtp
        # Gemma-style per-head q/k norms, 256 wide, q inside the q|gate buffer
        self._rms(self._p("qg"), self.ptr(a + "q_norm.weight"), self._p("qg"),
                  HEAD_DIM, N_HEADS, 2 * HEAD_DIM, T)
        self._rms(self._p("k"), self.ptr(a + "k_norm.weight"), self._p("k"),
                  HEAD_DIM, N_KV, HEAD_DIM, T)
        self._launch(self.k["rope_partial"],
                     ((N_HEADS + N_KV + 7) // 8, T), (256,),
                     [self._p("qg"), self._p("k"), N_HEADS, N_KV, HEAD_DIM, ROPE_DIM,
                      2 * HEAD_DIM, HEAD_DIM, pos0, self.rope_theta])
        q8 = self._draft_q8()
        self._launch(self.k["kv_store_q8" if q8 else "kv_store"], (N_KV, T), (256,),
                     [self._p("k"), self._p("v"), m.dk.ptr, m.dv.ptr,
                      0, pos0, N_KV, HEAD_DIM, m.cap])
        if q8 and (T <= self.split_max or APF_FH):
            if T <= self.split_max and ATT_DEC:
                self._attn_dec(self._p("qg"), m.dk.ptr, m.dv.ptr, self._p("att"), pos0,
                               m.cap, T)
            else:
                self._attn_fh(self._p("qg"), m.dk.ptr, m.dv.ptr, self._p("att"), pos0,
                              m.cap, T, q8=True)
        elif T <= self.split_max:
            # the GQA form reads a k/v row once for its 6 q heads; with the same
            # split_rows it is bit-identical to attention_split
            self._launch(self.k["attention_split_gqa" if ATT_GQA else "attention_split"],
                         (N_KV if ATT_GQA else N_HEADS, T, ATT_SPLIT), (256,),
                         [self._p("qg"), m.dk.ptr, m.dv.ptr, self.pacc, self.pm,
                          self.pd, 0, N_HEADS, N_KV, HEAD_DIM, 2 * HEAD_DIM, pos0,
                          m.cap, ATTN_SCALE, ATT_SPLIT_ROWS])
            self._launch(self.k["attention_merge"], (N_HEADS, T, ATT_SPLIT), (256,),
                         [self.pacc, self.pm, self.pd, self._p("att"), N_HEADS,
                          HEAD_DIM, HEAD_DIM])
        elif APF_FH:
            # an fp16 drafter cache (DRAFT_KV8 off, or an fp16 package)
            self._attn_fh(self._p("qg"), m.dk.ptr, m.dv.ptr, self._p("att"), pos0, m.cap, T,
                          q8=False)
        else:
            self._launch(self.k["attention_prefill"], (N_HEADS, (T + 7) // 8), (256,),
                         [self._p("qg"), m.dk.ptr, m.dv.ptr, self._p("att"), 0,
                          N_HEADS, N_KV, HEAD_DIM, 2 * HEAD_DIM, HEAD_DIM, pos0,
                          m.cap, ATTN_SCALE, T])
        self._launch(self.k["attn_gate"], ((N_HEADS * HEAD_DIM + 255) // 256, T),
                     (256,),
                     [self._p("att"), self._p("qg"), N_HEADS, HEAD_DIM, HEAD_DIM,
                      2 * HEAD_DIM, HEAD_DIM])

    def _mtp_forward(self, state, tokens: list[int], pos0: int, hbase: int,
                     h0_row: int, T: int) -> None:
        """Drafter rows for `tokens` at `pos0`, no head. Leaves the drafter's
        KV valid through pos0 + T - 1. In pieces of MTP_WARM_CHUNK rows, the
        `mcat` arena's depth: a prefill sub-chunk warms up to MAX_CHUNK - 1."""
        for i in range(0, T, MTP_WARM_CHUNK):
            n = min(MTP_WARM_CHUNK, T - i)
            cat = self._mtp_cat(tokens[i:i + n], pos0 + i, hbase, h0_row + i, n)
            self._gemv(cat, self.MTP + "fc", self._p("x"), n)
            self._mtp_layer(state, n, pos0 + i)
        state.mtp.mtp_len = max(state.mtp.mtp_len, pos0 + T)

    def _mtp_warm(self, state, tokens: list[int], pos0: int) -> None:
        """Fill the drafter's own KV over a prefill sub-chunk, and carry this
        chunk's last hidden forward as the next chunk's `htail`.

        Under the drafter's contract, the row at position q needs the target's
        hidden at q and the token at q+1, so over a chunk of T tokens at
        pos0..pos0+T-1 the rows that CAN be filled are pos0..pos0+T-2: they need
        h rows 0..T-2 and token indices 1..T-1, both contiguous. The LAST
        position, pos0+T-1, needs the token AFTER the chunk -- the first decoded
        token -- so it is the post-step draft's job, which runs with that token
        in hand and covers exactly that position. Nothing is left over.

        `pos0 > 0` needs one row peeled: the row at `pos0` reads the hidden at
        `pos0`, which is the PREVIOUS sub-chunk's last row and the one h row `h`
        no longer holds. It is a single T = 1 drafter call -- 212 MB of drafter
        weights for one row -- once per sub-chunk, against a prefill chunk that
        reads 12 GB of target weights. At `pos0 == 0` there is no such row and
        the whole range batches.
        """
        m = state.mtp
        self._mtp_grow(state, state.cap)
        T = len(tokens)
        if T == 0:
            return
        hk = self._p("hkeep")
        if T > 1:
            if pos0 == 0:
                self._mtp_forward(state, tokens[1:T], 0, hk, 0, T - 1)
            else:
                self._mtp_forward(state, tokens[1:2], pos0, m.htail.ptr, 0, 1)
                if T > 2:
                    self._mtp_forward(state, tokens[2:T], pos0 + 1, hk, 0, T - 2)
        # T == 1 (a decode step): the only row this could fill is the one the
        # post-step draft fills anyway, and it needs the token AFTER this one.
        # carry the last hidden of this chunk for the next one's peeled row
        self.dev.dtod(m.htail, self._view(hk + (T - 1) * HIDDEN * 2, HIDDEN * 2),
                      HIDDEN * 2)

    def _mtp_draft(self, state, tokens: list[int], pos0: int, hbase: int,
                   h0_row: int, pen=None) -> int:
        """Run the drafter over `tokens` at `pos0` -- T rows -- and return its
        prediction for the LAST row's next position (`pos0 + T + 1`).

        THE DRAFTER'S CONTRACT, which is vLLM's and is off by one from the
        obvious reading (see `set_inputs_first_pass` in
        _reference/vllm/vllm/v1/spec_decode/llm_base_proposer.py: it SHIFTS the
        target's token ids by one and leaves the positions and the hidden states
        UNSHIFTED):

            the drafter's row at position q is built from the TARGET's hidden at
            q and the token at q + 1, stores its KV at q, attends over the
            drafter's own 0..q, and predicts the token at q + 2.

        So row t here reads `hbase` row `h0_row + t` -- the target's hidden at
        position `pos0 + t` -- and `tokens[t]`, the token at `pos0 + t + 1`.

        The head runs over the last row only, so a T = 2 call costs one 954 MB
        pass over the shared 6-bit trellis plus one over the drafter's 212 MB --
        which is what makes the ACCEPT path's two rows cost the same as one.
        """
        T = len(tokens)
        m = state.mtp
        cat = self._mtp_cat(tokens, pos0, hbase, h0_row, T)
        self._gemv(cat, self.MTP + "fc", self._p("x"), T)
        self._mtp_layer(state, T, pos0)
        m.mtp_len = max(m.mtp_len, pos0 + T)
        self._rms(self._p("x"), self.ptr(self.MTP + "norm.weight"),
                  self._p("h"), HIDDEN, 1, HIDDEN, T)
        # the SHARED 6-bit head over the last row, the target's T = 1 head
        self._head(self._p("h") + (T - 1) * HIDDEN * 2, 1)
        if pen is not None:
            # the target samples this position under the request's penalties,
            # and a round accepts only an exact match: draft the PENALIZED
            # argmax (the top-1 after the same stack, same context). Any draft
            # keeps the stream exact; this one is the likelier match.
            spec, p_ids, o_ids = pen
            got = self._candidates_dev(self.w["logits2"], 1.0, 1, spec, p_ids, o_ids)
            if got is not None and got[1]:
                return got[1][0][1]
        return self._argmax(self.w["logits2"])

    def _mtp_next_draft(self, state, pos0: int, hbase: int, h0_row: int,
                        tokens: list[int]) -> None:
        """Queue the draft the round at `pos0` will need, and re-anchor `hprev`.

        Every path that changes what the state has covered comes through here, so
        the drafter's rows and `hprev` cannot drift apart: the rows written are
        exactly the positions the next round's draft must be able to attend to.
        """
        m = state.mtp
        # The target's final normed hidden at the position these rows describe,
        # saved BEFORE the drafter runs: the drafter's own layer writes the same
        # arena `hbase` usually points into, and `hprev` is what the next round's
        # draft reads. (A second, identical copy used to sit after `_mtp_draft`;
        # it is right only while `hbase` survives the drafter -- had a call site
        # ever passed the arena itself, it would have saved the DRAFTER's hidden
        # over the target's. One copy, before, is the correct one.)
        self.dev.dtod(m.hprev, self._view(hbase + (h0_row + len(tokens) - 1)
                                          * HIDDEN * 2, HIDDEN * 2), HIDDEN * 2)
        pen = None
        if m.spec is not None and self._pen_active(m.spec):
            # the output context the target will sample these positions under:
            # what the state holds past `fed`, then the tokens this call adds
            # beyond it (tokens[k] is the token at pos0 + 1 + k)
            p_ids, o_ids = self._pen_ctx(state)
            k0 = max(0, len(state.tokens) - pos0 - 1)
            pen = (m.spec, p_ids, o_ids + list(tokens[k0:]))
        m.draft = self._mtp_draft(state, tokens, pos0, hbase, h0_row, pen)
        m.draft2 = None
        q = pos0 + len(tokens)          # the chained row's position
        if MTP_K >= 2 and q < m.cap and q + 2 <= MAX_CONTEXT:
            # vLLM's chain: the row at q takes the first draft as its token and,
            # for the target hidden at q that does not exist yet, the drafter's
            # own post-norm output of the row before (still in `h`; `_mtp_cat`
            # reads it before the layer overwrites the arena). Its KV row is
            # provisional: the next round's draft rows rewrite q before reading.
            keep = m.mtp_len
            pen2 = None if pen is None else (pen[0], pen[1], pen[2] + [m.draft])
            m.draft2 = self._mtp_draft(state, [m.draft], q, self._p("h"), len(tokens) - 1,
                                       pen2)
            m.mtp_len = keep

    def _mtp_snapshot(self, state) -> None:
        """Copy the recurrent state a verify cannot rewind.

        `gdn_scan`'s state is a running product: a T = 2 verify advances it two
        positions in ONE call and there is no way back except a copy. So does
        the conv state, whose second-to-last frame would be the rejected draft.
        144 MiB + 5.6 MiB, one buffer, ~0.4 ms against a 78 ms forward. The KV
        cache needs no snapshot: it is position-addressable and always written
        before it is read, so a stale row is overwritten, never read.
        """
        self.dev.dtod(self.snap_s, state.s, GDN_S_BYTES)
        self.dev.dtod(self.snap_c[0], state.conv[0], GDN_CONV_BYTES)
        self.dev.dtod(self.snap_c[1], state.conv[1], GDN_CONV_BYTES)

    def _mtp_restore(self, state) -> None:
        self.dev.dtod(state.s, self.snap_s, GDN_S_BYTES)
        self.dev.dtod(state.conv[0], self.snap_c[0], GDN_CONV_BYTES)
        self.dev.dtod(state.conv[1], self.snap_c[1], GDN_CONV_BYTES)

    # -- sampling -----------------------------------------------------------

    def _candidates_dev(self, logits, temperature: float, top_k: int,
                        spec=None, prompt_ids=(), out_ids=()):
        """The candidate step on the DEVICE: (top, [(value, index), ...]) or None.

        `_candidates` below does this on the host and pays 496 KB of readback
        plus a Python pass over all 248 077 logits (38 ms under a python3 with
        no numpy). The device does it in three small passes and hands back only
        the candidates:

          exl3_cand_hist     the row's largest key + a per-block 512-bucket
                             histogram of the fp16 values' sortable keys
          exl3_cand_sum      those 32 histograms into one, and reset the counter
          exl3_cand_collect  the (value, index) pairs at or above a threshold

        The threshold is chosen from the histogram so that the collect returns
        the top_k-th value's bucket edge at the latest, i.e. a superset of the
        exact top_k and a subset of the host path's `{v > top - 30*temperature}`
        set -- and when the row has fewer than top_k candidates above the
        floor, the threshold IS that floor's successor, so the set is the same
        one the host loop would have collected. The ORDER (value desc, index
        desc) is imposed here on whatever comes back, so the candidate list is
        the host path's list, exactly.

        Returns None when the candidate count overflows CAND_CAP (the caller
        falls back to the host path, which is exact by construction).
        """
        chist, csum, cbuf = self.w["chist"], self.w["csum"], self.w["cand"]
        lp = logits.ptr
        self._launch(self.kx["exl3_cand_hist"], (CAND_BLOCKS,), (256,),
                     [lp, MIN_VOCAB_ID, chist.ptr, self.cand_perm])
        self._launch(self.kx["exl3_cand_sum"], (1,), (256,),
                     [chist.ptr, self.cand_perm, CAND_BLOCKS, csum.ptr,
                      cbuf.ptr])
        raw = self.dev.dtoh(csum, 4 * (CAND_BINS + 1))
        top_key, counts = struct.unpack_from("<I", raw, 0)[0], \
            struct.unpack_from(f"<{CAND_BINS}I", raw, 4)

        top = _key_value(top_key)
        floor = top - 30.0 * temperature
        span = self._pen_span(spec, top, out_ids) if spec else 0.0
        penalized = spec is not None and self._pen_active(spec)
        if penalized:
            floor -= span
        lo = _half_above(floor)                      # smallest fp16 > floor
        # Raise the floor to the bucket edge past `want` accounted candidates.
        # With penalties the reordering can displace tokens, so `want` carries
        # a fixed 256 of slack and the value margin stays small: the COUNT
        # bound is what keeps the collect under CAND_CAP -- a value window,
        # even one narrowed by the penalty span, gathers thousands of the flat
        # tail on a peaked row, overflows the buffer, and every token falls to
        # the host path (measured: decode 2.3 tok/s instead of 18.5). Without
        # penalties the raise is exactly the shipped one.
        want = top_k + (64 if penalized else 0)
        margin = min(2.0 * span, 2.0) if penalized else 0.0
        cum = 0
        for b in range(CAND_BINS - 1, -1, -1):
            cum += counts[b]
            if cum >= max(want, 1):
                edge = _key_value(b << CAND_SHIFT) - margin
                if edge > lo:
                    lo = edge
                break

        self._launch(self.kx["exl3_cand_collect"], (CAND_BLOCKS,), (256,),
                     [lp, MIN_VOCAB_ID, lo, cbuf.ptr + 8,
                      cbuf.ptr + 8 + 4 * CAND_CAP, cbuf.ptr, CAND_CAP])
        raw = self.dev.dtoh(cbuf, 8 + 8 * CAND_CAP)
        n = struct.unpack_from("<I", raw, 0)[0]
        if n > CAND_CAP:
            return None                              # overflow: use the host
        vals = struct.unpack_from(f"<{n}f", raw, 8)
        idxs = struct.unpack_from(f"<{n}I", raw, 8 + 4 * CAND_CAP)
        pairs = self._penalize(list(zip(vals, idxs)), spec, prompt_ids, out_ids)
        # nlargest over (value, index) pairs is the same order as sorting by
        # (-value, -index) and taking the head -- value desc, index desc -- at
        # O(n log top_k) instead of O(n log n), which matters when the penalty
        # window collects thousands
        cand = sorted(heapq.nlargest(top_k, pairs) if 0 < top_k < len(pairs)
                      else sorted(pairs, key=lambda p: (-p[0], -p[1])),
                      key=lambda p: (-p[0], -p[1]))
        if 0 < top_k < len(cand):
            cand = cand[:top_k]
        return top, cand

    def _argmax(self, logits) -> int:
        """The greedy token: the argmax of the fp16 row, first maximum wins.

        The two-kernel reduction `_build/`'s argmax pair uses (partial per
        block, then a merge), on the fp16 buffer the sampler reads rather than
        on a converted copy -- on the host this was a 248 077-element scan
        (0.3 ms under numpy, ~30 ms under a plain python3).
        """
        self._launch(self.kx["exl3_argmax16_partial"], (AMAX_BLOCKS,), (256,),
                     [logits.ptr, self.w["amax"].ptr, self.amax_idx,
                      MIN_VOCAB_ID])
        self._launch(self.kx["exl3_argmax16_final"], (1,), (256,),
                     [self.w["amax"].ptr, self.amax_idx, AMAX_BLOCKS,
                      self.amax_out, self.amax_out + 4])
        return struct.unpack("<i", self.dev.dtoh(
            self.w["amax"], 4, offset=AMAX_BLOCKS * 8))[0]

    def _candidates(self, raw: bytes, temperature: float, top_k: int,
                    spec=None, prompt_ids=(), out_ids=()):
        """(top value, [(value, index)], ...) in the order the sampler wants.

        Descending by value, ties by descending index -- what sorting
        (value, index) pairs gives, and what `heapq.nlargest` gives for the
        top_k branch. The numpy path is the shipped one; the loop below it is
        the same computation for a runtime without numpy (the CLI runs under a
        plain python3 on this machine, which has no numpy), and
        _build/check_sampler.py checks the two against each other on real logits
        rows, for several specs and seeds, candidate list and token both.
        """
        if _np is not None:
            flat = _np.frombuffer(raw, dtype="<f2").astype(_np.float64)
            top = float(flat.max())
            floor = top - 30.0 * temperature
            if spec is not None:
                floor -= self._pen_span(spec, top, out_ids)
            sel = _np.flatnonzero(flat > floor)
            pairs = self._penalize(list(zip(flat[sel].tolist(), sel.tolist())),
                                   spec, prompt_ids, out_ids)
            pairs.sort(key=lambda p: (-p[0], -p[1]))   # value desc, index desc
            if 0 < top_k < len(pairs):
                pairs = pairs[:top_k]
            return top, pairs
        values = struct.unpack(f"<{MIN_VOCAB_ID}e", raw)
        top = max(values)
        floor = top - 30.0 * temperature
        if spec is not None:
            floor -= self._pen_span(spec, top, out_ids)
        cand = [(v, i) for i, v in enumerate(values) if v > floor]
        cand = self._penalize(cand, spec, prompt_ids, out_ids)
        if 0 < top_k < len(cand):
            cand = heapq.nlargest(top_k, cand)
        else:
            cand.sort(reverse=True)
        return top, cand

    @staticmethod
    def _pen_active(spec) -> bool:
        """vLLM's `use_penalty` guard: neutral penalties cost nothing anywhere."""
        return (spec.repetition_penalty != 1.0 or spec.presence_penalty != 0.0
                or spec.frequency_penalty != 0.0)

    @staticmethod
    def _penalize(pairs, spec, prompt_ids, out_ids):
        """vLLM's penalty stack over the candidate pairs, on the RAW values.

        Its `layers/utils.py` apply_penalties, formula for formula and in its
        order: repetition (HF scale -- seen tokens: logit > 0 ? logit/p :
        logit*p -- over the PRESENCE union of prompt and output), then
        frequency (minus frequency x times-seen, OUTPUT only), then presence
        (minus presence, binary, OUTPUT only). The candidates are a superset of
        the top-k, so penalties applied here land exactly where applying them
        over the whole vocabulary would: penalties of this sign can only push
        tokens out of the cut, never into it.
        """
        if spec is None or (not prompt_ids and not out_ids):
            return pairs
        rp, pp, fp = (spec.repetition_penalty, spec.presence_penalty,
                      spec.frequency_penalty)
        if rp == 1.0 and pp == 0.0 and fp == 0.0:
            return pairs
        seen = set(prompt_ids) | set(out_ids)
        counts = {}
        for t in out_ids:
            counts[t] = counts.get(t, 0) + 1
        out = []
        for v, i in pairs:
            if rp != 1.0 and i in seen:
                v = v / rp if v > 0 else v * rp
            c = counts.get(i)
            if c:
                if fp != 0.0:
                    v -= fp * c
                if pp != 0.0:
                    v -= pp
            out.append((v, i))
        return out

    @staticmethod
    def _pen_span(spec, top, out_ids) -> float:
        """How far below its raw value a penalty can push a logit. The
        collection floor must widen by this, or a penalized token falls out of
        the candidate superset before the sort ever sees it.

        The frequency term is bounded by the LARGEST per-token count, not the
        output length (using the length put 60+ logits of margin on a 200-token
        output, the collect overflowed CAND_CAP and every token fell to the
        host path: decode 2.3 tok/s instead of 18.5, measured). And the whole
        span is capped at 12 raw logits: past that a token has been repeated so
        hard the penalty exists to kill it and the loop guard owns the case,
        while an unbounded span re-opens the same overflow."""
        if spec is None:
            return 0.0
        span = 0.0
        if spec.repetition_penalty != 1.0:
            span += abs(top) * abs(1.0 - 1.0 / spec.repetition_penalty) + 4.0
        span += abs(spec.presence_penalty)
        if spec.frequency_penalty != 0.0 and out_ids:
            counts = {}
            for t in out_ids:
                counts[t] = counts.get(t, 0) + 1
            span += abs(spec.frequency_penalty) * max(counts.values())
        return min(span, 12.0)

    def _pen_ctx(self, state):
        """(prompt ids, output ids) for the penalty stack. vLLM's split:
        repetition sees prompt + OUTPUT, presence/frequency see OUTPUT only,
        and the last generated token counts -- here it waits in `pending`
        until the next forward, so it joins the output side."""
        toks = state.tokens
        out = list(toks[state.fed:])
        if state.pending is not None:
            out.append(state.pending)
        return toks[:state.fed], out

    def _sample(self, logits, spec, prompt_ids=(), out_ids=()) -> int:
        """One token from the step's logits row, honouring the spec.

        The engine owns the policy; this executes it where the logits live. The
        draw stream is per-request and advanced once per sampled token
        (CONTRACT.md): `spec.seed` keys a generator kept across the request's
        steps, so two requests never share a stream and one request's stream is
        not restarted every token.

        The candidate step is where a decode token's host time went: measured
        (paired, one process) at 30-38 ms of a 128 ms token against 0.3 ms for
        the same work in numpy, because everything downstream of the candidate
        list works on at most `top_k` (40 by default) values. `_candidates`
        holds both host forms; `_candidates_dev` is the device one, and it is
        what the stdlib path (a python3 with no numpy) runs, since it removes
        the 496 KB readback AND the 248 077-element Python pass in one go. Any
        spec it cannot serve exactly -- top_k = 0, i.e. "keep every logit above
        the floor", which on this model's rows is 150 000+ of them, or an
        overflow of its candidate buffer -- falls back to the host path.

        Penalties (vLLM's stack, `_penalize`) act on the raw candidate values
        before the top-k cut. Greedy takes the argmax of the penalized row
        (vLLM argmaxes after the penalties): with penalties active it runs the
        candidate window widened by `_pen_span` instead of the device argmax,
        whose first-maximum tie-break it keeps (lowest index of the equal max).
        """
        pen = (prompt_ids, out_ids) if self._pen_active(spec) else None
        if spec.temperature <= 0 and pen is None:
            # Greedy: the argmax of the same fp16 row the stochastic path reads,
            # over the same MIN_VOCAB_ID prefix, and with the first-maximum
            # tie-break. (The argmax kernels in kernels.cu take fp32 and this
            # buffer is fp16, so handing them the row reads twice its length --
            # that path is gone rather than fixed; exl3_argmax16_partial reads
            # the fp16 row itself.)
            return self._argmax(logits)

        if spec.temperature <= 0:
            # greedy WITH penalties: vLLM argmaxes after the penalty stack, so
            # this is the argmax of the PENALIZED row, over the window the
            # widened floor collects. First maximum wins, as `_argmax` does.
            raw = self.dev.dtoh(logits, MIN_VOCAB_ID * 2)
            _, cand = self._candidates(raw, 0.0, 0, spec, prompt_ids, out_ids)
            return max(cand, key=lambda p: (p[0], -p[1]))[1]

        cand = None
        if spec.top_k > 0:
            got = self._candidates_dev(logits, spec.temperature, spec.top_k,
                                      spec, prompt_ids, out_ids)
            if got is not None:
                cand = got[1]
        elif pen is None and _np is not None and SAMPLE_NP:
            # no top-k: tens of thousands of candidates, which the list path
            # below sorted and walked in Python (+12 ms a token at T = 0.8,
            # top_p = 1). The same candidates, order and arithmetic in numpy.
            return self._sample_np(self.dev.dtoh(logits, MIN_VOCAB_ID * 2), spec)
        if cand is None:
            raw = self.dev.dtoh(logits, MIN_VOCAB_ID * 2)
            _, cand = self._candidates(raw, spec.temperature, spec.top_k,
                                      spec, prompt_ids, out_ids)
        return self._finish(cand, spec)

    def _sample_np(self, raw: bytes, spec, draw: float | None = None) -> int:
        """`_candidates` + `_finish` for a spec without top-k or penalties, on
        arrays: the floor, the (value desc, index desc) order, v / T, the
        min_p cut, pow(e, s - hi), top_p over a stable sort, the normalisations
        and the draw's running sum are the list path's operations in its order
        (np.power is libm's pow here, np.cumsum a left-to-right sum), and the
        two totals ARE Python's `sum` (compensated since 3.12, so not a cumsum)
        -- the same token for the same draw (_build/check_sampler_np.py)."""
        flat = _np.frombuffer(raw, dtype="<f2").astype(_np.float64)
        top = float(flat.max())
        sel = _np.flatnonzero(flat > top - 30.0 * spec.temperature)[::-1]   # index desc
        # value desc, ties keep index desc: a stable sort on an order-keeping
        # 16-bit key of the fp16 bits (-0 folded into +0, as the floats compare),
        # which numpy radix-sorts (20 ms -> ~2 at 247k candidates)
        bits = _np.frombuffer(raw, dtype="<u2")[sel]
        bits = _np.where(bits == 0x8000, 0, bits)
        key = _np.where(bits & 0x8000, bits, 0x7FFF - bits).astype(_np.uint16)
        order = _np.argsort(key, kind="stable")
        idx = sel[order]
        vals = flat[idx]
        scores = vals / spec.temperature
        hi = scores.max()
        if spec.min_p > 0:
            cut = hi + __import__("math").log(spec.min_p)
            scores = _np.where(scores >= cut, scores, -1.0e30)
            hi = scores.max()
        probs = _np.power(2.718281828459045, scores - hi)
        probs = probs / sum(probs.tolist())
        if 0 < spec.top_p < 1:
            # the candidates come value-descending, so probs is (almost surely)
            # non-increasing and the stable sort is the identity
            by = (_np.arange(len(probs)) if bool((probs[1:] <= probs[:-1]).all())
                  else _np.argsort(-probs, kind="stable"))
            cum = _np.cumsum(probs[by])
            hit = _np.flatnonzero(cum >= spec.top_p)
            keep = int(hit[0]) + 1 if hit.size else len(by)
            probs[by[keep:]] = 0.0
            total = sum(probs.tolist())
            if total > 0:
                probs = probs / total
        if draw is None:
            draw = self._rng(spec.seed).random()
        k = int(_np.searchsorted(_np.cumsum(probs), draw, side="right"))
        return int(idx[k]) if k < len(idx) else int(idx[-1])

    def _finish(self, cand, spec) -> int:
        """The sampler's tail: probabilities, top_p, min_p and the draw.

        Untouched since the round that vectorised `_candidates`, and it works on
        at most `top_k` values, which is why the candidate step is the only part
        that had to move.
        """
        scores = [v / spec.temperature for v, _ in cand]
        hi = max(scores)
        if spec.min_p > 0:
            # min_p BEFORE the softmax and the top_p cut, as vLLM orders the
            # chain (temperature -> min_p -> top-k/top-p): "prob < min_p * p_max"
            # is a threshold on the scaled logits, max + ln(min_p).
            cut = hi + __import__("math").log(spec.min_p)
            scores = [s if s >= cut else -1.0e30 for s in scores]
            hi = max(scores)
        probs = [pow(2.718281828459045, s - hi) for s in scores]
        total = sum(probs)
        probs = [p / total for p in probs]
        if 0 < spec.top_p < 1:
            order = sorted(range(len(probs)), key=lambda k: -probs[k])
            cum = 0.0
            keep = 0
            for k in order:
                cum += probs[k]
                keep += 1
                if cum >= spec.top_p:
                    break
            mask = set(order[:keep])
            probs = [p if k in mask else 0.0 for k, p in enumerate(probs)]
            total = sum(probs)
            if total > 0:
                probs = [p / total for p in probs]
        stream = self._rng(spec.seed)
        draw = stream.random()
        acc = 0.0
        for (_, token), pr in zip(cand, probs):
            acc += pr
            if draw < acc:
                return token
        return cand[-1][1]

    def _rng(self, seed):
        """The per-request draw stream, one generator per seed.

        Streams are cached so one request's draws continue across its steps.
        When the cache is full the OLDEST stream goes (insertion order, so the
        live requests' streams are the last to be touched) -- never a blanket
        clear(): that restarted a concurrent request's stream mid-generation,
        and a restarted stream repeats its draws, which repeats its tokens.
        """
        rng = self._streams.get(seed)
        if rng is None:
            while len(self._streams) >= 64:
                self._streams.pop(next(iter(self._streams)))
            rng = self._streams[seed] = random.Random(seed)
        else:
            # refresh recency (true LRU). Without this a live generation's
            # stream sits at the FRONT of insertion order and is the first one
            # evicted once 64 other seeds come and go -- and the restart re-
            # draws the stream from the top, which repeats the tokens: the
            # loop this whole cache exists to avoid.
            self._streams[seed] = self._streams.pop(seed)
        return rng


# ---------------------------------------------------------------------------
# the fused GEMV's shape, measured (_build/bench_wide.py, _build/NOTES.md)
# ---------------------------------------------------------------------------
# One warp per 16-column tile with the whole k axis leaves the walk with ONE
# cache line in flight per warp per k-step (the stride is nto*W*4 = 122 KB), so
# its rate is warps x bytes-per-k-step / round-trip time rather than the card's
# bandwidth: 129 GB/s against 572-605 measured for a stream, and the pure
# decode arithmetic without the walk runs at 624. Two shape parameters fix it,
# and the kernel has an entry point for each combination of them:
#
#   WT  tiles per warp, consecutive along the output axis (exl3_gemv_w4): WT
#       lines in flight per warp per k-step, WT times the bytes, the same
#       decode applied to each tile, x loaded once per token per k-step.
#   S   k-slices from gridDim.z, summed by exl3_sk_reduce (exl3_sk_reduce's
#       S fp32 partials are S*T*out floats, so this is for small T only).
#
# Measured at T = 1 on every module kind of this checkpoint: WT=4 with S from
# the block count is 1.3-3.4x over the one-tile kernel (gate_proj 115 -> 177
# GB/s, down_proj 56 -> 153, k_proj 21 -> 72, lm_head 230 -> 315). Above T ~ 4
# the token grid already fills the GPU and the partials stop being free, so the
# one-tile kernel keeps those calls.
EXL3_WT = 4                # tiles per warp on the wide path
EXL3_BLOCKS = int(__import__("os").environ.get("ORCA_BLOCKS", "288"))  # the block count a split aims for
EXL3_PT_BYTES = 4 << 20    # the partial arena (S*T*out*4 for the worst module)
EXL3_PT_SHARE = 0.08       # ... and the share of the trellis read it may take
# the verify's group of 3 on its own entry (exl3_gemv_w4a3fp, 2 blocks an SM; 3 an
# SM from this input width up -- down_proj, out_proj, o_proj measured 1.04-1.08x
# there, the 5120-wide inputs 0.8-1.0x). ORCA_M3_ENTRY=0: the m-general entry.
M3_ENTRY = int(__import__("os").environ.get("ORCA_M3_ENTRY", "1"))
# a decode step's projections of one input in one launch (exl3_gemv_w4a1fpn):
# in_proj_qkv + in_proj_z, q/k/v_proj, gate + up_proj. ORCA_GEMV_N=0: one each.
GEMV_N = int(__import__("os").environ.get("ORCA_GEMV_N", "1"))
GEMV_AB = int(__import__("os").environ.get("ORCA_GEMV_AB", "1"))  # a / b rows in that launch
# the T = 1 shape's k-split (which the verify pins too, so both are scored)
# where the one-input launches above measured better than the single-launch
# rule (EXL3_BLOCKS): F1 48.23 -> 47.99 ms, F3v 70.41 -> 69.37. Splits need not
# be powers of two.
DEC_SPLIT = {"in_proj_qkv": 5, "in_proj_z": 10, "q_proj": 4, "k_proj": 16, "v_proj": 16}
DEC_SPLIT_ON = int(__import__("os").environ.get("ORCA_DEC_SPLIT", "1"))
M3_OCC3_MIN_IN = 6144
EXL3_WIDE_T = 7            # T at or below which the wide path is used (5..7: two
                           # m = 4 groups, 143-164 ms a forward against xs_gemm16's 191)
# The prefill token group and the block count it needs to pay for itself:
# m = 16 halves the number of passes over a module's trellis (the walk is the
# cost -- see kernels_exl3.cu's prefill entry), and halves the block count with
# it, so a launch that would fall under ~2 blocks an SM keeps the general
# entry's m = 8. Measured pairs in _build/bench_prefill_m.py.
EXL3_M_PRE = 32            # == kernels_exl3.cu's EXL3_M_PRE_MAX
EXL3_M_PRE16 = 16          # ... and the group the stage serves below 32
EXL3_PRE_MIN_BLOCKS = 112  # ... and the shortest grid it is taken on (2/SM)
EXL3_PRE_M32_MIN_BLOCKS = 224   # m = 32 is only worth it with the grid to fill
# half2 elements a thread of the residual add carries; the launch is sized so
# the kernel still streams (see `_add`).
EXL3_ADD_PER_THREAD = 8
# The prefill group's input layout. True feeds the m = 16 path from
# had128_pre_lc, whose fp32 k-tiles are lane-ordered so the GEMV's x is ONE
# 16-byte load per token per k-step with no conversion (kernels_exl3.cu's
# had128_body_lc). It is bit-identical and it measures 1.29x SLOWER on this card
# (gate_proj at T = 512, m = 16: 41.5 ms against 32.3, paired in one process,
# _build/bench_prefill_m.py) -- a lane's 16-byte request quadruples the L1 bytes
# a warp-instruction asks for, and that costs more than the four fp16->fp32
# conversions it removes. The entries stay in the kernel behind this flag so the
# pair can be re-timed; the shipped value is False.
EXL3_PRE_X32 = False
# Prefill as a GEMM (kernels_exl3.cu's xp_gemm*): a weight is decoded once per
# 128 (or 64) tokens into shared memory and every thread runs an 8 x 8 register
# tile. Measured against the m = 16/32 entries above, paired in one process on
# real modules (_build/xg/bench_xp.py): 1.40x weighted at T = 512, 1.17x at
# T = 128, a LOSS below T ~ 96 except xp_gemm64 at T = 64 (1.26x). Narrow outputs
# (k_proj / v_proj: 32 blocks at T = 512) take the k-split entry instead.
XP_MIN_T = 128             # xp_gemm128 from here up
XP_HALF = int(__import__("os").environ.get("ORCA_XP_HALF", "1"))  # fp16-math prefill GEMM (xh_gemm*); 0 = xp_gemm128
# The decode GEMV's affine form (kernels_exl3.cu's exl3_gemv_w4a): the
# multiply on the FP64 pipe and the codebook's affine map applied per output, so
# each weight skips its fp16 rounding (<= half an fp16 ulp). Env-switchable for
# A/B: ORCA_AFFINE=0 restores the shipped exl3_gemv_w4.
EXL3_AFFINE = int(__import__("os").environ.get("ORCA_AFFINE", "1"))
EXL3_W4A1 = int(__import__("os").environ.get("ORCA_W4A1", "1"))
# the affine GEMV with its post-rotation fused (exl3_gemv_w4a1f / w4af)
EXL3_FUSED_POST = int(__import__("os").environ.get("ORCA_FUSED_POST", "1"))
GCNT_SLOTS = 4096          # (column group x token group) counters for it
# ... and with the pre-rotation folded into its prologue (exl3_gemv_w4a1fp / w4afp)
EXL3_FUSED_PRE = int(__import__("os").environ.get("ORCA_FUSED_PRE", "1"))
XP64_T = (64, 96)          # xp_gemm64 inside [lo, hi)
KV_GROW_STEP = 4096         # KV capacity: doubling up to this, then steps of it
XS_MIN_T = 8               # the fp16 prefill GEMM (XP_HALF, _xp_shape) from here up
LABEL_PREFIX_MIN = 16      # a shared prompt head shorter than this is not split off
# A whole forward's milliseconds by token count on this card, at the prefill
# dispatch below (_build/bench_small_prefill.py, Landed 21 in
# OPTIMIZE_2026-10-01_log.md): the cost steps with the GEMM's tile count, not
# with T, which is what `_label_groups` packs the label tails against. Past
# the table, ~5 ms a token.
FORWARD_MS = ((16, 198), (32, 207), (64, 323), (128, 621), (192, 913), (256, 1242),
              (320, 1646), (384, 1989), (448, 2340), (512, 2611))
# The fp16 prefill GEMM's entry and k-split per call (_xp_shape): a call costs
# a * BM * in / S (x 1e-6 us) per block-wave unit, where the busiest SM runs the
# grid's nb = out/128 * ceil(T/BM) * S blocks in waves of `occ` resident blocks
# and a partial last wave costs at least `q` blocks, plus XP_SPLIT_COST us per
# 1e6 partial floats of a split (its per-call constant fits to ~0). Fitted on
# every (entry, S <= 8) of the 7 projection shapes at 24 token counts 12..1024
# (_build/bench_xs_split.py, _build/fit_xp_cost.py): picks within 1.1% of the
# measured best at each count; fitted on 14 counts it held 0.2-1.6% at 9 unseen
# ones from 40 up, and the final fit 0.2-0.5% at 5, 8, 20 and 28.
# (entry family, BM, occ, a, q, T range it was measured over)
XP_COST = (("xs_gemm16", 16, 4, 2690.4, 3, 5, 16),
           ("xs_gemm32", 32, 4, 2352.4, 0.5, 5, 64),
           ("xh_gemm32", 32, 6, 1579.7, 5, 5, 64),
           ("xs_gemm64", 64, 3, 1510.6, 0.5, 5, 512),
           ("xh_gemm64", 64, 3, 1204.5, 2.5, 5, 512),
           ("xh_gemm128", 128, 2, 1198.2, 1.5, 64, 1 << 30),
           ("xh_gemm256", 256, 1, 1190.1, 2, 128, 1 << 30))
XP_SPLIT_COST = 19.12
# xh entries with a `t` twin (padding warps of the last tile skip their
# products): the cost of that tile at live fraction f is c0 + (1 - c0) f of a
# full one (measured, xh_bench: 256 at f .625 / .78 -> c0 .49 / .56; 128 at
# .75 -> .8; 64 lost), and the twin runs `pen` slower on full tiles (256: its
# plain entry is the two-k-tile body, 2.5% ahead of the twin's)
XP_TAIL = {"xh_gemm256": (0.52, 1.045), "xh_gemm128": (0.8, 1.02)}   # (c0, pen)
XP_SMS = 56
FH_ROWS = 64               # attention_prefill_fh: rows a block, kv rows a tile
FH_SPLIT_MAX = 32
LOG2E = 1.4426950408889634
XS_SPLIT_MAX = 8
XP_SPLIT_BLOCKS = 56       # below this many blocks, split k to fill the card
XP_SLOTS = 112             # resident xp_gemm128 blocks (2 per SM)
XP_PT_BYTES = 64 << 20     # split partials (S*T*out*4); _xp_shape caps S by it (8 MB lost 2-10% at 192..768 tokens)

KERNEL_NAMES = ("rmsnorm1p", "rmsnorm_gated", "l2norm_scaled", "rope_partial", "kv_store",
                "attention_split", "attention_merge", "attention_prefill", "attn_gate",
                "conv1d_causal", "gdn_scalars", "gdn_scan", "embed_gather", "silu_mul", "kv_store_q8",
                "attention_split_gqa_q8", "attention_prefill_fa_q8",
                "add_inplace", "argmax_partial", "argmax_final", "add_rmsnorm1p",
                "attention_split_gqa", "attention_prefill_fa", "gdn_scan_f",
                "attention_prefill_fh", "attention_prefill_fh_q8", "attention_merge_fh",
                "gdn_scan_rows", "attention_dec_q8", "attention_dec_merge", "gdn_scan_rows_f",
                "conv1d_causal_rows", "attention_dec_merge_gate", "attn_prep_q8")
KERNEL_X_NAMES = ("had128_pre", "had128_post", "exl3_gemv", "gemv_f16_f32",
                  "exl3_gemv_w4", "exl3_gemv_p4", "exl3_sk_reduce",
                  "exl3_gemv_pre", "exl3_gemv_pre32", "had128_pre_lc",
                  "xp_gemm128", "xp_gemm128s", "xp_gemm64", "xh_gemm32", "xh_gemm32s", "xh_gemm64", "xh_gemm64s", "xh_gemm128", "xh_gemm128s", "xh_gemm256", "xh_gemm256s", "xh_gemm128t", "xh_gemm128st", "xh_gemm256t", "xh_gemm256st", "xs_gemm16", "xs_gemm16s", "xs_gemm32", "xs_gemm32s", "xs_gemm64", "xs_gemm64s", "gemv_ab_f32", "gemm_ab_f32", "had128_pre_silu",
                  "had128_post_sk", "exl3_gemv_w4a", "exl3_gemv_w4a1",
                  "exl3_gemv_w4a1f", "exl3_gemv_w4af", "exl3_gemv_w4a1fp",
                  "exl3_gemv_w4afp", "exl3_gemv_w4a3fp", "exl3_gemv_w4a3fp3", "exl3_gemv_w4a1fpn",
                  "exl3_gemv_w4a3fpn",
                  "exl3_cand_hist", "exl3_cand_sum", "exl3_cand_collect",
                  "exl3_argmax16_partial", "exl3_argmax16_final")


def create():
    runtime = CudaRuntime()
    runtime._streams = {}
    return runtime
