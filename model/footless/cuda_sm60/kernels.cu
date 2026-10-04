// Architecture kernels for the OrcaSAQ-2-27B (Qwen3.5 dense hybrid) runtime on
// sm_60 (Tesla P100 / GP100). The exl3 fused-decode GEMV lives in
// `kernels_exl3.cu` — it is a different workstream's file, and nothing here
// decodes a trellis.
//
// Layout contract:  cuda_sm60/_build/DESIGN.md
// Model math:       _reference/ARCHITECTURE.md  (normative — "Traps, in one
//                   place" is the list this file is written against)
// Precedent/style:  models/Xing4.0-29B-A4B/footless/cuda_sm60/kernels.cu
//
// ----------------------------------------------------------------------------
// INTERFACE CONVENTIONS (DESIGN.md "Naming used in the kernel signatures")
// ----------------------------------------------------------------------------
//   * `layer` is the ORDINAL within its kind (0..47 GDN, 0..15 full attention),
//     never the block index. Kernels that address per-layer state take it
//     explicitly; the rest take a pointer the caller has already offset.
//   * `pos0` is the position of the first token of this call. RoPE uses
//     pos0+t; the causal conv reads pos0 == 0 as "no history".
//   * The chunk dimension is gridDim.y: one token (or one query tile, see
//     attention_prefill) per column. Every kernel reads the token index from
//     blockIdx.y unless it is a flat elementwise kernel.
//   * All geometry is an argument. The `#define`s below are IMPLEMENTATION
//     BOUNDS (block sizes, shared-memory tile extents, register-array extents),
//     never geometry, and every kernel that indexes by one of them guards it
//     first: MODEL_OPTIMIZE.md trap 5, "fixed-size private arrays silently bind
//     a dimension; guard or assert the bound". A guard that cannot be satisfied
//     returns without writing, so the runtime must check the geometry it
//     launches with (the precedent has `_check_geometry` for exactly this).
//   * Pointers are `const half* __restrict__` etc. No struct-by-value
//     parameters; every argument is an int/float/pointer, at most 16 per
//     kernel (the Xing bridge's `_ParamBlock` marshals those).
//
// ----------------------------------------------------------------------------
// PRECISION (DESIGN.md "Precision, deliberately chosen")
// ----------------------------------------------------------------------------
//   fp16: activations, norms, attention, KV cache, conv state.
//   fp32: A_log, dt_bias, a, b, beta, decay, GDN recurrent state, Hadamard.
//
//   * Norm GAMMAS are fp32 (rmsnorm1p's `w`, rmsnorm_gated's `w`). The task
//     pins rmsnorm1p's to fp32; the gated norm's is fp32 as well so that both
//     Gemma-style norms share one convention — the runtime must upload
//     `input_layernorm`, `post_attention_layernorm`, `model.norm`, `q_norm`,
//     `k_norm` and the GDN `norm` weights as fp32.
//   * `a` and `b` (the projections feeding gdn_scalars) are fp32, per
//     DESIGN.md's table ("A_log, dt_bias, a, b, beta, decay ... fp32"). The
//     runtime must emit those two projections in fp32.
//   * Division in the recurrences is `__fdiv_rn` (IEEE round-to-nearest), not
//     `/`: this file is built with --use_fast_math, which turns `/` into an
//     approximate reciprocal. model_optimize.md calls this out — "if a
//     precision test fails, check that before suspecting your math".
//
// ----------------------------------------------------------------------------
// WHAT EACH KERNEL ASSUMES ABOUT ITS BUFFERS (the internal contract)
// ----------------------------------------------------------------------------
//   Activations are fp16 rows. `rows` = rows per token, `pitch` = elements per
//   row; a token's slab is therefore rows*pitch contiguous elements and row r
//   of token t starts at (t*rows + r)*pitch. This covers every row shape in
//   this model: a 5120-wide block norm (rows=1, pitch=5120), a per-head 256
//   q/k norm (rows=24/4, pitch=256 packed, or rows=24, pitch=512 inside the
//   interleaved q|gate buffer), the GDN q/k norms (rows=16, pitch=128), the
//   GDN output norm (rows=48, pitch=128).
//
//   Full attention layer buffers:
//     qg    [T, 24, 512]  fp16  q_proj output for this layer, per head
//                               [q(256) | gate(256)] — trap 7. rope_partial
//                               rotates the q half in place, the attention
//                               kernels read q from it with q_pitch=512, and
//                               attn_gate reads the gate half with g_off=256.
//     k     [T, 4, 256]   fp16  k_proj output, contiguous; RoPE'd in place,
//                               then kv_store'd.
//     v     [T, 4, 256]   fp16  v_proj output, raw.
//     kcache/vcache [16, max_pos, 4, 256] fp16, position-addressable so
//                               truncate_state is free: row (layer, pos) of a
//                               kv head is at
//                               ((layer*max_pos + pos)*n_kv_heads + h)*head_dim.
//                               One buffer each; k holds POST-rope k.
//     z     [T, 24, 256]  fp16 (or qg itself with z_pitch=512, see attn_gate).
//
//   GDN layer buffers: q/k [T,16,128] fp16 (post-l2norm), v/z [T,48,128] fp16,
//     y [T,48,128] fp16, beta/decay [T,48] fp32, S [48,128,128] fp32 per layer
//     (read and rewritten in place by gdn_scan), conv state [dim,3] fp16 per
//     segment (dim-first — trap 5).
//
// ----------------------------------------------------------------------------
// WHAT THE BRIDGE MUST PROVIDE (interface facts this file cannot enforce)
// ----------------------------------------------------------------------------
//   1. Norm gammas are fp32: `rmsnorm1p` and `rmsnorm_gated` take `const float*`.
//   2. `gdn_scalars` takes `a` and `b` as fp32 (DESIGN.md's precision table).
//      Whatever produces those two projections must emit fp32 for them.
//   3. conv1d_causal: st_in and st_out are DISTINCT buffers (the last token's
//      block writes the new state while the first tokens' blocks read the old
//      one; there is no ordering between them). Double-buffer the 2.9 MB.
//   4. attention_split's partials are `pacc[T][n_q_heads][split][head_dim]` fp32
//      and `pm`/`pd` the same without the last axis; `split` is gridDim.z and
//      the buffer must be sized for the launch's split. attention_merge is
//      launched with grid (n_q_heads, T, split) and reads gridDim.z as the
//      count; the extra z blocks recompute the same output redundantly (which
//      is harmless, not a bug, but launching with gridDim.z = 1 would read the
//      split count as 1).
//   5. attention_split/prefill require head_dim % 64 == 0 and head_dim <= 256
//      (the lane layout and the shared-memory merge tile); they return without
//      writing if not, so the geometry check is the runtime's to make.
//   6a. gdn_scan takes v_stride for `v` and y_stride for `y` separately: `v` is
//       a view into the 10240-wide conv'd qkv row, while `y` is packed
//       [T, 48, 128]. One stride for both scrambles every token but the first.
//   6b. `out_scale` is the reference's `scale` (head_k_dim**-0.5): the reference
//       normalises q, multiplies it by that scale, and only then reads the state
//       out (`o = S . (scale*q)`). Scaling after the dot is the same arithmetic
//       with one more bit of precision (q stays unscaled in fp16) — but it is
//       NOT the same as scaling the gated norm's gamma, which scales the whole
//       branch instead of the read-out. Applying it here keeps the gated norm's
//       eps at the config's 1e-6, which is what the reference does.
//   6. gdn_scan's shape is hard-bound to K == V == 128 with a 256-thread block
//      (two threads per state row, halves of K as adjacent lanes). A different
//      geometry returns unwritten, by design.
//
// ----------------------------------------------------------------------------
// VERIFIED, ON THIS DEVICE, AND HOW
// ----------------------------------------------------------------------------
//   Every kernel in this file has been run on the P100 against a float64 numpy
//   oracle at the model's shapes (tests/test_kernels.py, 37 tests including 12
//   subtests; measured max errors are printed by each test):
//
//     rmsnorm1p        1.9e-3 abs / 8.6e-4 rel (fp16, n=5120 and n=256)
//     rmsnorm_gated    3.5e-3 abs / 7.0e-3 rel
//     l2norm_scaled    1.2e-4 abs / 5.9e-4 rel
//     rope_partial     fp16 rounding (5e-4 rel) at pos 0..4096 and at 200000;
//                      dims [64,256) bit-exact; chunk positions bit-exact
//     kv_store         bit-exact, positions addressed by (ordinal, pos, head)
//     attention_split  1.2e-4 abs (T=1, 8 splits) and 9.8e-4 abs (T=8 chunked,
//                      which equals eight T=1 launches and matches the oracle)
//     attention_prefill 9.7e-4 abs (T=8, 13 with a ragged tile, 40 with pos0=3);
//                      agrees with the split path to 4.9e-4
//     attn_gate        9.4e-4 abs; the gate half of the q|gate buffer bit-exact
//     conv1d_causal    3.8e-3 abs / 4.8e-4 rel; state bit-exact; 4+4 == 8 and
//                      2+1 == 3 EXACT (y bit-equal, state bit-equal); T=1/T=2
//                      pad from the state tail exactly; three segments with
//                      their own row offsets in one 10240-wide row
//     gdn_scalars      beta 4.6e-7 rel, decay 5.6e-6 rel; the softplus branch
//                      pinned at both ends (x=100 with exp(A)=1e-3 -> 0.904837,
//                      x=-40 -> 1.0)
//     gdn_scan         2.8e-5 abs on y, 8.0e-8 abs on the state; 4+4 == 8 EXACT
//     embed_gather     4.6e-4 rel on int8*row-scale dequantisation
//     silu_mul         7.7e-3 abs on 34816 (odd n covered)
//     add_inplace      2.0e-3 abs
//     argmax_*         136 cases incl. ties inside one thread's stride, ties
//                      across blocks, ties between the two merge walks, +-0.0,
//                      an all -inf row and more blocks than elements
//
//   The tests are mutation-checked (`_build/mutate_check.py`): 38 deliberate
//   single-edit bugs, one per trap, each required to make its named test fail.
//   All 38 do. That pass found three real holes in the tests as first written —
//   assert_close was blind to NaN (a mutant that left its partials unwritten
//   passed), the rope-frequency test could not separate the exponent from the
//   pairing, and two mutation entries were numerically equivalent rather than
//   observable — which is the point of doing it.
//
//   NOT verified here: nothing in this file is untested, but three things are
//   weaker than they look. (a) attention_prefill has been exercised to T=40
//   (two full chunks plus a ragged tile) and pos0>0, not at long context, so
//   the KV chunk loop's behaviour past a few thousand rows is unmeasured.
//   (b) gdn_scan's numerical drift over thousands of sequential tokens is not
//   measured (T=128 is the longest run here) — the arithmetic is one token's
//   worth, so the risk is accumulation, not logic. (c) embed_gather is tested
//   against a 64-row table, not the 248320-row one (a 1.27 GB upload would
//   dominate the suite); the row addressing is the same code either way.
// ----------------------------------------------------------------------------

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// implementation bounds (guarded at every use; see the header)
#define ATT_WARPS 8            // warps per attention block
#define ATT_THREADS (ATT_WARPS * 32)
#define ATT_NL2_MAX 4          // half2 groups a lane holds: head_dim/64 <= 4
#define ATT_HD_MAX 256         // head_dim <= 256 (shared-memory merge tile)
#define APF_BQ 8               // queries (one warp each) per attention_prefill block
#define APF_BK 32              // kv rows staged in shared memory per chunk
#define APF_THREADS 256
#define APF_TILE (APF_BK * ATT_HD_MAX)     // halves staged: 16 KB at 32x256
#define GDN_THREADS 256
#define GDN_KH_MAX 64          // K/2 columns in registers: K <= 128
#define GDN_SMEM_K 256         // K <= 256 (staged k and q)
#define AMAX_THREADS 256
#define AMAX_NEG_INF __int_as_float(0xff800000)

// fp16 tops out at 65504; rounding a larger value into it yields infinity, and
// one infinity destroys the rest of the forward (the next norm multiplies it by
// zero, a sum of two is NaN). The reference runs these activations in bf16,
// whose range is fp32's; fp16's narrower range is this package's choice, so a
// value that leaves it is pinned to the largest magnitude the storage holds
// instead of becoming infinite. NaN keeps its meaning: this only pins the
// range. Same helper, same reason, as the Xing kernels.
__device__ __forceinline__ __half f16_sat(float v) {
    return __float2half(v != v ? v : fminf(fmaxf(v, -65504.f), 65504.f));
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

// Every lane gets the total (the attention score is needed by all 32 lanes of
// the row's warp, not just lane 0).
__device__ __forceinline__ float warp_reduce_all(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Whole-block sum, two __syncthreads. `red` is caller-owned shared memory of at
// least 32 floats; the result is broadcast to every thread.
__device__ __forceinline__ float block_reduce_sum(float v, float* red) {
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    v = warp_reduce_sum(v);
    if (lane == 0) red[wid] = v;
    __syncthreads();
    if (wid == 0) {
        v = (lane < (int)(blockDim.x >> 5)) ? red[lane] : 0.f;
        v = warp_reduce_sum(v);
        red[0] = v;
    }
    __syncthreads();
    return red[0];
}

// silu and sigmoid in fp32 with a PRECISE divide. Under --use_fast_math `/`
// becomes an approximate reciprocal; these two feed the recurrence (silu gates
// the GDN output norm, sigmoid the attention gate), so they do not get to
// drift by 2 ulp for free. Past x = 88, __expf(-x) underflows to 0 and both
// saturate at 0 / 1, which is what the activation does.
__device__ __forceinline__ float silu_f(float x) { return __fdiv_rn(x, 1.f + __expf(-x)); }
__device__ __forceinline__ float sigmoid_f(float x) { return __fdiv_rn(1.f, 1.f + __expf(-x)); }

// ============================================================================
// Norms
// ============================================================================

// GemmaRMSNorm: y = x * rsqrt(mean(x^2) + eps) * (1 + w).
//
// MEAN-space eps (ss/n + eps), and the (1 + w) offset — the weight is a delta
// from 1, initialised to zeros. This is a different convention from
// rmsnorm_gated below (plain gamma) and from l2norm_scaled (SUM-space eps):
// mixing them is an O(1) error, so all three live under different names.
//
// rows/pitch: see the header. q/k_norm pass rows=24 (or 4) and pitch=256 for a
// packed buffer, or pitch=512 with the q half of the interleaved q|gate buffer.
// A zero row cannot NaN: mean 0 -> rsqrt(eps) -> the row zeroes out.
//
// grid (rows, T), block 256.
extern "C" __global__ void rmsnorm1p(const half* __restrict__ x,
                                     const float* __restrict__ w,
                                     half* __restrict__ y,
                                     int n, int rows, int pitch, float eps) {
    __shared__ float red[32];
    const int t = blockIdx.y;
    const int r = blockIdx.x;
    const half* xr = x + ((size_t)t * rows + r) * pitch;
    half* yr = y + ((size_t)t * rows + r) * pitch;

    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = __half2float(__ldg(&xr[i]));
        ss = fmaf(v, v, ss);
    }
    const float inv = rsqrtf(__fdiv_rn(block_reduce_sum(ss, red), (float)n) + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        yr[i] = f16_sat(__half2float(__ldg(&xr[i])) * inv
                        * (1.f + __ldg(&w[i])));
}

// The residual add fused with the rmsnorm1p that follows it:
//     x <- fp16(x + p);  y = rmsnorm1p(x, w)
// The add rounds exactly as add_inplace does and the output expression is
// rmsnorm1p's; what changes is the work split: one thread per 8 contiguous
// elements (16-byte loads and stores), so the sum of squares is accumulated in
// a different order than rmsnorm1p's 256-thread stride (fp32 rounding only).
// One launch instead of two and one pass instead of two; a decode step's 128
// residual adds went 15.5 + 5.5 us -> ~16 us with the stride kept, and this
// shape removes the 40 scalar round trips that kept it there.
// grid (T), block n/8 (640 for 5120); n a multiple of 8, n/8 <= 1024.
extern "C" __global__ void __launch_bounds__(1024)
add_rmsnorm1p(half* __restrict__ x, const half* __restrict__ p,
              const float* __restrict__ w, half* __restrict__ y,
              int n, float eps) {
    __shared__ float red[32];
    if ((n & 7) || (int)blockDim.x * 8 != n) return;
    const int t = blockIdx.x;
    const size_t base = (size_t)t * n + (size_t)threadIdx.x * 8;
    uint4 xv = *reinterpret_cast<const uint4*>(x + base);
    const uint4 pv = __ldg(reinterpret_cast<const uint4*>(p + base));
    const float4 w0 = __ldg(reinterpret_cast<const float4*>(w + threadIdx.x * 8));
    const float4 w1 = __ldg(reinterpret_cast<const float4*>(w + threadIdx.x * 8 + 4));
    half2* xh = reinterpret_cast<half2*>(&xv);
    const half2* ph = reinterpret_cast<const half2*>(&pv);
    float v[8];
    float ss = 0.f;
#pragma unroll
    for (int q = 0; q < 4; q++) {
        const float2 a = __half22float2(xh[q]);
        const float2 b = __half22float2(ph[q]);
        xh[q] = __floats2half2_rn(a.x + b.x, a.y + b.y);
        const float2 c = __half22float2(xh[q]);
        v[2 * q] = c.x;
        v[2 * q + 1] = c.y;
        ss = fmaf(c.x, c.x, ss);
        ss = fmaf(c.y, c.y, ss);
    }
    *reinterpret_cast<uint4*>(x + base) = xv;
    const float inv = rsqrtf(__fdiv_rn(block_reduce_sum(ss, red), (float)n) + eps);
    const float wv[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
    uint4 yv;
    half* yh = reinterpret_cast<half*>(&yv);
#pragma unroll
    for (int q = 0; q < 8; q++) yh[q] = f16_sat(v[q] * inv * (1.f + wv[q]));
    *reinterpret_cast<uint4*>(y + base) = yv;
}

// RMSNormGated (the GDN output norm), a DIFFERENT convention on purpose:
//
//     y = (o * rsqrt(mean(o^2) + eps) * w) * silu(z)
//
//   * plain gamma, NOT (1 + w): this is vLLM's RMSNormGated, whose weight is
//     initialised to ONES, not zeros. Using GemmaRMSNorm's (1+w) here would
//     scale these rows by ~2 and is the O(1) error DESIGN.md warns about.
//   * mean-space eps, same as rmsnorm1p.
//   * norm_before_gate=True: the rms is taken over `o` alone, and silu(z) is
//     multiplied AFTER the normalisation. (norm_before_gate=False would
//     normalise o*silu(z) instead — a different function.)
//   * one 128-long gamma shared by all 48 v-heads: rows = tokens*48, n = 128.
//
// grid (48, T), block 128.
extern "C" __global__ void rmsnorm_gated(const half* __restrict__ o,
                                         const half* __restrict__ z,
                                         const float* __restrict__ w,
                                         half* __restrict__ y,
                                         int n, int rows, int pitch, float eps) {
    __shared__ float red[32];
    const int t = blockIdx.y;
    const int r = blockIdx.x;
    const half* orow = o + ((size_t)t * rows + r) * pitch;
    const half* zrow = z + ((size_t)t * rows + r) * pitch;
    half* yrow = y + ((size_t)t * rows + r) * pitch;

    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = __half2float(__ldg(&orow[i]));
        ss = fmaf(v, v, ss);
    }
    const float inv = rsqrtf(__fdiv_rn(block_reduce_sum(ss, red), (float)n) + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float nn = __half2float(__ldg(&orow[i])) * inv * __ldg(&w[i]);
        const float g = __half2float(__ldg(&zrow[i]));
        yrow[i] = f16_sat(nn * silu_f(g));
    }
}

// L2 normalisation with a scale, SUM-space eps — the opposite eps convention
// from the two RMSNorms above:
//
//     y = (x / sqrt(sum(x^2) + eps)) * scale
//
// The reference divides first and multiplies by the scale after
// (fused_recurrent.py: `b_q = b_q / sqrt(sum + 1e-6)`, then `b_q *= scale`), and
// here the scale is fused in so the GDN's q scale (128**-0.5) and k's neutral
// 1.0 are one code path: the runtime launches this twice per layer, once for
// the q rows with scale = 128**-0.5 and once for the k rows with scale = 1.0.
// A zero row yields 0 (eps is inside the sqrt, so the denominator is finite).
//
// grid (rows, T), block 128.
extern "C" __global__ void l2norm_scaled(const half* __restrict__ x,
                                         half* __restrict__ y,
                                         int n, int rows, int pitch,
                                         float eps, float scale) {
    __shared__ float red[32];
    const int t = blockIdx.y;
    const int r = blockIdx.x;
    const half* xr = x + ((size_t)t * rows + r) * pitch;
    half* yr = y + ((size_t)t * rows + r) * pitch;

    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = __half2float(__ldg(&xr[i]));
        ss = fmaf(v, v, ss);
    }
    const float den = sqrtf(block_reduce_sum(ss, red) + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        yr[i] = f16_sat(__fdiv_rn(__half2float(__ldg(&xr[i])), den) * scale);
}

// ============================================================================
// RoPE, KV cache, attention
// ============================================================================

// Partial NeoX RoPE, in place, on q and k before the KV store.
//
// The first rope_dim of head_dim rotate, in (d, d + rope_dim/2) pairs; dims
// rope_dim..head_dim-1 are never touched (bit for bit — the test asserts the
// exact fp16 words). The frequency is theta^(-2j/rope_dim) over the ROTARY
// width (32 pairs here), NOT theta^(-2j/head_dim): frequencies built over the
// full 256 would rotate 64 dims at roughly half the correct rate, which is the
// classic bug this model's 0.25 partial-rotary factor invites.
//
// The angle is computed in DOUBLE: `(float)(pos) * inv_freq_fp32` loses the
// argument at long positions (a float32 200000 has 0.0156 of absolute
// resolution, so the reference's own fp32 angle carries ~1.5% of error in
// cos/sin there). Computing the frequency and the trigonometry in fp64 keeps
// this kernel at the accuracy of the float64 oracle it is tested against;
// `cosf`/`sinf` cannot be used here because --use_fast_math replaces them with
// the range-reduced approximations, which for an argument of O(1e5) are not
// merely approximate. Cost is one pow + one sincos per (head, token) per lane,
// ~50 us over a 2048-token prefill.
//
// One warp per head, lane j handles pair j (rope_dim/2 = 32 pairs = 32 lanes).
// q and k are handled in one launch: heads [0, n_q_heads) are q, the rest k.
// q's per-token slab is n_q_heads*q_pitch (512 inside the interleaved q|gate
// buffer, n_q_heads = 24), k's is n_k_heads*k_pitch (256, 4 heads).
//
// grid (ceil((n_q_heads+n_k_heads)/8), T), block 256.
extern "C" __global__ void rope_partial(half* __restrict__ q,
                                        half* __restrict__ k,
                                        int n_q_heads, int n_k_heads,
                                        int head_dim, int rope_dim,
                                        int q_pitch, int k_pitch,
                                        int pos0, float theta) {
    const int t = blockIdx.y;
    const int head = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (head >= n_q_heads + n_k_heads) return;
    if (rope_dim <= 0 || rope_dim > head_dim || (rope_dim & 1) != 0) return;

    const bool is_q = head < n_q_heads;
    const int h = is_q ? head : head - n_q_heads;
    const int nheads = is_q ? n_q_heads : n_k_heads;
    const int pitch = is_q ? q_pitch : k_pitch;
    half* row = (is_q ? q : k) + (size_t)t * nheads * pitch + (size_t)h * pitch;

    const int lane = threadIdx.x & 31;
    const int half_r = rope_dim >> 1;
    const double th = (double)theta;
    const int pos = pos0 + t;
    for (int j = lane; j < half_r; j += 32) {
        const double inv = pow(th, -2.0 * (double)j / (double)rope_dim);
        const double ang = (double)pos * inv;
        double s, c;
        sincos(ang, &s, &c);
        const float cf = (float)c;
        const float sf = (float)s;
        const float x0 = __half2float(row[j]);
        const float x1 = __half2float(row[j + half_r]);
        row[j] = f16_sat(x0 * cf - x1 * sf);
        row[j + half_r] = f16_sat(x1 * cf + x0 * sf);
    }
}

// Append post-RoPE k and raw v into the fp16 KV cache.
//
// The cache is [n_layers, max_pos, n_kv_heads, head_dim] per plane, addressed
// by (ordinal, pos, kv_head, dim) so `truncate_state` is free: it moves the
// sequence length, nothing else. `layer` is the ORDINAL within the full-
// attention kind (0..15), not the block index.
//
// grid (n_kv_heads, T), block 256.
extern "C" __global__ void kv_store(const half* __restrict__ k,
                                    const half* __restrict__ v,
                                    half* __restrict__ kcache,
                                    half* __restrict__ vcache,
                                    int layer, int pos0, int n_kv_heads,
                                    int head_dim, int max_pos) {
    const int t = blockIdx.y;
    const int h = blockIdx.x;
    const int pos = pos0 + t;
    if (pos >= max_pos) return;                 // never write past the cache
    const long long stride = (long long)n_kv_heads * head_dim;
    const long long off = ((long long)layer * max_pos + pos) * stride
                          + (long long)h * head_dim;
    const half* ks = k + ((size_t)t * n_kv_heads + h) * head_dim;
    const half* vs = v + ((size_t)t * n_kv_heads + h) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        kcache[off + d] = __ldg(&ks[d]);
        vcache[off + d] = __ldg(&vs[d]);
    }
}

// kv_store into an int8 cache (KV8): one buffer a layer, rows of int8
// [max_pos][n_kv][head_dim] then fp16 scales [max_pos][n_kv][head_dim / 32]
// (`layer` must be 0). Each 32-dim block -- one warp -- shares its absmax / 127
// as an fp16 scale and stores rint(x / scale) in [-127, 127].
// grid (n_kv, T), block head_dim (a multiple of 32).
extern "C" __global__ void kv_store_q8(const half* __restrict__ k,
                                       const half* __restrict__ v,
                                       signed char* __restrict__ kcache,
                                       signed char* __restrict__ vcache,
                                       int layer, int pos0, int n_kv_heads,
                                       int head_dim, int max_pos) {
    const int t = blockIdx.y;
    const int h = blockIdx.x;
    const int pos = pos0 + t;
    if (pos >= max_pos || layer != 0) return;
    const int d = threadIdx.x;
    const long long stride = (long long)n_kv_heads * head_dim;
    const long long off = (long long)pos * stride + (long long)h * head_dim + d;
    const long long soff = ((long long)pos * n_kv_heads + h) * (head_dim >> 5) + (d >> 5);
    const size_t src = ((size_t)t * n_kv_heads + h) * head_dim + d;
#pragma unroll
    for (int which = 0; which < 2; which++) {
        signed char* cache = which ? vcache : kcache;
        const float x = __half2float(which ? v[src] : k[src]);
        float a = fabsf(x);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
        const half sh = __float2half_rn(a / 127.0f);
        const float scale = __half2float(sh);
        const float q = scale > 0.f ? fminf(fmaxf(rintf(x / scale), -127.f), 127.f) : 0.f;
        cache[off] = (signed char)(int)q;
        if ((d & 31) == 0)
            reinterpret_cast<half*>(cache + (long long)max_pos * stride)[soff] = sh;
    }
}

// Causal GQA attention, decode shape: SPLIT over the KV context, one partial
// per (query, q_head, split) plus a merge.
//
// Why split: at T=1 one block per (query, head) is 24 blocks on a 56-SM part.
// Splitting the context 8 ways raises it to 192 blocks, which is the
// flash-decoding shape (a plain full-context version is the first thing that
// works and the wrong thing to ship). The partial carries (m, d, acc) of an
// online softmax, unnormalised; `attention_merge` combines them exactly as the
// online recurrence would have.
//
// The split a token asks for is computed from ITS OWN seqlen, not the launch's
// maximum:
//
//     St = clamp(seqlen / split_rows, 1, split)
//
// A block with s >= St writes an EMPTY partial (m = -1e30, d = 0, acc = 0)
// rather than skipping the write: the merge reads all `split` of them in order
// and exp(-1e30 - M) = 0 contributes nothing to either the denominator or the
// weighted sum. That is what makes a chunk's first tokens (with an own-seqlen
// St of 1) get the arithmetic of a single-partial launch, so a chunked step is
// the concatenation of single-token steps (the precedent measured a chunked/
// sequential divergence of exactly this kind when the split used the launch's
// maximum instead).
//
// Geometry: 24 q heads, 4 kv heads, head_dim 256, scale 256**-0.5, causal.
//   * each q head h reads kv head h/(n_q_heads/n_kv_heads) = h/6
//   * one warp owns the block's rows: warp w takes rows lo+w, lo+w+8, ...
//   * a lane holds head_dim/64 half2 groups (head_dim=256 -> 4 groups = 8 dims)
//     which makes the k and v row reads fully coalesced
//   * the 8 warps' per-warp partials are merged in shared memory at the end, so
//     the block emits ONE partial (not 8): the host allocates split partials
//     per (token, head), not split*8.
// All 32 lanes see the same score (warp_reduce_all), so the running max and
// denominator are identical across lanes and only the accumulator is per-lane.
//
// Partial layout: pacc[t][head][s][head_dim] fp32, pm/pd [t][head][s] fp32.
// grid (n_q_heads, T, split), block 256.
extern "C" __global__ void attention_split(const half* __restrict__ q,
                                           const half* __restrict__ kcache,
                                           const half* __restrict__ vcache,
                                           float* __restrict__ pacc,
                                           float* __restrict__ pm,
                                           float* __restrict__ pd,
                                           int layer, int n_q_heads, int n_kv_heads,
                                           int head_dim, int q_pitch, int pos0,
                                           int max_pos, float scale, int split_rows) {
    const int head = blockIdx.x;
    const int t = blockIdx.y;
    const int s = blockIdx.z;
    const int split = gridDim.z;
    if (n_q_heads % n_kv_heads != 0) return;
    if ((head_dim & 63) != 0) return;             // lane groups need head_dim%64==0
    if ((head_dim >> 6) > ATT_NL2_MAX) return;    // register bound
    if (head_dim > ATT_HD_MAX) return;            // smem merge bound

    const int hkv = head / (n_q_heads / n_kv_heads);
    const int seqlen = pos0 + t + 1;
    int st = split_rows > 0 ? seqlen / split_rows : split;
    st = st < 1 ? 1 : (st > split ? split : st);
    const int empty = (s >= st);
    const int lo = empty ? 0 : (int)((long long)seqlen * s / st);
    const int hi = empty ? 0 : (int)((long long)seqlen * (s + 1) / st);

    const int nl2 = head_dim >> 6;
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    const long long kv_stride = (long long)n_kv_heads * head_dim;
    const half* kbase = kcache + ((long long)layer * max_pos) * kv_stride + (long long)hkv * head_dim;
    const half* vbase = vcache + ((long long)layer * max_pos) * kv_stride + (long long)hkv * head_dim;

    const half2* q2 = reinterpret_cast<const half2*>(
        q + (size_t)t * n_q_heads * q_pitch + (size_t)head * q_pitch);
    float2 qf[ATT_NL2_MAX];
#pragma unroll
    for (int u = 0; u < ATT_NL2_MAX; u++)
        qf[u] = u < nl2 ? __half22float2(__ldg(&q2[lane + 32 * u]))
                        : make_float2(0.f, 0.f);

    float2 acc[ATT_NL2_MAX];
#pragma unroll
    for (int u = 0; u < ATT_NL2_MAX; u++) acc[u] = make_float2(0.f, 0.f);
    float m = -1e30f;
    float d = 0.f;

    for (int row = lo + wid; row < hi; row += nwarps) {
        const half2* k2 = reinterpret_cast<const half2*>(kbase + (long long)row * kv_stride);
        float sc = 0.f;
#pragma unroll
        for (int u = 0; u < ATT_NL2_MAX; u++) {
            if (u < nl2) {
                const float2 kk = __half22float2(__ldg(&k2[lane + 32 * u]));
                sc = fmaf(qf[u].x, kk.x, sc);
                sc = fmaf(qf[u].y, kk.y, sc);
            }
        }
        const float sco = warp_reduce_all(sc) * scale;

        float p;
        if (sco > m) {
            const float c = __expf(m - sco);
            m = sco;
            d = d * c + 1.f;
#pragma unroll
            for (int u = 0; u < ATT_NL2_MAX; u++) {
                acc[u].x *= c;
                acc[u].y *= c;
            }
            p = 1.f;
        } else {
            p = __expf(sco - m);
            d += p;
        }
        const half2* v2 = reinterpret_cast<const half2*>(vbase + (long long)row * kv_stride);
#pragma unroll
        for (int u = 0; u < ATT_NL2_MAX; u++) {
            if (u < nl2) {
                const float2 vv = __half22float2(__ldg(&v2[lane + 32 * u]));
                acc[u].x = fmaf(p, vv.x, acc[u].x);
                acc[u].y = fmaf(p, vv.y, acc[u].y);
            }
        }
    }

    // per-warp partials -> one block partial, in shared memory
    __shared__ float sm_acc[ATT_WARPS * ATT_HD_MAX];
    __shared__ float sm_m[ATT_WARPS];
    __shared__ float sm_d[ATT_WARPS];
    if (lane == 0) {
        sm_m[wid] = m;
        sm_d[wid] = d;
    }
#pragma unroll
    for (int u = 0; u < ATT_NL2_MAX; u++) {
        if (u < nl2) {
            sm_acc[wid * head_dim + 2 * (lane + 32 * u)] = acc[u].x;
            sm_acc[wid * head_dim + 2 * (lane + 32 * u) + 1] = acc[u].y;
        }
    }
    __syncthreads();

    float M = -1e30f;
    for (int w = 0; w < nwarps; w++) M = fmaxf(M, sm_m[w]);
    float num = 0.f;
    float den = 0.f;
    if (threadIdx.x < head_dim) {
        for (int w = 0; w < nwarps; w++) {
            const float e = __expf(sm_m[w] - M);
            num = fmaf(e, sm_acc[w * head_dim + threadIdx.x], num);
            den = fmaf(e, sm_d[w], den);
        }
    }
    const size_t part = ((size_t)t * n_q_heads + head) * split + s;
    if (threadIdx.x < head_dim) pacc[part * head_dim + threadIdx.x] = num;
    if (threadIdx.x == 0) {
        pm[part] = M;
        pd[part] = den;
    }
}

// attention_split for all G q heads of one kv head at once (GQA). Every
// per-head operation is attention_split's, in the same order -- the same rows
// per warp (lo + wid + 8k), the same lane split of the dot product, the same
// online softmax and the same per-warp merge -- so each head's partial is
// bit-identical to attention_split's; what changes is that a k row and a v row
// are read ONCE for the G heads that share them instead of G times (6x less
// cache traffic, which is what decode attention costs at long contexts).
// grid (n_kv_heads, T, split), block ATT_THREADS. Writes the same (pacc, pm,
// pd) layout, so attention_merge is unchanged.
template <int G>
static __device__ __forceinline__ void attention_split_gqa_body(
        const half* __restrict__ q, const half* __restrict__ kcache,
        const half* __restrict__ vcache, float* __restrict__ pacc,
        float* __restrict__ pm, float* __restrict__ pd, int layer, int n_q_heads,
        int n_kv_heads, int head_dim, int q_pitch, int pos0, int max_pos, float scale,
        int split_rows) {
    const int hkv = blockIdx.x;
    const int t = blockIdx.y;
    const int s = blockIdx.z;
    const int split = gridDim.z;
    const int seqlen = pos0 + t + 1;
    int st = split_rows > 0 ? seqlen / split_rows : split;
    st = st < 1 ? 1 : (st > split ? split : st);
    const int empty = (s >= st);
    const int lo = empty ? 0 : (int)((long long)seqlen * s / st);
    const int hi = empty ? 0 : (int)((long long)seqlen * (s + 1) / st);
    const int nl2 = head_dim >> 6;
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    const long long kv_stride = (long long)n_kv_heads * head_dim;
    const half* kbase = kcache + ((long long)layer * max_pos) * kv_stride + (long long)hkv * head_dim;
    const half* vbase = vcache + ((long long)layer * max_pos) * kv_stride + (long long)hkv * head_dim;

    float2 qf[G][ATT_NL2_MAX];
    float2 acc[G][ATT_NL2_MAX];
    float m[G], d[G];
#pragma unroll
    for (int h = 0; h < G; h++) {
        const half2* q2 = reinterpret_cast<const half2*>(
            q + (size_t)t * n_q_heads * q_pitch + (size_t)(hkv * G + h) * q_pitch);
#pragma unroll
        for (int u = 0; u < ATT_NL2_MAX; u++) {
            qf[h][u] = u < nl2 ? __half22float2(__ldg(&q2[lane + 32 * u]))
                               : make_float2(0.f, 0.f);
            acc[h][u] = make_float2(0.f, 0.f);
        }
        m[h] = -1e30f;
        d[h] = 0.f;
    }

    for (int row = lo + wid; row < hi; row += nwarps) {
        const half2* k2 = reinterpret_cast<const half2*>(kbase + (long long)row * kv_stride);
        const half2* v2 = reinterpret_cast<const half2*>(vbase + (long long)row * kv_stride);
        float2 kk[ATT_NL2_MAX], vv[ATT_NL2_MAX];
#pragma unroll
        for (int u = 0; u < ATT_NL2_MAX; u++) {
            kk[u] = u < nl2 ? __half22float2(__ldg(&k2[lane + 32 * u])) : make_float2(0.f, 0.f);
            vv[u] = u < nl2 ? __half22float2(__ldg(&v2[lane + 32 * u])) : make_float2(0.f, 0.f);
        }
#pragma unroll
        for (int h = 0; h < G; h++) {
            float sc = 0.f;
#pragma unroll
            for (int u = 0; u < ATT_NL2_MAX; u++) {
                if (u < nl2) {
                    sc = fmaf(qf[h][u].x, kk[u].x, sc);
                    sc = fmaf(qf[h][u].y, kk[u].y, sc);
                }
            }
            const float sco = warp_reduce_all(sc) * scale;
            float p;
            if (sco > m[h]) {
                const float c = __expf(m[h] - sco);
                m[h] = sco;
                d[h] = d[h] * c + 1.f;
#pragma unroll
                for (int u = 0; u < ATT_NL2_MAX; u++) {
                    acc[h][u].x *= c;
                    acc[h][u].y *= c;
                }
                p = 1.f;
            } else {
                p = __expf(sco - m[h]);
                d[h] += p;
            }
#pragma unroll
            for (int u = 0; u < ATT_NL2_MAX; u++) {
                if (u < nl2) {
                    acc[h][u].x = fmaf(p, vv[u].x, acc[h][u].x);
                    acc[h][u].y = fmaf(p, vv[u].y, acc[h][u].y);
                }
            }
        }
    }

    // attention_split's per-warp merge, one head at a time through one buffer
    __shared__ float sm_acc[ATT_WARPS * ATT_HD_MAX];
    __shared__ float sm_m[ATT_WARPS];
    __shared__ float sm_d[ATT_WARPS];
#pragma unroll
    for (int h = 0; h < G; h++) {
        if (lane == 0) {
            sm_m[wid] = m[h];
            sm_d[wid] = d[h];
        }
#pragma unroll
        for (int u = 0; u < ATT_NL2_MAX; u++) {
            if (u < nl2) {
                sm_acc[wid * head_dim + 2 * (lane + 32 * u)] = acc[h][u].x;
                sm_acc[wid * head_dim + 2 * (lane + 32 * u) + 1] = acc[h][u].y;
            }
        }
        __syncthreads();
        float M = -1e30f;
        for (int w = 0; w < nwarps; w++) M = fmaxf(M, sm_m[w]);
        float num = 0.f;
        float den = 0.f;
        if (threadIdx.x < head_dim) {
            for (int w = 0; w < nwarps; w++) {
                const float e = __expf(sm_m[w] - M);
                num = fmaf(e, sm_acc[w * head_dim + threadIdx.x], num);
                den = fmaf(e, sm_d[w], den);
            }
        }
        const size_t part = ((size_t)t * n_q_heads + hkv * G + h) * split + s;
        if (threadIdx.x < head_dim) pacc[part * head_dim + threadIdx.x] = num;
        if (threadIdx.x == 0) {
            pm[part] = M;
            pd[part] = den;
        }
        __syncthreads();
    }
}

#ifndef ATT_GQA_OCC
#define ATT_GQA_OCC 2
#endif
extern "C" __global__ void __launch_bounds__(ATT_THREADS, ATT_GQA_OCC)
attention_split_gqa(const half* __restrict__ q, const half* __restrict__ kcache,
                    const half* __restrict__ vcache, float* __restrict__ pacc,
                    float* __restrict__ pm, float* __restrict__ pd, int layer,
                    int n_q_heads, int n_kv_heads, int head_dim, int q_pitch, int pos0,
                    int max_pos, float scale, int split_rows) {
    if (n_q_heads % n_kv_heads != 0) return;
    if ((head_dim & 63) != 0 || (head_dim >> 6) > ATT_NL2_MAX || head_dim > ATT_HD_MAX) return;
    if ((int)blockDim.x != ATT_THREADS) return;
    switch (n_q_heads / n_kv_heads) {
    case 6: attention_split_gqa_body<6>(q, kcache, vcache, pacc, pm, pd, layer, n_q_heads,
                                        n_kv_heads, head_dim, q_pitch, pos0, max_pos, scale,
                                        split_rows); break;
    default: break;
    }
}

// The decode's per-head preparation in one launch (`attn_prep_q8`): what
// rmsnorm1p (q_norm, k_norm), rope_partial and kv_store_q8 do between the
// projections and the attention, a block a (head, token), one element a thread
// (head_dim = 256 = blockDim): the norm's single fmaf and block_reduce_sum, the
// rope pairs by warp 0 over the fp16-rounded normed row, and the int8 rows and
// scales by kv_store_q8's expressions -- the same values, written to the same
// places. grid (n_q_heads + n_kv_heads, T), block head_dim.
extern "C" __global__ void __launch_bounds__(256)
attn_prep_q8(half* __restrict__ qg, half* __restrict__ k, const half* __restrict__ v,
             const float* __restrict__ wq, const float* __restrict__ wk,
             signed char* __restrict__ kcache, signed char* __restrict__ vcache,
             int n_q_heads, int n_kv_heads, int head_dim, int q_pitch, int pos0,
             int max_pos, int rope_dim, float theta, float eps) {
    __shared__ float red[32];
    __shared__ half rs[256];
    if (head_dim != (int)blockDim.x || head_dim > 256) return;
    if (rope_dim <= 0 || rope_dim > head_dim || (rope_dim & 1) != 0) return;
    const int t = blockIdx.y, hd = blockIdx.x, d = threadIdx.x;
    const bool is_q = hd < n_q_heads;
    const int h = is_q ? hd : hd - n_q_heads;
    half* row = is_q ? qg + ((size_t)t * n_q_heads + h) * q_pitch
                     : k + ((size_t)t * n_kv_heads + h) * head_dim;
    // rmsnorm1p
    const float xv = __half2float(row[d]);
    const float ss = fmaf(xv, xv, 0.f);
    const float inv = rsqrtf(__fdiv_rn(block_reduce_sum(ss, red), (float)head_dim) + eps);
    rs[d] = f16_sat(xv * inv * (1.f + __ldg(&(is_q ? wq : wk)[d])));
    __syncthreads();
    // rope_partial, warp 0
    if (d < 32) {
        const int half_r = rope_dim >> 1;
        const double th = (double)theta;
        const int pos = pos0 + t;
        for (int j = d; j < half_r; j += 32) {
            const double invf = pow(th, -2.0 * (double)j / (double)rope_dim);
            const double ang = (double)pos * invf;
            double sn, cs;
            sincos(ang, &sn, &cs);
            const float cf = (float)cs;
            const float sf = (float)sn;
            const float x0 = __half2float(rs[j]);
            const float x1 = __half2float(rs[j + half_r]);
            rs[j] = f16_sat(x0 * cf - x1 * sf);
            rs[j + half_r] = f16_sat(x1 * cf + x0 * sf);
        }
    }
    __syncthreads();
    const half y = rs[d];
    row[d] = y;
    if (is_q) return;
    // kv_store_q8
    const int pos = pos0 + t;
    if (pos >= max_pos) return;
    const long long stride = (long long)n_kv_heads * head_dim;
    const long long off = (long long)pos * stride + (long long)h * head_dim + d;
    const long long soff = ((long long)pos * n_kv_heads + h) * (head_dim >> 5) + (d >> 5);
    const size_t src = ((size_t)t * n_kv_heads + h) * head_dim + d;
#pragma unroll
    for (int which = 0; which < 2; which++) {
        signed char* cache = which ? vcache : kcache;
        const float x = which ? __half2float(v[src]) : __half2float(y);
        float a = fabsf(x);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
        const half sh = __float2half_rn(a / 127.0f);
        const float scale = __half2float(sh);
        const float qv = scale > 0.f ? fminf(fmaxf(rintf(x / scale), -127.f), 127.f) : 0.f;
        cache[off] = (signed char)(int)qv;
        if ((d & 31) == 0)
            reinterpret_cast<half*>(cache + (long long)max_pos * stride)[soff] = sh;
    }
}

// attention_split_gqa over an int8 cache (KV8, kv_store_q8's layout; `layer`
// must be 0, head_dim 256). A lane owns 8 CONTIGUOUS dims (one 8-byte load of
// K and of V a row, inside one 32-dim scale block), so the block's scale
// multiplies the lane's partial dot product once and its p once; the partials
// and the merge are attention_split_gqa's.
template <int G>
static __device__ __forceinline__ void attention_split_gqa_q8_body(
        const half* __restrict__ q, const signed char* __restrict__ kcache,
        const signed char* __restrict__ vcache, float* __restrict__ pacc,
        float* __restrict__ pm, float* __restrict__ pd, int n_q_heads,
        int n_kv_heads, int head_dim, int q_pitch, int pos0, int max_pos, float scale,
        int split_rows) {
    const int hkv = blockIdx.x;
    const int t = blockIdx.y;
    const int s = blockIdx.z;
    const int split = gridDim.z;
    const int seqlen = pos0 + t + 1;
    int st = split_rows > 0 ? seqlen / split_rows : split;
    st = st < 1 ? 1 : (st > split ? split : st);
    const int empty = (s >= st);
    const int lo = empty ? 0 : (int)((long long)seqlen * s / st);
    const int hi = empty ? 0 : (int)((long long)seqlen * (s + 1) / st);
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    const long long kv_stride = (long long)n_kv_heads * head_dim;
    const int nsb = head_dim >> 5;
    const signed char* kb = kcache + (long long)hkv * head_dim + lane * 8;
    const signed char* vb = vcache + (long long)hkv * head_dim + lane * 8;
    const half* ksc = reinterpret_cast<const half*>(kcache + (long long)max_pos * kv_stride)
                      + hkv * nsb + (lane >> 2);
    const half* vsc = reinterpret_cast<const half*>(vcache + (long long)max_pos * kv_stride)
                      + hkv * nsb + (lane >> 2);
    const long long sstride = (long long)n_kv_heads * nsb;

    float qf[G][8], acc[G][8], m[G], d[G];
#pragma unroll
    for (int h = 0; h < G; h++) {
        const uint4 qv = __ldg(reinterpret_cast<const uint4*>(
            q + (size_t)t * n_q_heads * q_pitch + (size_t)(hkv * G + h) * q_pitch + lane * 8));
        const half2* q2 = reinterpret_cast<const half2*>(&qv);
#pragma unroll
        for (int u = 0; u < 4; u++) {
            const float2 f = __half22float2(q2[u]);
            qf[h][2 * u] = f.x;
            qf[h][2 * u + 1] = f.y;
        }
#pragma unroll
        for (int j = 0; j < 8; j++) acc[h][j] = 0.f;
        m[h] = -1e30f;
        d[h] = 0.f;
    }

    for (int row = lo + wid; row < hi; row += nwarps) {
        const uint2 kr = __ldg(reinterpret_cast<const uint2*>(kb + (long long)row * kv_stride));
        const uint2 vr = __ldg(reinterpret_cast<const uint2*>(vb + (long long)row * kv_stride));
        const float ks = __half2float(__ldg(ksc + (long long)row * sstride));
        const float vs = __half2float(__ldg(vsc + (long long)row * sstride));
        const signed char* kc = reinterpret_cast<const signed char*>(&kr);
        const signed char* vc = reinterpret_cast<const signed char*>(&vr);
        float kk[8], vv[8];
#pragma unroll
        for (int j = 0; j < 8; j++) {
            kk[j] = (float)kc[j];
            vv[j] = (float)vc[j];
        }
#pragma unroll
        for (int h = 0; h < G; h++) {
            float sc = 0.f;
#pragma unroll
            for (int j = 0; j < 8; j++) sc = fmaf(qf[h][j], kk[j], sc);
            const float sco = warp_reduce_all(sc * ks) * scale;
            float p;
            if (sco > m[h]) {
                const float c = __expf(m[h] - sco);
                m[h] = sco;
                d[h] = d[h] * c + 1.f;
#pragma unroll
                for (int j = 0; j < 8; j++) acc[h][j] *= c;
                p = 1.f;
            } else {
                p = __expf(sco - m[h]);
                d[h] += p;
            }
            const float pv = p * vs;
#pragma unroll
            for (int j = 0; j < 8; j++) acc[h][j] = fmaf(pv, vv[j], acc[h][j]);
        }
    }

    __shared__ float sm_acc[ATT_WARPS * ATT_HD_MAX];
    __shared__ float sm_m[ATT_WARPS];
    __shared__ float sm_d[ATT_WARPS];
#pragma unroll
    for (int h = 0; h < G; h++) {
        if (lane == 0) {
            sm_m[wid] = m[h];
            sm_d[wid] = d[h];
        }
#pragma unroll
        for (int j = 0; j < 8; j++) sm_acc[wid * head_dim + lane * 8 + j] = acc[h][j];
        __syncthreads();
        float M = -1e30f;
        for (int w = 0; w < nwarps; w++) M = fmaxf(M, sm_m[w]);
        float num = 0.f;
        float den = 0.f;
        if (threadIdx.x < head_dim) {
            for (int w = 0; w < nwarps; w++) {
                const float e = __expf(sm_m[w] - M);
                num = fmaf(e, sm_acc[w * head_dim + threadIdx.x], num);
                den = fmaf(e, sm_d[w], den);
            }
        }
        const size_t part = ((size_t)t * n_q_heads + hkv * G + h) * split + s;
        if (threadIdx.x < head_dim) pacc[part * head_dim + threadIdx.x] = num;
        if (threadIdx.x == 0) {
            pm[part] = M;
            pd[part] = den;
        }
        __syncthreads();
    }
}

extern "C" __global__ void __launch_bounds__(ATT_THREADS, ATT_GQA_OCC)
attention_split_gqa_q8(const half* __restrict__ q, const signed char* __restrict__ kcache,
                       const signed char* __restrict__ vcache, float* __restrict__ pacc,
                       float* __restrict__ pm, float* __restrict__ pd, int layer,
                       int n_q_heads, int n_kv_heads, int head_dim, int q_pitch, int pos0,
                       int max_pos, float scale, int split_rows) {
    if (layer != 0 || head_dim != 256 || head_dim > ATT_HD_MAX) return;
    if ((int)blockDim.x != ATT_THREADS) return;
    switch (n_q_heads / n_kv_heads) {
    case 6: attention_split_gqa_q8_body<6>(q, kcache, vcache, pacc, pm, pd, n_q_heads,
                                           n_kv_heads, head_dim, q_pitch, pos0, max_pos,
                                           scale, split_rows); break;
    default: break;
    }
}

// ============================================================================
// Decode attention over the int8 cache (`attention_dec_q8`)
// ============================================================================
// attention_split_gqa_q8 spent a warp a cache row: five shuffles, an exp and a
// branch per q head per row, ~120 GB/s at 64k. This kernel takes the cache in
// 64-row tiles, all 6 q heads of a kv head (and all T tokens of a verify) a
// block, so a tile is read once for every query row of the block:
//   * scores: thread (rp, part) dots rows rp and rp + 32 against the queries'
//     32 dims of scale block `part` (q staged in fp32, pre-scaled by
//     scale * log2 e), one scale multiply a row, three shuffles to sum parts;
//   * the tile's max per query row from per-warp maxima, online softmax in
//     exp2, P to shared memory; the row sums ride per warp;
//   * O += P V: thread (slot, dg) owns dims dg*4..+4 of rows slot + 4 i, V
//     dequantized exactly in fp32 (byte * fp16 scale); the four slots' sums
//     meet in shared memory at the end, in slot order.
// The kv range is split in whole tiles, and a token's split is a function of
// ITS OWN length only (tmin tiles a split at least, smax splits at most), and
// a token's arithmetic does not depend on which tokens share its block, so
// each row of a verify is bit-identical to the T = 1 call at its position. A
// tile outside a token's range adds p = 0 under a correction of exactly 1.
// Writes partials
// pacc [t][head][smax][256], pm (log2 units), pd for attention_dec_merge.
// The K of the next tile and the V of this one are in flight during the math.
// The entry takes one token a block (blockIdx.y): measured, three tokens a
// block (the tile read once) ran no faster at any depth than three blocks of
// one, which keep two blocks an SM and fill the card at short contexts.
// grid (n_kv_heads, tokens, >= the largest split), block 256; head_dim 256, G = 6.
#ifndef DD_OCC1
#define DD_OCC1 2
#endif
#define DD_BK 64

// four signed bytes -> four exact floats
__device__ __forceinline__ void dd_b4(unsigned int b, float (&f)[4]) {
    f[0] = (float)(int)(signed char)(b & 0xff);
    f[1] = (float)(int)(signed char)((b >> 8) & 0xff);
    f[2] = (float)(int)(signed char)((b >> 16) & 0xff);
    f[3] = (float)(int)(signed char)(b >> 24);
}

template <int T>
__device__ __forceinline__ void attention_dec_q8_body(
        const half* __restrict__ q, const signed char* __restrict__ kcache,
        const signed char* __restrict__ vcache, float* __restrict__ pacc,
        float* __restrict__ pm, float* __restrict__ pd, int n_q_heads, int n_kv_heads,
        int q_pitch, int pos0, int max_pos, float sl2, int tmin, int smax) {
    constexpr int G = 6, R = T * G, QP = 36;          // q pitch per 32-dim part (floats)
    constexpr int PVD = 4;                            // dims a thread in P V
    constexpr int NS = PVD;                           // row slots
    constexpr int RPS = DD_BK / NS;                   // rows a slot
    constexpr int NG = 256 / PVD;                     // dim groups
    __shared__ __align__(16) float qs[R * 8 * QP];
    __shared__ __align__(16) float psb[2][R * DD_BK];
    __shared__ __align__(16) float tmax[8 * R];
    __shared__ __align__(16) float psw[8 * R];
    __shared__ __align__(16) float vss[DD_BK * 8];
    const int hkv = blockIdx.x, s = blockIdx.z, tb = blockIdx.y * T;   // first token
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const long long kvs = (long long)n_kv_heads * 256;
    int tlo[T], thi[T];
    int ulo = 1 << 30, uhi = 0;
#pragma unroll
    for (int t = 0; t < T; t++) {
        const int len = pos0 + tb + t + 1;
        const int nkt = (len + DD_BK - 1) / DD_BK;
        int st = (nkt + tmin - 1) / tmin;
        st = st > smax ? smax : st;
        tlo[t] = s < st ? (int)((long long)nkt * s / st) : 0;
        thi[t] = s < st ? (int)((long long)nkt * (s + 1) / st) : 0;
        if (thi[t] > tlo[t]) { ulo = min(ulo, tlo[t]); uhi = max(uhi, thi[t]); }
    }
    if (uhi <= ulo) return;
    for (int i = tid; i < R * 64; i += 256) {
        const int r = i >> 6, c4 = (i & 63) * 4;
        const int t = r / G, g = r - t * G;
        const uint2 qv = __ldg(reinterpret_cast<const uint2*>(
            q + ((size_t)(tb + t) * n_q_heads + hkv * G + g) * q_pitch + c4));
        const float2 a = __half22float2(*reinterpret_cast<const half2*>(&qv.x));
        const float2 b = __half22float2(*reinterpret_cast<const half2*>(&qv.y));
        *reinterpret_cast<float4*>(&qs[r * 8 * QP + (c4 >> 5) * QP + (c4 & 31)]) =
            make_float4(a.x * sl2, a.y * sl2, b.x * sl2, b.y * sl2);
    }
    const int part = tid & 7, rp = tid >> 3;                   // scores
    const int dg = tid % NG, slot = tid / NG;                  // P V
    // a cache row is 2^ksh bytes and a scale row 2^ssh halves (n_kv_heads a
    // power of two): row offsets are 32-bit shifts, not 64-bit multiplies
    const int ksh = 31 - __clz(n_kv_heads * 256), ssh = 31 - __clz(n_kv_heads * 8);
    const signed char* kb = kcache + hkv * 256 + part * 32;
    const signed char* vb = vcache + hkv * 256 + dg * PVD;
    const half* ksb = reinterpret_cast<const half*>(kcache + (long long)max_pos * kvs) + hkv * 8 + part;
    const half* vsb = reinterpret_cast<const half*>(vcache + (long long)max_pos * kvs) + hkv * 8 + part;
    const int last = pos0 + tb + T - 1;                // loads clamp here (masked rows)
    float m[R], acc[R][PVD], dp[R], cr[R];
#pragma unroll
    for (int r = 0; r < R; r++) {
        m[r] = -INFINITY; dp[r] = 0.f;
#pragma unroll
        for (int j = 0; j < PVD; j++) acc[r][j] = 0.f;
    }
    __syncthreads();
    uint4 kr[2][2]; half ksr[2], vsr[2];
    auto loadk = [&](int tile) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const unsigned row = min(tile * DD_BK + rp + 32 * h, last);
            const uint4* kp = reinterpret_cast<const uint4*>(kb + (size_t)(row << ksh));
            kr[h][0] = __ldg(kp);
            kr[h][1] = __ldg(kp + 1);
            ksr[h] = __ldg(ksb + (size_t)(row << ssh));
            vsr[h] = __ldg(vsb + (size_t)(row << ssh));
        }
    };
    loadk(ulo);
    for (int tile = ulo; tile < uhi; tile++) {
        // V of this tile, in flight during the scores
        unsigned int vr[RPS][PVD / 4];
#pragma unroll
        for (int i = 0; i < RPS; i++) {
            const signed char* p = vb + (size_t)((unsigned)min(tile * DD_BK + slot + NS * i, last) << ksh);
            vr[i][0] = __ldg(reinterpret_cast<const unsigned int*>(p));
        }
        int act = 0;
#pragma unroll
        for (int t = 0; t < T; t++) act |= (tile >= tlo[t] && tile < thi[t]) << t;
        float* ps = psb[tile & 1];
        {
            float sa[R], sb[R];
#pragma unroll
            for (int r = 0; r < R; r++) { sa[r] = 0.f; sb[r] = 0.f; }
#pragma unroll
            for (int j4 = 0; j4 < 8; j4++) {
                float ka[4], kc[4];
                const uint4& wa = kr[0][j4 >> 2];
                const uint4& wb = kr[1][j4 >> 2];
                const int c = j4 & 3;
                dd_b4(c == 0 ? wa.x : c == 1 ? wa.y : c == 2 ? wa.z : wa.w, ka);
                dd_b4(c == 0 ? wb.x : c == 1 ? wb.y : c == 2 ? wb.z : wb.w, kc);
#pragma unroll
                for (int r = 0; r < R; r++) {
                    const float4 a = *reinterpret_cast<const float4*>(&qs[r * 8 * QP + part * QP + 4 * j4]);
                    sa[r] = fmaf(a.x, ka[0], sa[r]); sa[r] = fmaf(a.y, ka[1], sa[r]);
                    sa[r] = fmaf(a.z, ka[2], sa[r]); sa[r] = fmaf(a.w, ka[3], sa[r]);
                    sb[r] = fmaf(a.x, kc[0], sb[r]); sb[r] = fmaf(a.y, kc[1], sb[r]);
                    sb[r] = fmaf(a.z, kc[2], sb[r]); sb[r] = fmaf(a.w, kc[3], sb[r]);
                }
            }
            const float ksa = __half2float(ksr[0]), ksb_ = __half2float(ksr[1]);
            const int posa = tile * DD_BK + rp, posb = posa + 32;
#pragma unroll
            for (int r = 0; r < R; r++) {
                float va = sa[r] * ksa, vb_ = sb[r] * ksb_;
                va += __shfl_xor_sync(0xffffffffu, va, 1);
                vb_ += __shfl_xor_sync(0xffffffffu, vb_, 1);
                va += __shfl_xor_sync(0xffffffffu, va, 2);
                vb_ += __shfl_xor_sync(0xffffffffu, vb_, 2);
                va += __shfl_xor_sync(0xffffffffu, va, 4);
                vb_ += __shfl_xor_sync(0xffffffffu, vb_, 4);
                const int t = r / G;
                const bool on = (act >> t) & 1;
                const float sca = (on && posa <= pos0 + tb + t) ? va : -INFINITY;
                const float scb = (on && posb <= pos0 + tb + t) ? vb_ : -INFINITY;
                if ((r & 7) == part) {
                    ps[r * DD_BK + (rp % NS) * RPS + rp / NS] = sca;
                    ps[r * DD_BK + ((rp + 32) % NS) * RPS + (rp + 32) / NS] = scb;
                }
                float mx = fmaxf(sca, scb);
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 8));
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 16));
                if (lane == 0) tmax[r * 8 + w] = mx;
            }
            vss[rp * 8 + part] = __half2float(vsr[0]);
            vss[(rp + 32) * 8 + part] = __half2float(vsr[1]);
        }
        if (tile + 1 < uhi) loadk(tile + 1);
        __syncthreads();
#pragma unroll
        for (int r = 0; r < R; r++) {
            const float4 t0 = *reinterpret_cast<const float4*>(&tmax[r * 8]);
            const float4 t1 = *reinterpret_cast<const float4*>(&tmax[r * 8 + 4]);
            const float mt = fmaxf(fmaxf(fmaxf(t0.x, t0.y), fmaxf(t0.z, t0.w)),
                                   fmaxf(fmaxf(t1.x, t1.y), fmaxf(t1.z, t1.w)));
            const float mn = fmaxf(m[r], mt);
            const float c = mn == -INFINITY ? 1.f : exp2f(m[r] - mn);
            m[r] = mn;
            // an inactive row has c = 1 and p = 0: exact no-ops, so no branch
#pragma unroll
            for (int j = 0; j < PVD; j++) acc[r][j] *= c;
            cr[r] = c;
            float sum = 0.f;
            if ((r & 7) == part) {
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const int row = rp + 32 * h;
                    float& x = ps[r * DD_BK + (row % NS) * RPS + row / NS];
                    x = x == -INFINITY ? 0.f : exp2f(x - mn);
                    sum += x;
                }
            }
            sum += __shfl_xor_sync(0xffffffffu, sum, 8);
            sum += __shfl_xor_sync(0xffffffffu, sum, 16);
            if (lane == (r & 7)) psw[r * 8 + w] = sum;
        }
        __syncthreads();
#pragma unroll
        for (int r = 0; r < R; r++) {
            if (w == (r & 7)) {
                const float4 a = *reinterpret_cast<const float4*>(&psw[r * 8]);
                const float4 b = *reinterpret_cast<const float4*>(&psw[r * 8 + 4]);
                dp[r] = fmaf(dp[r], cr[r], ((a.x + a.y) + (a.z + a.w)) + ((b.x + b.y) + (b.z + b.w)));
            }
        }
        // O += P V
#pragma unroll
        for (int i4 = 0; i4 < RPS / 4; i4++) {
            float vf[4][PVD];
#pragma unroll
            for (int ii = 0; ii < 4; ii++) {
                const int i = 4 * i4 + ii;
                const float vs = vss[(slot + NS * i) * 8 + (dg * PVD >> 5)];
#pragma unroll
                for (int c = 0; c < PVD / 4; c++) {
                    float f[4];
                    dd_b4(vr[i][c], f);
#pragma unroll
                    for (int j = 0; j < 4; j++) vf[ii][4 * c + j] = f[j] * vs;
                }
            }
#pragma unroll
            for (int r = 0; r < R; r++) {
                const float4 p4 = *reinterpret_cast<const float4*>(&ps[r * DD_BK + slot * RPS + 4 * i4]);
                const float pp[4] = {p4.x, p4.y, p4.z, p4.w};
#pragma unroll
                for (int ii = 0; ii < 4; ii++) {
#pragma unroll
                    for (int j = 0; j < PVD; j++) acc[r][j] = fmaf(pp[ii], vf[ii][j], acc[r][j]);
                }
            }
        }
    }
    // the slots' partials, summed in slot order, RC query rows a round through
    // the (now free) q buffer
    constexpr int RW = (NS - 1) * 256 + 4;             // floats a query row (16 B aligned)
    constexpr int RC = (R * 8 * QP) / RW < 1 ? 1 : (R * 8 * QP) / RW;
    float* red = qs;
#pragma unroll
    for (int r0 = 0; r0 < R; r0 += RC) {
        __syncthreads();
#pragma unroll
        for (int r = r0; r < r0 + RC && r < R; r++) {
            if (slot > 0)
                *reinterpret_cast<float4*>(&red[(r - r0) * RW + (slot - 1) * 256 + dg * 4]) =
                    make_float4(acc[r][0], acc[r][1], acc[r][2], acc[r][3]);
            if (w == (r & 7) && lane == 0) red[(r - r0) * RW + (NS - 1) * 256] = dp[r];
        }
        __syncthreads();
#pragma unroll
        for (int r = r0; r < r0 + RC && r < R; r++) {
            const int t = r / G;
            if (slot == 0 && thi[t] > tlo[t]) {
                const size_t pi = ((size_t)(tb + t) * n_q_heads + hkv * G + (r - t * G)) * smax + s;
                float4 o = make_float4(acc[r][0], acc[r][1], acc[r][2], acc[r][3]);
#pragma unroll
                for (int k = 0; k < NS - 1; k++) {
                    const float4 b = *reinterpret_cast<const float4*>(&red[(r - r0) * RW + k * 256 + dg * 4]);
                    o.x += b.x; o.y += b.y; o.z += b.z; o.w += b.w;
                }
                *reinterpret_cast<float4*>(&pacc[pi * 256 + dg * 4]) = o;
                if (dg == 0) {
                    pm[pi] = m[r];
                    pd[pi] = red[(r - r0) * RW + (NS - 1) * 256];
                }
            }
        }
    }
}

#define DD_ENTRY(NAME, T, OCC)                                                            \
extern "C" __global__ void __launch_bounds__(256, OCC)                                \
NAME(const half* __restrict__ q, const signed char* __restrict__ kcache,             \
     const signed char* __restrict__ vcache, float* __restrict__ pacc,                \
     float* __restrict__ pm, float* __restrict__ pd, int n_q_heads, int n_kv_heads,   \
     int q_pitch, int pos0, int max_pos, float sl2, int tmin, int smax) {             \
    if (n_q_heads != 6 * n_kv_heads || (n_kv_heads & (n_kv_heads - 1))) return;      \
    attention_dec_q8_body<T>(q, kcache, vcache, pacc, pm, pd, n_q_heads, n_kv_heads,  \
                             q_pitch, pos0, max_pos, sl2, tmin, smax);                \
}
DD_ENTRY(attention_dec_q8, 1, DD_OCC1)

// out = sum_s 2^(m_s - M) acc_s / sum_s 2^(m_s - M) d_s over the token's own splits.
// The _gate entry also applies attn_gate (z *= sigmoid(gate), the gate at
// g + (t * n_q_heads + head) * g_pitch + g_off) to the rounded value, as
// attn_gate does after it: one launch instead of two, the same bits.
// grid (T, n_q_heads), block 256.
template <bool GATE>
__device__ __forceinline__ void attention_dec_merge_body(
        const float* __restrict__ pacc, const float* __restrict__ pm,
        const float* __restrict__ pd, half* __restrict__ z, int n_q_heads, int z_pitch,
        int pos0, int tmin, int smax, const half* __restrict__ g, int g_pitch, int g_off) {
    const int t = blockIdx.x, head = blockIdx.y, d = threadIdx.x;
    const int len = pos0 + t + 1;
    const int nkt = (len + DD_BK - 1) / DD_BK;
    int st = (nkt + tmin - 1) / tmin;
    st = st > smax ? smax : st;
    const size_t p0 = ((size_t)t * n_q_heads + head) * smax;
    float M = -INFINITY;
    for (int s = 0; s < st; s++) M = fmaxf(M, pm[p0 + s]);
    float num = 0.f, den = 0.f;
    for (int s = 0; s < st; s++) {
        const float e = exp2f(pm[p0 + s] - M);
        num = fmaf(e, pacc[(p0 + s) * 256 + d], num);
        den = fmaf(e, pd[p0 + s], den);
    }
    half o = f16_sat(num / den);
    if (GATE) {
        const float gv = __half2float(__ldg(&g[((size_t)t * n_q_heads + head) * g_pitch + g_off + d]));
        o = f16_sat(__half2float(o) * sigmoid_f(gv));
    }
    z[(size_t)t * n_q_heads * z_pitch + (size_t)head * z_pitch + d] = o;
}
extern "C" __global__ void attention_dec_merge(const float* __restrict__ pacc,
                                               const float* __restrict__ pm,
                                               const float* __restrict__ pd,
                                               half* __restrict__ z, int n_q_heads,
                                               int z_pitch, int pos0, int tmin, int smax) {
    attention_dec_merge_body<false>(pacc, pm, pd, z, n_q_heads, z_pitch, pos0, tmin, smax,
                                    nullptr, 0, 0);
}
extern "C" __global__ void attention_dec_merge_gate(const float* __restrict__ pacc,
                                                    const float* __restrict__ pm,
                                                    const float* __restrict__ pd,
                                                    half* __restrict__ z, int n_q_heads,
                                                    int z_pitch, int pos0, int tmin, int smax,
                                                    const half* __restrict__ g, int g_pitch,
                                                    int g_off) {
    attention_dec_merge_body<true>(pacc, pm, pd, z, n_q_heads, z_pitch, pos0, tmin, smax,
                                   g, g_pitch, g_off);
}

// The merge half of the split decode: M = max_s m_s, out = sum_s e_s z_s /
// sum_s e_s d_s with e_s = exp(m_s - M) — the same rescaling the online
// recurrence applies when the running max moves. With one non-empty partial it
// is exactly the un-split kernel's arithmetic (one weight of exp(0) = 1).
// Empty partials contribute exp(-1e30 - M) = 0 to both sums, and adding zero
// does not move a float, so a token that wanted 2 parts of a 16-part launch
// gets the same answer as one that wanted 2.
//
// z is written at z_pitch per head: 256 for a packed [T,24,256] z, or 512 to
// write it in place over the q half of the interleaved q|gate buffer.
// grid (n_q_heads, T) with gridDim.z = split, block 256.
extern "C" __global__ void attention_merge(const float* __restrict__ pacc,
                                          const float* __restrict__ pm,
                                          const float* __restrict__ pd,
                                          half* __restrict__ z,
                                          int n_q_heads, int head_dim, int z_pitch) {
    const int head = blockIdx.x;
    const int t = blockIdx.y;
    const int split = gridDim.z;
    // The grid keeps attention_split's shape (z = the split count, which is how
    // this kernel learns it); one block per (head, token) does the merge and
    // the other split-1 used to repeat it and store the same row.
    if (blockIdx.z != 0) return;
    const size_t p0 = ((size_t)t * n_q_heads + head) * split;
    float M = -1e30f;
    for (int s = 0; s < split; s++) M = fmaxf(M, pm[p0 + s]);
    half* zrow = z + (size_t)t * n_q_heads * z_pitch + (size_t)head * z_pitch;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float num = 0.f;
        float den = 0.f;
        for (int s = 0; s < split; s++) {
            const float e = __expf(pm[p0 + s] - M);
            num = fmaf(e, pacc[(p0 + s) * head_dim + d], num);
            den = fmaf(e, pd[p0 + s], den);
        }
        zrow[d] = f16_sat(num * __fdiv_rn(1.f, den));
    }
}

// Causal GQA attention, prefill shape: tile queries, stage the KV chunk in
// shared memory, online softmax in registers.
//
// MODEL_OPTIMIZE.md: "attention with T>1 must tile queries and stage cache in
// threadgroup memory with an online-softmax merge (8.7-15.3x, and the O(T^2)
// re-reads that flattened long-context prefill are gone)". This kernel is that
// shape:
//   * a block owns APF_BQ = 8 consecutive queries (one warp each) of ONE q head
//   * the KV cache is streamed in chunks of APF_BK = 32 rows; each chunk is
//     staged ONCE for the whole 8-query tile (k rows, then v rows through the
//     same 16 KB tile buffer), instead of being re-read per query
//   * each warp runs its own online softmax over the chunk (no cross-warp
//     merge: a warp's query is entirely its own), and a lane holds its 8 dims
//     of the accumulator throughout, so nothing spills
//   * the causal mask is per query: rows are visited in [0, pos0 + tq], where
//     tq is the warp's own token in the chunk, so a chunk's queries see exactly
//     the contexts single-token calls would have given them.
// A tile of 8 tokens means gridDim.y carries the QUERY TILE, not one token:
// the contract's one-token-per-column holds at APF_BQ = 1, and the tiling is
// the entire point of this kernel. (The decode path, attention_split, keeps the
// y axis = token exactly.)
//
// grid (n_q_heads, ceil(T/APF_BQ)), block 256.
extern "C" __global__ void attention_prefill(const half* __restrict__ q,
                                            const half* __restrict__ kcache,
                                            const half* __restrict__ vcache,
                                            half* __restrict__ z,
                                            int layer, int n_q_heads, int n_kv_heads,
                                            int head_dim, int q_pitch, int z_pitch,
                                            int pos0, int max_pos, float scale, int T) {
    const int head = blockIdx.x;
    const int w = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (n_q_heads % n_kv_heads != 0) return;
    if ((head_dim & 63) != 0) return;
    if ((head_dim >> 6) > ATT_NL2_MAX) return;
    if (head_dim > ATT_HD_MAX) return;

    const int hkv = head / (n_q_heads / n_kv_heads);
    const int tq = blockIdx.y * APF_BQ + w;          // this warp's token in the chunk
    const bool live = tq < T;
    const int nl2 = head_dim >> 6;
    const long long kv_stride = (long long)n_kv_heads * head_dim;
    const half* kbase = kcache + ((long long)layer * max_pos) * kv_stride + (long long)hkv * head_dim;
    const half* vbase = vcache + ((long long)layer * max_pos) * kv_stride + (long long)hkv * head_dim;

    __shared__ __align__(16) half tile[APF_TILE];

    float2 qf[ATT_NL2_MAX];
    float2 acc[ATT_NL2_MAX];
#pragma unroll
    for (int u = 0; u < ATT_NL2_MAX; u++) {
        qf[u] = make_float2(0.f, 0.f);
        acc[u] = make_float2(0.f, 0.f);
    }
    if (live) {
        const half2* q2 = reinterpret_cast<const half2*>(
            q + (size_t)tq * n_q_heads * q_pitch + (size_t)head * q_pitch);
#pragma unroll
        for (int u = 0; u < ATT_NL2_MAX; u++)
            if (u < nl2) qf[u] = __half22float2(__ldg(&q2[lane + 32 * u]));
    }
    float m = -1e30f;
    float d = 0.f;

    const int mypos = pos0 + tq;                     // last kv row this query may use
    const int last = min(blockIdx.y * APF_BQ + APF_BQ, T);   // tile's exclusive token end
    const int kv_end = pos0 + last;                  // cache rows written by this call
    const int u_per_row = head_dim >> 3;             // uint4 per row (head_dim%8==0)

    for (int kv0 = 0; kv0 < kv_end; kv0 += APF_BK) {
        // ---- stage APF_BK k rows (all threads; a row past the call's own
        //      tokens is zeroed and never read by a live query) ----
        for (int i = threadIdx.x; i < APF_BK * (head_dim >> 3); i += blockDim.x) {
            const int j = i / u_per_row;
            const int u = i - j * u_per_row;
            const int kv = kv0 + j;
            uint4 t4 = make_uint4(0, 0, 0, 0);
            if (kv < kv_end)
                t4 = __ldg(reinterpret_cast<const uint4*>(
                    kbase + (long long)kv * kv_stride) + u);
            *reinterpret_cast<uint4*>(tile + j * head_dim + (u << 3)) = t4;
        }
        __syncthreads();
        if (live) {
            for (int j = 0; j < APF_BK; j++) {
                const int kv = kv0 + j;
                if (kv > mypos) break;
                const half2* k2 = reinterpret_cast<const half2*>(tile + j * head_dim);
                float sc = 0.f;
#pragma unroll
                for (int u = 0; u < ATT_NL2_MAX; u++) {
                    if (u < nl2) {
                        const float2 kk = __half22float2(k2[lane + 32 * u]);
                        sc = fmaf(qf[u].x, kk.x, sc);
                        sc = fmaf(qf[u].y, kk.y, sc);
                    }
                }
                const float sco = warp_reduce_all(sc) * scale;
                float p;
                if (sco > m) {
                    const float c = __expf(m - sco);
                    m = sco;
                    d = d * c + 1.f;
#pragma unroll
                    for (int u = 0; u < ATT_NL2_MAX; u++) {
                        acc[u].x *= c;
                        acc[u].y *= c;
                    }
                    p = 1.f;
                } else {
                    p = __expf(sco - m);
                    d += p;
                }
                // the v row of the same position: read from the global cache
                // (staging it in the same tile buffer would need a second
                // barrier pair per chunk for nothing — the 8 warps of a tile
                // read the same row, so L1 serves 7 of the 8)
                const half2* v2 = reinterpret_cast<const half2*>(
                    vbase + (long long)kv * kv_stride);
#pragma unroll
                for (int u = 0; u < ATT_NL2_MAX; u++) {
                    if (u < nl2) {
                        const float2 vv = __half22float2(__ldg(&v2[lane + 32 * u]));
                        acc[u].x = fmaf(p, vv.x, acc[u].x);
                        acc[u].y = fmaf(p, vv.y, acc[u].y);
                    }
                }
            }
        }
        __syncthreads();
    }

    if (!live) return;
    const float inv = __fdiv_rn(1.f, d);
    half* zrow = z + (size_t)tq * n_q_heads * z_pitch + (size_t)head * z_pitch;
#pragma unroll
    for (int u = 0; u < ATT_NL2_MAX; u++) {
        if (u < nl2) {
            const float2 o = make_float2(acc[u].x * inv, acc[u].y * inv);
            *reinterpret_cast<half2*>(zrow + 2 * (lane + 32 * u)) = __floats2half2_rn(o.x, o.y);
        }
    }
}


// Attention output gate: x = x * sigmoid(gate), applied to the concatenated
// heads AFTER attention and BEFORE o_proj (trap 7).
//
// SIGMOID, not silu: `output_gate_type: "swish"` in the config belongs to the
// GDN's gated norm (rmsnorm_gated, where the activation is silu). Every
// reference hardcodes torch.sigmoid here, and using silu instead is a
// silently-wrong gate — the test mutates exactly this.
//
// z and g are separate pointers so the gate can be either a separate buffer or
// the q|gate buffer itself: with g == z, g_pitch = z_pitch = 512 and
// g_off = 256, this reads the gate half of each 512-wide head block and writes
// the value half, in place.
// grid (nx, T), block 256.
extern "C" __global__ void attn_gate(half* __restrict__ z,
                                     const half* __restrict__ g,
                                     int n_heads, int head_dim,
                                     int z_pitch, int g_pitch, int g_off) {
    const int t = blockIdx.y;
    const int n = n_heads * head_dim;
    half* zt = z + (size_t)t * n_heads * z_pitch;
    const half* gt = g + (size_t)t * n_heads * g_pitch + g_off;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const int h = i / head_dim;
        const int dd = i - h * head_dim;
        const float gv = __half2float(__ldg(&gt[h * g_pitch + dd]));
        half* p = zt + h * z_pitch + dd;
        *p = f16_sat(__half2float(*p) * sigmoid_f(gv));
    }
}

// ============================================================================
// GDN (linear attention)
// ============================================================================

// Depthwise causal conv, kernel `width`, no bias, silu AFTER the conv, over one
// segment of the qkv stream (the runtime launches it three times per layer: the
// q segment at offset 0, k at 2048, v at 4096, each with its own dim, its own
// weight slice and its own state — trap 2, the flat q|k|v layout).
//
// Semantics, exactly: with E = [state_frames(width-1) | x(T)] the frames of the
// extended sequence, E[i] = state[i] for i < width-1 (ZERO when pos0 == 0: a
// fresh sequence has no history, which is what makes prefix reuse correct) and
// E[width-1 + t] = x[t],
//
//     y[t][d] = silu( sum_j w[d][j] * E[t + j] )        j = 0..width-1
//
// i.e. tap 0 is the OLDEST (x[t-width+1]) and tap width-1 the newest, matching
// the checkpoint's [dim, 1, 4] with tap 0 oldest. The new state is the last
// width-1 frames of E — `st_out[d*sl + k] = E[T + k]` — which for T >= width-1
// is x's own tail and for a short call is "padded with the tail of the old
// state" (T=1 at pos0>0 gives (state[1], state[2], x[0])). Chunk invariance is
// a consequence: four tokens then four more is the eight-token call, because
// each call's E is the previous call's tail.
//
// The state is dim-first ([dim, width-1], trap 5), fp16 (trap 4), and holds
// PRE-conv frames (silu is applied after the conv, and to the output, never to
// what is stored). st_in and st_out must be DISTINCT buffers: the block that
// writes the new state is the last token's, and the blocks that read the old
// one are the first tokens', so they are different blocks with no ordering
// between them. (The runtime double-buffers the 2.9 MB per sequence.)
//
// grid (ceil(dim/256), T), block 256, one channel per thread.
extern "C" __global__ void conv1d_causal(const half* __restrict__ x,
                                         const half* __restrict__ w,
                                         const half* __restrict__ st_in,
                                         half* __restrict__ y,
                                         half* __restrict__ st_out,
                                         int dim, int T, int pos0,
                                         int row_stride, int width) {
    if (width < 1) return;
    const int t = blockIdx.y;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) return;
    const int sl = width - 1;                    // frames of history kept
    half* yr = y + (size_t)t * row_stride + d;   // `x`/`y` are segment-offset
    const half* st = st_in + (size_t)d * sl;

    // E[i] for i in [0, sl + T): the history frames first (ZERO at pos0 == 0),
    // then this call's frames, which live at x[(i - sl)][d] — a different token,
    // so the channel index has to come with the row stride.
    float acc = 0.f;
    for (int j = 0; j < width; j++) {
        const int i = t + j;
        float e = 0.f;
        if (i < sl) {
            if (pos0 > 0) e = __half2float(__ldg(&st[i]));
        } else {
            e = __half2float(__ldg(&x[(size_t)(i - sl) * row_stride + d]));
        }
        acc = fmaf(__half2float(__ldg(&w[(size_t)d * width + j])), e, acc);
    }
    *yr = f16_sat(silu_f(acc));

    // the last token's block rewrites the state: the last `sl` frames of E
    if (t == T - 1) {
        half* so = st_out + (size_t)d * sl;
        for (int k = 0; k < sl; k++) {
            const int i = T + k;
            float e = 0.f;
            if (i < sl) {
                if (pos0 > 0) e = __half2float(__ldg(&st[i]));
            } else {
                e = __half2float(__ldg(&x[(size_t)(i - sl) * row_stride + d]));
            }
            so[k] = f16_sat(e);
        }
    }
}

// conv1d_causal for a verify's T <= 3 rows in one launch: row t is the T = 1
// call at pos0 + t from the state row t - 1 left, and that state is written to
// st0 / st1 / st2 (row 0's may be st_in itself: a thread reads its channel's
// state before writing it) -- the per-row calls' values and stores exactly.
// Width 4 (3 frames of history). grid ceil(dim / 256), block 256.
extern "C" __global__ void conv1d_causal_rows(const half* __restrict__ x,
                                              const half* __restrict__ w,
                                              const half* st_in, half* y,
                                              half* st0, half* st1, half* st2,
                                              int dim, int T, int pos0, int row_stride) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim || T < 1 || T > 3) return;
    float e[3];
#pragma unroll
    for (int i = 0; i < 3; i++) e[i] = pos0 > 0 ? __half2float(st_in[(size_t)d * 3 + i]) : 0.f;
    float wv[4];
#pragma unroll
    for (int j = 0; j < 4; j++) wv[j] = __half2float(__ldg(&w[(size_t)d * 4 + j]));
    half* dst[3] = {st0, st1, st2};
#pragma unroll
    for (int t = 0; t < 3; t++) {
        if (t >= T) break;
        const float xt = __half2float(__ldg(&x[(size_t)t * row_stride + d]));
        float acc = 0.f;
        acc = fmaf(wv[0], e[0], acc);
        acc = fmaf(wv[1], e[1], acc);
        acc = fmaf(wv[2], e[2], acc);
        acc = fmaf(wv[3], xt, acc);
        y[(size_t)t * row_stride + d] = f16_sat(silu_f(acc));
        e[0] = e[1]; e[1] = e[2]; e[2] = __half2float(f16_sat(xt));
        half* so = dst[t] + (size_t)d * 3;
        so[0] = f16_sat(e[0]); so[1] = f16_sat(e[1]); so[2] = f16_sat(e[2]);
    }
}

// The two per-(token, v-head) scalars of the recurrence, in fp32:
//
//     beta  = sigmoid(b)
//     decay = exp(-exp(A_log[h]) * softplus(a[h] + dt_bias[h]))
//
// `h` is the V-HEAD index (0..47). softplus is PyTorch's, threshold 20 exactly:
// log1p(exp(x)) for x <= 20, x above — past 20 the two agree to an ulp and
// exp() would overflow for x > 88, which is what the branch is for. The sum
// a + dt_bias is in fp32 (trap 4: A_log/dt_bias are stored bf16 but are fp32
// math), the divide in sigmoid is __fdiv_rn (see the header: --use_fast_math's
// `/` is not precise, and beta and decay multiply decision-making quantities
// for the whole recurrence).
//
// Both outputs are [T, n_heads] fp32, contiguous; `a` and `b` are fp32 with
// their own strides so a shared arena with padding still works.
// grid (ceil(n_heads/256), T), block 256.
extern "C" __global__ void gdn_scalars(const float* __restrict__ a,
                                       const float* __restrict__ b,
                                       const float* __restrict__ A_log,
                                       const float* __restrict__ dt_bias,
                                       float* __restrict__ beta,
                                       float* __restrict__ decay,
                                       int T, int n_heads, int a_stride, int b_stride) {
    const int t = blockIdx.y;
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T || h >= n_heads) return;
    const float bb = __ldg(&b[(size_t)t * b_stride + h]);
    beta[(size_t)t * n_heads + h] = sigmoid_f(bb);
    const float x = __ldg(&a[(size_t)t * a_stride + h]) + __ldg(&dt_bias[h]);
    const float sp = x <= 20.f ? log1pf(__expf(x)) : x;
    const float g = -__expf(__ldg(&A_log[h])) * sp;
    decay[(size_t)t * n_heads + h] = __expf(g);
}

// The gated delta rule, one block per v-head, state read and rewritten IN PLACE
// (chunk invariance: two calls of 4 are one call of 8 — the state is the only
// thing that carries, and every per-token step below is the same arithmetic
// whichever call it happens in).
//
// Per token t, per v-head hv, with S a [V=128, K=128] fp32 matrix:
//
//     S      *= decay_t
//     m       = S · k_t                    (dot over K)
//     delta   = beta_t * (v_t - m)
//     S      += delta ⊗ k_t                (outer over V x K)
//     o_t     = S_new · q_t                (dot over K — the values just stored)
//
// The order is the reference's, line for line (fused_recurrent.py lines
// 128-137): decay, then the removal using the DECAYED state, then beta, then
// the write, then the read-back. o_t must use S_new, not S_old: the last two
// steps are a read-after-write inside one token.
//
// k here is the L2-normalised conv output and q is the L2-normalised conv
// output ALREADY scaled by 128**-0.5 (l2norm_scaled's scale argument) — the
// reference multiplies q by `scale = K**-0.5` after its l2norm.
//
// GQA: v-head hv reads k/q head hv/group (group = 48/16 = 3).
//
// Conditioning: with k L2-normalised and beta in (0,1) this recurrence is a
// contraction — |S| cannot grow without bound. If it diverges over a long
// sequence, suspect a missing L2 norm or a beta outside (0,1) BEFORE suspecting
// this kernel; both are checked by the oracles.
//
// Shape: one block per v-head (48 blocks), 256 threads, two threads per state
// ROW: thread 2*vi+half owns columns [half*K/2, (half+1)*K/2) of row vi. The
// partner of a row is the adjacent lane, so the two per-row reductions are
// __shfl_xor(...,1) — a cross-warp pair here would be silently wrong, since a
// shuffle cannot reach out of its own warp. The row's K/2 columns live in
// registers for the whole T loop (loaded once, stored once), and k/q are staged
// per token through shared memory so the inner loops are one smem read and one
// FMA per element.
//
// Prefill parallelism is 48 blocks: the recurrence is sequential in T and this
// kernel does not chunk it (a chunked scan is the prefill optimization; the
// arithmetic is the same recurrence in blocks). Decode is the case this shape
// is for.
//
// grid (n_vheads), block 256.
extern "C" __global__ void gdn_scan(const half* __restrict__ q,
                                    const half* __restrict__ k,
                                    const half* __restrict__ v,
                                    const float* __restrict__ beta,
                                    const float* __restrict__ decay,
                                    float* __restrict__ S,
                                    half* __restrict__ y,
                                    int T, int n_kheads, int n_vheads,
                                    int K, int V, int qk_stride, int v_stride,
                                    int y_stride, float out_scale, int group,
                                    float* __restrict__ Ck, long long ck_step) {
    if (group <= 0) return;
    if (K != 2 * GDN_KH_MAX) return;             // this shape is K = 128 (see below)
    if (K > GDN_SMEM_K) return;                  // smem bound
    if (2 * V != (int)blockDim.x) return;        // two threads per state row
    const int hv = blockIdx.x;
    const int hk = hv / group;
    if (hk >= n_kheads) return;
    const int vi = threadIdx.x >> 1;             // state row
    const int kpart = threadIdx.x & 1;           // which half of K
    const int KH = GDN_KH_MAX;                   // == K/2, by the guard above
    const int c0 = kpart * KH;
    float* Srow = S + ((size_t)hv * V + vi) * K + c0;

    // The K/2 columns of this thread's row, in REGISTERS for the whole T loop:
    // loaded once, stored once. All four loops below are fully unrolled over a
    // compile-time bound for exactly this reason — a runtime bound (or a guard
    // on a compile-time loop) makes `reg` a dynamic-indexed local array, which
    // ptxas then places on the stack: 256 bytes of stack frame, 0 spills, and
    // every one of the 192 accesses per token goes through L1 with a 256-byte
    // stride per thread. That is the one measurement-driven shape decision in
    // this kernel.
    float reg[GDN_KH_MAX];
    // 16-byte loads through the read-only path: a lane's 64 columns are 256
    // contiguous bytes, so the warp's first load brings each line once and the
    // next seven hit it. The scalar form (one 4-byte load a column, 32 lines a
    // warp-instruction, 8x the sectors) ran the T = 1 scan at ~80 GB/s.
    // Nothing reads S after this kernel writes it, so the non-coherent path is
    // safe; the values (and every operation below) are unchanged.
#pragma unroll
    for (int c = 0; c < GDN_KH_MAX; c += 4) {
        const float4 v4 = __ldg(reinterpret_cast<const float4*>(Srow + c));
        reg[c] = v4.x; reg[c + 1] = v4.y; reg[c + 2] = v4.z; reg[c + 3] = v4.w;
    }

    __shared__ half sh_k[GDN_SMEM_K];
    __shared__ half sh_q[GDN_SMEM_K];
    for (int t = 0; t < T; t++) {
        const float bt = __ldg(&beta[(size_t)t * n_vheads + hv]);
        const float dt = __ldg(&decay[(size_t)t * n_vheads + hv]);
        const half* krow = k + (size_t)t * qk_stride + (size_t)hk * K;
        const half* qrow = q + (size_t)t * qk_stride + (size_t)hk * K;
        for (int i = threadIdx.x; i < K; i += blockDim.x) {
            sh_k[i] = __ldg(&krow[i]);
            sh_q[i] = __ldg(&qrow[i]);
        }
        __syncthreads();

        // S *= decay, and the partial dot m = S . k
        const half* kk = sh_k + c0;
        const half* qq = sh_q + c0;
        float m = 0.f;
#pragma unroll
        for (int c = 0; c < GDN_KH_MAX; c++) {
            const float sv = reg[c] * dt;
            reg[c] = sv;
            m = fmaf(sv, __half2float(kk[c]), m);
        }
        m += __shfl_xor_sync(0xffffffffu, m, 1);   // the row's other half

        // delta = beta * (v - m); both threads of the row compute it
        const float vt = __half2float(__ldg(&v[(size_t)t * v_stride + (size_t)hv * V + vi]));
        const float delta = bt * (vt - m);

        // S += delta (x) k
#pragma unroll
        for (int c = 0; c < GDN_KH_MAX; c++)
            reg[c] = fmaf(delta, __half2float(kk[c]), reg[c]);

        // o = S_new . q
        float o = 0.f;
#pragma unroll
        for (int c = 0; c < GDN_KH_MAX; c++) o = fmaf(reg[c], __half2float(qq[c]), o);
        o += __shfl_xor_sync(0xffffffffu, o, 1);
        if (kpart == 0)
            y[(size_t)t * y_stride + (size_t)hv * V + vi] = f16_sat(o * out_scale);

        // mid-state checkpoints: the recurrence after every row but the last,
        // row t at Ck + t * ck_step, so a verify can fall back to the state
        // after any accepted prefix instead of re-running a forward
        // (_mtp_round's rejects). Null on every other path, and the copy is the
        // entry's own closing store, taken mid-loop.
        if (Ck != nullptr && t < T - 1) {
            float* ckrow = Ck + (size_t)t * ck_step + ((size_t)hv * V + vi) * K + c0;
#pragma unroll
            for (int c = 0; c < GDN_KH_MAX; c += 4)
                *reinterpret_cast<float4*>(ckrow + c) =
                    make_float4(reg[c], reg[c + 1], reg[c + 2], reg[c + 3]);
        }

        __syncthreads();                           // before the next staging
    }

#pragma unroll
    for (int c = 0; c < GDN_KH_MAX; c += 4)
        *reinterpret_cast<float4*>(Srow + c) =
            make_float4(reg[c], reg[c + 1], reg[c + 2], reg[c + 3]);
}

// gdn_scan with its two producers folded in, for the decode shapes (T <= 2):
// q and k arrive RAW (the conv output) and are L2-normalised here exactly as
// l2norm_scaled does it (128 threads a row, block_reduce_sum's order, the same
// division and rounding), and beta / decay are gdn_scalars' expressions. So
// the values are the three-launch path's bit for bit, in one launch. Prefill
// keeps the three launches (the per-token reductions would add two barriers a
// token to a sequential loop).
extern "C" __global__ void gdn_scan_f(const half* __restrict__ q,
                                    const half* __restrict__ k,
                                    const half* __restrict__ v,
                                    const float* __restrict__ beta,
                                    const float* __restrict__ decay,
                                    float* __restrict__ S,
                                    half* __restrict__ y,
                                    int T, int n_kheads, int n_vheads,
                                    int K, int V, int qk_stride, int v_stride,
                                    int y_stride, float out_scale, int group,
                                    float* __restrict__ Ck, long long ck_step,
                                    const float* __restrict__ ga,
                                    const float* __restrict__ gb,
                                    const float* __restrict__ A_log,
                                    const float* __restrict__ dt_bias,
                                    int ab_stride, float l2eps) {
    __shared__ float red[8];
    if (group <= 0) return;
    if (K != 2 * GDN_KH_MAX) return;             // this shape is K = 128 (see below)
    if (K > GDN_SMEM_K) return;                  // smem bound
    if (2 * V != (int)blockDim.x) return;        // two threads per state row
    const int hv = blockIdx.x;
    const int hk = hv / group;
    if (hk >= n_kheads) return;
    const int vi = threadIdx.x >> 1;             // state row
    const int kpart = threadIdx.x & 1;           // which half of K
    const int KH = GDN_KH_MAX;                   // == K/2, by the guard above
    const int c0 = kpart * KH;
    float* Srow = S + ((size_t)hv * V + vi) * K + c0;

    // The K/2 columns of this thread's row, in REGISTERS for the whole T loop:
    // loaded once, stored once. All four loops below are fully unrolled over a
    // compile-time bound for exactly this reason — a runtime bound (or a guard
    // on a compile-time loop) makes `reg` a dynamic-indexed local array, which
    // ptxas then places on the stack: 256 bytes of stack frame, 0 spills, and
    // every one of the 192 accesses per token goes through L1 with a 256-byte
    // stride per thread. That is the one measurement-driven shape decision in
    // this kernel.
    float reg[GDN_KH_MAX];
    // 16-byte loads through the read-only path: a lane's 64 columns are 256
    // contiguous bytes, so the warp's first load brings each line once and the
    // next seven hit it. The scalar form (one 4-byte load a column, 32 lines a
    // warp-instruction, 8x the sectors) ran the T = 1 scan at ~80 GB/s.
    // Nothing reads S after this kernel writes it, so the non-coherent path is
    // safe; the values (and every operation below) are unchanged.
#pragma unroll
    for (int c = 0; c < GDN_KH_MAX; c += 4) {
        const float4 v4 = __ldg(reinterpret_cast<const float4*>(Srow + c));
        reg[c] = v4.x; reg[c + 1] = v4.y; reg[c + 2] = v4.z; reg[c + 3] = v4.w;
    }

    __shared__ half sh_k[GDN_SMEM_K];
    __shared__ half sh_q[GDN_SMEM_K];
    for (int t = 0; t < T; t++) {
        // gdn_scalars, for this token and v-head
        const float bt = sigmoid_f(__ldg(&gb[(size_t)t * ab_stride + hv]));
        float dt;
        {
            const float xx = __ldg(&ga[(size_t)t * ab_stride + hv]) + __ldg(&dt_bias[hv]);
            const float sp = xx <= 20.f ? log1pf(__expf(xx)) : xx;
            dt = __expf(-__expf(__ldg(&A_log[hv])) * sp);
        }
        // l2norm_scaled of the k row (threads 0..127) and the q row (128..255)
        {
            const int half_ = threadIdx.x >> 7, i = threadIdx.x & 127;
            const half* row = (half_ ? q : k) + (size_t)t * qk_stride + (size_t)hk * K;
            const float v = __half2float(__ldg(&row[i]));
            float ss = warp_reduce_sum(fmaf(v, v, 0.f));
            const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
            if (lane == 0) red[wid] = ss;
            __syncthreads();
            if ((wid & 3) == 0) {
                float w4 = (lane < 4) ? red[(wid & 4) + lane] : 0.f;
                w4 = warp_reduce_sum(w4);
                if (lane == 0) red[wid] = w4;     // red[0] = k's, red[4] = q's
            }
            __syncthreads();
            const float den = sqrtf(red[half_ * 4] + l2eps);
            (half_ ? sh_q : sh_k)[i] = f16_sat(__fdiv_rn(v, den) * 1.0f);
        }
        __syncthreads();

        // S *= decay, and the partial dot m = S . k
        const half* kk = sh_k + c0;
        const half* qq = sh_q + c0;
        float m = 0.f;
#pragma unroll
        for (int c = 0; c < GDN_KH_MAX; c++) {
            const float sv = reg[c] * dt;
            reg[c] = sv;
            m = fmaf(sv, __half2float(kk[c]), m);
        }
        m += __shfl_xor_sync(0xffffffffu, m, 1);   // the row's other half

        // delta = beta * (v - m); both threads of the row compute it
        const float vt = __half2float(__ldg(&v[(size_t)t * v_stride + (size_t)hv * V + vi]));
        const float delta = bt * (vt - m);

        // S += delta (x) k
#pragma unroll
        for (int c = 0; c < GDN_KH_MAX; c++)
            reg[c] = fmaf(delta, __half2float(kk[c]), reg[c]);

        // o = S_new . q
        float o = 0.f;
#pragma unroll
        for (int c = 0; c < GDN_KH_MAX; c++) o = fmaf(reg[c], __half2float(qq[c]), o);
        o += __shfl_xor_sync(0xffffffffu, o, 1);
        if (kpart == 0)
            y[(size_t)t * y_stride + (size_t)hv * V + vi] = f16_sat(o * out_scale);

        // mid-state checkpoints: the recurrence after every row but the last,
        // row t at Ck + t * ck_step, so a verify can fall back to the state
        // after any accepted prefix instead of re-running a forward
        // (_mtp_round's rejects). Null on every other path, and the copy is the
        // entry's own closing store, taken mid-loop.
        if (Ck != nullptr && t < T - 1) {
            float* ckrow = Ck + (size_t)t * ck_step + ((size_t)hv * V + vi) * K + c0;
#pragma unroll
            for (int c = 0; c < GDN_KH_MAX; c += 4)
                *reinterpret_cast<float4*>(ckrow + c) =
                    make_float4(reg[c], reg[c + 1], reg[c + 2], reg[c + 3]);
        }

        __syncthreads();                           // before the next staging
    }

#pragma unroll
    for (int c = 0; c < GDN_KH_MAX; c += 4)
        *reinterpret_cast<float4*>(Srow + c) =
            make_float4(reg[c], reg[c + 1], reg[c + 2], reg[c + 3]);
}

// ============================================================================
// Elementwise and gather
// ============================================================================

// Row-gather + int8 dequantisation of `embed_tokens`, one token per column.
// The table is NOT materialised: only the gathered rows are read, and each is
// `w * scale[row]` with the per-row scale fp16 and w int8. 8 int8 = one 8-byte
// load, dequantised into 8 halves = one 16-byte store; the row's scale is
// converted once per row. hidden must be a multiple of 8 (5120 is).
// grid (nx, T), block 256.
extern "C" __global__ void embed_gather(const signed char* __restrict__ table,
                                        const half* __restrict__ scales,
                                        const int* __restrict__ ids,
                                        half* __restrict__ out,
                                        int hidden) {
    if ((hidden & 7) != 0) return;
    const int t = blockIdx.y;
    const int id = __ldg(&ids[t]);
    const int nx = gridDim.x;
    const int chunk = (((hidden + nx - 1) / nx) + 7) & ~7;
    const int i0 = blockIdx.x * chunk;
    const int i1 = i0 + chunk < hidden ? i0 + chunk : hidden;
    const float scf = __half2float(__ldg(&scales[id]));
    const signed char* src = table + (size_t)id * hidden;
    half* dst = out + (size_t)t * hidden;
    for (int i = i0 + threadIdx.x * 8; i < i1; i += blockDim.x * 8) {
        const long long w8 = *reinterpret_cast<const long long*>(src + i);
        uint4 packed;
        half* hp = reinterpret_cast<half*>(&packed);
#pragma unroll
        for (int u = 0; u < 8; u++)
            hp[u] = f16_sat((float)(signed char)((w8 >> (8 * u)) & 0xFF) * scf);
        *reinterpret_cast<uint4*>(dst + i) = packed;
    }
}

// silu(gate) * up, the SwiGLU MLP's activation, flat over n (both vectors are
// the projection outputs back to back). fp32 arithmetic, one rounding to fp16.
extern "C" __global__ void silu_mul(const half* __restrict__ g,
                                    const half* __restrict__ u,
                                    half* __restrict__ y, int n) {
    const int n2 = n >> 1;
    const half2* g2 = reinterpret_cast<const half2*>(g);
    const half2* u2 = reinterpret_cast<const half2*>(u);
    half2* y2 = reinterpret_cast<half2*>(y);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n2;
         i += gridDim.x * blockDim.x) {
        const float2 gv = __half22float2(__ldg(&g2[i]));
        const float2 uv = __half22float2(__ldg(&u2[i]));
        y2[i] = __floats2half2_rn(silu_f(gv.x) * uv.x, silu_f(gv.y) * uv.y);
    }
    if ((n & 1) && blockIdx.x == 0 && threadIdx.x == 0)
        y[n - 1] = f16_sat(silu_f(__half2float(g[n - 1])) * __half2float(u[n - 1]));
}

// x += y, the residual add, flat over n.
extern "C" __global__ void add_inplace(half* __restrict__ x,
                                       const half* __restrict__ y, int n) {
    const int n2 = n >> 1;
    half2* x2 = reinterpret_cast<half2*>(x);
    const half2* y2 = reinterpret_cast<const half2*>(y);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n2;
         i += gridDim.x * blockDim.x) {
        const float2 a = __half22float2(x2[i]);
        const float2 b = __half22float2(__ldg(&y2[i]));
        x2[i] = __floats2half2_rn(a.x + b.x, a.y + b.y);
    }
    if ((n & 1) && blockIdx.x == 0 && threadIdx.x == 0)
        x[n - 1] = f16_sat(__half2float(x[n - 1]) + __half2float(y[n - 1]));
}

// ============================================================================
// Greedy argmax (the correctness backbone: if greedy is wrong nothing else can
// be trusted, so the selection here is bit-exact rather than approximate)
// ============================================================================
//
// Candidates are combined by "larger value wins, else smaller index wins", which
// is the commutative, associative join of the total order a max/index walk
// imposes — the reduction's shape cannot change the winner, and ties (including
// +0.0 against -0.0, which compare equal) resolve to the SMALLEST index, as
// list.index does. No value is added, multiplied or rounded anywhere in this
// pair of kernels, so there is no drift to bound: the selection is identical
// bit for bit to a host walk.
//
// argmax_partial: grid (nblocks), block 256; keys/idxs are `nblocks` long. A
// block with no element (only possible when nblocks > n) still writes its
// candidate: value -inf with index n, which can never win against a real
// element unless the row is entirely -inf, in which case the answer is n (the
// caller's signal that no candidate exists).
extern "C" __global__ void argmax_partial(const float* __restrict__ v,
                                          float* __restrict__ keys,
                                          int* __restrict__ idxs, int n) {
    __shared__ float sk[AMAX_THREADS];
    __shared__ int si[AMAX_THREADS];
    const int t = threadIdx.x;
    const int per = (n + gridDim.x - 1) / gridDim.x;
    const int lo = (int)blockIdx.x * per;
    const int hi = lo + per < n ? lo + per : n;

    float bv = AMAX_NEG_INF;
    int bi = n;
    for (int i = lo + t; i < hi; i += AMAX_THREADS) {
        const float x = __ldg(&v[i]);
        if (x > bv || (x == bv && i < bi)) { bv = x; bi = i; }
    }
    sk[t] = bv;
    si[t] = bi;
    __syncthreads();
    for (int s = AMAX_THREADS >> 1; s > 0; s >>= 1) {
        if (t < s) {
            const float v2 = sk[t + s];
            const int i2 = si[t + s];
            if (v2 > sk[t] || (v2 == sk[t] && i2 < si[t])) { sk[t] = v2; si[t] = i2; }
        }
        __syncthreads();
    }
    if (t == 0) {
        keys[blockIdx.x] = sk[0];
        idxs[blockIdx.x] = si[0];
    }
}

// The merge: one block over the `n` partials, same join. grid (1), block 256.
extern "C" __global__ void argmax_final(const float* __restrict__ keys,
                                        const int* __restrict__ idxs,
                                        int n, int* __restrict__ out) {
    __shared__ float sk[AMAX_THREADS];
    __shared__ int si[AMAX_THREADS];
    const int t = threadIdx.x;
    float bv = AMAX_NEG_INF;
    int bi = n;
    for (int i = t; i < n; i += (int)blockDim.x) {
        const float x = keys[i];
        const int j = idxs[i];
        if (x > bv || (x == bv && j < bi)) { bv = x; bi = j; }
    }
    sk[t] = bv;
    si[t] = bi;
    __syncthreads();
    for (int s = (int)(blockDim.x >> 1); s > 0; s >>= 1) {
        if (t < s) {
            const float v2 = sk[t + s];
            const int i2 = si[t + s];
            if (v2 > sk[t] || (v2 == sk[t] && i2 < si[t])) { sk[t] = v2; si[t] = i2; }
        }
        __syncthreads();
    }
    if (t == 0) out[0] = si[0];
}

// ============================================================================
// NOT WRITTEN HERE, ON PURPOSE
// ============================================================================
// The exl3 trellis decode / fused GEMV (kernels_exl3.cu, another workstream),
// the Hadamard rotations around it, the top-k/top-p sampler (the Python side
// owns the sampling policy; only the device argmax above is not optional), and
// every projection's GEMV. gdn_scalars is the one place this file names a
// dtype the projections must produce: `a` and `b` are fp32.

// ============================================================================
// Causal prefill attention as two register-blocked GEMMs (FlashAttention-2
// shape, fp32 SIMT): a block owns FA_BQ = 64 queries of one q head and walks
// the kv rows in tiles of FA_BK = 64. Per tile: S = Q K^T (64 x 64 over the 256
// dims, staged 32 dims at a time as fp32, a 4 x 4 register tile per thread),
// the causal mask, the online softmax per row (a row's 64 scores live in the 16
// threads of one half-warp), then O += P V (P through shared memory, V staged
// 64 dims at a time, a 4 x 16 register tile per thread). The per-head kernel
// above spends a warp reduction and a softmax update per (query, key) pair and
// ran at ~6% of the FP32 rate at 4k tokens. Not bit-identical to it (another
// summation order); judged against the numpy oracle and by perplexity.
// grid (n_q_heads, ceil(T / FA_BQ)), block 256.
// ============================================================================
#define FA_BQ 64
#define FA_BK 64
#define FA_DC 32                // dims of Q/K staged per S step
#define FA_VC 64                // dims of V staged per O step
template <bool Q8>
static __device__ __forceinline__ void attention_prefill_fa_body(
        const half* __restrict__ q, const void* __restrict__ kcache,
        const void* __restrict__ vcache, half* __restrict__ z,
        int layer, int n_q_heads, int n_kv_heads, int head_dim,
        int q_pitch, int z_pitch, int pos0, int max_pos, float scale, int T) {
    // smem: phase S: Qc [FA_BQ][FA_DC+1] + Kc [FA_BK][FA_DC+1]; phase O: P [FA_BQ][FA_BK]
    // + Vc [FA_BK][FA_VC]; the two phases share one buffer.
    __shared__ __align__(16) float sm[FA_BQ * FA_BK + FA_BK * FA_VC];
    if (head_dim != 256 || n_q_heads % n_kv_heads != 0) return;
    if (Q8 && layer != 0) return;                    // an int8 cache is one layer a buffer
    const int head = blockIdx.x;
    const int hkv = head / (n_q_heads / n_kv_heads);
    const int tq0 = blockIdx.y * FA_BQ;
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const long long kv_stride = (long long)n_kv_heads * head_dim;
    const half* kbase = reinterpret_cast<const half*>(kcache)
                        + ((long long)layer * max_pos) * kv_stride + (long long)hkv * head_dim;
    const half* vbase = reinterpret_cast<const half*>(vcache)
                        + ((long long)layer * max_pos) * kv_stride + (long long)hkv * head_dim;
    // int8 (KV8): rows of int8 [max_pos][n_kv][head_dim], then the fp16 scales
    // of each 32-dim block [max_pos][n_kv][head_dim / 32]
    const signed char* k8 = reinterpret_cast<const signed char*>(kcache) + (long long)hkv * head_dim;
    const signed char* v8 = reinterpret_cast<const signed char*>(vcache) + (long long)hkv * head_dim;
    const half* ksc = reinterpret_cast<const half*>(
        reinterpret_cast<const signed char*>(kcache) + (long long)max_pos * kv_stride);
    const half* vsc = reinterpret_cast<const half*>(
        reinterpret_cast<const signed char*>(vcache) + (long long)max_pos * kv_stride);
    const int nsb = head_dim >> 5;
    const half* qbase = q + (size_t)head * q_pitch;
    float* Qc = sm;                                  // [FA_BQ][FA_DC + 1]
    float* Kc = sm + FA_BQ * (FA_DC + 1);            // [FA_BK][FA_DC + 1]
    float* P = sm;                                   // [FA_BQ][FA_BK]
    float* Vc = sm + FA_BQ * FA_BK;                  // [FA_BK][FA_VC]

    // this thread: query rows ty*4 + i, S columns tx*4 + j, O columns c*64 + tx*4 + j
    float o[4][16];
    float mrow[4], lrow[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        mrow[i] = -1e30f; lrow[i] = 0.f;
#pragma unroll
        for (int j = 0; j < 16; j++) o[i][j] = 0.f;
    }
    const int qlast = min(tq0 + FA_BQ, T) - 1;       // the block's last live query
    const int kv_end = pos0 + qlast + 1;             // rows any of its queries may see
    for (int kv0 = 0; kv0 < kv_end; kv0 += FA_BK) {
        float s[4][4];
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 4; j++) s[i][j] = 0.f;
        // ---- S = Q K^T, 32 dims at a time ----
        for (int dc = 0; dc < 256; dc += FA_DC) {
            // stage Qc and Kc: 64 rows x 32 dims each, as fp32 (8 halves a thread a matrix)
            {
                const int r = tid >> 2, c8 = (tid & 3) * 8;
                const int tq = tq0 + r;
                uint4 qv = make_uint4(0, 0, 0, 0);
                if (tq < T) qv = __ldg(reinterpret_cast<const uint4*>(
                                 qbase + (size_t)tq * n_q_heads * q_pitch + dc + c8));
                const int kv = kv0 + r;
                float kf[8];
                if (Q8) {
                    uint2 kb = make_uint2(0, 0);
                    float ks = 0.f;
                    if (kv < kv_end) {
                        kb = __ldg(reinterpret_cast<const uint2*>(k8 + (long long)kv * kv_stride + dc + c8));
                        ks = __half2float(__ldg(&ksc[((long long)kv * n_kv_heads + hkv) * nsb + (dc >> 5)]));
                    }
                    const signed char* kc = reinterpret_cast<const signed char*>(&kb);
#pragma unroll
                    for (int u = 0; u < 8; u++) kf[u] = (float)kc[u] * ks;
                } else {
                    uint4 kvv = make_uint4(0, 0, 0, 0);
                    if (kv < kv_end) kvv = __ldg(reinterpret_cast<const uint4*>(
                                         kbase + (long long)kv * kv_stride + dc + c8));
                    const half2* kh = reinterpret_cast<const half2*>(&kvv);
#pragma unroll
                    for (int u = 0; u < 4; u++) {
                        const float2 b = __half22float2(kh[u]);
                        kf[2 * u] = b.x;
                        kf[2 * u + 1] = b.y;
                    }
                }
                const half2* qh = reinterpret_cast<const half2*>(&qv);
#pragma unroll
                for (int u = 0; u < 4; u++) {
                    const float2 a = __half22float2(qh[u]);
                    Qc[r * (FA_DC + 1) + c8 + 2 * u] = a.x;
                    Qc[r * (FA_DC + 1) + c8 + 2 * u + 1] = a.y;
                    Kc[r * (FA_DC + 1) + c8 + 2 * u] = kf[2 * u];
                    Kc[r * (FA_DC + 1) + c8 + 2 * u + 1] = kf[2 * u + 1];
                }
            }
            __syncthreads();
#pragma unroll 8
            for (int d = 0; d < FA_DC; d++) {
                float a[4], b[4];
#pragma unroll
                for (int i = 0; i < 4; i++) a[i] = Qc[(ty * 4 + i) * (FA_DC + 1) + d];
#pragma unroll
                for (int j = 0; j < 4; j++) b[j] = Kc[(tx * 4 + j) * (FA_DC + 1) + d];
#pragma unroll
                for (int i = 0; i < 4; i++)
#pragma unroll
                    for (int j = 0; j < 4; j++) s[i][j] = fmaf(a[i], b[j], s[i][j]);
            }
            __syncthreads();
        }
        // ---- causal mask, online softmax per row (16 threads of a half-warp) ----
        float pr[4][4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int qpos = pos0 + tq0 + ty * 4 + i;
            float mx = -1e30f;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int kv = kv0 + tx * 4 + j;
                s[i][j] = (kv <= qpos) ? s[i][j] * scale : -1e30f;
                mx = fmaxf(mx, s[i][j]);
            }
#pragma unroll
            for (int off = 8; off > 0; off >>= 1)
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
            const float mnew = fmaxf(mrow[i], mx);
            const float corr = __expf(mrow[i] - mnew);
            float sum = 0.f;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                pr[i][j] = (s[i][j] > -1e29f) ? __expf(s[i][j] - mnew) : 0.f;
                sum += pr[i][j];
            }
#pragma unroll
            for (int off = 8; off > 0; off >>= 1)
                sum += __shfl_xor_sync(0xffffffffu, sum, off);
            lrow[i] = lrow[i] * corr + sum;
            mrow[i] = mnew;
#pragma unroll
            for (int j = 0; j < 16; j++) o[i][j] *= corr;
        }
        // ---- O += P V, 64 dims of V at a time ----
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 4; j++) P[(ty * 4 + i) * FA_BK + tx * 4 + j] = pr[i][j];
#pragma unroll
        for (int vc = 0; vc < 4; vc++) {
            {
                // stage Vc: 64 rows x 64 dims (16 halves a thread)
                const int r = tid >> 2, c16 = (tid & 3) * 16;
                const int kv = kv0 + r;
                float4* dst = reinterpret_cast<float4*>(Vc + r * FA_VC + c16);
                if (Q8) {
                    uint4 vb = make_uint4(0, 0, 0, 0);
                    float vs = 0.f;
                    if (kv < kv_end) {
                        vb = __ldg(reinterpret_cast<const uint4*>(
                            v8 + (long long)kv * kv_stride + vc * FA_VC + c16));
                        vs = __half2float(__ldg(&vsc[((long long)kv * n_kv_heads + hkv) * nsb
                                                     + ((vc * FA_VC + c16) >> 5)]));
                    }
                    const signed char* vcb = reinterpret_cast<const signed char*>(&vb);
#pragma unroll
                    for (int u = 0; u < 4; u++)
                        dst[u] = make_float4((float)vcb[4 * u] * vs, (float)vcb[4 * u + 1] * vs,
                                             (float)vcb[4 * u + 2] * vs, (float)vcb[4 * u + 3] * vs);
                } else {
                    uint4 v0 = make_uint4(0, 0, 0, 0), v1 = make_uint4(0, 0, 0, 0);
                    if (kv < kv_end) {
                        const uint4* src = reinterpret_cast<const uint4*>(
                            vbase + (long long)kv * kv_stride + vc * FA_VC + c16);
                        v0 = __ldg(src);
                        v1 = __ldg(src + 1);
                    }
                    const half2* h0 = reinterpret_cast<const half2*>(&v0);
                    const half2* h1 = reinterpret_cast<const half2*>(&v1);
#pragma unroll
                    for (int u = 0; u < 2; u++) {
                        const float2 a = __half22float2(h0[2 * u]), b = __half22float2(h0[2 * u + 1]);
                        dst[u] = make_float4(a.x, a.y, b.x, b.y);
                        const float2 c = __half22float2(h1[2 * u]), e = __half22float2(h1[2 * u + 1]);
                        dst[2 + u] = make_float4(c.x, c.y, e.x, e.y);
                    }
                }
            }
            __syncthreads();
#pragma unroll 8
            for (int k = 0; k < FA_BK; k++) {
                float pa[4];
#pragma unroll
                for (int i = 0; i < 4; i++) pa[i] = P[(ty * 4 + i) * FA_BK + k];
                const float4 vv = *reinterpret_cast<const float4*>(Vc + k * FA_VC + tx * 4);
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    o[i][vc * 4 + 0] = fmaf(pa[i], vv.x, o[i][vc * 4 + 0]);
                    o[i][vc * 4 + 1] = fmaf(pa[i], vv.y, o[i][vc * 4 + 1]);
                    o[i][vc * 4 + 2] = fmaf(pa[i], vv.z, o[i][vc * 4 + 2]);
                    o[i][vc * 4 + 3] = fmaf(pa[i], vv.w, o[i][vc * 4 + 3]);
                }
            }
            __syncthreads();
        }
    }
    // ---- normalize and store: row ty*4+i, columns vc*64 + tx*4 + j ----
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int tq = tq0 + ty * 4 + i;
        if (tq >= T) continue;
        const float inv = __fdiv_rn(1.f, lrow[i]);
        half* zrow = z + (size_t)tq * n_q_heads * z_pitch + (size_t)head * z_pitch;
#pragma unroll
        for (int vc = 0; vc < 4; vc++) {

            const int col = vc * FA_VC + tx * 4;
            half2 h0 = __floats2half2_rn(o[i][vc * 4] * inv, o[i][vc * 4 + 1] * inv);
            half2 h1 = __floats2half2_rn(o[i][vc * 4 + 2] * inv, o[i][vc * 4 + 3] * inv);
            *reinterpret_cast<half2*>(zrow + col) = h0;
            *reinterpret_cast<half2*>(zrow + col + 2) = h1;
        }
    }
}

extern "C" __global__ void __launch_bounds__(256, 2)
attention_prefill_fa(const half* __restrict__ q, const half* __restrict__ kcache,
                     const half* __restrict__ vcache, half* __restrict__ z,
                     int layer, int n_q_heads, int n_kv_heads, int head_dim,
                     int q_pitch, int z_pitch, int pos0, int max_pos, float scale, int T) {
    attention_prefill_fa_body<false>(q, kcache, vcache, z, layer, n_q_heads, n_kv_heads,
                                     head_dim, q_pitch, z_pitch, pos0, max_pos, scale, T);
}

// The same over an int8 cache (KV8, kv_store_q8's layout); `layer` must be 0.
extern "C" __global__ void __launch_bounds__(256, 2)
attention_prefill_fa_q8(const half* __restrict__ q, const signed char* __restrict__ kcache,
                        const signed char* __restrict__ vcache, half* __restrict__ z,
                        int layer, int n_q_heads, int n_kv_heads, int head_dim,
                        int q_pitch, int z_pitch, int pos0, int max_pos, float scale, int T) {
    attention_prefill_fa_body<true>(q, kcache, vcache, z, layer, n_q_heads, n_kv_heads,
                                    head_dim, q_pitch, z_pitch, pos0, max_pos, scale, T);
}

// ============================================================================
// Prefill attention, mixed precision, GQA rows, kv split (`attention_prefill_fh`)
// ============================================================================
// The FA-2 shape of attention_prefill_fa with three changes:
//   * ROWS are (token, q head) pairs of ONE kv head, token-major (row = t * G + g,
//     G = n_q_heads / n_kv_heads): a block's 64 rows share every K/V tile it
//     stages, and a short chunk fills them (T = 16 is 96 rows, 2 tiles, where a
//     per-head tile of 64 queries was 25% live).
//   * the KV range is SPLIT over gridDim.z in whole 64-row tiles: S > 1 writes
//     unnormalized partials (pacc [t][head][s][256], pm in natural-log units,
//     pd) for attention_merge_fh; S = 1 writes z. A short chunk deep in the
//     context ran 24 blocks on 56 SMs.
//   * S = Q K^T stays fp32 (FFMA over fp32-staged Q and K: a score's error is
//     an exponent's, and fp16 accumulation of it measured 7e-3 relative output
//     error against 2e-4); O += P V runs in fp16 (P rounded to half, HFMA2 over
//     the 64 rows of a tile, then widened into the fp32 accumulator with the
//     online-softmax correction): 4e-4 relative output error, against the int8
//     cache's own 1.4e-2.
// Per tile: S over 8 chunks of 32 dims (double-buffered, one barrier a chunk,
// the next chunk fetched into registers during the current FMAs), the causal
// mask and online softmax in exp2 (sl2 = scale * log2 e), P to shared memory as
// [kv][row], then V in 4 chunks of 16 rows aliasing the S buffers. The int8
// cache is dequantized while staging (K: byte * scale in fp32, exactly the
// float the fp32 kernel used; V: the byte made exact in fp16, one rounding of
// the product). The heaviest (latest) row tiles launch first.
// grid (n_kv_heads, ceil(T * G / 64), S), block 256; head_dim 256.
// ============================================================================
#define FH_BQ 64                 // (token, q head) rows a block
#define FH_BK 64                 // kv rows a tile
#define FH_DC 32                 // dims of Q / K a chunk
#define FH_FQP 36                // Qs / Ks pitch, floats (144 B: conflict-free LDS.128)
#define FH_VR 16                 // kv rows a V chunk
#define FH_VP 264                // Vs pitch, halves
#define FH_PP 72                 // Ps pitch, halves
#define FH_BUF (2 * FH_BQ * FH_FQP)                    // floats of one S buffer (Qs + Ks)
#define FH_SM_FLOATS (2 * FH_BUF + FH_BK * FH_PP / 2)  // S buffers (V aliases them) + P

union FhU4 { uint4 u; half2 h[4]; };

// 4 signed bytes -> 2 half2 of byte * s: the byte is exact in fp16 (0x64XX is
// 1024 + XX), the product rounds once.
__device__ __forceinline__ void fh_deq4(unsigned int b, half2 s, half2& lo, half2& hi) {
    const unsigned int u = b ^ 0x80808080u;                     // byte + 128
    const unsigned int a = __byte_perm(u, 0x64646464u, 0x4140);
    const unsigned int c = __byte_perm(u, 0x64646464u, 0x4342);
    const half2 off = __floats2half2_rn(1152.f, 1152.f);
    lo = __hmul2(__hsub2(*reinterpret_cast<const half2*>(&a), off), s);
    hi = __hmul2(__hsub2(*reinterpret_cast<const half2*>(&c), off), s);
}

template <bool Q8>
static __device__ __forceinline__ void attention_prefill_fh_body(
        const half* __restrict__ q, const void* __restrict__ kcache,
        const void* __restrict__ vcache, half* __restrict__ z,
        float* __restrict__ pacc, float* __restrict__ pm, float* __restrict__ pd,
        int n_q_heads, int n_kv_heads, int head_dim, int q_pitch, int z_pitch,
        int pos0, int max_pos, float sl2, int T) {
    __shared__ __align__(16) float sm[FH_SM_FLOATS];
    if (head_dim != 256 || n_q_heads % n_kv_heads != 0) return;
    const int G = n_q_heads / n_kv_heads;
    const int hkv = blockIdx.x;
    const int tile = gridDim.y - 1 - blockIdx.y;
    const int S = gridDim.z, sz = blockIdx.z;
    const int R0 = tile * FH_BQ;
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const long long kvs = (long long)n_kv_heads * 256;         // a cache row, elements
    const int kv_end = pos0 + min(R0 + FH_BQ - 1, T * G - 1) / G + 1;
    const int nkt = (pos0 + T + FH_BK - 1) / FH_BK;
    const int kv_lo = (int)((long long)sz * nkt / S) * FH_BK;
    const int kv_hi = min((int)((long long)(sz + 1) * nkt / S) * FH_BK, kv_end);

    half* Ps = reinterpret_cast<half*>(sm + 2 * FH_BUF);       // [kv][row]
    // staging: S chunk row sr, 8 dims at c8; V chunk kv row vr, 16 dims at c16
    const int sr = tid >> 2, c8 = (tid & 3) * 8;
    const int st = (R0 + sr) / G, sg = R0 + sr - st * G;
    const bool sq = st < T;
    const half* qsrc = q + ((size_t)(sq ? st : 0) * n_q_heads + hkv * G + sg) * q_pitch + c8;
    const int vr = tid >> 4, c16 = (tid & 15) * 16;
    const signed char* k8 = reinterpret_cast<const signed char*>(kcache) + hkv * 256;
    const signed char* v8 = reinterpret_cast<const signed char*>(vcache) + hkv * 256;
    const half* ksc = reinterpret_cast<const half*>(
        reinterpret_cast<const signed char*>(kcache) + (long long)max_pos * kvs);
    const half* vsc = reinterpret_cast<const half*>(
        reinterpret_cast<const signed char*>(vcache) + (long long)max_pos * kvs);
    const half* k16 = reinterpret_cast<const half*>(kcache) + hkv * 256;
    const half* v16 = reinterpret_cast<const half*>(vcache) + hkv * 256;

    // this thread: rows ty*4 + i; S columns tx + 16 j; O dims tx*8 + 0..7, 128 + tx*8 + 0..7
    int qpos[4];
    float o[4][16], mrow[4], lrow[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int t = (R0 + ty * 4 + i) / G;
        qpos[i] = t < T ? pos0 + t : -1;                       // a padding row sees nothing
        mrow[i] = -1e30f; lrow[i] = 0.f;
#pragma unroll
        for (int j = 0; j < 16; j++) o[i][j] = 0.f;
    }

    uint4 qn, kn16; uint2 kn8; half ksn;
    auto fetchS = [&](int kv0, int c) {
        const int dc = c * FH_DC;
        qn = sq ? __ldg(reinterpret_cast<const uint4*>(qsrc + dc)) : make_uint4(0, 0, 0, 0);
        const int kv = kv0 + sr;
        if (Q8) {
            if (kv < kv_end) {
                kn8 = __ldg(reinterpret_cast<const uint2*>(k8 + (long long)kv * kvs + dc + c8));
                ksn = __ldg(&ksc[((long long)kv * n_kv_heads + hkv) * 8 + c]);
            } else { kn8 = make_uint2(0, 0); ksn = __float2half(0.f); }
        } else {
            kn16 = kv < kv_end ? __ldg(reinterpret_cast<const uint4*>(k16 + (long long)kv * kvs + dc + c8))
                               : make_uint4(0, 0, 0, 0);
        }
    };
    auto storeS = [&](int b) {
        float* Qs = sm + b * FH_BUF;
        float* Ks = Qs + FH_BQ * FH_FQP;
        const half2* qh = reinterpret_cast<const half2*>(&qn);
        float4* qd = reinterpret_cast<float4*>(Qs + sr * FH_FQP + c8);
        float2 a = __half22float2(qh[0]), e = __half22float2(qh[1]);
        qd[0] = make_float4(a.x, a.y, e.x, e.y);
        a = __half22float2(qh[2]); e = __half22float2(qh[3]);
        qd[1] = make_float4(a.x, a.y, e.x, e.y);
        float4* kd = reinterpret_cast<float4*>(Ks + sr * FH_FQP + c8);
        if (Q8) {
            const float ks = __half2float(ksn);
            const signed char* kc = reinterpret_cast<const signed char*>(&kn8);
            kd[0] = make_float4((float)kc[0] * ks, (float)kc[1] * ks, (float)kc[2] * ks, (float)kc[3] * ks);
            kd[1] = make_float4((float)kc[4] * ks, (float)kc[5] * ks, (float)kc[6] * ks, (float)kc[7] * ks);
        } else {
            const half2* kh = reinterpret_cast<const half2*>(&kn16);
            a = __half22float2(kh[0]); e = __half22float2(kh[1]);
            kd[0] = make_float4(a.x, a.y, e.x, e.y);
            a = __half22float2(kh[2]); e = __half22float2(kh[3]);
            kd[1] = make_float4(a.x, a.y, e.x, e.y);
        }
    };
    uint4 vn8, vn16a, vn16b; half vsn;
    auto fetchV = [&](int kv0, int vc) {
        const int kv = kv0 + vc * FH_VR + vr;
        if (Q8) {
            if (kv < kv_end) {
                vn8 = __ldg(reinterpret_cast<const uint4*>(v8 + (long long)kv * kvs + c16));
                vsn = __ldg(&vsc[((long long)kv * n_kv_heads + hkv) * 8 + (c16 >> 5)]);
            } else { vn8 = make_uint4(0, 0, 0, 0); vsn = __float2half(0.f); }
        } else if (kv < kv_end) {                              // rows past kv_end may be stale
            const uint4* s = reinterpret_cast<const uint4*>(v16 + (long long)kv * kvs + c16);
            vn16a = __ldg(s); vn16b = __ldg(s + 1);
        } else {
            vn16a = vn16b = make_uint4(0, 0, 0, 0);
        }
    };
    auto storeV = [&](int b) {
        half* Vs = reinterpret_cast<half*>(sm) + b * (FH_VR * FH_VP);
        uint4* d = reinterpret_cast<uint4*>(Vs + vr * FH_VP + c16);
        if (Q8) {
            FhU4 w0, w1;
            const half2 s2 = __half2half2(vsn);
            fh_deq4(vn8.x, s2, w0.h[0], w0.h[1]);
            fh_deq4(vn8.y, s2, w0.h[2], w0.h[3]);
            fh_deq4(vn8.z, s2, w1.h[0], w1.h[1]);
            fh_deq4(vn8.w, s2, w1.h[2], w1.h[3]);
            d[0] = w0.u; d[1] = w1.u;
        } else {
            d[0] = vn16a; d[1] = vn16b;
        }
    };

    for (int kv0 = kv_lo; kv0 < kv_hi; kv0 += FH_BK) {
        // ---- S = Q K^T in fp32 ----
        float s[4][4];
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 4; j++) s[i][j] = 0.f;
        fetchS(kv0, 0);
#pragma unroll 1
        for (int c = 0; c < 256 / FH_DC; c++) {
            const int b = c & 1;
            storeS(b);                                         // its last reader was chunk c - 2
            __syncthreads();
            if (c + 1 < 256 / FH_DC) fetchS(kv0, c + 1);
            else fetchV(kv0, 0);
            const float* Qs = sm + b * FH_BUF;
            const float* Ks = Qs + FH_BQ * FH_FQP;
#pragma unroll
            for (int dd = 0; dd < FH_DC; dd += 4) {
                float4 kb[4];
#pragma unroll
                for (int j = 0; j < 4; j++)
                    kb[j] = *reinterpret_cast<const float4*>(Ks + (tx + 16 * j) * FH_FQP + dd);
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    const float4 qa = *reinterpret_cast<const float4*>(Qs + (ty * 4 + i) * FH_FQP + dd);
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        s[i][j] = fmaf(qa.x, kb[j].x, s[i][j]);
                        s[i][j] = fmaf(qa.y, kb[j].y, s[i][j]);
                        s[i][j] = fmaf(qa.z, kb[j].z, s[i][j]);
                        s[i][j] = fmaf(qa.w, kb[j].w, s[i][j]);
                    }
                }
            }
        }
        // ---- causal mask, online softmax (a row's 64 scores over the 16 tx lanes) ----
        float corr[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            float mx = -1e30f;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                s[i][j] = (kv0 + tx + 16 * j <= qpos[i]) ? s[i][j] * sl2 : -1e30f;
                mx = fmaxf(mx, s[i][j]);
            }
#pragma unroll
            for (int off = 8; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
            const float mnew = fmaxf(mrow[i], mx);
            corr[i] = exp2f(mrow[i] - mnew);
            float sum = 0.f;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                s[i][j] = (s[i][j] > -1e29f) ? exp2f(s[i][j] - mnew) : 0.f;
                sum += s[i][j];
            }
#pragma unroll
            for (int off = 8; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
            lrow[i] = lrow[i] * corr[i] + sum;
            mrow[i] = mnew;
        }
        // Ps's last reader was the previous tile's V loop, eight barriers ago
#pragma unroll
        for (int j = 0; j < 4; j++) {
            union { uint2 u; half2 h[2]; } pk;
            pk.h[0] = __floats2half2_rn(s[0][j], s[1][j]);
            pk.h[1] = __floats2half2_rn(s[2][j], s[3][j]);
            *reinterpret_cast<uint2*>(Ps + (tx + 16 * j) * FH_PP + ty * 4) = pk.u;
        }
        // ---- O += P V in fp16 over the tile, then into fp32 ----
        half2 oh[4][8];
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int u = 0; u < 8; u++) oh[i][u] = __float2half2_rn(0.f);
#pragma unroll 1
        for (int vc = 0; vc < FH_BK / FH_VR; vc++) {
            const int b = vc & 1;
            storeV(b);         // Vs[0] aliases S buffer 0 (last read by chunk 6); Vs[1]
            __syncthreads();   // reaches into buffer 1, written only after this barrier
            if (vc + 1 < FH_BK / FH_VR) fetchV(kv0, vc + 1);
            const half* Vs = reinterpret_cast<const half*>(sm) + b * (FH_VR * FH_VP);
#pragma unroll 4
            for (int kk = 0; kk < FH_VR; kk++) {
                union { uint2 u; half2 h[2]; } pp;
                pp.u = *reinterpret_cast<const uint2*>(Ps + (vc * FH_VR + kk) * FH_PP + ty * 4);
                FhU4 va, vb;
                va.u = *reinterpret_cast<const uint4*>(Vs + kk * FH_VP + tx * 8);
                vb.u = *reinterpret_cast<const uint4*>(Vs + kk * FH_VP + 128 + tx * 8);
                half2 pb[4];
                pb[0] = __low2half2(pp.h[0]); pb[1] = __high2half2(pp.h[0]);
                pb[2] = __low2half2(pp.h[1]); pb[3] = __high2half2(pp.h[1]);
#pragma unroll
                for (int i = 0; i < 4; i++)
#pragma unroll
                    for (int u = 0; u < 4; u++) {
                        oh[i][u] = __hfma2(pb[i], va.h[u], oh[i][u]);
                        oh[i][4 + u] = __hfma2(pb[i], vb.h[u], oh[i][4 + u]);
                    }
            }
        }
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int u = 0; u < 8; u++) {
                o[i][2 * u] = fmaf(o[i][2 * u], corr[i], __low2float(oh[i][u]));
                o[i][2 * u + 1] = fmaf(o[i][2 * u + 1], corr[i], __high2float(oh[i][u]));
            }
        __syncthreads();                                       // V read before the next S stores
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int R = R0 + ty * 4 + i, t = R / G;
        if (t >= T) continue;
        const int head = hkv * G + (R - t * G);
        if (S == 1) {
            const float inv = __fdiv_rn(1.f, lrow[i]);
            half* zr = z + (size_t)t * n_q_heads * z_pitch + (size_t)head * z_pitch;
#pragma unroll
            for (int h = 0; h < 2; h++) {
                union { uint4 u; half hv[8]; } w;
#pragma unroll
                for (int u = 0; u < 8; u++) w.hv[u] = f16_sat(o[i][8 * h + u] * inv);
                *reinterpret_cast<uint4*>(zr + 128 * h + tx * 8) = w.u;
            }
        } else {
            const size_t p = ((size_t)t * n_q_heads + head) * S + sz;
            float* pa = pacc + p * 256;
#pragma unroll
            for (int h = 0; h < 2; h++) {
                float4* d = reinterpret_cast<float4*>(pa + 128 * h + tx * 8);
                d[0] = make_float4(o[i][8 * h], o[i][8 * h + 1], o[i][8 * h + 2], o[i][8 * h + 3]);
                d[1] = make_float4(o[i][8 * h + 4], o[i][8 * h + 5], o[i][8 * h + 6], o[i][8 * h + 7]);
            }
            if (tx == 0) { pm[p] = mrow[i] * 0.6931471805599453f; pd[p] = lrow[i]; }
        }
    }
}

extern "C" __global__ void __launch_bounds__(256, 1)
attention_prefill_fh(const half* __restrict__ q, const half* __restrict__ kcache,
                     const half* __restrict__ vcache, half* __restrict__ z,
                     float* __restrict__ pacc, float* __restrict__ pm, float* __restrict__ pd,
                     int n_q_heads, int n_kv_heads, int head_dim, int q_pitch, int z_pitch,
                     int pos0, int max_pos, float sl2, int T) {
    attention_prefill_fh_body<false>(q, kcache, vcache, z, pacc, pm, pd, n_q_heads, n_kv_heads,
                                     head_dim, q_pitch, z_pitch, pos0, max_pos, sl2, T);
}

extern "C" __global__ void __launch_bounds__(256, 1)
attention_prefill_fh_q8(const half* __restrict__ q, const signed char* __restrict__ kcache,
                        const signed char* __restrict__ vcache, half* __restrict__ z,
                        float* __restrict__ pacc, float* __restrict__ pm, float* __restrict__ pd,
                        int n_q_heads, int n_kv_heads, int head_dim, int q_pitch, int z_pitch,
                        int pos0, int max_pos, float sl2, int T) {
    attention_prefill_fh_body<true>(q, kcache, vcache, z, pacc, pm, pd, n_q_heads, n_kv_heads,
                                    head_dim, q_pitch, z_pitch, pos0, max_pos, sl2, T);
}

// attention_prefill_fh's S partials of a (token, head) into its z row: the
// merge attention_merge does, one block a (token, head) and no idle z blocks.
// grid (T, n_q_heads), block head_dim (<= 1024).
extern "C" __global__ void attention_merge_fh(const float* __restrict__ pacc,
                                             const float* __restrict__ pm,
                                             const float* __restrict__ pd,
                                             half* __restrict__ z, int n_q_heads,
                                             int head_dim, int z_pitch, int S) {
    const int t = blockIdx.x, head = blockIdx.y, d = threadIdx.x;
    const size_t p0 = ((size_t)t * n_q_heads + head) * S;
    float M = -1e30f;
    for (int s = 0; s < S; s++) M = fmaxf(M, pm[p0 + s]);
    float num = 0.f, den = 0.f;
    for (int s = 0; s < S; s++) {
        const float e = __expf(pm[p0 + s] - M);
        num = fmaf(e, pacc[(p0 + s) * head_dim + d], num);
        den = fmaf(e, pd[p0 + s], den);
    }
    z[(size_t)t * n_q_heads * z_pitch + (size_t)head * z_pitch + d] = f16_sat(num * __fdiv_rn(1.f, den));
}

// ============================================================================
// gdn_scan for prefill: the state's rows over more blocks (`gdn_scan_rows`)
// ============================================================================
// The gated delta rule's rows are independent: row v of S needs only its own
// values, k_t, q_t, beta_t, decay_t and v_t[v]. gdn_scan gives a head ONE block
// (48 blocks on 56 SMs, each a 64-FMA dependent chain per dot, twice a token,
// with two barriers a token). Here a block owns GR_RB = 32 rows of a head
// (grid 48 x 4), GR_TPR = 8 threads a row (16 columns in registers), and k, q
// (as fp32), v, beta and decay are staged GR_TC tokens at a time, one barrier
// pair a stage. Per token: S *= decay, m = S . k, delta = beta (v - m),
// S += delta k, o = S . q -- gdn_scan's arithmetic per element; each dot sums
// 16 columns in two chains, then the row's 8 partials by xor-shuffle, so the
// values differ from gdn_scan's only in the fp32 order of those sums.
// grid (n_vheads, V / GR_RB), block GR_RB * GR_TPR (K = 128).
// ============================================================================
#ifndef GR_TPR
#define GR_TPR 8
#endif
#define GR_RB (256 / GR_TPR)
#define GR_NC (128 / GR_TPR)        // columns a thread
#define GR_TC 12
extern "C" __global__ void __launch_bounds__(GR_RB * GR_TPR, 4)
gdn_scan_rows(const half* __restrict__ q, const half* __restrict__ k,
              const half* __restrict__ v, const float* __restrict__ beta,
              const float* __restrict__ decay, float* __restrict__ S,
              half* __restrict__ y, int T, int n_kheads, int n_vheads, int K, int V,
              int qk_stride, int v_stride, int y_stride, float out_scale, int group) {
    __shared__ __align__(16) float sk[GR_TC][128];
    __shared__ __align__(16) float sq[GR_TC][128];
    __shared__ float sv[GR_TC][GR_RB];
    __shared__ float sb[GR_TC], sd[GR_TC];
    if (K != 128 || group <= 0 || V % GR_RB != 0) return;
    const int hv = blockIdx.x, hk = hv / group;
    if (hk >= n_kheads) return;
    const int tid = threadIdx.x, ri = tid / GR_TPR, c0 = (tid % GR_TPR) * 4;
    const int r0 = blockIdx.y * GR_RB, vi = r0 + ri;
    float* Srow = S + ((size_t)hv * V + vi) * K + c0;   // columns c0 + 32 j + 0..3
    float reg[GR_NC];
#pragma unroll
    for (int c = 0; c < GR_NC; c += 4) {
        const float4 v4 = __ldg(reinterpret_cast<const float4*>(Srow + GR_TPR * c));
        reg[c] = v4.x; reg[c + 1] = v4.y; reg[c + 2] = v4.z; reg[c + 3] = v4.w;
    }
    for (int t0 = 0; t0 < T; t0 += GR_TC) {
        const int n = min(GR_TC, T - t0);
        __syncthreads();                               // the previous stage is consumed
        for (int i = tid; i < n * 64; i += blockDim.x) {
            const int t = i >> 6, c = (i & 63) * 2;
            const size_t a = (size_t)(t0 + t) * qk_stride + (size_t)hk * K + c;
            const float2 kf = __half22float2(*reinterpret_cast<const half2*>(k + a));
            const float2 qf = __half22float2(*reinterpret_cast<const half2*>(q + a));
            sk[t][c] = kf.x; sk[t][c + 1] = kf.y;
            sq[t][c] = qf.x; sq[t][c + 1] = qf.y;
        }
        for (int i = tid; i < n * GR_RB; i += blockDim.x) {
            const int t = i / GR_RB, r = i % GR_RB;
            sv[t][r] = __half2float(v[(size_t)(t0 + t) * v_stride + (size_t)hv * V + r0 + r]);
        }
        if (tid < n) {
            sb[tid] = beta[(size_t)(t0 + tid) * n_vheads + hv];
            sd[tid] = decay[(size_t)(t0 + tid) * n_vheads + hv];
        }
        __syncthreads();
        for (int t = 0; t < n; t++) {
            const float dt = sd[t], bt = sb[t];
            float kk[GR_NC];
#pragma unroll
            for (int c = 0; c < GR_NC; c += 4) {
                const float4 f = *reinterpret_cast<const float4*>(&sk[t][c0 + GR_TPR * c]);
                kk[c] = f.x; kk[c + 1] = f.y; kk[c + 2] = f.z; kk[c + 3] = f.w;
            }
            float m0 = 0.f, m1 = 0.f;
#pragma unroll
            for (int c = 0; c < GR_NC; c += 2) {
                reg[c] *= dt; reg[c + 1] *= dt;
                m0 = fmaf(reg[c], kk[c], m0);
                m1 = fmaf(reg[c + 1], kk[c + 1], m1);
            }
            float m = m0 + m1;
#pragma unroll
            for (int off = 1; off < GR_TPR; off <<= 1) m += __shfl_xor_sync(0xffffffffu, m, off);
            const float delta = bt * (sv[t][ri] - m);
            float o0 = 0.f, o1 = 0.f;
#pragma unroll
            for (int c = 0; c < GR_NC; c += 4) {
                const float4 f = *reinterpret_cast<const float4*>(&sq[t][c0 + GR_TPR * c]);
                reg[c] = fmaf(delta, kk[c], reg[c]);
                reg[c + 1] = fmaf(delta, kk[c + 1], reg[c + 1]);
                reg[c + 2] = fmaf(delta, kk[c + 2], reg[c + 2]);
                reg[c + 3] = fmaf(delta, kk[c + 3], reg[c + 3]);
                o0 = fmaf(reg[c], f.x, o0);
                o1 = fmaf(reg[c + 1], f.y, o1);
                o0 = fmaf(reg[c + 2], f.z, o0);
                o1 = fmaf(reg[c + 3], f.w, o1);
            }
            float o = o0 + o1;
#pragma unroll
            for (int off = 1; off < GR_TPR; off <<= 1) o += __shfl_xor_sync(0xffffffffu, o, off);
            if (c0 == 0)
                y[(size_t)(t0 + t) * y_stride + (size_t)hv * V + vi] = f16_sat(o * out_scale);
        }
    }
#pragma unroll
    for (int c = 0; c < GR_NC; c += 4)
        *reinterpret_cast<float4*>(Srow + GR_TPR * c) = make_float4(reg[c], reg[c + 1], reg[c + 2], reg[c + 3]);
}

// conv1d_causal at T = 1 for one channel (width 4): the output, and the
// state it leaves when `st_out` is given -- its expressions and stores.
__device__ __forceinline__ half gdn_conv1(const half* __restrict__ x,
                                          const half* __restrict__ w,
                                          const half* __restrict__ st_in,
                                          half* __restrict__ st_out, int d, int pos0) {
    const half* st = st_in + (size_t)d * 3;
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const float e = j < 3 ? (pos0 > 0 ? __half2float(__ldg(&st[j])) : 0.f)
                              : __half2float(__ldg(&x[d]));
        acc = fmaf(__half2float(__ldg(&w[(size_t)d * 4 + j])), e, acc);
    }
    if (st_out != nullptr) {
        half* so = st_out + (size_t)d * 3;
#pragma unroll
        for (int k = 0; k < 3; k++) {
            const int i = 1 + k;
            const float e = i < 3 ? (pos0 > 0 ? __half2float(__ldg(&st[i])) : 0.f)
                                  : __half2float(__ldg(&x[d]));
            so[k] = f16_sat(e);
        }
    }
    return f16_sat(silu_f(acc));
}

// gdn_scan_f's shape for the decode and the verify (T <= 3) on gdn_scan_rows'
// rows: a block owns GR_RB rows of a head, GR_TPR threads a row, 4x the warps
// and 8-FMA chains where gdn_scan_f ran 128 rows a block in 64-FMA chains. k and
// q are L2-normalised in every block exactly as gdn_scan_f does it (the same
// reductions, division and fp16 rounding), beta and decay are its expressions,
// and the checkpoints after every row but the last are its layout; the dots
// sum in gdn_scan_rows' order. `cx` non-null (T = 1 only): q, k and v are
// the conv of the raw qkv row `cx` instead, gdn_conv1 per channel, the state
// into `cst_out` (never `cst_in`: other blocks still read it), so the decode
// skips conv1d_causal's launch. grid (n_vheads, V / GR_RB), block 256.
extern "C" __global__ void __launch_bounds__(GR_RB * GR_TPR, 4)
gdn_scan_rows_f(const half* __restrict__ q, const half* __restrict__ k,
                const half* __restrict__ v, float* __restrict__ S, half* __restrict__ y,
                int T, int n_kheads, int n_vheads, int K, int V, int qk_stride, int v_stride,
                int y_stride, float out_scale, int group, float* __restrict__ Ck,
                long long ck_step, const float* __restrict__ ga, const float* __restrict__ gb,
                const float* __restrict__ A_log, const float* __restrict__ dt_bias,
                int ab_stride, float l2eps, const half* __restrict__ cx,
                const half* __restrict__ cw, const half* __restrict__ cst_in,
                half* __restrict__ cst_out, int pos0) {
    __shared__ __align__(16) float sk[128];
    __shared__ __align__(16) float sq[128];
    __shared__ float red[8];
    if (K != 128 || group <= 0 || V % GR_RB != 0 || blockDim.x != 256) return;
    const int hv = blockIdx.x, hk = hv / group;
    if (hk >= n_kheads) return;
    const int tid = threadIdx.x, ri = tid / GR_TPR, c0 = (tid % GR_TPR) * 4;
    const int vi = blockIdx.y * GR_RB + ri;
    float* Srow = S + ((size_t)hv * V + vi) * K + c0;   // columns c0 + 32 j + 0..3
    float reg[GR_NC];
#pragma unroll
    for (int c = 0; c < GR_NC; c += 4) {
        const float4 v4 = __ldg(reinterpret_cast<const float4*>(Srow + GR_TPR * c));
        reg[c] = v4.x; reg[c + 1] = v4.y; reg[c + 2] = v4.z; reg[c + 3] = v4.w;
    }
    const float dtb = __ldg(&dt_bias[hv]), al = __expf(__ldg(&A_log[hv]));
    for (int t = 0; t < T; t++) {
        const float bt = sigmoid_f(__ldg(&gb[(size_t)t * ab_stride + hv]));
        float dt;
        {
            const float xx = __ldg(&ga[(size_t)t * ab_stride + hv]) + dtb;
            const float sp = xx <= 20.f ? log1pf(__expf(xx)) : xx;
            dt = __expf(-al * sp);
        }
        float vt;
        if (cx != nullptr) {
            // the conv of this row's v channel, conv1d_causal's T = 1 arithmetic
            // (every thread of the row the same value; one stores the state)
            const int ch = 2 * n_kheads * K + hv * V + vi;
            vt = __half2float(gdn_conv1(cx, cw, cst_in, c0 == 0 ? cst_out : nullptr, ch, pos0));
        } else {
            vt = __half2float(__ldg(&v[(size_t)t * v_stride + (size_t)hv * V + vi]));
        }
        {   // gdn_scan_f's l2norm: k by threads 0..127, q by 128..255
            const int half_ = tid >> 7, i = tid & 127;
            float x;
            if (cx != nullptr) {
                // this thread's q or k channel through the conv; its state is
                // stored once, by the head's first v-head's first row block
                const int ch = (half_ ? 0 : n_kheads * K) + hk * K + i;
                const bool own = (hv % group) == 0 && blockIdx.y == 0;
                x = __half2float(gdn_conv1(cx, cw, cst_in, own ? cst_out : nullptr, ch, pos0));
            } else {
                const half* row = (half_ ? q : k) + (size_t)t * qk_stride + (size_t)hk * K;
                x = __half2float(__ldg(&row[i]));
            }
            float ss = warp_reduce_sum(fmaf(x, x, 0.f));
            const int wid = tid >> 5, lane = tid & 31;
            if (lane == 0) red[wid] = ss;
            __syncthreads();
            if ((wid & 3) == 0) {
                float w4 = (lane < 4) ? red[(wid & 4) + lane] : 0.f;
                w4 = warp_reduce_sum(w4);
                if (lane == 0) red[wid] = w4;
            }
            __syncthreads();
            const float den = sqrtf(red[half_ * 4] + l2eps);
            (half_ ? sq : sk)[i] = __half2float(f16_sat(__fdiv_rn(x, den) * 1.0f));
        }
        __syncthreads();
        float kk[GR_NC];
#pragma unroll
        for (int c = 0; c < GR_NC; c += 4) {
            const float4 f = *reinterpret_cast<const float4*>(&sk[c0 + GR_TPR * c]);
            kk[c] = f.x; kk[c + 1] = f.y; kk[c + 2] = f.z; kk[c + 3] = f.w;
        }
        float m0 = 0.f, m1 = 0.f;
#pragma unroll
        for (int c = 0; c < GR_NC; c += 2) {
            reg[c] *= dt; reg[c + 1] *= dt;
            m0 = fmaf(reg[c], kk[c], m0);
            m1 = fmaf(reg[c + 1], kk[c + 1], m1);
        }
        float m = m0 + m1;
#pragma unroll
        for (int off = 1; off < GR_TPR; off <<= 1) m += __shfl_xor_sync(0xffffffffu, m, off);
        const float delta = bt * (vt - m);
        float o0 = 0.f, o1 = 0.f;
#pragma unroll
        for (int c = 0; c < GR_NC; c += 4) {
            const float4 f = *reinterpret_cast<const float4*>(&sq[c0 + GR_TPR * c]);
            reg[c] = fmaf(delta, kk[c], reg[c]);
            reg[c + 1] = fmaf(delta, kk[c + 1], reg[c + 1]);
            reg[c + 2] = fmaf(delta, kk[c + 2], reg[c + 2]);
            reg[c + 3] = fmaf(delta, kk[c + 3], reg[c + 3]);
            o0 = fmaf(reg[c], f.x, o0);
            o1 = fmaf(reg[c + 1], f.y, o1);
            o0 = fmaf(reg[c + 2], f.z, o0);
            o1 = fmaf(reg[c + 3], f.w, o1);
        }
        float o = o0 + o1;
#pragma unroll
        for (int off = 1; off < GR_TPR; off <<= 1) o += __shfl_xor_sync(0xffffffffu, o, off);
        if (c0 == 0)
            y[(size_t)t * y_stride + (size_t)hv * V + vi] = f16_sat(o * out_scale);
        if (Ck != nullptr && t < T - 1) {
            float* ckrow = Ck + (size_t)t * ck_step + ((size_t)hv * V + vi) * K + c0;
#pragma unroll
            for (int c = 0; c < GR_NC; c += 4)
                *reinterpret_cast<float4*>(ckrow + GR_TPR * c) =
                    make_float4(reg[c], reg[c + 1], reg[c + 2], reg[c + 3]);
        }
        __syncthreads();                               // sk / sq are reused next token
    }
#pragma unroll
    for (int c = 0; c < GR_NC; c += 4)
        *reinterpret_cast<float4*>(Srow + GR_TPR * c) = make_float4(reg[c], reg[c + 1], reg[c + 2], reg[c + 3]);
}
