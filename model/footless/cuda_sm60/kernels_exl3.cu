// ============================================================================
// kernels_exl3.cu -- the EXL3 trellis-decode GEMV/GEMM and the two Hadamard
// helpers of the OrcaSAQ-2-27B (Qwen3.5 dense hybrid) runtime, sm_60 / P100.
//
//   Layout contract:  cuda_sm60/_build/DESIGN.md
//   Decode spec:      cuda_sm60/_build/exl3_oracle.py -- a numpy decoder that is
//                     VALIDATED BIT-EXACT against the golden dumps for all five
//                     bitrates in this checkpoint
//                     (`python exl3_oracle.py <model_dir>` -> ALL TILES MATCH).
//                     Every line of the decode below is a port of that oracle,
//                     not of EXL3_FORMAT.md's prose. Where the two could differ
//                     the oracle wins, and the golden test settles it.
//   Format:           _reference/EXL3_FORMAT.md sections 3-5
//   Golden reference: cuda_sm60/_build/exl3_ref/golden/A_*.f16.bin, raw fp16
//                     (in, out) row-major, straight out of the checkpoint's
//                     safetensors by upstream reconstruct.cu on this device.
//   Tests:            tests/test_exl3.py
//   Sibling kernels:  kernels.cu (another workstream -- do not include it, do
//                     not edit it). Every helper here is `static` or
//                     `__forceinline__ static` so the two sources cannot
//                     collide on a symbol.
//
// ----------------------------------------------------------------------------
// WHAT THIS FILE IMPLEMENTS
// ----------------------------------------------------------------------------
// The deploy weight of a quantized projection is
//
//     W(out,in) = diag(suh) . H . A . H . diag(svh),   A = decode(trellis)
//
// with H = Sylvester(128)/sqrt(128) applied BLOCKWISE along each axis. H is
// never built and W is never materialised; the fused form is
//
//     xh = had128_pre (x, suh)            x * suh, Hadamard over the input axis
//     z  = exl3_gemv  (xh, trellis)       the GEMV over the raw trellis
//     y  = had128_post(z, svh)            Hadamard over the output axis, * svh
//
// `exl3_gemv` is the kernel that matters: it reads the int16 trellis, decodes
// each 16-bit window with the mul1 codebook in registers and accumulates
// straight into z. 8.3 bytes of trellis per fp16 A element are read and thrown
// away per token, and nothing else is.
//
// ----------------------------------------------------------------------------
// THE DECODE, AS THE ORACLE IMPLEMENTS IT (exl3_dq.cuh with the one sm_60
// substitution EXL3_FORMAT.md section 4 mandates: `__dp4a` does not exist on
// sm_60 and nvcc refuses it outright)
// ----------------------------------------------------------------------------
// A tile of the trellis is 16*bits int16 = 8*bits uint32 per 16x16 weight block.
// Window `t` (t = 0..255) of a tile is the 16-bit field starting at bit
// `t*bits + bits - 16 + 256*bits` of the tile's bit stream, taken modulo
// `bits*256` bits -- the `+256*bits` term and that modulo ARE the tail-biting
// wrap, and they are the first thing the mutation pass breaks on purpose.
//
// The mul1 (cb 2) codebook turns a window into fp16 with one integer multiply,
// a byte sum and one fused fp16 multiply-add:
//
//     sum = bytesum(w * 0x83DCD12D) + 0x6400
//     v   = fp16(sum_bits) * fp16(0x1EEE) + fp16(0xC931)      (one rounding)
//
// The oracle proves this equals an exact 1021-entry codebook built with exact
// rational arithmetic and round-half-to-even, so the decode is exact, and the
// golden dumps agree bit for bit. `exl3_dp4a_mul1` below computes the same sum
// either with the explicit byte sum the format document specifies, or with the
// `vabsdiff4` instruction upstream used before it switched to dp4a -- which is
// NATIVE on Pascal (the compile proves it; sm_86+ is where it is emulated).
// The two are bit-identical by construction (both are byte0+byte1+byte2+byte3,
// which cannot overflow: 4*255 + 0x6400 < 2^32) and both are tested against the
// golden dump. EXL3_VABSDIFF4 picks one; see the benchmark in _build/.
//
// ----------------------------------------------------------------------------
// THE PERMUTATION, AND WHY EACH LANE NEEDS ONLY TWO ACCUMULATORS
// ----------------------------------------------------------------------------
// Window p = 8*L + j of a tile (L = lane 0..31, j = 0..7) is the weight at
//
//     A = L % 8;  sL = L if A < 4 else L - 4
//     row = (sL % 4) * 2 + (0, 1, 8, 9)[j % 4]        row = input index
//     col = 2 * (sL // 8) + 8 * (j // 4) + (0 if A < 4 else 1)
//
// Since sL % 4 == L % 4 and sL // 8 == L // 8 for both halves of A, a lane's
// eight windows land in exactly TWO output columns -- `c = 2*(L//8) + (A < 4 ? 0
// : 1)` and `c + 8` -- at the four rows `r0 = (L%4)*2, r0+1, r0+8, r0+9`. That
// is why the accumulator is two floats per token and not eight: the decode's
// half2 pairs line up with (row, row+1) and (row+8, row+9) exactly.
//
// The four lanes `8*c_b .. 8*c_b+3` all own column 2*c_b (at rows 0..15 of the
// tile between them), so the tile's 16 output columns are the four-lane sums of
// those accumulators, reduced by a `__shfl_xor` pair and written by the one lane
// of each group of four whose A is 0 (columns c, c+8) or 4 (columns c+1, c+9).
//
// ----------------------------------------------------------------------------
// LAUNCH GEOMETRY (all of it is an argument; nothing is #defined)
// ----------------------------------------------------------------------------
//   exl3_gemv    grid (ceil(out/128), ceil(T/m)), block 256 (8 warps)
//                warp w of the block owns output tile blockIdx.x*8 + w, so a
//                block covers 128 output columns and the 8 tiles it reads per
//                k-step are contiguous (reconstruct.cu's blockIdx.x*8 shape).
//                `m` (1, 2, 4 or 8) is the number of tokens the block carries:
//                the decode of a k-tile is done ONCE and used m times, which is
//                what makes prefill not re-read the trellis T times.
//   had128_pre   grid (ceil((n/128)/8), rows), block 256; y = H128 (x * suh)
//   had128_post  grid (ceil((n/128)/8), rows), block 256; y = (H128 z) * svh
//   gemv_f16_f32 grid (ceil(out/8), T), block 256 (one warp per output row)
//                y fp32 = W fp16 (out,in) . x fp16 -- for in_proj_a/in_proj_b,
//                whose consumer gdn_scalars takes fp32.
//
// in and out must both be multiples of 16 (the tile geometry) and in/out must be
// multiples of 128 for the Hadamards (the reference checks divisibility:
// exllamav3 `TORCH_CHECK_DIV(input, 1, 128)`). A kernel whose geometry it cannot
// serve returns WITHOUT WRITING anything, so the runtime owns `_check_geometry`.
//
// ----------------------------------------------------------------------------
// PRECISION
// ----------------------------------------------------------------------------
// x, A and the outputs are fp16; the Hadamard interior is fp32 (DESIGN.md's
// table: "Hadamard | fp32 | matches the plugin's verified reconstruction path");
// the GEMV accumulates in fp32 and rounds once when it stores z. suh/svh are
// fp16 continuous scales (this checkpoint has no packed +/-1 form), multiplied
// in fp32 inside the Hadamard kernels.
//
// ----------------------------------------------------------------------------
// VERIFIED, ON THIS DEVICE, AND HOW (tests/test_exl3.py; measured numbers are
// printed by each test and repeated in the report next to this file)
// ----------------------------------------------------------------------------
//   * the A decode is BIT-EXACT against the golden dump -- all six golden
//     modules, five bitrates (2, 3, 3.5, 4, 6), both the first and the last
//     k-tile band, every column.
//   * the Hadamard butterflies are the natural-order Sylvester matrix
//     H[i][j] = (-1)^popcount(i&j)/sqrt(128), checked against a
//     directly-constructed matrix in numpy (_build/check_hadamard.py) and
//     against the kernels' own output at the model's shapes.
//   * the fused chain is checked against a float64 reference built from the
//     golden A for whole modules at the checkpoint's own suh/svh.
//   * the m = 1, 2, 4, 8 token-group paths agree bit for bit.
//   * the mutation pass (_build/mutate_exl3.py) breaks each of 25 things --
//     the +256*bits wrap (three ways), the aligned-variant window pairing, the
//     bits dispatch (four ways), the suh/svh sides, the Hadamard ordering (four
//     ways), the (0,1,8,9) row map, the A>=4 column offset, three codebook
//     constants, the k-tile stride, the four-lane reduction, the half-rate's bit
//     alternation and the token-group origin -- and requires the named test to
//     FAIL for each. 25/25 are caught; the one entry that survived the first run
//     was the lane-stage ORDER, which is not a bug at all (the stages act on
//     different index bits and commute, so no test could see it) and was
//     replaced by an element-order permutation, which is observable.
//   * the benchmark (_build/bench_exl3.py) attributes the time: on gate_proj at
//     T = 1 this kernel reads 39.0 MB of trellis in ~335 us = ~116 GB/s, the
//     same kernel with every load turned into an L1 hit runs ~69 us = ~567 GB/s
//     (a pure uint4 streaming read of the same buffer measures 570-574 GB/s),
//     so it is neither bandwidth- nor
//     ALU-bound: it is memory-LATENCY-bound on a 122 KB-strided walk, and the
//     arithmetic is essentially free. See the measured note above exl3_gemv_run
//     for the shared-memory staging variant that was tried and rejected (788 us)
//     and for the WT/split-K entries that are the fix (335 -> 240 us there).
//   * the wide path (exl3_gemv_w2/w4/w8 + exl3_sk_reduce) is held to the same
//     standard as the one-tile kernel (tests/test_exl3.py::TestWideGemv): WT
//     only changes how many tiles a warp owns, so its output is BIT-IDENTICAL
//     to exl3_gemv (18 module x WT pairs), a split is exact through the golden
//     one-hot decode (6 modules x S=4,8, first and last band) and within one
//     fp16 rounding otherwise (worst 4.4e-4 of the reference's scale).
//   NOT verified here: prefill-scale throughput (m > 1 is correct, not tuned;
//   the wide path is not used above T = 4), and the fused GEMV has not been
//   compared against a *served* logits stream -- that is the model-level test,
//   downstream of this file (PPL 5.6515 one tile, 5.6502 wide, checkpoint
//   5.6482).
// ============================================================================

#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// implementation bounds (guarded at every use, never geometry)
// ---------------------------------------------------------------------------
#define EXL3_THREADS 256        // threads per exl3_gemv block (8 warps)
#define EXL3_WARPS 8            // one output tile per warp
#define EXL3_M_MAX 8            // token groups the general entries instantiate
// The prefill entry's largest token group, and the measured crossover that
// picked it (_build/bench_prefill_m.py, T = 512, paired in one process, every
// module of the checkpoint; ratio against the m = 8 the runtime shipped):
//
//   m      gate_proj   down_proj   in_proj_qkv   k_proj
//   8        1.000       1.000        1.000       1.000
//   16       1.229       1.121        1.155       1.279
//   32       0.449       0.401        0.404       0.268
//
// So m = 16 is the optimum of this family and 32 is a LOSS, at every module:
// the accumulator array is 2*m floats per lane and 2*32 of them do not fit in
// the 128-register budget the walk's occupancy needs, so ptxas spills the loop
// (measured 2.7x worse than its own model, 1.9x worse than m = 16). m = 64 is
// worse still and needs OCC = 1 -- both needs an explicit -D to build, and the
// #error below refuses the occupancy combination that spills outright.
#ifndef EXL3_M_PRE_MAX
#define EXL3_M_PRE_MAX 32       // ... and the prefill entry (see exl3_gemv_pre)
#endif

// The prefill entry's token-loop unroll factor; 0 means "unroll fully", which
// is what every entry in this file does and what m = 16 needs: 2*m accumulators
// indexed by a RUNTIME loop variable go to local memory, and the partial unroll
// measured 6.6x slower for exactly that reason (212 ms against 32.3 on
// gate_proj at T = 512, m = 16 -- _build/bench_prefill_m.py). The partial form
// stays selectable for whoever wants to re-measure it.
#ifndef EXL3_TOKEN_UNROLL
#define EXL3_TOKEN_UNROLL 0
#endif

// The prefill entry's SHARED-MEMORY ACTIVATION STAGE, sized in BYTES per double
// buffer rather than in k-tiles, because the window that fits depends on the
// token group: the buffer is 128*SMW*M bytes and the budget that keeps
// EXL3_GEMV_BIG_OCC = 2 blocks resident is the SM's 64 KB, i.e. 32 KB a block.
// The card reports 0 bytes of reserved shared memory per block (checked with
// cuOccupancyMaxActiveBlocksPerMultiprocessor: exl3_gemv_pre returns 2 blocks/SM
// at 32 KB), so 32 KB is exactly affordable -- but only just, which is why this
// is a byte budget and not a tile count.
//
// `-DEXL3_SM_BYTES=0` reproduces the shipped loop exactly: the revert switch and
// the control arm of the A/B are the same thing.
#ifndef EXL3_SM_BYTES
#define EXL3_SM_BYTES 32768
#endif
// The window, in k-tiles, for a group of M tokens. M = 16 -> 16 k-tiles,
// M = 32 -> 8; both 32 KB. Only m = 16 and m = 32 stage (see the dispatch).
#define EXL3_STAGE_SMW(M) (EXL3_SM_BYTES / (128 * (M)))

#define HAD_THREADS 256         // had128_* block (8 warps, one 128-block each)
#define HAD_WARPS 8
#define EXL3_F16_MAX 65504.f    // fp16's finite range (kernels.cu's f16_sat)

// The mul1 (cb 2) codebook constants, EXL3_FORMAT.md section 4 / the oracle's
// MUL1 / CB_ADD / K_INV_BITS / K_BIAS_BITS.
#define EXL3_MUL1 0x83DCD12Du
#define EXL3_CB_ADD 0x6400u
#define EXL3_K_INV_BITS 0x1EEEu
#define EXL3_K_BIAS_BITS 0xC931u

// 1/sqrt(128): the Hadamard normalisation. exllamav3's had_r_128 spells it
// 0.088388347648; the double-precision value below is the same number rounded
// once (0.08838834764831845f == 0x3DB504F3), which is what the plugin's
// `_had` produces by dividing by BLOCK**0.5 in fp32.
#define EXL3_RSCALE 0.08838834764831845f

// The byte-sum vs vabsdiff4 choice for the codebook's __dp4a substitute.
// Both are bit-identical; see the header and the measured comparison.
#ifndef EXL3_VABSDIFF4
#define EXL3_VABSDIFF4 1
#endif

// How a lane's words are READ (never which words, and never what is done with
// them -- the decode cores below are byte-for-byte the same either way).
//
//   1 (default): one 16-byte load per aligned block, at most two blocks per
//                lane per tile. This is the round that took the GEMV off its
//                4-byte-request floor: the same traversal measures 177-201 GB/s
//                with 16-byte requests against 35-77 GB/s with 4-byte ones at
//                equal occupancy (_build/probe_mlp.py), and the kernel was at
//                137 GB/s.
//   0:           the four 4-byte __ldg calls the earlier rounds shipped, kept
//                compilable (`-DEXL3_WIDE_LOADS=0`) because the paired
//                comparison in _build/bench_loads.py needs both in one source.
//
// The block a word lives in is `word & ~3`: the tile holds W = 8*bits words,
// which is a multiple of four for all five bitrates, and every word index the
// offsets produce is already reduced mod W, so the block never leaves the tile
// (and `(W-4)..(W-1)` is in bounds). The tile as a whole is 16-byte aligned
// because each tile is W words = 4W bytes with W a multiple of 4 and the host
// refuses a trellis whose base pointer is not 16-byte aligned.
#ifndef EXL3_WIDE_LOADS
#define EXL3_WIDE_LOADS 0
#endif

// Ask for the next k-step's trellis lines while the current one is decoded.
// A pure latency play -- `prefetch.global.L2` produces no register result and
// cannot change any value. MEASURED AND OFF: at WT = 4 with the split the walk's
// round trip is already covered by the 16 resident warps' own requests, and an
// extra prefetch per tile per warp measures inside the run-to-run spread on
// every module of _build/bench_loads.py (gate_proj 244.2 vs 244.3 us at S = 8,
// q_proj 193.7 vs 192.0, in_proj_qkv 151.5 vs 147.5 -- two of them slower).
#ifndef EXL3_PREFETCH
#define EXL3_PREFETCH 0
#endif

// The k-walk's software pipeline, in k-steps: EXL3_PIPE k-steps' words are in
// registers at once, so stage d's DECODE runs while stage d + PIPE's loads are
// still in flight. This is the lever the two previous rounds identified and did
// not land (a depth-4 register double-buffer measured +12% at WT = 1 on the
// older shape, at 158 registers -- _build/kernels_exl3_pipe4_handoff.cu.txt).
//
// MEASURED AND OFF, at every depth and both occupancies. The reason it was
// expected to work was half right: the walk's round trip and the decode's issue
// do ADD at WT = 4 (walk probe, the shipped geometry: the decoder's own
// 4-loads-a-lane-a-tile walk is 178 us for gate_proj, the same arithmetic with
// every load an L1 hit is 69 us, the kernel is 250 -- the sum, not the max).
// But overlapping them in registers costs more than the overlap wins:
//
//   gate_proj (T = 1, WT = 4, S = 8, paired in one process, _build/bench_gemv_ab.py)
//     OCC = 2:  PIPE = 1  250.6 us | PIPE = 2  346.2 (1.39x WORSE)
//     OCC = 1:  PIPE = 1  250.5 us | PIPE = 2  283.8 (1.13x worse)
//   lm_head:    PIPE = 2 is 1.31x worse at OCC = 2 and 1.05x worse at OCC = 1.
//
// The register budget is what does it. The wide entry is pinned at 128
// registers (2 blocks of 256 threads), and its M = 1..8 dispatch means the
// M = 8 path's 64 accumulators are in the same allocation; a pipeline stage of
// WT = 4 tiles is 16 more registers, so PIPE = 2 lands on the cap with a spill
// and PIPE = 3/4 spill 40/284 bytes in the hot loop. The narrower decode entry
// below (M = 1/2 only, 101 registers unpipelined) does not rescue it: at
// PIPE = 1 it is NEUTRAL (1.004x), so the registers are not the whole story
// either -- an unpipelined loop with slack is simply not improved by holding
// k-ahead words, which says the k-step is limited by its REQUEST STREAM (see
// the walk probe: 4 requests a warp-k-step 82 us, 8 or 16 of them 178 us) and
// not by how many of them a warp has in flight.
//
// EXL3_PIPE = 1 is the unpipelined loop (the shipped kernel, bit for bit);
// values >= 2 are the pipeline depth. It is a -D flag with a default so both
// can be built from this one source and timed PAIRED in one process
// (_build/bench_gemv_ab.py); the entry it drives is exl3_gemv_p4, which no
// runtime path calls (tests/test_exl3.py::TestWideGemv holds it bit-identical
// to exl3_gemv_w4 so the flag cannot rot).
#ifndef EXL3_PIPE
#define EXL3_PIPE 1
#endif

// ---------------------------------------------------------------------------
// THE TWO-TOKEN X HOIST: m = 2 on the wide decode entry
// ---------------------------------------------------------------------------
// The shipped k-loop interleaves, per 16-column tile, the two x loads of token
// t with token t's 8 FMAs, and then does the same again for token t+1:
//
//     for i in 0..WT-1                       // WT = 4
//         load 4 trellis words for tile i
//         decode tile i
//         for t in 0..M-1                     // M = 2 at a verify
//             LDG x[t]        <- two 4-byte loads
//             8 FFMA into acc[..][..][t]
//
// so the second token's two loads are issued only after the first token's 8*WT
// FMAs, and a warp at m = 2 has 4 + 32 = 36 memory instructions in flight to
// cover them where it has 4 + 16 = 20 at m = 1. `EXL3_XHOIST_M` names the token
// group size this applies to; every other M keeps the shipped loop, byte for
// byte, and the guard is a folded template constant so the m = 1/4/8
// instantiations compile to exactly the code above.
//
// MEASURED, two cubins in one context, alternated, 11 real modules at the
// runtime's own geometry (S pinned to the T = 1 value, as the verify takes it),
// 1328 MHz / 55 C. exl3_gemv_w4, m = 2:
//
//                     T = 1     T = 2     T2/T1
//     shipped        4426.7    6380.3    1.4400
//     hoisted        4426.7    5176.4    1.1693      T2 x 0.8113
//     hoisted, x loads after the trellis loads   T2 x 0.8630
//     control: the same change at m = 1 only     T2 x 1.0000
//
// lm_head 0.773, down_proj 0.891, in_proj_qkv 0.878, q_proj 0.848, o_proj 0.894,
// gate_proj 0.868, in_proj_z 0.912, k_proj 0.921, and up_proj 1.043 -- the one
// module it LOSES on, at 3 bits, and it loses it at every placement tried.
//
// IT IS BIT-EXACT BY CONSTRUCTION and measured so: the same addresses, the
// same values, the same FMA order, the same accumulator registers. Only WHEN a
// load is issued changes. 11 modules x m in {1, 2, 4, 8}, output halves
// compared one by one: 0 differing out of 4 894 080.
//
// WHAT THIS IS NOT. It is not the FMA count: dropping token 1's entire FMA
// chain (8 FMAs x WT a k-step) measures T2 x 1.002 -- free. Nor is it the
// number of x loads: making token 1 share token 0's x (2 of 4 loads gone)
// measures T2 x 0.971, while REMOVING all four measures 0.879 and HOISTING
// all four measures 0.857. It is where in the k-step the loads are issued.
#ifndef EXL3_XHOIST_M
#define EXL3_XHOIST_M 2      // the m whose x is fetched up front
#endif
#ifndef EXL3_XFIRST
#define EXL3_XFIRST 1        // 1: before the WT trellis loads, 0: after them
#endif
#ifndef EXL3_XSPLIT
#define EXL3_XSPLIT 0        // >0: only the first EXL3_XSPLIT tokens go first
#endif

// Gather a tile's words with ONE warp-wide load per lane per tile plus four
// shuffles instead of four per-lane loads. The walk probe says this is the
// right shape for the WALK -- one load a lane a tile walks gate_proj's 39 MB in
// 82 us against 178 us for the decoder's four (2.17x, the same bytes and the
// same addresses) -- and attaching the decode to it says the opposite about the
// kernel:
//
//   gate_proj 378.6 us against the shipped 250.6 (1.51x WORSE)
//   down_proj 414.0 against 249.3 (1.66x), in_proj_qkv 244.7 against 150.7
//   (1.62x), lm_head 3569.6 against 3064.0 (1.17x, and that one does not even
//   take the shuffle path -- W = 48 > 32 falls through to the loads)
//
// so the walk's 82 us is not a rate the decode can be attached to. With one
// load per tile the whole tile's decode waits on that single load and then on
// four `shfl` round trips (the MIO pipe the loads themselves use), and the
// per-tile dependency chain that costs is bigger than the request slots the
// four independent loads were taking. Measured, three modules, consistent.
#ifndef EXL3_SHFL_LOADS
#define EXL3_SHFL_LOADS 0
#endif

// The sampler's candidate extraction (exl3_cand_hist / exl3_cand_collect) runs
// over one 248 077-element row, so its grid is small and fixed: the histogram's
// global-merge traffic is CAND_BLOCKS * CAND_BINS atomics.
#ifndef EXL3_CAND_BLOCKS
#define EXL3_CAND_BLOCKS 32
#endif

// The argmax pair's grid, one block per this many elements of the row. Same
// shape as kernels.cu's AMAX_BLOCKS (64 blocks of 256 threads over 248 077
// elements = 3877 each) -- keep the two in step; the runtime sizes the partial
// arrays for this.
#ifndef EXL3_AMAX_BLOCKS
#define EXL3_AMAX_BLOCKS 64
#endif

// The five decoders this checkpoint needs, as dispatch tags. `bits_x2` = 2*bits
// selects one unambiguously: 4 -> 2-bit, 6 -> 3-bit, 7 -> 3.5-bit (the half
// rate), 8 -> 4-bit, 12 -> 6-bit. 5-bit, 7-bit, 8-bit and the 1-bit aligned
// variant exist upstream but no module here uses them, so they are absent (and
// a bits_x2 outside the five returns without writing).
enum {
    EXL3_V_ALIGN2 = 0,   // dq8_aligned_2bits
    EXL3_V_DQ8_3 = 1,    // dq8<3, cb, 4>
    EXL3_V_HALF = 2,     // dq8_half<3, cb>   (3.5 bits, KA = 3)
    EXL3_V_ALIGN4 = 3,   // dq8_aligned_4bits
    EXL3_V_DQ4_6 = 4     // dq4<6, cb> twice
};

// ============================================================================
// small helpers
// ============================================================================

// fp16 tops out at 65504. An activation that leaves the range becomes an
// infinity and one infinity destroys the rest of the forward, so a value out of
// range is pinned to the largest magnitude the storage holds. NaN keeps its
// meaning. Same helper, same reason, as kernels.cu's f16_sat (and the same
// semantics: the clamp only).
static __device__ __forceinline__ float exl3_satf(float v) {
    return v != v ? v : fminf(fmaxf(v, -EXL3_F16_MAX), EXL3_F16_MAX);
}

// One lane's four consecutive halves: cuda_fp16.h in CUDA 12.6 has no `half4`
// (exllamav3 compiles against one that does), and __ldg has no overload for a
// struct, so the 8-byte vector is loaded as a uint2 and bit-cast through a
// union -- the same idiom as exllamav3's own half2_uint32.
union exl3_u2h2 {
    uint2 u;
    half2 h2[2];
};

static __device__ __forceinline__ half exl3_sat(float v) {
    return __float2half(exl3_satf(v));
}

// `fshift` from exl3_dq.cuh: a funnel shift across two uint32 words. The
// shift is the FULL 64-bit shift, not __funnelshift_r: its amount can reach 32
// (for bits = 3 the lane L = 3, 7, ... have s2 == 32 exactly), and
// __funnelshift_r takes its amount modulo 32, which would return `b` instead of
// `a` there. The oracle shifts a Python int and masks, so this matches it.
static __device__ __forceinline__ uint32_t exl3_fshift(uint32_t b, uint32_t a,
                                                       int shift) {
    const uint64_t merged = ((uint64_t)a << 32) | (uint64_t)b;
    return (uint32_t)(merged >> shift);
}

// `__dp4a(x, 0x01010101, 0x6400)` for sm_60, which has no DP4A (EXL3_FORMAT.md
// section 4: nvcc REFUSES it on this target).
//
//   (a) vabsdiff4 (default): the instruction upstream used before the dp4a
//       rewrite, `vabsdiff4(x, 0, acc)` == byte-sum(x) + acc. Native on Pascal.
//   (b) the explicit byte sum: four masks, three shifts, three adds.
// Both give byte0+byte1+byte2+byte3+0x6400, which cannot overflow (0x67FC max),
// so they are bit-identical to each other and to the oracle's LUT.
static __device__ __forceinline__ uint32_t exl3_dp4a_mul1(uint32_t x) {
#if EXL3_VABSDIFF4
    uint32_t sum;
    const uint32_t zero = 0u;
    const uint32_t acc = EXL3_CB_ADD;
    asm ("vabsdiff4.u32.u32.u32.add %0, %1, %2, %3;"
         : "=r"(sum) : "r"(x), "r"(zero), "r"(acc));
    return sum;
#else
    return (x & 0xffu) + ((x >> 8) & 0xffu) + ((x >> 16) & 0xffu)
           + ((x >> 24) & 0xffu) + EXL3_CB_ADD;
#endif
}

// `decode_mul1_product_2`: two windows -> one half2, the low half from x0.
// One integer multiply per window, the byte sum, then ONE fused fp16
// multiply-add (the oracle's exact-rational LUT proves the rounding).
static __device__ __forceinline__ half2 exl3_decode_pair(uint32_t x0, uint32_t x1) {
    const uint32_t s0 = exl3_dp4a_mul1(x0 * EXL3_MUL1);
    const uint32_t s1 = exl3_dp4a_mul1(x1 * EXL3_MUL1);
    const half2 k_inv = __half2half2(__ushort_as_half((unsigned short)EXL3_K_INV_BITS));
    const half2 k_bias = __half2half2(__ushort_as_half((unsigned short)EXL3_K_BIAS_BITS));
    const half2 h = __halves2half2(__ushort_as_half((unsigned short)(s0 & 0xffffu)),
                                   __ushort_as_half((unsigned short)(s1 & 0xffffu)));
    return __hfma2(h, k_inv, k_bias);
}

// ============================================================================
// the 128-point Hadamard, natural order
// ============================================================================
//
// H[i][j] = (-1)^popcount(i & j) / sqrt(128) -- what the plugin's Sylvester
// recursion and exllamav3's hadamard_data both produce. The butterfly below
// computes it with one warp per 128-block: element index = 4*lane + u
// (u = 0..3), stages on index bits 0 and 1 in registers and bits 2..6 as
// __shfl_xor with the sign flipped on (lane & s) -- the same shape, and the
// same element order, as exllamav3's had_hf_r_128_inner + shuffle_had_f4x32.
//
// A WRONG ORDERING HERE IS SILENT (it is a Hadamard matrix either way, so the
// output looks plausible and is garbage). _build/check_hadamard.py settles it:
// it builds the matrix from this exact simulation and compares it, element for
// element, against a directly-constructed (-1)^popcount(i&j)/sqrt(128), and the
// test suite checks the kernel itself against the same numpy matrix.
//
// Note the two orderings that do NOT work and were rejected: element index
// = u*32 + lane (lane-major) gives H with both axes permuted, and dropping the
// register stage's index mapping gives a different matrix again.
static __device__ __forceinline__ void exl3_fwt128(float (&v)[4], int lane) {
    // index bits 0 and 1: inside a lane's four elements
#pragma unroll
    for (int p = 0; p < 2; p++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            if (!(j & (1 << p))) {
                const float a = v[j];
                const float b = v[j ^ (1 << p)];
                v[j] = a + b;
                v[j ^ (1 << p)] = a - b;
            }
        }
    }
    // index bits 2..6: across the warp
#pragma unroll
    for (int s = 1; s < 32; s <<= 1) {
#pragma unroll
        for (int u = 0; u < 4; u++) {
            const float o = __shfl_xor_sync(0xffffffffu, v[u], s);
            v[u] = (lane & s) ? (o - v[u]) : (v[u] + o);
        }
    }
}

// y = H128 (x * suh), blockwise, fp32 interior, one fp16 rounding at the end.
// `pre` = 1 multiplies by the scale before the transform (the input side),
// 0 after (the output side). The two sides are genuinely different functions
// and the fused path needs the right one on each end: x * diag(suh) * H is
// had128_pre, (H * z) * diag(svh) is had128_post.
static __device__ __forceinline__ void had128_body(const half* __restrict__ x,
                                                   const half* __restrict__ scale,
                                                   half* __restrict__ y,
                                                   int n, int rows, int pre) {
    const int lane = threadIdx.x & 31;
    const int w = threadIdx.x >> 5;
    if (n <= 0 || (n & 127) != 0) return;        // the reference requires 128 | n
    const int blk = blockIdx.x * HAD_WARPS + w;
    if (blk >= (n >> 7)) return;
    const int t = blockIdx.y;
    if (t >= rows) return;

    const size_t base = (size_t)t * n + (size_t)blk * 128;
    exl3_u2h2 xv;
    xv.u = __ldg(reinterpret_cast<const uint2*>(x + base) + lane);
    exl3_u2h2 sv;
    sv.u = __ldg(reinterpret_cast<const uint2*>(scale + (size_t)blk * 128) + lane);
    // element index inside the 128-block = 4*lane + u, u = 0..3 in order
    float v[4] = { __half2float(__low2half(xv.h2[0])),
                   __half2float(__high2half(xv.h2[0])),
                   __half2float(__low2half(xv.h2[1])),
                   __half2float(__high2half(xv.h2[1])) };
    const float s[4] = { __half2float(__low2half(sv.h2[0])),
                         __half2float(__high2half(sv.h2[0])),
                         __half2float(__low2half(sv.h2[1])),
                         __half2float(__high2half(sv.h2[1])) };
    if (pre) {
#pragma unroll
        for (int u = 0; u < 4; u++) v[u] *= s[u];
        exl3_fwt128(v, lane);
#pragma unroll
        for (int u = 0; u < 4; u++) v[u] *= EXL3_RSCALE;
    } else {
        exl3_fwt128(v, lane);
#pragma unroll
        for (int u = 0; u < 4; u++) v[u] *= EXL3_RSCALE * s[u];
    }
    exl3_u2h2 out;
    out.h2[0] = __floats2half2_rn(exl3_satf(v[0]), exl3_satf(v[1]));
    out.h2[1] = __floats2half2_rn(exl3_satf(v[2]), exl3_satf(v[3]));
    *reinterpret_cast<uint2*>(y + base + 4 * (size_t)lane) = out.u;
}

// The same pre-rotation, for the PREFILL GEMV's single-load x: the 16 halves of
// every k-tile are reordered so that the four values ONE LANE needs
// (`_build/DESIGN.md`'s row map: rows r0, r0+1, r0+8, r0+9 with r0 = 2*(lane%4))
// are adjacent, and stored as fp32 so the GEMV needs no conversion.
//
// Why: the GEMV's per-token cost is what is left once m = 16 has amortised the
// walk, and measuring the x path out of it (a probe that replaces the two loads
// and their conversions with a constant, everything else identical) moved
// gate_proj at T = 512, m = 16 from 32.3 ms to 19.9 ms -- 38% of the kernel. Per
// token per k-step a lane does two 4-byte loads, four fp16->fp32 conversions and
// a 64-bit address computation to fetch four values that are 16 bytes apart;
// here they are one aligned 16-byte load and nothing else.
//
// The values are the SAME halves: the butterfly, the scale, the fp16 rounding
// and the saturation are had128_body's, and the reorder happens after the value
// is final -- `_round_to_half_then_fp32` is exact, so a GEMV fed from this
// buffer is bit-identical to one fed from the fp16 buffer. The permutation is
// within each 16-half k-tile: position 4g+j of the group holds element
// (2g, 2g+1, 2g+8, 2g+9)[j], which is a 2-shuffle fixup of the four lanes that
// hold the group (lane l holds elements 4l..4l+3 of the block).
static __device__ __forceinline__ void had128_body_lc(const half* __restrict__ x,
                                                      const half* __restrict__ scale,
                                                      float* __restrict__ y,
                                                      int n, int rows) {
    const int lane = threadIdx.x & 31;
    const int w = threadIdx.x >> 5;
    if (n <= 0 || (n & 127) != 0) return;        // the reference requires 128 | n
    const int blk = blockIdx.x * HAD_WARPS + w;
    if (blk >= (n >> 7)) return;
    const int t = blockIdx.y;
    if (t >= rows) return;

    const size_t base = (size_t)t * n + (size_t)blk * 128;
    exl3_u2h2 xv;
    xv.u = __ldg(reinterpret_cast<const uint2*>(x + base) + lane);
    exl3_u2h2 sv;
    sv.u = __ldg(reinterpret_cast<const uint2*>(scale + (size_t)blk * 128) + lane);
    float v[4] = { __half2float(__low2half(xv.h2[0])),
                   __half2float(__high2half(xv.h2[0])),
                   __half2float(__low2half(xv.h2[1])),
                   __half2float(__high2half(xv.h2[1])) };
    const float s[4] = { __half2float(__low2half(sv.h2[0])),
                         __half2float(__high2half(sv.h2[0])),
                         __half2float(__low2half(sv.h2[1])),
                         __half2float(__high2half(sv.h2[1])) };
#pragma unroll
    for (int u = 0; u < 4; u++) v[u] *= s[u];
    exl3_fwt128(v, lane);
#pragma unroll
    for (int u = 0; u < 4; u++) v[u] *= EXL3_RSCALE;

    // The fp16 rounding had128_body would have done, then the reorder among the
    // four lanes of this 16-half group, then the widening (exact).
    half2 h2[2];
    h2[0] = __floats2half2_rn(exl3_satf(v[0]), exl3_satf(v[1]));
    h2[1] = __floats2half2_rn(exl3_satf(v[2]), exl3_satf(v[3]));
    const int r = lane & 3;
    const int g = lane & ~3;                     // this group's first lane
    // Both halves come from each source lane and the TARGET selects: a shuffle
    // reads the SOURCE lane's copy of its operand, so selecting the operand with
    // the target's own lane bits (`h2[r & 1]`) hands back the source's choice
    // rather than the one this lane needs. That is exactly the silent-value
    // failure this kernel's test caught.
    const int sa = g + (r < 2 ? 0 : 1);
    const int sb = g + (r < 2 ? 2 : 3);
    const half2 a0 = __shfl_sync(0xffffffffu, h2[0], sa);
    const half2 a1 = __shfl_sync(0xffffffffu, h2[1], sa);
    const half2 b0 = __shfl_sync(0xffffffffu, h2[0], sb);
    const half2 b1 = __shfl_sync(0xffffffffu, h2[1], sb);
    const half2 a = (r & 1) ? a1 : a0;           // elements 2g, 2g+1
    const half2 b = (r & 1) ? b1 : b0;           // elements 2g+8, 2g+9
    float4 out;
    out.x = __low2float(a);
    out.y = __high2float(a);
    out.z = __low2float(b);
    out.w = __high2float(b);
    *reinterpret_cast<float4*>(y + base + 4 * (size_t)lane) = out;
}

// Step 1/2 of the fused form: xh = had128(x * suh), blockwise on the INPUT axis.
// grid (ceil((n/128)/8), rows), block 256. n = the input width.
extern "C" __global__ void had128_pre(const half* __restrict__ x,
                                      const half* __restrict__ suh,
                                      half* __restrict__ y, int n, int rows) {
    had128_body(x, suh, y, n, rows, 1);
}

// had128_pre over silu(g) * u: the MLP's down_proj input, rounded to fp16
// exactly as kernels.cu's silu_mul stores it (silu as __fdiv_rn(x, 1 + e^-x),
// the product's one rounding), then rotated as had128_pre -- the prefill path's
// silu_mul launch and its [T, 17408] round trip go. Bit-identical.
// grid/block as had128_pre.
extern "C" __global__ void had128_pre_silu(const half* __restrict__ g,
                                           const half* __restrict__ u,
                                           const half* __restrict__ suh,
                                           half* __restrict__ y, int n, int rows) {
    const int lane = threadIdx.x & 31;
    const int w = threadIdx.x >> 5;
    if (n <= 0 || (n & 127) != 0) return;
    const int blk = blockIdx.x * HAD_WARPS + w;
    if (blk >= (n >> 7)) return;
    const int t = blockIdx.y;
    if (t >= rows) return;
    const size_t base = (size_t)t * n + (size_t)blk * 128;
    exl3_u2h2 gv, uv, xv;
    gv.u = __ldg(reinterpret_cast<const uint2*>(g + base) + lane);
    uv.u = __ldg(reinterpret_cast<const uint2*>(u + base) + lane);
#pragma unroll
    for (int q = 0; q < 2; q++) {
        const float2 a = __half22float2(gv.h2[q]), b = __half22float2(uv.h2[q]);
        xv.h2[q] = __floats2half2_rn(__fdiv_rn(a.x, 1.f + __expf(-a.x)) * b.x,
                                     __fdiv_rn(a.y, 1.f + __expf(-a.y)) * b.y);
    }
    exl3_u2h2 sv;
    sv.u = __ldg(reinterpret_cast<const uint2*>(suh + (size_t)blk * 128) + lane);
    float v[4] = { __half2float(__low2half(xv.h2[0])), __half2float(__high2half(xv.h2[0])),
                   __half2float(__low2half(xv.h2[1])), __half2float(__high2half(xv.h2[1])) };
    const float s4[4] = { __half2float(__low2half(sv.h2[0])), __half2float(__high2half(sv.h2[0])),
                          __half2float(__low2half(sv.h2[1])), __half2float(__high2half(sv.h2[1])) };
#pragma unroll
    for (int q = 0; q < 4; q++) v[q] *= s4[q];
    exl3_fwt128(v, lane);
#pragma unroll
    for (int q = 0; q < 4; q++) v[q] *= EXL3_RSCALE;
    exl3_u2h2 out;
    out.h2[0] = __floats2half2_rn(exl3_satf(v[0]), exl3_satf(v[1]));
    out.h2[1] = __floats2half2_rn(exl3_satf(v[2]), exl3_satf(v[3]));
    *reinterpret_cast<uint2*>(y + base + 4 * (size_t)lane) = out.u;
}

// The pre-rotation in the prefill GEMV's single-load form: same values, same
// order of operations, fp32 out with each k-tile's 16 halves reordered so a
// lane's four values are one 16-byte load (see had128_body_lc).
extern "C" __global__ void had128_pre_lc(const half* __restrict__ x,
                                         const half* __restrict__ suh,
                                         float* __restrict__ y, int n, int rows) {
    had128_body_lc(x, suh, y, n, rows);
}

// exl3_sk_reduce and had128_post in one launch: the S fp32 partials of a split
// GEMV are summed in split order and rounded once (exactly what exl3_sk_reduce
// stores to z), and that fp16 value is what the post-rotation reads -- so the
// output is bit-identical to the two-launch path, without z's round trip or
// the second launch. part is [S, rows, n] fp32. grid/block as had128_post.
extern "C" __global__ void had128_post_sk(const float* __restrict__ part, int S,
                                          const half* __restrict__ svh,
                                          half* __restrict__ y, int n, int rows) {
    const int lane = threadIdx.x & 31;
    const int w = threadIdx.x >> 5;
    if (n <= 0 || (n & 127) != 0 || S <= 0) return;
    const int blk = blockIdx.x * HAD_WARPS + w;
    if (blk >= (n >> 7)) return;
    const int t = blockIdx.y;
    if (t >= rows) return;
    const size_t base = (size_t)t * n + (size_t)blk * 128;
    const size_t plane = (size_t)rows * n;
    const float4* pp = reinterpret_cast<const float4*>(part + base) + lane;
    const size_t p4 = plane >> 2;
    float4 a = __ldg(pp);
    int j = 1;
    for (; j + 4 <= S; j += 4) {             // four loads in flight, adds in order
        float4 b[4];
#pragma unroll
        for (int u = 0; u < 4; u++) b[u] = __ldg(pp + (size_t)(j + u) * p4);
#pragma unroll
        for (int u = 0; u < 4; u++) { a.x += b[u].x; a.y += b[u].y; a.z += b[u].z; a.w += b[u].w; }
    }
    for (; j < S; j++) {
        const float4 b = __ldg(pp + (size_t)j * p4);
        a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
    }
    float v[4] = { __half2float(exl3_sat(a.x)), __half2float(exl3_sat(a.y)),
                   __half2float(exl3_sat(a.z)), __half2float(exl3_sat(a.w)) };
    exl3_u2h2 sv;
    sv.u = __ldg(reinterpret_cast<const uint2*>(svh + (size_t)blk * 128) + lane);
    const float sc[4] = { __half2float(__low2half(sv.h2[0])), __half2float(__high2half(sv.h2[0])),
                          __half2float(__low2half(sv.h2[1])), __half2float(__high2half(sv.h2[1])) };
    exl3_fwt128(v, lane);
#pragma unroll
    for (int u = 0; u < 4; u++) v[u] *= EXL3_RSCALE * sc[u];
    exl3_u2h2 out;
    out.h2[0] = __floats2half2_rn(exl3_satf(v[0]), exl3_satf(v[1]));
    out.h2[1] = __floats2half2_rn(exl3_satf(v[2]), exl3_satf(v[3]));
    *reinterpret_cast<uint2*>(y + base + 4 * (size_t)lane) = out.u;
}

// Step 5: y = had128(z) * svh, blockwise on the OUTPUT axis (H then the scale).
// grid (ceil((n/128)/8), rows), block 256. n = the output width.
extern "C" __global__ void had128_post(const half* __restrict__ z,
                                       const half* __restrict__ svh,
                                       half* __restrict__ y, int n, int rows) {
    had128_body(z, svh, y, n, rows, 0);
}

// ============================================================================
// the trellis decode
// ============================================================================
//
// A lane's eight windows of a tile need at most two word pairs of the tile and
// two funnel-shift amounts. None of that depends on the k-tile, so it is drawn
// once per lane before the loop and reused for every tile the kernel touches.
struct Exl3Off {
    uint32_t i0[2];   // first word of each pair (already reduced mod the tile)
    uint32_t i2[2];   // second word of each pair
    int sh[2];        // the funnel-shift amount for that pair
#if EXL3_WIDE_LOADS
    // The same words again, as the wide load needs to see them: the two aligned
    // 16-byte blocks that between them own all four words (`wb`, ascending) and,
    // per word, which of the two it is in (bit 2 of `wsel`) and its slot inside
    // that block (bits 0..1). See exl3_offsets for why two blocks are always
    // enough. `wsel` is indexed by a compile-time constant at every use, so it
    // stays in registers.
    uint32_t wb[2];   // the two block indices
    uint32_t wsel[4]; // per needed word: (block << 2) | slot
#endif
};

// The word indices and shifts, per variant, exactly as exl3_dq.cuh computes
// them (and the oracle transcribes). `t` is the lane's window base, 8*lane.
static __device__ __forceinline__ Exl3Off exl3_offsets(int t, int vari, int W,
                                                       int bits) {
    Exl3Off o;
    const int bits2 = 2 * bits + 1;              // the half rate's 2-position period
    switch (vari) {
    case EXL3_V_DQ8_3: {
        // dq8<3, cb, 4>: b1 = (t + 257) * bits is the END of window 0, b2 the end
        // of window 7. One word pair, two funnel shifts 4*bits apart.
        const int b1 = (t + 257) * bits;
        const int b0 = b1 - 16, b2 = b1 + bits * 7;
        const int i0 = b0 / 32, i2 = (b2 - 1) / 32;
        const int s2 = (i2 + 1) * 32 - b2;
        o.i0[0] = o.i0[1] = (uint32_t)(i0 % W);
        o.i2[0] = o.i2[1] = (uint32_t)(i2 % W);
        o.sh[0] = s2;                    // windows 7..4
        o.sh[1] = s2 + bits * 4;         // windows 3..0
        break;
    }
    case EXL3_V_ALIGN4: {
        // dq8_aligned_4bits: stride 8 windows per word, previous word one back.
        const int i1 = t >> 3;
        const int i0 = (i1 + 31) & 31;
        o.i0[0] = o.i0[1] = (uint32_t)i0;
        o.i2[0] = o.i2[1] = (uint32_t)i1;
        o.sh[0] = 20;                    // the low group's funnel shift
        o.sh[1] = 0;                     // the high group reads `b` directly
        break;
    }
    case EXL3_V_ALIGN2: {
        // dq8_aligned_2bits: stride 16 windows per word, shift by (~t & 8) << 1.
        const int i1 = t >> 4;
        const int i0 = (i1 + 15) & 15;
        o.i0[0] = o.i0[1] = (uint32_t)i0;
        o.i2[0] = o.i2[1] = (uint32_t)i1;
        o.sh[0] = ((~t) & 8) << 1;
        o.sh[1] = 0;
        break;
    }
    case EXL3_V_HALF: {
        // dq8_half<KA, cb>: two groups of four windows, 18 + 3*KA bits each, at
        // their own word pairs. e7 ends window 7 (+ one tile for the wrap), e3
        // ends window 3, 2*bits2 earlier.
        const int gspan = 18 + 3 * bits;
        const int e7 = ((t >> 1) + 4) * bits2 + 128 * bits2;
        const int e3 = e7 - 2 * bits2;
        const int hi7 = (e7 - 1) / 32, lo7 = (e7 - gspan) / 32;
        const int hi3 = (e3 - 1) / 32, lo3 = (e3 - gspan) / 32;
        o.i0[0] = (uint32_t)(lo7 % W);
        o.i2[0] = (uint32_t)(hi7 % W);
        o.sh[0] = (hi7 + 1) * 32 - e7;
        o.i0[1] = (uint32_t)(lo3 % W);
        o.i2[1] = (uint32_t)(hi3 % W);
        o.sh[1] = (hi3 + 1) * 32 - e3;
        break;
    }
    default: {
        // dq4<bits, cb> twice: four windows per call, at t and t + 4.
        const int b0 = (t + 257) * bits - 16;
        const int b1 = b0 + 3 * bits;
        const int b2 = b1 + 16;
        const int i0 = b0 / 32, i2 = (b2 - 1) / 32;
        const int s2 = (i2 + 1) * 32 - b2;
        const int c0 = (t + 4 + 257) * bits - 16;
        const int c1 = c0 + 3 * bits;
        const int c2 = c1 + 16;
        const int j0 = c0 / 32, j2 = (c2 - 1) / 32;
        const int u2 = (j2 + 1) * 32 - c2;
        o.i0[0] = (uint32_t)(i0 % W);
        o.i2[0] = (uint32_t)(i2 % W);
        o.sh[0] = s2;
        o.i0[1] = (uint32_t)(j0 % W);
        o.i2[1] = (uint32_t)(j2 % W);
        o.sh[1] = u2;
        break;
    }
    }
#if EXL3_WIDE_LOADS
    // The wide load's bookkeeping, and the one place the five variants' word
    // geometry has to be believed.
    //
    // A lane needs the words at `i0[0]`, `i2[0]`, `i0[1]`, `i2[1]` -- for the
    // single-pair variants the last two are copies of the first two, by
    // construction. Two facts about that set make a 16-byte gather possible:
    //
    //   * It is contained in a run of at most THREE consecutive words of the
    //     tile. Two for the dq8 and dq8_aligned variants (their pair is 37, 30
    //     and 44 bits wide); three for the half rate (41 bits of span across its
    //     two 27-bit groups) and for dq4<6> (58 bits across its two 34-bit
    //     calls, 24 bits apart). A pair is either one word or two ADJACENT ones
    //     (the half rate's 26-bit group span is the case where both ends fall in
    //     the same word: it then reads that word twice, and the merged 64-bit
    //     funnel window is the tile's bit stream doubled, which is what the
    //     narrow path reads too).
    //   * The run WRAPS the tile for the lanes whose windows sit at the end of
    //     the bit stream: the field of a 3-bit tile's window 0 starts at bit 755
    //     of 768, so lane 0's pair is words 23 and 0. Hence the modulo in the
    //     offsets above, and hence `wb` is compared and not assumed ordered.
    //
    // Two aligned blocks own a run of three words whatever the alignment (the
    // block of its first word and the block of its last), and the wrap maps the
    // wrapped words into the tile's own first block, so the pair of blocks is
    // found by taking the min and max over the four words' blocks. The counts
    // were checked for all 32 lanes of all five variants (`_build/bench_loads.py`
    // re-derives them host side before it times anything).
    {
        const uint32_t idx[4] = { o.i0[0], o.i2[0], o.i0[1], o.i2[1] };
        uint32_t lo = idx[0] & ~3u, hi = lo;
#pragma unroll
        for (int k = 1; k < 4; k++) {
            const uint32_t b = idx[k] & ~3u;
            lo = b < lo ? b : lo;
            hi = b > hi ? b : hi;
        }
        o.wb[0] = lo;
        o.wb[1] = hi;
#pragma unroll
        for (int k = 0; k < 4; k++)
            o.wsel[k] = ((idx[k] & ~3u) == hi ? 4u : 0u) | (idx[k] & 3u);
    }
#endif
    return o;
}

// dq8<bits, cb, 4> (the 3-bit path): one word pair, the top four windows from a
// funnel shift and three shift-downs, the bottom four the same 4*bits later.
static __device__ __forceinline__ void exl3_dq8_3_core(uint32_t a, uint32_t b,
                                                       const Exl3Off& o, int bits,
                                                       half2 (&fr)[4]) {
    const uint32_t w7 = exl3_fshift(b, a, o.sh[0]);
    const uint32_t w6 = w7 >> bits, w5 = w6 >> bits, w4 = w5 >> bits;
    const uint32_t w3 = exl3_fshift(b, a, o.sh[1]);
    const uint32_t w2 = w3 >> bits, w1 = w2 >> bits, w0 = w1 >> bits;
    fr[0] = exl3_decode_pair(w0 & 0xffffu, w1 & 0xffffu);
    fr[1] = exl3_decode_pair(w2 & 0xffffu, w3 & 0xffffu);
    fr[2] = exl3_decode_pair(w4 & 0xffffu, w5 & 0xffffu);
    fr[3] = exl3_decode_pair(w6 & 0xffffu, w7 & 0xffffu);
}

// dq8_aligned_4bits: one word pair, the low three windows out of a 20-bit
// funnel shift and the high five straight out of the second word. The windows
// are PAIRED IN INDEX ORDER (w0,w1), (w2,w3), ... -- not in the order the
// shifts that produce them are written. That distinction is a real bug the
// oracle's history records, and the mutation pass re-introduces it on purpose.
static __device__ __forceinline__ void exl3_align4_core(uint32_t a, uint32_t b,
                                                        const Exl3Off& o,
                                                        half2 (&fr)[4]) {
    const uint32_t s = exl3_fshift(b, a, o.sh[0]);
    const uint32_t w0 = (s >> 8) & 0xffffu, w1 = (s >> 4) & 0xffffu, w2 = s & 0xffffu;
    const uint32_t w3 = (b >> 16) & 0xffffu, w4 = (b >> 12) & 0xffffu;
    const uint32_t w5 = (b >> 8) & 0xffffu, w6 = (b >> 4) & 0xffffu, w7 = b & 0xffffu;
    fr[0] = exl3_decode_pair(w0, w1);
    fr[1] = exl3_decode_pair(w2, w3);
    fr[2] = exl3_decode_pair(w4, w5);
    fr[3] = exl3_decode_pair(w6, w7);
}

// dq8_aligned_2bits: one funnel-shifted word, then w_k = word >> 2*(7-k).
static __device__ __forceinline__ void exl3_align2_core(uint32_t a, uint32_t b,
                                                        const Exl3Off& o,
                                                        half2 (&fr)[4]) {
    b = exl3_fshift(b, a, o.sh[0]);
    const uint32_t w0 = (b >> 14) & 0xffffu, w1 = (b >> 12) & 0xffffu;
    const uint32_t w2 = (b >> 10) & 0xffffu, w3 = (b >> 8) & 0xffffu;
    const uint32_t w4 = (b >> 6) & 0xffffu, w5 = (b >> 4) & 0xffffu;
    const uint32_t w6 = (b >> 2) & 0xffffu, w7 = b & 0xffffu;
    fr[0] = exl3_decode_pair(w0, w1);
    fr[1] = exl3_decode_pair(w2, w3);
    fr[2] = exl3_decode_pair(w4, w5);
    fr[3] = exl3_decode_pair(w6, w7);
}

// dq8_half<3, cb>: positions alternate 3 and 4 bits (KA and KA+1, odd positions
// carrying the extra bit; upstream's mask is 0xAAAA), so each group of four
// windows spans 18 + 3*KA = 27 bits and fits two words: one funnel shift, the
// other three windows the shift-downs by 4, 3, 4.
static __device__ __forceinline__ void exl3_half35_core(uint32_t a7, uint32_t b7,
                                                        uint32_t a3, uint32_t b3,
                                                        const Exl3Off& o,
                                                        half2 (&fr)[4]) {
    const uint32_t w7 = exl3_fshift(b7, a7, o.sh[0]);
    const uint32_t w6 = w7 >> 4, w5 = w6 >> 3, w4 = w5 >> 4;
    const uint32_t w3 = exl3_fshift(b3, a3, o.sh[1]);
    const uint32_t w2 = w3 >> 4, w1 = w2 >> 3, w0 = w1 >> 4;
    fr[0] = exl3_decode_pair(w0 & 0xffffu, w1 & 0xffffu);
    fr[1] = exl3_decode_pair(w2 & 0xffffu, w3 & 0xffffu);
    fr[2] = exl3_decode_pair(w4 & 0xffffu, w5 & 0xffffu);
    fr[3] = exl3_decode_pair(w6 & 0xffffu, w7 & 0xffffu);
}

// dq4<6, cb> twice (the 6-bit path): each call decodes four windows out of one
// word pair, four funnel shifts bits apart.
static __device__ __forceinline__ void exl3_dq4_6_core(uint32_t a0, uint32_t b0,
                                                       uint32_t a1, uint32_t b1,
                                                       const Exl3Off& o, int bits,
                                                       half2 (&fr)[4]) {
#pragma unroll
    for (int c = 0; c < 2; c++) {
        const uint32_t a = c ? a1 : a0;
        const uint32_t b = c ? b1 : b0;
        const int s2 = o.sh[c];
        const uint32_t w3 = exl3_fshift(b, a, s2) & 0xffffu;
        const uint32_t w2 = exl3_fshift(b, a, s2 + bits) & 0xffffu;
        const uint32_t w1 = exl3_fshift(b, a, s2 + 2 * bits) & 0xffffu;
        const uint32_t w0 = exl3_fshift(b, a, s2 + 3 * bits) & 0xffffu;
        fr[2 * c] = exl3_decode_pair(w0, w1);
        fr[2 * c + 1] = exl3_decode_pair(w2, w3);
    }
}

// Word `s & 3` of a 16-byte block. A 4-way select and not a dynamic register
// index: the four words are four registers, and indexing them by a runtime value
// would send the whole block to local memory.
static __device__ __forceinline__ uint32_t exl3_slot4(const uint4& v, uint32_t s) {
    return s == 0 ? v.x : s == 1 ? v.y : s == 2 ? v.z : v.w;
}

// One lane's word `k`, out of the two 16-byte blocks the offsets picked: bit 2
// of `sel` says which block, its low two bits say which word inside it. The wrap
// needs no case here -- the offsets reduced the word index mod W before the
// block was taken from it, so a wrapped word is named by the tile's own first
// block, exactly as the four-byte path names it with `tile[i0[0]]`.
static __device__ __forceinline__ uint32_t exl3_wword(const uint4& A, const uint4& B,
                                                      uint32_t sel) {
    const uint32_t s = sel & 3u;
    return (sel & 4u) ? exl3_slot4(B, s) : exl3_slot4(A, s);
}

// The words a lane's eight windows need, per variant: two for the decoders with
// a single word pair, four for the two-pair ones. Kept apart from the core so a
// caller can put several tiles' loads in flight before any of them is consumed.
//
// The WORD INDICES and the words they name are identical in all three branches
// below (same indices mod W, same values) -- only the way the bytes are fetched
// differs. That is what keeps the golden decode bit-exact through a change
// here: every branch gathers the same words, and every line of the decode
// arithmetic is untouched.
static __device__ __forceinline__ void exl3_load8(const uint32_t* __restrict__ tile,
                                                  const Exl3Off& o, int vari, int W,
                                                  uint32_t (&w)[4]) {
#if EXL3_WIDE_LOADS
    // Two 16-byte requests, both always issued. The second one is redundant for
    // the lanes whose whole run fits in the first block (three quarters of them
    // on a 3-bit tile) and reads the same address as the first for the lanes
    // whose two block indices are equal, but predicating it off is measurably
    // worse: `wb[1] != wb[0]` is a per-lane predicate, so the compiler puts the
    // second load behind a divergent branch and the two latencies stop
    // overlapping (measured on gate_proj at WT = 1: 611 us predicated against
    // the unconditional form below; see _build/bench_loads.py).
    const uint4 A = __ldg(reinterpret_cast<const uint4*>(tile + o.wb[0]));
    const uint4 B = __ldg(reinterpret_cast<const uint4*>(tile + o.wb[1]));
    w[0] = exl3_wword(A, B, o.wsel[0]);
    w[1] = exl3_wword(A, B, o.wsel[1]);
    if (vari == EXL3_V_HALF || vari == EXL3_V_DQ4_6) {
        w[2] = exl3_wword(A, B, o.wsel[2]);
        w[3] = exl3_wword(A, B, o.wsel[3]);
    } else {
        w[2] = w[3] = 0u;            // the pair is duplicated there anyway
    }
#elif EXL3_SHFL_LOADS
    // ONE load a lane a tile, then shuffles. MEASURED (_build/bench/walk_probe.cu,
    // the same __launch_bounds__ as this kernel, gate_proj's trellis at WT = 4
    // and S = 8, 39 MB): the walk's cost is the NUMBER OF LOAD INSTRUCTIONS, not
    // the bytes or the lines. The decoder's own 4-loads-a-lane-a-tile pattern
    // (16 a warp a k-step) walks the module in 178 us = 219 GB/s; the same bytes
    // with one load a lane a tile (4 a warp a k-step, the same address spread)
    // walk it in 82 us = 477 GB/s, 2.17x. Two loads a lane measure 178 us again,
    // and the one-16-byte-load walk measures 110 us = 353 GB/s. So the load has
    // to be a warp-wide gather of the whole tile:
    //
    //   lane L brings word L of the tile (clamped, so every lane's address is
    //   inside the tile and no neighbouring line is touched), and the four words
    //   a lane's windows need come back as four shuffles from the lanes that
    //   hold them.
    //
    // `shfl` is a register move of at most 32 lanes and its source may be a
    // runtime value, so the four indices the decode already computes name the
    // four source lanes directly. The words are the SAME words at the SAME
    // addresses (every index the offsets produce is already reduced mod W), so
    // the golden decode is bit-identical through this path.
    //
    // W > 32 (the 6-bit path, 48 words a tile) cannot be covered by one warp
    // load and keeps the four-load form.
    if (W <= 32) {
        const int lane = threadIdx.x & 31;
        const uint32_t mine = __ldg(&tile[lane < W ? lane : W - 1]);
        w[0] = __shfl_sync(0xffffffffu, mine, o.i0[0]);
        w[1] = __shfl_sync(0xffffffffu, mine, o.i2[0]);
        if (vari == EXL3_V_HALF || vari == EXL3_V_DQ4_6) {
            w[2] = __shfl_sync(0xffffffffu, mine, o.i0[1]);
            w[3] = __shfl_sync(0xffffffffu, mine, o.i2[1]);
        } else {
            w[2] = w[3] = 0u;        // the pair is duplicated there anyway
        }
    } else
#endif
    {
        w[0] = __ldg(&tile[o.i0[0]]);
        w[1] = __ldg(&tile[o.i2[0]]);
        if (vari == EXL3_V_HALF || vari == EXL3_V_DQ4_6) {
            w[2] = __ldg(&tile[o.i0[1]]);
            w[3] = __ldg(&tile[o.i2[1]]);
        } else {
            w[2] = w[3] = 0u;        // the pair is duplicated there anyway
        }
    }
}

// One lane's eight windows of one tile, in window order: fr[k] holds windows
// 2k and 2k+1 as (.x, .y). The words come from exl3_load8.
static __device__ __forceinline__ void exl3_decode8_w(const uint32_t (&w)[4],
                                                      const Exl3Off& o, int vari,
                                                      int bits, half2 (&fr)[4]) {
    switch (vari) {
    case EXL3_V_ALIGN2: exl3_align2_core(w[0], w[1], o, fr); break;
    case EXL3_V_DQ8_3: exl3_dq8_3_core(w[0], w[1], o, bits, fr); break;
    case EXL3_V_HALF: exl3_half35_core(w[0], w[1], w[2], w[3], o, fr); break;
    case EXL3_V_ALIGN4: exl3_align4_core(w[0], w[1], o, fr); break;
    default: exl3_dq4_6_core(w[0], w[1], w[2], w[3], o, bits, fr); break;
    }
}


// ============================================================================
// the fused decode GEMV/GEMM: z = xh @ A, A decoded on the fly and never stored
// ============================================================================
//
// Per k-tile (16 rows of A) and per lane: decode the lane's eight windows once,
// then consume them for each of the m tokens in the block's token group. The
// x values the lane needs are the four rows r0, r0+1, r0+8, r0+9 of the k-tile
// -- two half2 loads at indices (lane & 3) and (lane & 3) + 4, which every lane
// holding the same (lane & 3) shares, so the warp's 32 lanes read FOUR distinct
// addresses and the L1 broadcasts them. That is why this kernel needs no shared
// memory and no __syncthreads at all: the x traffic is a broadcast, and the
// trellis traffic is per-lane anyway.
//
// MEASURED, AND WHY IT IS THIS SHAPE (_build/bench_exl3.py, gate_proj, T = 1,
// 39.0 MB of trellis, P100): this version runs ~302 us = ~129 GB/s of trellis
// read. A variant that decodes the SAME tile for every k-tile -- every load an
// L1 hit, all the arithmetic unchanged -- runs ~63 us = ~624 GB/s, i.e. at the
// card's streaming ceiling (572-576 GB/s measured), so the ALU is essentially free and
// the whole cost is the latency of a walk whose stride is nto*W*4 = 122 KB.
// Staging the block's eight tiles per k-tile in shared memory with 16-byte
// loads (reconstruct.cu's shape, and the suggestion for this kernel) measures
// 788 us -- 2.4x SLOWER: the staging loop leaves one load in flight per warp
// and then the barrier makes all eight warps wait for it, 320 times over. The
// scatter that staging was meant to fix is cheaper than the barrier that fixes
// it. A register-level prefetch (the next k-tiles' words in flight while the
// current one's arithmetic runs) is the change that raises the memory-level
// parallelism here; it was built and measured, and it is worth 328 -> 292 us
// (+12%, 119 -> 134 GB/s) on this module at a depth of 4, at the cost of 158
// registers through this shared M dispatch and a second copy of the FMA chain
// for the ragged k tail. It is NOT the shipped version: the numbers by depth
// (1 -> 439, 2 -> 302, 3 -> 296, 4 -> 292, 6 -> 304, 8 -> 296 us) say the gain
// is real but the shape is not free, and two of the six modules measure a few
// percent SLOWER with it. It is kept, compiling and passing the decode tests, as
// _build/kernels_exl3_pipe4_handoff.cu.txt for whoever does the next round.
// ----------------------------------------------------------------------------
// TWO SHAPE PARAMETERS, AND THE MEASUREMENT THAT FIXED THEM
// ----------------------------------------------------------------------------
// The walk above reads its own tile once per k-step, nto*W*4 = 122 KB away from
// the previous one, and per k-step a warp has ONE cache line in flight. The rate
// is therefore warps-in-flight x bytes-per-k-step / the round-trip time, not the
// DRAM's own limit: measured across the checkpoint's modules (real trellis, T=1,
// _build/probe_splitk.py), the rate tracks the number of resident warps almost
// exactly (k_proj 1.1 warps/SM -> 12 GB/s, down_proj 5.7 -> 61, gate_proj 19.4
// -> 119), and a variant with 4x the blocks over the same bytes is 4x faster.
// Both parameters below multiply the parallelism; they are complementary, and
// the host picks them per module and per T.
//
//   WT  tiles per warp, CONSECUTIVE along the output axis. A warp still walks
//       the whole k axis, but owns WT 16-column tiles instead of one, so it has
//       WT line streams in flight per k-step and moves WT times the bytes per
//       k-step. The decode is UNCHANGED -- the same per-lane window assignment
//       is applied to each of the WT tiles, so nothing is redistributed -- and
//       the x values (the two half2 loads at (lane & 3) and (lane & 3) + 4) are
//       the same for all WT tiles and are loaded once per token per k-step.
//       WT costs registers only in the accumulators (2*WT*M floats) plus one
//       uint32[4] of pre-loaded words per tile, which is why the wide entries
//       are pinned at EXL3_GEMV_OCC blocks per SM (see the entry macro) and why
//       the runtime only asks for them when m <= 4: prefill's m = 8 would want
//       64 accumulators and spills.
//
//   S   k-slices, from gridDim.z. WT divides the blocks and S multiplies them,
//       so together they hold the SMs busy for every module: k_proj's out = 1024
//       is 2 blocks at WT = 4 without a split and 64 with the S the runtime
//       picks, down_proj (out = 5120) goes 10 -> 160, gate_proj (out = 17408)
//       34 -> 272. S = 1 (gridDim.z = 1) writes z fp16 through exactly the same
//       code as the one-tile kernel; S > 1 writes S fp32 partials and
//       exl3_sk_reduce sums them in split order and rounds once. The partials
//       are exact for a one-hot x (each split contributes w and exact zeros),
//       so the golden decode tests are bit-identical through a split as well.
//
// `kin` is THIS BLOCK's k-tile count (its slice), `koff` where the slice starts,
// and both pointers are pre-advanced by koff, so the loop below still counts
// 0..kin-1 and reads as the unsplit one does.
//
// ----------------------------------------------------------------------------
// WHAT IT BOUGHT (_build/bench_wide.py sweeps WT x S on real checkpoint data;
// these are the T = 1 numbers for the widest module of each kind, one process,
// back to back, so the card's clocks are the same for both columns)
// ----------------------------------------------------------------------------
//   module (bits)              one tile/warp       WT=4 + S       GB/s
//   gate_proj (3.5)  38.99 MB     335 us 116 GB/s   239 us   163  1.4x
//   gate_proj (4.0)  44.56 MB     264 us 168 GB/s   203 us   220  1.3x
//   down_proj (3.0)  33.42 MB     592 us  56 GB/s   219 us   153  2.7x
//   in_proj_qkv (3.0) 19.66 MB    234 us  84 GB/s   134 us   147  1.7x
//   out_proj (3.0)   11.80 MB     213 us  55 GB/s    86 us   137  2.5x
//   k_proj (3.5)      2.29 MB     107 us  22 GB/s    32 us    72  3.4x
//   lm_head (6.0)   953.55 MB    4144 us 230 GB/s  3007 us   317  1.4x
// The whole decode step (the engine's own timing, 512-token prompt, alternating
// in one process): 5.59 -> 8.31 tok/s with the prefill unmoved at 48.2/48.3.
// The GEMV's accounted total falls 128 -> 77 ms/token (_build/bench_modules.py,
// which charges the split's reduce to the GEMV).
//
// ----------------------------------------------------------------------------
// WHAT IS STILL THE WALL (re-measured this round, with the walk probe)
// ----------------------------------------------------------------------------
// `_build/bench/walk_probe.cu` repeats the walk at the SHIPPED geometry AND the
// shipped occupancy (`__launch_bounds__(256, 2)`, which _build/probe_stride.py's
// kernels did not carry), on gate_proj's real trellis, with the sum discarded
// so nothing can be optimised away. That is what makes these numbers comparable
// to the kernel itself:
//
//   mode                                     us      GB/s   requests a warp a k-step
//   the decoder's own 4 loads a lane a tile  177.8   219.3   16
//   one 4-byte load a lane a tile             81.8   476.8    4
//   one 16-byte load a lane a tile           110.5   352.9    4
//   two 16-byte loads a lane a tile          177.9   219.2    8   (= EXL3_WIDE_LOADS)
//   ... and the full kernel, same geometry   249.9   156.0   16 + the decode
//   the same bytes read contiguously         68.4   570.1
//
// The walk's cost is the number of LOAD INSTRUCTIONS per warp per k-step, and
// it is not linear in them: 16 and 8 cost the same 178 us, 4 costs 82-110. So
// the walk has a cliff between four and eight requests a k-step, and the
// decoder sits on the wrong side of it BY CONSTRUCTION: a lane's eight windows
// of a tile need words spread across the whole tile, so four lanes' worth of
// words cannot be had in fewer than four requests without either an extraction
// (tried, EXL3_WIDE_LOADS, and the two selects a word cost more than the 9%
// the width buys) or a redistribution (tried this round, EXL3_SHFL_LOADS, and
// the single load's dependency chain cost 51-66%). The walk's own ceiling at
// this geometry is therefore ~178 us for the module and the kernel is at 250,
// which is 178 + the 69 us of decode issue (no_stride, same arithmetic with
// every load an L1 hit) -- they ADD, and the one structure that would overlap
// them (EXL3_PIPE) costs more in registers than it buys in latency.
//
// ----------------------------------------------------------------------------
// AND THE SAMPLER'S CANDIDATE EXTRACTION, WHICH IS ALSO HERE NOW
// ----------------------------------------------------------------------------
// The four kernels at the end of this file (exl3_cand_hist, exl3_cand_sum,
// exl3_cand_collect, exl3_argmax16_partial/_final) turn the step's logits row
// into the sampler's candidate list on the DEVICE. On the host that cost
// 14.6 ms a token under numpy and 49.1 ms under a numpy-less python3 (496 KB
// read back, then a Python pass over all 248 077 logits); the whole device
// pipeline costs 0.094 and 0.096 ms. See the section above them.
// ============================================================================
// One tile's decode plus its FMAs, for the PIPELINED loop only (EXL3_PIPE >= 2,
// the entry the pipeline experiment drives; the shipped loop carries the same
// instructions inline -- see the note there for why sharing this helper with it
// cost the prefill path 1.71x). The arithmetic below is byte-for-byte the loop
// this file has shipped since the wide path landed, and the golden decode is
// bit-exact through either form.
template <int VARI, int M, int WT>
static __device__ __forceinline__ void exl3_tile_fma(const uint32_t (&w)[4],
                                                     const Exl3Off& o, int bits,
                                                     const half2* __restrict__ xbase,
                                                     int xstride, int k, int ntok,
                                                     int i, float (&acc)[WT][2][M]) {
    half2 fr[4];
    exl3_decode8_w(w, o, VARI, bits, fr);   // VARI is a constant here
    const float2 w00 = __half22float2(fr[0]);   // rows r0, r0+1, column c
    const float2 w01 = __half22float2(fr[1]);   // rows r0+8, r0+9, column c
    const float2 w10 = __half22float2(fr[2]);   // rows r0, r0+1, column c+8
    const float2 w11 = __half22float2(fr[3]);   // rows r0+8, r0+9, column c+8
#pragma unroll
    for (int t = 0; t < M; t++) {
        if (t < ntok) {
            const half2* xr = xbase + (size_t)t * xstride + k * 8;
            const float2 xa = __half22float2(__ldg(xr));         // rows r0, r0+1
            const float2 xb = __half22float2(__ldg(xr + 4));     // rows r0+8, r0+9
            acc[i][0][t] = fmaf(w00.x, xa.x,
                         fmaf(w00.y, xa.y,
                         fmaf(w01.x, xb.x,
                         fmaf(w01.y, xb.y, acc[i][0][t]))));
            acc[i][1][t] = fmaf(w10.x, xa.x,
                         fmaf(w10.y, xa.y,
                         fmaf(w11.x, xb.x,
                         fmaf(w11.y, xb.y, acc[i][1][t]))));
        }
    }
}

// X32 selects the INPUT's layout: 0 is the fp16 buffer had128_pre writes, 1 is
// the fp32 one with each k-tile's 16 halves lane-ordered (had128_pre_lc), where
// a lane's four x values are one aligned 16-byte load instead of two 4-byte
// loads, four conversions and the address arithmetic around them. The two agree
// bit for bit, because the reordered values are the same halves rounded the same
// way and widened exactly (tests/test_exl3.py::TestPrefillGemv).
// ---------------------------------------------------------------------------
// The shared-memory activation stage (EXL3_SMW > 0).
//
// The block's k-window is EXL3_SMW k-tiles of its WHOLE token group: M x SMW x
// 16 values, widened to fp32 and permuted so that the four rows a lane
// accumulates are one aligned 16-byte run,
//
//     dst[4*a + 0..3] <- rows {2a, 2a+1, 2a+8, 2a+9},   a = 0..3
//
// which is the same map had128_pre_lc writes to global and the same one
// _build/bench_prefill_m.py's PERM builds, so the staged values are the same
// halves widened by the same rounding and the token loop reads them with one
// LDS.128. (That map, and not the layout, is why the fp32 GLOBAL path lost:
// measured on this card, a 128-bit global request costs 1.66x the 32-bit one
// for the same bytes, while the same 128-bit access from shared memory is
// 1.39x FASTER than the shipped fetch. See _build/PREFILL_P100_OPTIONS.md.)
//
// One unit of work is a (token, k-tile, row-group) triple: the two halves of
// rows 2a,2a+1 and the two of 2a+8,2a+9, four widenings, one 16-byte store. The
// two 4-byte loads are the SAME broadcast pattern the shipped fetch uses -- 32
// lanes asking for 128 B and sharing four words -- and there are only
// M*SMW*4/256 of them a thread for a whole window, which is what keeps the
// stage off the L1's request-byte wall: 512 B of requests per k-step for the
// whole block against the shipped fetch's 32 KB.
//
// A window is clamped at both ends so a ragged token group (ntok < M) and a
// ragged k tail read in bounds. Neither clamp is ever stored: the out-of-range
// tokens' accumulators are discarded by the epilogue's `t < ntok` guard, which
// is the file's existing convention for the ragged tile reads.
// ONE array per kernel, not one per (M, SMW): the entry stages m = 16 and
// m = 32 with different windows over the SAME bytes, and a __shared__ inside a
// template allocates per instantiation -- two instantiations is two 32 KB
// arrays and 64 KB does not launch. So this is templated on the FLOAT COUNT
// (identical for every staged shape, since the byte budget is what is fixed)
// and the callers static_assert that their window fits in it.
// The `N > 0 ? N : 1` keeps the array legal in the EXL3_SM_BYTES == 0 build,
// where ptxas removes it with the dead branch (0 bytes smem in the log).
template <int N>
static __device__ __forceinline__ float* exl3_xstage_buf() {
    __shared__ float xs[N > 0 ? N : 1];
    return xs;
}

template <int M, int SMW>
static __device__ __forceinline__ void exl3_stage_window(
        float* __restrict__ dst, const half* __restrict__ x, int t0, int ntok,
        int in, int kfirst, int kend) {
    constexpr int SM = (SMW > 0 ? SMW : 1);   // 1 and not SMW: this function is
                                              // instantiated (then discarded) in
                                              // the SMW == 0 branch too, and the
                                              // division has to be legal there
    constexpr int UNITS = M * SM * 4;
    constexpr int PASSES = (UNITS + EXL3_THREADS - 1) / EXL3_THREADS;
#pragma unroll
    for (int s = 0; s < PASSES; s++) {
        const int u = threadIdx.x + s * EXL3_THREADS;
        if (UNITS % EXL3_THREADS != 0 && u >= UNITS) break;
        const int a = u & 3;                    // row group 2a, 2a+1, 2a+8, 2a+9
        const int rest = u >> 2;
        const int j = rest % SM;                // k-tile inside the window
        const int t = rest / SM;                // token inside the group
        const int tt = (t < ntok) ? t : ntok - 1;
        const int kk = (kfirst + j < kend) ? (kfirst + j) : (kend - 1);
        const half2* src = reinterpret_cast<const half2*>(
            x + (size_t)(t0 + tt) * in + (size_t)kk * 16) + a;
        const half2 lo = __ldg(src);            // rows 2a, 2a+1
        const half2 hi = __ldg(src + 4);        // rows 2a+8, 2a+9
        float4 v;
        v.x = __low2float(lo);
        v.y = __high2float(lo);
        v.z = __low2float(hi);
        v.w = __high2float(hi);
        *reinterpret_cast<float4*>(dst
                + ((size_t)j * M + t) * 16 + 4 * a) = v;
    }
}

template <int M, int VARI, int WT, int PIPE, int TUN = 0, int X32 = 0,
          bool FULL = false, int SMW = 0>
static __device__ __forceinline__ void exl3_gemv_run(const half* __restrict__ x,
                                                     const uint32_t* __restrict__ trellis,
                                                     half* __restrict__ z,
                                                     float* __restrict__ part,
                                                     int in, int out, int T,
                                                     int t0, int W, int bits,
                                                     int kin, int nto, int jt,
                                                     const Exl3Off& o, int koff,
                                                     int split) {
    static_assert(!(X32 && PIPE >= 2),
                  "the fp32 input layout is only wired into the unpipelined "
                  "loop; the pipelined variant stays in the file as the "
                  "measured dead end it is (EXL3_PIPE's note)");
    static_assert(!(SMW > 0 && (X32 || PIPE >= 2)),
                  "the shared-memory activation stage is wired into the fp16 "
                  "unpipelined loop only");
    const int lane = threadIdx.x & 31;
    float acc[WT][2][M];
#pragma unroll
    for (int i = 0; i < WT; i++)
#pragma unroll
        for (int u = 0; u < 2; u++)
#pragma unroll
            for (int t = 0; t < M; t++) acc[i][u][t] = 0.f;

    const int ntok = (t0 + M <= T) ? M : (T - t0);
    const int nleft = (nto - jt < WT) ? (nto - jt) : WT;
    const uint32_t* tile = trellis + (size_t)jt * W + (size_t)koff * nto * W;
    const half2* xbase = reinterpret_cast<const half2*>(x + (size_t)t0 * in
                                                        + (size_t)koff * 16)
                         + (lane & 3);
    const int xstride = in >> 1;                 // half2 per token
    const float* xbase32 = reinterpret_cast<const float*>(x)
                           + (size_t)t0 * in + (size_t)koff * 16 + (lane & 3) * 4;
    const size_t step = (size_t)nto * W;         // one k-tile's stride

    // This warp's WT tile bases. The ragged tail (`jt + WT > nto`) is the case
    // the previous shape handled by reading tile 0 for the tiles it does not
    // own; the clamp below reads the last tile it DOES own instead. Both are
    // only there to keep the address in bounds -- those accumulators are never
    // stored (`if (jt + i < nto)` in the epilogue) -- and the clamp is the one
    // that keeps the pipeline's per-tile addresses a constant offset, which is
    // what lets the whole k-loop address tiles as tp[i] + k*step.
    const uint32_t* tp[WT];
#pragma unroll
    for (int i = 0; i < WT; i++)
        tp[i] = tile + (size_t)(i < nleft ? i : nleft - 1) * W;

    // PIPE is a template parameter, not a #define, so this branch is resolved
    // before register allocation and the general entries (which pass PIPE = 1)
    // compile to exactly the loop they shipped with.
    if (PIPE >= 2) {
    // The pipelined walk. PIPE stages of words live in registers at once
    // (PIPE * WT * 4 uint32, which is the whole register cost of this variant);
    // the stage being decoded is d, and the load issued right after it is for
    // k + PIPE + d -- one full iteration of lead, so the round trip of a k-step
    // is covered by the decode of PIPE k-steps rather than adding to it.
    uint32_t w[PIPE][WT][4];
#pragma unroll
    for (int d = 0; d < PIPE; d++)
        if (d < kin)                             // a slice shorter than PIPE
#pragma unroll
            for (int i = 0; i < WT; i++)
                exl3_load8(tp[i] + (size_t)d * step, o, VARI, W, w[d][i]);

    int k = 0;
    for (; k + PIPE <= kin; k += PIPE) {
#pragma unroll
        for (int d = 0; d < PIPE; d++) {
#pragma unroll
            for (int i = 0; i < WT; i++)
                exl3_tile_fma<VARI, M, WT>(w[d][i], o, bits, xbase, xstride,
                                           k + d, ntok, i, acc);
            if (k + PIPE + d < kin) {            // refill this stage's slots
#pragma unroll
                for (int i = 0; i < WT; i++)
                    exl3_load8(tp[i] + (size_t)(k + PIPE + d) * step, o,
                               VARI, W, w[d][i]);
            }
        }
    }
    for (; k < kin; k++) {                       // the ragged k tail
#pragma unroll
        for (int i = 0; i < WT; i++) {
            uint32_t wt[4];
            exl3_load8(tp[i] + (size_t)k * step, o, VARI, W, wt);
            exl3_tile_fma<VARI, M, WT>(wt, o, bits, xbase, xstride, k, ntok, i, acc);
        }
    }
    } else if (SMW > 0) {
    // ------------------------------------------------------------------
    // The staged k-loop. Double buffered: the stage for window w+1 is written
    // at the FIRST k-step of window w, into the buffer window w-1 was read
    // from, and ONE __syncthreads at the end of each window both closes the
    // reads of the current buffer and publishes the writes to the other. So
    // every staging load is issued a whole window before the barrier that
    // waits on it -- which is the difference from the rejected trellis staging
    // (2.4x slower: its barrier serialised on one load in flight, 320 times
    // over). One barrier per window here, and ~40 of them a block.
    constexpr int SM = (SMW > 0 ? SMW : 1);       // dead when SMW == 0; 1 keeps
                                                  // the division legal there
    constexpr int WIN = SM * M * 16;              // floats in one buffer
    static_assert(2 * WIN <= EXL3_SM_BYTES / 4 + 1,
                  "the window does not fit the shared-memory budget");
    float* const xs = exl3_xstage_buf<EXL3_SM_BYTES / 4>();
    const int nw = (kin + SM - 1) / SM;
    const int la4 = (lane & 3) * 4;               // this lane's four rows
    exl3_stage_window<M, SMW>(xs, x, t0, ntok, in, koff, koff + kin);
    __syncthreads();
    for (int k = 0; k < kin; k++) {
        const int w = k / SM;                     // window this k-tile is in
        const int kj = k - w * SM;                // ... and its index in it
        const int wlen = ((kin - w * SM) < SM) ? (kin - w * SM) : SM;
        if (kj == 0 && w + 1 < nw)
            exl3_stage_window<M, SM>(xs + ((w + 1) & 1) * WIN, x, t0, ntok,
                                     in, koff + (w + 1) * SM, koff + kin);
        uint32_t ws[WT][4];
#pragma unroll
        for (int i = 0; i < WT; i++)
            exl3_load8(tp[i] + (size_t)k * step, o, VARI, W, ws[i]);
#pragma unroll
        for (int i = 0; i < WT; i++) {
            half2 fr[4];
            exl3_decode8_w(ws[i], o, VARI, bits, fr);
            const float2 w00 = __half22float2(fr[0]);   // rows r0, r0+1, column c
            const float2 w01 = __half22float2(fr[1]);   // rows r0+8, r0+9, column c
            const float2 w10 = __half22float2(fr[2]);   // rows r0, r0+1, column c+8
            const float2 w11 = __half22float2(fr[3]);   // rows r0+8, r0+9, column c+8
            const float* xb = xs + (w & 1) * WIN + (size_t)kj * 16 * M;
            // The same four FMAs, in the same order, over the same values: the
            // stage's permutation only decides WHICH register they arrive in.
#pragma unroll (TUN > 0 ? TUN : M)
            for (int t = 0; t < M; t++) {
                if (FULL || t < ntok) {
                    const float4 xv = *reinterpret_cast<const float4*>(
                        xb + (size_t)t * 16 + la4);
                    acc[i][0][t] = fmaf(w00.x, xv.x,
                             fmaf(w00.y, xv.y,
                             fmaf(w01.x, xv.z,
                             fmaf(w01.y, xv.w, acc[i][0][t]))));
                    acc[i][1][t] = fmaf(w10.x, xv.x,
                             fmaf(w10.y, xv.y,
                             fmaf(w11.x, xv.z,
                             fmaf(w11.y, xv.w, acc[i][1][t]))));
                }
            }
        }
        if (kj == wlen - 1) __syncthreads();      // publish w+1, close w
    }
    } else {
    for (int k = 0; k < kin; k++, tile += step) {
        // Every tile's words first, then every tile's decode: this is what puts
        // WT lines in flight per warp per k-step.
        uint32_t w[WT][4];
#if EXL3_PREFETCH
        // Ask for the NEXT k-step's lines while this one is still being
        // decoded. The walk's cost is the round trip, not the width (measured:
        // 16-byte requests move the same traversal 3.5x faster than 4-byte ones
        // in a pure walk, but only 1.09x inside this kernel -- see
        // _build/probe_stride.py and _build/bench_loads.py), and a prefetch
        // costs one instruction and no register, so it is the cheapest way to
        // overlap the two halves of the k-step. The lane asks for its own first
        // word (the warp's lanes cover the tile between them) and, at
        // EXL3_PREFETCH > 1, its last one -- the tail-biting wrap is the case
        // where those two are in different lines.
        {
            const uint32_t* nt = tile + step;
#pragma unroll
            for (int i = 0; i < WT; i++) {
                const uint32_t* tpp = nt + (size_t)(i < nleft ? i : 0) * W;
                asm volatile("prefetch.global.L2 [%0];" :: "l"(tpp + o.i0[0]));
#if EXL3_PREFETCH > 1
                asm volatile("prefetch.global.L2 [%0];" :: "l"(tpp + o.i2[1]));
#endif
            }
        }
#endif
        // Every token's x for this k-step, fetched ONCE and in one place, so
        // the M*2 loads are in flight together instead of each token's pair
        // queueing behind the previous token's FMAs (EXL3_XHOIST_M's note). The
        // arrays are one element wide at every m the hoist is not taken at, so
        // those instantiations cost no registers and compile to the shipped
        // loop.
        float2 xav[(M == EXL3_XHOIST_M) ? M : 1];
        float2 xbv[(M == EXL3_XHOIST_M) ? M : 1];
#if EXL3_XFIRST
#pragma unroll
        for (int t = 0; t < M; t++) {
            if (M == EXL3_XHOIST_M && t < ntok
                && (!EXL3_XSPLIT || t < EXL3_XSPLIT)) {
                const half2* xr = xbase + (size_t)t * xstride + k * 8;
                xav[t] = __half22float2(__ldg(xr));      // rows r0, r0+1
                xbv[t] = __half22float2(__ldg(xr + 4));  // rows r0+8, r0+9
            }
        }
#endif
#pragma unroll
        for (int i = 0; i < WT; i++)
            exl3_load8(tile + (size_t)(i < nleft ? i : 0) * W, o, VARI, W, w[i]);
#if !EXL3_XFIRST || EXL3_XSPLIT
#pragma unroll
        for (int t = 0; t < M; t++) {
            if (M == EXL3_XHOIST_M && t < ntok
                && (EXL3_XSPLIT ? t >= EXL3_XSPLIT : 1)) {
                const half2* xr = xbase + (size_t)t * xstride + k * 8;
                xav[t] = __half22float2(__ldg(xr));      // rows r0, r0+1
                xbv[t] = __half22float2(__ldg(xr + 4));  // rows r0+8, r0+9
            }
        }
#endif
        // INLINE, and not through exl3_tile_fma. The helper is the same
        // instructions and the wide path measures identically through it
        // (1.005x at WT = 4), but the ONE-TILE kernel, which is what prefill
        // runs at m = 8, is 1.71x SLOWER through it: 1329 us against 778 on
        // gate_proj's 39 MB, growing with M (1.09x at m = 1, 1.53x at m = 2,
        // 1.68x at m = 4). nvcc's scalar replacement of the accumulator array
        // does not survive the reference parameter, and at m = 8 that is 16
        // live accumulators per lane. Bisected to the helper alone (the same
        // file with it inlined: 1.001x). The pipelined path below still uses
        // it, where the loop is a dead-code experiment either way.
#pragma unroll
        for (int i = 0; i < WT; i++) {
            half2 fr[4];
            exl3_decode8_w(w[i], o, VARI, bits, fr);   // VARI is a constant here
            const float2 w00 = __half22float2(fr[0]);   // rows r0, r0+1, column c
            const float2 w01 = __half22float2(fr[1]);   // rows r0+8, r0+9, column c
            const float2 w10 = __half22float2(fr[2]);   // rows r0, r0+1, column c+8
            const float2 w11 = __half22float2(fr[3]);   // rows r0+8, r0+9, column c+8
            // The token loop's unroll factor: TUN = 0 (the default, and what
            // every entry but a deliberate experiment passes) is a full unroll,
            // which is what this kernel has always done. It is not optional at
            // m > 8 -- a partial unroll makes the accumulator index a runtime
            // value and the array goes to local memory: measured 6.6x slower
            // (212 ms against 32.3, gate_proj, T = 512, m = 16). TUN is a
            // template parameter and not a macro because nvcc 12.6 does not
            // expand in-file objects inside this pragma's argument (it reports
            // the macro as undefined); an -D value reaches it through the
            // entry's instantiation, which is ordinary code.
            //
            // FULL = this block's whole token group is live (`ntok == M`), i.e.
            // it is not the chunk's ragged tail -- 31 of a 32-group chunk at
            // m = 16, so for almost every block the guard below is a per-token
            // compare that can never fail. FULL makes it a template constant so
            // it is compiled away instead of being evaluated (or, worse,
            // forcing the whole token body through a predicate). MEASURED, and
            // it is the largest item this round found: gate_proj / up_proj /
            // in_proj_qkv, T = 512, m = 16, paired in one process against the
            // same file without it: 1.061 / 1.078 / 1.113x (32.3 -> 30.5,
            // 31.4 -> 29.1, 18.3 -> 16.5 ms), outputs bit-identical both at a
            // full chunk and at a ragged tail (_build/NOTES.md, the guard
            // round; _build/bench_prefill_m.py --cubin-b). The ragged block
            // still takes the guarded form, so no case is lost.
#pragma unroll (TUN > 0 ? TUN : M)
            for (int t = 0; t < M; t++) {
                if (FULL || t < ntok) {
                    float2 xa, xb;
                    if (M == EXL3_XHOIST_M) {
                        xa = xav[t];                        // the fetch above
                        xb = xbv[t];
                    } else if (X32) {
                        const float4 xv = __ldg(reinterpret_cast<const float4*>(
                            xbase32 + (size_t)t * in + (size_t)k * 16));
                        xa = make_float2(xv.x, xv.y);
                        xb = make_float2(xv.z, xv.w);
                    } else {
                        const half2* xr = xbase + (size_t)t * xstride + k * 8;
                        xa = __half22float2(__ldg(xr));      // rows r0, r0+1
                        xb = __half22float2(__ldg(xr + 4));  // rows r0+8, r0+9
                    }
                    acc[i][0][t] = fmaf(w00.x, xa.x,
                             fmaf(w00.y, xa.y,
                             fmaf(w01.x, xb.x,
                             fmaf(w01.y, xb.y, acc[i][0][t]))));
                    acc[i][1][t] = fmaf(w10.x, xa.x,
                             fmaf(w10.y, xa.y,
                             fmaf(w11.x, xb.x,
                             fmaf(w11.y, xb.y, acc[i][1][t]))));
                }
            }
        }
    }
    }

    // Every column of a tile is the sum of FOUR lanes' accumulators: the four
    // lanes that share `c_b = lane >> 3`, of which the A < 4 four own column
    // 2*c_b and the A >= 4 four own 2*c_b + 1 (each with its +8 partner). A
    // shfl_xor pair over the low two lane bits reduces inside each group of
    // four, and one lane per group stores. With a split (gridDim.z > 1) the
    // same reduced value goes to this split's fp32 plane instead of to z; the
    // four-lane stage and the arithmetic above it are untouched, which is what
    // keeps the split bit-identical on the golden one-hot tests.
    const int A = lane & 7;
    half* zrow = z + (size_t)t0 * out + (size_t)jt * 16;
    const int c = 2 * (lane >> 3) + (A >= 4 ? 1 : 0);
    float* prow = split ? part + ((size_t)blockIdx.z * T + t0) * out
                          + (size_t)jt * 16
                        : part;
#pragma unroll
    for (int i = 0; i < WT; i++) {
        if (jt + i < nto) {              // warp-uniform: only the ragged tail
#pragma unroll
            for (int t = 0; t < M; t++) {
                if (t < ntok) {
                    float s0 = acc[i][0][t], s1 = acc[i][1][t];
                    s0 += __shfl_xor_sync(0xffffffffu, s0, 1);
                    s0 += __shfl_xor_sync(0xffffffffu, s0, 2);
                    s1 += __shfl_xor_sync(0xffffffffu, s1, 1);
                    s1 += __shfl_xor_sync(0xffffffffu, s1, 2);
                    if (A == 0 || A == 4) {
                        if (split) {
                            float* p = prow + (size_t)t * out + (size_t)i * 16;
                            p[c] = s0;
                            p[c + 8] = s1;
                        } else {
                            half* row = zrow + (size_t)t * out + (size_t)i * 16;
                            row[c] = exl3_sat(s0);
                            row[c + 8] = exl3_sat(s1);
                        }
                    }
                }
            }
        }
    }
}

// Both dispatches are hoisted out of the k-loop: the variant switch inside the
// inner loop would cost a uniform branch per k-tile for nothing, and M has to be
// a template so the accumulators stay in registers (a runtime-indexed private
// array goes to local memory -- the trap kernels.cu's gdn_scan documents).
// (both macros read exl3_gemv_body's own locals: x, trellis, z, part, in, out,
//  T, t0, W, bits, kspan, nto, jt, o, koff, split)
#define EXL3_GEMV_M(M, VARI, WT, PIPE, TUN)                                    \
    EXL3_GEMV_MX(M, VARI, WT, PIPE, TUN, 0)
#define EXL3_GEMV_MX(M, VARI, WT, PIPE, TUN, X32)                              \
    exl3_gemv_run<M, VARI, WT, PIPE, TUN, X32>(x, trellis, z, part, in, out, T, \
                                               t0, W, bits, kspan, nto, jt, o, \
                                               koff, split)
#define EXL3_GEMV_DISPATCH(VARI, WT, PIPE)                                      \
    do {                                                                        \
        switch (m) {                                                            \
        case 1: EXL3_GEMV_M(1, VARI, WT, PIPE, 0); break;                          \
        case 2: EXL3_GEMV_M(2, VARI, WT, PIPE, 0); break;                          \
        case 4: EXL3_GEMV_M(4, VARI, WT, PIPE, 0); break;                          \
        case 8: EXL3_GEMV_M(8, VARI, WT, PIPE, 0); break;                          \
        default: break;                                                         \
        }                                                                       \
    } while (0)

// The decode entry's dispatch: M = 1 and 2 only, so the M = 8 path's 64
// accumulators cannot cost the pipelined loop a register. The runtime never
// asks the wide path for more than m = 2 (`_gemv_shape` is T <= 4 and m is the
// largest power of two <= T, and the pipelined entry is selected at m <= 2);
// a call with m = 4 or 8 on THIS entry writes nothing, exactly like any other
// unsupported m, and `exl3_gemv_w4` keeps serving those.
#define EXL3_GEMV_DISPATCH12(VARI, WT, PIPE)                                    \
    do {                                                                        \
        switch (m) {                                                            \
        case 1: EXL3_GEMV_M(1, VARI, WT, PIPE, 0); break;                          \
        case 2: EXL3_GEMV_M(2, VARI, WT, PIPE, 0); break;                          \
        default: break;                                                         \
        }                                                                       \
    } while (0)

// The prefill entry's token-loop unroll factor; 0 means "unroll fully", which
// is the general entries' behaviour and measured 2.0-2.3x slower for m > 8 (see
// the note at the loop). Swept in _build/bench_prefill_m.py.

// M = 64 is its own build decision, not a case: 2*64 = 128 accumulator registers
// per lane at WT = 1 is the ENTIRE 128-register budget at EXL3_GEMV_OCC = 2, so
// it only compiles into the prefill entry when the build asks for it
// (EXL3_M_PRE_MAX >= 64) AND that build raises the entry's occupancy choice
// along with it. Measured (`-DEXL3_M_PRE_MAX=64`, ptxas -v): at OCC = 2 the
// whole entry spills 2000 bytes and is slower at every m; at OCC = 1 it fits.
// A build without the case writes nothing for m = 64 -- the runtime's cap and
// this list must agree, which tests/test_exl3.py::TestPrefillGemv checks for
// every m the runtime can pass.
#if EXL3_M_PRE_MAX >= 64
#define EXL3_GEMV_M64_CASE(VARI, WT, PIPE) case 64: EXL3_GEMV_M(64, VARI, WT, PIPE, TUN); break;
#else
#define EXL3_GEMV_M64_CASE(VARI, WT, PIPE)
#endif

// The PREFILL dispatch: the token groups above 8 that the one-tile kernel
// cannot carry (see exl3_gemv_pre). 1..8 are here as well, so this entry
// serves any T on its own and the bit-identity of the m <= 8 cases against
// `exl3_gemv` can be checked directly (tests/test_exl3.py::TestPrefillGemv).
// A ragged token group is NOT a case here: m is the group SIZE, and the last
// block of a chunk carries `ntok = T - t0 < m` tokens through the same guard
// every entry already has.
//
// m = 32 used to be the dead case here -- the runtime used to cap at
// EXL3_M_PRE = 16 and the unstaged m = 32 was a 5.2x LOSS (measured at T = 512,
// gate_proj: 87.8 ms against m = 16's 21.7), because 2*32 accumulators do not
// fit beside the walk's per-token x addressing. The staged dispatch above it
// (EXL3_GEMV_DISPATCH_PRE_SW) brought it back: with that addressing out of the
// token loop, m = 32 fits at 128 registers with a 16-byte frame where the
// shipped build carried 112 and 208 bytes of spill, and it is worth another
// 1.03x-1.28x over the staged m = 16 (NOTES.md, the staged-activations round).
// EXL3_M_PRE_MAX and runtime.EXL3_M_PRE are 32 for that reason.
//
// The full-token-group form of one m: every token of the group is live, so the
// `t < ntok` guard is not evaluated at all -- 31 of a 32-group chunk's groups
// (see the note at the token loop for the 1.061-1.113x this was measured to be
// worth). FULL is a template parameter, so the guarded and unguarded forms are
// two instantiations rather than a runtime choice inside the loop.
#define EXL3_GEMV_MF(M, VARI, WT, PIPE, TUN)                                    \
    exl3_gemv_run<M, VARI, WT, PIPE, TUN, 0, true>(x, trellis, z, part, in, out, \
                                                   T, t0, W, bits, kspan, nto,   \
                                                   jt, o, koff, split)
// One m, either form. `full` is warp-uniform and known at launch, so the choice
// is one uniform branch per k-step, not per token.
#define EXL3_GEMV_M1C(M, VARI, WT, PIPE, TUN)                                   \
    do {                                                                        \
        if (full) { EXL3_GEMV_MF(M, VARI, WT, PIPE, TUN); }                     \
        else { EXL3_GEMV_M(M, VARI, WT, PIPE, TUN); }                           \
    } while (0)
#define EXL3_GEMV_MXS(M, VARI, WT, PIPE, TUN, X32, SMW, FULL)                  \
    exl3_gemv_run<M, VARI, WT, PIPE, TUN, X32, FULL, SMW>(                     \
        x, trellis, z, part, in, out, T, t0, W, bits, kspan, nto, jt, o, koff, \
        split)

#define EXL3_GEMV_DISPATCH_PRE(VARI, WT, PIPE)                                  \
    do {                                                                        \
        const bool full = (t0 + m <= T);                                        \
        switch (m) {                                                            \
        case 1: EXL3_GEMV_M1C(1, VARI, WT, PIPE, TUN); break;                   \
        case 2: EXL3_GEMV_M1C(2, VARI, WT, PIPE, TUN); break;                   \
        case 4: EXL3_GEMV_M1C(4, VARI, WT, PIPE, TUN); break;                   \
        case 8: EXL3_GEMV_M1C(8, VARI, WT, PIPE, TUN); break;                   \
        case 16: EXL3_GEMV_M1C(16, VARI, WT, PIPE, TUN); break;                 \
        case 32: EXL3_GEMV_M(32, VARI, WT, PIPE, TUN); break;                   \
        EXL3_GEMV_M64_CASE(VARI, WT, PIPE)                                      \
        default: break;                                                         \
        }                                                                       \
    } while (0)

// The same dispatch with the shared-memory activation stage. Only m = 16 is
// staged -- it is the shape the runtime picks for prefill (T >= 32 with a grid
// of at least EXL3_PRE_MIN_BLOCKS blocks), and it is the only one whose block
// width and k depth pay for a window; every other m falls through to the
// shipped dispatch, so this entry stays correct for all of them and the tests
// can compare it against the shipped one across m.
#define EXL3_GEMV_DISPATCH_PRE_SM(M, VARI, WT, PIPE)                            \
    do {                                                                        \
        const bool full = (t0 + m <= T);                                        \
        if (full) {                                                             \
            EXL3_GEMV_MXS(M, VARI, WT, PIPE, TUN, 0, EXL3_STAGE_SMW(M), true);  \
        } else {                                                                \
            EXL3_GEMV_MXS(M, VARI, WT, PIPE, TUN, 0, EXL3_STAGE_SMW(M), false); \
        }                                                                       \
    } while (0)
// The staged dispatch: m = 16 and m = 32 take the stage, everything else the
// shipped loop (so this entry still serves every m the tests drive it at).
#define EXL3_GEMV_DISPATCH_PRE_SW(VARI, WT, PIPE)                               \
    do {                                                                        \
        if (m == 16) {                                                          \
            EXL3_GEMV_DISPATCH_PRE_SM(16, VARI, WT, PIPE);                      \
        } else if (m == 32) {                                                   \
            EXL3_GEMV_DISPATCH_PRE_SM(32, VARI, WT, PIPE);                      \
        } else {                                                                \
            EXL3_GEMV_DISPATCH_PRE(VARI, WT, PIPE);                             \
        }                                                                       \
    } while (0)

// The same dispatch for the fp32-input entry (had128_pre_lc's layout).
#define EXL3_GEMV_DISPATCH_PRE32(VARI, WT, PIPE)                                \
    do {                                                                        \
        switch (m) {                                                            \
        case 1: EXL3_GEMV_MX(1, VARI, WT, PIPE, TUN, 1); break;                 \
        case 2: EXL3_GEMV_MX(2, VARI, WT, PIPE, TUN, 1); break;                 \
        case 4: EXL3_GEMV_MX(4, VARI, WT, PIPE, TUN, 1); break;                 \
        case 8: EXL3_GEMV_MX(8, VARI, WT, PIPE, TUN, 1); break;                 \
        case 16: EXL3_GEMV_MX(16, VARI, WT, PIPE, TUN, 1); break;               \
        default: break;                                                         \
        }                                                                       \
    } while (0)

// The body both entry points share. WT is the only difference between them.
template <int WT, bool NARROW = false, int PIPE = 1, bool PRE = false, int TUN = 0,
          bool PRE32 = false>
static __device__ __forceinline__ void exl3_gemv_body(const half* __restrict__ x,
                                                      const uint32_t* __restrict__ trellis,
                                                      half* __restrict__ z,
                                                      float* __restrict__ part,
                                                      int in, int out, int T,
                                                      int bits_x2, int m) {
    if (in <= 0 || out <= 0 || T <= 0) return;
    if ((in & 15) || (out & 15)) return;      // 16x16 tiles need both multiples
    int vari = -1;
    switch (bits_x2) {
    case 4: vari = EXL3_V_ALIGN2; break;
    case 6: vari = EXL3_V_DQ8_3; break;
    case 7: vari = EXL3_V_HALF; break;
    case 8: vari = EXL3_V_ALIGN4; break;
    case 12: vari = EXL3_V_DQ4_6; break;
    default: return;
    }
    const int w = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int W = bits_x2 * 4;                // uint32 per 16x16 tile
    const int kin = in >> 4;                  // k-tiles (16 input rows each)
    const int nto = out >> 4;                 // tiles along the output axis
    const int jt = (blockIdx.x * EXL3_WARPS + w) * WT;
    if (jt >= nto) return;                    // guard: this warp owns no tile
    const int t0 = blockIdx.y * m;
    if (t0 >= T) return;
    const int bits = bits_x2 >> 1;
    const Exl3Off o = exl3_offsets(lane * 8, vari, W, bits);

    // The k-slice this block carries: S = gridDim.z, split evenly, the first
    // `kin % S` slices one tile longer. S = 1 is the whole k range.
    const int S = gridDim.z;
    const int s = blockIdx.z;
    const int kslice = kin / S;
    const int koff = s * kslice + min(s, kin % S);
    const int kspan = kslice + (s < kin % S ? 1 : 0);
    const int split = (S > 1);
    if (kspan <= 0) return;                   // more splits than k-tiles

    if (NARROW) {
        switch (vari) {
        case EXL3_V_ALIGN2: EXL3_GEMV_DISPATCH12(EXL3_V_ALIGN2, WT, PIPE); break;
        case EXL3_V_DQ8_3: EXL3_GEMV_DISPATCH12(EXL3_V_DQ8_3, WT, PIPE); break;
        case EXL3_V_HALF: EXL3_GEMV_DISPATCH12(EXL3_V_HALF, WT, PIPE); break;
        case EXL3_V_ALIGN4: EXL3_GEMV_DISPATCH12(EXL3_V_ALIGN4, WT, PIPE); break;
        default: EXL3_GEMV_DISPATCH12(EXL3_V_DQ4_6, WT, PIPE); break;
        }
    } else if (PRE32) {
        switch (vari) {
        case EXL3_V_ALIGN2: EXL3_GEMV_DISPATCH_PRE32(EXL3_V_ALIGN2, WT, PIPE); break;
        case EXL3_V_DQ8_3: EXL3_GEMV_DISPATCH_PRE32(EXL3_V_DQ8_3, WT, PIPE); break;
        case EXL3_V_HALF: EXL3_GEMV_DISPATCH_PRE32(EXL3_V_HALF, WT, PIPE); break;
        case EXL3_V_ALIGN4: EXL3_GEMV_DISPATCH_PRE32(EXL3_V_ALIGN4, WT, PIPE); break;
        default: EXL3_GEMV_DISPATCH_PRE32(EXL3_V_DQ4_6, WT, PIPE); break;
        }
    } else if (PRE) {
#if EXL3_SM_BYTES > 0
        switch (vari) {
        case EXL3_V_ALIGN2: EXL3_GEMV_DISPATCH_PRE_SW(EXL3_V_ALIGN2, WT, PIPE); break;
        case EXL3_V_DQ8_3: EXL3_GEMV_DISPATCH_PRE_SW(EXL3_V_DQ8_3, WT, PIPE); break;
        case EXL3_V_HALF: EXL3_GEMV_DISPATCH_PRE_SW(EXL3_V_HALF, WT, PIPE); break;
        case EXL3_V_ALIGN4: EXL3_GEMV_DISPATCH_PRE_SW(EXL3_V_ALIGN4, WT, PIPE); break;
        default: EXL3_GEMV_DISPATCH_PRE_SW(EXL3_V_DQ4_6, WT, PIPE); break;
        }
#else
        switch (vari) {
        case EXL3_V_ALIGN2: EXL3_GEMV_DISPATCH_PRE(EXL3_V_ALIGN2, WT, PIPE); break;
        case EXL3_V_DQ8_3: EXL3_GEMV_DISPATCH_PRE(EXL3_V_DQ8_3, WT, PIPE); break;
        case EXL3_V_HALF: EXL3_GEMV_DISPATCH_PRE(EXL3_V_HALF, WT, PIPE); break;
        case EXL3_V_ALIGN4: EXL3_GEMV_DISPATCH_PRE(EXL3_V_ALIGN4, WT, PIPE); break;
        default: EXL3_GEMV_DISPATCH_PRE(EXL3_V_DQ4_6, WT, PIPE); break;
        }
#endif
    } else {
        switch (vari) {
        case EXL3_V_ALIGN2: EXL3_GEMV_DISPATCH(EXL3_V_ALIGN2, WT, PIPE); break;
        case EXL3_V_DQ8_3: EXL3_GEMV_DISPATCH(EXL3_V_DQ8_3, WT, PIPE); break;
        case EXL3_V_HALF: EXL3_GEMV_DISPATCH(EXL3_V_HALF, WT, PIPE); break;
        case EXL3_V_ALIGN4: EXL3_GEMV_DISPATCH(EXL3_V_ALIGN4, WT, PIPE); break;
        default: EXL3_GEMV_DISPATCH(EXL3_V_DQ4_6, WT, PIPE); break;
        }
    }
}

// z = xh @ A for one projection. See the header for the geometry.
//
//   x        [T, in]  fp16, already rotated and scaled by had128_pre
//   trellis  the `.trellis` tensor of the module, raw: (in/16, out/16, 8*bits)
//            uint32, i.e. the int16 payload reinterpreted little-endian -- the
//            word indices the decoders take are the same bytes the oracle reads.
//   z        [T, out] fp16
//   bits_x2  2*bits: 4, 6, 7, 8 or 12. Anything else writes nothing.
//   m        tokens per block: 1, 2, 4 or 8. Anything else writes nothing.
//   grid     (ceil(out/128), ceil(T/m)) -- one tile per warp, the whole k range.
//            This is the entry the tests drive; exl3_gemv_w2/w4 below are the
//            wide ones the runtime uses, and WT = 1 makes this identical to them.
extern "C" __global__ void exl3_gemv(const half* __restrict__ x,
                                     const uint32_t* __restrict__ trellis,
                                     half* __restrict__ z,
                                     int in, int out, int T,
                                     int bits_x2, int m) {
    exl3_gemv_body<1>(x, trellis, z, nullptr, in, out, T, bits_x2, m);
}

// The runtime's entries: WT consecutive tiles per warp and a k-split on
// gridDim.z. `part` is [gridDim.z, T, out] fp32 and is written only when
// gridDim.z > 1 (then `z` is not touched; exl3_sk_reduce finishes the job).
// WT = 2 halves the blocks and doubles the bytes per warp-k-step; WT = 4 does
// that four times. Blocks = ceil(nto / (8*WT)) * S.
//
// __launch_bounds__ pins the occupancy at EXL3_GEMV_OCC blocks per SM (Pascal
// has 64K registers per SM, so that is the register budget of the whole entry,
// M = 8 included). The wide entries are driven with m <= 2 by the runtime and
// the M = 4/8 paths there are the ones that pay for it; the M = 1 path -- the
// one that matters for decode -- keeps its registers. Without the bound ptxas
// gives exl3_gemv_w4 144 registers, i.e. one block per SM, and WT's whole point
// is the number of warps per SM that each have WT lines in flight.
#ifndef EXL3_GEMV_OCC
#define EXL3_GEMV_OCC 2
#endif
// The general entries: M = 1..8, the unpipelined k-loop (PIPE = 1), exactly
// what this file shipped before the pipeline existed.
#define EXL3_GEMV_ENTRY(NAME, WT)                                              \
    extern "C" __global__ void __launch_bounds__(EXL3_THREADS, EXL3_GEMV_OCC)  \
    NAME(const half* __restrict__ x,                                           \
         const uint32_t* __restrict__ trellis,                                 \
         half* __restrict__ z,                                                 \
         float* __restrict__ part, int in, int out,                            \
         int T, int bits_x2, int m) {                                          \
        exl3_gemv_body<WT, false, 1>(x, trellis, z, part, in, out, T,          \
                                     bits_x2, m);                              \
    }

EXL3_GEMV_ENTRY(exl3_gemv_w2, 2)
EXL3_GEMV_ENTRY(exl3_gemv_w4, 4)
EXL3_GEMV_ENTRY(exl3_gemv_w8, 8)

// The pipelined decode entry: same geometry as exl3_gemv_w4 (WT = 4, split on
// gridDim.z), the k-walk EXL3_PIPE deep, and M = 1/2 only -- which is what buys
// the registers the pipeline needs. The runtime asks for this one at m <= 2 and
// for exl3_gemv_w4 above that, so decode (m = 1 always) takes it and prefill's
// token groups do not. Bit-identical to exl3_gemv_w4 by construction: PIPE only
// changes WHEN a k-tile's words are read, never which words or what is done
// with them (tests/test_exl3.py::TestPipelinedGemv checks that against the
// one-tile kernel and against exl3_gemv_w4 on every module).
#define EXL3_GEMV_ENTRY_PIPED(NAME, WT, PIPE)                                  \
    extern "C" __global__ void __launch_bounds__(EXL3_THREADS, EXL3_GEMV_OCC)  \
    NAME(const half* __restrict__ x,                                           \
         const uint32_t* __restrict__ trellis,                                 \
         half* __restrict__ z,                                                 \
         float* __restrict__ part, int in, int out,                            \
         int T, int bits_x2, int m) {                                          \
        exl3_gemv_body<WT, true, PIPE>(x, trellis, z, part, in, out, T,        \
                                       bits_x2, m);                            \
    }

EXL3_GEMV_ENTRY_PIPED(exl3_gemv_p4, 4, EXL3_PIPE)

// ============================================================================
// THE PREFILL ENTRY: m > 8, and the measurement that sizes it
// ============================================================================
// Prefill's cost is not the trellis bytes, it is how many times the WALK
// repeats: the one-tile kernel walks the whole k axis for each group of m
// tokens, so a T-token chunk pays `ceil(T/m)` full passes over the module's
// trellis, and the walk's cost is per-warp-k-step LATENCY (measured in bench/
// walk_probe.cu at ~1 us for every request shape this decode can produce), not
// bandwidth. So the chunk's GEMV time is
//
//     (T/m) * (walk + m * fma) = T*walk/m + T*fma
//
// -- linear in T either way, but with a term that falls as 1/m and a term that
// does not. Every module here is at m = 8 today (the largest the general entry
// instantiates) and the fma term is what is left on the floor: at m = 8 the
// walk is 5/6 of a prefill GEMV's time on gate_proj (_build/bench_prefill_m.py,
// paired in one process, T = 512).
//
// The register arithmetic for going further is the accumulator array: 2*M
// floats per lane at WT = 1, so M = 16 is 32 registers, M = 32 is 64, M = 64
// is 128 -- the last of which does not fit beside the decode's own words at
// EXL3_GEMV_OCC = 2 blocks per SM, which is why the entry below carries its own
// occupancy (EXL3_GEMV_BIG_OCC) and the measured table that picks it.
//
// What was left after m = 16 was the token loop's own guard and the x fetch.
// The guard is gone from the full groups (1.061-1.113x, the EXL3_GEMV_MF note).
// The x fetch is gone as well, and HOW it went is worth stating because two
// earlier rounds read the same probe the other way: it was never only the
// instructions. _build/PREFILL_P100_OPTIONS.md measures the card's L1 at a wall
// of ~32 BYTES OF REQUEST per cycle per SM, with the shipped fetch (2 LDG.32 and
// four HADD2.F32 a token, 32 KB of requests a warp-k-step) at 87% of it -- which
// is why removing the four conversions alone bought 1.15x instead of the 1.27x
// their slot count promises, and why every WIDER global request lost badly
// (1xLDG.128 and 2xLDG.64 both 1.66x SLOWER for the same bytes; had128_pre_lc
// 1.29x, EXL3_SHFL_LOADS 1.51x, the rejected trellis staging 2.4x). One 128-bit
// load a token from SHARED MEMORY, which is a separate array from the L1 on this
// part, measures 1.391x on the same instruction count. Hence EXL3_SM_BYTES.
//
// The stage also unblocks m = 32, which the register note below had to reject:
// with the per-token x addressing and its 64-bit pointer pair out of the token
// loop, 2*32 accumulators now fit, and the decode -- 22% of the k-step and
// independent of the group size -- is paid half as often. Measured against the
// SHIPPED kernel, paired in one process, T = 512: 1.29x-1.42x at m = 16,
// 1.30x-2.49x at m = 32, and bit-identical output at both (the group size only
// changes WHICH block computes a (token, column) pair). At the engine level a
// 512-token chunk goes 8083 -> 4843 ms, 1.669x, reproduced to 0.02% between the
// two cool reps. _build/NOTES.md, the "staged activations" round, has the rest.
#ifndef EXL3_GEMV_BIG_OCC
#define EXL3_GEMV_BIG_OCC 2
#endif
#if EXL3_M_PRE_MAX >= 64 && EXL3_GEMV_BIG_OCC > 1
#error "m = 64 needs 2*64 accumulator registers, so EXL3_GEMV_OCC must be 1 \
there (measured: 2000 bytes of spill in the hot loop at OCC 2, 1.9x slower \
than m = 32)"
#endif

// WT = 1 (one 16-column tile per warp, the shape every prefill GEMV uses),
// M = 1..64, PIPE = 1 (the unpipelined k-loop: a k-step with 32 tokens on it is
// not the round trip a pipeline was supposed to cover -- see the EXL3_PIPE note).
// The entry is selected by the runtime for T above the wide path's range, and
// only for the m values it instantiates; anything else writes nothing.
//
// BIT-IDENTITY: m changes only WHICH BLOCK computes a given (token, column) --
// the k loop, the window→lane map, the four-lane shuffle reduction and the fp32
// order are the same code at every m, so two launches that differ only in m are
// element-for-element identical (tests/test_exl3.py::TestPrefillGemv checks the
// m >= 16 cases against m = 8 and against the golden decode).
extern "C" __global__ void __launch_bounds__(EXL3_THREADS, EXL3_GEMV_BIG_OCC)
exl3_gemv_pre(const half* __restrict__ x,
              const uint32_t* __restrict__ trellis,
              half* __restrict__ z,
              int in, int out, int T, int bits_x2, int m) {
    exl3_gemv_body<1, false, 1, true, EXL3_TOKEN_UNROLL>(x, trellis, z, nullptr,
                                                         in, out, T, bits_x2, m);
}

// The same entry over the SINGLE-LOAD input layout: `x` is had128_pre_lc's fp32
// buffer (`in` floats a row, each k-tile's 16 halves lane-ordered), so a lane's
// four x values are one 16-byte load.
//
// DO NOT READ THE NEXT PARAGRAPH AS A WIN. This entry measured 32.3 -> 41.5 ms
// against the fp16 one (gate_proj, T = 512, m = 16, paired in one process) --
// 1.29x SLOWER -- and `runtime.EXL3_PRE_X32` is False for that reason. The
// 21.1 ms that used to be quoted here was the x-removed *probe's* number, i.e.
// the cost of the slot count this entry does NOT remove: it replaces two 4-byte
// loads and four conversions with ONE 16-byte load, whose 32 lanes ask the L1
// for 512 B (64 B unique) instead of 256 B, and that 4x in request bytes costs
// more than the conversions save. The guard round measured the rest of the
// story: making the x loads L1-resident with the SAME slot count is exactly
// 1.000x, and REMOVING them is 0.61x -- so it is the instructions, not where
// they hit or how wide they are. Keep the fp16 path.
//
// (The fused values are bit-identical to the fp16 path by construction; see
// had128_body_lc.)
extern "C" __global__ void __launch_bounds__(EXL3_THREADS, EXL3_GEMV_BIG_OCC)
exl3_gemv_pre32(const half* __restrict__ x,
                const uint32_t* __restrict__ trellis,
                half* __restrict__ z,
                int in, int out, int T, int bits_x2, int m) {
    exl3_gemv_body<1, false, 1, false, EXL3_TOKEN_UNROLL, true>(
        x, trellis, z, nullptr, in, out, T, bits_x2, m);
}

// z = sat(sum_s part[s]) over the S fp32 partials of a split launch, in the
// order the splits were laid out (split 0 first, i.e. ascending k) and with one
// fp16 rounding at the end -- the same rounding the unsplit kernel does, so a
// split changes only the fp32 associativity. n = T*out.
extern "C" __global__ void exl3_sk_reduce(const float* __restrict__ part,
                                          half* __restrict__ z, int n, int S) {
    if (n <= 0 || S <= 0 || part == nullptr || z == nullptr) return;
    const int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        float s = part[i];
        for (int j = 1; j < S; j++) s += part[(size_t)j * n + i];
        z[i] = exl3_sat(s);
    }
}

// ============================================================================
// the sampler's candidate extraction (device side)
// ============================================================================
//
// `_sample` used to turn the 248 077 logits of a step into its candidate list on
// the host: `struct.unpack` of the whole fp16 row into Python floats, a `max`,
// and a comprehension over all of them. That is 38 ms of a decode token under a
// plain python3 (the machine's interpreter has no numpy) against 0.3 ms under
// numpy, and the row is 496 KB to read back either way. The reduction below
// replaces it: the device reduces the row to a histogram of the fp16 values'
// sortable keys plus the row's maximum (2 KB back), the host picks a threshold
// from that histogram, and a second pass returns ONLY the candidates at or above
// it -- tens of entries, 8 KB back in a fixed buffer.
//
// The pieces, and why this shape:
//
//   exl3_key16      fp16 bits -> a u16 whose ASCENDING order is the values'
//                   ascending order (the standard float-to-sortable-integer
//                   trick, 16-bit wide here). It makes "above a threshold" a
//                   key comparison and it makes the histogram a monotone
//                   bucketization of the row. -0.0 is folded onto +0.0 so equal
//                   VALUES share a key -- the sampler's order is (value desc,
//                   index desc), and a row where -0.0 and +0.0 straddle the
//                   boundary must not be split by the bucket.
//   exl3_cand_hist  one pass: the row's largest key per block, and a block-local
//                   512-bucket histogram of the keys (bucket = key >> 7, so the
//                   512 buckets tile the whole key space). Each block writes its
//                   own 2 KB histogram; there is no global atomic anywhere, so
//                   nothing needs zeroing between tokens (see exl3_cand_sum).
//   exl3_cand_sum   one block: sums the per-block histograms and maxima, and
//                   resets the candidate counter the collect pass fills.
//   exl3_cand_collect  one pass: every value >= `lo` appends its (value, index)
//                   pair to the output buffer, with a global counter and a cap.
//                   Only the elements at or above the threshold do an atomic,
//                   so the atomic traffic is the candidate count (tens).
//
// The host picks `lo` as max(the bucket edge the histogram says holds the
// top_k-th value, the smallest fp16 strictly above the sampling floor) -- the
// first is what makes the readback small, the second is what keeps the
// candidate set a SUBSET of `{v : v > top - 30 * temperature}`, which is the
// set the old host loop selected from. `top - 30 * temperature` is the floor
// `_candidates` uses, and the candidate ORDER is imposed by the host on the
// returned list, so the list, and the token a seed draws from it, are the same
// as the host loop's (tests/test_exl3.py::TestSamplerCandidates and
// _build/check_sampler.py hold both to it, on real logits rows).
#define CAND_BINS 512            // buckets over the 16-bit key space
#define CAND_SHIFT 7             // 512 << 7 == 65536

// fp16 bits -> a sortable key (ascending bits == ascending value), with -0.0
// folded onto +0.0. NaN maps to the top of the space (a row with a NaN logit is
// already undefined in the host path: numpy's max propagates it and nothing is
// above the resulting NaN floor).
static __device__ __forceinline__ unsigned int exl3_key16(unsigned short u) {
    if (u == 0x8000u) u = 0x0000u;              // -0.0 -> +0.0
    return (u & 0x8000u) ? (unsigned int)(0xFFFFu - u) : (unsigned int)(u | 0x8000u);
}

// The row's keys, per block: `perm[blockIdx.x]` = the block's largest key and
// `per[blockIdx.x * CAND_BINS + b]` = how many of its elements fall in bucket b.
// grid (EXL3_CAND_BLOCKS), block 256.
extern "C" __global__ void __launch_bounds__(EXL3_THREADS)
exl3_cand_hist(const half* __restrict__ v, int n, unsigned int* __restrict__ per,
               unsigned int* __restrict__ perm) {
    if (n <= 0 || v == nullptr || per == nullptr || perm == nullptr) return;
    __shared__ unsigned int sh[CAND_BINS];
    __shared__ unsigned int shmax;
    const int t = threadIdx.x;
    for (int b = t; b < CAND_BINS; b += EXL3_THREADS) sh[b] = 0u;
    if (t == 0) shmax = 0u;
    __syncthreads();
    unsigned int lmax = 0u;
    const int stride = gridDim.x * EXL3_THREADS;
    for (int i = blockIdx.x * EXL3_THREADS + t; i < n; i += stride) {
        const unsigned int k = exl3_key16(__half_as_ushort(__ldg(&v[i])));
        if (k > lmax) lmax = k;
        atomicAdd(&sh[k >> CAND_SHIFT], 1u);
    }
    // Every thread folds its own maximum in: `lmax` is per-thread, so a
    // block-level reduction is a shared atomicMax from each of them (256 a
    // block, and they are the only shared atomics outside the histogram).
    atomicMax(&shmax, lmax);
    __syncthreads();
    unsigned int* row = per + (size_t)blockIdx.x * CAND_BINS;
    for (int b = t; b < CAND_BINS; b += EXL3_THREADS) row[b] = sh[b];
    if (t == 0) perm[blockIdx.x] = shmax;
}

// Sum the per-block histograms and maxima into one 2 KB histogram plus the
// row's largest key, and reset the candidate counter. grid (1), block 256.
// `out[0]` = the largest key, `out[1 + b]` = bucket b's count.
extern "C" __global__ void __launch_bounds__(EXL3_THREADS)
exl3_cand_sum(const unsigned int* __restrict__ per,
              const unsigned int* __restrict__ perm, int nblocks,
              unsigned int* __restrict__ out, unsigned int* __restrict__ cnt) {
    const int t = threadIdx.x;
    for (int b = t; b < CAND_BINS; b += EXL3_THREADS) {
        unsigned int s = 0u;
        for (int k = 0; k < nblocks; k++) s += per[(size_t)k * CAND_BINS + b];
        out[1 + b] = s;
    }
    if (t == 0) {
        unsigned int m = 0u;
        for (int k = 0; k < nblocks; k++) if (perm[k] > m) m = perm[k];
        out[0] = m;
        if (cnt != nullptr) cnt[0] = 0u;
    }
}

// Every value >= `lo`, as (value f32, index u32) pairs in `vals`/`idxs`, with
// the count in cnt[0]. `slot >= cap` means the count overflowed the buffer: the
// pair is dropped but cnt still counts it, so the caller sees the overflow and
// can fall back. grid (EXL3_CAND_BLOCKS), block 256.
extern "C" __global__ void __launch_bounds__(EXL3_THREADS)
exl3_cand_collect(const half* __restrict__ v, int n, float lo,
                  float* __restrict__ vals, unsigned int* __restrict__ idxs,
                  unsigned int* __restrict__ cnt, int cap) {
    if (n <= 0 || v == nullptr || vals == nullptr || idxs == nullptr
        || cnt == nullptr || cap <= 0) return;
    const int stride = gridDim.x * EXL3_THREADS;
    for (int i = blockIdx.x * EXL3_THREADS + threadIdx.x; i < n; i += stride) {
        const float x = __half2float(__ldg(&v[i]));
        if (x >= lo) {
            const unsigned int slot = atomicAdd(cnt, 1u);
            if (slot < (unsigned int)cap) {
                vals[slot] = x;
                idxs[slot] = (unsigned int)i;
            }
        }
    }
}

// The greedy path's argmax over an fp16 row: the same reduction `argmax_partial`
// + `argmax_final` (kernels.cu) do for fp32, with the same join -- largest
// value, and the SMALLEST index among equal values, which is what
// `max(range(n), key=lambda i: (values[i], -i))` returns. It exists because the
// greedy path used to unpack the whole row on the host: 0.3 ms with numpy, and
// the same ~30 ms the stochastic path paid under a numpy-less python3.
//
// grid (EXL3_AMAX_BLOCKS), block 256; keys/idxs are that long.
extern "C" __global__ void __launch_bounds__(EXL3_THREADS)
exl3_argmax16_partial(const half* __restrict__ v, float* __restrict__ keys,
                      int* __restrict__ idxs, int n) {
    __shared__ float sk[EXL3_THREADS];
    __shared__ int si[EXL3_THREADS];
    const int t = threadIdx.x;
    const int per = (n + gridDim.x - 1) / gridDim.x;
    const int lo = (int)blockIdx.x * per;
    const int hi = lo + per < n ? lo + per : n;

    float bv = -EXL3_F16_MAX;
    int bi = n;
    for (int i = lo + t; i < hi; i += EXL3_THREADS) {
        const float x = __half2float(__ldg(&v[i]));
        if (x > bv || (x == bv && i < bi)) { bv = x; bi = i; }
    }
    sk[t] = bv;
    si[t] = bi;
    __syncthreads();
    for (int s = EXL3_THREADS >> 1; s > 0; s >>= 1) {
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

// The merge, and the winner's value alongside its index (the sampler's floor is
// derived from the value). grid (1), block 256, n = the partial count.
extern "C" __global__ void __launch_bounds__(EXL3_THREADS)
exl3_argmax16_final(const float* __restrict__ keys, const int* __restrict__ idxs,
                    int n, int* __restrict__ out, float* __restrict__ vmax) {
    __shared__ float sk[EXL3_THREADS];
    __shared__ int si[EXL3_THREADS];
    const int t = threadIdx.x;
    float bv = -EXL3_F16_MAX;
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
    if (t == 0) {
        out[0] = si[0];
        vmax[0] = sk[0];
    }
}

// ============================================================================
// plain fp16 GEMV with fp32 output (the two unquantized GDN projections)
// ============================================================================
//
// y[t][o] = sum_i x[t][i] * w[o][i], w fp16 (out, in) row-major. This exists
// because `gdn_scalars` (kernels.cu) takes `a` and `b` as fp32: in_proj_a and
// in_proj_b are the only projections with no trellis, and their dtype is the
// interface, not a choice. One warp per output row, lanes striding the row in
// half2, fp32 accumulation and a warp butterfly sum (the order differs from a
// linear sum by rounding only; the oracle is float64 and the test reports the
// measured error).
// grid (ceil(out/8), T), block 256.
extern "C" __global__ void gemv_f16_f32(const half* __restrict__ x,
                                        const half* __restrict__ w,
                                        float* __restrict__ y,
                                        int in, int out) {
    if (in <= 0 || out <= 0 || (in & 1)) return;
    const int o = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int t = blockIdx.y;
    if (o >= out) return;
    const int lane = threadIdx.x & 31;
    const half2* xr = reinterpret_cast<const half2*>(x + (size_t)t * in);
    const half2* wr = reinterpret_cast<const half2*>(w + (size_t)o * in);
    float acc = 0.f;
    for (int i = lane; i < (in >> 1); i += 32) {
        const float2 xv = __half22float2(__ldg(&xr[i]));
        const float2 wv = __half22float2(__ldg(&wr[i]));
        acc = fmaf(xv.x, wv.x, acc);
        acc = fmaf(xv.y, wv.y, acc);
    }
#pragma unroll
    for (int s = 16; s > 0; s >>= 1)
        acc += __shfl_xor_sync(0xffffffffu, acc, s);
    if (lane == 0) y[(size_t)t * out + o] = acc;
}

// Both of a GDN layer's fp16 projections (in_proj_a and in_proj_b, `out` rows
// each) in ONE launch, one 128-thread block per output row: gemv_f16_f32 gave
// each row one warp and the launch 6 blocks, so the pair cost ~15 us twice on
// an idle card. Same products, fp32 sum in a different order (16-byte loads,
// then a block reduction). grid (2*out, T), block 128; in a multiple of 8.
extern "C" __global__ void __launch_bounds__(128)
gemv_ab_f32(const half* __restrict__ x, const half* __restrict__ wa,
            const half* __restrict__ wb, float* __restrict__ ya,
            float* __restrict__ yb, int in, int out) {
    __shared__ float red[4];
    if (in <= 0 || out <= 0 || (in & 7)) return;
    const int ob = blockIdx.x, t = blockIdx.y;
    const bool second = ob >= out;
    const int o = second ? ob - out : ob;
    const uint4* wr = reinterpret_cast<const uint4*>((second ? wb : wa) + (size_t)o * in);
    const uint4* xr = reinterpret_cast<const uint4*>(x + (size_t)t * in);
    float acc = 0.f;
    for (int i = threadIdx.x; i < (in >> 3); i += 128) {
        const uint4 xv = __ldg(&xr[i]);
        const uint4 wv = __ldg(&wr[i]);
        const half2* xh = reinterpret_cast<const half2*>(&xv);
        const half2* wh = reinterpret_cast<const half2*>(&wv);
#pragma unroll
        for (int q = 0; q < 4; q++) {
            const float2 a = __half22float2(xh[q]);
            const float2 b = __half22float2(wh[q]);
            acc = fmaf(a.x, b.x, acc);
            acc = fmaf(a.y, b.y, acc);
        }
    }
#pragma unroll
    for (int s = 16; s > 0; s >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, s);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x == 0)
        (second ? yb : ya)[(size_t)t * out + o] = (red[0] + red[1]) + (red[2] + red[3]);
}

// gemv_ab_f32 for a prefill chunk: the pair as a small tiled GEMM. A block
// owns 32 tokens x the `out` (<= 48) rows of ONE of the two matrices
// (blockIdx.y); k is staged 64 at a time as fp32 (x [k][token], w [k][row]), a
// thread accumulates 2 tokens x 3 rows. The same fp32 products as gemv_ab_f32,
// summed in k order (one chain an output): another fp32 sum order. The GEMV
// re-read both weight matrices once a token; here once a 32 tokens.
// MEASURED (T = 1024, one layer): 1058 -> 558 us; below ~300 tokens the
// per-block k walk (80 stages) is slower than the GEMV (runtime GAB_MIN_T).
// grid (ceil(T / 32), 2), block 256; out <= 48, in % 64 == 0.
#define GAB_TB 32
#define GAB_KC 64
extern "C" __global__ void __launch_bounds__(256)
gemm_ab_f32(const half* __restrict__ x, const half* __restrict__ wa,
            const half* __restrict__ wb, float* __restrict__ ya,
            float* __restrict__ yb, int in, int out, int T) {
    __shared__ __align__(16) float xs[GAB_KC][GAB_TB + 4];
    __shared__ __align__(16) float ws[GAB_KC][48 + 4];
    if (in <= 0 || out <= 0 || out > 48 || (in % GAB_KC)) return;
    const int tid = threadIdx.x, tt = tid & 15, tr = tid >> 4;   // tokens 2 tt.., rows 3 tr..
    const int t0 = blockIdx.x * GAB_TB;
    const half* w = blockIdx.y ? wb : wa;
    float* y = blockIdx.y ? yb : ya;
    float acc[2][3];
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 3; j++) acc[i][j] = 0.f;
    for (int k0 = 0; k0 < in; k0 += GAB_KC) {
        {   // x: 32 tokens x 64 k, one uint4 (8 halves) a thread
            const int t = tid >> 3, kq = (tid & 7) * 8;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (t0 + t < T) v = __ldg(reinterpret_cast<const uint4*>(x + (size_t)(t0 + t) * in + k0 + kq));
            const half* h = reinterpret_cast<const half*>(&v);
#pragma unroll
            for (int q = 0; q < 8; q++) xs[kq + q][t] = __half2float(h[q]);
        }
        for (int i = tid; i < 48 * 8; i += 256) {   // w: 48 rows x 64 k
            const int r = i >> 3, kq = (i & 7) * 8;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (r < out) v = __ldg(reinterpret_cast<const uint4*>(w + (size_t)r * in + k0 + kq));
            const half* h = reinterpret_cast<const half*>(&v);
#pragma unroll
            for (int q = 0; q < 8; q++) ws[kq + q][r] = __half2float(h[q]);
        }
        __syncthreads();
#pragma unroll 8
        for (int k = 0; k < GAB_KC; k++) {
            const float2 a = *reinterpret_cast<const float2*>(&xs[k][2 * tt]);
            const float b0 = ws[k][3 * tr], b1 = ws[k][3 * tr + 1], b2 = ws[k][3 * tr + 2];
            acc[0][0] = fmaf(a.x, b0, acc[0][0]); acc[1][0] = fmaf(a.y, b0, acc[1][0]);
            acc[0][1] = fmaf(a.x, b1, acc[0][1]); acc[1][1] = fmaf(a.y, b1, acc[1][1]);
            acc[0][2] = fmaf(a.x, b2, acc[0][2]); acc[1][2] = fmaf(a.y, b2, acc[1][2]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const int t = t0 + 2 * tt + i;
        if (t >= T) continue;
#pragma unroll
        for (int j = 0; j < 3; j++)
            if (3 * tr + j < out) y[(size_t)t * out + 3 * tr + j] = acc[i][j];
    }
}

// ============================================================================
// PREFILL AS A GEMM (xp_gemm*): decode each weight ONCE per BM tokens
// ============================================================================
// The prefill entry above decodes a weight once per m <= 32 tokens and streams
// x past it; it runs at ~60% of the card's FP32 rate. Here a block owns a
// 128-column x BM-token output tile: per k-tile (16 input rows) each of its 8
// warps decodes ONE trellis tile with the shipped decoder (same words, same
// fp16 weights, widened exactly to fp32) into shared memory, the block stages
// the 16 x BM activations beside it, and every thread accumulates an 8 x TM
// register tile -- 4 LDS.128 per 64 FFMA. Double buffered: the next k-tile's
// trellis words and x are fetched into registers before the current k-tile's
// FMAs, one barrier per k-tile. MEASURED (_build/xg/bench_xp.py, T = 512,
// paired in one process against exl3_gemv_pre at the runtime's m): 1.40x
// weighted over the layers, gate/up/in_proj_qkv 1.44-1.47x (gate_proj 11.7
// ms = 7.8 TFLOPS, 82% of FP32 peak); 1.17x at T = 128, a loss below T ~ 96
// (xp_gemm64 wins at T = 64). The weights are bit-identical to the GEMV's;
// only the fp32 summation order differs (sequential over k here), which moves
// ~0.7% of the fp16 outputs by one ulp.
//
// xp_gemm128s is the same tile with a k-split on gridDim.z (fp32 partials to
// `part`, exl3_sk_reduce finishes) for the outputs too narrow to fill the card
// (k_proj / v_proj: 1.5x at S = 3). It is NOT used where the plain entry fills
// the card: its loop is the same instructions but measures ~15% slower per
// unit of work (register allocation), so splitting down_proj's 1.4 waves gains
// nothing.
//
// grid (out / 128, ceil(T / BM), S), block 256; `in` a multiple of 16, `out` of 128.
#define XP_BN 128
template <int VARI, int BM, bool SPLIT>
static __device__ __forceinline__ void xp_body(const half* __restrict__ x,
                                               const uint32_t* __restrict__ trellis,
                                               half* __restrict__ z, float* __restrict__ part,
                                               int in, int out, int T, int bits, float* smem) {
    constexpr int TM = BM / 16;                    // tokens per thread
    float (*Xs)[16][BM] = reinterpret_cast<float (*)[16][BM]>(smem);
    float (*Ws)[16][XP_BN] = reinterpret_cast<float (*)[16][XP_BN]>(smem + 2 * 16 * BM);
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int tx = tid & 15, ty = tid >> 4;
    const int W = (VARI == EXL3_V_ALIGN2) ? 16 : (VARI == EXL3_V_DQ8_3) ? 24
                : (VARI == EXL3_V_HALF) ? 28 : (VARI == EXL3_V_ALIGN4) ? 32 : 48;
    const int kall = in >> 4, nto = out >> 4;
    const int S = SPLIT ? gridDim.z : 1, sz = SPLIT ? blockIdx.z : 0;
    const int kslice = SPLIT ? kall / S : kall;
    const int koff = SPLIT ? sz * kslice + min(sz, kall % S) : 0;
    const int kin = SPLIT ? kslice + (sz < kall % S ? 1 : 0) : kall;
    const int col0 = blockIdx.x * XP_BN, tok0 = blockIdx.y * BM;
    const int jt = (col0 >> 4) + w;                // this warp's tile column
    const Exl3Off o = exl3_offsets(lane * 8, VARI, W, bits);
    // the lane's 8 weights are rows {r0, r0+1, r0+8, r0+9} x cols {cc, cc+8}
    const int r0 = 2 * (lane & 3);
    const int cc = 16 * w + 2 * (lane >> 3) + ((lane & 7) >= 4 ? 1 : 0);
    constexpr int XH = BM / 16;                    // x halves per thread per k-tile
    constexpr int XG = 16 / XH;                    // threads per token
    const size_t tstep = (size_t)nto * W;
    const uint32_t* tbase = trellis + (size_t)jt * W + (SPLIT ? (size_t)koff * tstep : 0);
    if (SPLIT) x += (size_t)koff * 16;
    const int xt = tid / XG, xr = (tid % XG) * XH;  // this thread's token, first row

    uint32_t wn[4];
    uint4 xn;
    auto fetch = [&](int kt) {
        exl3_load8(tbase + (size_t)kt * tstep, o, VARI, W, wn);
        const int tok = tok0 + xt;
        if (tok < T) {
            const half* src = x + (size_t)tok * in + kt * 16 + xr;
            if (XH == 8) xn = __ldg(reinterpret_cast<const uint4*>(src));
            else { const uint2 v = __ldg(reinterpret_cast<const uint2*>(src)); xn = make_uint4(v.x, v.y, 0, 0); }
        } else {
            xn = make_uint4(0, 0, 0, 0);           // a ragged tail multiplies zeros
        }
    };
    auto stage = [&](int buf) {
        half2 fr[4];
        exl3_decode8_w(wn, o, VARI, bits, fr);
        const float2 a = __half22float2(fr[0]), b = __half22float2(fr[1]);
        const float2 c = __half22float2(fr[2]), d = __half22float2(fr[3]);
        Ws[buf][r0][cc] = a.x;      Ws[buf][r0 + 1][cc] = a.y;
        Ws[buf][r0 + 8][cc] = b.x;  Ws[buf][r0 + 9][cc] = b.y;
        Ws[buf][r0][cc + 8] = c.x;  Ws[buf][r0 + 1][cc + 8] = c.y;
        Ws[buf][r0 + 8][cc + 8] = d.x; Ws[buf][r0 + 9][cc + 8] = d.y;
        const half2* h = reinterpret_cast<const half2*>(&xn);
#pragma unroll
        for (int q = 0; q < XH / 2; q++) {
            const float2 f = __half22float2(h[q]);
            Xs[buf][xr + 2 * q][xt] = f.x;
            Xs[buf][xr + 2 * q + 1][xt] = f.y;
        }
    };

    float acc[TM][8];
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < 8; j++) acc[i][j] = 0.f;

    fetch(0);
    stage(0);
    if (kin > 1) fetch(1);
    __syncthreads();
    for (int kt = 0; kt < kin; kt++) {
        const int buf = kt & 1;
        if (kt + 1 < kin) {
            stage(buf ^ 1);                       // the buffer kt-1 was read from
            if (kt + 2 < kin) fetch(kt + 2);
        }
#pragma unroll
        for (int kk = 0; kk < 16; kk++) {
            float av[TM], bv[8];
            const float4 a0 = *reinterpret_cast<const float4*>(&Xs[buf][kk][ty * 4]);
            av[0] = a0.x; av[1] = a0.y; av[2] = a0.z; av[3] = a0.w;
            if (TM == 8) {
                const float4 a1 = *reinterpret_cast<const float4*>(&Xs[buf][kk][BM / 2 + ty * 4]);
                av[4 % TM] = a1.x; av[5 % TM] = a1.y; av[6 % TM] = a1.z; av[7 % TM] = a1.w;
            }
            const float4 b0 = *reinterpret_cast<const float4*>(&Ws[buf][kk][tx * 4]);
            const float4 b1 = *reinterpret_cast<const float4*>(&Ws[buf][kk][64 + tx * 4]);
            bv[0] = b0.x; bv[1] = b0.y; bv[2] = b0.z; bv[3] = b0.w;
            bv[4] = b1.x; bv[5] = b1.y; bv[6] = b1.z; bv[7] = b1.w;
#pragma unroll
            for (int i = 0; i < TM; i++)
#pragma unroll
                for (int j = 0; j < 8; j++) acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
        }
        __syncthreads();                          // publish buf^1, close buf
    }
#pragma unroll
    for (int i = 0; i < TM; i++) {
        const int tok = tok0 + ((TM == 8) ? ((i < 4) ? ty * 4 + i : BM / 2 + ty * 4 + i - 4)
                                          : ty * 4 + i);
        if (tok >= T) continue;
        if (SPLIT) {
            float* pr = part + ((size_t)sz * T + tok) * out + col0;
            *reinterpret_cast<float4*>(pr + tx * 4) =
                make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
            *reinterpret_cast<float4*>(pr + 64 + tx * 4) =
                make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
            continue;
        }
        half* zr = z + (size_t)tok * out + col0;
        exl3_u2h2 o0, o1;
        o0.h2[0] = __floats2half2_rn(exl3_satf(acc[i][0]), exl3_satf(acc[i][1]));
        o0.h2[1] = __floats2half2_rn(exl3_satf(acc[i][2]), exl3_satf(acc[i][3]));
        o1.h2[0] = __floats2half2_rn(exl3_satf(acc[i][4]), exl3_satf(acc[i][5]));
        o1.h2[1] = __floats2half2_rn(exl3_satf(acc[i][6]), exl3_satf(acc[i][7]));
        *reinterpret_cast<uint2*>(zr + tx * 4) = o0.u;
        *reinterpret_cast<uint2*>(zr + 64 + tx * 4) = o1.u;
    }
}

// The shared buffer lives in the entry, not in the template: a __shared__
// array inside xp_body would be allocated once per (VARI, BM) instantiation.
#define XP_ENTRY(NAME, BM, OCC, SPLIT)                                          \
    extern "C" __global__ void __launch_bounds__(256, OCC)                     \
    NAME(const half* __restrict__ x, const uint32_t* __restrict__ tr,          \
         half* __restrict__ z, float* __restrict__ part, int in, int out,      \
         int T, int bits_x2) {                                                 \
        __shared__ __align__(16) float smem[2 * 16 * (BM + XP_BN)];            \
        if (in <= 0 || out <= 0 || T <= 0 || (in & 15) || (out & 127)) return;  \
        const int bits = bits_x2 >> 1;                                         \
        switch (bits_x2) {                                                     \
        case 4: xp_body<EXL3_V_ALIGN2, BM, SPLIT>(x, tr, z, part, in, out, T, bits, smem); break; \
        case 6: xp_body<EXL3_V_DQ8_3, BM, SPLIT>(x, tr, z, part, in, out, T, bits, smem); break;  \
        case 7: xp_body<EXL3_V_HALF, BM, SPLIT>(x, tr, z, part, in, out, T, bits, smem); break;   \
        case 8: xp_body<EXL3_V_ALIGN4, BM, SPLIT>(x, tr, z, part, in, out, T, bits, smem); break; \
        case 12: xp_body<EXL3_V_DQ4_6, BM, SPLIT>(x, tr, z, part, in, out, T, bits, smem); break; \
        default: break;                                                        \
        }                                                                      \
    }
XP_ENTRY(xp_gemm128, 128, 2, false)
XP_ENTRY(xp_gemm128s, 128, 2, true)
XP_ENTRY(xp_gemm64, 64, 3, false)
// ============================================================================
// PREFILL GEMM IN fp16 MATH (xh_gemm*): HFMA2 with fp32 accumulation per 32 k
// ============================================================================
// Same weights, same tile walk as xp_gemm128, but the products run on the
// card's 2x-rate fp16 pipe: HFMA2 accumulates FT * 16 = 32 products per output
// in fp16, then the partial is widened into an fp32 accumulator (the flush is
// unconditional because k-tiles are walked in groups of FT). x is staged as
// plain halves; the HFMA2 operand selector (.H0_H0 / .H1_H1) broadcasts a token
// over a column pair for free. NT = 512 threads own a 256-token tile (warps 0-7
// decode the 8 trellis tiles, every thread stages one uint4 of x), so a decoded
// weight feeds twice the FMAs of xp_gemm128. Pointers advance per k-tile (no
// 64-bit index math in the loop).
// MEASURED (_build/xg/xp_ceiling.py, T = 1024, sustained 1328 MHz): gate_proj
// 23.45 -> 15.33 ms, down_proj 23.73 -> 15.34 ms (1.53x; 62% of the HFMA2
// peak). Numerics vs xp_gemm128 on N(0, 0.5) x: mean |dz| 6.7e-4 of rms(z),
// max 5.7e-3; on real activations 5.3e-4 of rms(z). PPL / KL: _build/OPTIMIZE_2026-10-01_log.md.
//
// The `s` entries split k in groups of FT k-tiles over gridDim.z into fp32
// partials (exl3_sk_reduce finishes), for the grids that leave SMs idle in the
// last wave (5120-wide outputs at 256..1023 tokens: 40 or 80 blocks on 56 SMs).
// NT = 128 / 64 (xh_gemm64 / 32) keep the 8 x 8 thread tile on 64 / 32 tokens:
// 4 / 2 warps decode the 8 trellis tiles, 2 / 4 each. At 3 / 6 blocks per SM
// they hold 164-168 registers without spills (at 4 / 8 they spilled and ran
// 10-15% slower). Against xs_gemm64 / 32 at their best split: 0.80 / 0.78 of
// the model's GEMM time at 64 / 32 tokens (_build/bench_xs_split.py).
//
// grid (out / 128, ceil(T / (NT / 2)), S), block NT; `in` a multiple of 16 * FT.
// xh_body's products over one staged k-tile: NI = 8 the thread's two token
// halves, NI = 4 the first only (the second is the last tile's padding).
template <int NI, int BM>
static __device__ __forceinline__ void xh_ktile(const half (*Xs)[BM], const half (*Ws)[XP_BN],
                                                int tx, int ty, half2 (&hacc)[8][4]) {
#pragma unroll
    for (int kk = 0; kk < 16; kk++) {
        half2 av[8], bv[4];
        const uint2 a0 = *reinterpret_cast<const uint2*>(&Xs[kk][ty * 4]);
        const uint2 b0 = *reinterpret_cast<const uint2*>(&Ws[kk][tx * 4]);
        const uint2 b1 = *reinterpret_cast<const uint2*>(&Ws[kk][64 + tx * 4]);
        const half2* pa0 = reinterpret_cast<const half2*>(&a0);
        const half2* pb0 = reinterpret_cast<const half2*>(&b0);
        const half2* pb1 = reinterpret_cast<const half2*>(&b1);
        av[0] = __low2half2(pa0[0]); av[1] = __high2half2(pa0[0]);
        av[2] = __low2half2(pa0[1]); av[3] = __high2half2(pa0[1]);
        if (NI == 8) {
            const uint2 a1 = *reinterpret_cast<const uint2*>(&Xs[kk][BM / 2 + ty * 4]);
            const half2* pa1 = reinterpret_cast<const half2*>(&a1);
            av[4 % NI] = __low2half2(pa1[0]); av[5 % NI] = __high2half2(pa1[0]);
            av[6 % NI] = __low2half2(pa1[1]); av[7 % NI] = __high2half2(pa1[1]);
        }
        bv[0] = pb0[0]; bv[1] = pb0[1]; bv[2] = pb1[0]; bv[3] = pb1[1];
#pragma unroll
        for (int i = 0; i < NI; i++)
#pragma unroll
            for (int j = 0; j < 4; j++) hacc[i][j] = __hfma2(av[i], bv[j], hacc[i][j]);
    }
}

template <int VARI, int FT, int NT, bool SPLIT, bool TAIL>
static __device__ __forceinline__ void xh_body(const half* __restrict__ x,
                                                const uint32_t* __restrict__ trellis,
                                                half* __restrict__ z, float* __restrict__ part,
                                                int in, int out, int T, int bits, uint32_t* smem) {
    constexpr int BM = NT / 2;
    half (*Xs)[16][BM] = reinterpret_cast<half (*)[16][BM]>(smem);
    half (*Ws)[16][XP_BN] = reinterpret_cast<half (*)[16][XP_BN]>(smem + 16 * BM);
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int tx = tid & 15, ty = tid >> 4;
    const int W = (VARI == EXL3_V_ALIGN2) ? 16 : (VARI == EXL3_V_DQ8_3) ? 24
                : (VARI == EXL3_V_HALF) ? 28 : (VARI == EXL3_V_ALIGN4) ? 32 : 48;
    const int kgall = in / (16 * FT), nto = out >> 4;     // k-tile GROUPS
    const int S = SPLIT ? gridDim.z : 1, sz = SPLIT ? blockIdx.z : 0;
    const int kgs = kgall / S;
    const int kgoff = SPLIT ? sz * kgs + min(sz, kgall % S) : 0;
    const int kin = FT * (SPLIT ? kgs + (sz < kgall % S ? 1 : 0) : kgall);
    const int koff = FT * kgoff;
    const int col0 = blockIdx.x * XP_BN, tok0 = blockIdx.y * BM;
    const bool dec = w < 8;
    const int jt = (col0 >> 4) + (w & 7);
    const Exl3Off o = exl3_offsets(lane * 8, VARI, W, bits);
    const int r0 = 2 * (lane & 3);
    const int cc = 16 * (w & 7) + 2 * (lane >> 3) + ((lane & 7) >= 4 ? 1 : 0);
    const size_t tstep = (size_t)nto * W;
    const uint32_t* tbase = trellis + (size_t)jt * W + (SPLIT ? (size_t)koff * tstep : 0);
    const int xt = tid >> 1, xr = (tid & 1) * 8;

    // NT = 128 / 64 has 4 / 2 warps for the 8 trellis tiles: each decodes DT of them
    constexpr int NW = NT / 32, DT = NW >= 8 ? 1 : 8 / NW;
    uint32_t wn[DT][4];
    uint4 xn;
    const uint32_t* tp = tbase;
    const bool xin = tok0 + xt < T;
    const uint4* xp = reinterpret_cast<const uint4*>(x + (size_t)(xin ? tok0 + xt : 0) * in + (size_t)koff * 16 + xr);
    auto fetch = [&](int) {                        // called for kt = 0, 1, 2, ... in order
        if (dec) {
#pragma unroll
            for (int d = 0; d < DT; d++) exl3_load8(tp + d * NW * W, o, VARI, W, wn[d]);
        }
        tp += tstep;
        xn = xin ? __ldg(xp) : make_uint4(0, 0, 0, 0);
        xp += 2;
    };
    auto stage = [&](int buf) {
        if (dec) {
#pragma unroll
            for (int d = 0; d < DT; d++) {
                const int c = cc + 16 * NW * d;
                half2 fr[4];
                exl3_decode8_w(wn[d], o, VARI, bits, fr);
                Ws[buf][r0][c] = __low2half(fr[0]);      Ws[buf][r0 + 1][c] = __high2half(fr[0]);
                Ws[buf][r0 + 8][c] = __low2half(fr[1]);  Ws[buf][r0 + 9][c] = __high2half(fr[1]);
                Ws[buf][r0][c + 8] = __low2half(fr[2]);  Ws[buf][r0 + 1][c + 8] = __high2half(fr[2]);
                Ws[buf][r0 + 8][c + 8] = __low2half(fr[3]); Ws[buf][r0 + 9][c + 8] = __high2half(fr[3]);
            }
        }
        const half2* h = reinterpret_cast<const half2*>(&xn);
#pragma unroll
        for (int q = 0; q < 4; q++) {
            Xs[buf][xr + 2 * q][xt] = __low2half(h[q]);
            Xs[buf][xr + 2 * q + 1][xt] = __high2half(h[q]);
        }
    };

    // a warp's tokens are 8 w .. 8 w + 7 of each half of the tile: a half wholly
    // past T (the last tile's padding) skips its products -- the warp still
    // decodes and stages for the others
    const bool liveA = tok0 + 8 * w < T, liveB = tok0 + BM / 2 + 8 * w < T;
    float acc[8][8];
    half2 hacc[8][4];
#pragma unroll
    for (int i = 0; i < 8; i++) {
#pragma unroll
        for (int j = 0; j < 8; j++) acc[i][j] = 0.f;
#pragma unroll
        for (int j = 0; j < 4; j++) hacc[i][j] = __float2half2_rn(0.f);
    }

    if (kin > 0) {
        fetch(0);
        stage(0);
        if (kin > 1) fetch(1);
    }
    __syncthreads();
    for (int kg = 0; kg < kin; kg += FT) {
#pragma unroll
        for (int u = 0; u < FT; u++) {
            const int kt = kg + u;
            const int buf = (FT % 2 == 0) ? (u & 1) : (kt & 1);
            if (kt + 1 < kin) {
                stage(buf ^ 1);
                if (kt + 2 < kin) fetch(kt + 2);
            }
            if (!TAIL || liveB) xh_ktile<8, BM>(Xs[buf], Ws[buf], tx, ty, hacc);
            else if (liveA) xh_ktile<4, BM>(Xs[buf], Ws[buf], tx, ty, hacc);
            __syncthreads();
        }
#pragma unroll
        for (int i = 0; i < 8; i++)
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const float2 f = __half22float2(hacc[i][j]);
                acc[i][2 * j] += f.x;
                acc[i][2 * j + 1] += f.y;
                hacc[i][j] = __float2half2_rn(0.f);
            }
    }
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const int tok = tok0 + (i >> 2) * (BM / 2) + ty * 4 + (i & 3);
        if (tok >= T) continue;
        if (SPLIT) {
            float* pr = part + ((size_t)sz * T + tok) * out + col0;
            *reinterpret_cast<float4*>(pr + tx * 4) =
                make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
            *reinterpret_cast<float4*>(pr + 64 + tx * 4) =
                make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
            continue;
        }
        half* zr = z + (size_t)tok * out + col0;
        exl3_u2h2 o0, o1;
        o0.h2[0] = __floats2half2_rn(exl3_satf(acc[i][0]), exl3_satf(acc[i][1]));
        o0.h2[1] = __floats2half2_rn(exl3_satf(acc[i][2]), exl3_satf(acc[i][3]));
        o1.h2[0] = __floats2half2_rn(exl3_satf(acc[i][4]), exl3_satf(acc[i][5]));
        o1.h2[1] = __floats2half2_rn(exl3_satf(acc[i][6]), exl3_satf(acc[i][7]));
        *reinterpret_cast<uint2*>(zr + tx * 4) = o0.u;
        *reinterpret_cast<uint2*>(zr + 64 + tx * 4) = o1.u;
    }
}

#define XH_ENTRY(NAME, FT, NT, OCC, SPLIT, TAIL)                                     \
    extern "C" __global__ void __launch_bounds__(NT, OCC)                      \
    NAME(const half* __restrict__ x, const uint32_t* __restrict__ tr,          \
         half* __restrict__ z, float* __restrict__ part, int in, int out,      \
         int T, int bits_x2) {                                                 \
        __shared__ __align__(16) uint32_t smem[16 * (NT / 2) + 16 * XP_BN];    \
        if (in <= 0 || out <= 0 || T <= 0 || (in % (16 * FT)) || (out & 127)) return; \
        const int bits = bits_x2 >> 1;                                         \
        switch (bits_x2) {                                                     \
        case 4: xh_body<EXL3_V_ALIGN2, FT, NT, SPLIT, TAIL>(x, tr, z, part, in, out, T, bits, smem); break; \
        case 6: xh_body<EXL3_V_DQ8_3, FT, NT, SPLIT, TAIL>(x, tr, z, part, in, out, T, bits, smem); break;  \
        case 7: xh_body<EXL3_V_HALF, FT, NT, SPLIT, TAIL>(x, tr, z, part, in, out, T, bits, smem); break;   \
        case 8: xh_body<EXL3_V_ALIGN4, FT, NT, SPLIT, TAIL>(x, tr, z, part, in, out, T, bits, smem); break; \
        case 12: xh_body<EXL3_V_DQ4_6, FT, NT, SPLIT, TAIL>(x, tr, z, part, in, out, T, bits, smem); break; \
        default: break;                                                        \
        }                                                                      \
    }
// xh_body for the 256-token tile (NT = 512) with TWO k-tiles a barrier: every
// warp decodes (16 trellis tiles a pair: 8 columns x 2 k-tiles over the 16
// warps, where xh_body left half of them idle at each barrier), x for both
// k-tiles staged together, operands by LDS.128 (a thread's 8 tokens and 8
// columns contiguous). The same products, flushes and order as xh_body: bit-
// identical. Measured (xh_bench, T = 1024, all eight projection shapes): 1.025x.
// TAIL: a warp owns tokens 16 w .. 16 w + 15; one wholly past T skips its
// products (the last tile's padding) but still decodes and stages.
template <int VARI, int NT, bool SPLIT, bool TAIL>
static __device__ __forceinline__ void xh2_body(const half* __restrict__ x,
                                                 const uint32_t* __restrict__ trellis,
                                                 half* __restrict__ z, float* __restrict__ part,
                                                 int in, int out, int T, int bits, uint32_t* smem) {
    constexpr int BM = NT / 2;
    half (*Xs)[32][BM] = reinterpret_cast<half (*)[32][BM]>(smem);
    half (*Ws)[32][XP_BN] = reinterpret_cast<half (*)[32][XP_BN]>(smem + 32 * BM);
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int tx = tid & 15, ty = tid >> 4;
    const int W = (VARI == EXL3_V_ALIGN2) ? 16 : (VARI == EXL3_V_DQ8_3) ? 24
                : (VARI == EXL3_V_HALF) ? 28 : (VARI == EXL3_V_ALIGN4) ? 32 : 48;
    const int kgall = in / 32, nto = out >> 4;           // k-tile PAIRS
    const int S = SPLIT ? gridDim.z : 1, sz = SPLIT ? blockIdx.z : 0;
    const int kgs = kgall / S;
    const int kgoff = SPLIT ? sz * kgs + min(sz, kgall % S) : 0;
    const int np = SPLIT ? kgs + (sz < kgall % S ? 1 : 0) : kgall;
    const int koff = 2 * kgoff;
    const int col0 = blockIdx.x * XP_BN, tok0 = blockIdx.y * BM;
    constexpr int NW = NT / 32, DJ = 16 / NW;             // decode jobs a warp a pair
    const Exl3Off o = exl3_offsets(lane * 8, VARI, W, bits);
    const int r0 = 2 * (lane & 3);
    const int ccl = 2 * (lane >> 3) + ((lane & 7) >= 4 ? 1 : 0);
    const size_t tstep = (size_t)nto * W;
    const uint32_t* tp = trellis + (size_t)(col0 >> 4) * W + (size_t)koff * tstep;
    const int xt = tid >> 1, xr = (tid & 1) * 8;
    uint32_t wn[DJ][4];
    uint4 xn0, xn1;
    const bool xin = tok0 + xt < T;
    const uint4* xp = reinterpret_cast<const uint4*>(x + (size_t)(xin ? tok0 + xt : 0) * in + (size_t)koff * 16 + xr);
    auto fetch = [&]() {
#pragma unroll
        for (int d = 0; d < DJ; d++) {
            const int id = w + NW * d;
            exl3_load8(tp + (size_t)(id >> 3) * tstep + (id & 7) * W, o, VARI, W, wn[d]);
        }
        tp += 2 * tstep;
        xn0 = xin ? __ldg(xp) : make_uint4(0, 0, 0, 0);
        xn1 = xin ? __ldg(xp + 2) : make_uint4(0, 0, 0, 0);
        xp += 4;
    };
    auto stage = [&](int buf) {
#pragma unroll
        for (int d = 0; d < DJ; d++) {
            const int id = w + NW * d;
            const int kr = (id >> 3) * 16 + r0, c = 16 * (id & 7) + ccl;
            half2 fr[4];
            exl3_decode8_w(wn[d], o, VARI, bits, fr);
            Ws[buf][kr][c] = __low2half(fr[0]);      Ws[buf][kr + 1][c] = __high2half(fr[0]);
            Ws[buf][kr + 8][c] = __low2half(fr[1]);  Ws[buf][kr + 9][c] = __high2half(fr[1]);
            Ws[buf][kr][c + 8] = __low2half(fr[2]);  Ws[buf][kr + 1][c + 8] = __high2half(fr[2]);
            Ws[buf][kr + 8][c + 8] = __low2half(fr[3]); Ws[buf][kr + 9][c + 8] = __high2half(fr[3]);
        }
        const half2* h0 = reinterpret_cast<const half2*>(&xn0);
        const half2* h1 = reinterpret_cast<const half2*>(&xn1);
#pragma unroll
        for (int q = 0; q < 4; q++) {
            Xs[buf][xr + 2 * q][xt] = __low2half(h0[q]);
            Xs[buf][xr + 2 * q + 1][xt] = __high2half(h0[q]);
            Xs[buf][16 + xr + 2 * q][xt] = __low2half(h1[q]);
            Xs[buf][16 + xr + 2 * q + 1][xt] = __high2half(h1[q]);
        }
    };

    float acc[8][8];
    half2 hacc[8][4];
#pragma unroll
    for (int i = 0; i < 8; i++) {
#pragma unroll
        for (int j = 0; j < 8; j++) acc[i][j] = 0.f;
#pragma unroll
        for (int j = 0; j < 4; j++) hacc[i][j] = __float2half2_rn(0.f);
    }
    if (np > 0) {
        fetch();
        stage(0);
        if (np > 1) fetch();
    }
    __syncthreads();
    const bool live = !TAIL || tok0 + 16 * w < T;
    for (int p = 0; p < np; p++) {
        const int buf = p & 1;
        if (p + 1 < np) {
            stage(buf ^ 1);
            if (p + 2 < np) fetch();
        }
        if (live) {
#pragma unroll
        for (int kk = 0; kk < 32; kk++) {
            half2 av[8], bv[4];
            const uint4 a0 = *reinterpret_cast<const uint4*>(&Xs[buf][kk][ty * 8]);
            const uint4 b0 = *reinterpret_cast<const uint4*>(&Ws[buf][kk][tx * 8]);
            const half2* pa0 = reinterpret_cast<const half2*>(&a0);
            const half2* pb0 = reinterpret_cast<const half2*>(&b0);
#pragma unroll
            for (int u = 0; u < 4; u++) {
                av[2 * u] = __low2half2(pa0[u]); av[2 * u + 1] = __high2half2(pa0[u]);
                bv[u] = pb0[u];
            }
#pragma unroll
            for (int i = 0; i < 8; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) hacc[i][j] = __hfma2(av[i], bv[j], hacc[i][j]);
        }
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 8; i++)
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const float2 f = __half22float2(hacc[i][j]);
                acc[i][2 * j] += f.x;
                acc[i][2 * j + 1] += f.y;
                hacc[i][j] = __float2half2_rn(0.f);
            }
    }
    constexpr int CO = 4;
    const int ct = tx * 8;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const int tok = tok0 + ty * 8 + i;
        if (tok >= T) continue;
        if (SPLIT) {
            float* pr = part + ((size_t)sz * T + tok) * out + col0;
            *reinterpret_cast<float4*>(pr + ct) = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
            *reinterpret_cast<float4*>(pr + CO + ct) = make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
            continue;
        }
        half* zr = z + (size_t)tok * out + col0;
        exl3_u2h2 o0, o1;
        o0.h2[0] = __floats2half2_rn(exl3_satf(acc[i][0]), exl3_satf(acc[i][1]));
        o0.h2[1] = __floats2half2_rn(exl3_satf(acc[i][2]), exl3_satf(acc[i][3]));
        o1.h2[0] = __floats2half2_rn(exl3_satf(acc[i][4]), exl3_satf(acc[i][5]));
        o1.h2[1] = __floats2half2_rn(exl3_satf(acc[i][6]), exl3_satf(acc[i][7]));
        *reinterpret_cast<uint2*>(zr + ct) = o0.u;
        *reinterpret_cast<uint2*>(zr + CO + ct) = o1.u;
    }
}

#define XH2_ENTRY(NAME, NT, OCC, SPLIT, TAIL)                                   \
    extern "C" __global__ void __launch_bounds__(NT, OCC)                      \
    NAME(const half* __restrict__ x, const uint32_t* __restrict__ tr,          \
         half* __restrict__ z, float* __restrict__ part, int in, int out,      \
         int T, int bits_x2) {                                                 \
        __shared__ __align__(16) uint32_t smem[32 * (NT / 2) + 32 * XP_BN];    \
        if (in <= 0 || out <= 0 || T <= 0 || (in % 32) || (out & 127)) return; \
        const int bits = bits_x2 >> 1;                                         \
        switch (bits_x2) {                                                     \
        case 4: xh2_body<EXL3_V_ALIGN2, NT, SPLIT, TAIL>(x, tr, z, part, in, out, T, bits, smem); break; \
        case 6: xh2_body<EXL3_V_DQ8_3, NT, SPLIT, TAIL>(x, tr, z, part, in, out, T, bits, smem); break;  \
        case 7: xh2_body<EXL3_V_HALF, NT, SPLIT, TAIL>(x, tr, z, part, in, out, T, bits, smem); break;   \
        case 8: xh2_body<EXL3_V_ALIGN4, NT, SPLIT, TAIL>(x, tr, z, part, in, out, T, bits, smem); break; \
        case 12: xh2_body<EXL3_V_DQ4_6, NT, SPLIT, TAIL>(x, tr, z, part, in, out, T, bits, smem); break; \
        default: break;                                                        \
        }                                                                      \
    }
XH2_ENTRY(xh_gemm256, 512, 1, false, false)
XH2_ENTRY(xh_gemm256s, 512, 1, true, false)
XH_ENTRY(xh_gemm256t, 2, 512, 1, false, true)
XH_ENTRY(xh_gemm256st, 2, 512, 1, true, true)
XH_ENTRY(xh_gemm128, 2, 256, 2, false, false)
XH_ENTRY(xh_gemm128t, 2, 256, 2, false, true)
XH_ENTRY(xh_gemm128s, 2, 256, 2, true, false)
XH_ENTRY(xh_gemm128st, 2, 256, 2, true, true)
XH_ENTRY(xh_gemm64, 2, 128, 3, false, false)
XH_ENTRY(xh_gemm64t, 2, 128, 3, false, true)
XH_ENTRY(xh_gemm64s, 2, 128, 3, true, false)
XH_ENTRY(xh_gemm64st, 2, 128, 3, true, true)
XH_ENTRY(xh_gemm32, 2, 64, 6, false, false)
XH_ENTRY(xh_gemm32t, 2, 64, 6, false, true)
XH_ENTRY(xh_gemm32s, 2, 64, 6, true, false)
XH_ENTRY(xh_gemm32st, 2, 64, 6, true, true)

// ============================================================================
// SHORT PREFILLS (xs_gemm16/32/64, `s`: k-split): the fp16 tile at BM = 16,
// 32 or 64 tokens
// ============================================================================
// A prompt of tens of tokens through xh_gemm128 pays a 128-token tile's FMAs
// (~900 ms a forward whatever T), and through the older GEMV it walks the
// trellis once per 8-32 tokens. Here a block owns 128 columns x BM tokens, a
// thread 8 columns x BM/16 tokens; per k-tile each warp decodes one trellis
// tile (as xh_gemm128) and the FMAs are BM/128 of xh's. Every output takes the
// same 32-product fp16 partials in the same k order as xh_gemm*, so without a
// split the values are xh's bit for bit. The split entries cut k in pairs of
// k-tiles over gridDim.z into fp32 partials (exl3_sk_reduce finishes), for the
// grids too small to fill the card (down_proj, k/v_proj at short T).
//
// grid (out / 128, ceil(T / BM), S), block 256; `in` a multiple of 32; BM 16,
// 32 or 64.
template <int VARI, int BM, bool SPLIT>
static __device__ __forceinline__ void xs_body(const half* __restrict__ x,
                                               const uint32_t* __restrict__ trellis,
                                               half* __restrict__ z, float* __restrict__ part,
                                               int in, int out, int T, int bits, uint32_t* smem) {
    constexpr int FT = 2, TM = BM / 16, XG = 16 / TM;
    half (*Xs)[16][BM] = reinterpret_cast<half (*)[16][BM]>(smem);
    half (*Ws)[16][XP_BN] = reinterpret_cast<half (*)[16][XP_BN]>(smem + 16 * BM);
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int tx = tid & 15, ty = tid >> 4;
    const int W = (VARI == EXL3_V_ALIGN2) ? 16 : (VARI == EXL3_V_DQ8_3) ? 24
                : (VARI == EXL3_V_HALF) ? 28 : (VARI == EXL3_V_ALIGN4) ? 32 : 48;
    const int kpall = in >> 5, nto = out >> 4;           // k-tile PAIRS
    const int S = SPLIT ? gridDim.z : 1, sz = SPLIT ? blockIdx.z : 0;
    const int kps = kpall / S;
    const int kpoff = SPLIT ? sz * kps + min(sz, kpall % S) : 0;
    const int kin = FT * (SPLIT ? kps + (sz < kpall % S ? 1 : 0) : kpall);
    const int koff = FT * kpoff;
    const int col0 = blockIdx.x * XP_BN, tok0 = blockIdx.y * BM;
    const int jt = (col0 >> 4) + w;
    const Exl3Off o = exl3_offsets(lane * 8, VARI, W, bits);
    const int r0 = 2 * (lane & 3);
    const int cc = 16 * w + 2 * (lane >> 3) + ((lane & 7) >= 4 ? 1 : 0);
    const size_t tstep = (size_t)nto * W;
    const int xt = tid / XG, xr = (tid % XG) * TM;
    const bool xin = tok0 + xt < T;
    const half* xp = x + (size_t)(xin ? tok0 + xt : 0) * in + (size_t)koff * 16 + xr;
    const uint32_t* tp = trellis + (size_t)jt * W + (size_t)koff * tstep;

    uint32_t wn[4];
    uint2 xn;
    auto fetch = [&]() {                           // k-tiles in order
        exl3_load8(tp, o, VARI, W, wn);
        tp += tstep;
        if (TM == 4) xn = xin ? __ldg(reinterpret_cast<const uint2*>(xp)) : make_uint2(0, 0);
        else if (TM == 2) xn = make_uint2(xin ? __ldg(reinterpret_cast<const uint32_t*>(xp)) : 0u, 0u);
        else xn = make_uint2(xin ? (uint32_t)__ldg(reinterpret_cast<const unsigned short*>(xp)) : 0u, 0u);
        xp += 16;
    };
    auto stage = [&](int buf) {
        half2 fr[4];
        exl3_decode8_w(wn, o, VARI, bits, fr);
        Ws[buf][r0][cc] = __low2half(fr[0]);      Ws[buf][r0 + 1][cc] = __high2half(fr[0]);
        Ws[buf][r0 + 8][cc] = __low2half(fr[1]);  Ws[buf][r0 + 9][cc] = __high2half(fr[1]);
        Ws[buf][r0][cc + 8] = __low2half(fr[2]);  Ws[buf][r0 + 1][cc + 8] = __high2half(fr[2]);
        Ws[buf][r0 + 8][cc + 8] = __low2half(fr[3]); Ws[buf][r0 + 9][cc + 8] = __high2half(fr[3]);
        const half* h = reinterpret_cast<const half*>(&xn);
#pragma unroll
        for (int q = 0; q < TM; q++) Xs[buf][xr + q][xt] = h[q];
    };

    float acc[TM][8];
    half2 hacc[TM][4];
#pragma unroll
    for (int i = 0; i < TM; i++) {
#pragma unroll
        for (int j = 0; j < 8; j++) acc[i][j] = 0.f;
#pragma unroll
        for (int j = 0; j < 4; j++) hacc[i][j] = __float2half2_rn(0.f);
    }
    if (kin > 0) {
        fetch();
        stage(0);
        if (kin > 1) fetch();
    }
    __syncthreads();
    for (int kg = 0; kg < kin; kg += FT) {
#pragma unroll
        for (int u = 0; u < FT; u++) {
            const int kt = kg + u;
            const int buf = u & 1;
            if (kt + 1 < kin) {
                stage(buf ^ 1);
                if (kt + 2 < kin) fetch();
            }
#pragma unroll
            for (int kk = 0; kk < 16; kk++) {
                half2 av[TM], bv[4];
                if (TM == 4) {
                    const uint2 a0 = *reinterpret_cast<const uint2*>(&Xs[buf][kk][ty * TM]);
                    const half2* pa = reinterpret_cast<const half2*>(&a0);
                    av[0] = __low2half2(pa[0]); av[1 % TM] = __high2half2(pa[0]);
                    av[2 % TM] = __low2half2(pa[1]); av[3 % TM] = __high2half2(pa[1]);
                } else if (TM == 2) {
                    const half2 a0 = *reinterpret_cast<const half2*>(&Xs[buf][kk][ty * TM]);
                    av[0] = __low2half2(a0); av[1 % TM] = __high2half2(a0);
                } else {
                    av[0] = __half2half2(Xs[buf][kk][ty]);
                }
                const uint2 b0 = *reinterpret_cast<const uint2*>(&Ws[buf][kk][tx * 4]);
                const uint2 b1 = *reinterpret_cast<const uint2*>(&Ws[buf][kk][64 + tx * 4]);
                const half2* pb0 = reinterpret_cast<const half2*>(&b0);
                const half2* pb1 = reinterpret_cast<const half2*>(&b1);
                bv[0] = pb0[0]; bv[1] = pb0[1]; bv[2] = pb1[0]; bv[3] = pb1[1];
#pragma unroll
                for (int i = 0; i < TM; i++)
#pragma unroll
                    for (int j = 0; j < 4; j++) hacc[i][j] = __hfma2(av[i], bv[j], hacc[i][j]);
            }
            __syncthreads();
        }
#pragma unroll
        for (int i = 0; i < TM; i++)
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const float2 f = __half22float2(hacc[i][j]);
                acc[i][2 * j] += f.x;
                acc[i][2 * j + 1] += f.y;
                hacc[i][j] = __float2half2_rn(0.f);
            }
    }
#pragma unroll
    for (int i = 0; i < TM; i++) {
        const int tok = tok0 + ty * TM + i;
        if (tok >= T) continue;
        if (SPLIT) {
            float* pr = part + ((size_t)sz * T + tok) * out + col0;
            *reinterpret_cast<float4*>(pr + tx * 4) =
                make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
            *reinterpret_cast<float4*>(pr + 64 + tx * 4) =
                make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
            continue;
        }
        half* zr = z + (size_t)tok * out + col0;
        exl3_u2h2 o0, o1;
        o0.h2[0] = __floats2half2_rn(exl3_satf(acc[i][0]), exl3_satf(acc[i][1]));
        o0.h2[1] = __floats2half2_rn(exl3_satf(acc[i][2]), exl3_satf(acc[i][3]));
        o1.h2[0] = __floats2half2_rn(exl3_satf(acc[i][4]), exl3_satf(acc[i][5]));
        o1.h2[1] = __floats2half2_rn(exl3_satf(acc[i][6]), exl3_satf(acc[i][7]));
        *reinterpret_cast<uint2*>(zr + tx * 4) = o0.u;
        *reinterpret_cast<uint2*>(zr + 64 + tx * 4) = o1.u;
    }
}

#define XS_ENTRY(NAME, BM, OCC, SPLIT)                                          \
    extern "C" __global__ void __launch_bounds__(256, OCC)                     \
    NAME(const half* __restrict__ x, const uint32_t* __restrict__ tr,          \
         half* __restrict__ z, float* __restrict__ part, int in, int out,      \
         int T, int bits_x2) {                                                 \
        __shared__ __align__(16) uint32_t smem[16 * (BM) + 16 * XP_BN];        \
        if (in <= 0 || out <= 0 || T <= 0 || (in & 31) || (out & 127)) return; \
        const int bits = bits_x2 >> 1;                                         \
        switch (bits_x2) {                                                     \
        case 4: xs_body<EXL3_V_ALIGN2, BM, SPLIT>(x, tr, z, part, in, out, T, bits, smem); break; \
        case 6: xs_body<EXL3_V_DQ8_3, BM, SPLIT>(x, tr, z, part, in, out, T, bits, smem); break;  \
        case 7: xs_body<EXL3_V_HALF, BM, SPLIT>(x, tr, z, part, in, out, T, bits, smem); break;   \
        case 8: xs_body<EXL3_V_ALIGN4, BM, SPLIT>(x, tr, z, part, in, out, T, bits, smem); break; \
        case 12: xs_body<EXL3_V_DQ4_6, BM, SPLIT>(x, tr, z, part, in, out, T, bits, smem); break; \
        default: break;                                                        \
        }                                                                      \
    }
XS_ENTRY(xs_gemm16, 16, 4, false)
XS_ENTRY(xs_gemm16s, 16, 4, true)
XS_ENTRY(xs_gemm32, 32, 4, false)
XS_ENTRY(xs_gemm32s, 32, 4, true)
XS_ENTRY(xs_gemm64, 64, 3, false)
XS_ENTRY(xs_gemm64s, 64, 3, true)

// ============================================================================
// THE DECODE GEMV WITHOUT THE PER-WEIGHT fp16 ROUND TRIP (exl3_gemv_w4a)
// ============================================================================
// Decode (T <= 4) is ALU-bound on this card, not memory-bound: per weight the
// shipped wide loop spends a 32/clk shift, a 3-XMAD multiply (ptxas never emits
// the 2-XMAD form for w * MUL1), a 32/clk VABSDIFF4, half a PRMT, half an
// HFMA2 and a 32/clk f16->f32 convert before its FFMA (rates measured in
// _build/xg/ipeak.cu). This entry is the same walk and the same windows with
// two changes per weight:
//   * the multiply runs on the FP64 pipe: fma(2^52 + w, M, 2^52 (1 - M)) is
//     2^52 + w*M exactly (w*M < 2^48), so its low word is (w*M) mod 2^32 --
//     checked over all 65536 windows (_build/xg/dpeak.cu);
//   * the codebook's affine map is applied once per output instead of once per
//     weight: v = k_inv * (1024 + s) + k_bias, so sum(v x) = k_inv * sum(s x)
//     + (1024 k_inv + k_bias) * sum(x). VABSDIFF4 accumulates onto 0x4B000000,
//     making s one FADD away (2^23 + s - 2^23, exact).
// The weights are therefore the codebook's values WITHOUT the fp16 rounding the
// shipped decode applies to each of them (<= half an fp16 ulp apart); the
// partial sums, the split, the four-lane reduction and the epilogue are the
// wide entry's. Measured per module (bench_xg.py, T = 1, paired): 1.09-1.18x;
// it must be judged end to end (PPL), not bit for bit.
static __device__ __forceinline__ uint32_t exl3_mul1_d(uint32_t w) {
    const double a = __hiloint2double(0x43300000, (int)w);
    const double r = fma(a, 2212286765.0, -4503599627370496.0 * 2212286764.0);
    return (uint32_t)__double2loint(r);
}

// A lane's eight 16-bit windows (clean, upper half zero), in window order, from
// the words exl3_load8 fetched -- the shipped cores' shifts, without the decode.
// A shifted window is one BFE (ptxas otherwise emits SHR + AND for it).
static __device__ __forceinline__ uint32_t exl3_bfe16(uint32_t v, uint32_t pos) {
    uint32_t d;
    asm("bfe.u32 %0, %1, %2, 16;" : "=r"(d) : "r"(v), "r"(pos));
    return d;
}
static __device__ __forceinline__ void exl3_windows8(const uint32_t (&w)[4], const Exl3Off& o,
                                                     int vari, int bits, uint32_t (&wv)[8]) {
    switch (vari) {
    case EXL3_V_ALIGN2: {
        const uint32_t b = exl3_fshift(w[1], w[0], o.sh[0]);
#pragma unroll
        for (int j = 0; j < 7; j++) wv[j] = exl3_bfe16(b, 2 * (7 - j));
        wv[7] = b & 0xffffu;
        break;
    }
    case EXL3_V_DQ8_3: {
        const uint32_t w7 = exl3_fshift(w[1], w[0], o.sh[0]);
        const uint32_t w3 = exl3_fshift(w[1], w[0], o.sh[1]);
#pragma unroll
        for (int j = 0; j < 3; j++) {
            wv[4 + j] = exl3_bfe16(w7, bits * (3 - j));
            wv[j] = exl3_bfe16(w3, bits * (3 - j));
        }
        wv[7] = w7 & 0xffffu;
        wv[3] = w3 & 0xffffu;
        break;
    }
    case EXL3_V_HALF: {
        const uint32_t w7 = exl3_fshift(w[1], w[0], o.sh[0]);
        const uint32_t w3 = exl3_fshift(w[3], w[2], o.sh[1]);
        constexpr int S4[3] = {11, 7, 4};
#pragma unroll
        for (int j = 0; j < 3; j++) {
            wv[4 + j] = exl3_bfe16(w7, S4[j]);
            wv[j] = exl3_bfe16(w3, S4[j]);
        }
        wv[7] = w7 & 0xffffu;
        wv[3] = w3 & 0xffffu;
        break;
    }
    case EXL3_V_ALIGN4: {
        const uint32_t s = exl3_fshift(w[1], w[0], o.sh[0]);
        const uint32_t b = w[1];
        wv[0] = exl3_bfe16(s, 8); wv[1] = exl3_bfe16(s, 4); wv[2] = s & 0xffffu;
        wv[3] = b >> 16; wv[4] = exl3_bfe16(b, 12);
        wv[5] = exl3_bfe16(b, 8); wv[6] = exl3_bfe16(b, 4); wv[7] = b & 0xffffu;
        break;
    }
    default: {
#pragma unroll
        for (int c = 0; c < 2; c++) {
            const uint32_t a = c ? w[2] : w[0];
            const uint32_t b = c ? w[3] : w[1];
            const int s2 = o.sh[c];
            wv[4 * c + 3] = exl3_fshift(b, a, s2) & 0xffffu;
            wv[4 * c + 2] = exl3_fshift(b, a, s2 + bits) & 0xffffu;
            wv[4 * c + 1] = exl3_fshift(b, a, s2 + 2 * bits) & 0xffffu;
            wv[4 * c + 0] = exl3_fshift(b, a, s2 + 3 * bits) & 0xffffu;
        }
        break;
    }
    }
}

// s of one window (0..1020) as the fp32 whose BITS are s: s * 2^-149, a
// subnormal, exact. The FMAs that take it are the IEEE (non-FTZ) form, whose
// rate on this card equals the FTZ one (_build/xg/dnorm.cu), against x * 2^100,
// so each product is s * x * 2^-49 exactly as a normal number; the epilogue
// scales the sum back by 2^49. Saves the FADD that 2^23 + s - 2^23 cost.
static __device__ __forceinline__ float exl3_s_den(uint32_t w) {
    uint32_t s;
    const uint32_t zero = 0u;
    asm ("vabsdiff4.u32.u32.u32.add %0, %1, %2, %3;"
         : "=r"(s) : "r"(exl3_mul1_d(w)), "r"(zero), "r"(zero));
    return __uint_as_float(s);
}
static __device__ __forceinline__ float exl3_fma_ieee(float a, float b, float c) {
    float d;
    asm ("fma.rn.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
    return d;
}
#define EXL3_X_UP 1.2676506002282294e30f        // 2^100
#define EXL3_ACC_DOWN 562949953421312.0f        // 2^49

template <int M, int VARI, int WT, bool FULLW, bool FULLT, bool XSM = false>
static __device__ __forceinline__ void exl3_gemv_run_a(const half* __restrict__ x,
                                                       const uint32_t* __restrict__ trellis,
                                                       half* __restrict__ z,
                                                       float* __restrict__ part,
                                                       int in, int out, int T, int t0,
                                                       int W, int bits, int kin, int nto,
                                                       int jt, const Exl3Off& o, int koff,
                                                       int split, int zidx = -1) {
    const int lane = threadIdx.x & 31;
    float acc[WT][2][M];
#pragma unroll
    for (int i = 0; i < WT; i++)
#pragma unroll
        for (int u = 0; u < 2; u++)
#pragma unroll
            for (int t = 0; t < M; t++) acc[i][u][t] = 0.f;
    float xsum[M];
#pragma unroll
    for (int t = 0; t < M; t++) xsum[t] = 0.f;
    const int ntok = (t0 + M <= T) ? M : (T - t0);
    const int nleft = (nto - jt < WT) ? (nto - jt) : WT;
    const uint32_t* tile = trellis + (size_t)jt * W + (size_t)koff * nto * W;
    // XSM: x is this block's pre-rotated k-slice in shared memory, a row of
    // kin * 16 halves per token (the fused pre-rotation's prologue wrote it)
    const half2* xbase = XSM ? reinterpret_cast<const half2*>(x) + (lane & 3)
                             : reinterpret_cast<const half2*>(x + (size_t)t0 * in
                                                              + (size_t)koff * 16) + (lane & 3);
    const int xstride = XSM ? kin * 8 : in >> 1;
    const size_t step = (size_t)nto * W;
    constexpr int WC = (VARI == EXL3_V_ALIGN2) ? 16 : (VARI == EXL3_V_DQ8_3) ? 24
                     : (VARI == EXL3_V_HALF) ? 28 : (VARI == EXL3_V_ALIGN4) ? 32 : 48;
    constexpr bool TWO = (VARI == EXL3_V_HALF || VARI == EXL3_V_DQ4_6);
    const uint32_t* pw0 = tile + o.i0[0];
    const uint32_t* pw1 = tile + o.i2[0];
    const uint32_t* pw2 = tile + o.i0[1];
    const uint32_t* pw3 = tile + o.i2[1];
    for (int k = 0; k < kin; k++, tile += step) {
        uint32_t w[WT][4];
        float2 xav[M], xbv[M];
#pragma unroll
        for (int t = 0; t < M; t++) {
            if (FULLT || t < ntok) {
                const half2* xr = xbase + (size_t)t * xstride + k * 8;
                const float2 xa = __half22float2(XSM ? xr[0] : __ldg(xr));
                const float2 xb = __half22float2(XSM ? xr[4] : __ldg(xr + 4));
                xsum[t] += (xa.x + xa.y) + (xb.x + xb.y);
                xav[t] = make_float2(xa.x * EXL3_X_UP, xa.y * EXL3_X_UP);
                xbv[t] = make_float2(xb.x * EXL3_X_UP, xb.y * EXL3_X_UP);
            }
        }
        if (FULLW) {
            // per-lane word pointers, the WT tiles at compile-time offsets
#pragma unroll
            for (int i = 0; i < WT; i++) {
                w[i][0] = __ldg(pw0 + i * WC);
                w[i][1] = __ldg(pw1 + i * WC);
                if (TWO) { w[i][2] = __ldg(pw2 + i * WC); w[i][3] = __ldg(pw3 + i * WC); }
                else { w[i][2] = w[i][3] = 0u; }
            }
            pw0 += step; pw1 += step;
            if (TWO) { pw2 += step; pw3 += step; }
        } else {
#pragma unroll
            for (int i = 0; i < WT; i++)
                exl3_load8(tile + (size_t)(i < nleft ? i : 0) * W, o, VARI, W, w[i]);
        }
#pragma unroll
        for (int i = 0; i < WT; i++) {
            uint32_t wv[8];
            exl3_windows8(w[i], o, VARI, bits, wv);
            float f[8];
#pragma unroll
            for (int q = 0; q < 8; q++) f[q] = exl3_s_den(wv[q]);
#pragma unroll
            for (int t = 0; t < M; t++) {
                if (FULLT || t < ntok) {
                    const float2 xa = xav[t], xb = xbv[t];
                    acc[i][0][t] = exl3_fma_ieee(f[0], xa.x, exl3_fma_ieee(f[1], xa.y,
                                   exl3_fma_ieee(f[2], xb.x, exl3_fma_ieee(f[3], xb.y, acc[i][0][t]))));
                    acc[i][1][t] = exl3_fma_ieee(f[4], xa.x, exl3_fma_ieee(f[5], xa.y,
                                   exl3_fma_ieee(f[6], xb.x, exl3_fma_ieee(f[7], xb.y, acc[i][1][t]))));
                }
            }
        }
    }
    const float kinv = __half2float(__ushort_as_half((unsigned short)EXL3_K_INV_BITS)) * EXL3_ACC_DOWN;
    const float kb = 1024.f * __half2float(__ushort_as_half((unsigned short)EXL3_K_INV_BITS))
                     + __half2float(__ushort_as_half((unsigned short)EXL3_K_BIAS_BITS));
    const int A = lane & 7;
    half* zrow = z + (size_t)t0 * out + (size_t)jt * 16;
    const int c = 2 * (lane >> 3) + (A >= 4 ? 1 : 0);
    // the split's slice: blockIdx.z, or the caller's (a launch of several modules)
    const int zs = zidx < 0 ? (int)blockIdx.z : zidx;
    float* prow = split ? part + ((size_t)zs * T + t0) * out + (size_t)jt * 16 : part;
#pragma unroll
    for (int i = 0; i < WT; i++) {
        if (jt + i < nto) {
#pragma unroll
            for (int t = 0; t < M; t++) {
                if (FULLT || t < ntok) {
                    float s0 = acc[i][0][t], s1 = acc[i][1][t], xs = xsum[t];
                    s0 += __shfl_xor_sync(0xffffffffu, s0, 1);
                    s0 += __shfl_xor_sync(0xffffffffu, s0, 2);
                    s1 += __shfl_xor_sync(0xffffffffu, s1, 1);
                    s1 += __shfl_xor_sync(0xffffffffu, s1, 2);
                    xs += __shfl_xor_sync(0xffffffffu, xs, 1);
                    xs += __shfl_xor_sync(0xffffffffu, xs, 2);
                    s0 = fmaf(kinv, s0, kb * xs);
                    s1 = fmaf(kinv, s1, kb * xs);
                    if (A == 0 || A == 4) {
                        if (split) {
                            float* p = prow + (size_t)t * out + (size_t)i * 16;
                            p[c] = s0;
                            p[c + 8] = s1;
                        } else {
                            half* row = zrow + (size_t)t * out + (size_t)i * 16;
                            row[c] = exl3_sat(s0);
                            row[c + 8] = exl3_sat(s1);
                        }
                    }
                }
            }
        }
    }
}

// Same grid, arguments and split as exl3_gemv_w4; m = 1, 2 or 4.
extern "C" __global__ void __launch_bounds__(EXL3_THREADS, EXL3_GEMV_OCC)
exl3_gemv_w4a(const half* __restrict__ x, const uint32_t* __restrict__ trellis,
              half* __restrict__ z, float* __restrict__ part, int in, int out,
              int T, int bits_x2, int m) {
    constexpr int WT = 4;
    if (in <= 0 || out <= 0 || T <= 0 || (in & 15) || (out & 15)) return;
    int vari;
    switch (bits_x2) {
    case 4: vari = EXL3_V_ALIGN2; break;
    case 6: vari = EXL3_V_DQ8_3; break;
    case 7: vari = EXL3_V_HALF; break;
    case 8: vari = EXL3_V_ALIGN4; break;
    case 12: vari = EXL3_V_DQ4_6; break;
    default: return;
    }
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int W = bits_x2 * 4, kall = in >> 4, nto = out >> 4;
    const int jt = (blockIdx.x * EXL3_WARPS + w) * WT;
    if (jt >= nto) return;
    const int t0 = blockIdx.y * m;
    if (t0 >= T) return;
    const int bits = bits_x2 >> 1;
    const Exl3Off o = exl3_offsets(lane * 8, vari, W, bits);
    const int S = gridDim.z, s = blockIdx.z;
    const int kslice = kall / S;
    const int koff = s * kslice + min(s, kall % S);
    const int kspan = kslice + (s < kall % S ? 1 : 0);
    if (kspan <= 0) return;
    // FULLT: the whole token group is live (every decode step and every
    // verify), so the per-token guard is compiled out instead of branching.
#define EXL3_A_MF(VARI, FW)                                                         \
    switch (m) {                                                                    \
    case 1: exl3_gemv_run_a<1, VARI, WT, FW, true>(x, trellis, z, part, in, out, T, t0, W, \
                                         bits, kspan, nto, jt, o, koff, S > 1); break; \
    case 2: if (t0 + 2 <= T) exl3_gemv_run_a<2, VARI, WT, FW, true>(x, trellis, z, part, in, \
                out, T, t0, W, bits, kspan, nto, jt, o, koff, S > 1);                 \
            else exl3_gemv_run_a<2, VARI, WT, FW, false>(x, trellis, z, part, in, out, T, \
                t0, W, bits, kspan, nto, jt, o, koff, S > 1); break;                  \
    case 3: if (t0 + 3 <= T) exl3_gemv_run_a<3, VARI, WT, FW, true>(x, trellis, z, part, in, \
                out, T, t0, W, bits, kspan, nto, jt, o, koff, S > 1);                 \
            else exl3_gemv_run_a<3, VARI, WT, FW, false>(x, trellis, z, part, in, out, T, \
                t0, W, bits, kspan, nto, jt, o, koff, S > 1); break;                  \
    case 4: exl3_gemv_run_a<4, VARI, WT, FW, false>(x, trellis, z, part, in, out, T, t0, W, \
                                         bits, kspan, nto, jt, o, koff, S > 1); break; \
    default: break;                                                                 \
    }
    // the ragged last column group (nto % WT != 0) keeps the clamped loads
#define EXL3_A_M(VARI)                                                              \
    if (jt + WT <= nto) { EXL3_A_MF(VARI, true) } else { EXL3_A_MF(VARI, false) }
    switch (vari) {
    case EXL3_V_ALIGN2: EXL3_A_M(EXL3_V_ALIGN2); break;
    case EXL3_V_DQ8_3: EXL3_A_M(EXL3_V_DQ8_3); break;
    case EXL3_V_HALF: EXL3_A_M(EXL3_V_HALF); break;
    case EXL3_V_ALIGN4: EXL3_A_M(EXL3_V_ALIGN4); break;
    default: EXL3_A_M(EXL3_V_DQ4_6); break;
    }
#undef EXL3_A_M
#undef EXL3_A_MF
}

// The same entry for m = 1 only, at 3 blocks per SM (A/B probe).
#ifndef EXL3_A1_OCC
#define EXL3_A1_OCC 3
#endif
extern "C" __global__ void __launch_bounds__(EXL3_THREADS, EXL3_A1_OCC)
exl3_gemv_w4a1(const half* __restrict__ x, const uint32_t* __restrict__ trellis,
              half* __restrict__ z, float* __restrict__ part, int in, int out,
              int T, int bits_x2, int m) {
    constexpr int WT = 4;
    if (in <= 0 || out <= 0 || T <= 0 || (in & 15) || (out & 15)) return;
    int vari;
    switch (bits_x2) {
    case 4: vari = EXL3_V_ALIGN2; break;
    case 6: vari = EXL3_V_DQ8_3; break;
    case 7: vari = EXL3_V_HALF; break;
    case 8: vari = EXL3_V_ALIGN4; break;
    case 12: vari = EXL3_V_DQ4_6; break;
    default: return;
    }
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int W = bits_x2 * 4, kall = in >> 4, nto = out >> 4;
    const int jt = (blockIdx.x * EXL3_WARPS + w) * WT;
    if (jt >= nto) return;
    const int t0 = blockIdx.y * m;
    if (t0 >= T) return;
    const int bits = bits_x2 >> 1;
    const Exl3Off o = exl3_offsets(lane * 8, vari, W, bits);
    const int S = gridDim.z, s = blockIdx.z;
    const int kslice = kall / S;
    const int koff = s * kslice + min(s, kall % S);
    const int kspan = kslice + (s < kall % S ? 1 : 0);
    if (kspan <= 0) return;
    // FULLT: the whole token group is live (every decode step and every
    // verify), so the per-token guard is compiled out instead of branching.
#define EXL3_A_MF(VARI, FW)                                                         \
    switch (m) {                                                                    \
    case 1: exl3_gemv_run_a<1, VARI, WT, FW, true>(x, trellis, z, part, in, out, T, t0, W, \
                                         bits, kspan, nto, jt, o, koff, S > 1); break; \
    default: break;                                                                 \
    }
    // the ragged last column group (nto % WT != 0) keeps the clamped loads
#define EXL3_A_M(VARI)                                                              \
    if (jt + WT <= nto) { EXL3_A_MF(VARI, true) } else { EXL3_A_MF(VARI, false) }
    switch (vari) {
    case EXL3_V_ALIGN2: EXL3_A_M(EXL3_V_ALIGN2); break;
    case EXL3_V_DQ8_3: EXL3_A_M(EXL3_V_DQ8_3); break;
    case EXL3_V_HALF: EXL3_A_M(EXL3_V_HALF); break;
    case EXL3_V_ALIGN4: EXL3_A_M(EXL3_V_ALIGN4); break;
    default: EXL3_A_M(EXL3_V_DQ4_6); break;
    }
#undef EXL3_A_M
#undef EXL3_A_MF
}


// ============================================================================
// THE DECODE GEMV WITH ITS POST-ROTATION FUSED (exl3_gemv_w4af / w4a1f)
// ============================================================================
// exl3_gemv_w4a(1) followed by had128_post_sk in ONE launch. Every block writes
// its fp32 partial (also at S = 1); the LAST of a column group's S blocks to
// finish -- found by a per-(column group, token group) counter that the last
// block resets, so nothing is cleared between launches -- sums the S partials in
// split order, rounds once and post-rotates exactly as had128_post_sk does, so
// the output is bit-identical to the two-launch path. Which block is last
// changes nothing but who does the work. Removes one launch (and its gap) per
// projection: 400 a decode step. Needs out a multiple of 512 (a block's
// 4 x 128-column rotation blocks); every module here is.
template <int WT, int MAXM>
static __device__ __forceinline__ void exl3_post_tail(const float* __restrict__ part,
                                                      const half* __restrict__ svh,
                                                      half* __restrict__ y,
                                                      int* __restrict__ cnt,
                                                      int out, int T, int t0, int ntok) {
    __shared__ int last;
    const int S = gridDim.z;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        int* c = cnt + blockIdx.y * gridDim.x + blockIdx.x;
        const int prev = (S > 1) ? atomicAdd(c, 1) : 0;
        last = (prev == S - 1);
        if (last && S > 1) *c = 0;
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    constexpr int NHB = EXL3_WARPS * WT * 16 / 128;   // rotation blocks a block owns
    const size_t plane = (size_t)T * out;
    for (int job = w; job < NHB * ntok; job += EXL3_WARPS) {
        const int hb = job % NHB, t = job / NHB;
        const int col0 = (blockIdx.x * EXL3_WARPS * WT) * 16 + hb * 128;
        if (col0 >= out) continue;
        const size_t base = (size_t)(t0 + t) * out + col0;
        const float4* pp = reinterpret_cast<const float4*>(part + base) + lane;
        const size_t p4 = plane >> 2;
        float4 a = __ldcg(pp);
        int j = 1;
        for (; j + 4 <= S; j += 4) {         // four in flight, summed in order
            float4 b[4];
#pragma unroll
            for (int u = 0; u < 4; u++) b[u] = __ldcg(pp + (size_t)(j + u) * p4);
#pragma unroll
            for (int u = 0; u < 4; u++) { a.x += b[u].x; a.y += b[u].y; a.z += b[u].z; a.w += b[u].w; }
        }
        for (; j < S; j++) {
            const float4 b = __ldcg(pp + (size_t)j * p4);
            a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
        }
        float v[4] = { __half2float(exl3_sat(a.x)), __half2float(exl3_sat(a.y)),
                       __half2float(exl3_sat(a.z)), __half2float(exl3_sat(a.w)) };
        exl3_u2h2 sv;
        sv.u = __ldg(reinterpret_cast<const uint2*>(svh + col0) + lane);
        const float sc[4] = { __half2float(__low2half(sv.h2[0])), __half2float(__high2half(sv.h2[0])),
                              __half2float(__low2half(sv.h2[1])), __half2float(__high2half(sv.h2[1])) };
        exl3_fwt128(v, lane);
#pragma unroll
        for (int u = 0; u < 4; u++) v[u] *= EXL3_RSCALE * sc[u];
        exl3_u2h2 o2;
        o2.h2[0] = __floats2half2_rn(exl3_satf(v[0]), exl3_satf(v[1]));
        o2.h2[1] = __floats2half2_rn(exl3_satf(v[2]), exl3_satf(v[3]));
        *reinterpret_cast<uint2*>(y + base + 4 * (size_t)lane) = o2.u;
    }
}

#define EXL3_AF_ENTRY(NAME, OCC, MSET)                                              \
extern "C" __global__ void __launch_bounds__(EXL3_THREADS, OCC)                     \
NAME(const half* __restrict__ x, const uint32_t* __restrict__ trellis,              \
     float* __restrict__ part, const half* __restrict__ svh, half* __restrict__ y,  \
     int* __restrict__ cnt, int in, int out, int T, int bits_x2, int m) {           \
    constexpr int WT = 4;                                                           \
    if (in <= 0 || out <= 0 || T <= 0 || (in & 15) || (out & 511)) return;         \
    int vari;                                                                       \
    switch (bits_x2) {                                                              \
    case 4: vari = EXL3_V_ALIGN2; break;                                            \
    case 6: vari = EXL3_V_DQ8_3; break;                                             \
    case 7: vari = EXL3_V_HALF; break;                                              \
    case 8: vari = EXL3_V_ALIGN4; break;                                            \
    case 12: vari = EXL3_V_DQ4_6; break;                                            \
    default: return;                                                                \
    }                                                                               \
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;                        \
    const int W = bits_x2 * 4, kall = in >> 4, nto = out >> 4;                      \
    const int jt = (blockIdx.x * EXL3_WARPS + w) * WT;                              \
    const int t0 = blockIdx.y * m;                                                  \
    if (t0 >= T) return;                                                            \
    const int ntok = (t0 + m <= T) ? m : (T - t0);                                  \
    const int bits = bits_x2 >> 1;                                                  \
    const Exl3Off o = exl3_offsets(lane * 8, vari, W, bits);                        \
    const int S = gridDim.z, s = blockIdx.z;                                        \
    const int kslice = kall / S;                                                    \
    const int koff = s * kslice + min(s, kall % S);                                 \
    const int kspan = kslice + (s < kall % S ? 1 : 0);                              \
    half* z = nullptr;                                                              \
    if (jt < nto && kspan > 0) {                                                    \
        MSET                                                                        \
    }                                                                               \
    exl3_post_tail<WT, 4>(part, svh, y, cnt, out, T, t0, ntok);                     \
}
#define EXL3_AF_CALL(M, VARI, FT)                                                   \
    exl3_gemv_run_a<M, VARI, WT, true, FT>(x, trellis, z, part, in, out, T, t0, W,  \
                                           bits, kspan, nto, jt, o, koff, 1)
#define EXL3_AF_VARI(MCASES)                                                        \
    switch (vari) {                                                                 \
    case EXL3_V_ALIGN2: { constexpr int V_ = EXL3_V_ALIGN2; MCASES } break;          \
    case EXL3_V_DQ8_3: { constexpr int V_ = EXL3_V_DQ8_3; MCASES } break;            \
    case EXL3_V_HALF: { constexpr int V_ = EXL3_V_HALF; MCASES } break;              \
    case EXL3_V_ALIGN4: { constexpr int V_ = EXL3_V_ALIGN4; MCASES } break;          \
    default: { constexpr int V_ = EXL3_V_DQ4_6; MCASES } break;                     \
    }
// m = 1 only, 3 blocks an SM (the decode step and both heads)
EXL3_AF_ENTRY(exl3_gemv_w4a1f, EXL3_A1_OCC,
              EXL3_AF_VARI(if (m == 1) EXL3_AF_CALL(1, V_, true);))
// m = 2 and 4 (the MTP verify, and T = 3/4 calls)
EXL3_AF_ENTRY(exl3_gemv_w4af, EXL3_GEMV_OCC,
              EXL3_AF_VARI(if (m == 2) { if (ntok == 2) EXL3_AF_CALL(2, V_, true);
                                         else EXL3_AF_CALL(2, V_, false); }
                           else if (m == 3) { if (ntok == 3) EXL3_AF_CALL(3, V_, true);
                                              else EXL3_AF_CALL(3, V_, false); }
                           else if (m == 4) EXL3_AF_CALL(4, V_, false);
                           else if (m == 1) EXL3_AF_CALL(1, V_, true);))


// ============================================================================
// ... AND THE PRE-ROTATION IN THE PROLOGUE (exl3_gemv_w4a1fp / w4afp)
// ============================================================================
// had128_pre folded in as well: each block rotates the 128-blocks of x that
// its k-slice touches (one warp each, had128_body's exact arithmetic: scale by
// suh, the butterfly, the 1/sqrt(128), the fp16 rounding) into shared memory,
// and the k-loop reads x from there. The values are had128_pre's, so the
// output is bit-identical to had128_pre + exl3_gemv_w4a(1)f; one launch per
// projection instead of two. Dynamic shared memory: m * kspan_max * 32 bytes.
// `xup` non-null: x is silu(xin) * xup, rounded to fp16 exactly as kernels.cu's
// silu_mul stores it (the MLP's down_proj input), so silu_mul's launch goes too.
static __device__ __forceinline__ float exl3_silu(float x) {
    return __fdiv_rn(x, 1.f + __expf(-x));
}
static __device__ __forceinline__ void exl3_pre_prologue(const half* __restrict__ xin,
                                                         const half* __restrict__ xup,
                                                         const half* __restrict__ suh,
                                                         half* __restrict__ xs,
                                                         int in, int t0, int ntok,
                                                         int koff, int kspan) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int e0 = koff * 16, e1 = (koff + kspan) * 16;   // the slice, in elements
    const int hb0 = e0 >> 7, hb1 = (e1 - 1) >> 7;
    const int nhb = hb1 - hb0 + 1;
    const int ld = kspan * 16;
    for (int job = w; job < nhb * ntok; job += EXL3_WARPS) {
        const int hb = hb0 + job % nhb, t = job / nhb;
        const size_t base = (size_t)(t0 + t) * in + (size_t)hb * 128;
        exl3_u2h2 xv, sv;
        xv.u = __ldg(reinterpret_cast<const uint2*>(xin + base) + lane);
        sv.u = __ldg(reinterpret_cast<const uint2*>(suh + (size_t)hb * 128) + lane);
        if (xup != nullptr) {
            exl3_u2h2 uv;
            uv.u = __ldg(reinterpret_cast<const uint2*>(xup + base) + lane);
#pragma unroll
            for (int q = 0; q < 2; q++) {
                const float2 g = __half22float2(xv.h2[q]);
                const float2 u = __half22float2(uv.h2[q]);
                xv.h2[q] = __floats2half2_rn(exl3_silu(g.x) * u.x, exl3_silu(g.y) * u.y);
            }
        }
        float v[4] = { __half2float(__low2half(xv.h2[0])), __half2float(__high2half(xv.h2[0])),
                       __half2float(__low2half(xv.h2[1])), __half2float(__high2half(xv.h2[1])) };
        const float sc[4] = { __half2float(__low2half(sv.h2[0])), __half2float(__high2half(sv.h2[0])),
                              __half2float(__low2half(sv.h2[1])), __half2float(__high2half(sv.h2[1])) };
#pragma unroll
        for (int u = 0; u < 4; u++) v[u] *= sc[u];
        exl3_fwt128(v, lane);
#pragma unroll
        for (int u = 0; u < 4; u++) v[u] *= EXL3_RSCALE;
        exl3_u2h2 o2;
        o2.h2[0] = __floats2half2_rn(exl3_satf(v[0]), exl3_satf(v[1]));
        o2.h2[1] = __floats2half2_rn(exl3_satf(v[2]), exl3_satf(v[3]));
        const int e = hb * 128 + 4 * lane;            // this lane's 4 elements
        if (e >= e0 && e + 4 <= e1)
            *reinterpret_cast<uint2*>(xs + (size_t)t * ld + (e - e0)) = o2.u;
    }
    __syncthreads();
}

#define EXL3_AFP_ENTRY(NAME, OCC, ...)                                              \
extern "C" __global__ void __launch_bounds__(EXL3_THREADS, OCC)                     \
NAME(const half* __restrict__ xin, const half* __restrict__ xup,                    \
     const half* __restrict__ suh,                                                  \
     const uint32_t* __restrict__ trellis, float* __restrict__ part,                \
     const half* __restrict__ svh, half* __restrict__ y, int* __restrict__ cnt,     \
     int in, int out, int T, int bits_x2, int m) {                                  \
    constexpr int WT = 4;                                                           \
    extern __shared__ __align__(16) half xs_dyn[];                                  \
    if (in <= 0 || out <= 0 || T <= 0 || (in & 127) || (out % (EXL3_WARPS * WT * 16))) return; \
    int vari;                                                                       \
    switch (bits_x2) {                                                              \
    case 4: vari = EXL3_V_ALIGN2; break;                                            \
    case 6: vari = EXL3_V_DQ8_3; break;                                             \
    case 7: vari = EXL3_V_HALF; break;                                              \
    case 8: vari = EXL3_V_ALIGN4; break;                                            \
    case 12: vari = EXL3_V_DQ4_6; break;                                            \
    default: return;                                                                \
    }                                                                               \
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;                        \
    const int W = bits_x2 * 4, kall = in >> 4, nto = out >> 4;                      \
    const int jt = (blockIdx.x * EXL3_WARPS + w) * WT;                              \
    const int t0 = blockIdx.y * m;                                                  \
    if (t0 >= T) return;                                                            \
    const int ntok = (t0 + m <= T) ? m : (T - t0);                                  \
    const int bits = bits_x2 >> 1;                                                  \
    const Exl3Off o = exl3_offsets(lane * 8, vari, W, bits);                        \
    const int S = gridDim.z, s = blockIdx.z;                                        \
    const int kslice = kall / S;                                                    \
    const int koff = s * kslice + min(s, kall % S);                                 \
    const int kspan = kslice + (s < kall % S ? 1 : 0);                              \
    if (kspan > 0) exl3_pre_prologue(xin, xup, suh, xs_dyn, in, t0, ntok, koff, kspan); \
    const half* x = xs_dyn;                                                         \
    half* z = nullptr;                                                              \
    if (jt < nto && kspan > 0) {                                                    \
        __VA_ARGS__                                                                 \
    }                                                                               \
    exl3_post_tail<WT, 4>(part, svh, y, cnt, out, T, t0, ntok);                     \
}
#define EXL3_AFP_CALL(M, VARI, FT)                                                  \
    exl3_gemv_run_a<M, VARI, WT, true, FT, true>(x, trellis, z, part, in, out, T,   \
                                                 t0, W, bits, kspan, nto, jt, o, koff, 1)
EXL3_AFP_ENTRY(exl3_gemv_w4a1fp, EXL3_A1_OCC,
               EXL3_AF_VARI(if (m == 1) EXL3_AFP_CALL(1, V_, true);))
EXL3_AFP_ENTRY(exl3_gemv_w4afp, EXL3_GEMV_OCC,
               EXL3_AF_VARI(if (m == 2) { if (ntok == 2) EXL3_AFP_CALL(2, V_, true);
                                          else EXL3_AFP_CALL(2, V_, false); }
                            else if (m == 3) { if (ntok == 3) EXL3_AFP_CALL(3, V_, true);
                                               else EXL3_AFP_CALL(3, V_, false); }
                            else if (m == 4) EXL3_AFP_CALL(4, V_, false);
                            else if (m == 1) EXL3_AFP_CALL(1, V_, true);))
// m = 3 alone (the K = 2 verify's full group of 3), at 2 and at 3 blocks an SM:
// the m-general entry above carries every group size's code and runs the same
// group 1.03x slower; which occupancy wins depends on the module's shape
// (runtime M3_OCC3_MIN_IN). Same arithmetic, so the same bits.
EXL3_AFP_ENTRY(exl3_gemv_w4a3fp, EXL3_GEMV_OCC,
               EXL3_AF_VARI(if (m == 3 && ntok == 3) EXL3_AFP_CALL(3, V_, true);))
EXL3_AFP_ENTRY(exl3_gemv_w4a3fp3, 3,
               EXL3_AF_VARI(if (m == 3 && ntok == 3) EXL3_AFP_CALL(3, V_, true);))

// ============================================================================
// SEVERAL PROJECTIONS OF ONE INPUT IN ONE LAUNCH (exl3_gemv_w4a1fpn / w4a3fpn)
// ============================================================================
// exl3_gemv_w4a1fp / w4a3fp for up to three modules that read the same x (in_proj_qkv and
// in_proj_z; q_proj, k_proj and v_proj): blockIdx.x walks module 0's blocks,
// then module 1's, then module 2's, each with ITS OWN k-split, partials, counters
// and rotations -- every block does exactly what it does in the module's own
// launch, so the outputs are bit-identical to the separate launches. What goes
// is the launches' gaps and their partial last waves (k_proj and v_proj ran
// 64 blocks each, in_proj_z 192 = 1.14 waves).
struct Exl3Fm {                 // one module of a multi-module launch
    const half* suh; const uint32_t* trellis; const half* svh; half* y; float* part;
    int* cnt; int out, bits_x2, S;
};
// the last block of a column group's S: exl3_post_tail with the group and split
// count given instead of read from the grid
template <int WT>
static __device__ __forceinline__ void exl3_post_tail_n(const float* __restrict__ part,
                                                        const half* __restrict__ svh,
                                                        half* __restrict__ y,
                                                        int* __restrict__ c, int S,
                                                        int cg, int out, int T, int ntok) {
    __shared__ int last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        const int prev = (S > 1) ? atomicAdd(c, 1) : 0;
        last = (prev == S - 1);
        if (last && S > 1) *c = 0;
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    constexpr int NHB = EXL3_WARPS * WT * 16 / 128;
    const size_t plane = (size_t)T * out;
    for (int job = w; job < NHB * ntok; job += EXL3_WARPS) {
        const int hb = job % NHB, t = job / NHB;
        const int col0 = (cg * EXL3_WARPS * WT) * 16 + hb * 128;
        if (col0 >= out) continue;
        const size_t base = (size_t)t * out + col0;
        const float4* pp = reinterpret_cast<const float4*>(part + base) + lane;
        const size_t p4 = plane >> 2;
        float4 a = __ldcg(pp);
        int j = 1;
        for (; j + 4 <= S; j += 4) {
            float4 b[4];
#pragma unroll
            for (int u = 0; u < 4; u++) b[u] = __ldcg(pp + (size_t)(j + u) * p4);
#pragma unroll
            for (int u = 0; u < 4; u++) { a.x += b[u].x; a.y += b[u].y; a.z += b[u].z; a.w += b[u].w; }
        }
        for (; j < S; j++) {
            const float4 b = __ldcg(pp + (size_t)j * p4);
            a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
        }
        float v[4] = { __half2float(exl3_sat(a.x)), __half2float(exl3_sat(a.y)),
                       __half2float(exl3_sat(a.z)), __half2float(exl3_sat(a.w)) };
        exl3_u2h2 sv;
        sv.u = __ldg(reinterpret_cast<const uint2*>(svh + col0) + lane);
        const float sc[4] = { __half2float(__low2half(sv.h2[0])), __half2float(__high2half(sv.h2[0])),
                              __half2float(__low2half(sv.h2[1])), __half2float(__high2half(sv.h2[1])) };
        exl3_fwt128(v, lane);
#pragma unroll
        for (int u = 0; u < 4; u++) v[u] *= EXL3_RSCALE * sc[u];
        exl3_u2h2 o2;
        o2.h2[0] = __floats2half2_rn(exl3_satf(v[0]), exl3_satf(v[1]));
        o2.h2[1] = __floats2half2_rn(exl3_satf(v[2]), exl3_satf(v[3]));
        *reinterpret_cast<uint2*>(y + base + 4 * (size_t)lane) = o2.u;
    }
}

// grid (sum of the modules' column groups x splits [+ ab.out * M], 1, 1); `in` shared; one group
// of M = 1 (the decode step) or 3 (the K = 2 verify) tokens.
// Dynamic shared memory: M * the largest module's kspan_max * 32 bytes.
// The fp16 in_proj_a / in_proj_b rows (gemv_ab_f32's work) as extra blocks
// after the trellis modules': two rows a block, one a 128-thread half, each
// half gemv_ab_f32's loop and reduction exactly. `ab` null: none.
struct Exl3Ab { const half* wa; const half* wb; float* ya; float* yb; int out; };
template <int M>
static __device__ __forceinline__ void exl3_ab_rows(const half* __restrict__ x, const Exl3Ab& ab,
                                                    int in, int e) {
    __shared__ float red[8];
    const int h = threadIdx.x >> 7, tl = threadIdx.x & 127;
    const int row = 2 * (e % ab.out) + h, t = e / ab.out;   // rows 0..2 out - 1, then tokens
    const bool second = row >= ab.out;
    const int o = second ? row - ab.out : row;
    const uint4* wr = reinterpret_cast<const uint4*>((second ? ab.wb : ab.wa) + (size_t)o * in);
    const uint4* xr = reinterpret_cast<const uint4*>(x + (size_t)t * in);
    float acc = 0.f;
    for (int i = tl; i < (in >> 3); i += 128) {
        const uint4 xv = __ldg(&xr[i]);
        const uint4 wv = __ldg(&wr[i]);
        const half2* xh = reinterpret_cast<const half2*>(&xv);
        const half2* wh = reinterpret_cast<const half2*>(&wv);
#pragma unroll
        for (int q = 0; q < 4; q++) {
            const float2 a = __half22float2(xh[q]);
            const float2 b = __half22float2(wh[q]);
            acc = fmaf(a.x, b.x, acc);
            acc = fmaf(a.y, b.y, acc);
        }
    }
#pragma unroll
    for (int s = 16; s > 0; s >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, s);
    if ((tl & 31) == 0) red[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (tl == 0)
        (second ? ab.yb : ab.ya)[(size_t)t * ab.out + o] =
            (red[4 * h] + red[4 * h + 1]) + (red[4 * h + 2] + red[4 * h + 3]);
}

template <int M>
static __device__ __forceinline__ void exl3_gemv_fpn_body(const half* __restrict__ xin,
                                                          int in, int nmod, const Exl3Fm& m0,
                                                          const Exl3Fm& m1, const Exl3Fm& m2,
                                                          const Exl3Ab& ab) {
    constexpr int WT = 4;
    extern __shared__ __align__(16) half xs_dynn[];
    int b = blockIdx.x;
    const int bx0 = -(-(m0.out >> 4) / (EXL3_WARPS * WT));
    const int bx1 = -(-(m1.out >> 4) / (EXL3_WARPS * WT));
    const int bx2 = -(-(m2.out >> 4) / (EXL3_WARPS * WT));
    const int nbt = bx0 * m0.S + bx1 * m1.S + (nmod > 2 ? bx2 * m2.S : 0);
    if (b >= nbt) {                                   // the a / b rows
        if (ab.wa != nullptr && b - nbt < ab.out * M) exl3_ab_rows<M>(xin, ab, in, b - nbt);
        return;
    }
    Exl3Fm f = m0;
    int bx = bx0;
    if (b >= bx0 * m0.S) {
        b -= bx0 * m0.S; f = m1; bx = bx1;
        if (nmod > 2 && b >= bx1 * m1.S) { b -= bx1 * m1.S; f = m2; bx = bx2; }
    }
    if (in <= 0 || (in & 127) || f.out <= 0 || (f.out % (EXL3_WARPS * WT * 16))) return;
    int vari;
    switch (f.bits_x2) {
    case 4: vari = EXL3_V_ALIGN2; break;
    case 6: vari = EXL3_V_DQ8_3; break;
    case 7: vari = EXL3_V_HALF; break;
    case 8: vari = EXL3_V_ALIGN4; break;
    case 12: vari = EXL3_V_DQ4_6; break;
    default: return;
    }
    const int cg = b % bx, s = b / bx, S = f.S;
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int W = f.bits_x2 * 4, kall = in >> 4, nto = f.out >> 4;
    const int jt = (cg * EXL3_WARPS + w) * WT;
    const int T = M, t0 = 0, out = f.out;
    const int bits = f.bits_x2 >> 1;
    const Exl3Off o = exl3_offsets(lane * 8, vari, W, bits);
    const int kslice = kall / S;
    const int koff = s * kslice + min(s, kall % S);
    const int kspan = kslice + (s < kall % S ? 1 : 0);
    if (kspan > 0) exl3_pre_prologue(xin, nullptr, f.suh, xs_dynn, in, t0, M, koff, kspan);
    const half* x = xs_dynn;
    const uint32_t* trellis = f.trellis;
    float* part = f.part;
    half* z = nullptr;
    if (jt < nto && kspan > 0) {
#define EXL3_FN_CALL exl3_gemv_run_a<M, V_, WT, true, true, true>(x, trellis, z, part, in, \
                     out, T, t0, W, bits, kspan, nto, jt, o, koff, 1, s);
        EXL3_AF_VARI(EXL3_FN_CALL)
#undef EXL3_FN_CALL
    }
    exl3_post_tail_n<WT>(part, f.svh, f.y, f.cnt + cg, S, cg, out, T, M);
}
extern "C" __global__ void __launch_bounds__(EXL3_THREADS, EXL3_A1_OCC)
exl3_gemv_w4a1fpn(const half* __restrict__ xin, int in, int nmod, Exl3Fm m0, Exl3Fm m1,
                  Exl3Fm m2, Exl3Ab ab) {
    exl3_gemv_fpn_body<1>(xin, in, nmod, m0, m1, m2, ab);
}
extern "C" __global__ void __launch_bounds__(EXL3_THREADS, EXL3_GEMV_OCC)
exl3_gemv_w4a3fpn(const half* __restrict__ xin, int in, int nmod, Exl3Fm m0, Exl3Fm m1,
                  Exl3Fm m2, Exl3Ab ab) {
    exl3_gemv_fpn_body<3>(xin, in, nmod, m0, m1, m2, ab);
}
