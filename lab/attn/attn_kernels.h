// Inkling 2 Mini attention lab kernels for gfx1151 (wave32, WMMA f16). MIT.
// Exact math: notes/B-forward-spec.md 2.2 C-D. Layout: K and V caches are f16 [cell][8 kv heads][128] (one cell = one
// 2 KiB row per tensor = one Gufo kAppendRows/kWindowRows row). Cell of position p is p % cap (cap >= max ctx for global
// layers, >= 1024 + max ubatch for SWA ring layers).
//
//   kv_write   : raw K/V (post-conv, f32) -> L2clip(K) / RMS(V) -> f16 cache
//   qr_prep    : raw Q/R -> clip, (global: *tau), Q also * 1/sqrt(128) -> Qn f16, Rn f32      (prefill)
//   attn_decode<NQ> + decode_reduce : batch 1..4 queries, split-K, fused Q/R prologue, sink, gate optional
//   attn_prefill<NTT> : flash style, WMMA f16, no materialised mask, bias via per-key dot with f16 B^T rows
// Gate (o * sigmoid(gate)) is fused in decode_reduce / prefill epilogue when a gate pointer is given (nullptr = skip).
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <math.h>

namespace ia {
constexpr int HD = 128, NH = 24, NKV = 8, GQ = 3, RD = 16, WIN = 1024, CELL = NKV * HD;  // CELL = 1024 halves = 2 KiB
typedef _Float16 h16;
typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef _Float16 v2h __attribute__((ext_vector_type(2)));
typedef float v8f __attribute__((ext_vector_type(8)));

__device__ __forceinline__ float wsum32(float x) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) x += __shfl_xor(x, o, 32);
  return x;
}
__device__ __forceinline__ float tau_of(int pos) {
  float x = (float)(pos + 1) * (1.0f / 8192.0f);
  return 1.0f + 0.1f * logf(fmaxf(1.0f, x));
}
__device__ __forceinline__ v16h cat16(uint4 a, uint4 b) {
  union { uint4 u[2]; v16h v; } x; x.u[0] = a; x.u[1] = b; return x.v;
}
__device__ __forceinline__ v2h as_v2h(uint32_t u) { return __builtin_bit_cast(v2h, u); }
__device__ __forceinline__ uint32_t as_u32(v2h v) { return __builtin_bit_cast(uint32_t, v); }
__device__ __forceinline__ void cbar() { asm volatile("" ::: "memory"); }
__device__ __forceinline__ float2 h2_to_f2(uint32_t u) {
  v2h v = as_v2h(u); return make_float2((float)v[0], (float)v[1]);
}
// bias = R . Bt[d]  (R as 8 packed half2, Bt row = 16 f16 = 32 B)
__device__ __forceinline__ float bias_dot(const v2h* rp, const __half* bt, int d) {
  const uint4* bp = (const uint4*)(bt + (size_t)d * RD);
  uint4 b0 = bp[0], b1 = bp[1];
  float s = 0.f;
  s = __builtin_amdgcn_fdot2(rp[0], as_v2h(b0.x), s, false);
  s = __builtin_amdgcn_fdot2(rp[1], as_v2h(b0.y), s, false);
  s = __builtin_amdgcn_fdot2(rp[2], as_v2h(b0.z), s, false);
  s = __builtin_amdgcn_fdot2(rp[3], as_v2h(b0.w), s, false);
  s = __builtin_amdgcn_fdot2(rp[4], as_v2h(b1.x), s, false);
  s = __builtin_amdgcn_fdot2(rp[5], as_v2h(b1.y), s, false);
  s = __builtin_amdgcn_fdot2(rp[6], as_v2h(b1.z), s, false);
  s = __builtin_amdgcn_fdot2(rp[7], as_v2h(b1.w), s, false);
  return s;
}

// ------------------------------------------------------------------ kv_write
struct KvWriteArgs {
  const float* k; const float* v;   // raw [n][8][128] f32 (K post-conv pre-clip; V post-conv)
  int n; int pos0; int cap;
  __half* kc; __half* vc;
  float qk_thr; float eps;
};
// one wave per (token, kv head); 4 waves/block
__global__ __launch_bounds__(128) void kv_write(KvWriteArgs a) {
  const int lane = threadIdx.x & 31, gw = blockIdx.x * 4 + (threadIdx.x >> 5);
  if (gw >= a.n * NKV) return;
  const int t = gw / NKV, kvh = gw % NKV;
  const size_t off = (size_t)t * CELL + kvh * HD;
  float4 kx = ((const float4*)(a.k + off))[lane];
  float4 vx = ((const float4*)(a.v + off))[lane];
  float ks = wsum32(kx.x * kx.x + kx.y * kx.y + kx.z * kx.z + kx.w * kx.w);
  float vs = wsum32(vx.x * vx.x + vx.y * vx.y + vx.z * vx.z + vx.w * vx.w);
  float kf = fminf(1.0f, a.qk_thr / (sqrtf(ks) + 1e-6f));
  float vf = rsqrtf(vs * (1.0f / HD) + a.eps);
  int cell = (a.pos0 + t) % a.cap;
  __half2* kp = (__half2*)(a.kc + (size_t)cell * CELL + kvh * HD + lane * 4);
  __half2* vp = (__half2*)(a.vc + (size_t)cell * CELL + kvh * HD + lane * 4);
  kp[0] = __floats2half2_rn(kx.x * kf, kx.y * kf); kp[1] = __floats2half2_rn(kx.z * kf, kx.w * kf);
  vp[0] = __floats2half2_rn(vx.x * vf, vx.y * vf); vp[1] = __floats2half2_rn(vx.z * vf, vx.w * vf);
}

// ------------------------------------------------------------------ qr_prep (prefill)
struct PrepArgs {
  const float* q; const float* r;          // raw [n][24][128], [n][24][16]
  const float* qscale; const float* rclip; // [24][128], [24][16]
  int n; int pos0; int swa; float qk_thr; float r_thr;
  __half* qn; float* rn;                   // [n][24][128] f16 (x tau x 1/sqrt128), [n][24][16] f32 (x tau)
};
__global__ __launch_bounds__(128) void qr_prep(PrepArgs a) {
  const int lane = threadIdx.x & 31, gw = blockIdx.x * 4 + (threadIdx.x >> 5);
  if (gw >= a.n * NH) return;
  const int t = gw / NH, h = gw % NH;
  const float tau = a.swa ? 1.0f : tau_of(a.pos0 + t);
  float4 x = ((const float4*)(a.q + (size_t)gw * HD))[lane];
  float4 s = ((const float4*)(a.qscale + h * HD))[lane];
  x.x *= s.x; x.y *= s.y; x.z *= s.z; x.w *= s.w;
  float ss = wsum32(x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w);
  float sc = fminf(1.0f, a.qk_thr / (sqrtf(ss) + 1e-6f)) * tau * 0.08838834764831845f;
  __half2* qp = (__half2*)(a.qn + (size_t)gw * HD + lane * 4);
  qp[0] = __floats2half2_rn(x.x * sc, x.y * sc); qp[1] = __floats2half2_rn(x.z * sc, x.w * sc);
  float rv = 0.f;
  if (lane < RD) rv = a.r[(size_t)gw * RD + lane] * a.rclip[h * RD + lane];
  float rs = wsum32(rv * rv);
  float rf = fminf(1.0f, a.r_thr / (sqrtf(rs) + 1e-6f)) * tau;
  if (lane < RD) a.rn[(size_t)gw * RD + lane] = rv * rf;
}

// ------------------------------------------------------------------ decode
struct DecArgs {
  const __half* kc; const __half* vc; int cap; int swa;
  int nq; int p0;                        // queries at positions p0 .. p0+nq-1 (already written to the cache)
  const float* q; const float* r;        // raw [nq][24][128], [nq][24][16]
  const float* qscale; const float* rclip;
  const __half* bt;                      // B^T [1024][16] f16
  float qk_thr; float r_thr;
  int kb; int chunk;                     // first key position, keys per split
  float* ws_o; float* ws_m; float* ws_l; // [nsplit][8][12][128], [nsplit][8][12] x2
  // optional single-kernel mode (small nsplit): the last block of each kv head does the split reduction
  int fuse; int nsplit; unsigned* counters;  // counters[8], zero-initialised once, self-resetting
  const float* sink; const float* gate; float* out;
};

template <int NQ, int NW>
__global__ __launch_bounds__(NW * 32) void attn_decode(DecArgs a) {
  constexpr int NR = 3 * NQ;
  __shared__ __attribute__((aligned(16))) __half sQ[16][HD + 8];
  __shared__ float sR[16][16];
  __shared__ __attribute__((aligned(16))) float sP[NW][16][16];
  __shared__ float sA[NW][16];
  __shared__ __attribute__((aligned(16))) float sO[NW][NR][HD];
  __shared__ float sM[NW][16], sL[NW][16];
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  const int kvh = blockIdx.x, split = blockIdx.y;
  const int pmax = a.p0 + a.nq - 1;
  const int lo = a.kb + split * a.chunk;
  const int hi = min(lo + a.chunk, pmax + 1);

  // ---- prologue: clip / scale Q and R rows for this kv head's 3 heads x NQ tokens
  for (int r = w; r < 16; r += NW) {
    if (r < NR) {
      const int t = r / 3, h = kvh * 3 + r % 3;
      const float tau = a.swa ? 1.0f : tau_of(a.p0 + t);
      float4 x = ((const float4*)(a.q + ((size_t)t * NH + h) * HD))[lane];
      float4 s = ((const float4*)(a.qscale + h * HD))[lane];
      x.x *= s.x; x.y *= s.y; x.z *= s.z; x.w *= s.w;
      float ss = wsum32(x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w);
      float sc = fminf(1.0f, a.qk_thr / (sqrtf(ss) + 1e-6f)) * tau * 0.08838834764831845f;
      __half2* qp = (__half2*)&sQ[r][lane * 4];
      qp[0] = __floats2half2_rn(x.x * sc, x.y * sc); qp[1] = __floats2half2_rn(x.z * sc, x.w * sc);
      float rv = 0.f;
      if (lane < RD) rv = a.r[((size_t)t * NH + h) * RD + lane] * a.rclip[h * RD + lane];
      float rs = wsum32(rv * rv);
      float rf = fminf(1.0f, a.r_thr / (sqrtf(rs) + 1e-6f)) * tau;
      if (lane < RD) sR[r][lane] = rv * rf;
    } else {
      __half2* qp = (__half2*)&sQ[r][lane * 4];
      qp[0] = __floats2half2_rn(0.f, 0.f); qp[1] = __floats2half2_rn(0.f, 0.f);
      if (lane < RD) sR[r][lane] = 0.f;
    }
  }
  __syncthreads();
  const int q16 = lane & 15, half = lane >> 4;
  v16h bq[8];
#pragma unroll
  for (int s = 0; s < 8; s++) bq[s] = cat16(*(const uint4*)&sQ[q16][s * 16], *(const uint4*)&sQ[q16][s * 16 + 8]);
  v2h rp[8];
#pragma unroll
  for (int c = 0; c < 8; c++) { rp[c][0] = (h16)sR[q16][2 * c]; rp[c][1] = (h16)sR[q16][2 * c + 1]; }
  const int posq = (q16 < NR) ? a.p0 + q16 / 3 : -1;

  float m = -INFINITY, l = 0.f;
  float acc[NR][4];
#pragma unroll
  for (int r = 0; r < NR; r++) { acc[r][0] = acc[r][1] = acc[r][2] = acc[r][3] = 0.f; }

  for (int jt = lo + w * 16; jt < hi; jt += NW * 16) {
    const int pk = min(jt + q16, hi - 1);
    const uint4* kp = (const uint4*)(a.kc + (size_t)(pk % a.cap) * CELL + kvh * HD);
    uint4 kr[16];
#pragma unroll
    for (int u = 0; u < 16; u++) kr[u] = kp[u];
    uint2 vr[16];
#pragma unroll
    for (int k = 0; k < 16; k++) {
      const int pv = min(jt + k, hi - 1);
      vr[k] = *(const uint2*)(a.vc + (size_t)(pv % a.cap) * CELL + kvh * HD + lane * 4);
    }
    v8f c = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int s = 0; s < 8; s++) c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(cat16(kr[2 * s], kr[2 * s + 1]), bq[s], c);

    float sv[8], mx = -INFINITY;
    const bool needb = (posq >= 0) && (posq - (jt + 15) < WIN);
    const bool anyb = __any(needb);
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const int pos = jt + 2 * i + half, d = posq - pos;
      const bool ok = (d >= 0) && (pos < hi) && (!a.swa || d < WIN);
      float s = c[i];
      if (anyb && ok && d < WIN) s += bias_dot(rp, a.bt, d);
      sv[i] = ok ? s : -INFINITY;
      mx = fmaxf(mx, sv[i]);
    }
    mx = fmaxf(mx, __shfl_xor(mx, 16, 32));
    const float mn = fmaxf(m, mx);
    float alpha = 1.f, ps = 0.f;
    if (mn != -INFINITY) {
      alpha = __expf(m - mn);
#pragma unroll
      for (int i = 0; i < 8; i++) { float p = __expf(sv[i] - mn); ps += p; sP[w][2 * i + half][q16] = p; }
    } else {
#pragma unroll
      for (int i = 0; i < 8; i++) sP[w][2 * i + half][q16] = 0.f;
    }
    l = l * alpha + ps; m = mn;
    sA[w][q16] = alpha;
    cbar();
    float al[NR];
#pragma unroll
    for (int r = 0; r < NR; r++) al[r] = sA[w][r];
#pragma unroll
    for (int r = 0; r < NR; r++) { acc[r][0] *= al[r]; acc[r][1] *= al[r]; acc[r][2] *= al[r]; acc[r][3] *= al[r]; }
#pragma unroll
    for (int k = 0; k < 16; k++) {
      float2 v01 = h2_to_f2(vr[k].x), v23 = h2_to_f2(vr[k].y);
      float p[NR];
#pragma unroll
      for (int g = 0; g < (NR + 3) / 4; g++) {
        float4 t4 = *(const float4*)&sP[w][k][g * 4];
        if (g * 4 + 0 < NR) p[g * 4 + 0] = t4.x;
        if (g * 4 + 1 < NR) p[g * 4 + 1] = t4.y;
        if (g * 4 + 2 < NR) p[g * 4 + 2] = t4.z;
        if (g * 4 + 3 < NR) p[g * 4 + 3] = t4.w;
      }
#pragma unroll
      for (int r = 0; r < NR; r++) {
        acc[r][0] = fmaf(p[r], v01.x, acc[r][0]); acc[r][1] = fmaf(p[r], v01.y, acc[r][1]);
        acc[r][2] = fmaf(p[r], v23.x, acc[r][2]); acc[r][3] = fmaf(p[r], v23.y, acc[r][3]);
      }
    }
    cbar();
  }
  // ---- merge the NW waves of this block
  const float ltot = l + __shfl_xor(l, 16, 32);
  if (lane < 16) { sM[w][lane] = m; sL[w][lane] = ltot; }
#pragma unroll
  for (int r = 0; r < NR; r++) *(float4*)&sO[w][r][lane * 4] = make_float4(acc[r][0], acc[r][1], acc[r][2], acc[r][3]);
  __syncthreads();
  const size_t base = ((size_t)split * NKV + kvh) * 12;
  for (int idx = tid; idx < NR * HD; idx += NW * 32) {
    const int r = idx / HD, dim = idx % HD;
    float M = -INFINITY;
#pragma unroll
    for (int w2 = 0; w2 < NW; w2++) M = fmaxf(M, sM[w2][r]);
    float o = 0.f;
    if (M != -INFINITY) {
#pragma unroll
      for (int w2 = 0; w2 < NW; w2++) o += __expf(sM[w2][r] - M) * sO[w2][r][dim];
    }
    a.ws_o[(base + r) * HD + dim] = o;
  }
  if (tid < NR) {
    float M = -INFINITY;
#pragma unroll
    for (int w2 = 0; w2 < NW; w2++) M = fmaxf(M, sM[w2][tid]);
    float L = 0.f;
    if (M != -INFINITY) {
#pragma unroll
      for (int w2 = 0; w2 < NW; w2++) L += __expf(sM[w2][tid] - M) * sL[w2][tid];
    }
    a.ws_m[base + tid] = M; a.ws_l[base + tid] = L;
  }
  if (a.fuse) {
    __shared__ int sLast;
    __threadfence();
    __syncthreads();
    if (tid == 0) sLast = (atomicAdd(&a.counters[kvh], 1u) == (unsigned)(a.nsplit - 1));
    __syncthreads();
    if (sLast) {
      __threadfence();
      if (tid == 0) a.counters[kvh] = 0;
      auto ld = [](const float* p) { return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); };
      for (int idx = tid; idx < NR * HD; idx += NW * 32) {
        const int r = idx / HD, dim = idx % HD, t = r / 3, h = kvh * 3 + r % 3;
        const float sk = a.sink[t * NH + h];
        float M = sk;
        for (int s = 0; s < a.nsplit; s++) M = fmaxf(M, ld(&a.ws_m[((size_t)s * NKV + kvh) * 12 + r]));
        float Z = __expf(sk - M), o = 0.f;
        for (int s = 0; s < a.nsplit; s++) {
          const size_t b = ((size_t)s * NKV + kvh) * 12 + r;
          const float ms = ld(&a.ws_m[b]);
          if (ms != -INFINITY) { const float e = __expf(ms - M); Z += e * ld(&a.ws_l[b]); o += e * ld(&a.ws_o[b * HD + dim]); }
        }
        o /= Z;
        if (a.gate) o *= 1.0f / (1.0f + __expf(-a.gate[(size_t)t * (NH * HD) + h * HD + dim]));
        a.out[((size_t)t * NH + h) * HD + dim] = o;
      }
    }
  }
}

struct ReduceArgs {
  const float* ws_o; const float* ws_m; const float* ws_l; int nsplit;
  const float* sink;   // [nq][24]
  const float* gate;   // [nq][24*128] or nullptr
  float* out;          // [nq][24][128]
};
__global__ __launch_bounds__(128) void decode_reduce(ReduceArgs a) {
  __shared__ float sc[1024];
  __shared__ float sred[4], sZ;
  const int t = blockIdx.x / NH, h = blockIdx.x % NH, kvh = h / 3, r = t * 3 + h % 3, dim = threadIdx.x;
  const float sk = a.sink[t * NH + h];
  // pass 1: per-split (m, l) -> global max M and denominator Z (all 128 threads share the nsplit loop)
  float mmax = sk;
  for (int s = dim; s < a.nsplit; s += 128) { sc[s] = a.ws_m[((size_t)s * NKV + kvh) * 12 + r]; mmax = fmaxf(mmax, sc[s]); }
  mmax = fmaxf(mmax, __shfl_xor(mmax, 16, 32)); mmax = fmaxf(mmax, __shfl_xor(mmax, 8, 32)); mmax = fmaxf(mmax, __shfl_xor(mmax, 4, 32));
  mmax = fmaxf(mmax, __shfl_xor(mmax, 2, 32)); mmax = fmaxf(mmax, __shfl_xor(mmax, 1, 32));
  if ((dim & 31) == 0) sred[dim >> 5] = mmax;
  __syncthreads();
  const float M = fmaxf(fmaxf(sred[0], sred[1]), fmaxf(sred[2], sred[3]));
  float zp = 0.f;
  for (int s = dim; s < a.nsplit; s += 128) {
    const float ms = sc[s];
    const float e = (ms == -INFINITY) ? 0.f : __expf(ms - M);
    sc[s] = e; zp += e * ((ms == -INFINITY) ? 0.f : a.ws_l[((size_t)s * NKV + kvh) * 12 + r]);
  }
  zp = wsum32(zp);
  __syncthreads();
  if ((dim & 31) == 0) sred[dim >> 5] = zp;
  __syncthreads();
  const float Z = __expf(sk - M) + sred[0] + sred[1] + sred[2] + sred[3];
  // pass 2: weighted sum, independent loads
  float o = 0.f;
#pragma unroll 8
  for (int s = 0; s < a.nsplit; s++) o += sc[s] * a.ws_o[(((size_t)s * NKV + kvh) * 12 + r) * HD + dim];
  o /= Z;
  if (a.gate) o *= 1.0f / (1.0f + __expf(-a.gate[(size_t)t * (NH * HD) + h * HD + dim]));
  a.out[((size_t)t * NH + h) * HD + dim] = o;
}

// ------------------------------------------------------------------ prefill
struct PfArgs {
  const __half* kc; const __half* vc; int cap; int swa;
  int U; int p0;                      // queries at positions p0 .. p0+U-1 (already in the cache)
  const __half* qn; const float* rn;  // from qr_prep
  const float* sink;                  // [U][24]
  const float* gate;                  // [U][3072] or nullptr
  const __half* bt;
  float* out;                         // [U][24][128] f32
};
constexpr int PF_KT = 64;       // keys per LDS tile
constexpr int PF_KS = HD + 8;   // sK row stride (halves)
constexpr int PF_VS = PF_KT + 8; // sVt row stride (halves)

template <int NTT>
__global__ __launch_bounds__(NTT * 96) void attn_prefill(PfArgs a) {
  constexpr int NT = NTT * 96;
  __shared__ __attribute__((aligned(16))) __half sK[PF_KT][PF_KS];
  __shared__ __attribute__((aligned(16))) __half sVt[HD][PF_VS];
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  const int kvh = blockIdx.y;
  const int hh = w % 3, tt = w / 3, h = kvh * 3 + hh;
  const int q16 = lane & 15, half = lane >> 4;
  const int tbase = blockIdx.x * 16 * NTT;
  const int tb = tbase + tt * 16;              // first token of this wave
  const int t = tb + q16;
  const bool rowvalid = t < a.U;
  const int tcl = min(t, a.U - 1);
  const int posq = rowvalid ? a.p0 + t : -1;

  v16h bq[8];
  {
    const uint4* qp = (const uint4*)(a.qn + ((size_t)tcl * NH + h) * HD);
#pragma unroll
    for (int s = 0; s < 8; s++) bq[s] = cat16(qp[2 * s], qp[2 * s + 1]);
  }
  v2h rp[8];
  {
    const float2* rq = (const float2*)(a.rn + ((size_t)tcl * NH + h) * RD);
#pragma unroll
    for (int c = 0; c < 8; c++) { float2 f = rq[c]; rp[c][0] = (h16)f.x; rp[c][1] = (h16)f.y; }
  }
  float m = rowvalid ? a.sink[(size_t)t * NH + h] : 0.f;
  float l = (half == 0) ? 1.f : 0.f;
  v8f acc[8];
#pragma unroll
  for (int d = 0; d < 8; d++) acc[d] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};

  const int pfirst = a.p0 + tbase;
  const int plast = min(a.p0 + a.U - 1, a.p0 + tbase + 16 * NTT - 1);
  const int kfirst = a.swa ? max(0, pfirst - (WIN - 1)) : 0;
  const int ntiles = (plast - kfirst + PF_KT) / PF_KT;
  const int wmin = a.p0 + tb, wmax = a.p0 + tb + 15;  // wave-uniform position range of its rows

  for (int tile = 0; tile < ntiles; tile++) {
    const int kbase = kfirst + tile * PF_KT;
    __syncthreads();
    for (int idx = tid; idx < PF_KT * 16; idx += NT) {
      const int key = idx >> 4, c = idx & 15;
      const int pos = min(kbase + key, plast);
      *(uint4*)&sK[key][c * 8] = *(const uint4*)(a.kc + (size_t)(pos % a.cap) * CELL + kvh * HD + c * 8);
    }
    for (int idx = tid; idx < (PF_KT / 2) * 16; idx += NT) {
      const int kp = idx >> 4, c = idx & 15;
      const int p0_ = min(kbase + 2 * kp, plast), p1_ = min(kbase + 2 * kp + 1, plast);
      uint4 v0 = *(const uint4*)(a.vc + (size_t)(p0_ % a.cap) * CELL + kvh * HD + c * 8);
      uint4 v1 = *(const uint4*)(a.vc + (size_t)(p1_ % a.cap) * CELL + kvh * HD + c * 8);
      uint32_t* dst = (uint32_t*)&sVt[c * 8][2 * kp];
      constexpr int RS = PF_VS / 2;   // row stride in words
      uint32_t x0[4] = {v0.x, v0.y, v0.z, v0.w}, x1[4] = {v1.x, v1.y, v1.z, v1.w};
#pragma unroll
      for (int e = 0; e < 4; e++) {
        dst[(2 * e) * RS] = (x0[e] & 0xffffu) | (x1[e] << 16);
        dst[(2 * e + 1) * RS] = (x0[e] >> 16) | (x1[e] & 0xffff0000u);
      }
    }
    __syncthreads();
#pragma unroll
    for (int kt = 0; kt < PF_KT / 16; kt++) {
      const int pos0 = kbase + kt * 16;
      if (pos0 > wmax) continue;
      if (a.swa && (wmin - (pos0 + 15)) >= WIN) continue;
      v8f c = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int s = 0; s < 8; s++)
        c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
            cat16(*(const uint4*)&sK[kt * 16 + q16][s * 16], *(const uint4*)&sK[kt * 16 + q16][s * 16 + 8]), bq[s], c);
      float sv[8], mx = -INFINITY;
      const bool needb = rowvalid && (posq - (pos0 + 15) < WIN);
      const bool anyb = __any(needb);
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const int pos = pos0 + 2 * i + half, d = posq - pos;
        const bool ok = (d >= 0) && (pos <= plast) && (!a.swa || d < WIN);
        float s = c[i];
        if (anyb && ok && d < WIN) s += bias_dot(rp, a.bt, d);
        sv[i] = ok ? s : -INFINITY;
        mx = fmaxf(mx, sv[i]);
      }
      mx = fmaxf(mx, __shfl_xor(mx, 16, 32));
      const float mn = fmaxf(m, mx);
      const float alpha = __expf(m - mn);
      float p[8], ps = 0.f;
#pragma unroll
      for (int i = 0; i < 8; i++) { p[i] = __expf(sv[i] - mn); ps += p[i]; }
      l = l * alpha + ps; m = mn;
      // P as WMMA B operand: slot s <-> key s; even slots live in half 0 lanes, odd slots in half 1 lanes
      uint32_t mine[4], theirs[4];
#pragma unroll
      for (int i = 0; i < 4; i++) {
        v2h pr; pr[0] = (h16)p[2 * i]; pr[1] = (h16)p[2 * i + 1]; mine[i] = as_u32(pr);
        theirs[i] = __shfl_xor(mine[i], 16, 32);
      }
      // mine[i] packs regs (2i, 2i+1): keys (4i+half, 4i+2+half)
      v16h pf;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        v2h mv = as_v2h(mine[i]), tv = as_v2h(theirs[i]);
        // half0 holds even keys 4i, 4i+2 ; half1 holds odd keys 4i+1, 4i+3
        pf[4 * i + 0] = half ? tv[0] : mv[0];
        pf[4 * i + 1] = half ? mv[0] : tv[0];
        pf[4 * i + 2] = half ? tv[1] : mv[1];
        pf[4 * i + 3] = half ? mv[1] : tv[1];
      }
#pragma unroll
      for (int dt = 0; dt < 8; dt++) {
        acc[dt] *= alpha;
        const __half* vp = &sVt[dt * 16 + q16][kt * 16];
        acc[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(cat16(*(const uint4*)vp, *(const uint4*)(vp + 8)), pf, acc[dt]);
      }
    }
  }
  if (!rowvalid) return;
  const float inv = 1.0f / (l + __shfl_xor(l, 16, 32));
  float* op = a.out + ((size_t)t * NH + h) * HD;
  const float* gp = a.gate ? a.gate + (size_t)t * (NH * HD) + h * HD : nullptr;
#pragma unroll
  for (int dt = 0; dt < 8; dt++)
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const int dim = dt * 16 + 2 * i + half;
      float o = acc[dt][i] * inv;
      if (gp) o *= 1.0f / (1.0f + __expf(-gp[dim]));
      op[dim] = o;
    }
}
}  // namespace ia
