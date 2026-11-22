// Fast kernels for Inkling 2 Mini (milestone M2). See fast.hpp.
// MIT. Ported from ext/lab (kern.h, kern2.h, attn/attn_kernels.h of this repository).
#include "src/hip/fast.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <stdexcept>
#include <string>

namespace gufo::models::inkling2::fast {
namespace {

constexpr int WS = 32;
typedef uint32_t v4u __attribute__((ext_vector_type(4)));
typedef uint32_t v4u_a2 __attribute__((ext_vector_type(4), aligned(2)));
typedef uint32_t v4u_a4 __attribute__((ext_vector_type(4), aligned(4)));

constexpr int FMT_Q8 = 0, FMT_NVFP4 = 1;
template <int F> struct Fm;
template <> struct Fm<FMT_Q8> { static constexpr int BYTES = 34, UEL = 32; };
template <> struct Fm<FMT_NVFP4> { static constexpr int BYTES = 36, UEL = 64; };

__device__ __forceinline__ int dot4(uint32_t a, uint32_t b, int c) {
  return __builtin_amdgcn_sudot4(true, (int)a, true, (int)b, c, false);
}
__device__ __forceinline__ uint32_t dec4(uint32_t n) {
  uint32_t sel = n & 0x07070707u;
  uint32_t pos = __builtin_amdgcn_perm(0x0C080604u, 0x03020100u, sel);
  uint32_t neg = __builtin_amdgcn_perm(0xF4F8FAFCu, 0xFDFEFF00u, sel);
  uint32_t m = ((n >> 3) & 0x01010101u) * 255u;
  return (neg & m) | (pos & ~m);
}
__device__ __forceinline__ float ue4m3(uint32_t x) {
  uint32_t e = (x >> 3) & 15u, m = x & 7u;
  float nrm = __uint_as_float(((e + 119u) << 23) | (m << 20));
  float sub = (float)m * (1.0f / 1024.0f);
  float r = e ? nrm : sub;
  return (x == 0x7Fu) ? 0.f : r;
}
__device__ __forceinline__ uint32_t pack4(float a, float b, float c, float d) {
  return ((uint32_t)(uint8_t)(int8_t)(int)rintf(a)) | ((uint32_t)(uint8_t)(int8_t)(int)rintf(b) << 8) |
         ((uint32_t)(uint8_t)(int8_t)(int)rintf(c) << 16) |
         ((uint32_t)(uint8_t)(int8_t)(int)rintf(d) << 24);
}

// Quantize activation vectors (contiguous, K floats each) to int8 with a per-16 f32 scale in LDS,
// plus a second int8 plane holding the quantisation residual (about 14 bit of activation
// precision in total). Plane 1 lives at aq + NV*CPU*NU and asc + NV*K/16.
template <int K, int NV, int UEL, int NPER = NV>
__device__ __forceinline__ void quant_act(const float* __restrict__ x, v4u* aq, float* asc, int res,
                                          int ldt = 0) {
  constexpr int NG = K / 16, NU = K / UEL, CPU = UEL / 16, PLQ = NV * CPU * NU, PLS = NV * NG;
  for (int it = threadIdx.x; it < NV * NG; it += blockDim.x) {
    int v = it / NG, g = it % NG;
    const float4* xp = (const float4*)(x + (size_t)(v / NPER) * ldt + (size_t)(v % NPER) * K + g * 16);
    float f[16];
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const float4 t = xp[i];
      f[4 * i] = t.x; f[4 * i + 1] = t.y; f[4 * i + 2] = t.z; f[4 * i + 3] = t.w;
      amax = fmaxf(amax, fmaxf(fmaxf(fabsf(t.x), fabsf(t.y)), fmaxf(fabsf(t.z), fabsf(t.w))));
    }
    const float sc = amax / 127.f, id = amax > 0.f ? 127.f / amax : 0.f;
    float r[16];
    uint32_t q[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
      float q0 = rintf(f[4 * i] * id), q1 = rintf(f[4 * i + 1] * id), q2 = rintf(f[4 * i + 2] * id),
            q3 = rintf(f[4 * i + 3] * id);
      q[i] = pack4(q0, q1, q2, q3);
      r[4 * i] = f[4 * i] - q0 * sc; r[4 * i + 1] = f[4 * i + 1] - q1 * sc;
      r[4 * i + 2] = f[4 * i + 2] - q2 * sc; r[4 * i + 3] = f[4 * i + 3] - q3 * sc;
    }
    asc[v * NG + g] = sc;
    const int idx = (v * CPU + g % CPU) * NU + g / CPU;
    aq[idx] = v4u{q[0], q[1], q[2], q[3]};
    if (res) {
      // residual: |r| <= sc/2; scale sc/254 maps it onto +-127
      const float sc2 = sc * (1.0f / 254.0f), id2 = sc > 0.f ? 254.0f / sc : 0.f;
      uint32_t q2p[4];
#pragma unroll
      for (int i = 0; i < 4; i++)
        q2p[i] = pack4(r[4 * i] * id2, r[4 * i + 1] * id2, r[4 * i + 2] * id2, r[4 * i + 3] * id2);
      asc[PLS + v * NG + g] = sc2;
      aq[PLQ + idx] = v4u{q2p[0], q2p[1], q2p[2], q2p[3]};
    }
  }
  __syncthreads();
}

template <int B, int K, int VS = 1>
__device__ __forceinline__ void q8_unit(const uint8_t* p, const v4u* aq, const float* asc, int u,
                                        int vb, float (&acc)[B]) {
  constexpr int NU = K / 32;
  v4u w0 = *(const v4u_a2*)(p + 2);
  v4u w1 = *(const v4u_a2*)(p + 18);
  float d = __half2float(*(const __half*)p);
#pragma unroll
  for (int b = 0; b < B; b++) {
    v4u a0 = aq[((vb + b * VS) * 2 + 0) * NU + u], a1 = aq[((vb + b * VS) * 2 + 1) * NU + u];
    int s0 = dot4(w0.x, a0.x, 0);
    s0 = dot4(w0.y, a0.y, s0);
    s0 = dot4(w0.z, a0.z, s0);
    s0 = dot4(w0.w, a0.w, s0);
    int s1 = dot4(w1.x, a1.x, 0);
    s1 = dot4(w1.y, a1.y, s1);
    s1 = dot4(w1.z, a1.z, s1);
    s1 = dot4(w1.w, a1.w, s1);
    const float* sc = asc + (vb + b * VS) * (K / 16) + 2 * u;
    acc[b] += d * (sc[0] * (float)s0 + sc[1] * (float)s1);
  }
}

template <int B, int K, int VS = 1>
__device__ __forceinline__ void nv_unit(const uint8_t* p, const v4u* aq, const float* asc, int u,
                                        int vb, float (&acc)[B]) {
  constexpr int NU = K / 64, NB16 = K / 16;
  uint32_t sw = *(const uint32_t*)p;
  v4u q0 = *(const v4u_a4*)(p + 4);
  v4u q1 = *(const v4u_a4*)(p + 20);
  uint32_t w[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
  for (int s = 0; s < 4; s++) {
    uint32_t w0 = w[2 * s], w1 = w[2 * s + 1];
    uint32_t l0 = dec4(w0 & 0x0F0F0F0Fu), l1 = dec4(w1 & 0x0F0F0F0Fu);
    uint32_t h0 = dec4((w0 >> 4) & 0x0F0F0F0Fu), h1 = dec4((w1 >> 4) & 0x0F0F0F0Fu);
    float ds = ue4m3((sw >> (8 * s)) & 0xFFu);
#pragma unroll
    for (int b = 0; b < B; b++) {
      v4u a = aq[((vb + b * VS) * 4 + s) * NU + u];
      int t = dot4(l0, a.x, 0);
      t = dot4(l1, a.y, t);
      t = dot4(h0, a.z, t);
      t = dot4(h1, a.w, t);
      acc[b] += ds * asc[(vb + b * VS) * NB16 + u * 4 + s] * (float)t;
    }
  }
}

template <int FMT, int K, int LPR, int B, int VS = 1>
__device__ __forceinline__ void row_dot(const uint8_t* row, const v4u* aq, const float* asc, int plq,
                                        int pls, int res, int vb, int lir, float (&acc)[B]) {
  constexpr int NU = K / Fm<FMT>::UEL, BY = Fm<FMT>::BYTES;
#pragma unroll
  for (int b = 0; b < B; b++) acc[b] = 0.f;
#pragma unroll
  for (int u0 = 0; u0 < NU; u0 += LPR) {
    int u = u0 + lir;
    if ((NU % LPR == 0) || u < NU) {
      if (FMT == FMT_Q8) {
        q8_unit<B, K, VS>(row + (size_t)u * BY, aq, asc, u, vb, acc);
        if (res) q8_unit<B, K, VS>(row + (size_t)u * BY, aq + plq, asc + pls, u, vb, acc);
      } else {
        nv_unit<B, K, VS>(row + (size_t)u * BY, aq, asc, u, vb, acc);
        if (res) nv_unit<B, K, VS>(row + (size_t)u * BY, aq + plq, asc + pls, u, vb, acc);
      }
    }
  }
#pragma unroll
  for (int b = 0; b < B; b++)
#pragma unroll
    for (int off = LPR / 2; off > 0; off >>= 1) acc[b] += __shfl_xor(acc[b], off, WS);
}

// ---------------- dense GEMV (B = 1) with optional softcap ----------------
template <int FMT, int K, int LPR, int NIT, int BW, int NB = 1>
__global__ void __launch_bounds__(BW * WS)
    k_gemv(const uint8_t* __restrict__ W, const float* __restrict__ x, float* __restrict__ y, int M,
           float softcap, int res) {
  constexpr int NU = K / Fm<FMT>::UEL, CPU = Fm<FMT>::UEL / 16, RPW = WS / LPR;
  __shared__ v4u aq[2 * NB * CPU * NU];
  __shared__ float asc[2 * NB * (K / 16)];
  quant_act<K, NB, Fm<FMT>::UEL>(x, aq, asc, res);
  int wave = threadIdx.x / WS, lane = threadIdx.x % WS, g = lane / LPR, lir = lane % LPR;
  size_t rb = (size_t)NU * Fm<FMT>::BYTES;
  int base = (blockIdx.x * BW + wave) * RPW * NIT;
#pragma unroll
  for (int it = 0; it < NIT; it++) {
    int row = base + it * RPW + g;
    int rr = row < M ? row : M - 1;
    float acc[NB];
    row_dot<FMT, K, LPR, NB>(W + rb * rr, aq, asc, NB * CPU * NU, NB * (K / 16), res, 0, lir, acc);
    if (row < M && lir == 0)
#pragma unroll
      for (int b = 0; b < NB; ++b)
        y[(size_t)b * M + row] = softcap > 0.f ? softcap * tanhf(acc[b] / softcap) : acc[b];
  }
}

// ---------------- fused multi-matrix Q8_0 GEMV (K = 2304) ----------------
struct MultiP {
  const uint8_t* W[5];
  int M[5];
  int in[5];
  int yoff[5];
  int cum[6];
  int n;
  int res;
  int ldx;  // token stride of x (floats)
  int ldy;  // token stride of y
  // tiny F32 / F16 / BF16 matrices handled by the first `ntb` blocks
  int tn, ntb;
  const void* tw[3];
  int ttype[3], trows[3], tin[3], tys[3];
  float* ty[3];
};

__device__ __forceinline__ float wsum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
  return v;
}
// One wave = one row of a tiny matrix (K = 2304, F32 / F16 / BF16) against NB un-quantised input vectors.
template <int NB>
__device__ __forceinline__ void tiny_row(const MultiP& P, const float* __restrict__ x, int row, int lane) {
  int m = 0;
  while (m < P.tn - 1 && row >= P.trows[m]) {
    row -= P.trows[m];
    ++m;
  }
  if (row >= P.trows[m]) return;
  float acc[NB];
#pragma unroll
  for (int b = 0; b < NB; ++b) acc[b] = 0.f;
  const int ty = P.ttype[m];
  const float* xv = x + (size_t)P.tin[m] * 2304;
  if (ty == 0) {
    const float4* wp = (const float4*)((const float*)P.tw[m] + (size_t)row * 2304);
    float4 wv[18];
#pragma unroll
    for (int i = 0; i < 18; ++i) wv[i] = wp[lane + 32 * i];
#pragma unroll
    for (int b = 0; b < NB; ++b) {
      const float4* xp = (const float4*)(xv + (size_t)b * P.ldx);
#pragma unroll
      for (int i = 0; i < 18; ++i) {
        const float4 a = xp[lane + 32 * i];
        acc[b] += wv[i].x * a.x + wv[i].y * a.y + wv[i].z * a.z + wv[i].w * a.w;
      }
    }
  } else {
    // F16 / BF16: 2304 halves = 288 x 16 B; 9 chunks of 8 per lane
    const uint4* wp = (const uint4*)((const uint16_t*)P.tw[m] + (size_t)row * 2304);
    uint4 wv[9];
#pragma unroll
    for (int i = 0; i < 9; ++i) wv[i] = wp[lane + 32 * i];
#pragma unroll
    for (int b = 0; b < NB; ++b) {
      const float4* xp = (const float4*)(xv + (size_t)b * P.ldx);
#pragma unroll
      for (int i = 0; i < 9; ++i) {
        const uint32_t u[4] = {wv[i].x, wv[i].y, wv[i].z, wv[i].w};
        const float4 a0 = xp[2 * (lane + 32 * i)], a1 = xp[2 * (lane + 32 * i) + 1];
        const float a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
#pragma unroll
        for (int h = 0; h < 4; ++h) {
          float lo, hi;
          if (ty == 1) {
            lo = __half2float(__ushort_as_half((unsigned short)(u[h] & 0xFFFFu)));
            hi = __half2float(__ushort_as_half((unsigned short)(u[h] >> 16)));
          } else {
            lo = __uint_as_float(u[h] << 16);
            hi = __uint_as_float(u[h] & 0xFFFF0000u);
          }
          acc[b] += lo * a[2 * h] + hi * a[2 * h + 1];
        }
      }
    }
  }
#pragma unroll
  for (int b = 0; b < NB; ++b) {
    const float v = wsum(acc[b]);
    if (lane == 0) P.ty[m][(size_t)b * P.tys[m] + row] = v;
  }
}

template <int K, int LPR, int NIT, int BW, int NI, int NB = 1>
__global__ void __launch_bounds__(BW * WS)
    k_gemv_multi(MultiP P, const float* __restrict__ x, float* __restrict__ y, int Mtot) {
  if (blockIdx.x < P.ntb) {  // the tiny matrices first, so they never queue behind the long GEMV
    tiny_row<NB>(P, x, blockIdx.x * BW + threadIdx.x / WS, threadIdx.x % WS);
    return;
  }
  const int bx = blockIdx.x - P.ntb;
  constexpr int NU = K / 32, RPW = WS / LPR;
  __shared__ v4u aq[2 * NB * NI * 2 * NU];
  __shared__ float asc[2 * NB * NI * (K / 16)];
  quant_act<K, NB * NI, 32, NI>(x, aq, asc, P.res, P.ldx);
  int wave = threadIdx.x / WS, lane = threadIdx.x % WS, g = lane / LPR, lir = lane % LPR;
  size_t rb = (size_t)NU * 34;
  int base = (bx * BW + wave) * RPW * NIT;
#pragma unroll
  for (int it = 0; it < NIT; it++) {
    int row = base + it * RPW + g;
    int rr = row < Mtot ? row : Mtot - 1;
    int m = 0;
#pragma unroll
    for (int i = 1; i < 5; i++)
      if (i < P.n && rr >= P.cum[i]) m = i;
    const uint8_t* Wm = P.W[0];
    int inm = P.in[0], yo = P.yoff[0], c0 = P.cum[0];
#pragma unroll
    for (int i = 1; i < 5; i++)
      if (i == m) {
        Wm = P.W[i];
        inm = P.in[i];
        yo = P.yoff[i];
        c0 = P.cum[i];
      }
    int lr = rr - c0;
    float acc[NB];
    row_dot<FMT_Q8, K, LPR, NB, NI>(Wm + rb * lr, aq, asc, NB * NI * 2 * NU, NB * NI * (K / 16), P.res, inm, lir, acc);
    if (row < Mtot && lir == 0)
#pragma unroll
      for (int b = 0; b < NB; ++b) y[(size_t)b * P.ldy + yo + lr] = acc[b];
  }
}

// ---------------- router: F32 [128][2304] GEMV + sigmoid + biased top-8 ----------------
template <int K, bool H>
__global__ void __launch_bounds__(256)
    k_router(const void* __restrict__ Wv, const float* __restrict__ bias, const float* __restrict__ x0,
             float* __restrict__ logit0, unsigned* __restrict__ counter0, int* __restrict__ ids0,
             float* __restrict__ wts0, float rscale) {
  const int tk = blockIdx.y;  // token of the batch
  const float* x = x0 + (size_t)tk * K;
  float* logit = logit0 + tk * 128;
  unsigned* counter = counter0 + tk;
  int* ids = ids0 + tk * 8;
  float* wts = wts0 + tk * 8;
  int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
  int row = blockIdx.x * 8 + wave;
  const float4* xp = (const float4*)x;
  float s = 0.f;
  if (H) {
    const uint2* wp = (const uint2*)((const uint16_t*)Wv + (size_t)row * K);
    uint2 wv[K / 128];
#pragma unroll
    for (int i = 0; i < K / 128; i++) wv[i] = wp[lane + 32 * i];
#pragma unroll
    for (int i = 0; i < K / 128; i++) {
      const float4 b = xp[lane + 32 * i];
      s += __half2float(__ushort_as_half((unsigned short)(wv[i].x & 0xFFFFu))) * b.x +
           __half2float(__ushort_as_half((unsigned short)(wv[i].x >> 16))) * b.y +
           __half2float(__ushort_as_half((unsigned short)(wv[i].y & 0xFFFFu))) * b.z +
           __half2float(__ushort_as_half((unsigned short)(wv[i].y >> 16))) * b.w;
    }
  } else {
    const float4* wp = (const float4*)((const float*)Wv + (size_t)row * K);
#pragma unroll
    for (int i = 0; i < K / 128; i++) {
      float4 a = wp[lane + 32 * i], b = xp[lane + 32 * i];
      s += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) s += __shfl_xor(s, o, 32);
  if (lane == 0) logit[row] = s;
  __threadfence();
  __syncthreads();
  __shared__ unsigned last;
  if (threadIdx.x == 0) {
    unsigned t = atomicAdd(&counter[0], 1u);
    last = (t == gridDim.x - 1);
  }
  __syncthreads();
  if (!last) return;
  __threadfence();
  if (wave != 0) return;
  float sc[4], sel[4];
#pragma unroll
  for (int k = 0; k < 4; k++) {
    int i = lane + 32 * k;
    float l = __builtin_nontemporal_load(&logit[i]);
    sc[k] = 1.f / (1.f + expf(-l));
    sel[k] = sc[k] + bias[i];
  }
  float ssum = 0.f;
  float myw[8];
  int myi[8];
#pragma unroll
  for (int r = 0; r < 8; r++) {
    float bv = -1e30f;
    int bi = 0;
#pragma unroll
    for (int k = 0; k < 4; k++)
      if (sel[k] > bv) {
        bv = sel[k];
        bi = lane + 32 * k;
      }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      float ov = __shfl_xor(bv, o, 32);
      int oi = __shfl_xor(bi, o, 32);
      if (ov > bv || (ov == bv && oi < bi)) {
        bv = ov;
        bi = oi;
      }
    }
    float scv = 0.f;
#pragma unroll
    for (int k = 0; k < 4; k++)
      if (lane + 32 * k == bi) {
        scv = sc[k];
        sel[k] = -1e30f;
      }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) scv += __shfl_xor(scv, o, 32);
    myw[r] = scv;
    myi[r] = bi;
    ssum += scv;
  }
  if (lane < 8) {
    float w = 0.f;
    int id = 0;
#pragma unroll
    for (int r = 0; r < 8; r++)
      if (r == lane) {
        w = myw[r];
        id = myi[r];
      }
    ids[lane] = id;
    wts[lane] = w / (ssum + 1e-20f) * rscale;
  }
  if (lane == 0) counter[0] = 0;
}

// ---------------- MoE gate/up + SITU ----------------
__device__ __forceinline__ float situ(float g, float u, float beta, float lb) {
  return beta * tanhf(g / beta) * (1.f / (1.f + expf(-g))) * (lb * tanhf(u / lb));
}
template <int FMT, int K, int LPR, int NIT, int BW>
__global__ void __launch_bounds__(BW * WS)
    k_moe_gu(const uint8_t* __restrict__ W, const float* __restrict__ escale,
             const float* __restrict__ x, const int* __restrict__ ids, float* __restrict__ a, int FF,
             float beta, float lb, int res) {
  constexpr int NU = K / Fm<FMT>::UEL, CPU = Fm<FMT>::UEL / 16, RPP = WS / LPR / 2;
  __shared__ v4u aq[2 * CPU * NU];
  __shared__ float asc[2 * (K / 16)];
  // Grid x is the (token, expert) assignment and y the row block, so the blocks of different tokens that chose the
  // same expert read its rows at about the same time (cache hits instead of a second DRAM read).
  const int p = blockIdx.x, rbk = blockIdx.y;
  quant_act<K, 1, Fm<FMT>::UEL>(x + (size_t)(p >> 3) * K, aq, asc, res);
  int e = ids[p];
  float sc = escale ? escale[e] : 1.f;
  size_t rb = (size_t)NU * Fm<FMT>::BYTES;
  const uint8_t* We = W + (size_t)e * 2 * FF * rb;
  int wave = threadIdx.x / WS, lane = threadIdx.x % WS, g = lane / LPR, lir = lane % LPR,
      isup = g & 1, pg = g >> 1;
  int base = (rbk * BW + wave) * RPP * NIT;
#pragma unroll
  for (int it = 0; it < NIT; it++) {
    int j = base + it * RPP + pg;
    int jj = j < FF ? j : FF - 1;
    float acc[1];
    row_dot<FMT, K, LPR, 1>(We + rb * (isup ? FF + jj : jj), aq, asc, CPU * NU, K / 16, res, 0, lir, acc);
    float other = __shfl_xor(acc[0], LPR, WS);
    if (!isup && lir == 0 && j < FF) a[(size_t)p * FF + j] = situ(acc[0] * sc, other * sc, beta, lb);
  }
}

// ---------------- MoE down + weighted sum ----------------
template <int FMT, int K, int LPR, int NIT, int BW>
__global__ void __launch_bounds__(BW * WS)
    k_moe_down(const uint8_t* __restrict__ W, const float* __restrict__ escale,
               const float* __restrict__ a, const int* __restrict__ ids,
               const float* __restrict__ wts, float* __restrict__ y, int D, int res) {
  constexpr int NU = K / Fm<FMT>::UEL, CPU = Fm<FMT>::UEL / 16, RPW = WS / LPR;
  __shared__ v4u aq[2 * 8 * CPU * NU];
  __shared__ float asc[2 * 8 * (K / 16)];
  const int tk = blockIdx.y;  // token of the batch
  a += (size_t)tk * 8 * K;
  ids += tk * 8;
  wts += tk * 8;
  y += (size_t)tk * D;
  quant_act<K, 8, Fm<FMT>::UEL>(a, aq, asc, res);
  size_t rb = (size_t)NU * Fm<FMT>::BYTES;
  int wave = threadIdx.x / WS, lane = threadIdx.x % WS, g = lane / LPR, lir = lane % LPR;
  int base = (blockIdx.x * BW + wave) * RPW * NIT;
  int eid[8];
  float ew[8];
#pragma unroll
  for (int s = 0; s < 8; s++) {
    eid[s] = ids[s];
    ew[s] = wts[s] * (escale ? escale[eid[s]] : 1.f);
  }
#pragma unroll
  for (int it = 0; it < NIT; it++) {
    int row = base + it * RPW + g;
    int rr = row < D ? row : D - 1;
    float sum = 0.f;
#pragma unroll
    for (int s = 0; s < 8; s++) {
      float acc[1];
      row_dot<FMT, K, LPR, 1>(W + ((size_t)eid[s] * D + rr) * rb, aq, asc, 8 * CPU * NU, 8 * (K / 16), res, s, lir, acc);
      sum += ew[s] * acc[0];
    }
    if (row < D && lir == 0) y[row] = sum;
  }
}

// ---------------- MoE down, one expert per block row (weighted partial outputs) ----------------
template <int FMT, int K, int LPR, int NIT, int BW>
__global__ void __launch_bounds__(BW * WS)
    k_moe_down_e(const uint8_t* __restrict__ W, const float* __restrict__ escale, const float* __restrict__ a,
                 const int* __restrict__ ids, const float* __restrict__ wts, float* __restrict__ yp, int D, int res) {
  constexpr int NU = K / Fm<FMT>::UEL, CPU = Fm<FMT>::UEL / 16, RPW = WS / LPR;
  __shared__ v4u aq[2 * CPU * NU];
  __shared__ float asc[2 * (K / 16)];
  const int p = blockIdx.x, rbk = blockIdx.y;  // assignment (token * 8 + slot) first, as in k_moe_gu
  quant_act<K, 1, Fm<FMT>::UEL>(a + (size_t)p * K, aq, asc, res);
  const int e = ids[p];
  const float ew = wts[p] * (escale ? escale[e] : 1.f);
  size_t rb = (size_t)NU * Fm<FMT>::BYTES;
  int wave = threadIdx.x / WS, lane = threadIdx.x % WS, g = lane / LPR, lir = lane % LPR;
  int base = (rbk * BW + wave) * RPW * NIT;
#pragma unroll
  for (int it = 0; it < NIT; it++) {
    int row = base + it * RPW + g;
    int rr = row < D ? row : D - 1;
    float acc[1];
    row_dot<FMT, K, LPR, 1>(W + ((size_t)e * D + rr) * rb, aq, asc, CPU * NU, K / 16, res, 0, lir, acc);
    if (row < D && lir == 0) yp[(size_t)p * D + row] = ew * acc[0];
  }
}

// ---------------- small ops ----------------
__device__ __forceinline__ float block_sum(float v, float* red /*[32]*/) {
  const int wid = threadIdx.x / 32, lane = threadIdx.x % 32, nw = blockDim.x / 32;
  v = wsum(v);
  __syncthreads();
  if (lane == 0) red[wid] = v;
  __syncthreads();
  float t = 0.f;
  for (int i = 0; i < nw; ++i) t += red[i];
  return t;
}


// post-conv + residual: out = (x + w . history) * scale; block: blk = cur, cur = out; else cur += out.
__device__ __forceinline__ void conv_resid_elem(const ConvResidArgs& a, int c, int tok = 0) {
  const size_t to = (size_t)tok * 2304;
  float* st = a.slot[tok * 48].conv + (size_t)a.conv_off * 3 + (size_t)c * 3;
  float xc;
  if (a.nsum == 8) {
    float v[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) v[e] = a.x[((size_t)tok * 8 + e) * 2304 + c];
    xc = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) xc += v[e];
  } else if (a.nsum > 1) {
    xc = 0.f;
    for (int e = 0; e < a.nsum; ++e) xc += a.x[((size_t)tok * a.nsum + e) * 2304 + c];
  } else {
    xc = a.x[to + c];
  }
  const float xin[4] = {st[0], st[1], st[2], xc};
  float y = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) y += a.w[(size_t)c * 4 + i] * xin[i];
  st[0] = xin[1]; st[1] = xin[2]; st[2] = xin[3];
  const float o = (xin[3] + y) * a.scale;
  const float cv = a.cur[to + c];
  if (a.blk) {
    a.blk[to + c] = cv;
    a.cur[to + c] = o;
  } else {
    a.cur[to + c] = cv + o;
  }
}
// The same for a whole 512-thread block (k_mix prologue): every load of the 5 elements per thread is issued before
// the first one is consumed, instead of one dependent chain per element.
__device__ __forceinline__ void conv_resid_block(const ConvResidArgs& a, int tok) {
  constexpr int R = 5;  // ceil(2304 / 512)
  const size_t to = (size_t)tok * 2304;
  const SlotEntry* sl = a.slot + tok * 48;
  float xin[R][4], wv[R][4], cv[R];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const int c = threadIdx.x + r * 512;
    if (c < 2304) {
      const float* st = sl->conv + (size_t)a.conv_off * 3 + (size_t)c * 3;
      xin[r][0] = st[0]; xin[r][1] = st[1]; xin[r][2] = st[2];
      float xc;
      if (a.nsum == 8) {
        float v[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) v[e] = a.x[((size_t)tok * 8 + e) * 2304 + c];
        xc = 0.f;
#pragma unroll
        for (int e = 0; e < 8; ++e) xc += v[e];
      } else {
        xc = a.x[to + c];
      }
      xin[r][3] = xc;
#pragma unroll
      for (int i = 0; i < 4; ++i) wv[r][i] = a.w[(size_t)c * 4 + i];
      cv[r] = a.cur[to + c];
    }
  }
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const int c = threadIdx.x + r * 512;
    if (c < 2304) {
      float* st = sl->conv + (size_t)a.conv_off * 3 + (size_t)c * 3;
      float y = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i) y += wv[r][i] * xin[r][i];
      st[0] = xin[r][1]; st[1] = xin[r][2]; st[2] = xin[r][3];
      const float o = (xin[r][3] + y) * a.scale;
      if (a.blk) {
        a.blk[to + c] = cv[r];
        a.cur[to + c] = o;
      } else {
        a.cur[to + c] = cv[r] + o;
      }
    }
  }
}

// ---- multi-row launches (0.1.3 Stage 2) --------------------------------------------------------------------
// Several rows of one session at consecutive positions share a launch. Row m of a session reads the committed history
// (the slot, never written during the launch) for the inputs before the launch and the RAW inputs of the session's
// earlier rows of the launch (rows tok - 1, tok - 2, tok - 3 while m allows) for the rest: exactly the four inputs the
// sequential decode would hold, computed by the same expressions, so the result is bit-identical. The state after this
// row goes to the row's snapshot (RowInfo::snap) or, when the session has one row only, in place as before.
__device__ __forceinline__ float conv_x_of(const ConvResidArgs& a, int tok, int c) {
  float xc;
  if (a.nsum == 8) {
    float v[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) v[e] = a.x[((size_t)tok * 8 + e) * 2304 + c];
    xc = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) xc += v[e];
  } else if (a.nsum > 1) {
    xc = 0.f;
    for (int e = 0; e < a.nsum; ++e) xc += a.x[((size_t)tok * a.nsum + e) * 2304 + c];
  } else {
    xc = a.x[(size_t)tok * 2304 + c];
  }
  return xc;
}
__device__ __forceinline__ void conv_resid_elem_mr(const ConvResidArgs& a, int c, int tok) {
  const size_t to = (size_t)tok * 2304;
  const RowInfo ri = a.rows[tok];
  const float* st = a.slot[tok * 48].conv + (size_t)a.conv_off * 3 + (size_t)c * 3;
  float xin[4];
#pragma unroll
  for (int i = 0; i < 3; ++i) xin[i] = (ri.m - 3 + i < 0) ? st[ri.m + i] : conv_x_of(a, tok - 3 + i, c);
  xin[3] = conv_x_of(a, tok, c);
  float y = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) y += a.w[(size_t)c * 4 + i] * xin[i];
  float* so = ri.snap ? ri.snap + ((size_t)a.visit * 6656 + a.conv_off) * 3 + (size_t)c * 3 : const_cast<float*>(st);
  so[0] = xin[1]; so[1] = xin[2]; so[2] = xin[3];
  const float o = (xin[3] + y) * a.scale;
  const float cv = a.cur[to + c];
  if (a.blk) {
    a.blk[to + c] = cv;
    a.cur[to + c] = o;
  } else {
    a.cur[to + c] = cv + o;
  }
}
__device__ __forceinline__ void conv_resid_block_mr(const ConvResidArgs& a, int tok) {
  constexpr int R = 5;  // ceil(2304 / 512)
  const size_t to = (size_t)tok * 2304;
  const RowInfo ri = a.rows[tok];
  const float* sl = a.slot[tok * 48].conv;
  float xin[R][4], wv[R][4], cv[R];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const int c = threadIdx.x + r * 512;
    if (c < 2304) {
      const float* st = sl + (size_t)a.conv_off * 3 + (size_t)c * 3;
#pragma unroll
      for (int i = 0; i < 3; ++i) xin[r][i] = (ri.m - 3 + i < 0) ? st[ri.m + i] : conv_x_of(a, tok - 3 + i, c);
      xin[r][3] = conv_x_of(a, tok, c);
#pragma unroll
      for (int i = 0; i < 4; ++i) wv[r][i] = a.w[(size_t)c * 4 + i];
      cv[r] = a.cur[to + c];
    }
  }
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const int c = threadIdx.x + r * 512;
    if (c < 2304) {
      float* so = ri.snap ? ri.snap + ((size_t)a.visit * 6656 + a.conv_off) * 3 + (size_t)c * 3
                          : const_cast<float*>(sl) + (size_t)a.conv_off * 3 + (size_t)c * 3;
      float y = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i) y += wv[r][i] * xin[r][i];
      so[0] = xin[r][1]; so[1] = xin[r][2]; so[2] = xin[r][3];
      const float o = (xin[r][3] + y) * a.scale;
      if (a.blk) {
        a.blk[to + c] = cv[r];
        a.cur[to + c] = o;
      } else {
        a.cur[to + c] = cv[r] + o;
      }
    }
  }
}
__global__ void __launch_bounds__(256) k_conv_resid_mr(ConvResidArgs a) {
  const int c = blockIdx.x * 256 + threadIdx.x;
  if (c < 2304) conv_resid_elem_mr(a, c, blockIdx.z);
}
__global__ void __launch_bounds__(256) k_conv_resid(ConvResidArgs a) {
  const int c = blockIdx.x * 256 + threadIdx.x;
  if (c < 2304) conv_resid_elem(a, c, blockIdx.z);
}

// Attention-residual mix, one block (16 waves) per query row. Optional prologue: the previous
// op's post-conv + residual (single-row jobs); optional epilogue: RMS norm with weights.
template <bool MR>
__global__ void __launch_bounds__(512) k_mix(MixJob j, float eps) {
  __shared__ float sc[8], coef[8], red[16];
  const int q = blockIdx.x, tok = blockIdx.y, wave = threadIdx.x / 32, lane = threadIdx.x % 32;
  const size_t toff = (size_t)tok * 2304;
  if (j.has_cr) {
    if constexpr (MR) conv_resid_block_mr(j.cr, tok);
    else conv_resid_block(j.cr, tok);
    __syncthreads();
  }
  const float* qv = j.query + (size_t)q * 2304;
  if (wave < j.n) {
    const float4* xp = (const float4*)(j.st[wave] + toff);
    const float4* qp = (const float4*)qv;
    float ss = 0.f, d = 0.f;
#pragma unroll
    for (int i = 0; i < 18; ++i) {
      const float4 x = xp[lane + 32 * i], y = qp[lane + 32 * i];
      ss += x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w;
      d += x.x * y.x + x.y * y.y + x.z * y.z + x.w * y.w;
    }
    ss = wsum(ss);
    d = wsum(d);
    if (lane == 0) {
      const float inv = rsqrtf(ss * (1.0f / 2304.0f) + eps);
      sc[wave] = d * inv * 0.020833333f;  // 1/sqrt(2304) = 1/48
      coef[wave] = inv;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float m = -INFINITY;
    for (int i = 0; i < j.n; ++i) m = fmaxf(m, sc[i]);
    float z = 0.f;
    for (int i = 0; i < j.n; ++i) {
      sc[i] = expf(sc[i] - m);
      z += sc[i];
    }
    for (int i = 0; i < j.n; ++i) sc[i] = sc[i] / z * coef[i];
  }
  __syncthreads();
  float4 a[2];
  float ssq = 0.f;
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int i = threadIdx.x + r * 512;
    a[r] = make_float4(0.f, 0.f, 0.f, 0.f);
    if (i < 576) {
      for (int k = 0; k < j.n; ++k) {
        const float4 x = ((const float4*)(j.st[k] + toff))[i];
        a[r].x += sc[k] * x.x; a[r].y += sc[k] * x.y; a[r].z += sc[k] * x.z; a[r].w += sc[k] * x.w;
      }
      ssq += a[r].x * a[r].x + a[r].y * a[r].y + a[r].z * a[r].z + a[r].w * a[r].w;
    }
  }
  float scale = 1.0f;
  const float* nw = j.nw[q];
  if (nw) {
    ssq = wsum(ssq);
    if (lane == 0) red[wave] = ssq;
    __syncthreads();
    float t = 0.f;
    for (int k = 0; k < 16; ++k) t += red[k];
    scale = rsqrtf(t * (1.0f / 2304.0f) + eps);
  }
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int i = threadIdx.x + r * 512;
    if (i < 576) {
      float4 o = a[r];
      if (nw) {
        const float4 w = ((const float4*)nw)[i];
        o.x *= scale * w.x; o.y *= scale * w.y; o.z *= scale * w.z; o.w *= scale * w.w;
      }
      float* ob = j.outq[q] ? j.outq[q] + toff
                            : j.out + (j.ostride ? (size_t)tok * j.ostride : (size_t)tok * j.nq * 2304) + (size_t)q * 2304;
      ((float4*)ob)[i] = o;
    }
  }
}

struct NormP {
  NormJobs j;
};
__global__ void __launch_bounds__(256) k_rms(NormJobs j, float eps) {
  __shared__ float red[32];
  const int r = blockIdx.x, tk = blockIdx.y;
  const float* x = j.x[r] + (size_t)tk * j.xs[r];
  const float* w = j.w[r];
  float* o = j.out[r] + (size_t)tk * j.os[r];
  float ss = 0.f;
  for (int i = threadIdx.x; i < 2304; i += blockDim.x) ss += x[i] * x[i];
  const float tot = block_sum(ss, red);
  const float inv = rsqrtf(tot * (1.0f / 2304.0f) + eps);
  for (int i = threadIdx.x; i < 2304; i += blockDim.x) o[i] = x[i] * inv * (w ? w[i] : 1.0f);
}

// Loop controller gate: g64[row] = down[row] . [x/rms(x); msg/rms(msg)]. One block per (row, stream).
__device__ __forceinline__ float4 ld4w(const void* base, size_t i4, bool h) {  // 4 weights starting at element 4*i4
  if (!h) return ((const float4*)base)[i4];
  const uint2 u = ((const uint2*)base)[i4];
  return make_float4(__half2float(__ushort_as_half((unsigned short)(u.x & 0xFFFFu))),
                     __half2float(__ushort_as_half((unsigned short)(u.x >> 16))),
                     __half2float(__ushort_as_half((unsigned short)(u.y & 0xFFFFu))),
                     __half2float(__ushort_as_half((unsigned short)(u.y >> 16))));
}
template <bool H>
__global__ void __launch_bounds__(256) k_ctl_gate(CtlJobs j, float eps) {
  __shared__ float red[4][8];
  const int row = blockIdx.x, s = blockIdx.y, tk = blockIdx.z;
  const float4* x = (const float4*)(j.x[s] + (size_t)tk * j.xs[s]);
  const float4* m = (const float4*)(j.msg[s] + (size_t)tk * j.ms[s]);
  const size_t drow = (size_t)row * 1152;  // 4608 / 4 groups of four weights per row
  float sx = 0.f, sm = 0.f, dx = 0.f, dm = 0.f;
  for (int i = threadIdx.x; i < 576; i += blockDim.x) {
    const float4 xv = x[i], mv = m[i], d0 = ld4w(j.down[s], drow + i, H), d1 = ld4w(j.down[s], drow + 576 + i, H);
    sx += xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w;
    sm += mv.x * mv.x + mv.y * mv.y + mv.z * mv.z + mv.w * mv.w;
    dx += d0.x * xv.x + d0.y * xv.y + d0.z * xv.z + d0.w * xv.w;
    dm += d1.x * mv.x + d1.y * mv.y + d1.z * mv.z + d1.w * mv.w;
  }
  sx = wsum(sx); sm = wsum(sm); dx = wsum(dx); dm = wsum(dm);
  const int wid = threadIdx.x / 32, lane = threadIdx.x % 32;
  if (lane == 0) {
    red[0][wid] = sx; red[1][wid] = sm; red[2][wid] = dx; red[3][wid] = dm;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float t[4] = {0.f, 0.f, 0.f, 0.f};
    for (int w = 0; w < 8; ++w)
      for (int k = 0; k < 4; ++k) t[k] += red[k][w];
    j.g64[s][(size_t)tk * j.gs[s] + row] = t[2] * rsqrtf(t[0] * (1.0f / 2304.0f) + eps) + t[3] * rsqrtf(t[1] * (1.0f / 2304.0f) + eps);
  }
}
template <bool H>
__global__ void __launch_bounds__(256) k_ctl_apply(CtlJobs j) {
  const int i = blockIdx.x * 256 + threadIdx.x, s = blockIdx.y, tk = blockIdx.z;
  if (i >= 2304) return;
  const float4* g = (const float4*)(j.g64[s] + (size_t)tk * j.gs[s]);
  float ga = 0.f, fo = 0.f;
#pragma unroll
  for (int k = 0; k < 16; ++k) {
    const float4 gv = g[k], a = ld4w(j.gain[s], (size_t)i * 16 + k, H), b = ld4w(j.forget[s], (size_t)i * 16 + k, H);
    ga += a.x * gv.x + a.y * gv.y + a.z * gv.z + a.w * gv.w;
    fo += b.x * gv.x + b.y * gv.y + b.z * gv.z + b.w * gv.w;
  }
  const float cv = j.x[s][(size_t)tk * j.xs[s] + i];
  j.out[s][(size_t)tk * j.os[s] + i] = cv - tanhf(fo) * cv + tanhf(ga) * j.msg[s][(size_t)tk * j.ms[s] + i];
}

// y[r] = W[r] . x for a few small matrices (K = 2304). One wave per row.
__device__ __forceinline__ float ld_w(const void* w, int type, size_t i) {
  if (type == 0) return ((const float*)w)[i];
  if (type == 1) return __half2float(((const __half*)w)[i]);
  return __uint_as_float((uint32_t)((const uint16_t*)w)[i] << 16);
}
__global__ void __launch_bounds__(256) k_tiny(TinyJobs j) {
  const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
  int row = blockIdx.x * 8 + wave;
  int m = 0;
  while (m < j.n && row >= j.rows[m]) {
    row -= j.rows[m];
    ++m;
  }
  if (m >= j.n) return;
  const void* W = j.w[m];
  const int ty = j.type[m];
  const float* x = j.x[m] + (size_t)blockIdx.y * j.xs[m];
  float s = 0.f;
  for (int i = lane; i < 2304; i += 32) s += ld_w(W, ty, (size_t)row * 2304 + i) * x[i];
  s = wsum(s);
  if (lane == 0) j.y[m][(size_t)blockIdx.y * j.ys[m] + row] = s;
}

// K/V conv (+ conv-state update), K clip / V rms, f16 KV write. One block per kv head.
template <bool MR>
__global__ void __launch_bounds__(128) k_kvprep(KvPrepArgs a) {
  __shared__ float red[32];
  const int kvh = blockIdx.x, d = threadIdx.x, c = kvh * 128 + d;
  const int tk = blockIdx.z;
  const int pos = a.pos[tk];
  const SlotEntry* slot = a.slot + tk * 48;
  const float* ak = a.k + (size_t)tk * a.ld;
  const float* av = a.v + (size_t)tk * a.ld;
  float* sk = slot->conv + (size_t)a.conv_off_k * 3 + (size_t)c * 3;
  float* sv = slot->conv + (size_t)a.conv_off_v * 3 + (size_t)c * 3;
  float kx[4], vx[4];
  float *ok = sk, *ov = sv;  // where the state after this row goes
  if constexpr (MR) {
    // see conv_resid_elem_mr: the earlier rows of the session in this launch supply their raw inputs
    const RowInfo ri = a.rows[tk];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      const bool hist = ri.m - 3 + i < 0;
      kx[i] = hist ? sk[ri.m + i] : a.k[(size_t)(tk - 3 + i) * a.ld + c];
      vx[i] = hist ? sv[ri.m + i] : a.v[(size_t)(tk - 3 + i) * a.ld + c];
    }
    kx[3] = ak[c];
    vx[3] = av[c];
    if (ri.snap) {
      ok = ri.snap + ((size_t)a.visit * 6656 + a.conv_off_k) * 3 + (size_t)c * 3;
      ov = ri.snap + ((size_t)a.visit * 6656 + a.conv_off_v) * 3 + (size_t)c * 3;
    }
  } else {
    kx[0] = sk[0]; kx[1] = sk[1]; kx[2] = sk[2]; kx[3] = ak[c];
    vx[0] = sv[0]; vx[1] = sv[1]; vx[2] = sv[2]; vx[3] = av[c];
  }
  float ky = 0.f, vy = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    ky += a.wk[(size_t)c * 4 + i] * kx[i];
    vy += a.wv[(size_t)c * 4 + i] * vx[i];
  }
  if (a.dk) {
    const float* dk = a.dk + (size_t)tk * a.ldd;
    const float* dv = a.dv + (size_t)tk * a.ldd;
#pragma unroll
    for (int lag = 0; lag < 4; ++lag) {
      ky += dk[kvh * 4 + lag] * kx[3 - lag];
      vy += dv[kvh * 4 + lag] * vx[3 - lag];
    }
  }
  ok[0] = kx[1]; ok[1] = kx[2]; ok[2] = kx[3];
  ov[0] = vx[1]; ov[1] = vx[2]; ov[2] = vx[3];
  const float kc = (kx[3] + ky) * a.scale, vc = (vx[3] + vy) * a.scale;
  const float kss = block_sum(kc * kc, red);
  const float vss = block_sum(vc * vc, red);
  const float kf = fminf(1.0f, a.qk_thr / (sqrtf(kss) + 1e-6f));
  const float vf = rsqrtf(vss * (1.0f / 128.0f) + a.eps);
  const int cell = pos % slot->cap;
  ((__half*)slot->k)[(size_t)cell * 1024 + c] = __float2half(kc * kf);
  ((__half*)slot->v)[(size_t)cell * 1024 + c] = __float2half(vc * vf);
}

// Plain-load sweep over byte ranges (MALL warm-up for the next GEMV). Result is discarded.
__global__ void __launch_bounds__(256) k_prefetch(PrefetchJob j, float* sink) {
  float acc = 0.f;
  for (int r = 0; r < j.n; ++r) {
    const float4* a = (const float4*)j.p[r];
    const size_t n = j.bytes[r] / 16;
#pragma unroll 8
    for (size_t i = (size_t)blockIdx.x * 256 + threadIdx.x; i < n; i += (size_t)gridDim.x * 256) {
      const float4 v = a[i];
      acc += v.x + v.w;
    }
  }
  if (acc == 1234.5f) sink[0] = acc;
}

// single-block counting sort of (token, slot) pairs by expert
__global__ void __launch_bounds__(1024) k_route_build(const int* ids, int P, int E, int* off, int* sorted) {
  __shared__ int cnt[256], cur[256];
  for (int e = threadIdx.x; e < E; e += blockDim.x) cnt[e] = 0;
  __syncthreads();
  for (int p = threadIdx.x; p < P; p += blockDim.x) atomicAdd(&cnt[ids[p]], 1);
  __syncthreads();
  if (threadIdx.x == 0) {
    int run = 0;
    for (int e = 0; e < E; ++e) {
      off[e] = run;
      cur[e] = run;
      run += cnt[e];
    }
    off[E] = run;
  }
  __syncthreads();
  for (int p = threadIdx.x; p < P; p += blockDim.x) sorted[atomicAdd(&cur[ids[p]], 1)] = p;
}


// ---------------- prefill: tiled SIMT kernels for the skinny F32 matrices ----------------
// out[t][r] = sum_k W[r][k] * in[t][k] with the input read from up to two segments (x, msg), each
// RMS-normalised per token when `rms`. Block: 8 tokens x 64 rows, K in chunks of 128.
struct GateP {
  const float* x; int ldx;
  const float* msg; int ldm;
  const float* W; int ldw;
  int k1, k2, rms;
  float* out; int ldo;
  int rows;  // valid output rows of the tile range (<= 64 per tile; 0 = all)
};
struct GateJobs {
  GateP j[3];
};
__global__ void __launch_bounds__(256) k_gate_p(GateJobs jobs, int T, float eps) {
  const GateP& p = jobs.j[blockIdx.z];
  __shared__ float Wt[64][129];
  __shared__ float xs[8][128];
  __shared__ float inv[2][8];
  const int tid = threadIdx.x, t0 = blockIdx.x * 8, r0 = blockIdx.y * 64;
  const int wave = tid / 32, lane = tid % 32;
  // per-token inverse RMS of each segment (wave = token)
  {
    const int t = min(t0 + wave, T - 1);
    float s1 = 1.f, s2 = 1.f;
    if (p.rms) {
      float a = 0.f, b = 0.f;
      for (int i = lane; i < p.k1; i += 32) { float v = p.x[(size_t)t * p.ldx + i]; a += v * v; }
      for (int i = lane; i < p.k2; i += 32) { float v = p.msg[(size_t)t * p.ldm + i]; b += v * v; }
      a = wsum(a);
      b = wsum(b);
      s1 = rsqrtf(a / p.k1 + eps);
      s2 = p.k2 ? rsqrtf(b / p.k2 + eps) : 1.f;
    }
    if (lane == 0) { inv[0][wave] = s1; inv[1][wave] = s2; }
  }
  __syncthreads();
  const int r = tid & 63, tg = tid >> 6;
  float a0 = 0.f, a1 = 0.f;
  const int ktot = p.k1 + p.k2;
  for (int kc = 0; kc < ktot; kc += 128) {
    __syncthreads();
    // weight tile 64 x 128
    for (int i = tid; i < 64 * 32; i += 256) {
      const int row = i >> 5, c4 = i & 31;
      const float4 v = *(const float4*)(p.W + (size_t)(r0 + row) * p.ldw + kc + c4 * 4);
      Wt[row][c4 * 4] = v.x; Wt[row][c4 * 4 + 1] = v.y; Wt[row][c4 * 4 + 2] = v.z; Wt[row][c4 * 4 + 3] = v.w;
    }
    // activation tile 8 x 128
    const bool seg2 = kc >= p.k1;
    const float* src = seg2 ? p.msg : p.x;
    const int ld = seg2 ? p.ldm : p.ldx, kk = seg2 ? kc - p.k1 : kc;
    for (int i = tid; i < 8 * 32; i += 256) {
      const int tk = i >> 5, c4 = i & 31;
      const int t = min(t0 + tk, T - 1);
      float4 v = *(const float4*)(src + (size_t)t * ld + kk + c4 * 4);
      const float sc = inv[seg2 ? 1 : 0][tk];
      xs[tk][c4 * 4] = v.x * sc; xs[tk][c4 * 4 + 1] = v.y * sc; xs[tk][c4 * 4 + 2] = v.z * sc; xs[tk][c4 * 4 + 3] = v.w * sc;
    }
    __syncthreads();
#pragma unroll 8
    for (int k = 0; k < 128; ++k) {
      const float w = Wt[r][k];
      a0 += w * xs[2 * tg][k];
      a1 += w * xs[2 * tg + 1][k];
    }
  }
  const int ta = t0 + 2 * tg, tb = ta + 1;
  if (p.rows && r0 + r >= p.rows) return;
  if (ta < T) p.out[(size_t)ta * p.ldo + r0 + r] = a0;
  if (tb < T) p.out[(size_t)tb * p.ldo + r0 + r] = a1;
}

// out = x - tanh(forget) x + tanh(gain) msg with gain/forget = W g64 (K = 64). Block: TT tokens x 64
// channels. TT = 32 with 512 threads (4 tokens per thread): the 64x64 gain/forget tiles are read from
// L2 once per block instead of once per 8 tokens, which is what this kernel was paying for (the
// per-token weight traffic drops 4x). The per-output K accumulation order is unchanged.
constexpr int kCtlTT = 32, kCtlTPT = 4;
__global__ void __launch_bounds__(512) k_ctl_apply_p(CtlPJobs jobs, int T) {
  const CtlPJob& p = jobs.j[blockIdx.z];
  __shared__ float wg[64][65], wf[64][65];
  __shared__ float gs[kCtlTT][64];
  const int tid = threadIdx.x, t0 = blockIdx.x * kCtlTT, c0 = blockIdx.y * 64;
  for (int i = tid; i < 64 * 16; i += 512) {
    const int row = i >> 4, c4 = i & 15;
    const float4 a = *(const float4*)(p.gain + (size_t)(c0 + row) * 64 + c4 * 4);
    const float4 b = *(const float4*)(p.forget + (size_t)(c0 + row) * 64 + c4 * 4);
    wg[row][c4 * 4] = a.x; wg[row][c4 * 4 + 1] = a.y; wg[row][c4 * 4 + 2] = a.z; wg[row][c4 * 4 + 3] = a.w;
    wf[row][c4 * 4] = b.x; wf[row][c4 * 4 + 1] = b.y; wf[row][c4 * 4 + 2] = b.z; wf[row][c4 * 4 + 3] = b.w;
  }
  for (int i = tid; i < kCtlTT * 16; i += 512) {
    const int tk = i >> 4, c4 = i & 15;
    const int t = min(t0 + tk, T - 1);
    const float4 v = *(const float4*)(p.g64 + (size_t)t * 64 + c4 * 4);
    gs[tk][c4 * 4] = v.x; gs[tk][c4 * 4 + 1] = v.y; gs[tk][c4 * 4 + 2] = v.z; gs[tk][c4 * 4 + 3] = v.w;
  }
  __syncthreads();
  const int c = tid & 63, tg = tid >> 6;
  float ga[kCtlTPT], fo[kCtlTPT];
#pragma unroll
  for (int u = 0; u < kCtlTPT; ++u) { ga[u] = 0.f; fo[u] = 0.f; }
#pragma unroll 4
  for (int k = 0; k < 64; ++k) {
    const float wgv = wg[c][k], wfv = wf[c][k];
#pragma unroll
    for (int u = 0; u < kCtlTPT; ++u) {
      const float g = gs[tg * kCtlTPT + u][k];
      ga[u] += wgv * g;
      fo[u] += wfv * g;
    }
  }
#pragma unroll
  for (int u = 0; u < kCtlTPT; ++u) {
    const int tk = tg * kCtlTPT + u, t = t0 + tk;
    if (t < T) {
      const float cv = p.x[(size_t)t * p.ldx + c0 + c];
      p.out[(size_t)t * 2304 + c0 + c] = cv - tanhf(fo[u]) * cv + tanhf(ga[u]) * p.msg[(size_t)t * p.ldm + c0 + c];
    }
  }
}


// sigmoid + biased top-8 + renormalise, one wave per token (router logits [T][128])
__global__ void __launch_bounds__(256) k_topk_p(const float* __restrict__ logit, const float* __restrict__ bias,
                                                 int* __restrict__ ids, float* __restrict__ wts, int T,
                                                 float rscale) {
  const int lane = threadIdx.x % 32, tok = blockIdx.x * 8 + threadIdx.x / 32;
  if (tok >= T) return;
  float sc[4], sel[4];
#pragma unroll
  for (int k = 0; k < 4; k++) {
    int i = lane + 32 * k;
    float l = logit[(size_t)tok * 128 + i];
    sc[k] = 1.f / (1.f + expf(-l));
    sel[k] = sc[k] + bias[i];
  }
  float ssum = 0.f;
  float myw[8];
  int myi[8];
#pragma unroll
  for (int r = 0; r < 8; r++) {
    float bv = -1e30f;
    int bi = 0;
#pragma unroll
    for (int k = 0; k < 4; k++)
      if (sel[k] > bv) {
        bv = sel[k];
        bi = lane + 32 * k;
      }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      float ov = __shfl_xor(bv, o, 32);
      int oi = __shfl_xor(bi, o, 32);
      if (ov > bv || (ov == bv && oi < bi)) {
        bv = ov;
        bi = oi;
      }
    }
    float scv = 0.f;
#pragma unroll
    for (int k = 0; k < 4; k++)
      if (lane + 32 * k == bi) {
        scv = sc[k];
        sel[k] = -1e30f;
      }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) scv += __shfl_xor(scv, o, 32);
    myw[r] = scv;
    myi[r] = bi;
    ssum += scv;
  }
  if (lane < 8) {
    float w = 0.f;
    int id = 0;
#pragma unroll
    for (int r = 0; r < 8; r++)
      if (r == lane) {
        w = myw[r];
        id = myi[r];
      }
    ids[(size_t)tok * 8 + lane] = id;
    wts[(size_t)tok * 8 + lane] = w / (ssum + 1e-20f) * rscale;
  }
}

// ---------------- attention decode (lab/attn/attn_kernels.h, device-side position) ----------------
namespace ia {
constexpr int HD = 128, NH = 24, NKV = 8, RD = 16, WIN = 1024, CELL = NKV * HD;
typedef _Float16 h16;
typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef _Float16 v2h __attribute__((ext_vector_type(2)));
typedef float v8f __attribute__((ext_vector_type(8)));

__device__ __forceinline__ float tau_of(int pos, float floor_, float alpha) {
  float x = (float)(pos + 1) / floor_;
  return 1.0f + alpha * logf(fmaxf(1.0f, x));
}
__device__ __forceinline__ v16h cat16(uint4 a, uint4 b) {
  union { uint4 u[2]; v16h v; } x;
  x.u[0] = a;
  x.u[1] = b;
  return x.v;
}
__device__ __forceinline__ v2h as_v2h(uint32_t u) { return __builtin_bit_cast(v2h, u); }
__device__ __forceinline__ void cbar() { asm volatile("" ::: "memory"); }
__device__ __forceinline__ float2 h2_to_f2(uint32_t u) {
  v2h v = as_v2h(u);
  return make_float2((float)v[0], (float)v[1]);
}
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

// NQ = 1 (3 rows: the 3 q heads of one kv head)
template <int NW>
__global__ __launch_bounds__(NW * 32) void attn_decode(AttnArgs a) {
  constexpr int NR = 3;
  __shared__ __attribute__((aligned(16))) __half sQ[16][HD + 8];
  __shared__ float sR[16][16];
  __shared__ __attribute__((aligned(16))) float sP[NW][16][16];
  __shared__ float sA[NW][16];
  __shared__ __attribute__((aligned(16))) float sO[NW][NR][HD];
  __shared__ float sM[NW][16], sL[NW][16];
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  const int kvh = blockIdx.x, split = blockIdx.y, tk = blockIdx.z;
  const int p0 = a.pos[tk];
  const SlotEntry* slot = a.slot + tk * 48;
  const float* aq_ = a.q + (size_t)tk * a.ldp;
  const float* ar_ = a.r + (size_t)tk * a.ldp;
  float* ws_o = a.ws_o + (size_t)tk * a.ws_tok * 128;
  float* ws_m = a.ws_m + (size_t)tk * a.ws_tok;
  float* ws_l = a.ws_l + (size_t)tk * a.ws_tok;
  const __half* kc = (const __half*)slot->k;
  const __half* vc = (const __half*)slot->v;
  const int cap = slot->cap;
  const int kb = a.swa ? max(0, p0 - (WIN - 1)) : 0;
  // Multi-row launch (global layers): this row splits by the bucket of its own position, as a single-token decode
  // at that position would; splits beyond its own count have nothing to do.
  const int chunk = a.rows ? a.rows[tk].chunk : a.chunk;
  if (a.rows && split >= a.rows[tk].nsplit) return;
  const int lo = kb + split * chunk;
  const int hi = min(lo + chunk, p0 + 1);
  const __half* bt = (const __half*)a.bt;

  for (int r = w; r < 16; r += NW) {
    if (r < NR) {
      const int h = kvh * 3 + r;
      const float tau = a.swa ? 1.0f : tau_of(p0, a.tau_floor, a.tau_alpha);
      float4 x = ((const float4*)(aq_ + (size_t)h * HD))[lane];
      float4 s = ((const float4*)(a.qscale + h * HD))[lane];
      x.x *= s.x; x.y *= s.y; x.z *= s.z; x.w *= s.w;
      float ss = wsum(x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w);
      float sc = fminf(1.0f, a.qk_thr / (sqrtf(ss) + 1e-6f)) * tau * 0.08838834764831845f;
      __half2* qp = (__half2*)&sQ[r][lane * 4];
      qp[0] = __floats2half2_rn(x.x * sc, x.y * sc);
      qp[1] = __floats2half2_rn(x.z * sc, x.w * sc);
      float rv = 0.f;
      if (lane < RD) rv = ar_[(size_t)h * RD + lane] * a.rclip[h * RD + lane];
      float rs = wsum(rv * rv);
      float rf = fminf(1.0f, a.r_thr / (sqrtf(rs) + 1e-6f)) * tau;
      if (lane < RD) sR[r][lane] = rv * rf;
    } else {
      __half2* qp = (__half2*)&sQ[r][lane * 4];
      qp[0] = __floats2half2_rn(0.f, 0.f);
      qp[1] = __floats2half2_rn(0.f, 0.f);
      if (lane < RD) sR[r][lane] = 0.f;
    }
  }
  __syncthreads();
  const int q16 = lane & 15, half = lane >> 4;
  v16h bq[8];
#pragma unroll
  for (int s = 0; s < 8; s++)
    bq[s] = cat16(*(const uint4*)&sQ[q16][s * 16], *(const uint4*)&sQ[q16][s * 16 + 8]);
  v2h rp[8];
#pragma unroll
  for (int c = 0; c < 8; c++) {
    rp[c][0] = (h16)sR[q16][2 * c];
    rp[c][1] = (h16)sR[q16][2 * c + 1];
  }
  const int posq = (q16 < NR) ? p0 : -1;

  float m = -INFINITY, l = 0.f;
  float acc[NR][4];
#pragma unroll
  for (int r = 0; r < NR; r++) acc[r][0] = acc[r][1] = acc[r][2] = acc[r][3] = 0.f;

  for (int jt = lo + w * 16; jt < hi; jt += NW * 16) {
    const int pk = min(jt + q16, hi - 1);
    const uint4* kp = (const uint4*)(kc + (size_t)(pk % cap) * CELL + kvh * HD);
    uint4 kr[16];
#pragma unroll
    for (int u = 0; u < 16; u++) kr[u] = kp[u];
    uint2 vr[16];
#pragma unroll
    for (int k = 0; k < 16; k++) {
      const int pv = min(jt + k, hi - 1);
      vr[k] = *(const uint2*)(vc + (size_t)(pv % cap) * CELL + kvh * HD + lane * 4);
    }
    v8f c = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int s = 0; s < 8; s++)
      c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(cat16(kr[2 * s], kr[2 * s + 1]), bq[s], c);

    float sv[8], mx = -INFINITY;
    const bool needb = (posq >= 0) && (posq - (jt + 15) < WIN);
    const bool anyb = __any(needb);
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const int pos = jt + 2 * i + half, d = posq - pos;
      const bool ok = (d >= 0) && (pos < hi) && (!a.swa || d < WIN);
      float s = c[i];
      if (anyb && ok && d < WIN) s += bias_dot(rp, bt, d);
      sv[i] = ok ? s : -INFINITY;
      mx = fmaxf(mx, sv[i]);
    }
    mx = fmaxf(mx, __shfl_xor(mx, 16, 32));
    const float mn = fmaxf(m, mx);
    float alpha = 1.f, ps = 0.f;
    if (mn != -INFINITY) {
      alpha = __expf(m - mn);
#pragma unroll
      for (int i = 0; i < 8; i++) {
        float p = __expf(sv[i] - mn);
        ps += p;
        sP[w][2 * i + half][q16] = p;
      }
    } else {
#pragma unroll
      for (int i = 0; i < 8; i++) sP[w][2 * i + half][q16] = 0.f;
    }
    l = l * alpha + ps;
    m = mn;
    sA[w][q16] = alpha;
    cbar();
    float al[NR];
#pragma unroll
    for (int r = 0; r < NR; r++) al[r] = sA[w][r];
#pragma unroll
    for (int r = 0; r < NR; r++) {
      acc[r][0] *= al[r]; acc[r][1] *= al[r]; acc[r][2] *= al[r]; acc[r][3] *= al[r];
    }
#pragma unroll
    for (int k = 0; k < 16; k++) {
      float2 v01 = h2_to_f2(vr[k].x), v23 = h2_to_f2(vr[k].y);
      float p[NR];
      {
        float4 t4 = *(const float4*)&sP[w][k][0];
        p[0] = t4.x; p[1] = t4.y; p[2] = t4.z;
      }
#pragma unroll
      for (int r = 0; r < NR; r++) {
        acc[r][0] = fmaf(p[r], v01.x, acc[r][0]);
        acc[r][1] = fmaf(p[r], v01.y, acc[r][1]);
        acc[r][2] = fmaf(p[r], v23.x, acc[r][2]);
        acc[r][3] = fmaf(p[r], v23.y, acc[r][3]);
      }
    }
    cbar();
  }
  const float ltot = l + __shfl_xor(l, 16, 32);
  if (lane < 16) {
    sM[w][lane] = m;
    sL[w][lane] = ltot;
  }
#pragma unroll
  for (int r = 0; r < NR; r++)
    *(float4*)&sO[w][r][lane * 4] = make_float4(acc[r][0], acc[r][1], acc[r][2], acc[r][3]);
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
    ws_o[(base + r) * HD + dim] = o;
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
    ws_m[base + tid] = M;
    ws_l[base + tid] = L;
  }
}

__global__ __launch_bounds__(128) void decode_reduce(AttnArgs a) {
  __shared__ float sc[1024];
  __shared__ float sred[4], sZ;
  const int h = blockIdx.x, kvh = h / 3, r = h % 3, dim = threadIdx.x, tk = blockIdx.z;
  const float* ws_o = a.ws_o + (size_t)tk * a.ws_tok * 128;
  const float* ws_m = a.ws_m + (size_t)tk * a.ws_tok;
  const float* ws_l = a.ws_l + (size_t)tk * a.ws_tok;
  const float sk = a.sink[tk * 24 + h];
  const int nsplit = a.rows ? a.rows[tk].nsplit : a.nsplit;
  float mmax = sk;
  for (int s = dim; s < nsplit; s += 128) {
    sc[s] = ws_m[((size_t)s * NKV + kvh) * 12 + r];
    mmax = fmaxf(mmax, sc[s]);
  }
  mmax = fmaxf(mmax, __shfl_xor(mmax, 16, 32));
  mmax = fmaxf(mmax, __shfl_xor(mmax, 8, 32));
  mmax = fmaxf(mmax, __shfl_xor(mmax, 4, 32));
  mmax = fmaxf(mmax, __shfl_xor(mmax, 2, 32));
  mmax = fmaxf(mmax, __shfl_xor(mmax, 1, 32));
  if ((dim & 31) == 0) sred[dim >> 5] = mmax;
  __syncthreads();
  const float M = fmaxf(fmaxf(sred[0], sred[1]), fmaxf(sred[2], sred[3]));
  float zp = 0.f;
  for (int s = dim; s < nsplit; s += 128) {
    const float ms = sc[s];
    const float e = (ms == -INFINITY) ? 0.f : __expf(ms - M);
    sc[s] = e;
    zp += e * ((ms == -INFINITY) ? 0.f : ws_l[((size_t)s * NKV + kvh) * 12 + r]);
  }
  zp = wsum(zp);
  __syncthreads();
  if ((dim & 31) == 0) sred[dim >> 5] = zp;
  __syncthreads();
  const float Z = __expf(sk - M) + sred[0] + sred[1] + sred[2] + sred[3];
  float o = 0.f;
#pragma unroll 8
  for (int s = 0; s < nsplit; s++) o += sc[s] * ws_o[(((size_t)s * NKV + kvh) * 12 + r) * HD + dim];
  o /= Z;
  if (a.gate) o *= 1.0f / (1.0f + __expf(-a.gate[(size_t)tk * a.ldp + (size_t)h * HD + dim]));
  a.out[(size_t)tk * 3072 + (size_t)h * HD + dim] = o;
}
}  // namespace ia


// ================================================================================================
// Prefill (T > 1): LDS-tiled f16-WMMA grouped GEMM (lab kern3.h / kern4.h `k_gemm4`), GPU-side
// expert grouping, and the WMMA flash attention (lab/attn/attn_kernels.h `attn_prefill`).
// ================================================================================================
typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef _Float16 v16h_a16 __attribute__((ext_vector_type(16), aligned(16)));
typedef uint32_t v4u_a16 __attribute__((ext_vector_type(4), aligned(16)));
typedef uint32_t v8u __attribute__((ext_vector_type(8)));

struct GP3 {
  const uint8_t* W;
  size_t wexp;
  const float* escale;
  const _Float16* ah;
  const int* actrow;
  const int* nitems;
  const int* item_e;
  const int* item_start;
  const int* item_n;
  float* out;
  _Float16* outh;
  int ldo;
  const int* postok;
  const int* pair;  // EPI 2: (token * 8 + slot) of each sorted position: deterministic, non-atomic write
  const float* posw;
  float beta, lb;
};

__device__ __forceinline__ void nib4_to_half(uint32_t n, uint32_t& a, uint32_t& b) {
  uint32_t m = n & 0x07070707u, t = (n << 4) & 0x80808080u;
  uint32_t sA = __builtin_amdgcn_perm(m, 0x0C0C0C0Cu, 0x05000400u),
           sB = __builtin_amdgcn_perm(m, 0x0C0C0C0Cu, 0x07000600u);
  a = __builtin_amdgcn_perm(0x4A484644u, 0x42403C00u, sA) | __builtin_amdgcn_perm(t, 0u, 0x050C040Cu);
  b = __builtin_amdgcn_perm(0x4A484644u, 0x42403C00u, sB) | __builtin_amdgcn_perm(t, 0u, 0x070C060Cu);
}
__device__ __forceinline__ v16h dq_nv16(uint32_t w0, uint32_t w1, float ds) {
  v8u r;
  uint32_t a, b;
  nib4_to_half(w0 & 0x0F0F0F0Fu, a, b); r[0] = a; r[1] = b;
  nib4_to_half(w1 & 0x0F0F0F0Fu, a, b); r[2] = a; r[3] = b;
  nib4_to_half((w0 >> 4) & 0x0F0F0F0Fu, a, b); r[4] = a; r[5] = b;
  nib4_to_half((w1 >> 4) & 0x0F0F0F0Fu, a, b); r[6] = a; r[7] = b;
  return __builtin_bit_cast(v16h, r) * (_Float16)ds;
}
__device__ __forceinline__ v16h dq_q8_16(v4u w, float d) {
  v8u r;
  uint32_t x[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
  for (int i = 0; i < 4; i++) {
    uint32_t u = x[i] ^ 0x80808080u;
    r[2 * i] = __builtin_amdgcn_perm(u, 0x64646464u, 0x00050004u);
    r[2 * i + 1] = __builtin_amdgcn_perm(u, 0x64646464u, 0x00070006u);
  }
  return (__builtin_bit_cast(v16h, r) - (_Float16)1152.0f) * (_Float16)d;
}

// EPI 0: f32 out[pos*ldo+row]; EPI 1: gate/up + SITU -> outh[pos*FF+j]; EPI 2: down, atomicAdd into
// out[postok[pos]*ldo+row] weighted by posw*expert scale.
template <int FMT, int EPI, int BN, int K, int RG>
__global__ void __launch_bounds__(64 * RG) k_gemm4(GP3 p, int rgn) {
  constexpr int FF = 1152, STR = 72, NT = 64 * RG, BM = 32 * RG, TTW = BN / 32, NK = K / 64,
                NCH = (BN * 8 + NT - 1) / NT;
  constexpr size_t RB = FMT == FMT_Q8 ? (K / 32) * 34 : (K / 64) * 36;
  __shared__ _Float16 Ws[2][BM * STR];
  __shared__ _Float16 As[2][BN * STR];
  const int total = *p.nitems * rgn;
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, ln = lane & 15, hi = lane >> 4,
            rg = wave % RG, th = wave / RG;
  for (int unit = blockIdx.x; unit < total; unit += gridDim.x) {
    const int item = unit / rgn, rgi = unit % rgn;
    const int e = p.item_e[item], start = p.item_start[item], n = p.item_n[item];
    const uint8_t* We = p.W + (size_t)e * p.wexp;
    int wr = tid >> 1, wh = tid & 1, srow;
    if (EPI == 1) {
      int h2 = wr >> 5, sub = (wr >> 4) & 1, r = wr & 15;
      srow = (sub ? FF : 0) + rgi * (16 * RG) + h2 * 16 + r;
    } else {
      srow = rgi * BM + wr;
    }
    const uint8_t* wp = We + (size_t)srow * RB;
    uint32_t raw[9];
    v4u areg[NCH];
    const _Float16* ap[NCH];
    int arow_l[NCH];
    bool aok[NCH];
#pragma unroll
    for (int i = 0; i < NCH; i++) {
      int idx = tid + NT * i, row = idx >> 3, ch = idx & 7;
      aok[i] = idx < BN * 8;
      arow_l[i] = row * STR + ch * 8;
      int pos = row < n ? start + row : start;
      int ar = p.actrow ? p.actrow[pos] : pos;
      ap[i] = p.ah + (size_t)ar * K + ch * 8;
    }
    auto gload = [&](int kb) {
      if (FMT == FMT_NVFP4) {
        const uint8_t* b = wp + (size_t)kb * 36;
        raw[0] = *(const uint32_t*)b;
        v4u q = *(const v4u_a4*)(b + 4 + 16 * wh);
        raw[1] = q.x; raw[2] = q.y; raw[3] = q.z; raw[4] = q.w;
      } else {
        const uint8_t* b = wp + (size_t)(2 * kb + wh) * 34;
        raw[0] = *(const uint16_t*)b;
        v4u a = *(const v4u_a2*)(b + 2), c = *(const v4u_a2*)(b + 18);
        raw[1] = a.x; raw[2] = a.y; raw[3] = a.z; raw[4] = a.w;
        raw[5] = c.x; raw[6] = c.y; raw[7] = c.z; raw[8] = c.w;
      }
#pragma unroll
      for (int i = 0; i < NCH; i++)
        if (aok[i]) areg[i] = *(const v4u*)(ap[i] + kb * 64);
    };
    auto sstore = [&](int buf) {
      _Float16* dst = &Ws[buf][wr * STR + 32 * wh];
      if (FMT == FMT_NVFP4) {
#pragma unroll
        for (int s2 = 0; s2 < 2; s2++) {
          float ds = ue4m3((raw[0] >> (16 * wh + 8 * s2)) & 0xFFu);
          *(v16h_a16*)(dst + 16 * s2) = dq_nv16(raw[1 + 2 * s2], raw[2 + 2 * s2], ds);
        }
      } else {
        float d = __half2float(__ushort_as_half((unsigned short)raw[0]));
        *(v16h_a16*)dst = dq_q8_16((v4u){raw[1], raw[2], raw[3], raw[4]}, d);
        *(v16h_a16*)(dst + 16) = dq_q8_16((v4u){raw[5], raw[6], raw[7], raw[8]}, d);
      }
#pragma unroll
      for (int i = 0; i < NCH; i++)
        if (aok[i]) *(v4u_a16*)&As[buf][arow_l[i]] = areg[i];
    };
    v8f facc[TTW][2];
#pragma unroll
    for (int t = 0; t < TTW; t++) {
      facc[t][0] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
      facc[t][1] = facc[t][0];
    }
    gload(0);
    sstore(0);
    __syncthreads();
    for (int kb = 0; kb < NK; kb++) {
      int cur = kb & 1;
      if (kb + 1 < NK) gload(kb + 1);
#pragma unroll
      for (int ks = 0; ks < 4; ks++) {
        v16h bf[2];
#pragma unroll
        for (int r = 0; r < 2; r++) bf[r] = *(const v16h_a16*)&Ws[cur][(rg * 32 + 16 * r + ln) * STR + ks * 16];
#pragma unroll
        for (int t = 0; t < TTW; t++) {
          v16h a = *(const v16h_a16*)&As[cur][(th * (BN / 2) + 16 * t + ln) * STR + ks * 16];
#pragma unroll
          for (int r = 0; r < 2; r++)
            facc[t][r] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, bf[r], facc[t][r]);
        }
      }
      if (kb + 1 < NK) sstore(cur ^ 1);
      __syncthreads();
    }
    float esc = p.escale ? p.escale[e] : 1.f;
#pragma unroll
    for (int t = 0; t < TTW; t++)
#pragma unroll
      for (int i = 0; i < 8; i++) {
        int pl = th * (BN / 2) + 16 * t + 2 * i + hi;
        if (pl >= n) continue;
        int pd = start + pl;
        if (EPI == 0) {
          int row = rgi * BM + rg * 32 + ln;
          p.out[(size_t)pd * p.ldo + row] = facc[t][0][i];
          p.out[(size_t)pd * p.ldo + row + 16] = facc[t][1][i];
        } else if (EPI == 1) {
          int j = rgi * (16 * RG) + rg * 16 + ln;
          p.outh[(size_t)pd * FF + j] =
              (_Float16)situ(facc[t][0][i] * esc, facc[t][1][i] * esc, p.beta, p.lb);
        } else {
          float w = p.posw[pd] * esc;
          // One row per (token, slot) pair; k_sum_pairs adds the 8 slots in a fixed order. (An f32
          // atomicAdd here made two identical prefills differ in the last bits.)
          size_t o = (size_t)p.pair[pd] * p.ldo;
          int row = rgi * BM + rg * 32 + ln;
          p.out[o + row] = w * facc[t][0][i];
          p.out[o + row + 16] = w * facc[t][1][i];
        }
      }
    __syncthreads();
  }
}

// y[t][r] = sum over the 8 slots of pairs[t * 8 + k][r], in slot order (deterministic).
__global__ void k_sum_pairs(const float* __restrict__ pairs, float* __restrict__ y, int d) {
  const size_t t = blockIdx.x;
  for (int r = threadIdx.x * 4; r < d; r += blockDim.x * 4) {
    float4 a = *(const float4*)(pairs + (t * 8) * d + r);
#pragma unroll
    for (int k = 1; k < 8; k++) {
      float4 b = *(const float4*)(pairs + (t * 8 + k) * d + r);
      a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
    }
    *(float4*)(y + t * d + r) = a;
  }
}

__global__ void k_cvt_f16(const float* __restrict__ x, _Float16* __restrict__ h, size_t n) {
  size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (i + 3 < n) {
    float4 v = *(const float4*)(x + i);
    h[i] = (_Float16)v.x; h[i + 1] = (_Float16)v.y; h[i + 2] = (_Float16)v.z; h[i + 3] = (_Float16)v.w;
  } else {
    for (; i < n; ++i) h[i] = (_Float16)x[i];
  }
}

// GPU-side expert grouping (E = 128)
__global__ void k_hist(const int* __restrict__ ids, int P, int* cnt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < P) atomicAdd(&cnt[ids[i]], 1);
}
__global__ void k_scan_items(const int* cnt, int tile, int* goff, int* fill, int* nitems, int* ie,
                             int* is, int* in) {
  __shared__ int sc[128], st[128];
  int e = threadIdx.x;
  int c = cnt[e], tiles = (c + tile - 1) / tile;
  sc[e] = c;
  st[e] = tiles;
  __syncthreads();
  for (int o = 1; o < 128; o <<= 1) {
    int a = e >= o ? sc[e - o] : 0, b = e >= o ? st[e - o] : 0;
    __syncthreads();
    sc[e] += a;
    st[e] += b;
    __syncthreads();
  }
  int ec = sc[e] - c, et = st[e] - tiles;
  goff[e] = ec;
  fill[e] = 0;
  for (int t = 0; t < tiles; t++) {
    int idx = et + t;
    ie[idx] = e;
    is[idx] = ec + t * tile;
    in[idx] = min(tile, c - t * tile);
  }
  if (e == 127) {
    *nitems = st[127];
    goff[128] = sc[127];
  }
}
__global__ void k_scatter(const int* __restrict__ ids, const float* __restrict__ w, int P,
                          const int* goff, int* fill, int* postok, int* pair, float* posw) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= P) return;
  int e = ids[i];
  int pos = goff[e] + atomicAdd(&fill[e], 1);
  postok[pos] = i / 8;
  pair[pos] = i;
  posw[pos] = w[i];
}

// ---------------- prefill attention prologue / KV write / flash attention ----------------
__global__ __launch_bounds__(128) void k_kv_write(const float* __restrict__ k, const float* __restrict__ v,
                                                  int n, int pos0, int cap, __half* kc, __half* vc,
                                                  float qk_thr, float eps) {
  using namespace ia;
  const int lane = threadIdx.x & 31, gw = blockIdx.x * 4 + (threadIdx.x >> 5);
  if (gw >= n * NKV) return;
  const int t = gw / NKV, kvh = gw % NKV;
  const size_t off = (size_t)t * CELL + kvh * HD;
  float4 kx = ((const float4*)(k + off))[lane];
  float4 vx = ((const float4*)(v + off))[lane];
  float ks = wsum(kx.x * kx.x + kx.y * kx.y + kx.z * kx.z + kx.w * kx.w);
  float vs = wsum(vx.x * vx.x + vx.y * vx.y + vx.z * vx.z + vx.w * vx.w);
  float kf = fminf(1.0f, qk_thr / (sqrtf(ks) + 1e-6f));
  float vf = rsqrtf(vs * (1.0f / HD) + eps);
  int cell = (pos0 + t) % cap;
  __half2* kp = (__half2*)(kc + (size_t)cell * CELL + kvh * HD + lane * 4);
  __half2* vp = (__half2*)(vc + (size_t)cell * CELL + kvh * HD + lane * 4);
  kp[0] = __floats2half2_rn(kx.x * kf, kx.y * kf);
  kp[1] = __floats2half2_rn(kx.z * kf, kx.w * kf);
  vp[0] = __floats2half2_rn(vx.x * vf, vx.y * vf);
  vp[1] = __floats2half2_rn(vx.z * vf, vx.w * vf);
}

__global__ __launch_bounds__(128) void k_qr_prep(const float* __restrict__ q, const float* __restrict__ r,
                                                 const float* __restrict__ qscale, const float* __restrict__ rclip,
                                                 int n, int pos0, int swa, float qk_thr, float r_thr,
                                                 float tau_floor, float tau_alpha, __half* qn, float* rn) {
  using namespace ia;
  const int lane = threadIdx.x & 31, gw = blockIdx.x * 4 + (threadIdx.x >> 5);
  if (gw >= n * NH) return;
  const int t = gw / NH, h = gw % NH;
  const float tau = swa ? 1.0f : tau_of(pos0 + t, tau_floor, tau_alpha);
  float4 x = ((const float4*)(q + (size_t)gw * HD))[lane];
  float4 s = ((const float4*)(qscale + h * HD))[lane];
  x.x *= s.x; x.y *= s.y; x.z *= s.z; x.w *= s.w;
  float ss = wsum(x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w);
  float sc = fminf(1.0f, qk_thr / (sqrtf(ss) + 1e-6f)) * tau * 0.08838834764831845f;
  __half2* qp = (__half2*)(qn + (size_t)gw * HD + lane * 4);
  qp[0] = __floats2half2_rn(x.x * sc, x.y * sc);
  qp[1] = __floats2half2_rn(x.z * sc, x.w * sc);
  float rv = 0.f;
  if (lane < RD) rv = r[(size_t)gw * RD + lane] * rclip[h * RD + lane];
  float rs = wsum(rv * rv);
  float rf = fminf(1.0f, r_thr / (sqrtf(rs) + 1e-6f)) * tau;
  if (lane < RD) rn[(size_t)gw * RD + lane] = rv * rf;
}

struct PfArgs {
  const __half* kc;
  const __half* vc;
  int cap, swa, U, p0;
  const __half* qn;
  const float* rn;
  const float* sink;
  const float* gate;
  const __half* bt;
  float* out;
};
namespace ia {
constexpr int PF_KT = 64, PF_KS = HD + 8, PF_VS = PF_KT + 8;
__device__ __forceinline__ uint32_t as_u32(v2h v) { return __builtin_bit_cast(uint32_t, v); }
}  // namespace ia

template <int NTT>
__global__ __launch_bounds__(NTT * 96) void attn_prefill(PfArgs a) {
  using namespace ia;
  constexpr int NT = NTT * 96;
  __shared__ __attribute__((aligned(16))) __half sK[PF_KT][PF_KS];
  __shared__ __attribute__((aligned(16))) __half sVt[HD][PF_VS];
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  const int kvh = blockIdx.y;
  const int hh = w % 3, tt = w / 3, h = kvh * 3 + hh;
  const int q16 = lane & 15, half = lane >> 4;
  const int tbase = blockIdx.x * 16 * NTT;
  const int tb = tbase + tt * 16;
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
    for (int c = 0; c < 8; c++) {
      float2 f = rq[c];
      rp[c][0] = (h16)f.x;
      rp[c][1] = (h16)f.y;
    }
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
  const int wmin = a.p0 + tb, wmax = a.p0 + tb + 15;

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
      constexpr int RS = PF_VS / 2;
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
            cat16(*(const uint4*)&sK[kt * 16 + q16][s * 16], *(const uint4*)&sK[kt * 16 + q16][s * 16 + 8]),
            bq[s], c);
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
      for (int i = 0; i < 8; i++) {
        p[i] = __expf(sv[i] - mn);
        ps += p[i];
      }
      l = l * alpha + ps;
      m = mn;
      uint32_t mine[4], theirs[4];
#pragma unroll
      for (int i = 0; i < 4; i++) {
        v2h pr;
        pr[0] = (h16)p[2 * i];
        pr[1] = (h16)p[2 * i + 1];
        mine[i] = as_u32(pr);
        theirs[i] = __shfl_xor(mine[i], 16, 32);
      }
      v16h pf;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        v2h mv = as_v2h(mine[i]), tv = as_v2h(theirs[i]);
        pf[4 * i + 0] = half ? tv[0] : mv[0];
        pf[4 * i + 1] = half ? mv[0] : tv[0];
        pf[4 * i + 2] = half ? tv[1] : mv[1];
        pf[4 * i + 3] = half ? mv[1] : tv[1];
      }
#pragma unroll
      for (int dt = 0; dt < 8; dt++) {
        acc[dt] *= alpha;
        const __half* vp = &sVt[dt * 16 + q16][kt * 16];
        acc[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(cat16(*(const uint4*)vp, *(const uint4*)(vp + 8)),
                                                             pf, acc[dt]);
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

template <int N> using I = std::integral_constant<int, N>;
void Chk(const char* what) {
  const hipError_t e = hipGetLastError();
  if (e != hipSuccess)
    throw std::runtime_error(std::string("inkling2 fast ") + what + ": " + hipGetErrorString(e));
}

}  // namespace

int g_tune[4] = {0, 0, 0, 0};
int g_res = 1;  // 1: second int8 plane with the quantisation residual of the activations
void SetActResidual(bool on) { g_res = on ? 1 : 0; }
void SetTune(int proj, int gu, int down) {
  g_tune[0] = proj;
  g_tune[1] = gu;
  g_tune[2] = down;
}

// ---------------------------------------------------------------------------------------------
void ProjMulti(hipStream_t s, const ProjSpec& p, const float* x, int ni, float* y, int nb, int ldx, int ldy) {
  MultiP mp{};
  mp.ldx = ldx;
  mp.ldy = ldy;
  int cum = 0;
  for (int i = 0; i < p.n; ++i) {
    mp.W[i] = static_cast<const uint8_t*>(p.w[i]);
    mp.M[i] = p.m[i];
    mp.in[i] = p.in[i];
    mp.yoff[i] = cum;
    mp.cum[i] = cum;
    cum += p.m[i];
  }
  mp.cum[p.n] = cum;
  mp.n = p.n;
  mp.res = g_res;
  int trows_total = 0;
  mp.tn = p.tn;
  for (int i = 0; i < p.tn; ++i) {
    mp.tw[i] = p.tw[i];
    mp.ttype[i] = p.ttype[i] == 0 ? 0 : (p.ttype[i] == 1 ? 1 : 2);
    mp.trows[i] = p.trows[i];
    mp.tin[i] = p.tin[i];
    mp.ty[i] = p.ty[i];
    mp.tys[i] = p.tys[i];
    trows_total += p.trows[i];
  }
  auto run = [&](auto lpr, auto nit, auto bw) {
    constexpr int LPR = decltype(lpr)::value, NIT = decltype(nit)::value, BW = decltype(bw)::value;
    constexpr int RPB = BW * (WS / LPR) * NIT;
    mp.ntb = (trows_total + BW - 1) / BW;
    const dim3 grid((cum + RPB - 1) / RPB + mp.ntb);
    // One launch of `n` tokens starting at token `t0` (tokens are independent: any split gives the same results).
    auto launch = [&](int n, int t0) {
      MultiP m2 = mp;
      for (int i = 0; i < m2.tn; ++i) m2.ty[i] += (size_t)t0 * m2.tys[i];
      const float* x2 = x + (size_t)t0 * ldx;
      float* y2 = y + (size_t)t0 * ldy;
      if (ni == 1) {
        switch (n) {
          case 1: k_gemv_multi<2304, LPR, NIT, BW, 1, 1><<<grid, BW * WS, 0, s>>>(m2, x2, y2, cum); break;
          case 2: k_gemv_multi<2304, LPR, NIT, BW, 1, 2><<<grid, BW * WS, 0, s>>>(m2, x2, y2, cum); break;
          case 3: k_gemv_multi<2304, LPR, NIT, BW, 1, 3><<<grid, BW * WS, 0, s>>>(m2, x2, y2, cum); break;
          case 4: k_gemv_multi<2304, LPR, NIT, BW, 1, 4><<<grid, BW * WS, 0, s>>>(m2, x2, y2, cum); break;
          default:
            // 5..8 tokens (multi-row launches): only the default shape is instantiated
            if constexpr (LPR == 16 && NIT == 4 && BW == 4) {
              switch (n) {
                case 5: k_gemv_multi<2304, LPR, NIT, BW, 1, 5><<<grid, BW * WS, 0, s>>>(m2, x2, y2, cum); break;
                case 6: k_gemv_multi<2304, LPR, NIT, BW, 1, 6><<<grid, BW * WS, 0, s>>>(m2, x2, y2, cum); break;
                case 7: k_gemv_multi<2304, LPR, NIT, BW, 1, 7><<<grid, BW * WS, 0, s>>>(m2, x2, y2, cum); break;
                default: k_gemv_multi<2304, LPR, NIT, BW, 1, 8><<<grid, BW * WS, 0, s>>>(m2, x2, y2, cum); break;
              }
            } else {
              throw std::runtime_error("inkling2 fast: more than 4 tokens need the default projection shape");
            }
        }
      } else {
        // three inputs per token: more than 3 tokens would need more than 64 KiB of LDS
        switch (n) {
          case 1: k_gemv_multi<2304, LPR, NIT, BW, 3, 1><<<grid, BW * WS, 0, s>>>(m2, x2, y2, cum); break;
          case 2: k_gemv_multi<2304, LPR, NIT, BW, 3, 2><<<grid, BW * WS, 0, s>>>(m2, x2, y2, cum); break;
          default: k_gemv_multi<2304, LPR, NIT, BW, 3, 3><<<grid, BW * WS, 0, s>>>(m2, x2, y2, cum); break;
        }
      }
    };
    if (ni == 1 || nb <= 3) {
      launch(nb, 0);
    } else {
      // 4 tokens run as two pairs (as in 0.1.2), 5..8 in groups of three (the last group 1..3)
      const int step = nb == 4 ? 2 : 3;
      for (int t0 = 0; t0 < nb; t0 += step) launch(std::min(step, nb - t0), t0);
    }
  };
  switch (g_tune[0]) {
    case 1: run(I<16>{}, I<2>{}, I<4>{}); break;
    case 2: run(I<16>{}, I<4>{}, I<8>{}); break;
    case 3: run(I<32>{}, I<2>{}, I<4>{}); break;
    case 4: run(I<8>{}, I<4>{}, I<4>{}); break;
    case 5: run(I<16>{}, I<8>{}, I<4>{}); break;
    default: run(I<16>{}, I<4>{}, I<4>{}); break;
  }
  Chk("proj_multi");
}

void GemvQ8(hipStream_t s, int k, const void* w, const float* x, float* y, int m, float softcap, int nb) {
  const uint8_t* W = static_cast<const uint8_t*>(w);
  if (k == 2304) {
    constexpr int LPR = 32, NIT = 1, BW = 4, RPB = BW * (WS / LPR) * NIT;
    const dim3 grid((m + RPB - 1) / RPB);
    switch (nb) {
      case 1: k_gemv<FMT_Q8, 2304, LPR, NIT, BW, 1><<<grid, BW * WS, 0, s>>>(W, x, y, m, softcap, g_res); break;
      case 2: k_gemv<FMT_Q8, 2304, LPR, NIT, BW, 2><<<grid, BW * WS, 0, s>>>(W, x, y, m, softcap, g_res); break;
      case 3: k_gemv<FMT_Q8, 2304, LPR, NIT, BW, 3><<<grid, BW * WS, 0, s>>>(W, x, y, m, softcap, g_res); break;
      case 4: k_gemv<FMT_Q8, 2304, LPR, NIT, BW, 4><<<grid, BW * WS, 0, s>>>(W, x, y, m, softcap, g_res); break;
      case 5: k_gemv<FMT_Q8, 2304, LPR, NIT, BW, 5><<<grid, BW * WS, 0, s>>>(W, x, y, m, softcap, g_res); break;
      case 6: k_gemv<FMT_Q8, 2304, LPR, NIT, BW, 6><<<grid, BW * WS, 0, s>>>(W, x, y, m, softcap, g_res); break;
      case 7: k_gemv<FMT_Q8, 2304, LPR, NIT, BW, 7><<<grid, BW * WS, 0, s>>>(W, x, y, m, softcap, g_res); break;
      default: k_gemv<FMT_Q8, 2304, LPR, NIT, BW, 8><<<grid, BW * WS, 0, s>>>(W, x, y, m, softcap, g_res); break;
    }
  } else if (k == 3072) {
    constexpr int LPR = 16, NIT = 1, BW = 4, RPB = BW * (WS / LPR) * NIT;
    const dim3 grid((m + RPB - 1) / RPB);
    switch (nb) {
      case 1: k_gemv<FMT_Q8, 3072, LPR, NIT, BW, 1><<<grid, BW * WS, 0, s>>>(W, x, y, m, softcap, g_res); break;
      case 2: k_gemv<FMT_Q8, 3072, LPR, NIT, BW, 2><<<grid, BW * WS, 0, s>>>(W, x, y, m, softcap, g_res); break;
      case 3: k_gemv<FMT_Q8, 3072, LPR, NIT, BW, 3><<<grid, BW * WS, 0, s>>>(W, x, y, m, softcap, g_res); break;
      case 4: k_gemv<FMT_Q8, 3072, LPR, NIT, BW, 4><<<grid, BW * WS, 0, s>>>(W, x, y, m, softcap, g_res); break;
      case 5: k_gemv<FMT_Q8, 3072, LPR, NIT, BW, 5><<<grid, BW * WS, 0, s>>>(W, x, y, m, softcap, g_res); break;
      case 6: k_gemv<FMT_Q8, 3072, LPR, NIT, BW, 6><<<grid, BW * WS, 0, s>>>(W, x, y, m, softcap, g_res); break;
      case 7: k_gemv<FMT_Q8, 3072, LPR, NIT, BW, 7><<<grid, BW * WS, 0, s>>>(W, x, y, m, softcap, g_res); break;
      default: k_gemv<FMT_Q8, 3072, LPR, NIT, BW, 8><<<grid, BW * WS, 0, s>>>(W, x, y, m, softcap, g_res); break;
    }
  } else {
    throw std::runtime_error("inkling2 fast: unsupported gemv K");
  }
  Chk("gemv");
}

void Router(hipStream_t s, const void* w, const float* bias, const float* x, float* logit_ws,
            unsigned* counter, int* ids, float* wts, float rscale, int nb, bool half) {
  if (half)
    k_router<2304, true><<<dim3(16, nb), 256, 0, s>>>(w, bias, x, logit_ws, counter, ids, wts, rscale);
  else
    k_router<2304, false><<<dim3(16, nb), 256, 0, s>>>(w, bias, x, logit_ws, counter, ids, wts, rscale);
  Chk("router");
}

void MoeGu(hipStream_t s, int fmt, const void* w, const float* escale, const float* x,
           const int* ids, float* a, int ff, float beta, float lb, int nb) {
  const uint8_t* W = static_cast<const uint8_t*>(w);
  auto run = [&](auto fm, auto lpr, auto nit, auto bw) {
    constexpr int FM = decltype(fm)::value, LPR = decltype(lpr)::value, NIT = decltype(nit)::value,
                  BW = decltype(bw)::value;
    constexpr int RPB = BW * (WS / LPR / 2) * NIT;
    k_moe_gu<FM, 2304, LPR, NIT, BW>
        <<<dim3(8 * nb, (ff + RPB - 1) / RPB), BW * WS, 0, s>>>(W, escale, x, ids, a, ff, beta, lb, g_res);
  };
  if (fmt == kFmtNvfp4) {
    switch (g_tune[1]) {
      case 1: run(I<FMT_NVFP4>{}, I<16>{}, I<2>{}, I<4>{}); break;
      case 2: run(I<FMT_NVFP4>{}, I<16>{}, I<1>{}, I<8>{}); break;
      case 3: run(I<FMT_NVFP4>{}, I<8>{}, I<1>{}, I<4>{}); break;
      case 4: run(I<FMT_NVFP4>{}, I<8>{}, I<2>{}, I<4>{}); break;
      case 5: run(I<FMT_NVFP4>{}, I<16>{}, I<1>{}, I<2>{}); break;
      default: run(I<FMT_NVFP4>{}, I<16>{}, I<1>{}, I<4>{}); break;
    }
  } else {
    run(I<FMT_Q8>{}, I<16>{}, I<1>{}, I<4>{});
  }
  Chk("moe_gu");
}

void MoeDown(hipStream_t s, int fmt, const void* w, const float* escale, const float* a,
             const int* ids, const float* wts, float* y, int d, int nb) {
  const uint8_t* W = static_cast<const uint8_t*>(w);
  auto run = [&](auto fm, auto lpr, auto nit, auto bw) {
    constexpr int FM = decltype(fm)::value, LPR = decltype(lpr)::value, NIT = decltype(nit)::value,
                  BW = decltype(bw)::value;
    constexpr int RPB = BW * (WS / LPR) * NIT;
    k_moe_down<FM, 1152, LPR, NIT, BW><<<dim3((d + RPB - 1) / RPB, nb), BW * WS, 0, s>>>(W, escale, a, ids, wts, y, d, g_res);
  };
  if (fmt == kFmtNvfp4) {
    switch (g_tune[2]) {
      case 1: run(I<FMT_NVFP4>{}, I<32>{}, I<1>{}, I<4>{}); break;
      case 2: run(I<FMT_NVFP4>{}, I<16>{}, I<1>{}, I<8>{}); break;
      case 3: run(I<FMT_NVFP4>{}, I<16>{}, I<2>{}, I<4>{}); break;
      case 4: run(I<FMT_NVFP4>{}, I<32>{}, I<2>{}, I<4>{}); break;
      case 5: run(I<FMT_NVFP4>{}, I<32>{}, I<1>{}, I<2>{}); break;
      default: run(I<FMT_NVFP4>{}, I<32>{}, I<1>{}, I<8>{}); break;
    }
  } else {
    run(I<FMT_Q8>{}, I<8>{}, I<4>{}, I<4>{});
  }
  Chk("moe_down");
}

void MoeDownE(hipStream_t s, int fmt, const void* w, const float* escale, const float* a, const int* ids,
              const float* wts, float* yp, int d, int nb) {
  const uint8_t* W = static_cast<const uint8_t*>(w);
  auto run = [&](auto fm, auto lpr, auto nit, auto bw) {
    constexpr int FM = decltype(fm)::value, LPR = decltype(lpr)::value, NIT = decltype(nit)::value,
                  BW = decltype(bw)::value;
    constexpr int RPB = BW * (WS / LPR) * NIT;
    k_moe_down_e<FM, 1152, LPR, NIT, BW>
        <<<dim3(8 * nb, (d + RPB - 1) / RPB), BW * WS, 0, s>>>(W, escale, a, ids, wts, yp, d, g_res);
  };
  if (fmt == kFmtNvfp4) {
    switch (g_tune[2]) {
      case 11: run(I<FMT_NVFP4>{}, I<16>{}, I<1>{}, I<4>{}); break;
      case 12: run(I<FMT_NVFP4>{}, I<8>{}, I<1>{}, I<4>{}); break;
      case 13: run(I<FMT_NVFP4>{}, I<32>{}, I<1>{}, I<4>{}); break;
      case 14: run(I<FMT_NVFP4>{}, I<16>{}, I<2>{}, I<4>{}); break;
      case 15: run(I<FMT_NVFP4>{}, I<8>{}, I<2>{}, I<4>{}); break;
      case 16: run(I<FMT_NVFP4>{}, I<16>{}, I<1>{}, I<8>{}); break;
      case 17: run(I<FMT_NVFP4>{}, I<8>{}, I<1>{}, I<8>{}); break;
      case 18: run(I<FMT_NVFP4>{}, I<8>{}, I<1>{}, I<2>{}); break;
      case 19: run(I<FMT_NVFP4>{}, I<4>{}, I<1>{}, I<4>{}); break;
      case 20: run(I<FMT_NVFP4>{}, I<8>{}, I<2>{}, I<8>{}); break;
      case 21: run(I<FMT_NVFP4>{}, I<8>{}, I<1>{}, I<1>{}); break;
      default: run(I<FMT_NVFP4>{}, I<8>{}, I<1>{}, I<4>{}); break;
    }
  } else {
    switch (g_tune[2]) {
      case 31: run(I<FMT_Q8>{}, I<16>{}, I<1>{}, I<4>{}); break;
      case 32: run(I<FMT_Q8>{}, I<8>{}, I<1>{}, I<4>{}); break;
      case 33: run(I<FMT_Q8>{}, I<16>{}, I<2>{}, I<4>{}); break;
      case 34: run(I<FMT_Q8>{}, I<32>{}, I<1>{}, I<4>{}); break;
      default: run(I<FMT_Q8>{}, I<8>{}, I<4>{}, I<4>{}); break;
    }
  }
  Chk("moe_down_e");
}

void Mix(hipStream_t s, const MixJob& j, float eps) {
  if (j.has_cr && j.cr.rows) k_mix<true><<<dim3(j.nq, j.ntok), 512, 0, s>>>(j, eps);
  else k_mix<false><<<dim3(j.nq, j.ntok), 512, 0, s>>>(j, eps);
  Chk("mix");
}
void Rms(hipStream_t s, const NormJobs& j, float eps, int nb) {
  k_rms<<<dim3(j.n, nb), 256, 0, s>>>(j, eps);
  Chk("rms");
}
void CtlGate(hipStream_t s, const CtlJobs& j, float eps, int nb) {
  if (j.half) k_ctl_gate<true><<<dim3(64, j.n, nb), 256, 0, s>>>(j, eps);
  else k_ctl_gate<false><<<dim3(64, j.n, nb), 256, 0, s>>>(j, eps);
  Chk("ctl_gate");
}
void CtlApply(hipStream_t s, const CtlJobs& j, int nb) {
  if (j.half) k_ctl_apply<true><<<dim3(9, j.n, nb), 256, 0, s>>>(j);
  else k_ctl_apply<false><<<dim3(9, j.n, nb), 256, 0, s>>>(j);
  Chk("ctl_apply");
}
void Tiny(hipStream_t s, const TinyJobs& j, int nb) {
  int rows = 0;
  for (int i = 0; i < j.n; ++i) rows += j.rows[i];
  k_tiny<<<dim3((rows + 7) / 8, nb), 256, 0, s>>>(j);
  Chk("tiny");
}
void KvPrep(hipStream_t s, const KvPrepArgs& a, int nb) {
  if (a.rows) k_kvprep<true><<<dim3(8, 1, nb), 128, 0, s>>>(a);
  else k_kvprep<false><<<dim3(8, 1, nb), 128, 0, s>>>(a);
  Chk("kvprep");
}
void AttnDecode(hipStream_t s, const AttnArgs& a, int nb) {
  ia::attn_decode<4><<<dim3(8, a.nsplit, nb), 128, 0, s>>>(a);
  ia::decode_reduce<<<dim3(24, 1, nb), 128, 0, s>>>(a);
  Chk("attn_decode");
}
void ConvResid(hipStream_t s, const ConvResidArgs& a, int nb) {
  if (a.rows) k_conv_resid_mr<<<dim3(9, 1, nb), 256, 0, s>>>(a);
  else k_conv_resid<<<dim3(9, 1, nb), 256, 0, s>>>(a);
  Chk("conv_resid");
}
void Prefetch(hipStream_t s, const PrefetchJob& j, float* sink, int blocks) {
  k_prefetch<<<blocks, 256, 0, s>>>(j, sink);
  Chk("prefetch");
}
void RouteBuild(hipStream_t s, const int* ids, int p, int e, int* off, int* sorted) {
  k_route_build<<<1, 1024, 0, s>>>(ids, p, e, off, sorted);
  Chk("route_build");
}


// ---------------------------------------------------------------------------------------------
// prefill host API
void CvtF16(hipStream_t s, const float* x, void* h, size_t n) {
  k_cvt_f16<<<(unsigned)((n / 4 + 255) / 256), 256, 0, s>>>(x, static_cast<_Float16*>(h), n);
  Chk("cvt_f16");
}

void GemmDense(hipStream_t s, const void* w, int m, int k, const void* ah, float* out,
               const GemmItems& it) {
  GP3 p{};
  p.W = static_cast<const uint8_t*>(w);
  p.wexp = 0;
  p.ah = static_cast<const _Float16*>(ah);
  p.nitems = it.nitems;
  p.item_e = it.item_e;
  p.item_start = it.item_start;
  p.item_n = it.item_n;
  p.out = out;
  p.ldo = m;
  constexpr int RG = 2, BM = 32 * RG;
  if (m % BM) throw std::runtime_error("inkling2 fast: dense gemm rows not a multiple of 64");
  const int rgn = m / BM;
  const dim3 grid(it.max_items * rgn);
  if (k == 2304)
    k_gemm4<FMT_Q8, 0, 64, 2304, RG><<<grid, 64 * RG, 0, s>>>(p, rgn);
  else if (k == 3072)
    k_gemm4<FMT_Q8, 0, 64, 3072, RG><<<grid, 64 * RG, 0, s>>>(p, rgn);
  else
    throw std::runtime_error("inkling2 fast: unsupported dense gemm K");
  Chk("gemm_dense");
}

void MoeRoute(hipStream_t s, const int* ids, const float* wts, int p, int tile, const MoeWs& ws) {
  hipMemsetAsync(ws.cnt, 0, 128 * 4, s);
  k_hist<<<(p + 255) / 256, 256, 0, s>>>(ids, p, ws.cnt);
  k_scan_items<<<1, 128, 0, s>>>(ws.cnt, tile, ws.goff, ws.fill, ws.nitems, ws.item_e, ws.item_start,
                                 ws.item_n);
  k_scatter<<<(p + 255) / 256, 256, 0, s>>>(ids, wts, p, ws.goff, ws.fill, ws.postok, ws.pair, ws.posw);
  Chk("moe_route");
}

template <int FMT, int BN, int RG>
static void gu_launch(hipStream_t s, GP3& p, int max_items) {
  constexpr int rgn = 1152 / (16 * RG);
  k_gemm4<FMT, 1, BN, 2304, RG><<<dim3(max_items * rgn), 64 * RG, 0, s>>>(p, rgn);
}
template <int FMT, int BN, int RG>
static void dn_launch(hipStream_t s, GP3& p, int max_items) {
  constexpr int rgn = 2304 / (32 * RG);
  k_gemm4<FMT, 2, BN, 1152, RG><<<dim3(max_items * rgn), 64 * RG, 0, s>>>(p, rgn);
}

void MoeGuPrefill(hipStream_t s, int fmt, int tile, const void* w, const float* escale, const void* ah,
                  void* act16, const MoeWs& ws, int max_items, float beta, float lb, int ff) {
  GP3 p{};
  p.W = static_cast<const uint8_t*>(w);
  p.wexp = (size_t)(fmt == kFmtNvfp4 ? 2304 / 64 * 36 : 2304 / 32 * 34) * 2 * ff;
  p.escale = escale;
  p.ah = static_cast<const _Float16*>(ah);
  p.actrow = ws.postok;
  p.nitems = ws.nitems;
  p.item_e = ws.item_e;
  p.item_start = ws.item_start;
  p.item_n = ws.item_n;
  p.outh = static_cast<_Float16*>(act16);
  p.beta = beta;
  p.lb = lb;
  if (fmt == kFmtNvfp4) {
    if (tile == 32) gu_launch<FMT_NVFP4, 32, 2>(s, p, max_items);
    else gu_launch<FMT_NVFP4, 64, 4>(s, p, max_items);
  } else {
    if (tile == 32) gu_launch<FMT_Q8, 32, 2>(s, p, max_items);
    else gu_launch<FMT_Q8, 64, 4>(s, p, max_items);
  }
  Chk("moe_gu_prefill");
}

void MoeDownPrefill(hipStream_t s, int fmt, int tile, const void* w, const float* escale,
                    const void* act16, float* y, float* pairs, int ntok, const MoeWs& ws, int max_items,
                    int d) {
  GP3 p{};
  p.W = static_cast<const uint8_t*>(w);
  p.wexp = (size_t)(fmt == kFmtNvfp4 ? 1152 / 64 * 36 : 1152 / 32 * 34) * d;
  p.escale = escale;
  p.ah = static_cast<const _Float16*>(act16);
  p.actrow = nullptr;
  p.nitems = ws.nitems;
  p.item_e = ws.item_e;
  p.item_start = ws.item_start;
  p.item_n = ws.item_n;
  p.out = pairs;
  p.ldo = d;
  p.postok = ws.postok;
  p.pair = ws.pair;
  p.posw = ws.posw;
  if (fmt == kFmtNvfp4) {
    if (tile == 32) dn_launch<FMT_NVFP4, 32, 2>(s, p, max_items);
    else dn_launch<FMT_NVFP4, 64, 4>(s, p, max_items);
  } else {
    if (tile == 32) dn_launch<FMT_Q8, 32, 2>(s, p, max_items);
    else dn_launch<FMT_Q8, 64, 4>(s, p, max_items);
  }
  k_sum_pairs<<<ntok, 256, 0, s>>>(pairs, y, d);
  Chk("moe_down_prefill");
}

void AttnPrefill(hipStream_t s, const PrefillAttn& a) {
  k_kv_write<<<(a.n * 8 + 3) / 4, 128, 0, s>>>(a.k, a.v, a.n, a.pos0, a.cap, static_cast<__half*>(a.kc),
                                              static_cast<__half*>(a.vc), a.qk_thr, a.eps);
  k_qr_prep<<<(a.n * 24 + 3) / 4, 128, 0, s>>>(a.q, a.r, a.qscale, a.rclip, a.n, a.pos0, a.swa, a.qk_thr,
                                                a.r_thr, a.tau_floor, a.tau_alpha,
                                                static_cast<__half*>(a.qn), a.rn);
  PfArgs pa{};
  pa.kc = static_cast<const __half*>(a.kc);
  pa.vc = static_cast<const __half*>(a.vc);
  pa.cap = a.cap;
  pa.swa = a.swa;
  pa.U = a.n;
  pa.p0 = a.pos0;
  pa.qn = static_cast<const __half*>(a.qn);
  pa.rn = a.rn;
  pa.sink = a.sink;
  pa.gate = a.gate;
  pa.bt = static_cast<const __half*>(a.bt);
  pa.out = a.out;
  attn_prefill<4><<<dim3((a.n + 63) / 64, 8), 384, 0, s>>>(pa);
  Chk("attn_prefill");
}


void CtlPrefill(hipStream_t s, const CtlPJobs& jobs, int t, float eps) {
  GateJobs g{};
  for (int i = 0; i < jobs.n; ++i) {
    const CtlPJob& c = jobs.j[i];
    g.j[i] = {c.x, c.ldx, c.msg, c.ldm, c.down, 4608, 2304, 2304, 1, c.g64, 64, 0};
  }
  k_gate_p<<<dim3((t + 7) / 8, 1, jobs.n), 256, 0, s>>>(g, t, eps);
  k_ctl_apply_p<<<dim3((t + kCtlTT - 1) / kCtlTT, 36, jobs.n), 512, 0, s>>>(jobs, t);
  Chk("ctl_prefill");
}

void RouterPrefill(hipStream_t s, const float* w, const float* x, float* logits, int t) {
  GateJobs g{};
  g.j[0] = {x, 2304, nullptr, 0, w, 2304, 2304, 0, 0, logits, 128, 0};
  k_gate_p<<<dim3((t + 7) / 8, 2, 1), 256, 0, s>>>(g, t, 0.f);
  Chk("router_prefill");
}


void TopkPrefill(hipStream_t s, const float* logits, const float* bias, int* ids, float* wts, int t,
                 float rscale) {
  k_topk_p<<<(t + 7) / 8, 256, 0, s>>>(logits, bias, ids, wts, t, rscale);
  Chk("topk_prefill");
}

void SkinnyPrefill(hipStream_t s, const float* x, int ldx, const float* w, int rows, float* out,
                   int ldo, int t) {
  GateJobs g{};
  g.j[0] = {x, ldx, nullptr, 0, w, 2304, 2304, 0, 0, out, ldo, rows};
  k_gate_p<<<dim3((t + 7) / 8, 1, 1), 256, 0, s>>>(g, t, 0.f);
  Chk("skinny_prefill");
}

}  // namespace gufo::models::inkling2::fast
