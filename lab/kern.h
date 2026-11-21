// Standalone HIP decode kernels for Inkling 2 Mini (gfx1151). MIT.
// Block layouts follow ggml-common.h (block_q8_0: f16 d + 32 x i8 = 34 B; block_nvfp4: 4 x ue4m3 + 32 B nibbles = 36 B, 64 elems).
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#ifndef WS
#define WS 32   // wavefront size this TU is compiled for (-mwavefrontsize64 -DWS=64 for wave64)
#endif
typedef uint32_t v4u    __attribute__((ext_vector_type(4)));
typedef uint32_t v4u_a2 __attribute__((ext_vector_type(4), aligned(2)));
typedef uint32_t v4u_a4 __attribute__((ext_vector_type(4), aligned(4)));

enum { FMT_Q8 = 0, FMT_NVFP4 = 1 };
template <int F> struct Fm;
template <> struct Fm<FMT_Q8>    { static constexpr int BYTES = 34, UEL = 32; };
template <> struct Fm<FMT_NVFP4> { static constexpr int BYTES = 36, UEL = 64; };

__device__ __forceinline__ int dot4(uint32_t a, uint32_t b, int c) {
  return __builtin_amdgcn_sudot4(true, (int)a, true, (int)b, c, false);   // v_dot4_i32_iu8, both signed
}
// 4 e2m1 nibbles (one per byte, 0..15) -> 4 signed int8 (value*2: 0,1,2,3,4,6,8,12 / negatives)
__device__ __forceinline__ uint32_t dec4(uint32_t n) {
  uint32_t sel = n & 0x07070707u;
  uint32_t pos = __builtin_amdgcn_perm(0x0C080604u, 0x03020100u, sel);
  uint32_t neg = __builtin_amdgcn_perm(0xF4F8FAFCu, 0xFDFEFF00u, sel);
  uint32_t m = ((n >> 3) & 0x01010101u) * 255u;
  return (neg & m) | (pos & ~m);
}
// ggml_ue4m3_to_fp32 (includes the 0.5 factor)
__device__ __forceinline__ float ue4m3(uint32_t x) {
  uint32_t e = (x >> 3) & 15u, m = x & 7u;
  float nrm = __uint_as_float(((e + 119u) << 23) | (m << 20));
  float sub = (float)m * (1.0f / 1024.0f);
  float r = e ? nrm : sub;
  return (x == 0x7Fu) ? 0.f : r;
}
__device__ __forceinline__ uint32_t pack4(float a, float b, float c, float d) {
  return ((uint32_t)(uint8_t)(int8_t)(int)rintf(a)) | ((uint32_t)(uint8_t)(int8_t)(int)rintf(b) << 8) |
         ((uint32_t)(uint8_t)(int8_t)(int)rintf(c) << 16) | ((uint32_t)(uint8_t)(int8_t)(int)rintf(d) << 24);
}

// Quantize NV activation vectors (contiguous, K floats each) to int8 with a per-32 f32 scale into LDS.
// LDS layout: aq[(v*CPU + cc)*NU + u] = 16 B chunk cc of unit u (consecutive units -> consecutive 16 B: conflict-free).
template <int K, int NV, int UEL>
__device__ __forceinline__ void quant_act(const float* __restrict__ x, v4u* aq, float* asc) {
  constexpr int NB32 = K / 32, NU = K / UEL, CPU = UEL / 16;
  for (int it = threadIdx.x; it < NV * NB32; it += blockDim.x) {
    int v = it / NB32, j = it % NB32;
    const float4* xp = (const float4*)(x + (size_t)v * K + j * 32);
    float4 f[8]; float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      f[i] = xp[i];
      amax = fmaxf(amax, fmaxf(fmaxf(fabsf(f[i].x), fabsf(f[i].y)), fmaxf(fabsf(f[i].z), fabsf(f[i].w))));
    }
    float id = amax > 0.f ? 127.f / amax : 0.f;
    asc[v * NB32 + j] = amax / 127.f;
    uint32_t q[8];
#pragma unroll
    for (int i = 0; i < 8; i++) q[i] = pack4(f[i].x * id, f[i].y * id, f[i].z * id, f[i].w * id);
#pragma unroll
    for (int h = 0; h < 2; h++) {
      int c16 = j * 2 + h, u = c16 / CPU, cc = c16 % CPU;
      v4u t = {q[4 * h], q[4 * h + 1], q[4 * h + 2], q[4 * h + 3]};
      aq[(v * CPU + cc) * NU + u] = t;
    }
  }
  __syncthreads();
}

template <int B, int K>
__device__ __forceinline__ void q8_unit(const uint8_t* p, const v4u* aq, const float* asc, int u, int vb, float (&acc)[B]) {
  constexpr int NU = K / 32;
#ifdef Q8_SAFE
  const uint16_t* h = (const uint16_t*)(p + 2); uint32_t t[8];
  for (int i = 0; i < 8; i++) t[i] = (uint32_t)h[2 * i] | ((uint32_t)h[2 * i + 1] << 16);
  v4u w0 = {t[0], t[1], t[2], t[3]}, w1 = {t[4], t[5], t[6], t[7]};
#else
  v4u w0 = *(const v4u_a2*)(p + 2);
  v4u w1 = *(const v4u_a2*)(p + 18);
#endif
  float d = __half2float(*(const __half*)p);
#pragma unroll
  for (int b = 0; b < B; b++) {
    v4u a0 = aq[((vb + b) * 2 + 0) * NU + u], a1 = aq[((vb + b) * 2 + 1) * NU + u];
    int s = dot4(w0.x, a0.x, 0); s = dot4(w0.y, a0.y, s); s = dot4(w0.z, a0.z, s); s = dot4(w0.w, a0.w, s);
    s = dot4(w1.x, a1.x, s); s = dot4(w1.y, a1.y, s); s = dot4(w1.z, a1.z, s); s = dot4(w1.w, a1.w, s);
    acc[b] += d * asc[(vb + b) * NU + u] * (float)s;
  }
}

template <int B, int K>
__device__ __forceinline__ void nv_unit(const uint8_t* p, const v4u* aq, const float* asc, int u, int vb, float (&acc)[B]) {
  constexpr int NU = K / 64, NB32 = K / 32;
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
      v4u a = aq[((vb + b) * 4 + s) * NU + u];
      int t = dot4(l0, a.x, 0); t = dot4(l1, a.y, t); t = dot4(h0, a.z, t); t = dot4(h1, a.w, t);
      acc[b] += ds * asc[(vb + b) * NB32 + u * 2 + (s >> 1)] * (float)t;
    }
  }
}

// One row (K elements) dotted with B vectors starting at vector index vb; LPR lanes cooperate, result reduced over the LPR lanes.
template <int FMT, int K, int LPR, int B>
__device__ __forceinline__ void row_dot(const uint8_t* row, const v4u* aq, const float* asc, int vb, int lir, float (&acc)[B]) {
  constexpr int NU = K / Fm<FMT>::UEL, BY = Fm<FMT>::BYTES;
#pragma unroll
  for (int b = 0; b < B; b++) acc[b] = 0.f;
#pragma unroll
  for (int u0 = 0; u0 < NU; u0 += LPR) {
    int u = u0 + lir;
    if ((NU % LPR == 0) || u < NU) {
      if (FMT == FMT_Q8) q8_unit<B, K>(row + (size_t)u * BY, aq, asc, u, vb, acc);
      else               nv_unit<B, K>(row + (size_t)u * BY, aq, asc, u, vb, acc);
    }
  }
#pragma unroll
  for (int b = 0; b < B; b++)
#pragma unroll
    for (int off = LPR / 2; off > 0; off >>= 1) acc[b] += __shfl_xor(acc[b], off, WS);
}

// ---------------- dense GEMV: y[b][m] = W[m,:] . x[b,:]  (optional softcap) ----------------
template <int FMT, int K, int LPR, int B, int NIT, int BW>
__global__ void __launch_bounds__(BW * WS) k_gemv(const uint8_t* __restrict__ W, const float* __restrict__ x, float* __restrict__ y, int M, float softcap) {
  constexpr int NU = K / Fm<FMT>::UEL, CPU = Fm<FMT>::UEL / 16, RPW = WS / LPR;
  __shared__ v4u aq[B * CPU * NU];
  __shared__ float asc[B * (K / 32)];
  quant_act<K, B, Fm<FMT>::UEL>(x, aq, asc);
  int wave = threadIdx.x / WS, lane = threadIdx.x % WS, g = lane / LPR, lir = lane % LPR;
  size_t rb = (size_t)NU * Fm<FMT>::BYTES;
  int base = (blockIdx.x * BW + wave) * RPW * NIT;
#pragma unroll
  for (int it = 0; it < NIT; it++) {
    int row = base + it * RPW + g;
    int rr = row < M ? row : M - 1;
    float acc[B];
    row_dot<FMT, K, LPR, B>(W + rb * rr, aq, asc, 0, lir, acc);
    if (row < M && lir == 0) {
#pragma unroll
      for (int b = 0; b < B; b++) y[(size_t)b * M + row] = softcap > 0.f ? softcap * tanhf(acc[b] / softcap) : acc[b];
    }
  }
}

// ---------------- MoE gate/up + SITU: one launch for all (token, expert) pairs ----------------
// W: layer base [n_exp][2*FF rows][K]; rows 0..FF-1 = gate, FF..2FF-1 = up. a[p][FF], p = tok*8+slot
__device__ __forceinline__ float situ(float g, float u) {
  return 4.f * tanhf(g * 0.25f) * (1.f / (1.f + expf(-g))) * 25.f * tanhf(u * 0.04f);
}
template <int FMT, int K, int LPR, int NIT, int BW>
__global__ void __launch_bounds__(BW * WS) k_moe_gu(const uint8_t* __restrict__ W, const float* __restrict__ escale, const float* __restrict__ x,
                                                     const int* __restrict__ ids, float* __restrict__ a, int FF) {
  constexpr int NU = K / Fm<FMT>::UEL, CPU = Fm<FMT>::UEL / 16, RPP = WS / LPR / 2;
  __shared__ v4u aq[CPU * NU];
  __shared__ float asc[K / 32];
  int p = blockIdx.y, tok = p / 8;
  quant_act<K, 1, Fm<FMT>::UEL>(x + (size_t)tok * K, aq, asc);
  int e = ids[p];
  float sc = escale ? escale[e] : 1.f;
  size_t rb = (size_t)NU * Fm<FMT>::BYTES;
  const uint8_t* We = W + (size_t)e * 2 * FF * rb;
  int wave = threadIdx.x / WS, lane = threadIdx.x % WS, g = lane / LPR, lir = lane % LPR, isup = g & 1, pg = g >> 1;
  int base = (blockIdx.x * BW + wave) * RPP * NIT;
#pragma unroll
  for (int it = 0; it < NIT; it++) {
    int j = base + it * RPP + pg;
    int jj = j < FF ? j : FF - 1;
    float acc[1];
    row_dot<FMT, K, LPR, 1>(We + rb * (isup ? FF + jj : jj), aq, asc, 0, lir, acc);
    float other = __shfl_xor(acc[0], LPR, WS);
    if (!isup && lir == 0 && j < FF) a[(size_t)p * FF + j] = situ(acc[0] * sc, other * sc);
  }
}

// ---------------- MoE down + weighted sum (+ residual): y[tok][r] = sum_e w_e * s_e * (W2[id_e][r,:] . a_e) ----------------
template <int FMT, int K, int LPR, int NIT, int BW>
__global__ void __launch_bounds__(BW * WS) k_moe_down(const uint8_t* __restrict__ W, const float* __restrict__ escale, const float* __restrict__ a,
                                                       const int* __restrict__ ids, const float* __restrict__ wts, const float* __restrict__ resid,
                                                       float* __restrict__ y, int D) {
  constexpr int NU = K / Fm<FMT>::UEL, CPU = Fm<FMT>::UEL / 16, RPW = WS / LPR;
  __shared__ v4u aq[8 * CPU * NU];
  __shared__ float asc[8 * (K / 32)];
  int tok = blockIdx.y;
  quant_act<K, 8, Fm<FMT>::UEL>(a + (size_t)tok * 8 * K, aq, asc);
  size_t rb = (size_t)NU * Fm<FMT>::BYTES;
  int wave = threadIdx.x / WS, lane = threadIdx.x % WS, g = lane / LPR, lir = lane % LPR;
  int base = (blockIdx.x * BW + wave) * RPW * NIT;
  int eid[8]; float ew[8];
#pragma unroll
  for (int s = 0; s < 8; s++) { eid[s] = ids[tok * 8 + s]; ew[s] = wts[tok * 8 + s] * (escale ? escale[eid[s]] : 1.f); }
#pragma unroll
  for (int it = 0; it < NIT; it++) {
    int row = base + it * RPW + g;
    int rr = row < D ? row : D - 1;
    float sum = 0.f;
#pragma unroll
    for (int s = 0; s < 8; s++) {
      float acc[1];
      row_dot<FMT, K, LPR, 1>(W + ((size_t)eid[s] * D + rr) * rb, aq, asc, s, lir, acc);
      sum += ew[s] * acc[0];
    }
    if (row < D && lir == 0) y[(size_t)tok * D + row] = sum + (resid ? resid[(size_t)tok * D + row] : 0.f);
  }
}
