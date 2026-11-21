// Task H2 kernels: fused multi-matrix Q8_0 GEMV, router+top8, WMMA prefill GEMMs (dense + grouped MoE). MIT.
#pragma once
#include "kern.h"
typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef _Float16 v16h __attribute__((ext_vector_type(16)));

// ---------------- (1) multi-matrix Q8_0 GEMV: several matrices (K=2304) over up to NI input vectors in one launch ----------------
struct MultiP { const uint8_t* W[6]; int M[6]; int in[6]; int yoff[6]; int cum[7]; int n; };
template <int K, int LPR, int B, int NIT, int BW, int NI>
__global__ void __launch_bounds__(BW * WS) k_gemv_multi(MultiP P, const float* __restrict__ x, float* __restrict__ y, int Mtot) {
  constexpr int NU = K / 32, RPW = WS / LPR;
  __shared__ v4u aq[NI * B * 2 * NU];
  __shared__ float asc[NI * B * (K / 32)];
  quant_act<K, NI * B, 32>(x, aq, asc);   // x = [NI][B][K]
  int wave = threadIdx.x / WS, lane = threadIdx.x % WS, g = lane / LPR, lir = lane % LPR;
  size_t rb = (size_t)NU * 34;
  int base = (blockIdx.x * BW + wave) * RPW * NIT;
#pragma unroll
  for (int it = 0; it < NIT; it++) {
    int row = base + it * RPW + g;
    int rr = row < Mtot ? row : Mtot - 1;
    int m = 0;
#pragma unroll
    for (int i = 1; i < 6; i++) if (i < P.n && rr >= P.cum[i]) m = i;
    const uint8_t* Wm = P.W[0]; int inm = P.in[0], Mm = P.M[0], yo = P.yoff[0], c0 = P.cum[0];
#pragma unroll
    for (int i = 1; i < 6; i++) if (i == m) { Wm = P.W[i]; inm = P.in[i]; Mm = P.M[i]; yo = P.yoff[i]; c0 = P.cum[i]; }
    int lr = rr - c0;
    float acc[B];
    row_dot<FMT_Q8, K, LPR, B>(Wm + rb * lr, aq, asc, inm * B, lir, acc);
    if (row < Mtot && lir == 0) {
#pragma unroll
      for (int b = 0; b < B; b++) y[(size_t)yo * B + (size_t)b * Mm + lr] = acc[b];
    }
  }
}

// ---------------- router: F32 [128][K] GEMV + sigmoid + biased top-8 + renorm*8 (one launch, last-block-done top-k) ----------------
template <int K>
__global__ void __launch_bounds__(256) k_router(const float* __restrict__ W, const float* __restrict__ bias, const float* __restrict__ x,
                                                float* __restrict__ logit, unsigned* __restrict__ counter, int* __restrict__ ids, float* __restrict__ wts) {
  int tok = blockIdx.y, wave = threadIdx.x / 32, lane = threadIdx.x % 32;
  int row = blockIdx.x * 8 + wave;
  const float4* wp = (const float4*)(W + (size_t)row * K); const float4* xp = (const float4*)(x + (size_t)tok * K);
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < K / 128; i++) { float4 a = wp[lane + 32 * i], b = xp[lane + 32 * i]; s += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) s += __shfl_xor(s, o, 32);
  if (lane == 0) logit[tok * 128 + row] = s;
  __threadfence();
  __syncthreads();
  __shared__ unsigned last;
  if (threadIdx.x == 0) { unsigned t = atomicAdd(&counter[tok], 1u); last = (t == gridDim.x - 1); }
  __syncthreads();
  if (!last) return;
  __threadfence();
  if (wave != 0) return;
  float sc[4], sel[4];
#pragma unroll
  for (int k = 0; k < 4; k++) { int i = lane + 32 * k; float l = __builtin_nontemporal_load(&logit[tok * 128 + i]); sc[k] = 1.f / (1.f + expf(-l)); sel[k] = sc[k] + bias[i]; }
  float ssum = 0.f; float myw[8]; int myi[8];
#pragma unroll
  for (int r = 0; r < 8; r++) {
    float bv = -1e30f; int bi = 0;
#pragma unroll
    for (int k = 0; k < 4; k++) if (sel[k] > bv) { bv = sel[k]; bi = lane + 32 * k; }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { float ov = __shfl_xor(bv, o, 32); int oi = __shfl_xor(bi, o, 32); if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; } }
    float scv = 0.f;
#pragma unroll
    for (int k = 0; k < 4; k++) if (lane + 32 * k == bi) { scv = sc[k]; sel[k] = -1e30f; }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) scv += __shfl_xor(scv, o, 32);
    myw[r] = scv; myi[r] = bi; ssum += scv;
  }
  if (lane < 8) { float w = 0.f; int id = 0;
#pragma unroll
    for (int r = 0; r < 8; r++) if (r == lane) { w = myw[r]; id = myi[r]; }
    ids[tok * 8 + lane] = id; wts[tok * 8 + lane] = w / (ssum + 1e-20f) * 8.f; }
  if (lane == 0) counter[tok] = 0;
}

// ---------------- (2) prefill GEMM with WMMA (gfx11 wave32). D[token][weight row] = A(activations) x B(weights^T) ----------------
// MODE 0: W8A8 iu8 WMMA (act int8 per-32 scale), fp32 dequant epilogue per K-block; MODE 1: dequant weights to f16, f16 WMMA (act f16)
// EPI 0 store out[pos*ldo+row]; EPI 1 SITU (gate rows j + up rows FF+j, expert scale), a[pos][FF]; EPI 2 down: atomicAdd(y[postok[pos]][row], w*escale*acc)
struct GP {
  const uint8_t* W; size_t wexp; const float* escale; const int8_t* aq; const _Float16* ah; const float* asc; int Rpad;
  const int* actrow; const int* item_e; const int* item_start; const int* item_n; float* out; int ldo; const int* postok; const float* posw;
};
__device__ __forceinline__ v16h deq16(v4u w, float s) {   // 16 int8 -> f16 * s
  v16h r; uint32_t a[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
  for (int i = 0; i < 16; i++) r[i] = (_Float16)((float)(int)(int8_t)(a[i >> 2] >> (8 * (i & 3))) * s);
  return r;
}
template <int FMT, int MODE, int EPI, int TT, int K>
__global__ void __launch_bounds__(128) k_gemm(GP p) {
  constexpr int FF = 1152;
  int item = blockIdx.x, e = p.item_e[item], start = p.item_start[item], n = p.item_n[item];
  int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, ln = lane & 15, hi = lane >> 4;
  int rt = blockIdx.y * 4 + wave;
  constexpr size_t rb = FMT == FMT_Q8 ? (K / 32) * 34 : (K / 64) * 36;
  const uint8_t* We = p.W + (size_t)e * p.wexp;
  int rowbase[2];
  if (EPI == 1) { rowbase[0] = rt * 16; rowbase[1] = FF + rt * 16; } else { rowbase[0] = rt * 32; rowbase[1] = rt * 32 + 16; }
  const uint8_t* wr[2] = {We + (size_t)(rowbase[0] + ln) * rb, We + (size_t)(rowbase[1] + ln) * rb};
  int arow_a[TT], arow_d[TT][8], posd[TT][8];
#pragma unroll
  for (int t = 0; t < TT; t++) {
    int pa = (16 * t + ln < n) ? start + 16 * t + ln : start; arow_a[t] = p.actrow ? p.actrow[pa] : pa;
#pragma unroll
    for (int i = 0; i < 8; i++) { int pl = 16 * t + 2 * i + hi; int pd = pl < n ? start + pl : -1; posd[t][i] = pd; int pc = pd >= 0 ? pd : start; arow_d[t][i] = p.actrow ? p.actrow[pc] : pc; }
  }
  v8f facc[TT][2];
#pragma unroll
  for (int t = 0; t < TT; t++) { facc[t][0] = (v8f){0, 0, 0, 0, 0, 0, 0, 0}; facc[t][1] = facc[t][0]; }
  const v8i z8 = {0, 0, 0, 0, 0, 0, 0, 0};
  if (FMT == FMT_Q8) {
#pragma unroll(MODE == 1 ? 2 : 1)
    for (int kb = 0; kb < K / 32; kb++) {
      float d[2]; v4u w0[2], w1[2];
#pragma unroll
      for (int r = 0; r < 2; r++) { const uint8_t* b = wr[r] + kb * 34; d[r] = __half2float(*(const __half*)b); w0[r] = *(const v4u_a2*)(b + 2); w1[r] = *(const v4u_a2*)(b + 18); }
      if (MODE == 0) {
        float sa[TT][8];
#pragma unroll
        for (int t = 0; t < TT; t++)
#pragma unroll
          for (int i = 0; i < 8; i++) sa[t][i] = p.asc[(size_t)kb * p.Rpad + arow_d[t][i]];
#pragma unroll
        for (int t = 0; t < TT; t++) {
          v4i a0 = *(const v4i*)(p.aq + (size_t)arow_a[t] * K + kb * 32), a1 = *(const v4i*)(p.aq + (size_t)arow_a[t] * K + kb * 32 + 16);
#pragma unroll
          for (int r = 0; r < 2; r++) {
            v8i c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a0, true, (v4i){(int)w0[r].x, (int)w0[r].y, (int)w0[r].z, (int)w0[r].w}, z8, false);
            c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a1, true, (v4i){(int)w1[r].x, (int)w1[r].y, (int)w1[r].z, (int)w1[r].w}, c, false);
#pragma unroll
            for (int i = 0; i < 8; i++) facc[t][r][i] += (float)c[i] * (d[r] * sa[t][i]);
          }
        }
      } else {
        v16h wh0[2], wh1[2];
#pragma unroll
        for (int r = 0; r < 2; r++) { wh0[r] = deq16(w0[r], d[r]); wh1[r] = deq16(w1[r], d[r]); }
#pragma unroll
        for (int t = 0; t < TT; t++) {
          v16h a0 = *(const v16h*)(p.ah + (size_t)arow_a[t] * K + kb * 32), a1 = *(const v16h*)(p.ah + (size_t)arow_a[t] * K + kb * 32 + 16);
#pragma unroll
          for (int r = 0; r < 2; r++) { facc[t][r] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, wh0[r], facc[t][r]); facc[t][r] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, wh1[r], facc[t][r]); }
        }
      }
    }
  } else {
#pragma unroll(MODE == 1 ? 2 : 1)
    for (int kb = 0; kb < K / 64; kb++) {
      uint32_t sw[2]; uint32_t qw[2][8];
#pragma unroll
      for (int r = 0; r < 2; r++) { const uint8_t* b = wr[r] + kb * 36; sw[r] = *(const uint32_t*)b; v4u q0 = *(const v4u_a4*)(b + 4), q1 = *(const v4u_a4*)(b + 20);
        qw[r][0] = q0.x; qw[r][1] = q0.y; qw[r][2] = q0.z; qw[r][3] = q0.w; qw[r][4] = q1.x; qw[r][5] = q1.y; qw[r][6] = q1.z; qw[r][7] = q1.w; }
#pragma unroll
      for (int s = 0; s < 4; s++) {
        v4u wf[2]; float ds[2];
#pragma unroll
        for (int r = 0; r < 2; r++) { uint32_t w0 = qw[r][2 * s], w1 = qw[r][2 * s + 1];
          wf[r] = (v4u){dec4(w0 & 0x0F0F0F0Fu), dec4(w1 & 0x0F0F0F0Fu), dec4((w0 >> 4) & 0x0F0F0F0Fu), dec4((w1 >> 4) & 0x0F0F0F0Fu)}; ds[r] = ue4m3((sw[r] >> (8 * s)) & 0xFFu); }
        if (MODE == 0) {
          int kb32 = kb * 2 + (s >> 1);
#pragma unroll
          for (int t = 0; t < TT; t++) {
            v4i a = *(const v4i*)(p.aq + (size_t)arow_a[t] * K + kb * 64 + s * 16);
            float sa[8];
#pragma unroll
            for (int i = 0; i < 8; i++) sa[i] = p.asc[(size_t)kb32 * p.Rpad + arow_d[t][i]];
#pragma unroll
            for (int r = 0; r < 2; r++) {
              v8i c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, (v4i){(int)wf[r].x, (int)wf[r].y, (int)wf[r].z, (int)wf[r].w}, z8, false);
#pragma unroll
              for (int i = 0; i < 8; i++) facc[t][r][i] += (float)c[i] * (ds[r] * sa[i]);
            }
          }
        } else {
          v16h wh[2];
#pragma unroll
          for (int r = 0; r < 2; r++) wh[r] = deq16(wf[r], ds[r]);
#pragma unroll
          for (int t = 0; t < TT; t++) {
            v16h a = *(const v16h*)(p.ah + (size_t)arow_a[t] * K + kb * 64 + s * 16);
#pragma unroll
            for (int r = 0; r < 2; r++) facc[t][r] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, wh[r], facc[t][r]);
          }
        }
      }
    }
  }
  float esc = p.escale ? p.escale[e] : 1.f;
#pragma unroll
  for (int t = 0; t < TT; t++)
#pragma unroll
    for (int i = 0; i < 8; i++) {
      int pd = posd[t][i]; if (pd < 0) continue;
      if (EPI == 0) { p.out[(size_t)pd * p.ldo + rowbase[0] + ln] = facc[t][0][i]; p.out[(size_t)pd * p.ldo + rowbase[1] + ln] = facc[t][1][i]; }
      else if (EPI == 1) { p.out[(size_t)pd * FF + rt * 16 + ln] = situ(facc[t][0][i] * esc, facc[t][1][i] * esc); }
      else { float w = p.posw[pd] * esc; size_t o = (size_t)p.postok[pd] * p.ldo; atomicAdd(&p.out[o + rowbase[0] + ln], w * facc[t][0][i]); atomicAdd(&p.out[o + rowbase[1] + ln], w * facc[t][1][i]); }
    }
}

// activation prep for prefill: x[rows][K] f32 -> int8 [rows][K] + scale[kb*Rpad+row] (per 32), or f16
__global__ void k_quant_rows_i8(const float* __restrict__ x, int8_t* __restrict__ q, float* __restrict__ sc, int rows, int K, int Rpad) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x; int nb = K / 32; if (idx >= rows * nb) return;
  int row = idx % rows, kb = idx / rows;
  const float4* xp = (const float4*)(x + (size_t)row * K + kb * 32);
  float4 f[8]; float amax = 0.f;
#pragma unroll
  for (int i = 0; i < 8; i++) { f[i] = xp[i]; amax = fmaxf(amax, fmaxf(fmaxf(fabsf(f[i].x), fabsf(f[i].y)), fmaxf(fabsf(f[i].z), fabsf(f[i].w)))); }
  float id = amax > 0.f ? 127.f / amax : 0.f; sc[(size_t)kb * Rpad + row] = amax / 127.f;
  uint4* qp = (uint4*)(q + (size_t)row * K + kb * 32);
#pragma unroll
  for (int h = 0; h < 2; h++) qp[h] = make_uint4(pack4(f[4 * h].x * id, f[4 * h].y * id, f[4 * h].z * id, f[4 * h].w * id), pack4(f[4 * h + 1].x * id, f[4 * h + 1].y * id, f[4 * h + 1].z * id, f[4 * h + 1].w * id),
                                                  pack4(f[4 * h + 2].x * id, f[4 * h + 2].y * id, f[4 * h + 2].z * id, f[4 * h + 2].w * id), pack4(f[4 * h + 3].x * id, f[4 * h + 3].y * id, f[4 * h + 3].z * id, f[4 * h + 3].w * id));
}
__global__ void k_cvt_f16(const float* __restrict__ x, _Float16* __restrict__ h, size_t n) {
  size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 4; if (i >= n) return;
  float4 f = *(const float4*)(x + i); h[i] = (_Float16)f.x; h[i + 1] = (_Float16)f.y; h[i + 2] = (_Float16)f.z; h[i + 3] = (_Float16)f.w;
}
