// Task H3: LDS-tiled f16-WMMA grouped GEMM with dequant-to-LDS (scales folded into f16 weights), double-buffered; GPU-side expert grouping. MIT.
#pragma once
#include "kern2.h"
#ifndef WPE
#define WPE 6
#endif
typedef _Float16 v16h_a16 __attribute__((ext_vector_type(16), aligned(16)));
typedef uint32_t v4u_a16 __attribute__((ext_vector_type(4), aligned(16)));

struct GP3 {
  const uint8_t* W; size_t wexp; const float* escale; const _Float16* ah; const int* actrow;
  const int* nitems; const int* item_e; const int* item_start; const int* item_n;
  float* out; _Float16* outh; int ldo; const int* postok; const float* posw;
};
// Tile: 64 weight rows x BN tokens, K-step 64 (= one NVFP4 block / two Q8_0 blocks), 4 waves, wave tile 32 rows x BN/2 tokens.
// EPI 0 store f32 out[pos*ldo+row]; EPI 1 SITU (rows: gate j / up FF+j paired per wave) -> outh[pos*FF+j] f16; EPI 2 down: atomicAdd y[postok[pos]]
template <int FMT, int EPI, int BN, int K>
__global__ void __launch_bounds__(128) __attribute__((amdgpu_waves_per_eu(WPE, 16))) k_gemm3(GP3 p) {
  constexpr int FF = 1152, STR = 72, BM = 64, TTW = BN / 32, NK = K / 64, NCH = BN / 16;
  constexpr size_t RB = FMT == FMT_Q8 ? (K / 32) * 34 : (K / 64) * 36;
  int item = blockIdx.x; if (item >= *p.nitems) return;
  int e = p.item_e[item], start = p.item_start[item], n = p.item_n[item];
  int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, ln = lane & 15, hi = lane >> 4, half = wave & 1, th = wave >> 1;
  __shared__ _Float16 Ws[2][BM * STR];
  __shared__ _Float16 As[2][BN * STR];
  const uint8_t* We = p.W + (size_t)e * p.wexp;
  // ---- weight loader mapping ----
  int wr = tid >> 1, wh = tid & 1, srow;
  if (EPI == 1) { int h2 = wr >> 5, sub = (wr >> 4) & 1, r = wr & 15; srow = (sub ? FF : 0) + blockIdx.y * 32 + h2 * 16 + r; } else srow = blockIdx.y * 64 + wr;
  const uint8_t* wp = We + (size_t)srow * RB;
  uint32_t raw[9];
  // ---- activation loader mapping ----
  v4u areg[NCH]; const _Float16* ap[NCH]; int arow_l[NCH];
#pragma unroll
  for (int i = 0; i < NCH; i++) { int idx = tid + 128 * i; int row = idx >> 3, ch = idx & 7; arow_l[i] = row * STR + ch * 8;
    int pos = row < n ? start + row : start; int ar = p.actrow ? p.actrow[pos] : pos; ap[i] = p.ah + (size_t)ar * K + ch * 8; }
  auto gload = [&](int kb) {
    if (FMT == FMT_NVFP4) { const uint8_t* b = wp + (size_t)kb * 36; raw[0] = *(const uint32_t*)b; v4u q = *(const v4u_a4*)(b + 4 + 16 * wh); raw[1] = q.x; raw[2] = q.y; raw[3] = q.z; raw[4] = q.w; }
    else { const uint8_t* b = wp + (size_t)(2 * kb + wh) * 34; raw[0] = *(const uint16_t*)b; v4u a = *(const v4u_a2*)(b + 2), c = *(const v4u_a2*)(b + 18);
      raw[1] = a.x; raw[2] = a.y; raw[3] = a.z; raw[4] = a.w; raw[5] = c.x; raw[6] = c.y; raw[7] = c.z; raw[8] = c.w; }
#pragma unroll
    for (int i = 0; i < NCH; i++) areg[i] = *(const v4u*)(ap[i] + kb * 64);
  };
  auto sstore = [&](int buf) {
    _Float16* dst = &Ws[buf][wr * STR + 32 * wh];
    if (FMT == FMT_NVFP4) {
#pragma unroll
      for (int s2 = 0; s2 < 2; s2++) { float ds = ue4m3((raw[0] >> (16 * wh + 8 * s2)) & 0xFFu); uint32_t w0 = raw[1 + 2 * s2], w1 = raw[2 + 2 * s2];
        v4u wf = {dec4(w0 & 0x0F0F0F0Fu), dec4(w1 & 0x0F0F0F0Fu), dec4((w0 >> 4) & 0x0F0F0F0Fu), dec4((w1 >> 4) & 0x0F0F0F0Fu)};
        v16h h = deq16(wf, ds); *(v16h_a16*)(dst + 16 * s2) = h; }
    } else {
      float d = __half2float(__ushort_as_half((unsigned short)raw[0]));
      v16h h0 = deq16((v4u){raw[1], raw[2], raw[3], raw[4]}, d), h1 = deq16((v4u){raw[5], raw[6], raw[7], raw[8]}, d);
      *(v16h_a16*)dst = h0; *(v16h_a16*)(dst + 16) = h1;
    }
#pragma unroll
    for (int i = 0; i < NCH; i++) *(v4u_a16*)&As[buf][arow_l[i]] = areg[i];
  };
  v8f facc[TTW][2];
#pragma unroll
  for (int t = 0; t < TTW; t++) { facc[t][0] = (v8f){0, 0, 0, 0, 0, 0, 0, 0}; facc[t][1] = facc[t][0]; }
  gload(0); sstore(0); __syncthreads();
  for (int kb = 0; kb < NK; kb++) {
    int cur = kb & 1;
    if (kb + 1 < NK) gload(kb + 1);
#pragma unroll
    for (int ks = 0; ks < 4; ks++) {
      v16h bf[2];
#pragma unroll
      for (int r = 0; r < 2; r++) bf[r] = *(const v16h_a16*)&Ws[cur][(half * 32 + 16 * r + ln) * STR + ks * 16];
#pragma unroll
      for (int t = 0; t < TTW; t++) {
        v16h a = *(const v16h_a16*)&As[cur][(th * (BN / 2) + 16 * t + ln) * STR + ks * 16];
#pragma unroll
        for (int r = 0; r < 2; r++) facc[t][r] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, bf[r], facc[t][r]);
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
      int pl = th * (BN / 2) + 16 * t + 2 * i + hi; if (pl >= n) continue; int pd = start + pl;
      if (EPI == 0) { int row = blockIdx.y * 64 + half * 32 + ln; p.out[(size_t)pd * p.ldo + row] = facc[t][0][i]; p.out[(size_t)pd * p.ldo + row + 16] = facc[t][1][i]; }
      else if (EPI == 1) { int j = blockIdx.y * 32 + half * 16 + ln; p.outh[(size_t)pd * FF + j] = (_Float16)situ(facc[t][0][i] * esc, facc[t][1][i] * esc); }
      else if (EPI == 3) { float w = p.posw[pd] * esc; int row = blockIdx.y * 64 + half * 32 + ln; p.out[(size_t)pd * p.ldo + row] = w * facc[t][0][i]; p.out[(size_t)pd * p.ldo + row + 16] = w * facc[t][1][i]; }
      else { float w = p.posw[pd] * esc; size_t o = (size_t)p.postok[pd] * p.ldo; int row = blockIdx.y * 64 + half * 32 + ln;
        atomicAdd(&p.out[o + row], w * facc[t][0][i]); atomicAdd(&p.out[o + row + 16], w * facc[t][1][i]); }
    }
}

// ---------------- GPU-side expert grouping ----------------
__global__ void k_hist(const int* __restrict__ ids, int P, int* cnt) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < P) atomicAdd(&cnt[ids[i]], 1); }
__global__ void k_scan_items(const int* cnt, int tile, int* goff, int* fill, int* nitems, int* ie, int* is, int* in) {   // 1 block x 128 threads
  __shared__ int sc[128], st[128]; int e = threadIdx.x; int c = cnt[e], tiles = (c + tile - 1) / tile; sc[e] = c; st[e] = tiles; __syncthreads();
  for (int o = 1; o < 128; o <<= 1) { int a = e >= o ? sc[e - o] : 0, b = e >= o ? st[e - o] : 0; __syncthreads(); sc[e] += a; st[e] += b; __syncthreads(); }
  int ec = sc[e] - c, et = st[e] - tiles; goff[e] = ec; fill[e] = 0;
  for (int t = 0; t < tiles; t++) { int idx = et + t; ie[idx] = e; is[idx] = ec + t * tile; in[idx] = min(tile, c - t * tile); }
  if (e == 127) { *nitems = st[127]; goff[128] = sc[127]; }
}
__global__ void k_scatter(const int* __restrict__ ids, const float* __restrict__ w, int P, const int* goff, int* fill, int* postok, float* posw, int* invpos) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= P) return; int e = ids[i]; int pos = goff[e] + atomicAdd(&fill[e], 1); postok[pos] = i / 8; posw[pos] = w[i]; invpos[i] = pos;
}

// per-slot outputs y2[pos][D] -> y[token][D] = sum of the token's 8 slots (deterministic, no atomics)
__global__ void k_reduce8(const float* __restrict__ y2, const int* __restrict__ invpos, float* __restrict__ y, int D) {
  int t = blockIdx.x; for (int r = threadIdx.x * 4; r < D; r += blockDim.x * 4) { float4 s = make_float4(0, 0, 0, 0);
#pragma unroll
    for (int k = 0; k < 8; k++) { float4 v = *(const float4*)(y2 + (size_t)invpos[t * 8 + k] * D + r); s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w; }
    *(float4*)(y + (size_t)t * D + r) = s; }
}
