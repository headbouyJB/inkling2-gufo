#pragma once
#include "kern4.h"
// Task H4b: k_gemm5 = k_gemm4 with 2-stage-deep register prefetch (global loads issued 2 K-steps ahead of the LDS write).
template <int FMT, int EPI, int BN, int K, int RG, int PERSIST>
__global__ void __launch_bounds__(64 * RG) k_gemm5(GP3 p, int rgn) {
  constexpr int FF = 1152, STR = 72, NW = 2 * RG, NT = 64 * RG, BM = 32 * RG, TTW = BN / 32, NK = K / 64, NCH = (BN * 8 + NT - 1) / NT;
  constexpr size_t RB = FMT == FMT_Q8 ? (K / 32) * 34 : (K / 64) * 36;
  __shared__ _Float16 Ws[2][BM * STR];
  __shared__ _Float16 As[2][BN * STR];
  int total = *p.nitems * rgn;
  int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, ln = lane & 15, hi = lane >> 4, rg = wave % RG, th = wave / RG;
  for (int unit = blockIdx.x; unit < total; unit += gridDim.x) {
    int item = unit / rgn, rgi = unit % rgn;
    int e = p.item_e[item], start = p.item_start[item], n = p.item_n[item];
    const uint8_t* We = p.W + (size_t)e * p.wexp;
    int wr = tid >> 1, wh = tid & 1, srow;
    if (EPI == 1) { int h2 = wr >> 5, sub = (wr >> 4) & 1, r = wr & 15; srow = (sub ? FF : 0) + rgi * (16 * RG) + h2 * 16 + r; } else srow = rgi * BM + wr;
    const uint8_t* wp = We + (size_t)srow * RB;
    struct Stg { uint32_t raw[9]; v4u areg[NCH]; };
    const _Float16* ap[NCH]; int arow_l[NCH]; bool aok[NCH];
#pragma unroll
    for (int i = 0; i < NCH; i++) { int idx = tid + NT * i; int row = idx >> 3, ch = idx & 7; aok[i] = idx < BN * 8; arow_l[i] = row * STR + ch * 8;
      int pos = row < n ? start + row : start; int ar = p.actrow ? p.actrow[pos] : pos; ap[i] = p.ah + (size_t)ar * K + ch * 8; }
    auto gload = [&](int kb, Stg& S) {
      if (FMT == FMT_NVFP4) { const uint8_t* b = wp + (size_t)kb * 36; S.raw[0] = *(const uint32_t*)b; v4u q = *(const v4u_a4*)(b + 4 + 16 * wh); S.raw[1] = q.x; S.raw[2] = q.y; S.raw[3] = q.z; S.raw[4] = q.w; }
      else { const uint8_t* b = wp + (size_t)(2 * kb + wh) * 34; S.raw[0] = *(const uint16_t*)b; v4u a = *(const v4u_a2*)(b + 2), c = *(const v4u_a2*)(b + 18);
        S.raw[1] = a.x; S.raw[2] = a.y; S.raw[3] = a.z; S.raw[4] = a.w; S.raw[5] = c.x; S.raw[6] = c.y; S.raw[7] = c.z; S.raw[8] = c.w; }
#pragma unroll
      for (int i = 0; i < NCH; i++) if (aok[i]) S.areg[i] = *(const v4u*)(ap[i] + kb * 64);
    };
    auto sstore = [&](int buf, Stg& S) {
      _Float16* dst = &Ws[buf][wr * STR + 32 * wh];
      if (FMT == FMT_NVFP4) {
#pragma unroll
        for (int s2 = 0; s2 < 2; s2++) { float ds = ue4m3((S.raw[0] >> (16 * wh + 8 * s2)) & 0xFFu); *(v16h_a16*)(dst + 16 * s2) = dq_nv16(S.raw[1 + 2 * s2], S.raw[2 + 2 * s2], ds); }
      } else {
        float d = __half2float(__ushort_as_half((unsigned short)S.raw[0]));
        *(v16h_a16*)dst = dq_q8_16((v4u){S.raw[1], S.raw[2], S.raw[3], S.raw[4]}, d); *(v16h_a16*)(dst + 16) = dq_q8_16((v4u){S.raw[5], S.raw[6], S.raw[7], S.raw[8]}, d);
      }
#pragma unroll
      for (int i = 0; i < NCH; i++) if (aok[i]) *(v4u_a16*)&As[buf][arow_l[i]] = S.areg[i];
    };
    v8f facc[TTW][2];
#pragma unroll
    for (int t = 0; t < TTW; t++) { facc[t][0] = (v8f){0, 0, 0, 0, 0, 0, 0, 0}; facc[t][1] = facc[t][0]; }
    Stg S0, S1;
    gload(0, S0); sstore(0, S0); if (NK > 1) gload(1, S1); if (NK > 2) gload(2, S0); __syncthreads();
    auto body = [&](int kb, Stg& Sn) {
      int cur = kb & 1;
#pragma unroll
      for (int ks = 0; ks < 4; ks++) {
        v16h bf[2];
#pragma unroll
        for (int r = 0; r < 2; r++) bf[r] = *(const v16h_a16*)&Ws[cur][(rg * 32 + 16 * r + ln) * STR + ks * 16];
#pragma unroll
        for (int t = 0; t < TTW; t++) {
          v16h a = *(const v16h_a16*)&As[cur][(th * (BN / 2) + 16 * t + ln) * STR + ks * 16];
#pragma unroll
          for (int r = 0; r < 2; r++) facc[t][r] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, bf[r], facc[t][r]);
        }
      }
      if (kb + 1 < NK) sstore(cur ^ 1, Sn);
      if (kb + 3 < NK) gload(kb + 3, Sn);
      __syncthreads();
    };
    for (int kb = 0; kb < NK; kb += 2) { body(kb, S1); if (kb + 1 < NK) body(kb + 1, S0); }
    float esc = p.escale ? p.escale[e] : 1.f;
#pragma unroll
    for (int t = 0; t < TTW; t++)
#pragma unroll
      for (int i = 0; i < 8; i++) {
        int pl = th * (BN / 2) + 16 * t + 2 * i + hi; if (pl >= n) continue; int pd = start + pl;
        if (EPI == 0) { int row = rgi * BM + rg * 32 + ln; p.out[(size_t)pd * p.ldo + row] = facc[t][0][i]; p.out[(size_t)pd * p.ldo + row + 16] = facc[t][1][i]; }
        else if (EPI == 1) { int j = rgi * (16 * RG) + rg * 16 + ln; p.outh[(size_t)pd * FF + j] = (_Float16)situ(facc[t][0][i] * esc, facc[t][1][i] * esc); }
        else { float w = p.posw[pd] * esc; size_t o = (size_t)p.postok[pd] * p.ldo; int row = rgi * BM + rg * 32 + ln;
          atomicAdd(&p.out[o + row], w * facc[t][0][i]); atomicAdd(&p.out[o + row + 16], w * facc[t][1][i]); }
      }
    __syncthreads();
  }
}
