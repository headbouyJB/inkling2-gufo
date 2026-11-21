// Task H4 lab: sweeps of k_gemm4 variants (packed-half dequant, 4/8 waves, persistent CTAs) vs k_gemm3, MoE gate_up/down + dense. MIT.
#define main lab3_main
#include "lab3.cpp"
#undef main
#include "kern5.h"
#ifndef GEMM
#define GEMM k_gemm5
#endif

template <int FMT, int EPI, int BN, int K, int RG, int PS> static void L4(const GP3& p, int grid, int rgn) { GEMM<FMT, EPI, BN, K, RG, PS><<<grid, 64 * RG>>>(p, rgn); }
template <int FMT, int EPI, int K, int RG, int PS> static bool L4bn(int bn, const GP3& p, int grid, int rgn) {
  if (bn == 32) L4<FMT, EPI, 32, K, RG, PS>(p, grid, rgn); else if (bn == 64) L4<FMT, EPI, 64, K, RG, PS>(p, grid, rgn);
  else if constexpr (RG == 2) { if (bn == 128) L4<FMT, EPI, 128, K, RG, PS>(p, grid, rgn); else return false; } else return false; return true; }
template <int FMT, int EPI, int K> static bool L4v(int bn, int rg, int ps, const GP3& p, int grid, int rgn) {
  if (rg == 2) return ps ? L4bn<FMT, EPI, K, 2, 1>(bn, p, grid, rgn) : L4bn<FMT, EPI, K, 2, 0>(bn, p, grid, rgn);
  return ps ? L4bn<FMT, EPI, K, 4, 1>(bn, p, grid, rgn) : L4bn<FMT, EPI, K, 4, 0>(bn, p, grid, rgn); }
static bool disp4(int fmt, int stage, int bn, int rg, int ps, const GP3& p, int grid, int rgn) {
  if (fmt == FMT_NVFP4) return stage == 0 ? L4v<FMT_NVFP4, 1, 2304>(bn, rg, ps, p, grid, rgn) : L4v<FMT_NVFP4, 2, 1152>(bn, rg, ps, p, grid, rgn);
  return stage == 0 ? L4v<FMT_Q8, 1, 2304>(bn, rg, ps, p, grid, rgn) : L4v<FMT_Q8, 2, 1152>(bn, rg, ps, p, grid, rgn);
}
struct Var { int bn, rg, ps, ctas; };

int main(int argc, char** argv) {
  std::string gpath = std::string(getenv("HOME")) + "/models/inkling2-mini/unsloth/Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file"; bool do_val = true;
  for (int i = 1; i < argc; i++) if (std::string(argv[i]) == "--noval") do_val = false;
  Gguf g; if (!g.open(gpath.c_str())) return 1; CK(hipEventCreate(&ev0)); CK(hipEventCreate(&ev1));
  const int FF = 1152, D = 2304; int NLw = 3;
  std::vector<Var> vars; for (int bn : {32, 64, 128}) for (int rg : {2, 4}) { if (rg == 4 && bn == 128) continue; vars.push_back({bn, rg, 0, 0}); vars.push_back({bn, rg, 1, 160}); vars.push_back({bn, rg, 1, 400}); }
  for (int fmt : {FMT_NVFP4, FMT_Q8}) {
    if (getenv("FMTONLY") && atoi(getenv("FMTONLY")) != fmt) continue;
    const char* tag = fmt == FMT_NVFP4 ? "nvfp4" : "q8"; int l0 = fmt == FMT_NVFP4 ? 0 : 20;
    std::vector<uint8_t*> gu, dn; std::vector<float*> sgu, sdn;
    for (int l = l0; l < l0 + NLw; l++) { gu.push_back(upload(g, LN(l, "ffn_gate_up_exps.weight"))); dn.push_back(upload(g, LN(l, "ffn_down_exps.weight")));
      if (fmt == FMT_NVFP4) { sgu.push_back((float*)upload(g, LN(l, "ffn_gate_up_exps.scale"))); sdn.push_back((float*)upload(g, LN(l, "ffn_down_exps.scale"))); } else { sgu.push_back(nullptr); sdn.push_back(nullptr); } }
    size_t rb2 = rowbytes(fmt, 2304), rb1 = rowbytes(fmt, 1152); size_t gub = 2 * FF * rb2, dnb = (size_t)D * rb1;
    for (int U : {512, 2048}) {
      if (getenv("UONLY") && atoi(getenv("UONLY")) != U) continue;
      int P = U * 8; std::vector<int> ids; std::vector<float> wv; gen_ids(U, 21, ids, wv); int* idd = dupv(ids); float* wd = dupv(wv);
      std::vector<float> xh((size_t)U * 2304); { auto x = gen_x(2304, 7); for (int t = 0; t < U; t++) memcpy(&xh[(size_t)t * 2304], x.data(), 2304 * 4); }
      float* xd = dupv(xh); _Float16* xhh = (_Float16*)dallocb((size_t)U * 2304 * 2); cvt16(xd, xhh, (size_t)U * 2304);
      _Float16* ad = (_Float16*)dallocb((size_t)P * FF * 2); _Float16* ad4 = (_Float16*)dallocb((size_t)P * FF * 2); float* yd = (float*)dallocb((size_t)U * D * 4); float* yd4 = (float*)dallocb((size_t)U * D * 4);
      { std::vector<_Float16> ah((size_t)P * FF); std::mt19937 rr(3); std::normal_distribution<float> nd(0, 0.5f); for (auto& v : ah) v = (_Float16)nd(rr); CK(hipMemcpy(ad, ah.data(), ah.size() * 2, hipMemcpyHostToDevice)); }
      double bgu3 = 1e30, bdn3 = 1e30; int bn3g = 0, bn3d = 0;
      std::vector<RouteDev> rds;
      for (int bn : {32, 64, 128}) { int cap = P / bn + 128; RouteDev rd = route_alloc(P, cap); route_run(rd, idd, wd, P, bn); CK(hipDeviceSynchronize()); rds.push_back(rd);
        double u1 = time_us([&](int i) { GP3 p{}; p.W = gu[i % NLw]; p.wexp = gub; p.escale = sgu[i % NLw]; p.ah = xhh; p.actrow = rd.postok; p.nitems = rd.nitems; p.item_e = rd.ie; p.item_start = rd.is; p.item_n = rd.in; p.outh = ad4; p.ldo = FF; moe3d(fmt, 0, bn, p, cap); }, 40, 100);
        double u2 = time_us([&](int i) { CK(hipMemsetAsync(yd, 0, (size_t)U * D * 4, 0)); GP3 q{}; q.W = dn[i % NLw]; q.wexp = dnb; q.escale = sdn[i % NLw]; q.ah = ad; q.nitems = rd.nitems; q.item_e = rd.ie; q.item_start = rd.is; q.item_n = rd.in; q.out = yd; q.ldo = D; q.postok = rd.postok; q.posw = rd.posw; moe3d(fmt, 1, bn, q, cap); }, 40, 100);
        printf("PERF4 %s U=%d base(k_gemm3) BN=%d gate_up %9.1f us | down %9.1f us\n", tag, U, bn, u1, u2); if (u1 < bgu3) { bgu3 = u1; bn3g = bn; } if (u2 < bdn3) { bdn3 = u2; bn3d = bn; } }
      double bg = 1e30, bd = 1e30; Var vg{}, vd{};
      for (auto& v : vars) { int ri = v.bn == 32 ? 0 : v.bn == 64 ? 1 : 2; RouteDev& rd = rds[ri]; int cap = P / v.bn + 128; int rgn1 = FF / (16 * v.rg), rgn2 = D / (32 * v.rg);
        int g1 = v.ps ? v.ctas : cap * rgn1, g2 = v.ps ? v.ctas : cap * rgn2;
        GP3 p0{}; p0.W = gu[0]; p0.wexp = gub; p0.escale = sgu[0]; p0.ah = xhh; p0.actrow = rd.postok; p0.nitems = rd.nitems; p0.item_e = rd.ie; p0.item_start = rd.is; p0.item_n = rd.in; p0.outh = ad4; p0.ldo = FF;
        if (!disp4(fmt, 0, v.bn, v.rg, v.ps, p0, g1, rgn1)) continue;
        double u1 = time_us([&](int i) { GP3 p = p0; p.W = gu[i % NLw]; p.escale = sgu[i % NLw]; disp4(fmt, 0, v.bn, v.rg, v.ps, p, g1, rgn1); }, 40, 100);
        double u2 = time_us([&](int i) { CK(hipMemsetAsync(yd4, 0, (size_t)U * D * 4, 0)); GP3 q{}; q.W = dn[i % NLw]; q.wexp = dnb; q.escale = sdn[i % NLw]; q.ah = ad; q.nitems = rd.nitems; q.item_e = rd.ie; q.item_start = rd.is; q.item_n = rd.in; q.out = yd4; q.ldo = D; q.postok = rd.postok; q.posw = rd.posw; disp4(fmt, 1, v.bn, v.rg, v.ps, q, g2, rgn2); }, 40, 100);
        printf("PERF4 %s U=%d gemm4 BN=%d RG=%d persist=%d ctas=%d gate_up %9.1f us %5.1f TF | down %9.1f us %5.1f TF\n", tag, U, v.bn, v.rg, v.ps, v.ctas, u1, 2.0 * 2304 * 2304 * P / u1 * 1e-6, u2, 2.0 * 2304 * 1152 * P / u2 * 1e-6);
        if (u1 < bg) { bg = u1; vg = v; } if (u2 < bd) { bd = u2; vd = v; } }
      printf("BEST4 %s U=%d: base gate_up %.1f (BN=%d) down %.1f (BN=%d) | gemm4 gate_up %.1f (BN=%d RG=%d ps=%d ctas=%d) down %.1f (BN=%d RG=%d ps=%d ctas=%d) | total %.1f us vs base %.1f us\n", tag, U, bgu3, bn3g, bdn3, bn3d, bg, vg.bn, vg.rg, vg.ps, vg.ctas, bd, vd.bn, vd.rg, vd.ps, vd.ctas, bg + bd, bgu3 + bdn3);
      if (do_val) {   // gemm4 best variants vs k_gemm3 same-BN outputs (k_gemm3 is validated vs CPU in h3)
        for (int st = 0; st < 2; st++) { Var v = st ? vd : vg; int ri = v.bn == 32 ? 0 : v.bn == 64 ? 1 : 2; RouteDev& rd = rds[ri]; int cap = P / v.bn + 128; int rgn1 = FF / (16 * v.rg), rgn2 = D / (32 * v.rg);
          int g1 = v.ps ? v.ctas : cap * rgn1, g2 = v.ps ? v.ctas : cap * rgn2;
          if (st == 0) { GP3 p{}; p.W = gu[0]; p.wexp = gub; p.escale = sgu[0]; p.ah = xhh; p.actrow = rd.postok; p.nitems = rd.nitems; p.item_e = rd.ie; p.item_start = rd.is; p.item_n = rd.in; p.ldo = FF;
            p.outh = ad; CK(hipMemset(ad, 0, (size_t)P * FF * 2)); moe3d(fmt, 0, v.bn, p, cap); p.outh = ad4; CK(hipMemset(ad4, 0, (size_t)P * FF * 2)); disp4(fmt, 0, v.bn, v.rg, v.ps, p, g1, rgn1); CK(hipDeviceSynchronize());
            std::vector<_Float16> a((size_t)P * FF), b((size_t)P * FF); CK(hipMemcpy(a.data(), ad, a.size() * 2, hipMemcpyDeviceToHost)); CK(hipMemcpy(b.data(), ad4, b.size() * 2, hipMemcpyDeviceToHost));
            std::vector<float> bf(b.size()); std::vector<double> af(a.size()); for (size_t i = 0; i < a.size(); i++) { bf[i] = (float)b[i]; af[i] = (float)a[i]; } Err e = cmp(bf.data(), af.data(), a.size()); printf("VAL4 %s U=%d gate_up gemm4 vs gemm3 (diff of validated kernel): err/rms=%.2e maxrel=%.2e\n", tag, U, e.rms_rel, e.max_rel);
            CK(hipMemcpy(ad, ad, 0, hipMemcpyDeviceToDevice)); }
          else { GP3 q{}; q.W = dn[0]; q.wexp = dnb; q.escale = sdn[0]; q.ah = ad4; q.nitems = rd.nitems; q.item_e = rd.ie; q.item_start = rd.is; q.item_n = rd.in; q.ldo = D; q.postok = rd.postok; q.posw = rd.posw;
            q.out = yd; CK(hipMemset(yd, 0, (size_t)U * D * 4)); moe3d(fmt, 1, v.bn, q, cap); q.out = yd4; CK(hipMemset(yd4, 0, (size_t)U * D * 4)); disp4(fmt, 1, v.bn, v.rg, v.ps, q, g2, rgn2); CK(hipDeviceSynchronize());
            std::vector<float> a((size_t)U * D), b((size_t)U * D); CK(hipMemcpy(a.data(), yd, a.size() * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(b.data(), yd4, b.size() * 4, hipMemcpyDeviceToHost)); std::vector<double> ad_(a.begin(), a.end());
            Err e = cmp(b.data(), ad_.data(), a.size()); printf("VAL4 %s U=%d down gemm4 vs gemm3: err/rms=%.2e maxrel=%.2e\n", tag, U, e.rms_rel, e.max_rel); } }
      }
    }
  }
  return 0;
}
