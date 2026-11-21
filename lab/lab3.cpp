// Task H3 lab: LDS-tiled f16 WMMA grouped GEMM (dense Q8 + MoE NVFP4/Q8_0) with GPU-side expert grouping. MIT.
#include "common.h"
#include "kern3.h"
#include <numeric>
static void* dallocb(size_t n) { void* p; CK(hipMalloc(&p, n)); return p; }
template <class T> static T* dupv(const std::vector<T>& v) { T* p = (T*)dallocb(v.size() * sizeof(T)); CK(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice)); return p; }

template <int FMT, int EPI, int BN, int K> static void L3(const GP3& p, int cap, int rg) { k_gemm3<FMT, EPI, BN, K><<<dim3(cap, rg), 128>>>(p); }
static void dense3(int bn, int K, const GP3& p, int cap, int M) {
  int rg = M / 64;
  if (K == 2304) { if (bn == 64) L3<FMT_Q8, 0, 64, 2304>(p, cap, rg); else L3<FMT_Q8, 0, 128, 2304>(p, cap, rg); }
  else           { if (bn == 64) L3<FMT_Q8, 0, 64, 3072>(p, cap, rg); else L3<FMT_Q8, 0, 128, 3072>(p, cap, rg); }
}
template <int FMT> static void moe3(int stage, int bn, const GP3& p, int cap) {
  if (stage == 0) { if (bn == 32) L3<FMT, 1, 32, 2304>(p, cap, 36); else if (bn == 64) L3<FMT, 1, 64, 2304>(p, cap, 36); else L3<FMT, 1, 128, 2304>(p, cap, 36); }
  else if (stage == 1) { if (bn == 32) L3<FMT, 2, 32, 1152>(p, cap, 36); else if (bn == 64) L3<FMT, 2, 64, 1152>(p, cap, 36); else L3<FMT, 2, 128, 1152>(p, cap, 36); }
  else            { if (bn == 32) L3<FMT, 3, 32, 1152>(p, cap, 36); else if (bn == 64) L3<FMT, 3, 64, 1152>(p, cap, 36); else L3<FMT, 3, 128, 1152>(p, cap, 36); }
}
static void moe3d(int fmt, int stage, int bn, const GP3& p, int cap) { if (fmt == FMT_NVFP4) moe3<FMT_NVFP4>(stage, bn, p, cap); else moe3<FMT_Q8>(stage, bn, p, cap); }

struct RouteDev { int *cnt, *goff, *fill, *nitems, *ie, *is, *in, *postok, *invpos; float* posw; int cap; };
static RouteDev route_alloc(int P, int cap) { RouteDev r; r.cap = cap; r.cnt = (int*)dallocb(128 * 4); r.goff = (int*)dallocb(129 * 4); r.fill = (int*)dallocb(128 * 4); r.nitems = (int*)dallocb(4);
  r.ie = (int*)dallocb(cap * 4); r.is = (int*)dallocb(cap * 4); r.in = (int*)dallocb(cap * 4); r.postok = (int*)dallocb((size_t)P * 4); r.posw = (float*)dallocb((size_t)P * 4); r.invpos = (int*)dallocb((size_t)P * 4); return r; }
static void route_run(RouteDev& r, const int* ids, const float* w, int P, int tile) {
  CK(hipMemsetAsync(r.cnt, 0, 128 * 4, 0)); k_hist<<<(P + 255) / 256, 256>>>(ids, P, r.cnt); k_scan_items<<<1, 128>>>(r.cnt, tile, r.goff, r.fill, r.nitems, r.ie, r.is, r.in);
  k_scatter<<<(P + 255) / 256, 256>>>(ids, w, P, r.goff, r.fill, r.postok, r.posw, r.invpos);
}
static void gen_ids(int U, int seed, std::vector<int>& ids, std::vector<float>& w) {
  std::mt19937 g(seed); ids.resize(U * 8); w.resize(U * 8);
  for (int t = 0; t < U; t++) { std::vector<int> perm(128); std::iota(perm.begin(), perm.end(), 0); std::shuffle(perm.begin(), perm.end(), g); float s = 0;
    for (int k = 0; k < 8; k++) { ids[t * 8 + k] = perm[k]; w[t * 8 + k] = 0.2f + (g() % 1000) / 1000.f; s += w[t * 8 + k]; } for (int k = 0; k < 8; k++) w[t * 8 + k] *= 8.f / s; }
}
static void cvt16(const float* x, _Float16* h, size_t n) { k_cvt_f16<<<(unsigned)((n / 4 + 255) / 256), 256>>>(x, h, n); }

static void run_dense(const Gguf& g, bool do_val, bool do_perf) {
  struct Dn { const char* name; const char* tn; int K, M; std::vector<uint8_t*> p; };
  std::vector<Dn> ds = {{"attn_q", "attn_q.weight", 2304, 3072, {}}, {"attn_k", "attn_k.weight", 2304, 1024, {}}, {"attn_v", "attn_v.weight", 2304, 1024, {}},
    {"attn_gate", "attn_gate.weight", 2304, 3072, {}}, {"attn_output", "attn_output.weight", 3072, 2304, {}}, {"attn_rel", "attn_rel.weight", 2304, 384, {}}};
  for (auto& d : ds) for (int l = 0; l < 24; l++) d.p.push_back(upload(g, LN(l, d.tn)));
  auto items_for = [&](int U, int tile, std::vector<int>& ie, std::vector<int>& is, std::vector<int>& in) { for (int s = 0; s < U; s += tile) { ie.push_back(0); is.push_back(s); in.push_back(std::min(tile, U - s)); } };
  if (do_val) { int U = 70;
    for (int di : {0, 4}) { auto& d = ds[di]; int K = d.K, M = d.M;
      std::vector<float> xh((size_t)U * K); for (int t = 0; t < U; t++) { auto x = gen_x(K, 500 + t); memcpy(&xh[(size_t)t * K], x.data(), K * 4); }
      float* xd = dupv(xh); _Float16* ah = (_Float16*)dallocb((size_t)U * K * 2); cvt16(xd, ah, (size_t)U * K); float* od = (float*)dallocb((size_t)U * M * 4);
      std::vector<uint8_t> Wh(rowbytes(FMT_Q8, K) * M); CK(hipMemcpy(Wh.data(), d.p[3], Wh.size(), hipMemcpyDeviceToHost));
      std::vector<double> r32((size_t)U * M); std::vector<std::vector<double>> xr(U, std::vector<double>(K)); for (int t = 0; t < U; t++) for (int i = 0; i < K; i++) xr[t][i] = xh[(size_t)t * K + i];
      pfor(M, [&](int m) { std::vector<float> w(K); dequant_row(FMT_Q8, Wh.data() + rowbytes(FMT_Q8, K) * m, K, w.data()); for (int t = 0; t < U; t++) r32[(size_t)t * M + m] = dotd(w.data(), xr[t].data(), K); });
      for (int bn : {64, 128}) { std::vector<int> ie, is, in; items_for(U, bn, ie, is, in); int ni = ie.size(); int* nid = dupv(std::vector<int>{ni});
        GP3 p{}; p.W = d.p[3]; p.ah = ah; p.nitems = nid; p.item_e = dupv(ie); p.item_start = dupv(is); p.item_n = dupv(in); p.out = od; p.ldo = M;
        CK(hipMemset(od, 0, (size_t)U * M * 4)); dense3(bn, K, p, ni, M); CK(hipDeviceSynchronize());
        std::vector<float> oh((size_t)U * M); CK(hipMemcpy(oh.data(), od, oh.size() * 4, hipMemcpyDeviceToHost)); Err e32 = cmp(oh.data(), r32.data(), oh.size());
        printf("VAL gemm3_q8 %-11s BN=%d vs-fp32act-ref: err/rms=%.2e maxrel=%.2e\n", d.name, bn, e32.rms_rel, e32.max_rel); }
      CK(hipFree(xd)); CK(hipFree(ah)); CK(hipFree(od)); } }
  if (do_perf) for (int U : {512, 1024, 2048}) { double layer = 0;
    for (auto& d : ds) { int K = d.K, M = d.M; std::vector<float> xh((size_t)U * K); { auto x = gen_x(K, 9); for (int t = 0; t < U; t++) memcpy(&xh[(size_t)t * K], x.data(), K * 4); }
      float* xd = dupv(xh); _Float16* ah = (_Float16*)dallocb((size_t)U * K * 2); cvt16(xd, ah, (size_t)U * K); float* od = (float*)dallocb((size_t)U * M * 4);
      double best = 1e30; int bb = 0;
      for (int bn : {64, 128}) { std::vector<int> ie, is, in; items_for(U, bn, ie, is, in); int ni = ie.size(); int* nid = dupv(std::vector<int>{ni}); int *ied = dupv(ie), *isd = dupv(is), *ind = dupv(in);
        double us = time_us([&](int i) { GP3 p{}; p.W = d.p[i % 24]; p.ah = ah; p.nitems = nid; p.item_e = ied; p.item_start = isd; p.item_n = ind; p.out = od; p.ldo = M; dense3(bn, K, p, ni, M); }, 40, 200);
        printf("PERF gemm3_q8 U=%d %-11s BN=%d %9.1f us %6.1f TFLOPS\n", U, d.name, bn, us, 2.0 * M * K * U / us * 1e-6); if (us < best) { best = us; bb = bn; } }
      printf("BEST gemm3_q8 U=%d %-11s BN=%d %9.1f us %6.1f TFLOPS\n", U, d.name, bb, best, 2.0 * M * K * U / best * 1e-6); layer += best; CK(hipFree(xd)); CK(hipFree(ah)); CK(hipFree(od)); }
    printf("LAYERDENSE3 U=%d dense Q8 projections f16-LDS %.0f us (+cvt of inputs not included; ~10-30 us)\n", U, layer); }
}

static void run_moe(const Gguf& g, bool do_val, bool do_perf) {
  const int FF = 1152, D = 2304, NE = 128;
  for (int fmt : {FMT_NVFP4, FMT_Q8}) {
    if (getenv("FMTONLY") && atoi(getenv("FMTONLY")) != fmt) continue;
    const char* tag = fmt == FMT_NVFP4 ? "nvfp4" : "q8";
    std::vector<uint8_t*> gu, dn; std::vector<float*> sgu, sdn; int l0 = fmt == FMT_NVFP4 ? 0 : 20; int NLw = 3;
    for (int l = l0; l < l0 + NLw; l++) { gu.push_back(upload(g, LN(l, "ffn_gate_up_exps.weight"))); dn.push_back(upload(g, LN(l, "ffn_down_exps.weight")));
      if (fmt == FMT_NVFP4) { sgu.push_back((float*)upload(g, LN(l, "ffn_gate_up_exps.scale"))); sdn.push_back((float*)upload(g, LN(l, "ffn_down_exps.scale"))); } else { sgu.push_back(nullptr); sdn.push_back(nullptr); } }
    size_t rb2 = rowbytes(fmt, 2304), rb1 = rowbytes(fmt, 1152); size_t gub = 2 * FF * rb2, dnb = (size_t)D * rb1;
    if (do_val) {
      int U = 64; std::vector<int> ids; std::vector<float> wv; gen_ids(U, 11, ids, wv); int P = U * 8;
      std::vector<float> xh((size_t)U * 2304); for (int t = 0; t < U; t++) { auto x = gen_x(2304, 600 + t); memcpy(&xh[(size_t)t * 2304], x.data(), 2304 * 4); }
      std::vector<uint8_t> Wg(gub * NE), Wd(dnb * NE); CK(hipMemcpy(Wg.data(), gu[0], Wg.size(), hipMemcpyDeviceToHost)); CK(hipMemcpy(Wd.data(), dn[0], Wd.size(), hipMemcpyDeviceToHost));
      std::vector<float> sg(NE, 1.f), sd(NE, 1.f); if (fmt == FMT_NVFP4) { CK(hipMemcpy(sg.data(), sgu[0], NE * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(sd.data(), sdn[0], NE * 4, hipMemcpyDeviceToHost)); }
      int* idd = dupv(ids); float* wd = dupv(wv); float* xd = dupv(xh); _Float16* xhh = (_Float16*)dallocb((size_t)U * 2304 * 2); cvt16(xd, xhh, (size_t)U * 2304);
      _Float16* ad = (_Float16*)dallocb((size_t)P * FF * 2); _Float16* ad2 = (_Float16*)dallocb((size_t)P * FF * 2); float* aF = (float*)dallocb((size_t)P * FF * 4); float* yd = (float*)dallocb((size_t)U * D * 4);
      for (int bn : {32, 64, 128}) {
        int cap = P / bn + 128; RouteDev rd = route_alloc(P, cap); route_run(rd, idd, wd, P, bn); CK(hipDeviceSynchronize());
        std::vector<int> goff(129), postok(P); std::vector<float> posw(P); CK(hipMemcpy(goff.data(), rd.goff, 129 * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(postok.data(), rd.postok, P * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(posw.data(), rd.posw, P * 4, hipMemcpyDeviceToHost));
        // check grouping: every pair appears once with the right expert
        { std::vector<int> seen(P, 0); int bad = 0; for (int e = 0; e < NE; e++) for (int q = goff[e]; q < goff[e + 1]; q++) { int t = postok[q]; bool ok = false; for (int k = 0; k < 8; k++) if (ids[t * 8 + k] == e && fabsf(wv[t * 8 + k] - posw[q]) < 1e-6f) ok = true; if (!ok) bad++; seen[q]++; }
          for (int q = 0; q < P; q++) if (seen[q] != 1) bad++; if (goff[128] != P) bad++; printf("VAL route_gpu BN=%d bad=%d\n", bn, bad); }
        std::vector<std::vector<double>> xr(U, std::vector<double>(2304)); for (int t = 0; t < U; t++) for (int i = 0; i < 2304; i++) xr[t][i] = xh[(size_t)t * 2304 + i];
        std::vector<double> a32((size_t)P * FF);
        pfor(NE * FF, [&](int idx) { int e = idx / FF, j = idx % FF; std::vector<float> wg(2304), wu(2304); dequant_row(fmt, Wg.data() + e * gub + rb2 * j, 2304, wg.data()); dequant_row(fmt, Wg.data() + e * gub + rb2 * (FF + j), 2304, wu.data());
          for (int q = goff[e]; q < goff[e + 1]; q++) { const double* xx = xr[postok[q]].data(); double gv = dotd(wg.data(), xx, 2304) * sg[e], uv = dotd(wu.data(), xx, 2304) * sg[e];
              a32[(size_t)q * FF + j] = 4 * tanh(gv / 4) * (1 / (1 + exp(-gv))) * 25 * tanh(uv / 25); } });
        GP3 p{}; p.W = gu[0]; p.wexp = gub; p.escale = sgu[0]; p.ah = xhh; p.actrow = rd.postok; p.nitems = rd.nitems; p.item_e = rd.ie; p.item_start = rd.is; p.item_n = rd.in; p.outh = ad; p.ldo = FF;
        CK(hipMemset(ad, 0, (size_t)P * FF * 2)); moe3d(fmt, 0, bn, p, cap); CK(hipDeviceSynchronize());
        std::vector<_Float16> ahh((size_t)P * FF); CK(hipMemcpy(ahh.data(), ad, ahh.size() * 2, hipMemcpyDeviceToHost)); std::vector<float> ah(ahh.size()); for (size_t i = 0; i < ah.size(); i++) ah[i] = (float)ahh[i];
        Err e32 = cmp(ah.data(), a32.data(), ah.size()); printf("VAL moe3_gu_%s BN=%d vs-fp32act-ref: err/rms=%.2e maxrel=%.2e\n", tag, bn, e32.rms_rel, e32.max_rel);
        // down from the fp32 reference a
        std::vector<float> aref((size_t)P * FF); for (size_t i = 0; i < aref.size(); i++) aref[i] = (float)a32[i]; CK(hipMemcpy(aF, aref.data(), aref.size() * 4, hipMemcpyHostToDevice)); cvt16(aF, ad2, (size_t)P * FF);
        std::vector<std::vector<double>> ar(P, std::vector<double>(FF)); for (int q = 0; q < P; q++) for (int i = 0; i < FF; i++) ar[q][i] = aref[(size_t)q * FF + i];
        std::vector<double> y32((size_t)U * D, 0); std::vector<std::vector<double>> c32(P, std::vector<double>(D));
        pfor(NE * D, [&](int idx) { int e = idx / D, r = idx % D; std::vector<float> wr(FF); dequant_row(fmt, Wd.data() + ((size_t)e * D + r) * rb1, FF, wr.data()); for (int q = goff[e]; q < goff[e + 1]; q++) c32[q][r] = (double)posw[q] * sd[e] * dotd(wr.data(), ar[q].data(), FF); });
        for (int q = 0; q < P; q++) for (int r = 0; r < D; r++) y32[(size_t)postok[q] * D + r] += c32[q][r];
        GP3 q{}; q.W = dn[0]; q.wexp = dnb; q.escale = sdn[0]; q.ah = ad2; q.nitems = rd.nitems; q.item_e = rd.ie; q.item_start = rd.is; q.item_n = rd.in; q.out = yd; q.ldo = D; q.postok = rd.postok; q.posw = rd.posw;
        CK(hipMemset(yd, 0, (size_t)U * D * 4)); moe3d(fmt, 1, bn, q, cap); CK(hipDeviceSynchronize());
        std::vector<float> yh((size_t)U * D); CK(hipMemcpy(yh.data(), yd, yh.size() * 4, hipMemcpyDeviceToHost)); Err d32 = cmp(yh.data(), y32.data(), yh.size());
        printf("VAL moe3_down_%s BN=%d vs-fp32act-ref: err/rms=%.2e maxrel=%.2e\n", tag, bn, d32.rms_rel, d32.max_rel);
        { float* y2 = (float*)dallocb((size_t)P * D * 4); q.out = y2; moe3d(fmt, 2, bn, q, cap); k_reduce8<<<U, 256>>>(y2, rd.invpos, yd, D); CK(hipDeviceSynchronize()); CK(hipMemcpy(yh.data(), yd, yh.size() * 4, hipMemcpyDeviceToHost)); Err r32 = cmp(yh.data(), y32.data(), yh.size());
          printf("VAL moe3_down_reduce_%s BN=%d vs-fp32act-ref: err/rms=%.2e maxrel=%.2e\n", tag, bn, r32.rms_rel, r32.max_rel); CK(hipFree(y2)); } }
    }
    if (do_perf) for (int U : {512, 1024, 2048}) {
      if (getenv("UONLY") && atoi(getenv("UONLY")) != U) continue;
      int P = U * 8; std::vector<int> ids; std::vector<float> wv; gen_ids(U, 21, ids, wv); int* idd = dupv(ids); float* wd = dupv(wv);
      std::vector<float> xh((size_t)U * 2304); { auto x = gen_x(2304, 7); for (int t = 0; t < U; t++) memcpy(&xh[(size_t)t * 2304], x.data(), 2304 * 4); }
      float* xd = dupv(xh); _Float16* xhh = (_Float16*)dallocb((size_t)U * 2304 * 2); cvt16(xd, xhh, (size_t)U * 2304);
      _Float16* ad = (_Float16*)dallocb((size_t)P * FF * 2); float* yd = (float*)dallocb((size_t)U * D * 4);
      { std::vector<_Float16> ah((size_t)P * FF); std::mt19937 rr(3); std::normal_distribution<float> nd(0, 0.5f); for (auto& v : ah) v = (_Float16)nd(rr); CK(hipMemcpy(ad, ah.data(), ah.size() * 2, hipMemcpyHostToDevice)); }
      double bgu = 1e30, bdn = 1e30, brt = 1e30; int tgu = 0, tdn = 0;
      for (int bn : {32, 64, 128}) { int cap = P / bn + 128; RouteDev rd = route_alloc(P, cap);
        double rt = time_us([&](int) { route_run(rd, idd, wd, P, bn); }, 20, 500); route_run(rd, idd, wd, P, bn);
        double us1 = time_us([&](int i) { GP3 p{}; p.W = gu[i % NLw]; p.wexp = gub; p.escale = sgu[i % NLw]; p.ah = xhh; p.actrow = rd.postok; p.nitems = rd.nitems; p.item_e = rd.ie; p.item_start = rd.is; p.item_n = rd.in; p.outh = ad; p.ldo = FF; moe3d(fmt, 0, bn, p, cap); }, 40, 100);
        double us2 = time_us([&](int i) { CK(hipMemsetAsync(yd, 0, (size_t)U * D * 4, 0)); GP3 q{}; q.W = dn[i % NLw]; q.wexp = dnb; q.escale = sdn[i % NLw]; q.ah = ad; q.nitems = rd.nitems; q.item_e = rd.ie; q.item_start = rd.is; q.item_n = rd.in; q.out = yd; q.ldo = D; q.postok = rd.postok; q.posw = rd.posw; moe3d(fmt, 1, bn, q, cap); }, 40, 100);
        float* y2 = (float*)dallocb((size_t)P * D * 4);
        double us3 = time_us([&](int i) { GP3 q{}; q.W = dn[i % NLw]; q.wexp = dnb; q.escale = sdn[i % NLw]; q.ah = ad; q.nitems = rd.nitems; q.item_e = rd.ie; q.item_start = rd.is; q.item_n = rd.in; q.out = y2; q.ldo = D; q.postok = rd.postok; q.posw = rd.posw; moe3d(fmt, 2, bn, q, cap); k_reduce8<<<U, 256>>>(y2, rd.invpos, yd, D); }, 40, 100);
        CK(hipFree(y2));
        if (us3 < us2) us2 = us3;   // best of atomics vs per-slot write + reduce
        double flg = 2.0 * 2304 * 2304 * P, fld = 2.0 * 2304 * 1152 * P;
        printf("PERF moe3_%s U=%d BN=%d route %.1f us | gate_up %9.1f us %5.1f TFLOPS | down %9.1f us %5.1f TFLOPS (best of atomics/%.0f us reduce-variant)\n", tag, U, bn, rt, us1, flg / us1 * 1e-6, us2, fld / us2 * 1e-6, us3);
        if (us1 < bgu) { bgu = us1; tgu = bn; } if (us2 < bdn) { bdn = us2; tdn = bn; } brt = std::min(brt, rt); }
      double wbytes = 128.0 * (gub + dnb);
      printf("BEST moe3_%s U=%d: gate_up %.1f us (BN=%d), down %.1f us (BN=%d), route %.1f us, total %.1f us; weight-stream floor %.0f us\n", tag, U, bgu, tgu, bdn, tdn, brt, bgu + bdn + brt, wbytes / 242e3);
    }
  }
}
int main(int argc, char** argv) {
  std::string only; bool do_val = true, do_perf = true; std::string gpath = std::string(getenv("HOME")) + "/models/inkling2-mini/unsloth/Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file";
  for (int i = 1; i < argc; i++) { std::string a = argv[i]; if (a == "--only" && i + 1 < argc) only = argv[++i]; else if (a == "--noval") do_val = false; else if (a == "--noperf") do_perf = false; }
  Gguf g; if (!g.open(gpath.c_str())) { fprintf(stderr, "cannot open gguf\n"); return 1; }
  CK(hipEventCreate(&ev0)); CK(hipEventCreate(&ev1));
  if (only.empty() || only == "dense") run_dense(g, do_val, do_perf);
  if (only.empty() || only == "moe") run_moe(g, do_val, do_perf);
  return 0;
}
