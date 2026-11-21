// Task H2 lab: fused multi-matrix GEMV, router, WMMA prefill GEMMs (dense + grouped MoE) on real GGUF tensors. MIT.
#include "common.h"
#include "kern2.h"
#include <numeric>

static bool want(const std::string& only, const char* k) { return only.empty() || only == k; }
static void* dallocb(size_t n) { void* p; CK(hipMalloc(&p, n)); return p; }
template <class T> static T* dupv(const std::vector<T>& v) { T* p = (T*)dallocb(v.size() * sizeof(T)); CK(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice)); return p; }

// ================= (1) multi-matrix GEMV =================
struct Sep { const uint8_t* W; int M; int in; };
template <int B, int LPR, int NIT, int NI> static void launch_multi(const MultiP& P, const float* x, float* y, int Mtot, hipStream_t s) {
  constexpr int BW = 4, RPB = BW * (WS / LPR) * NIT;
  k_gemv_multi<2304, LPR, B, NIT, BW, NI><<<(Mtot + RPB - 1) / RPB, BW * WS, 0, s>>>(P, x, y, Mtot);
}
template <int B, int NI> static void multi_dispatch(int cfg, const MultiP& P, const float* x, float* y, int Mtot) {
  switch (cfg) { case 0: launch_multi<B, 8, 1, NI>(P, x, y, Mtot, 0); break; case 1: launch_multi<B, 8, 2, NI>(P, x, y, Mtot, 0); break; case 2: launch_multi<B, 16, 1, NI>(P, x, y, Mtot, 0); break;
    case 3: launch_multi<B, 16, 2, NI>(P, x, y, Mtot, 0); break; case 4: launch_multi<B, 16, 4, NI>(P, x, y, Mtot, 0); break; default: launch_multi<B, 32, 2, NI>(P, x, y, Mtot, 0); break; }
}
static const char* cfgname[6] = {"lpr8_nit1", "lpr8_nit2", "lpr16_nit1", "lpr16_nit2", "lpr16_nit4", "lpr32_nit2"};
template <int B> static void sep_launch(const Sep& s, const float* x, float* y, hipStream_t st) {   // baseline = lab.cpp best-ish config lpr16/nit2/bw4
  constexpr int RPB = 4 * (WS / 16) * 2; k_gemv<FMT_Q8, 2304, 16, B, 2, 4><<<(s.M + RPB - 1) / RPB, 4 * WS, 0, st>>>(s.W, x, y, s.M, 0.f);
}

static void run_multi(const Gguf& g, bool do_val) {
  const char* names[5] = {"attn_q.weight", "attn_gate.weight", "attn_rel.weight", "attn_k.weight", "attn_v.weight"};
  const int Ms[5] = {3072, 3072, 384, 1024, 1024};
  std::vector<uint8_t*> W[5]; for (int i = 0; i < 5; i++) for (int l = 0; l < 24; l++) W[i].push_back(upload(g, LN(l, names[i])));
  for (int B : {1, 2, 4}) for (int NI : {1, 3}) {
    int Mtot = 0; for (int i = 0; i < 5; i++) Mtot += Ms[i];
    std::vector<float> xh((size_t)NI * B * 2304); for (int v = 0; v < NI * B; v++) { auto x = gen_x(2304, 300 + v); memcpy(&xh[(size_t)v * 2304], x.data(), 2304 * 4); }
    float* xd = dupv(xh); float* ym = (float*)dallocb((size_t)Mtot * B * 4); float* ys = (float*)dallocb((size_t)Mtot * B * 4);
    int inp[5] = {0, 0, 0, NI == 3 ? 1 : 0, NI == 3 ? 2 : 0};
    auto mk = [&](int l) { MultiP P{}; P.n = 5; int c = 0; for (int i = 0; i < 5; i++) { P.W[i] = W[i][l]; P.M[i] = Ms[i]; P.in[i] = inp[i]; P.yoff[i] = c; P.cum[i] = c; c += Ms[i]; } P.cum[5] = c; return P; };
    auto run_sep = [&](int l, float* y) {   // y layout per matrix [b][M] at yoff*B, input slice (in*B*K)
      int c = 0; for (int i = 0; i < 5; i++) { const float* xi = xd + (size_t)inp[i] * B * 2304; Sep s{W[i][l], Ms[i], inp[i]};
        if (B == 1) sep_launch<1>(s, xi, y + (size_t)c * B, 0); else if (B == 2) sep_launch<2>(s, xi, y + (size_t)c * B, 0); else sep_launch<4>(s, xi, y + (size_t)c * B, 0); c += Ms[i]; } };
    auto run_m = [&](int cfg, int l, float* y) { MultiP P = mk(l);
      if (B == 1) { if (NI == 1) multi_dispatch<1, 1>(cfg, P, xd, y, Mtot); else multi_dispatch<1, 3>(cfg, P, xd, y, Mtot); }
      else if (B == 2) { if (NI == 1) multi_dispatch<2, 1>(cfg, P, xd, y, Mtot); else multi_dispatch<2, 3>(cfg, P, xd, y, Mtot); }
      else { if (NI == 1) multi_dispatch<4, 1>(cfg, P, xd, y, Mtot); else multi_dispatch<4, 3>(cfg, P, xd, y, Mtot); } };
    double bytes = 0; for (int i = 0; i < 5; i++) bytes += (double)Ms[i] * 2448;
    if (do_val) { run_sep(3, ys); CK(hipDeviceSynchronize()); std::vector<float> a((size_t)Mtot * B), b((size_t)Mtot * B);
      CK(hipMemcpy(a.data(), ys, a.size() * 4, hipMemcpyDeviceToHost));
      for (int cfg = 0; cfg < 6; cfg++) { CK(hipMemset(ym, 0, a.size() * 4)); run_m(cfg, 3, ym); CK(hipDeviceSynchronize()); CK(hipMemcpy(b.data(), ym, b.size() * 4, hipMemcpyDeviceToHost));
        double md = 0; for (size_t i = 0; i < a.size(); i++) md = std::max(md, (double)fabsf(a[i] - b[i])); printf("VAL multi B=%d NI=%d %s max|multi-separate|=%.3g\n", B, NI, cfgname[cfg], md); } }
    double ts = time_us([&](int it) { run_sep(it % 24, ys); });
    printf("PERF multi B=%d NI=%d separate(5 launches) %8.2f us %7.1f GB/s (%.1f MB)\n", B, NI, ts, bytes / ts * 1e-3, bytes / 1e6);
    double best = 1e30; int bc = 0;
    for (int cfg = 0; cfg < 6; cfg++) { double us = time_us([&](int it) { run_m(cfg, it % 24, ym); }); printf("PERF multi B=%d NI=%d fused %-11s %8.2f us %7.1f GB/s\n", B, NI, cfgname[cfg], us, bytes / us * 1e-3); if (us < best) { best = us; bc = cfg; } }
    printf("BEST multi B=%d NI=%d fused %s %.2f us vs separate %.2f us (%.2fx)\n", B, NI, cfgname[bc], best, ts, ts / best);
    CK(hipFree(xd)); CK(hipFree(ym)); CK(hipFree(ys));
  }
}

// ================= router =================
static void run_router(const Gguf& g, bool do_val) {
  std::vector<float*> Wd, bd; std::vector<std::vector<float>> Wh, bh;
  for (int l = 0; l < 24; l++) { Wd.push_back((float*)upload(g, LN(l, "ffn_gate_inp.weight"))); bd.push_back((float*)upload(g, LN(l, "exp_probs_b.bias"))); }
  float* logit = (float*)dallocb(8 * 128 * 4); unsigned* cnt = (unsigned*)dallocb(8 * 4); CK(hipMemset(cnt, 0, 32));
  int* ids = (int*)dallocb(8 * 8 * 4); float* wts = (float*)dallocb(8 * 8 * 4);
  for (int ntok : {1, 4, 512}) {
    std::vector<float> xh((size_t)ntok * 2304); for (int b = 0; b < ntok; b++) { auto x = gen_x(2304, 400 + b); memcpy(&xh[(size_t)b * 2304], x.data(), 2304 * 4); }
    float* xd = dupv(xh);
    float* lg = (float*)dallocb((size_t)ntok * 128 * 4); unsigned* ct = (unsigned*)dallocb(ntok * 4); CK(hipMemset(ct, 0, ntok * 4));
    int* id = (int*)dallocb((size_t)ntok * 8 * 4); float* wt = (float*)dallocb((size_t)ntok * 8 * 4);
    auto run = [&](int l) { k_router<2304><<<dim3(16, ntok), 256>>>(Wd[l], bd[l], xd, lg, ct, id, wt); };
    if (do_val && ntok <= 4) {
      run(5); CK(hipDeviceSynchronize()); std::vector<int> ih(ntok * 8); std::vector<float> wh(ntok * 8); CK(hipMemcpy(ih.data(), id, ih.size() * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(wh.data(), wt, wh.size() * 4, hipMemcpyDeviceToHost));
      std::vector<float> W((size_t)128 * 2304), bias(128); CK(hipMemcpy(W.data(), Wd[5], W.size() * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(bias.data(), bd[5], 512, hipMemcpyDeviceToHost));
      int idbad = 0; double wmax = 0;
      for (int t = 0; t < ntok; t++) { std::vector<double> sc(128), sel(128); for (int r = 0; r < 128; r++) { double s = 0; for (int k = 0; k < 2304; k++) s += (double)W[(size_t)r * 2304 + k] * xh[(size_t)t * 2304 + k]; sc[r] = 1 / (1 + exp(-s)); sel[r] = sc[r] + bias[r]; }
        std::vector<int> ord(128); std::iota(ord.begin(), ord.end(), 0); std::sort(ord.begin(), ord.end(), [&](int a, int b) { return sel[a] > sel[b]; });
        double ss = 0; for (int k = 0; k < 8; k++) ss += sc[ord[k]];
        for (int k = 0; k < 8; k++) { if (ih[t * 8 + k] != ord[k]) idbad++; wmax = std::max(wmax, fabs(wh[t * 8 + k] - sc[ord[k]] / (ss + 1e-20) * 8)); } }
      printf("VAL router ntok=%d id mismatches (ordered)=%d max|w err|=%.3g\n", ntok, idbad, wmax);
    }
    double us = time_us([&](int it) { run(it % 24); });
    printf("PERF router ntok=%d %8.2f us (1.18 MB F32 weights, %.1f GB/s)\n", ntok, us, 1.18 / us * 1e3);
    CK(hipFree(xd)); CK(hipFree(lg)); CK(hipFree(ct)); CK(hipFree(id)); CK(hipFree(wt));
  }
}

// ================= (2) prefill GEMMs =================
struct Route { int U, P; std::vector<int> ids, postok, goff, cnt; std::vector<float> posw; };
static Route make_route(int U, int seed) {
  Route r; r.U = U; r.P = U * 8; std::mt19937 g(seed); r.ids.resize(U * 8); r.cnt.assign(128, 0); std::vector<float> w(U * 8);
  for (int t = 0; t < U; t++) { std::vector<int> perm(128); std::iota(perm.begin(), perm.end(), 0); std::shuffle(perm.begin(), perm.end(), g); float s = 0;
    for (int k = 0; k < 8; k++) { r.ids[t * 8 + k] = perm[k]; w[t * 8 + k] = 0.2f + (g() % 1000) / 1000.f; s += w[t * 8 + k]; } for (int k = 0; k < 8; k++) w[t * 8 + k] *= 8.f / s; }
  for (int p = 0; p < U * 8; p++) r.cnt[r.ids[p]]++;
  r.goff.assign(129, 0); for (int e = 0; e < 128; e++) r.goff[e + 1] = r.goff[e] + r.cnt[e];
  r.postok.resize(r.P); r.posw.resize(r.P); std::vector<int> fill(128, 0);
  for (int p = 0; p < U * 8; p++) { int e = r.ids[p]; int pos = r.goff[e] + fill[e]++; r.postok[pos] = p / 8; r.posw[pos] = w[p]; }
  return r;
}
struct Items { std::vector<int> e, start, n; };
static Items make_items(const Route& r, int tile) {
  Items it; for (int e = 0; e < 128; e++) for (int s = 0; s < r.cnt[e]; s += tile) { it.e.push_back(e); it.start.push_back(r.goff[e] + s); it.n.push_back(std::min(tile, r.cnt[e] - s)); } return it;
}
struct DevItems { int* e; int* s; int* n; int count; };
static DevItems up_items(const Items& it) { return {dupv(it.e), dupv(it.start), dupv(it.n), (int)it.e.size()}; }

template <int FMT, int MODE, int EPI, int TT, int K> static void launch_gemm(const GP& p, int nitems, int rowgroups) { k_gemm<FMT, MODE, EPI, TT, K><<<dim3(nitems, rowgroups), 128>>>(p); }
template <int FMT, int MODE, int EPI, int TT, int K> static void launch_gemm_(const GP& p, int ni, int rg) { launch_gemm<FMT, MODE, EPI, TT, K>(p, ni, rg); }

struct ActBufs { int8_t* q = nullptr; float* sc = nullptr; _Float16* h = nullptr; };
static void prep_act(int mode, const float* x, ActBufs& a, int rows, int K) {
  if (mode == 0) { int n = rows * (K / 32); k_quant_rows_i8<<<(n + 255) / 256, 256>>>(x, a.q, a.sc, rows, K, rows); }
  else { size_t n = (size_t)rows * K; k_cvt_f16<<<(unsigned)((n / 4 + 255) / 256), 256>>>(x, a.h, n); }
}
static ActBufs alloc_act(int rows, int K) { ActBufs a; a.q = (int8_t*)dallocb((size_t)rows * K); a.sc = (float*)dallocb((size_t)rows * (K / 32) * 4); a.h = (_Float16*)dallocb((size_t)rows * K * 2); return a; }

// dense dispatch (K in {2304,3072}, TT in {2,4})
static void dense_launch(int mode, int tt, int K, GP p, int ni, int M) {
  int rg = M / 128;
#define D_(K_, MODE_, TT_) launch_gemm_<FMT_Q8, MODE_, 0, TT_, K_>(p, ni, rg)
  if (K == 2304) { if (mode == 0) { if (tt == 2) D_(2304, 0, 2); else D_(2304, 0, 4); } else { if (tt == 2) D_(2304, 1, 2); else D_(2304, 1, 4); } }
  else           { if (mode == 0) { if (tt == 2) D_(3072, 0, 2); else D_(3072, 0, 4); } else { if (tt == 2) D_(3072, 1, 2); else D_(3072, 1, 4); } }
#undef D_
}
template <int FMT> static void moe_launch(int stage, int mode, int tt, GP p, int ni) {   // stage 0 gate_up SITU (K=2304), 1 down (K=1152)
#define M_(ST, MODE_, TT_) if (stage == 0) launch_gemm_<FMT, MODE_, 1, TT_, 2304>(p, ni, 18); else launch_gemm_<FMT, MODE_, 2, TT_, 1152>(p, ni, 18)
  if (mode == 0) { if (tt == 2) { M_(0, 0, 2); } else { M_(0, 0, 4); } } else { if (tt == 2) { M_(0, 1, 2); } else { M_(0, 1, 4); } }
#undef M_
}
static void moe_dispatch(int fmt, int stage, int mode, int tt, GP p, int ni) { if (fmt == FMT_NVFP4) moe_launch<FMT_NVFP4>(stage, mode, tt, p, ni); else moe_launch<FMT_Q8>(stage, mode, tt, p, ni); }

static void run_dense(const Gguf& g, bool do_val, bool do_perf) {
  struct Dn { const char* name; const char* tn; int K, M; std::vector<uint8_t*> p; };
  std::vector<Dn> ds = {{"attn_q", "attn_q.weight", 2304, 3072, {}}, {"attn_k", "attn_k.weight", 2304, 1024, {}}, {"attn_v", "attn_v.weight", 2304, 1024, {}},
    {"attn_gate", "attn_gate.weight", 2304, 3072, {}}, {"attn_output", "attn_output.weight", 3072, 2304, {}}, {"attn_rel", "attn_rel.weight", 2304, 384, {}}};
  for (auto& d : ds) for (int l = 0; l < 24; l++) d.p.push_back(upload(g, LN(l, d.tn)));
  if (do_val) {
    int U = 70; // partial last tile on purpose
    for (int di : {0, 4}) { auto& d = ds[di]; int K = d.K, M = d.M;
      std::vector<float> xh((size_t)U * K); for (int t = 0; t < U; t++) { auto x = gen_x(K, 500 + t); memcpy(&xh[(size_t)t * K], x.data(), K * 4); }
      float* xd = dupv(xh); ActBufs ab = alloc_act(U, K); float* od = (float*)dallocb((size_t)U * M * 4);
      std::vector<uint8_t> Wh(rowbytes(FMT_Q8, K) * M); CK(hipMemcpy(Wh.data(), d.p[3], Wh.size(), hipMemcpyDeviceToHost));
      std::vector<double> r32((size_t)U * M), rq((size_t)U * M); std::vector<std::vector<double>> xr(U, std::vector<double>(K)), xq(U, std::vector<double>(K));
      for (int t = 0; t < U; t++) { quant_act_ref(&xh[(size_t)t * K], K, xq[t].data()); for (int i = 0; i < K; i++) xr[t][i] = xh[(size_t)t * K + i]; }
      pfor(M, [&](int m) { std::vector<float> w(K); dequant_row(FMT_Q8, Wh.data() + rowbytes(FMT_Q8, K) * m, K, w.data()); for (int t = 0; t < U; t++) { r32[(size_t)t * M + m] = dotd(w.data(), xr[t].data(), K); rq[(size_t)t * M + m] = dotd(w.data(), xq[t].data(), K); } });
      for (int mode = 0; mode < 2; mode++) for (int tt : {2, 4}) {
        prep_act(mode, xd, ab, U, K); int tile = tt * 16; Items it; for (int s = 0; s < U; s += tile) { it.e.push_back(0); it.start.push_back(s); it.n.push_back(std::min(tile, U - s)); }
        DevItems di2 = up_items(it); GP p{}; p.W = d.p[3]; p.wexp = 0; p.aq = ab.q; p.ah = ab.h; p.asc = ab.sc; p.Rpad = U; p.item_e = di2.e; p.item_start = di2.s; p.item_n = di2.n; p.out = od; p.ldo = M;
        CK(hipMemset(od, 0, (size_t)U * M * 4)); dense_launch(mode, tt, K, p, di2.count, M); CK(hipDeviceSynchronize());
        std::vector<float> oh((size_t)U * M); CK(hipMemcpy(oh.data(), od, oh.size() * 4, hipMemcpyDeviceToHost)); Err e32 = cmp(oh.data(), r32.data(), oh.size()), eq = cmp(oh.data(), rq.data(), oh.size());
        printf("VAL gemm_q8 %-11s %s TT=%d vs-q8act-ref: err/rms=%.2e maxrel=%.2e | vs-fp32act-ref: err/rms=%.2e maxrel=%.2e\n", d.name, mode ? "f16 " : "iu8 ", tt, eq.rms_rel, eq.max_rel, e32.rms_rel, e32.max_rel);
        CK(hipFree(di2.e)); CK(hipFree(di2.s)); CK(hipFree(di2.n)); }
      CK(hipFree(xd)); CK(hipFree(od)); }
  }
  if (do_perf) for (int U : {512, 1024, 2048}) {
    if (getenv("UONLY") && atoi(getenv("UONLY")) != U) continue;
    double layer[2] = {0, 0}; double layer_gq[2] = {0, 0};
    for (auto& d : ds) { int K = d.K, M = d.M;
      std::vector<float> xh((size_t)U * K); { auto x = gen_x(K, 9); for (int t = 0; t < U; t++) memcpy(&xh[(size_t)t * K], x.data(), K * 4); for (size_t i = 0; i < xh.size(); i += 7) xh[i] *= 1.0f + (i % 13) * 0.01f; }
      float* xd = dupv(xh); ActBufs ab = alloc_act(U, K); float* od = (float*)dallocb((size_t)U * M * 4);
      for (int mode = 0; mode < 2; mode++) { prep_act(mode, xd, ab, U, K); double qus = time_us([&](int) { prep_act(mode, xd, ab, U, K); });
        double best = 1e30; int bt = 0;
        for (int tt : {2, 4}) { int tile = tt * 16; Items it; for (int s = 0; s < U; s += tile) { it.e.push_back(0); it.start.push_back(s); it.n.push_back(std::min(tile, U - s)); } DevItems di2 = up_items(it);
          double us = time_us([&](int i) { GP p{}; p.W = d.p[i % 24]; p.aq = ab.q; p.ah = ab.h; p.asc = ab.sc; p.Rpad = U; p.item_e = di2.e; p.item_start = di2.s; p.item_n = di2.n; p.out = od; p.ldo = M; dense_launch(mode, tt, K, p, di2.count, M); }, 40, 200);
          double tops = 2.0 * M * K * U / us * 1e-6; printf("PERF gemm_q8 U=%d %-11s %s TT=%d %9.1f us %6.1f T%s\n", U, d.name, mode ? "f16" : "iu8", tt, us, tops, mode ? "FLOPS" : "OPS");
          if (us < best) { best = us; bt = tt; } CK(hipFree(di2.e)); CK(hipFree(di2.s)); CK(hipFree(di2.n)); }
        printf("BEST gemm_q8 U=%d %-11s %s TT=%d %9.1f us %6.1f T (quant %.1f us)\n", U, d.name, mode ? "f16" : "iu8", bt, best, 2.0 * M * K * U / best * 1e-6, qus);
        layer[mode] += best; layer_gq[mode] += qus; }
      CK(hipFree(xd)); CK(hipFree(od)); }
    printf("LAYERDENSE U=%d dense Q8 projections (q,k,v,gate,o,rel) iu8 %.0f us (+quant %.0f) | f16 %.0f us (+cvt %.0f)\n", U, layer[0], layer_gq[0], layer[1], layer_gq[1]);
  }
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
      int U = 64; Route R = make_route(U, 11); std::vector<float> xh((size_t)U * 2304); for (int t = 0; t < U; t++) { auto x = gen_x(2304, 600 + t); memcpy(&xh[(size_t)t * 2304], x.data(), 2304 * 4); }
      std::vector<uint8_t> Wg(gub * NE), Wd(dnb * NE); CK(hipMemcpy(Wg.data(), gu[0], Wg.size(), hipMemcpyDeviceToHost)); CK(hipMemcpy(Wd.data(), dn[0], Wd.size(), hipMemcpyDeviceToHost));
      std::vector<float> sg(NE, 1.f), sd(NE, 1.f); if (fmt == FMT_NVFP4) { CK(hipMemcpy(sg.data(), sgu[0], NE * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(sd.data(), sdn[0], NE * 4, hipMemcpyDeviceToHost)); }
      std::vector<std::vector<double>> xr(U, std::vector<double>(2304)), xq(U, std::vector<double>(2304)); for (int t = 0; t < U; t++) { quant_act_ref(&xh[(size_t)t * 2304], 2304, xq[t].data()); for (int i = 0; i < 2304; i++) xr[t][i] = xh[(size_t)t * 2304 + i]; }
      std::vector<int> epos(R.P); for (int e = 0; e < NE; e++) for (int p = R.goff[e]; p < R.goff[e + 1]; p++) epos[p] = e;
      std::vector<double> a32((size_t)R.P * FF), aq((size_t)R.P * FF);
      pfor(NE * FF, [&](int idx) { int e = idx / FF, j = idx % FF; std::vector<float> wg(2304), wu(2304); dequant_row(fmt, Wg.data() + e * gub + rb2 * j, 2304, wg.data()); dequant_row(fmt, Wg.data() + e * gub + rb2 * (FF + j), 2304, wu.data());
        for (int p = R.goff[e]; p < R.goff[e + 1]; p++) for (int m = 0; m < 2; m++) { const double* xx = m ? xq[R.postok[p]].data() : xr[R.postok[p]].data(); double gv = dotd(wg.data(), xx, 2304) * sg[e], uv = dotd(wu.data(), xx, 2304) * sg[e];
            (m ? aq : a32)[(size_t)p * FF + j] = 4 * tanh(gv / 4) * (1 / (1 + exp(-gv))) * 25 * tanh(uv / 25); } });
      float* xd = dupv(xh); float* ad = (float*)dallocb((size_t)R.P * FF * 4); float* yd = (float*)dallocb((size_t)U * D * 4); int* ptok = dupv(R.postok); float* pw = dupv(R.posw);
      ActBufs ax = alloc_act(U, 2304), aa = alloc_act(R.P, FF);
      // down reference from the fp32 a reference
      std::vector<float> aref((size_t)R.P * FF); for (size_t i = 0; i < aref.size(); i++) aref[i] = (float)a32[i];
      std::vector<std::vector<double>> ar(R.P, std::vector<double>(FF)), aqq(R.P, std::vector<double>(FF)); for (int p = 0; p < R.P; p++) { quant_act_ref(&aref[(size_t)p * FF], FF, aqq[p].data()); for (int i = 0; i < FF; i++) ar[p][i] = aref[(size_t)p * FF + i]; }
      std::vector<double> y32((size_t)U * D, 0), yq((size_t)U * D, 0); std::vector<std::vector<double>> c32(R.P, std::vector<double>(D)), cq(R.P, std::vector<double>(D));
      pfor(NE * D, [&](int idx) { int e = idx / D, r = idx % D; std::vector<float> wr(FF); dequant_row(fmt, Wd.data() + ((size_t)e * D + r) * rb1, FF, wr.data());
        for (int p = R.goff[e]; p < R.goff[e + 1]; p++) { double w = (double)R.posw[p] * sd[e]; c32[p][r] = w * dotd(wr.data(), ar[p].data(), FF); cq[p][r] = w * dotd(wr.data(), aqq[p].data(), FF); } });
      for (int p = 0; p < R.P; p++) for (int r = 0; r < D; r++) { y32[(size_t)R.postok[p] * D + r] += c32[p][r]; yq[(size_t)R.postok[p] * D + r] += cq[p][r]; }
      for (int mode = 0; mode < 2; mode++) for (int tt : {2, 4}) {
        Items it = make_items(R, tt * 16); DevItems di = up_items(it);
        prep_act(mode, xd, ax, U, 2304); GP p{}; p.W = gu[0]; p.wexp = gub; p.escale = sgu[0]; p.aq = ax.q; p.ah = ax.h; p.asc = ax.sc; p.Rpad = U; p.actrow = dupv(R.postok); p.item_e = di.e; p.item_start = di.s; p.item_n = di.n; p.out = ad; p.ldo = FF;
        CK(hipMemset(ad, 0, (size_t)R.P * FF * 4)); moe_dispatch(fmt, 0, mode, tt, p, di.count); CK(hipDeviceSynchronize());
        std::vector<float> ah((size_t)R.P * FF); CK(hipMemcpy(ah.data(), ad, ah.size() * 4, hipMemcpyDeviceToHost)); Err e32 = cmp(ah.data(), a32.data(), ah.size()), eq = cmp(ah.data(), aq.data(), ah.size());
        printf("VAL moe_gu_%s %s TT=%d vs-q8act-ref: err/rms=%.2e maxrel=%.2e | vs-fp32act-ref: err/rms=%.2e maxrel=%.2e\n", tag, mode ? "f16 " : "iu8 ", tt, eq.rms_rel, eq.max_rel, e32.rms_rel, e32.max_rel);
        CK(hipMemcpy(ad, aref.data(), aref.size() * 4, hipMemcpyHostToDevice)); prep_act(mode, ad, aa, R.P, FF);
        GP q{}; q.W = dn[0]; q.wexp = dnb; q.escale = sdn[0]; q.aq = aa.q; q.ah = aa.h; q.asc = aa.sc; q.Rpad = R.P; q.item_e = di.e; q.item_start = di.s; q.item_n = di.n; q.out = yd; q.ldo = D; q.postok = ptok; q.posw = pw;
        CK(hipMemset(yd, 0, (size_t)U * D * 4)); moe_dispatch(fmt, 1, mode, tt, q, di.count); CK(hipDeviceSynchronize());
        std::vector<float> yh((size_t)U * D); CK(hipMemcpy(yh.data(), yd, yh.size() * 4, hipMemcpyDeviceToHost)); Err d32 = cmp(yh.data(), y32.data(), yh.size()), dq = cmp(yh.data(), yq.data(), yh.size());
        printf("VAL moe_down_%s %s TT=%d vs-q8act-ref: err/rms=%.2e maxrel=%.2e | vs-fp32act-ref: err/rms=%.2e maxrel=%.2e\n", tag, mode ? "f16 " : "iu8 ", tt, dq.rms_rel, dq.max_rel, d32.rms_rel, d32.max_rel);
        CK(hipFree(di.e)); CK(hipFree(di.s)); CK(hipFree(di.n)); }
    }
    if (do_perf) for (int U : {512, 1024, 2048}) {
      if (getenv("UONLY") && atoi(getenv("UONLY")) != U) continue;
      Route R = make_route(U, 21); std::vector<float> xh((size_t)U * 2304); { auto x = gen_x(2304, 7); for (int t = 0; t < U; t++) memcpy(&xh[(size_t)t * 2304], x.data(), 2304 * 4); }
      float* xd = dupv(xh); float* ad = (float*)dallocb((size_t)R.P * FF * 4); float* yd = (float*)dallocb((size_t)U * D * 4); int* ptok = dupv(R.postok); float* pw = dupv(R.posw); int* arow = dupv(R.postok);
      { std::vector<float> ah((size_t)R.P * FF); std::mt19937 rr(3); std::normal_distribution<float> nd(0, 0.5f); for (auto& v : ah) v = nd(rr); CK(hipMemcpy(ad, ah.data(), ah.size() * 4, hipMemcpyHostToDevice)); }
      ActBufs ax = alloc_act(U, 2304), aa = alloc_act(R.P, FF);
      double bestsum[2] = {0, 0};
      for (int mode = 0; mode < 2; mode++) {
        double qgu = time_us([&](int) { prep_act(mode, xd, ax, U, 2304); }), qdn = time_us([&](int) { prep_act(mode, ad, aa, R.P, FF); });
        double bgu = 1e30, bdn = 1e30; int tgu = 0, tdn = 0;
        for (int tt : {2, 4}) { Items it = make_items(R, tt * 16); DevItems di = up_items(it);
          prep_act(mode, xd, ax, U, 2304); prep_act(mode, ad, aa, R.P, FF);
          double us1 = time_us([&](int i) { GP p{}; p.W = gu[i % NLw]; p.wexp = gub; p.escale = sgu[i % NLw]; p.aq = ax.q; p.ah = ax.h; p.asc = ax.sc; p.Rpad = U; p.actrow = arow; p.item_e = di.e; p.item_start = di.s; p.item_n = di.n; p.out = ad; p.ldo = FF; moe_dispatch(fmt, 0, mode, tt, p, di.count); }, 40, 100);
          CK(hipMemcpy(ad, ad, 0, hipMemcpyDeviceToDevice));
          double us2 = time_us([&](int i) { CK(hipMemsetAsync(yd, 0, (size_t)U * D * 4, 0)); GP q{}; q.W = dn[i % NLw]; q.wexp = dnb; q.escale = sdn[i % NLw]; q.aq = aa.q; q.ah = aa.h; q.asc = aa.sc; q.Rpad = R.P; q.item_e = di.e; q.item_start = di.s; q.item_n = di.n; q.out = yd; q.ldo = D; q.postok = ptok; q.posw = pw; moe_dispatch(fmt, 1, mode, tt, q, di.count); }, 40, 100);
          double flg = 2.0 * 2304 * 2304 * R.P, fld = 2.0 * 2304 * 1152 * R.P;
          printf("PERF moe_gu_%s U=%d %s TT=%d %9.1f us %6.1f T  | moe_down_%s %s TT=%d %9.1f us %6.1f T (incl memset)\n", tag, U, mode ? "f16" : "iu8", tt, us1, flg / us1 * 1e-6, tag, mode ? "f16" : "iu8", tt, us2, fld / us2 * 1e-6);
          if (us1 < bgu) { bgu = us1; tgu = tt; } if (us2 < bdn) { bdn = us2; tdn = tt; } CK(hipFree(di.e)); CK(hipFree(di.s)); CK(hipFree(di.n)); }
        double wbytes = 128.0 * (gub + dnb);
        printf("BEST moe_%s U=%d %s: gate_up %.1f us (TT=%d, quant %.1f us), down %.1f us (TT=%d, quant %.1f us), total %.1f us; weight-stream floor @242GB/s %.0f us\n", tag, U, mode ? "f16" : "iu8", bgu, tgu, qgu, bdn, tdn, qdn, bgu + bdn + qgu + qdn, wbytes / 242e3);
        bestsum[mode] = bgu + bdn + qgu + qdn; }
      printf("LAYERMOE_%s U=%d iu8 %.0f us | f16 %.0f us\n", tag, U, bestsum[0], bestsum[1]);
      CK(hipFree(xd)); CK(hipFree(ad)); CK(hipFree(yd));
    }
    for (auto p : gu) CK(hipFree(p)); for (auto p : dn) CK(hipFree(p));
  }
}

int main(int argc, char** argv) {
  std::string only; bool do_val = true, do_perf = true; std::string gpath = std::string(getenv("HOME")) + "/models/inkling2-mini/unsloth/Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file";
  for (int i = 1; i < argc; i++) { std::string a = argv[i]; if (a == "--only" && i + 1 < argc) only = argv[++i]; else if (a == "--noval") do_val = false; else if (a == "--noperf") do_perf = false; }
  Gguf g; if (!g.open(gpath.c_str())) { fprintf(stderr, "cannot open gguf\n"); return 1; }
  CK(hipEventCreate(&ev0)); CK(hipEventCreate(&ev1));
  if (want(only, "multi")) run_multi(g, do_val);
  if (want(only, "router")) run_router(g, do_val);
  if (want(only, "dense")) run_dense(g, do_val, do_perf);
  if (want(only, "moe")) run_moe(g, do_val, do_perf);
  return 0;
}
