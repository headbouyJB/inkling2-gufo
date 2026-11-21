// Kernel lab: validation + benchmark of decode-critical Inkling 2 Mini matrix kernels on real GGUF tensors. MIT.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <thread>
#include <vector>
#include <string>
#include "gguf.h"
#include "kern.h"

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1);} } while (0)
static void pfor(int n, const std::function<void(int)>& f) {
  int nt = std::min<int>(std::thread::hardware_concurrency(), 32); std::vector<std::thread> th;
  for (int t = 0; t < nt; t++) th.emplace_back([&, t] { for (int i = t; i < n; i += nt) f(i); });
  for (auto& x : th) x.join();
}
// ---------------- CPU reference (exact dequant, double accumulate) ----------------
static float h2f(uint16_t h) {
  int s = h >> 15, e = (h >> 10) & 31, m = h & 1023; float v;
  if (e == 0) v = ldexpf((float)m, -24); else if (e == 31) v = m ? NAN : INFINITY; else v = ldexpf(1.f + m / 1024.f, e - 15);
  return s ? -v : v;
}
static float ue4m3_ref(uint8_t x) {   // ggml_ue4m3_to_fp32
  if (x == 0 || x == 0x7F) return 0.f; int e = (x >> 3) & 15, m = x & 7; float raw = e == 0 ? ldexpf((float)m, -9) : ldexpf(1.f + m / 8.f, e - 7); return raw * 0.5f;
}
static const int8_t KV[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
static void dequant_row(int fmt, const uint8_t* row, int K, float* out) {
  if (fmt == FMT_Q8) { for (int b = 0; b < K / 32; b++) { const uint8_t* p = row + b * 34; uint16_t h; memcpy(&h, p, 2); float d = h2f(h);
      for (int i = 0; i < 32; i++) out[b * 32 + i] = d * (float)(int8_t)p[2 + i]; } }
  else { for (int b = 0; b < K / 64; b++) { const uint8_t* p = row + b * 36; for (int s = 0; s < 4; s++) { float d = ue4m3_ref(p[s]);
        for (int j = 0; j < 8; j++) { uint8_t q = p[4 + s * 8 + j]; out[b * 64 + s * 16 + j] = KV[q & 15] * d; out[b * 64 + s * 16 + j + 8] = KV[q >> 4] * d; } } } }
}
static void quant_act_ref(const float* x, int K, double* xq) {   // same scheme as kernel: per-32 amax/127, round-to-nearest-even
  for (int j = 0; j < K / 32; j++) { float amax = 0; for (int i = 0; i < 32; i++) amax = fmaxf(amax, fabsf(x[j * 32 + i]));
    float id = amax > 0 ? 127.f / amax : 0.f, sc = amax / 127.f; for (int i = 0; i < 32; i++) xq[j * 32 + i] = (double)(rintf(x[j * 32 + i] * id) * sc); }
}
static double dotd(const float* w, const double* x, int K) { double s = 0; for (int i = 0; i < K; i++) s += (double)w[i] * x[i]; return s; }
static size_t rowbytes(int fmt, int K) { return fmt == FMT_Q8 ? (size_t)K / 32 * 34 : (size_t)K / 64 * 36; }

struct Err { double rms_rel = 0, max_rel = 0; };
static Err cmp(const float* k, const double* r, size_t n) {
  double ss = 0; for (size_t i = 0; i < n; i++) ss += r[i] * r[i]; double rms = sqrt(ss / n), mx = 0, mr = 0;
  for (size_t i = 0; i < n; i++) { double e = fabs(k[i] - r[i]); mx = std::max(mx, e); mr = std::max(mr, e / std::max(fabs(r[i]), 0.01 * rms)); }
  return {mx / rms, mr};
}
static std::vector<float> gen_x(int K, int seed) {
  std::mt19937 g(seed); std::normal_distribution<float> nd(0, 1); std::vector<float> x(K); double ss = 0;
  for (int i = 0; i < K; i++) { x[i] = nd(g) * (g() % 100 == 0 ? 5.f : 1.f); ss += x[i] * x[i]; }
  float s = 1.f / sqrtf(ss / K); for (auto& v : x) v *= s; return x;   // RMS-normalized with ~1% outliers
}
// ---------------- variants ----------------
using GemvFn = std::function<void(const uint8_t*, const float*, float*, int, float, hipStream_t)>;
using GuFn   = std::function<void(const uint8_t*, const float*, const float*, const int*, float*, int ntok, hipStream_t)>;
using DownFn = std::function<void(const uint8_t*, const float*, const float*, const int*, const float*, const float*, float*, int ntok, hipStream_t)>;
template <int FMT, int K, int LPR, int B, int NIT, int BW> GemvFn mk_gemv() {
  return [](const uint8_t* W, const float* x, float* y, int M, float sc, hipStream_t s) { constexpr int RPB = BW * (WS / LPR) * NIT;
    k_gemv<FMT, K, LPR, B, NIT, BW><<<(M + RPB - 1) / RPB, BW * WS, 0, s>>>(W, x, y, M, sc); };
}
template <int FMT, int K, int LPR, int NIT, int BW> GuFn mk_gu() {
  return [](const uint8_t* W, const float* sc, const float* x, const int* ids, float* a, int nt, hipStream_t s) { constexpr int RPB = BW * (WS / LPR / 2) * NIT;
    k_moe_gu<FMT, K, LPR, NIT, BW><<<dim3((1152 + RPB - 1) / RPB, nt * 8), BW * WS, 0, s>>>(W, sc, x, ids, a, 1152); };
}
template <int FMT, int K, int LPR, int NIT, int BW> DownFn mk_down() {
  return [](const uint8_t* W, const float* sc, const float* a, const int* ids, const float* wt, const float* res, float* y, int nt, hipStream_t s) { constexpr int RPB = BW * (WS / LPR) * NIT;
    k_moe_down<FMT, K, LPR, NIT, BW><<<dim3((2304 + RPB - 1) / RPB, nt), BW * WS, 0, s>>>(W, sc, a, ids, wt, res, y, 2304); };
}
template <class F> struct Named { std::string name; F fn; };
#define NAMEDV(v, L, N, W, expr) v.push_back({"lpr" #L "_nit" #N "_bw" #W, expr});
template <int K, int B> void reg_gemv(std::vector<Named<GemvFn>>& v) {
#define G(L, N, W) NAMEDV(v, L, N, W, (mk_gemv<FMT_Q8, K, L, B, N, W>()))
  G(4,1,4) G(4,2,4) G(4,4,4) G(8,1,4) G(8,2,4) G(8,4,4) G(16,1,4) G(16,2,4) G(16,4,4) G(8,1,8) G(8,2,8) G(4,2,8) G(32,1,4) G(32,2,4)
#undef G
}
template <int FMT> void reg_moe(std::vector<Named<GuFn>>& gu, std::vector<Named<DownFn>>& dn) {
#define M_(L, N, W) NAMEDV(gu, L, N, W, (mk_gu<FMT, 2304, L, N, W>())) NAMEDV(dn, L, N, W, (mk_down<FMT, 1152, L, N, W>()))
  M_(2,1,4) M_(2,2,4) M_(2,4,4) M_(4,1,4) M_(4,2,4) M_(4,4,4) M_(8,1,4) M_(8,2,4) M_(8,4,4) M_(16,1,4) M_(16,2,4) M_(16,4,4) M_(4,2,8) M_(8,2,8)
#undef M_
#define GU_(L, N, W) NAMEDV(gu, L, N, W, (mk_gu<FMT, 2304, L, N, W>()))
#define DN_(L, N, W) NAMEDV(dn, L, N, W, (mk_down<FMT, 1152, L, N, W>()))
  GU_(16,1,2) GU_(16,1,8) GU_(8,1,2) GU_(8,1,8) GU_(16,2,8)
  DN_(32,1,4) DN_(32,1,2) DN_(32,1,8) DN_(16,1,2) DN_(16,1,8) DN_(8,1,2) DN_(8,1,8) DN_(4,1,8) DN_(32,2,4)
#undef GU_
#undef DN_
}
// ---------------- timing ----------------
static hipEvent_t ev0, ev1;
static double time_us(const std::function<void(int)>& f, double target_ms = 25, int maxit = 3000) {
  for (int i = 0; i < 5; i++) f(i); CK(hipDeviceSynchronize());
  CK(hipEventRecord(ev0)); for (int i = 0; i < 10; i++) f(i); CK(hipEventRecord(ev1)); CK(hipEventSynchronize(ev1));
  float ms; CK(hipEventElapsedTime(&ms, ev0, ev1)); int it = (int)std::min<double>(maxit, std::max(20.0, target_ms / (ms / 10)));
  double best = 1e30;
  for (int r = 0; r < 3; r++) { CK(hipEventRecord(ev0)); for (int i = 0; i < it; i++) f(i + 10 * r); CK(hipEventRecord(ev1)); CK(hipEventSynchronize(ev1));
    CK(hipEventElapsedTime(&ms, ev0, ev1)); best = std::min<double>(best, ms * 1000.0 / it); }
  return best;
}
template <class T> T* dalloc(size_t n) { T* p; CK(hipMalloc(&p, n * sizeof(T))); return p; }
template <class T> T* dup(const std::vector<T>& v) { T* p = dalloc<T>(v.size()); CK(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice)); return p; }
static uint8_t* upload(const Gguf& g, const std::string& n) { const TInfo& t = g.t.at(n); uint8_t* p = dalloc<uint8_t>(t.bytes); CK(hipMemcpy(p, g.ptr(n), t.bytes, hipMemcpyHostToDevice)); return p; }
static std::string LN(int l, const char* s) { return "blk." + std::to_string(l) + "." + s; }
static bool match(const std::string& s, const std::string& f) { return f.empty() || s.find(f) != std::string::npos; }

int main(int argc, char** argv) {
  std::string only; int NLN = 4; bool do_val = true, do_perf = true; std::string gpath = std::string(getenv("HOME")) + "/models/inkling2-mini/unsloth/Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file";
  for (int i = 1; i < argc; i++) { std::string a = argv[i]; if (a == "--only" && i + 1 < argc) only = argv[++i]; else if (a == "--noval") do_val = false; else if (a == "--noperf") do_perf = false;
    else if (a == "--gguf" && i + 1 < argc) gpath = argv[++i]; else if (a == "--nvlayers" && i + 1 < argc) NLN = atoi(argv[++i]); }
  Gguf g; if (!g.open(gpath.c_str())) { fprintf(stderr, "cannot open gguf\n"); return 1; }
  CK(hipEventCreate(&ev0)); CK(hipEventCreate(&ev1));
  hipDeviceProp_t pr; CK(hipGetDeviceProperties(&pr, 0)); printf("# device %s warp=%d WS=%d\n", pr.gcnArchName, pr.warpSize, WS);
  const int FF = 1152, D = 2304, NE = 128, NL = 24;
  // ---- upload ----
  struct Dense { const char* name; const char* tn; int K, M; std::vector<uint8_t*> p; };
  std::vector<Dense> dense = {{"attn_q", "attn_q.weight", 2304, 3072, {}}, {"attn_k", "attn_k.weight", 2304, 1024, {}}, {"attn_v", "attn_v.weight", 2304, 1024, {}},
    {"attn_gate", "attn_gate.weight", 2304, 3072, {}}, {"attn_output", "attn_output.weight", 3072, 2304, {}}, {"attn_rel", "attn_rel.weight", 2304, 384, {}}};
  for (auto& d : dense) { if (!only.empty() && !match(d.name, only) && !match("dense", only) && !match("gemv", only)) continue; for (int l = 0; l < NL; l++) { const TInfo& t = g.t.at(LN(l, d.tn));
      if (t.type != 8 || t.ne[0] != (uint64_t)d.K || t.ne[1] != (uint64_t)d.M) { fprintf(stderr, "unexpected tensor %s\n", t.name.c_str()); return 1; } d.p.push_back(upload(g, LN(l, d.tn))); } }
  uint8_t* lm = nullptr; if (match("lm_head", only) || only.empty()) lm = upload(g, "output.weight");
  struct MoeW { std::vector<uint8_t*> gu, dn; std::vector<float*> sgu, sdn; std::vector<int> layer; };
  MoeW nv, q8;
  bool want_moe = only.empty() || match("moe", only) || match("nvfp4", only) || match("q8x", only) || match("gu", only) || match("down", only);
  if (want_moe) {
    for (int l = 0; l < NLN; l++) { nv.layer.push_back(l); nv.gu.push_back(upload(g, LN(l, "ffn_gate_up_exps.weight"))); nv.dn.push_back(upload(g, LN(l, "ffn_down_exps.weight")));
      nv.sgu.push_back((float*)upload(g, LN(l, "ffn_gate_up_exps.scale"))); nv.sdn.push_back((float*)upload(g, LN(l, "ffn_down_exps.scale"))); }
    for (int l = 20; l < 24; l++) { q8.layer.push_back(l); q8.gu.push_back(upload(g, LN(l, "ffn_gate_up_exps.weight"))); q8.dn.push_back(upload(g, LN(l, "ffn_down_exps.weight"))); }
  }
  printf("# tensors uploaded\n");
  const int Bs[3] = {1, 2, 4};
  // ================= DENSE Q8_0 GEMV (+ lm_head softcap) =================
  auto run_dense = [&](const char* name, int K, int M, const std::vector<uint8_t*>& ptrs, float softcap, int layer_for_val) {
    if (ptrs.empty()) return;
    for (int bi = 0; bi < 3; bi++) {
      int B = Bs[bi]; std::vector<Named<GemvFn>> vs;
      if (K == 2304) { if (B == 1) reg_gemv<2304, 1>(vs); else if (B == 2) reg_gemv<2304, 2>(vs); else reg_gemv<2304, 4>(vs); }
      else           { if (B == 1) reg_gemv<3072, 1>(vs); else if (B == 2) reg_gemv<3072, 2>(vs); else reg_gemv<3072, 4>(vs); }
      std::vector<float> xh(B * K); for (int b = 0; b < B; b++) { auto x = gen_x(K, 100 + b); memcpy(&xh[b * K], x.data(), K * 4); }
      float* xd = dup(xh); float* yd = dalloc<float>((size_t)B * M);
      if (do_val) {
        const uint8_t* W = ptrs[layer_for_val % ptrs.size()]; std::vector<uint8_t> Wh(rowbytes(FMT_Q8, K) * M); CK(hipMemcpy(Wh.data(), W, Wh.size(), hipMemcpyDeviceToHost));
        std::vector<double> r32((size_t)B * M), rq((size_t)B * M); std::vector<std::vector<double>> xq(B, std::vector<double>(K)), xr(B, std::vector<double>(K));
        for (int b = 0; b < B; b++) { quant_act_ref(&xh[b * K], K, xq[b].data()); for (int i = 0; i < K; i++) xr[b][i] = xh[b * K + i]; }
        pfor(M, [&](int m) { std::vector<float> w(K); dequant_row(FMT_Q8, Wh.data() + rowbytes(FMT_Q8, K) * m, K, w.data());
          for (int b = 0; b < B; b++) { r32[(size_t)b * M + m] = dotd(w.data(), xr[b].data(), K); rq[(size_t)b * M + m] = dotd(w.data(), xq[b].data(), K); } });
        if (softcap > 0) for (size_t i = 0; i < r32.size(); i++) { r32[i] = softcap * tanh(r32[i] / softcap); rq[i] = softcap * tanh(rq[i] / softcap); }
        Err w32{}, wq{}; std::vector<float> yh((size_t)B * M); int nfail = 0;
        for (auto& v : vs) { CK(hipMemset(yd, 0, yh.size() * 4)); v.fn(W, xd, yd, M, softcap, 0); CK(hipDeviceSynchronize()); CK(hipMemcpy(yh.data(), yd, yh.size() * 4, hipMemcpyDeviceToHost));
          Err e32 = cmp(yh.data(), r32.data(), yh.size()), eq = cmp(yh.data(), rq.data(), yh.size());
          if (getenv("DBG") && &v == &vs[0]) for (int i = 0; i < 8; i++) printf("DBG m=%d kern=%g q8ref=%g fp32ref=%g\n", i, yh[i], rq[i], r32[i]);
          if (eq.rms_rel > 1e-4) { nfail++; printf("VALFAIL gemv_q8 %s B=%d %s q8ref max/rms=%.3e\n", name, B, v.name.c_str(), eq.rms_rel); }
          w32.rms_rel = std::max(w32.rms_rel, e32.rms_rel); w32.max_rel = std::max(w32.max_rel, e32.max_rel); wq.rms_rel = std::max(wq.rms_rel, eq.rms_rel); wq.max_rel = std::max(wq.max_rel, eq.max_rel); }
        printf("VAL gemv_q8 %-12s B=%d  vs-q8act-ref: maxerr/rms=%.2e maxrel=%.2e | vs-fp32act-ref: maxerr/rms=%.2e maxrel=%.2e | variants failed=%d/%zu\n", name, B, wq.rms_rel, wq.max_rel, w32.rms_rel, w32.max_rel, nfail, vs.size());
      }
      if (do_perf) {
        double bytes = (double)rowbytes(FMT_Q8, K) * M; double best = 1e30; std::string bn;
        for (auto& v : vs) { double us = time_us([&](int it) { v.fn(ptrs[it % ptrs.size()], xd, yd, M, softcap, 0); }, M > 100000 ? 60 : 25, M > 100000 ? 40 : 3000);
          printf("PERF gemv_q8 %-12s B=%d %-16s %9.2f us %7.1f GB/s\n", name, B, v.name.c_str(), us, bytes / us * 1e-3); if (us < best) { best = us; bn = v.name; } }
        printf("BEST gemv_q8 %-12s B=%d %-16s %9.2f us %7.1f GB/s (%.1f MB)\n", name, B, bn.c_str(), best, bytes / best * 1e-3, bytes / 1e6);
      }
      hipFree(xd); hipFree(yd);
    }
  };
  for (auto& d : dense) if (!d.p.empty() && (only.empty() || match(d.name, only) || match("dense", only) || match("gemv", only))) run_dense(d.name, d.K, d.M, d.p, 0.f, 3);
  if (lm && match("lm_head", only)) run_dense("lm_head", 2304, 201024, {lm}, 30.f, 0);
  // ================= MoE =================
  auto run_moe = [&](const char* tag, int fmt, MoeW& w) {
    if (w.gu.empty()) return;
    std::vector<Named<GuFn>> gus; std::vector<Named<DownFn>> dns; if (fmt == FMT_NVFP4) reg_moe<FMT_NVFP4>(gus, dns); else reg_moe<FMT_Q8>(gus, dns);
    const int NID = 64; std::mt19937 rng(7); std::vector<int> ids(NID * 32); std::vector<float> wts(NID * 32);
    for (int p = 0; p < NID * 4; p++) { std::vector<int> perm(NE); for (int i = 0; i < NE; i++) perm[i] = i; std::shuffle(perm.begin(), perm.end(), rng);
      float s = 0; for (int e = 0; e < 8; e++) { ids[p * 8 + e] = perm[e]; wts[p * 8 + e] = 0.2f + (rng() % 1000) / 1000.f; s += wts[p * 8 + e]; } for (int e = 0; e < 8; e++) wts[p * 8 + e] *= 8.f / s; }
    std::vector<float> xh(4 * 2304); for (int b = 0; b < 4; b++) { auto x = gen_x(2304, 200 + b); memcpy(&xh[b * 2304], x.data(), 2304 * 4); }
    int* idd = dup(ids); float* wtd = dup(wts); float* xd = dup(xh); float* ad = dalloc<float>(32 * FF); float* yd = dalloc<float>(4 * D);
    std::vector<float> rh(4 * D); { std::mt19937 r2(5); std::normal_distribution<float> nd(0, 1); for (auto& v : rh) v = nd(r2); } float* resd = dup(rh);
    size_t rb2 = rowbytes(fmt, 2304), rb1 = rowbytes(fmt, 1152); size_t gub = 2 * FF * rb2, dnb = D * rb1;
    // ---- validation (layer idx 0, id pattern 0, B tokens=2 distinct) ----
    std::vector<float> a_ref_f;   // fp32-act reference a for token 0..1, used as down input
    if (do_val && (only == "moe" || match(std::string(tag) + " gu", only))) {
      int nt = 2; std::vector<uint8_t> Wh(gub * NE); CK(hipMemcpy(Wh.data(), w.gu[0], Wh.size(), hipMemcpyDeviceToHost));
      std::vector<float> sch(NE, 1.f); if (fmt == FMT_NVFP4) CK(hipMemcpy(sch.data(), w.sgu[0], NE * 4, hipMemcpyDeviceToHost));
      std::vector<double> r32(nt * 8 * FF), rq(nt * 8 * FF);
      for (int t = 0; t < nt; t++) { std::vector<double> xr(2304), xq(2304); quant_act_ref(&xh[t * 2304], 2304, xq.data()); for (int i = 0; i < 2304; i++) xr[i] = xh[t * 2304 + i];
        pfor(8 * FF, [&](int idx) { int s = idx / FF, j = idx % FF; int e = ids[t * 8 + s]; std::vector<float> wg(2304), wu(2304);
          dequant_row(fmt, Wh.data() + e * gub + rb2 * j, 2304, wg.data()); dequant_row(fmt, Wh.data() + e * gub + rb2 * (FF + j), 2304, wu.data());
          for (int m = 0; m < 2; m++) { const double* xx = m ? xq.data() : xr.data(); double gv = dotd(wg.data(), xx, 2304) * sch[e], uv = dotd(wu.data(), xx, 2304) * sch[e];
            double a = 4 * tanh(gv / 4) * (1 / (1 + exp(-gv))) * 25 * tanh(uv / 25); (m ? rq : r32)[(size_t)(t * 8 + s) * FF + j] = a; } }); }
      a_ref_f.resize(r32.size()); for (size_t i = 0; i < r32.size(); i++) a_ref_f[i] = (float)r32[i];
      std::vector<float> ah(r32.size()); Err w32{}, wq{}; int nfail = 0;
      for (auto& v : gus) { CK(hipMemset(ad, 0, ah.size() * 4)); v.fn(w.gu[0], fmt == FMT_NVFP4 ? w.sgu[0] : nullptr, xd, idd, ad, nt, 0); CK(hipDeviceSynchronize());
        CK(hipMemcpy(ah.data(), ad, ah.size() * 4, hipMemcpyDeviceToHost)); Err e32 = cmp(ah.data(), r32.data(), ah.size()), eq = cmp(ah.data(), rq.data(), ah.size());
        if (eq.rms_rel > 1e-4) { nfail++; printf("VALFAIL moe_gu_%s %s q8ref max/rms=%.3e\n", tag, v.name.c_str(), eq.rms_rel); }
        w32.rms_rel = std::max(w32.rms_rel, e32.rms_rel); w32.max_rel = std::max(w32.max_rel, e32.max_rel); wq.rms_rel = std::max(wq.rms_rel, eq.rms_rel); wq.max_rel = std::max(wq.max_rel, eq.max_rel); }
      printf("VAL moe_gu_%s  vs-q8act-ref: maxerr/rms=%.2e maxrel=%.2e | vs-fp32act-ref: maxerr/rms=%.2e maxrel=%.2e | failed=%d/%zu\n", tag, wq.rms_rel, wq.max_rel, w32.rms_rel, w32.max_rel, nfail, gus.size());
    }
    if (do_val && (only == "moe" || match(std::string(tag) + " down", only))) {
      int nt = 2; if (a_ref_f.empty()) { a_ref_f.resize(nt * 8 * FF); std::mt19937 r3(9); std::normal_distribution<float> nd(0, 0.5f); for (auto& v : a_ref_f) v = nd(r3); }
      CK(hipMemcpy(ad, a_ref_f.data(), a_ref_f.size() * 4, hipMemcpyHostToDevice));
      std::vector<uint8_t> Wh(dnb * NE); CK(hipMemcpy(Wh.data(), w.dn[0], Wh.size(), hipMemcpyDeviceToHost));
      std::vector<float> sch(NE, 1.f); if (fmt == FMT_NVFP4) CK(hipMemcpy(sch.data(), w.sdn[0], NE * 4, hipMemcpyDeviceToHost));
      std::vector<double> r32(nt * D), rq(nt * D);
      for (int t = 0; t < nt; t++) { std::vector<std::vector<double>> xr(8, std::vector<double>(FF)), xq(8, std::vector<double>(FF));
        for (int s = 0; s < 8; s++) { quant_act_ref(&a_ref_f[(t * 8 + s) * FF], FF, xq[s].data()); for (int i = 0; i < FF; i++) xr[s][i] = a_ref_f[(t * 8 + s) * FF + i]; }
        pfor(D, [&](int r) { double s32 = 0, sq = 0; std::vector<float> wr(FF); for (int s = 0; s < 8; s++) { int e = ids[t * 8 + s]; dequant_row(fmt, Wh.data() + (size_t)(e * D + r) * rb1, FF, wr.data());
            double ww = (double)wts[t * 8 + s] * sch[e]; s32 += ww * dotd(wr.data(), xr[s].data(), FF); sq += ww * dotd(wr.data(), xq[s].data(), FF); }
          r32[t * D + r] = s32 + rh[t * D + r]; rq[t * D + r] = sq + rh[t * D + r]; }); }
      std::vector<float> yh(nt * D); Err w32{}, wq{}; int nfail = 0;
      for (auto& v : dns) { CK(hipMemset(yd, 0, yh.size() * 4)); v.fn(w.dn[0], fmt == FMT_NVFP4 ? w.sdn[0] : nullptr, ad, idd, wtd, resd, yd, nt, 0); CK(hipDeviceSynchronize());
        CK(hipMemcpy(yh.data(), yd, yh.size() * 4, hipMemcpyDeviceToHost)); Err e32 = cmp(yh.data(), r32.data(), yh.size()), eq = cmp(yh.data(), rq.data(), yh.size());
        if (eq.rms_rel > 1e-4) { nfail++; printf("VALFAIL moe_down_%s %s q8ref max/rms=%.3e\n", tag, v.name.c_str(), eq.rms_rel); }
        w32.rms_rel = std::max(w32.rms_rel, e32.rms_rel); w32.max_rel = std::max(w32.max_rel, e32.max_rel); wq.rms_rel = std::max(wq.rms_rel, eq.rms_rel); wq.max_rel = std::max(wq.max_rel, eq.max_rel); }
      printf("VAL moe_down_%s  vs-q8act-ref: maxerr/rms=%.2e maxrel=%.2e | vs-fp32act-ref: maxerr/rms=%.2e maxrel=%.2e | failed=%d/%zu\n", tag, wq.rms_rel, wq.max_rel, w32.rms_rel, w32.max_rel, nfail, dns.size());
    }
    // ---- perf ----
    if (do_perf) {
      int NLW = w.gu.size();
      // fill ad with realistic a (run one gu)
      for (int bi = 0; bi < 3; bi++) { int B = Bs[bi];
        double uniq = 0; for (int i = 0; i < NID; i++) { std::vector<int> s(ids.begin() + i * 32, ids.begin() + i * 32 + B * 8); std::sort(s.begin(), s.end()); uniq += std::unique(s.begin(), s.end()) - s.begin(); } uniq /= NID;
        if ((only == "moe" || match(std::string(tag) + " gu", only))) { double bytes = uniq * (gub + (fmt == FMT_NVFP4 ? 4 : 0)); double best = 1e30; std::string bn;
          for (auto& v : gus) { double us = time_us([&](int it) { int l = it % NLW; v.fn(w.gu[l], fmt == FMT_NVFP4 ? w.sgu[l] : nullptr, xd, idd + (it % NID) * 32, ad, B, 0); });
            printf("PERF moe_gu_%s B=%d %-16s %9.2f us %7.1f GB/s\n", tag, B, v.name.c_str(), us, bytes / us * 1e-3); if (us < best) { best = us; bn = v.name; } }
          printf("BEST moe_gu_%s B=%d %-16s %9.2f us %7.1f GB/s (%.1f MB read, uniq experts %.2f)\n", tag, B, bn.c_str(), best, bytes / best * 1e-3, bytes / 1e6, uniq); }
        if ((only == "moe" || match(std::string(tag) + " down", only))) { double bytes = uniq * (dnb + (fmt == FMT_NVFP4 ? 4 : 0)); double best = 1e30; std::string bn;
          for (auto& v : dns) { double us = time_us([&](int it) { int l = it % NLW; v.fn(w.dn[l], fmt == FMT_NVFP4 ? w.sdn[l] : nullptr, ad, idd + (it % NID) * 32, wtd + (it % NID) * 32, resd, yd, B, 0); });
            printf("PERF moe_down_%s B=%d %-16s %9.2f us %7.1f GB/s\n", tag, B, v.name.c_str(), us, bytes / us * 1e-3); if (us < best) { best = us; bn = v.name; } }
          printf("BEST moe_down_%s B=%d %-16s %9.2f us %7.1f GB/s (%.1f MB read, uniq experts %.2f)\n", tag, B, bn.c_str(), best, bytes / best * 1e-3, bytes / 1e6, uniq); }
      }
    }
    hipFree(idd); hipFree(wtd); hipFree(xd); hipFree(ad); hipFree(yd); hipFree(resd);
  };
  run_moe("nvfp4", FMT_NVFP4, nv);
  run_moe("q8", FMT_Q8, q8);
  return 0;
}
