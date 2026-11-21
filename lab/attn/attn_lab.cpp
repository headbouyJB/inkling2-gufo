// Standalone lab: correctness (double CPU reference) + benchmarks for the Inkling 2 Mini attention kernels. MIT.
// usage: attn_lab [test|bench|all|tune]   (run through $GPU_RUN)
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>
#include "attn_kernels.h"
using namespace ia;

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1);} } while (0)
static void pfor(int n, const std::function<void(int)>& f) {
  int nt = std::min<int>(std::thread::hardware_concurrency(), 32); std::vector<std::thread> th;
  for (int t = 0; t < nt; t++) th.emplace_back([&, t] { for (int i = t; i < n; i += nt) f(i); });
  for (auto& x : th) x.join();
}
// ---------------- rng
struct Rng { uint64_t s; explicit Rng(uint64_t x) : s(x * 0x9E3779B97F4A7C15ull + 88172645463325252ull) {}
  uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
  float u() { return (next() >> 40) * (1.0f / 16777216.f); }
  float n() { return (u() + u() + u() + u() - 2.f) * 1.7320508f; } };
static const double QK_THR = 22.627, R_THR = 8.0;

// ---------------- cache on host (f16 bits) with realistic rows
struct HostCache {
  int cap = 0; std::vector<_Float16> k, v;   // [cap][8][128]
  void init(int c) { cap = c; k.assign((size_t)c * CELL, (_Float16)0); v.assign((size_t)c * CELL, (_Float16)0); }
  // fill positions [pa, pb] with clipped K / RMS-normed V (same math as kv_write, but double precision)
  void fill(int pa, int pb, uint64_t seed) {
    pfor(pb - pa + 1, [&](int i) {
      int pos = pa + i; Rng r(seed * 1000003ull + pos); size_t cell = (size_t)(pos % cap);
      for (int h = 0; h < NKV; h++) {
        double kx[HD], vx[HD], ks = 0, vs = 0;
        float sk = 1.5f * expf(0.35f * r.n()), sv = expf(0.5f * r.n());
        for (int d = 0; d < HD; d++) { kx[d] = r.n() * sk; vx[d] = r.n() * sv; ks += kx[d] * kx[d]; vs += vx[d] * vx[d]; }
        double kf = std::min(1.0, QK_THR / (sqrt(ks) + 1e-6)), vf = 1.0 / sqrt(vs / HD + 1e-6);
        for (int d = 0; d < HD; d++) { k[cell * CELL + h * HD + d] = (_Float16)(float)(kx[d] * kf); v[cell * CELL + h * HD + d] = (_Float16)(float)(vx[d] * vf); }
      }
    });
  }
};
struct Params {  // per-layer weights (random but realistic) + table
  std::vector<float> qscale, rclip, B;   // [24][128], [24][16], [16][1024]
  std::vector<_Float16> bt;              // [1024][16] f16 transposed
  void init(uint64_t seed) {
    Rng r(seed); qscale.resize(NH * HD); rclip.resize(NH * RD); B.resize(RD * WIN); bt.resize(WIN * RD);
    for (auto& x : qscale) x = 1.0f + 0.25f * r.n();
    for (auto& x : rclip) x = 1.0f + 0.25f * r.n();
    for (auto& x : B) x = 0.12f * r.n();
    for (int c = 0; c < RD; c++) for (int d = 0; d < WIN; d++) bt[d * RD + c] = (_Float16)B[c * WIN + d];
  }
};
struct DevBuf { void* p = nullptr; size_t n = 0; ~DevBuf() { if (p) hipFree(p); }
  void alloc(size_t bytes) { if (p) hipFree(p); CK(hipMalloc(&p, bytes)); n = bytes; }
  template <class T> void up(const std::vector<T>& v) { alloc(v.size() * sizeof(T)); CK(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice)); } };

// ---------------- CPU reference for one (position, head) row, double. rawQ[128], rawR[16]
static void ref_row(const HostCache& c, bool swa, int pos, const float* rawQ, const float* rawR, int h, const Params& P, float sink,
                    const float* gate /*[128] or null*/, double* out) {
  double q[HD], rr[RD], qs = 0, rs = 0;
  for (int d = 0; d < HD; d++) { q[d] = (double)rawQ[d] * P.qscale[h * HD + d]; qs += q[d] * q[d]; }
  for (int d = 0; d < RD; d++) { rr[d] = (double)rawR[d] * P.rclip[h * RD + d]; rs += rr[d] * rr[d]; }
  double tau = 1.0;
  if (!swa) tau = 1.0 + 0.1 * log(std::max(1.0, (double)(pos + 1) / 8192.0));
  double qf = std::min(1.0, QK_THR / (sqrt(qs) + 1e-6)) * tau, rf = std::min(1.0, R_THR / (sqrt(rs) + 1e-6)) * tau;
  for (int d = 0; d < HD; d++) q[d] *= qf;
  for (int d = 0; d < RD; d++) rr[d] *= rf;
  const int kv = h / GQ, j0 = swa ? std::max(0, pos - (WIN - 1)) : 0;
  std::vector<double> s(pos - j0 + 1);
  double mx = sink;
  for (int j = j0; j <= pos; j++) {
    const _Float16* kp = &c.k[(size_t)(j % c.cap) * CELL + kv * HD];
    double a = 0; for (int d = 0; d < HD; d++) a += q[d] * (double)(float)kp[d];
    a /= sqrt(128.0);
    int dist = pos - j;
    if (dist < WIN) { double b = 0; for (int cc = 0; cc < RD; cc++) b += rr[cc] * (double)P.B[cc * WIN + dist]; a += b; }
    s[j - j0] = a; mx = std::max(mx, a);
  }
  double Z = exp(sink - mx);
  for (auto& x : s) { x = exp(x - mx); Z += x; }
  double o[HD] = {0};
  for (int j = j0; j <= pos; j++) {
    const _Float16* vp = &c.v[(size_t)(j % c.cap) * CELL + kv * HD];
    double p = s[j - j0];
    for (int d = 0; d < HD; d++) o[d] += p * (double)(float)vp[d];
  }
  for (int d = 0; d < HD; d++) { o[d] /= Z; if (gate) o[d] *= 1.0 / (1.0 + exp(-(double)gate[d])); out[d] = o[d]; }
}

// ---------------- device-side contexts
struct Dev {
  DevBuf kc, vc, qscale, rclip, bt, ws_o, ws_m, ws_l, counters;
  int cap = 0;
  void setup_params(const Params& P) { qscale.up(P.qscale); rclip.up(P.rclip); bt.up(P.bt);
    size_t ws = (size_t)1024 * NKV * 12; ws_o.alloc(ws * HD * 4); ws_m.alloc(ws * 4); ws_l.alloc(ws * 4); counters.alloc(NKV * 4); CK(hipMemset(counters.p, 0, NKV * 4)); }
  void setup_cache(const HostCache& c) { cap = c.cap; kc.up(c.k); vc.up(c.v); }
};
static int g_chunk_override = 0; static bool g_nofuse = true;   // fused last-block reduction measured slower (35 us vs 27 us SWA): off by default
static int pick_chunk(int nkeys, int nq = 1) {
  if (g_chunk_override) return g_chunk_override;
  int ch = nkeys / 32; ch = std::min(std::max(ch, nq >= 3 ? 256 : 128), 2048); ch = (ch + 63) / 64 * 64; return ch;   // ~32 splits, 2048 keys max per split
}
// launches decode main kernel (+ reduce). returns nsplit.
template <int NQ> static void launch_dec_nq(const DecArgs& a, int nsplit, hipStream_t s) {
  hipLaunchKernelGGL((attn_decode<NQ, 4>), dim3(NKV, nsplit), dim3(128), 0, s, a);
}
static int run_decode(Dev& d, bool swa, int p0, int nq, const float* q, const float* r, const float* sink, const float* gate, float* out,
                      hipStream_t s, int chunk_ov = 0, bool do_reduce = true) {
  DecArgs a{}; a.kc = (const __half*)d.kc.p; a.vc = (const __half*)d.vc.p; a.cap = d.cap; a.swa = swa; a.nq = nq; a.p0 = p0;
  a.q = q; a.r = r; a.qscale = (const float*)d.qscale.p; a.rclip = (const float*)d.rclip.p; a.bt = (const __half*)d.bt.p;
  a.qk_thr = (float)QK_THR; a.r_thr = (float)R_THR;
  int pmax = p0 + nq - 1; a.kb = swa ? std::max(0, p0 - (WIN - 1)) : 0;
  int nkeys = pmax - a.kb + 1; int chunk = chunk_ov ? chunk_ov : pick_chunk(nkeys, nq);
  int nsplit = (nkeys + chunk - 1) / chunk; if (nsplit > 1024) { fprintf(stderr, "nsplit too big\n"); exit(1); }
  a.nsplit = nsplit; a.fuse = (nsplit <= 16 && do_reduce && !g_nofuse) ? 1 : 0; a.counters = (unsigned*)d.counters.p; a.sink = sink; a.gate = gate; a.out = out;
  a.chunk = chunk; a.ws_o = (float*)d.ws_o.p; a.ws_m = (float*)d.ws_m.p; a.ws_l = (float*)d.ws_l.p;
  switch (nq) { case 1: launch_dec_nq<1>(a, nsplit, s); break; case 2: launch_dec_nq<2>(a, nsplit, s); break;
    case 3: launch_dec_nq<3>(a, nsplit, s); break; default: launch_dec_nq<4>(a, nsplit, s); }
  if (do_reduce && !a.fuse) {
    ReduceArgs ra{a.ws_o, a.ws_m, a.ws_l, nsplit, sink, gate, out};
    hipLaunchKernelGGL(decode_reduce, dim3(nq * NH), dim3(128), 0, s, ra);
  }
  return nsplit;
}
static void run_prefill(Dev& d, bool swa, int p0, int U, const float* q, const float* r, const float* sink, const float* gate, float* out,
                        __half* qn, float* rn, hipStream_t s, bool prep = true, int NTT = 2) {
  if (prep) {
    PrepArgs pa{q, r, (const float*)d.qscale.p, (const float*)d.rclip.p, U, p0, (int)swa, (float)QK_THR, (float)R_THR, qn, rn};
    hipLaunchKernelGGL(qr_prep, dim3((U * NH + 3) / 4), dim3(128), 0, s, pa);
  }
  PfArgs a{}; a.kc = (const __half*)d.kc.p; a.vc = (const __half*)d.vc.p; a.cap = d.cap; a.swa = swa; a.U = U; a.p0 = p0;
  a.qn = qn; a.rn = rn; a.sink = sink; a.gate = gate; a.bt = (const __half*)d.bt.p; a.out = out;
  if (NTT == 2) hipLaunchKernelGGL((attn_prefill<2>), dim3((U + 31) / 32, NKV), dim3(192), 0, s, a);
  else if (NTT == 1) hipLaunchKernelGGL((attn_prefill<1>), dim3((U + 15) / 16, NKV), dim3(96), 0, s, a);
  else if (NTT == 3) hipLaunchKernelGGL((attn_prefill<3>), dim3((U + 47) / 48, NKV), dim3(288), 0, s, a);
  else if (NTT == 6) hipLaunchKernelGGL((attn_prefill<6>), dim3((U + 95) / 96, NKV), dim3(576), 0, s, a);
  else if (NTT == 8) hipLaunchKernelGGL((attn_prefill<8>), dim3((U + 127) / 128, NKV), dim3(768), 0, s, a);
  else hipLaunchKernelGGL((attn_prefill<4>), dim3((U + 63) / 64, NKV), dim3(384), 0, s, a);
}

// ---------------- error metrics
struct Err { double maxabs = 0, scale = 0, rms = 0, rmsref = 0; int bad = 0; };
static void accum(Err& e, const double* ref, const float* got, int n) {
  for (int i = 0; i < n; i++) {
    double dlt = fabs(ref[i] - got[i]); if (!(dlt == dlt)) { e.bad++; dlt = 1e9; }
    e.maxabs = std::max(e.maxabs, dlt); e.scale = std::max(e.scale, fabs(ref[i])); e.rms += dlt * dlt; e.rmsref += ref[i] * ref[i];
  }
}
static int g_fail = 0;
static void report(const char* name, const Err& e, double tol_rel = 4e-3) {
  double rel = e.maxabs / std::max(e.scale, 1e-6), relrms = sqrt(e.rms / std::max(e.rmsref, 1e-30));
  bool ok = rel < tol_rel && e.bad == 0;
  printf("  %-52s max|err|=%.3e  scale=%.3e  max/scale=%.2e  rms_rel=%.2e  nan=%d  %s\n", name, e.maxabs, e.scale, rel, relrms, e.bad, ok ? "PASS" : "FAIL");
  if (!ok) g_fail++;
}

static void make_inputs(Rng& r, int n, std::vector<float>& q, std::vector<float>& rr, std::vector<float>& sink, std::vector<float>& gate, float sinkmode) {
  q.resize((size_t)n * NH * HD); rr.resize((size_t)n * NH * RD); sink.resize((size_t)n * NH); gate.resize((size_t)n * NH * HD);
  for (int i = 0; i < n * NH; i++) {
    float sc = 2.4f * expf(0.3f * r.n());
    for (int d = 0; d < HD; d++) q[(size_t)i * HD + d] = r.n() * sc;
    float rsc = 1.6f * expf(0.3f * r.n());
    for (int d = 0; d < RD; d++) rr[(size_t)i * RD + d] = r.n() * rsc;
    sink[i] = (sinkmode == 0.f) ? 1.5f * r.n() : sinkmode;
  }
  for (auto& g : gate) g = 1.5f * r.n();
}

// ---------------- tests
static HostCache g_big;      // global cache, cap = CAPG
static const int CAPG = 200016;
static Dev g_devbig;
static Params g_P;

static void test_kv_write() {
  printf("[kv_write] vs double reference (clip K, RMS V)\n");
  int n = 37, cap = 50, pos0 = 30;   // wraps the ring (cells 30..49, 0..16)
  Rng r(5); std::vector<float> k(n * CELL), v(n * CELL);
  for (int i = 0; i < n * NKV; i++) { float sk = 2.0f * expf(0.5f * r.n()), sv = expf(0.7f * r.n()); for (int d = 0; d < HD; d++) { k[i * HD + d] = r.n() * sk; v[i * HD + d] = r.n() * sv; } }
  DevBuf dk, dv, kc, vc; dk.up(k); dv.up(v); kc.alloc((size_t)cap * CELL * 2); vc.alloc((size_t)cap * CELL * 2);
  CK(hipMemset(kc.p, 0, (size_t)cap * CELL * 2)); CK(hipMemset(vc.p, 0, (size_t)cap * CELL * 2));
  KvWriteArgs a{(const float*)dk.p, (const float*)dv.p, n, pos0, cap, (__half*)kc.p, (__half*)vc.p, (float)QK_THR, 1e-6f};
  hipLaunchKernelGGL(kv_write, dim3((n * NKV + 3) / 4), dim3(128), 0, 0, a); CK(hipDeviceSynchronize());
  std::vector<_Float16> hk((size_t)cap * CELL), hv((size_t)cap * CELL);
  CK(hipMemcpy(hk.data(), kc.p, hk.size() * 2, hipMemcpyDeviceToHost)); CK(hipMemcpy(hv.data(), vc.p, hv.size() * 2, hipMemcpyDeviceToHost));
  Err ek, ev; int clipped = 0;
  for (int t = 0; t < n; t++) for (int h = 0; h < NKV; h++) {
    double ks = 0, vs = 0; for (int d = 0; d < HD; d++) { double x = k[(t * NKV + h) * HD + d], y = v[(t * NKV + h) * HD + d]; ks += x * x; vs += y * y; }
    double kf = std::min(1.0, QK_THR / (sqrt(ks) + 1e-6)), vf = 1.0 / sqrt(vs / HD + 1e-6); if (kf < 1) clipped++;
    int cell = (pos0 + t) % cap; double rk[HD], rv[HD]; float gk[HD], gv[HD];
    for (int d = 0; d < HD; d++) { rk[d] = k[(t * NKV + h) * HD + d] * kf; rv[d] = v[(t * NKV + h) * HD + d] * vf; gk[d] = (float)hk[(size_t)cell * CELL + h * HD + d]; gv[d] = (float)hv[(size_t)cell * CELL + h * HD + d]; }
    accum(ek, rk, gk, HD); accum(ev, rv, gv, HD);
  }
  printf("  (%d of %d K rows hit the clip)\n", clipped, n * NKV);
  report("kv_write K (f16 rounding only)", ek, 1e-3); report("kv_write V (f16 rounding only)", ev, 1e-3);
}

struct DecCase { const char* name; bool swa; int p0; int nq; float sinkmode; bool gate; int chunk; };
static void test_decode_case(const DecCase& c, Dev& dev, const HostCache& hc) {
  Rng r(1234 + c.p0 * 7 + c.nq); std::vector<float> q, rr, sink, gate; make_inputs(r, c.nq, q, rr, sink, gate, c.sinkmode);
  DevBuf dq, dr, ds, dg, dout; dq.up(q); dr.up(rr); ds.up(sink); dg.up(gate); dout.alloc((size_t)c.nq * NH * HD * 4);
  run_decode(dev, c.swa, c.p0, c.nq, (const float*)dq.p, (const float*)dr.p, (const float*)ds.p, c.gate ? (const float*)dg.p : nullptr, (float*)dout.p, 0, c.chunk);
  CK(hipDeviceSynchronize());
  std::vector<float> out((size_t)c.nq * NH * HD); CK(hipMemcpy(out.data(), dout.p, out.size() * 4, hipMemcpyDeviceToHost));
  Err e; std::vector<double> ref((size_t)c.nq * NH * HD);
  pfor(c.nq * NH, [&](int i) { int t = i / NH, h = i % NH;
    ref_row(hc, c.swa, c.p0 + t, &q[(size_t)i * HD], &rr[(size_t)i * RD], h, g_P, sink[i], c.gate ? &gate[(size_t)i * HD] : nullptr, &ref[(size_t)i * HD]); });
  accum(e, ref.data(), out.data(), (int)ref.size());
  char nm[128]; snprintf(nm, sizeof nm, "%s %s p0=%d nq=%d", c.swa ? "SWA " : "glob", c.name, c.p0, c.nq); report(nm, e);
}

static HostCache g_ring; static Dev g_devring;
static void ring_prepare(int p0, int nq, int cap) {   // ring cache holding the window positions ending at p0+nq-1
  g_ring.init(cap); int pmax = p0 + nq - 1; int pa = std::max(0, pmax - cap + 1);
  g_ring.fill(pa, pmax, 99);   // same seed => same row for same position
  g_devring.cap = cap; g_devring.kc.up(g_ring.k); g_devring.vc.up(g_ring.v);
}
static void test_decode() {
  printf("[decode] global layer, cap=%d, cache 200K rows (seed-deterministic)\n", CAPG);
  std::vector<DecCase> gl = {
    {"ctx=1", false, 0, 1, 0.f, true, 0}, {"ctx=2", false, 1, 1, 0.f, false, 0}, {"d=1023 edge", false, 1023, 1, 0.f, true, 0}, {"d=1024 edge", false, 1024, 1, 0.f, true, 0},
    {"ctx=1025", false, 1024, 1, 0.f, true, 0}, {"ctx=1026", false, 1025, 1, 0.f, false, 0}, {"pos=8191 tau=1", false, 8191, 1, 0.f, true, 0}, {"pos=8192 tau~1", false, 8192, 1, 0.f, true, 0},
    {"pos=20000 tau>1", false, 20000, 1, 0.f, true, 0}, {"ctx=33K", false, 32999, 1, 0.f, true, 0}, {"ctx=200K", false, 199999, 1, 0.f, true, 0},
    {"ctx=200K chunk=448", false, 199999, 1, 0.f, false, 448}, {"nq=2 crossing 8192", false, 8191, 2, 0.f, true, 0}, {"nq=3", false, 5000, 3, 0.f, true, 0}, {"nq=4 ctx~33K", false, 33000, 4, 0.f, true, 0},
    {"nq=4 ctx=200K", false, 199990, 4, 0.f, true, 0}, {"nq=4 ctx=1..4 (causal in batch)", false, 0, 4, 0.f, true, 0},
    {"sink +40 (dominant)", false, 33000, 1, 40.f, true, 0}, {"sink -40 (negligible)", false, 33000, 1, -40.f, true, 0}, {"sink +40 ctx1", false, 0, 1, 40.f, false, 0}};
  for (auto& c : gl) test_decode_case(c, g_devbig, g_big);
  printf("[decode] SWA ring layer (window 1024, cap=1280 or 1536)\n");
  struct SC { const char* name; int p0; int nq; int cap; float sm; };
  std::vector<SC> sl = {{"pos=0", 0, 1, 1280, 0.f}, {"pos=1022", 1022, 1, 1280, 0.f}, {"pos=1023 (d=1023 oldest)", 1023, 1, 1280, 0.f}, {"pos=1024 (key0 drops)", 1024, 1, 1280, 0.f},
    {"pos=1025", 1025, 1, 1280, 0.f}, {"pos=5000 ring wrapped", 5000, 1, 1280, 0.f}, {"pos=33000 ring wrapped", 33000, 1, 1280, 0.f}, {"nq=4 pos=5000", 5000, 4, 1280, 0.f},
    {"nq=4 pos=1022 crosses edge", 1022, 4, 1280, 0.f}, {"nq=2 pos=300", 300, 2, 1280, 0.f}, {"sink +40", 5000, 1, 1280, 40.f}, {"cap=1536 pos=7777", 7777, 1, 1536, 0.f}};
  for (auto& s : sl) { ring_prepare(s.p0, s.nq, s.cap); DecCase c{s.name, true, s.p0, s.nq, s.sm, true, 0}; test_decode_case(c, g_devring, g_ring); }
  g_nofuse = true; { ring_prepare(5000, 4, 1280); DecCase c{"nq=4 pos=5000 (separate reduce kernel)", true, 5000, 4, 0.f, true, 0}; test_decode_case(c, g_devring, g_ring); } g_nofuse = false;
  { DecCase c{"ctx=33K fused forced 16 splits", false, 32999, 1, 0.f, true, 2048}; test_decode_case(c, g_devbig, g_big); }
}

struct PfCase { const char* name; bool swa; int p0; int U; bool gate; int NTT; };
static void test_prefill_case(const PfCase& c, Dev& dev, const HostCache& hc) {
  Rng r(777 + c.p0 + c.U); std::vector<float> q, rr, sink, gate; make_inputs(r, c.U, q, rr, sink, gate, 0.f);
  DevBuf dq, dr, ds, dg, dout, dqn, drn; dq.up(q); dr.up(rr); ds.up(sink); dg.up(gate); dout.alloc((size_t)c.U * NH * HD * 4);
  dqn.alloc((size_t)c.U * NH * HD * 2); drn.alloc((size_t)c.U * NH * RD * 4);
  run_prefill(dev, c.swa, c.p0, c.U, (const float*)dq.p, (const float*)dr.p, (const float*)ds.p, c.gate ? (const float*)dg.p : nullptr, (float*)dout.p,
              (__half*)dqn.p, (float*)drn.p, 0, true, c.NTT);
  CK(hipDeviceSynchronize());
  std::vector<float> out((size_t)c.U * NH * HD); CK(hipMemcpy(out.data(), dout.p, out.size() * 4, hipMemcpyDeviceToHost));
  // sampled query rows: edges, positions around 1023/1024 distance, random
  std::vector<int> ts = {0, 1, 2, 15, 16, 17, c.U / 2, c.U - 2, c.U - 1}; Rng rs(c.U * 31 + c.p0);
  for (int i = 0; i < 12; i++) ts.push_back((int)(rs.next() % c.U));
  std::sort(ts.begin(), ts.end()); ts.erase(std::unique(ts.begin(), ts.end()), ts.end());
  std::vector<std::pair<int, int>> rows; for (int t : ts) if (t >= 0 && t < c.U) for (int h = 0; h < NH; h++) rows.push_back({t, h});
  std::vector<double> ref(rows.size() * HD); std::vector<float> got(rows.size() * HD);
  pfor((int)rows.size(), [&](int i) { int t = rows[i].first, h = rows[i].second; size_t ix = (size_t)t * NH + h;
    ref_row(hc, c.swa, c.p0 + t, &q[ix * HD], &rr[ix * RD], h, g_P, sink[ix], c.gate ? &gate[ix * HD] : nullptr, &ref[(size_t)i * HD]);
    memcpy(&got[(size_t)i * HD], &out[ix * HD], HD * 4); });
  Err e; accum(e, ref.data(), got.data(), (int)ref.size());
  // any NaN anywhere in the output?
  int nn = 0; for (float x : out) if (!(x == x)) nn++; e.bad += nn;
  char nm[128]; snprintf(nm, sizeof nm, "%s %s p0=%d U=%d NTT=%d", c.swa ? "SWA " : "glob", c.name, c.p0, c.U, c.NTT); report(nm, e);
}
static void test_prefill() {
  printf("[prefill] global layer (sampled rows vs double reference; NaN scan over all rows)\n");
  std::vector<PfCase> gl = {{"prefix 0", false, 0, 512, true, 2}, {"prefix 0 NTT=1", false, 0, 512, false, 1}, {"prefix 0 NTT=4", false, 0, 512, false, 4}, {"prefix 4K", false, 4096, 512, true, 2},
    {"prefix 8000 crosses tau", false, 8000, 512, true, 2}, {"prefix 1000 crosses d=1024", false, 1000, 512, false, 2}, {"prefix 32K", false, 32768, 512, true, 2},
    {"U=100 odd", false, 77, 100, true, 2}, {"U=1", false, 5, 1, false, 2}, {"U=2048 prefix 0", false, 0, 2048, true, 2}, {"prefix 200K-2K", false, 197000, 512, false, 2}};
  for (auto& c : gl) test_prefill_case(c, g_devbig, g_big);
  printf("[prefill] SWA ring layer cap=1536 (window 1024 + ubatch 512)\n");
  struct SC { const char* name; int p0; int U; int cap; };
  std::vector<SC> sl = {{"prefix 0", 0, 512, 1536}, {"prefix 4K", 4096, 512, 1536}, {"prefix 32K", 32768, 512, 1536}, {"prefix 1000 (window edge)", 1000, 512, 1536}, {"prefix 600", 600, 512, 1536},
    {"U=1 prefix 3000", 3000, 1, 1536}, {"U=100 prefix 1300", 1300, 100, 1536}, {"U=1024 cap=2048", 5000, 1024, 2048}, {"U=2048 cap=3072", 40000, 2048, 3072}};
  for (auto& s : sl) {
    // ring contains positions [p0 + U - cap .. p0 + U - 1]; window needs [p0-1023 .. p0+U-1]
    int pmax = s.p0 + s.U - 1; g_ring.init(s.cap); g_ring.fill(std::max(0, pmax - s.cap + 1), pmax, 99);
    g_devring.cap = s.cap; g_devring.kc.up(g_ring.k); g_devring.vc.up(g_ring.v);
    PfCase c{s.name, true, s.p0, s.U, true, 2}; test_prefill_case(c, g_devring, g_ring);
  }
}

// ---------------- benchmarks
__global__ void fill_half(__half* p, size_t n, uint32_t seed) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  uint32_t x = (uint32_t)i * 2654435761u + seed; x ^= x >> 15; x *= 2246822519u; x ^= x >> 13;
  p[i] = __float2half(((x & 0xffff) * (1.0f / 32768.f) - 1.0f) * 0.5f);
}
__global__ void fill_f32(float* p, size_t n, uint32_t seed, float sc) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  uint32_t x = (uint32_t)i * 2654435761u + seed; x ^= x >> 15; x *= 2246822519u; x ^= x >> 13;
  p[i] = ((x & 0xffff) * (1.0f / 32768.f) - 1.0f) * sc;
}
static double now_ms() { return 0; }
template <class F> static float time_it(F f, int warm, int iters) {
  hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
  for (int i = 0; i < warm; i++) f(i); CK(hipDeviceSynchronize());
  CK(hipEventRecord(a, 0)); for (int i = 0; i < iters; i++) f(i); CK(hipEventRecord(b, 0)); CK(hipEventSynchronize(b));
  float ms; CK(hipEventElapsedTime(&ms, a, b)); return ms / iters;
}
struct BenchCtx { std::vector<Dev> dv; int ncopies; };
static void bench_decode_one(const char* label, bool swa, int ctx, int nq, int cap, int chunk_ov, bool tune_print, int force_copies = 0) {
  size_t cache_bytes = (size_t)cap * CELL * 2 * 2; int ncopies = (int)std::max<size_t>(1, std::min<size_t>(64, (768ull << 20) / cache_bytes));
  if (force_copies) ncopies = force_copies;
  std::vector<Dev> dv(ncopies); Params P; P.init(3);
  for (int i = 0; i < ncopies; i++) {
    dv[i].setup_params(P); dv[i].cap = cap; dv[i].kc.alloc((size_t)cap * CELL * 2); dv[i].vc.alloc((size_t)cap * CELL * 2);
    size_t n = (size_t)cap * CELL; fill_half<<<(n + 255) / 256, 256>>>((__half*)dv[i].kc.p, n, 1 + i); fill_half<<<(n + 255) / 256, 256>>>((__half*)dv[i].vc.p, n, 77 + i);
  }
  DevBuf dq, dr, ds, dout; dq.alloc(nq * NH * HD * 4); dr.alloc(nq * NH * RD * 4); ds.alloc(nq * NH * 4); dout.alloc(nq * NH * HD * 4);
  fill_f32<<<(nq * NH * HD + 255) / 256, 256>>>((float*)dq.p, nq * NH * HD, 5, 3.0f); fill_f32<<<(nq * NH * RD + 255) / 256, 256>>>((float*)dr.p, nq * NH * RD, 6, 2.0f);
  fill_f32<<<1, 128>>>((float*)ds.p, nq * NH, 7, 1.0f); CK(hipDeviceSynchronize());
  int p0 = ctx - nq; int pmax = p0 + nq - 1; int kb = swa ? std::max(0, p0 - (WIN - 1)) : 0; int nkeys = pmax - kb + 1;
  int nsp = 0;
  float tmain = time_it([&](int i) { nsp = run_decode(dv[i % ncopies], swa, p0, nq, (const float*)dq.p, (const float*)dr.p, (const float*)ds.p, nullptr, (float*)dout.p, 0, chunk_ov, false); }, 3, 20);
  float tall = time_it([&](int i) { run_decode(dv[i % ncopies], swa, p0, nq, (const float*)dq.p, (const float*)dr.p, (const float*)ds.p, nullptr, (float*)dout.p, 0, chunk_ov, true); }, 3, 20);
  double bytes = (double)nkeys * NKV * 512.0;   // K+V f16 for all 8 kv heads
  printf("  %-26s nq=%d keys=%-8d splits=%-4d  main %8.1f us  main+reduce %8.1f us  KV-read %6.1f GB/s (on main+reduce)  copies=%d\n", label, nq, nkeys, nsp, tmain * 1e3, tall * 1e3, bytes / (tall * 1e-3) / 1e9, ncopies);
  (void)tune_print;
}
static void bench_decode() {
  printf("[decode perf] per layer visit, KV f16 [cell][8][128]; GB/s = K+V bytes touched / (main+reduce) time; caches rotated (>=768 MB footprint) to defeat MALL\n");
  int ctxs[] = {1024, 32768, 131072, 200000};
  for (int nq : {1, 4}) {
    for (int ctx : ctxs) { char l[64]; snprintf(l, sizeof l, "global ctx=%d", ctx); bench_decode_one(l, false, ctx, nq, ctx + 16, 0, false); }
    bench_decode_one("global ctx=1M", false, 1048576, nq, 1048576 + 16, 0, false);
    bench_decode_one("SWA window 1024", true, 8000, nq, 1280, 0, false);
  }
}
static void tune_decode() {
  printf("[decode tuning] chunk sweep (keys per split)\n");
  for (int ctx : {32768, 200000}) for (int chunk : {64, 128, 256, 512, 1024, 2048, 4096}) { if (ctx / chunk > 1000) continue; char l[64]; snprintf(l, sizeof l, "glob ctx=%d chunk=%d", ctx, chunk); bench_decode_one(l, false, ctx, 1, ctx + 16, chunk, false); }
  for (int chunk : {64, 128, 256, 512, 1024}) { char l[64]; snprintf(l, sizeof l, "SWA chunk=%d", chunk); bench_decode_one(l, true, 8000, 1, 1280, chunk, false); }
  for (int chunk : {64, 128, 256, 512, 1024}) { char l[64]; snprintf(l, sizeof l, "SWA nq4 chunk=%d", chunk); bench_decode_one(l, true, 8000, 4, 1280, chunk, false); }
}
static void bench_prefill_one(const char* label, bool swa, int p0, int U, int cap, int NTT) {
  Dev dv; Params P; P.init(3); dv.setup_params(P); dv.cap = cap;
  dv.kc.alloc((size_t)cap * CELL * 2); dv.vc.alloc((size_t)cap * CELL * 2); size_t n = (size_t)cap * CELL;
  fill_half<<<(n + 255) / 256, 256>>>((__half*)dv.kc.p, n, 1); fill_half<<<(n + 255) / 256, 256>>>((__half*)dv.vc.p, n, 2);
  DevBuf dq, dr, ds, dout, dqn, drn; dq.alloc((size_t)U * NH * HD * 4); dr.alloc((size_t)U * NH * RD * 4); ds.alloc((size_t)U * NH * 4); dout.alloc((size_t)U * NH * HD * 4);
  dqn.alloc((size_t)U * NH * HD * 2); drn.alloc((size_t)U * NH * RD * 4);
  fill_f32<<<((size_t)U * NH * HD + 255) / 256, 256>>>((float*)dq.p, (size_t)U * NH * HD, 5, 3.0f); fill_f32<<<((size_t)U * NH * RD + 255) / 256, 256>>>((float*)dr.p, (size_t)U * NH * RD, 6, 2.0f);
  fill_f32<<<((size_t)U * NH + 255) / 256, 256>>>((float*)ds.p, (size_t)U * NH, 7, 1.0f); CK(hipDeviceSynchronize());
  float t = time_it([&](int) { run_prefill(dv, swa, p0, U, (const float*)dq.p, (const float*)dr.p, (const float*)ds.p, nullptr, (float*)dout.p, (__half*)dqn.p, (float*)drn.p, 0, true, NTT); }, 2, 8);
  float tp = time_it([&](int) { run_prefill(dv, swa, p0, U, (const float*)dq.p, (const float*)dr.p, (const float*)ds.p, nullptr, (float*)dout.p, (__half*)dqn.p, (float*)drn.p, 0, false, NTT); }, 2, 8);
  double pairs = 0; for (int i = 0; i < U; i++) { int pos = p0 + i; pairs += swa ? std::min(pos + 1, WIN) : pos + 1; }
  double flops = pairs * NH * 4.0 * HD;
  printf("  %-24s U=%-5d prefix=%-6d NTT=%d  attn %8.3f ms (+prep %6.3f)  useful %6.2f TFLOPS  %8.0f tok/s/layer  (key-pairs %.3g)\n", label, U, p0, NTT, tp, t - tp, flops / (tp * 1e-3) / 1e12, U / (tp * 1e-3), pairs);
}
static void bench_prefill() {
  printf("[prefill perf] one layer visit, U query tokens over prefix+ubatch keys (f16 KV, WMMA f16, bias+sink fused, no mask tensor)\n");
  for (int U : {512, 2048}) for (int pre : {0, 4096, 32768}) { char l[64]; snprintf(l, sizeof l, "global"); bench_prefill_one(l, false, pre, U, pre + U + 16, 4); }
  for (int U : {512, 2048}) for (int pre : {0, 4096, 32768}) { char l[64]; snprintf(l, sizeof l, "SWA"); bench_prefill_one(l, true, pre, U, 1024 + U + 256, 4); }
  for (int NTT : {1, 2}) { bench_prefill_one("global NTT variant", false, 4096, 512, 4096 + 512 + 16, NTT); bench_prefill_one("SWA NTT variant", true, 4096, 512, 1792, NTT); }
}

static void bench_cmp() {
  printf("[shapes matching llama.cpp test-backend-ops perf cases: hsk=hsv=128, 8 kv heads, f16 KV, causal-with-bias here vs mask there]\n");
  printf(" decode, 1 query, warm cache (one copy, repeated: small contexts live in the 32 MB MALL, like the llama.cpp test):\n");
  for (int ctx : {4096, 8192, 16384}) { char l[64]; snprintf(l, sizeof l, "glob kv=%d warm", ctx); bench_decode_one(l, false, ctx, 1, ctx + 16, 0, false, 1); }
  printf(" prefill nb=U over kv=U (prefix 0):\n");
  for (int NTT : {4, 6}) bench_prefill_one("global nb=kv=4096", false, 0, 4096, 4096 + 16, NTT);
  bench_prefill_one("SWA nb=4096", true, 0, 4096, 1024 + 4096 + 256, 4);
}
int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0); std::string mode = argc > 1 ? argv[1] : "all";
  hipDeviceProp_t pr; CK(hipGetDeviceProperties(&pr, 0)); printf("device %s, CUs %d, L2 %d KiB\n", pr.name, pr.multiProcessorCount, pr.l2CacheSize / 1024);
  if (mode == "test" || mode == "all") {
    g_P.init(11); g_devbig.setup_params(g_P);
    test_kv_write();
    printf("building 200K-row host cache...\n"); fflush(stdout);
    g_big.init(CAPG); g_big.fill(0, CAPG - 1, 99); g_devbig.setup_cache(g_big);
    g_devring.setup_params(g_P);
    test_decode(); test_prefill();
    printf("TEST SUMMARY: %d failing cases\n", g_fail);
  }
  if (mode == "bench" || mode == "all") { bench_decode(); bench_prefill(); }
  if (mode == "tune") tune_decode();
  if (mode == "cmp") bench_cmp();
  if (mode == "ptune") { for (int NTT : {3, 4, 6, 8}) { bench_prefill_one("global", false, 32768, 512, 32768 + 512 + 16, NTT); bench_prefill_one("SWA", true, 32768, 512, 1792, NTT); bench_prefill_one("global U2048", false, 4096, 2048, 4096 + 2048 + 16, NTT); } }
  return g_fail ? 1 : 0;
}
