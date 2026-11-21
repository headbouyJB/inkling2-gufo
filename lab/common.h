#pragma once
// shared helpers (copied from lab.cpp). Kernel lab: validation + benchmark of decode-critical Inkling 2 Mini matrix kernels on real GGUF tensors. MIT.
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

