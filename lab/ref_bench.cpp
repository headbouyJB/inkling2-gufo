// llama.cpp (ggml-hip) reference timing for the same shapes: MUL_MAT (dense Q8_0, lm_head) and MUL_MAT_ID (NVFP4 / Q8_0 experts).
// Weights are random bytes (timing only). N independent nodes over N distinct weight tensors per graph => cache-cold; reports per-node us.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <algorithm>
#include <string>
static ggml_backend_t be;
static double run(ggml_context* ctx, std::vector<ggml_tensor*> outs, int reps) {
  ggml_cgraph* gf = ggml_new_graph_custom(ctx, 4096, false); for (auto o : outs) ggml_build_forward_expand(gf, o);
  ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be)); ggml_gallocr_alloc_graph(ga, gf);
  for (int i = 0; i < 3; i++) ggml_backend_graph_compute(be, gf); ggml_backend_synchronize(be);
  double best = 1e30;
  for (int r = 0; r < reps; r++) { auto t0 = std::chrono::steady_clock::now(); ggml_backend_graph_compute(be, gf); ggml_backend_synchronize(be);
    best = std::min(best, std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count()); }
  ggml_gallocr_free(ga); return best / outs.size();
}
static void fill_rand(ggml_tensor* t) { std::vector<uint8_t> buf(1 << 24); std::mt19937 g(1); for (auto& b : buf) b = g() & 0xFF; size_t n = ggml_nbytes(t);
  for (size_t o = 0; o < n; o += buf.size()) ggml_backend_tensor_set(t, buf.data(), o, std::min(buf.size(), n - o)); }
int main(int argc, char** argv) {
  ggml_backend_load_all_from_path(argc > 1 ? argv[1] : "."); be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr); if (!be) be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr);
  if (!be) { fprintf(stderr, "no gpu backend\n"); return 1; }
  printf("# backend %s\n", ggml_backend_name(be));
  if (argc > 2 && std::string(argv[2]) == "prefill") {
    struct PS { const char* name; int K, M; };
    PS dn[] = {{"attn_q", 2304, 3072}, {"attn_k", 2304, 1024}, {"attn_v", 2304, 1024}, {"attn_gate", 2304, 3072}, {"attn_output", 3072, 2304}, {"attn_rel", 2304, 384}};
    for (int U : {512, 1024, 2048}) {
      double tot = 0;
      for (auto& s : dn) { ggml_init_params ip = {ggml_tensor_overhead() * 4096 + (1 << 20), nullptr, true}; ggml_context* ctx = ggml_init(ip);
        ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s.K, U); std::vector<ggml_tensor*> W, outs;
        for (int i = 0; i < 6; i++) { W.push_back(ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, s.K, s.M)); outs.push_back(ggml_mul_mat(ctx, W.back(), x)); }
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be); for (auto w : W) fill_rand(w);
        std::vector<float> xh((size_t)s.K * U, 0.5f); ggml_backend_tensor_set(x, xh.data(), 0, xh.size() * 4);
        double us = run(ctx, outs, 8); tot += us; printf("REFP mul_mat q8_0 U=%d %-12s %9.1f us %6.1f TOPS\n", U, s.name, us, 2.0 * s.K * s.M * U / us * 1e-6);
        ggml_backend_buffer_free(buf); ggml_free(ctx); }
      printf("REFP dense_layer U=%d total %.0f us\n", U, tot);
      struct MS2 { const char* name; ggml_type t; int K, M; bool gate; int N; };
      MS2 ms[] = {{"gu_nvfp4", GGML_TYPE_NVFP4, 2304, 2304, true, 3}, {"down_nvfp4", GGML_TYPE_NVFP4, 1152, 2304, false, 3}, {"gu_q8_0", GGML_TYPE_Q8_0, 2304, 2304, true, 2}, {"down_q8_0", GGML_TYPE_Q8_0, 1152, 2304, false, 2}};
      for (auto& s : ms) { ggml_init_params ip = {ggml_tensor_overhead() * 4096 + (1 << 20), nullptr, true}; ggml_context* ctx = ggml_init(ip);
        ggml_tensor* ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 8, U); ggml_tensor* x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, s.K, s.gate ? 1 : 8, U);
        std::vector<ggml_tensor*> W, outs; for (int i = 0; i < s.N; i++) { W.push_back(ggml_new_tensor_3d(ctx, s.t, s.K, s.M, 128)); outs.push_back(ggml_mul_mat_id(ctx, W.back(), x, ids)); }
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be); for (auto w : W) fill_rand(w);
        std::mt19937 g(3); std::vector<int> ih(8 * U), perm(128); for (int b = 0; b < U; b++) { for (int i = 0; i < 128; i++) perm[i] = i; std::shuffle(perm.begin(), perm.end(), g); for (int e = 0; e < 8; e++) ih[b * 8 + e] = perm[e]; }
        ggml_backend_tensor_set(ids, ih.data(), 0, ih.size() * 4); std::vector<float> xh((size_t)s.K * (s.gate ? 1 : 8) * U, 0.5f); ggml_backend_tensor_set(x, xh.data(), 0, xh.size() * 4);
        double us = run(ctx, outs, 6); printf("REFP mul_mat_id %-11s U=%d %10.1f us %6.1f TOPS\n", s.name, U, us, 2.0 * s.K * s.M * 8.0 * U / us * 1e-6);
        ggml_backend_buffer_free(buf); ggml_free(ctx); }
    }
    return 0;
  }
  struct Sh { const char* name; int K, M; int N; };
  Sh dense[] = {{"attn_q", 2304, 3072, 24}, {"attn_k", 2304, 1024, 24}, {"attn_gate", 2304, 3072, 24}, {"attn_output", 3072, 2304, 24}, {"attn_rel", 2304, 384, 24}, {"lm_head", 2304, 201024, 1}};
  for (auto& s : dense) for (int B : {1, 2, 4}) {
    ggml_init_params ip = {ggml_tensor_overhead() * 4096 + (1 << 20), nullptr, true}; ggml_context* ctx = ggml_init(ip);
    std::vector<ggml_tensor*> W, outs; ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s.K, B); ggml_set_input(x);
    for (int i = 0; i < s.N; i++) { W.push_back(ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, s.K, s.M)); outs.push_back(ggml_mul_mat(ctx, W.back(), x)); }
    if (s.N == 1) { for (int i = 1; i < 6; i++) {} }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be); for (auto w : W) fill_rand(w);
    std::vector<float> xh(s.K * B, 0.5f); ggml_backend_tensor_set(x, xh.data(), 0, xh.size() * 4);
    double us = run(ctx, outs, s.N == 1 ? 20 : 30); double bytes = (double)s.M * s.K / 32 * 34;
    printf("REF mul_mat q8_0 %-12s B=%d %9.2f us %7.1f GB/s\n", s.name, B, us, bytes / us * 1e-3);
    ggml_backend_buffer_free(buf); ggml_free(ctx);
  }
  struct MS { const char* name; ggml_type t; int K, M, nused, N; bool gate; };
  MS moe[] = {{"gu_nvfp4", GGML_TYPE_NVFP4, 2304, 2304, 8, 8, true}, {"down_nvfp4", GGML_TYPE_NVFP4, 1152, 2304, 8, 8, false},
              {"gu_q8_0", GGML_TYPE_Q8_0, 2304, 2304, 8, 4, true}, {"down_q8_0", GGML_TYPE_Q8_0, 1152, 2304, 8, 4, false}};
  for (auto& s : moe) for (int B : {1, 2, 4}) {
    ggml_init_params ip = {ggml_tensor_overhead() * 4096 + (1 << 20), nullptr, true}; ggml_context* ctx = ggml_init(ip);
    ggml_tensor* ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 8, B); ggml_set_input(ids);
    ggml_tensor* x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, s.K, s.gate ? 1 : 8, B); ggml_set_input(x);
    std::vector<ggml_tensor*> W, outs;
    for (int i = 0; i < s.N; i++) { W.push_back(ggml_new_tensor_3d(ctx, s.t, s.K, s.M, 128)); outs.push_back(ggml_mul_mat_id(ctx, W.back(), x, ids)); }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be); for (auto w : W) fill_rand(w);
    std::mt19937 g(3); std::vector<int> ih(8 * B), perm(128); std::vector<int> uq;
    for (int b = 0; b < B; b++) { for (int i = 0; i < 128; i++) perm[i] = i; std::shuffle(perm.begin(), perm.end(), g); for (int e = 0; e < 8; e++) ih[b * 8 + e] = perm[e]; }
    uq = ih; std::sort(uq.begin(), uq.end()); int nu = std::unique(uq.begin(), uq.end()) - uq.begin();
    ggml_backend_tensor_set(ids, ih.data(), 0, ih.size() * 4);
    std::vector<float> xh((size_t)s.K * (s.gate ? 1 : 8) * B, 0.5f); ggml_backend_tensor_set(x, xh.data(), 0, xh.size() * 4);
    double us = run(ctx, outs, 30); double bytes = (double)nu * s.M * s.K / (s.t == GGML_TYPE_NVFP4 ? 64 : 32) * (s.t == GGML_TYPE_NVFP4 ? 36 : 34);
    printf("REF mul_mat_id %-11s B=%d %9.2f us %7.1f GB/s (uniq experts %d)\n", s.name, B, us, bytes / us * 1e-3, nu);
    ggml_backend_buffer_free(buf); ggml_free(ctx);
  }
  return 0;
}
