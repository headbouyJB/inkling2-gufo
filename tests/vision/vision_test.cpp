// Inkling 2 vision encoder test tool (MIT). Modes:
//   vision_test gen OUTDIR                       write the three synthetic PNGs
//   vision_test run MMPROJ IMG OUT.bin [--exact | --emulate-q8] [--reps N]
//                                                encode IMG, write [n_tokens][2304] f32 (see header below), time it
//   vision_test cmp A.bin B.bin                  per-token cosine / max-abs-error summary
//   vision_test tokens WxH ...                   print token counts for sizes (no GPU)
// .bin layout: int32 n_tokens, int32 dim, int32 tok_w, int32 tok_h, then n_tokens*dim float32.
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../../src/vision/vision_encoder.hpp"

using namespace gufo::models::inkling2::vision;

// ---- synthetic images ---------------------------------------------------------------------------
namespace {
struct Img { int w, h; std::vector<uint8_t> px; };

Img Make(int w, int h) { return {w, h, std::vector<uint8_t>(size_t(w) * h * 3)}; }
void Put(Img& im, int x, int y, int r, int g, int b) {
  if (x < 0 || y < 0 || x >= im.w || y >= im.h) return;
  uint8_t* p = &im.px[(size_t(y) * im.w + x) * 3];
  p[0] = uint8_t(r); p[1] = uint8_t(g); p[2] = uint8_t(b);
}
void Rect(Img& im, int x0, int y0, int x1, int y1, int r, int g, int b) {
  for (int y = y0; y < y1; y++) for (int x = x0; x < x1; x++) Put(im, x, y, r, g, b);
}
void Disc(Img& im, int cx, int cy, int rad, int r, int g, int b) {
  for (int y = cy - rad; y <= cy + rad; y++)
    for (int x = cx - rad; x <= cx + rad; x++)
      if ((x - cx) * (x - cx) + (y - cy) * (y - cy) <= rad * rad) Put(im, x, y, r, g, b);
}
void Gradient(Img& im) {
  for (int y = 0; y < im.h; y++) for (int x = 0; x < im.w; x++)
    Put(im, x, y, 40 + 180 * x / im.w, 60 + 120 * y / im.h, 200 - 150 * x / im.w);
}
// 5x7 glyphs for "INKLG2 " (rows, MSB = left)
const uint8_t kGlyph[7][7] = {
  {0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x1F},  // I
  {0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11},  // N
  {0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11},  // K
  {0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F},  // L
  {0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F},  // G
  {0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F},  // 2
  {0, 0, 0, 0, 0, 0, 0}};
void Text(Img& im, int x0, int y0, const char* s, int scale, int r, int g, int b) {
  const char* order = "INKLG2 ";
  for (; *s; s++, x0 += 6 * scale) {
    const char* f = std::strchr(order, *s);
    if (!f) continue;
    const uint8_t* gl = kGlyph[f - order];
    for (int row = 0; row < 7; row++) for (int col = 0; col < 5; col++)
      if (gl[row] & (0x10 >> col)) Rect(im, x0 + col * scale, y0 + row * scale, x0 + (col + 1) * scale, y0 + (row + 1) * scale, r, g, b);
  }
}
uint32_t lcg(uint32_t& s) { s = s * 1664525u + 1013904223u; return s >> 8; }

Img Shapes512x320() {
  Img im = Make(512, 320);
  Gradient(im);
  Rect(im, 30, 30, 190, 120, 230, 40, 40);
  Rect(im, 210, 60, 300, 200, 20, 20, 20);
  Disc(im, 400, 100, 60, 250, 230, 40);
  Disc(im, 120, 230, 45, 255, 255, 255);
  Text(im, 20, 150, "INKLING 2", 5, 255, 255, 255);
  Text(im, 250, 250, "LINK 22", 6, 10, 10, 10);
  return im;
}
Img Tiny40() {
  Img im = Make(40, 40);
  Gradient(im);
  Disc(im, 20, 20, 12, 255, 200, 0);
  Rect(im, 0, 34, 40, 40, 0, 0, 0);
  return im;
}
Img Big1280x720() {
  Img im = Make(1280, 720);
  Gradient(im);
  uint32_t s = 12345;
  for (int i = 0; i < 60; i++) {
    int x = lcg(s) % 1280, y = lcg(s) % 720, a = 20 + lcg(s) % 80;
    if (i % 2) Disc(im, x, y, a, lcg(s) & 255, lcg(s) & 255, lcg(s) & 255);
    else Rect(im, x, y, x + a * 2, y + a, lcg(s) & 255, lcg(s) & 255, lcg(s) & 255);
  }
  Text(im, 100, 100, "INKLING 2 22", 12, 255, 255, 255);
  Text(im, 600, 500, "KING GLINK", 9, 0, 0, 0);
  for (int i = 0; i < 20000; i++) Put(im, lcg(s) % 1280, lcg(s) % 720, lcg(s) & 255, lcg(s) & 255, lcg(s) & 255);
  return im;
}
bool SavePng(const Img& im, const std::string& path) {
  return stbi_write_png(path.c_str(), im.w, im.h, 3, im.px.data(), im.w * 3) != 0;
}

double Now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: see header\n"); return 2; }
  const std::string mode = argv[1];
  if (mode == "gen" && argc >= 3) {
    const std::string d = argv[2];
    bool ok = SavePng(Shapes512x320(), d + "/shapes_512x320.png") && SavePng(Tiny40(), d + "/tiny_40x40.png") &&
              SavePng(Big1280x720(), d + "/big_1280x720.png");
    return ok ? 0 : 1;
  }
  if (mode == "tokens") {
    for (int i = 2; i < argc; i++) {
      int w = 0, h = 0;
      if (std::sscanf(argv[i], "%dx%d", &w, &h) != 2) continue;
      ImagePlan p = PlanImage(w, h);
      std::printf("%dx%d -> pad %dx%d grid %dx%d tokens %dx%d = %d%s\n", w, h, p.pad_w, p.pad_h, p.grid_w, p.grid_h,
                  p.tok_w, p.tok_h, p.n_tokens, p.downscaled ? " (downscaled)" : "");
    }
    return 0;
  }
  if (mode == "cmp" && argc >= 4) {
    auto rd = [](const char* path, int* hdr, std::vector<float>* v) {
      FILE* f = std::fopen(path, "rb");
      if (!f || std::fread(hdr, 4, 4, f) != 4) return false;
      v->resize(size_t(hdr[0]) * hdr[1]);
      bool ok = std::fread(v->data(), 4, v->size(), f) == v->size();
      std::fclose(f);
      return ok;
    };
    int ha[4], hb[4];
    std::vector<float> a, b;
    if (!rd(argv[2], ha, &a) || !rd(argv[3], hb, &b) || ha[0] != hb[0] || ha[1] != hb[1]) { std::fprintf(stderr, "bad inputs / shape mismatch\n"); return 1; }
    const int n = ha[0], d = ha[1];
    double cmin = 2, csum = 0, emax = 0, esum = 0, rel_sum = 0, rel_max = 0;
    int cmin_tok = -1, emax_tok = -1;
    for (int t = 0; t < n; t++) {
      double dot = 0, na = 0, nb = 0, em = 0, se = 0;
      for (int i = 0; i < d; i++) {
        double x = a[size_t(t) * d + i], y = b[size_t(t) * d + i];
        dot += x * y; na += x * x; nb += y * y;
        em = std::max(em, std::fabs(x - y)); se += (x - y) * (x - y);
      }
      double cs = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30), rel = std::sqrt(se) / (std::sqrt(nb) + 1e-30);
      if (cs < cmin) { cmin = cs; cmin_tok = t; }
      if (em > emax) { emax = em; emax_tok = t; }
      csum += cs; esum += em; rel_sum += rel; rel_max = std::max(rel_max, rel);
    }
    std::printf("tokens %d dim %d | cos min %.6f (tok %d) mean %.6f | max|err| %.4g (tok %d) mean-of-token-max %.4g | rel-L2 mean %.4g max %.4g\n",
                n, d, cmin, cmin_tok, csum / n, emax, emax_tok, esum / n, rel_sum / n, rel_max);
    return 0;
  }
  if (mode == "bench" && argc >= 4) {  // bench MMPROJ WxH... : time encodes of synthetic images (WMMA)
    VisionEncoder enc;
    std::string err;
    if (!enc.Load(argv[2], &err)) { std::fprintf(stderr, "load: %s\n", err.c_str()); return 1; }
    hipStream_t s;
    hipStreamCreate(&s);
    for (int i = 3; i < argc; i++) {
      int w = 0, h = 0;
      if (std::sscanf(argv[i], "%dx%d", &w, &h) != 2) continue;
      Img im = Make(w, h);
      uint32_t st = 777;
      for (size_t k = 0; k < im.px.size(); k++) im.px[k] = uint8_t(lcg(st));
      VisionEncoder::Result r;
      std::vector<double> ms;
      for (int rep = 0; rep < 4; rep++) {
        hipStreamSynchronize(s);
        double a = Now();
        if (!enc.Encode(im.px.data(), w, h, s, &r, &err)) { std::fprintf(stderr, "encode: %s\n", err.c_str()); return 1; }
        hipStreamSynchronize(s);
        ms.push_back((Now() - a) * 1e3);
      }
      std::sort(ms.begin() + 1, ms.end());
      std::printf("%dx%d -> %d tokens (%d patches): warm-up %.0f ms, median %.1f ms (%.2f ms/token)\n", w, h, r.n_tokens,
                  r.plan.grid_w * r.plan.grid_h, ms[0], ms[2], ms[2] / r.n_tokens);
    }
    return 0;
  }
  if (mode == "run" && argc >= 5) {
    bool exact = false, emu = false;
    int reps = 3;
    for (int i = 5; i < argc; i++) {
      if (!std::strcmp(argv[i], "--exact")) exact = true;
      else if (!std::strcmp(argv[i], "--emulate-q8")) { exact = true; emu = true; }
      else if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) reps = std::atoi(argv[++i]);
    }
    ImageU8 img;
    std::string err;
    if (!DecodeImageFile(argv[3], &img, &err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    VisionEncoder enc;
    double t0 = Now();
    if (!enc.Load(argv[2], &err, exact)) { std::fprintf(stderr, "load: %s\n", err.c_str()); return 1; }
    hipDeviceSynchronize();
    std::printf("load %.2fs\n", Now() - t0);
    VisionEncoder::Options o;
    o.emulate_q8_acts = emu;
    enc.set_options(o);
    hipStream_t s;
    hipStreamCreate(&s);
    VisionEncoder::Result r;
    std::vector<double> ms;
    for (int i = 0; i < reps + 1; i++) {  // first run = warm-up (allocations, code load)
      hipStreamSynchronize(s);
      double a = Now();
      if (!enc.Encode(img.rgb.data(), img.w, img.h, s, &r, &err)) { std::fprintf(stderr, "encode: %s\n", err.c_str()); return 1; }
      hipStreamSynchronize(s);
      ms.push_back((Now() - a) * 1e3);
    }
    std::sort(ms.begin() + 1, ms.end());
    std::printf("%s %dx%d -> %d tokens (%dx%d, pad %dx%d, %d patches) | encode %s: first %.1f ms, median of %d = %.1f ms\n",
                argv[3], img.w, img.h, r.n_tokens, r.tok_w, r.tok_h, r.plan.pad_w, r.plan.pad_h, r.plan.grid_w * r.plan.grid_h,
                exact ? "exact-f32" : "wmma", ms[0], reps, ms[1 + reps / 2]);
    std::vector<float> host(size_t(r.n_tokens) * r.dim);
    hipMemcpy(host.data(), r.embd, host.size() * 4, hipMemcpyDeviceToHost);
    FILE* f = std::fopen(argv[4], "wb");
    int hdr[4] = {r.n_tokens, r.dim, r.tok_w, r.tok_h};
    std::fwrite(hdr, 4, 4, f);
    std::fwrite(host.data(), 4, host.size(), f);
    std::fclose(f);
    return 0;
  }
  std::fprintf(stderr, "bad arguments\n");
  return 2;
}
