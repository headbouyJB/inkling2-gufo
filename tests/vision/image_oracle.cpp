// Image prompt oracle for the Inkling 2 Mini HIP engine (milestone M5).
//
//   inkling2_image_oracle --model GGUF --mmproj GGUF --image FILE --question TEXT --out dump.bin
//                         [--prompt-out prompt.txt] [--ids-out ids.bin] [--ctx N] [--reps N] [--system TEXT]
//
// Renders the same prompt as the runner (one user message: the image, then the question; reasoning
// off, so the prompt ends with <|message_model|><|content_text|>), encodes the image with
// VisionEncoder, injects the rows through Engine::Forward(InputRows) and writes the logits of the last
// position in the I2OR layout (one record; see oracle.cpp). --prompt-out writes the text the llama.cpp
// side feeds mtmd (the image block replaced by the media marker). Prints timings.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include <gufo/ext/gguf_reader.hpp>
#include <gufo/ext/text.hpp>

#include "src/engine.hpp"
#include "src/vision/vision_encoder.hpp"
#include "text/gufo_adapter.hpp"
#include "text/tokenizer.hpp"

namespace g = gufo::ext;
namespace ga = inkling2::text::gufo_adapter;
using Clock = std::chrono::steady_clock;
static double Ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }

int main(int argc, char** argv) {
  std::string append, model, mmproj, image, question = "What is in this image?", out, prompt_out, ids_out, system;
  std::uint32_t ctx = 0;
  int reps = 3;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (k == "--model") model = next();
    else if (k == "--mmproj") mmproj = next();
    else if (k == "--image") image = next();
    else if (k == "--question") question = next();
    else if (k == "--system") system = next();
    else if (k == "--append") append = next();
    else if (k == "--out") out = next();
    else if (k == "--prompt-out") prompt_out = next();
    else if (k == "--ids-out") ids_out = next();
    else if (k == "--ctx") ctx = static_cast<std::uint32_t>(std::strtoul(next().c_str(), nullptr, 10));
    else if (k == "--reps") reps = std::atoi(next().c_str());
    else { std::fprintf(stderr, "unknown option %s\n", k.c_str()); return 2; }
  }
  if (model.empty() || mmproj.empty() || image.empty() || out.empty()) {
    std::fprintf(stderr, "usage: inkling2_image_oracle --model G --mmproj G --image F --out F [--question T]\n");
    return 2;
  }
  std::string err;
  namespace vis = gufo::models::inkling2::vision;
  vis::ImageU8 img;
  if (!vis::DecodeImageFile(image, &img, &err)) { std::fprintf(stderr, "image: %s\n", err.c_str()); return 1; }
  const std::size_t n_img = static_cast<std::size_t>(vis::VisionEncoder::CountTokens(img.w, img.h));

  auto tok = inkling2::text::Tokenizer::CreateFromGguf(model, &err);
  if (!tok) { std::fprintf(stderr, "tokenizer: %s\n", err.c_str()); return 1; }
  g::ChatRequest request;
  request.reasoning.enabled = false;
  if (!system.empty()) request.messages.push_back(g::ChatMessage(g::ChatRole::kSystem, system));
  g::ChatMessage m(g::ChatRole::kUser, question);
  g::ChatMessage::ImagePart part;
  part.offset = 0;
  part.bytes = std::make_shared<const std::vector<std::uint8_t>>(1, 0);  // only its presence matters here
  m.images.push_back(part);
  request.messages.push_back(m);
  inkling2::text::PromptTokens pt;
  const std::vector<std::size_t> counts{n_img};
  auto ids_opt = ga::RenderAndTokenize(*tok, request, &counts, &pt);
  std::vector<std::int32_t> ids(ids_opt->begin(), ids_opt->end());
  const std::size_t first = pt.image_positions.at(0);
  // --append TEXT: teacher-force these tokens after the prompt (dump the logits at the last prompt token and at
  // each appended one: 1 + n records); their ids go to <out>.append.ids for the llama.cpp side.
  const std::size_t n_prompt = ids.size();
  std::vector<std::int32_t> extra;
  if (!append.empty()) {
    std::vector<inkling2::text::TokenId> e;
    tok->EncodeOrdinary(append, &e);
    extra.assign(e.begin(), e.end());
    ids.insert(ids.end(), extra.begin(), extra.end());
    std::ofstream f(out + ".append.ids", std::ios::binary);
    f.write(reinterpret_cast<const char*>(extra.data()), static_cast<std::streamsize>(extra.size() * 4));
  }
  std::fprintf(stderr, "image %dx%d -> %zu rows at tokens [%zu,%zu), prompt %zu tokens\n", img.w, img.h, n_img, first,
               first + n_img, ids.size());
  if (!ids_out.empty()) {
    std::ofstream f(ids_out, std::ios::binary);
    f.write(reinterpret_cast<const char*>(ids.data()), static_cast<std::streamsize>(ids.size() * 4));
  }
  if (!prompt_out.empty()) {
    // text for mtmd: the tokens before <|content_image|> (id at first-1), the media marker, the tokens after the
    // placeholders (from <|end_message|> on)
    std::string text;
    for (std::size_t i = 0; i + 1 < first; ++i) text += tok->TokenBytes(ids[i]);
    text += "<__media__>";
    for (std::size_t i = first + n_img; i < ids.size(); ++i) text += tok->TokenBytes(ids[i]);
    std::ofstream f(prompt_out, std::ios::binary);
    f << text;
  }

  gufo::core::GgufOpenOptions open;
  open.allow_unsized_tensors = true;
  std::shared_ptr<const gufo::core::GgufReader> reader = gufo::core::GgufReader::OpenFile(model, &err, open);
  if (!reader) { std::fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
  auto engine = gufo::models::inkling2::Engine::Create(reader, ctx ? ctx : static_cast<std::uint32_t>(ids.size() + 64), &err);
  if (!engine) { std::fprintf(stderr, "engine: %s\n", err.c_str()); return 1; }
  vis::VisionEncoder enc;
  if (!enc.Load(mmproj, &err)) { std::fprintf(stderr, "mmproj: %s\n", err.c_str()); return 1; }
  hipStream_t stream;
  (void)hipStreamCreate(&stream);

  float* rows = nullptr;
  (void)hipMalloc(&rows, n_img * 2304 * sizeof(float));
  double enc_ms_first = 0, enc_ms_best = 1e30;
  for (int r = 0; r < std::max(reps, 1); ++r) {
    vis::VisionEncoder::Result res;
    const auto t0 = Clock::now();
    if (!enc.Encode(img.rgb.data(), img.w, img.h, stream, &res, &err)) { std::fprintf(stderr, "encode: %s\n", err.c_str()); return 1; }
    (void)hipMemcpyAsync(rows, res.embd, n_img * 2304 * sizeof(float), hipMemcpyDeviceToDevice, stream);
    (void)hipStreamSynchronize(stream);
    const double ms = Ms(t0, Clock::now());
    if (r == 0) enc_ms_first = ms;
    enc_ms_best = std::min(enc_ms_best, ms);
  }
  std::fprintf(stderr, "encode: first %.1f ms, best %.1f ms (%zu rows)\n", enc_ms_first, enc_ms_best, n_img);

  const std::size_t V = engine->config().n_vocab_text;
  std::ofstream o(out, std::ios::binary);
  const std::uint32_t version = 1, n_pos = static_cast<std::uint32_t>(1 + extra.size()), top = 64;
  o.write("I2OR", 4);
  o.write(reinterpret_cast<const char*>(&version), 4);
  o.write(reinterpret_cast<const char*>(&n_pos), 4);
  o.write(reinterpret_cast<const char*>(&top), 4);
  std::vector<int> order(V);
  auto dump = [&](const float* l) {
    double mx = -1e30;
    for (std::size_t i = 0; i < V; ++i) mx = std::max<double>(mx, l[i]);
    double sum = 0;
    for (std::size_t i = 0; i < V; ++i) sum += std::exp(static_cast<double>(l[i]) - mx);
    const float lse = static_cast<float>(mx + std::log(sum));
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + 64, order.end(), [&](int x, int y) { return l[x] > l[y] || (l[x] == l[y] && x < y); });
    std::int32_t tid[64];
    float tl[64];
    for (int i = 0; i < 64; ++i) { tid[i] = order[i]; tl[i] = l[order[i]]; }
    o.write(reinterpret_cast<const char*>(&lse), 4);
    o.write(reinterpret_cast<const char*>(tid), sizeof tid);
    o.write(reinterpret_cast<const char*>(tl), sizeof tl);
    o.write(reinterpret_cast<const char*>(tl), sizeof tl);
    std::fprintf(stderr, "top-1 token %d (%s) logit %.3f\n", tid[0], std::string(tok->TokenBytes(tid[0])).c_str(), tl[0]);
  };
  std::vector<gufo::models::inkling2::InputRows> inj{{first, n_img, rows}};
  double fwd_best = 1e30;
  for (int r = 0; r < std::max(reps, 1); ++r) {
    auto session = engine->NewSession();
    const auto t0 = Clock::now();
    const bool dump_now = r + 1 == std::max(reps, 1);
    if (extra.empty() || !dump_now) {
      engine->Forward(*session, ids, {}, nullptr, inj);
      if (dump_now) dump(session->logits().data());
    } else {
      engine->Forward(*session, ids,
                      [&](std::size_t f, std::size_t count, const float* d) {
                        for (std::size_t i = 0; i < count; ++i)
                          if (f + i + 1 >= n_prompt) dump(d + i * V);
                      },
                      nullptr, inj);
    }
    fwd_best = std::min(fwd_best, Ms(t0, Clock::now()));
  }
  std::fprintf(stderr, "prefill of %zu tokens (%zu image rows): best %.1f ms = %.0f tok/s\n", ids.size(), n_img, fwd_best,
               ids.size() / (fwd_best / 1000.0));
  return 0;
}
