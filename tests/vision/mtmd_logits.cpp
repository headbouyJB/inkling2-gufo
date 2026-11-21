// llama.cpp side of the M5 image check: runs a prompt with one media marker through libmtmd + libllama
// (text model on the GPU or CPU) and writes the logits of the last position in the I2OR layout
// (one record, like tests/vision/image_oracle.cpp). MIT; links libmtmd + libllama.
//   mtmd_logits MMPROJ TEXT.gguf IMAGE PROMPT.txt OUT.bin [ngl] [threads]
// PROMPT.txt: the prompt with "<__media__>" where the image goes; special tokens are parsed. env
// MTMD_IDS_OUT=file writes the token ids of the text chunks (-1 marks the image rows). MTMD_NBATCH=N sets
// n_batch = n_ubatch (default 2048). MTMD_GEN=N continues greedily for up to N tokens (stops at <|end_message|>
// unless MTMD_GEN_PAST is set; MTMD_VISION_CPU=1 runs the projector on the CPU) and prints "GEN: <text>". MTMD_APPEND_IDS=file teacher-forces token ids.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

int main(int argc, char** argv) {
  if (argc < 6) { std::fprintf(stderr, "usage: mtmd_logits MMPROJ TEXT.gguf IMAGE PROMPT.txt OUT.bin [ngl] [threads]\n"); return 2; }
  const int ngl = argc > 6 ? std::atoi(argv[6]) : 99;
  const int threads = argc > 7 ? std::atoi(argv[7]) : 16;
  std::ifstream pf(argv[4], std::ios::binary);
  std::ostringstream ps; ps << pf.rdbuf();
  const std::string prompt = ps.str();
  llama_backend_init();
  ggml_backend_load_all();
  llama_model_params mp = llama_model_default_params();
  mp.n_gpu_layers = ngl;
  llama_model* model = llama_model_load_from_file(argv[2], mp);
  if (!model) { std::fprintf(stderr, "text model load failed\n"); return 1; }
  const llama_vocab* vocab = llama_model_get_vocab(model);
  const int V = llama_vocab_n_tokens(vocab);
  mtmd_context_params cp = mtmd_context_params_default();
  cp.use_gpu = ngl > 0 && !std::getenv("MTMD_VISION_CPU");  // MTMD_VISION_CPU=1: the projector on the CPU, text on the GPU
  cp.n_threads = threads;
  cp.print_timings = true;
  cp.warmup = false;
  mtmd_context* mctx = mtmd_init_from_file(argv[1], model, cp);
  if (!mctx) { std::fprintf(stderr, "mtmd init failed\n"); return 1; }
  mtmd_bitmap* bmp = mtmd_helper_bitmap_init_from_file(mctx, argv[3], false, mtmd_helper_init_opt_default()).bitmap;
  if (!bmp) { std::fprintf(stderr, "image load failed\n"); return 1; }
  mtmd_input_chunks* chunks = mtmd_input_chunks_init();
  mtmd_input_text txt{prompt.c_str(), prompt.size(), false, true};
  const mtmd_bitmap* bs[1] = {bmp};
  if (mtmd_tokenize(mctx, chunks, &txt, bs, 1) != 0) { std::fprintf(stderr, "tokenize failed\n"); return 1; }
  std::vector<int32_t> ids;
  size_t n_total = 0;
  for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
    const mtmd_input_chunk* c = mtmd_input_chunks_get(chunks, i);
    if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
      size_t n = 0;
      const llama_token* t = mtmd_input_chunk_get_tokens_text(c, &n);
      ids.insert(ids.end(), t, t + n);
      n_total += n;
    } else {
      const size_t n = mtmd_image_tokens_get_n_tokens(mtmd_input_chunk_get_tokens_image(c));
      ids.insert(ids.end(), n, -1);
      n_total += n;
    }
  }
  if (const char* p = std::getenv("MTMD_IDS_OUT")) {
    std::ofstream f(p, std::ios::binary);
    f.write(reinterpret_cast<const char*>(ids.data()), static_cast<std::streamsize>(ids.size() * 4));
  }
  const int n_batch = std::getenv("MTMD_NBATCH") ? std::atoi(std::getenv("MTMD_NBATCH")) : 2048;  // also n_ubatch
  llama_context_params lp = llama_context_default_params();
  lp.n_ctx = static_cast<uint32_t>(n_total + 64);
  lp.n_batch = n_batch;
  lp.n_ubatch = n_batch;
  lp.n_threads = threads;
  lp.n_threads_batch = threads;
  llama_context* lctx = llama_init_from_model(model, lp);
  if (!lctx) { std::fprintf(stderr, "context failed\n"); return 1; }
  llama_pos n_past = 0;
  if (mtmd_helper_eval_chunks(mctx, lctx, chunks, 0, 0, n_batch, true, &n_past) != 0) { std::fprintf(stderr, "eval failed\n"); return 1; }
  std::fprintf(stderr, "prompt %zu positions, n_past %d\n", n_total, (int)n_past);
  // MTMD_APPEND_IDS=file: teacher-force these token ids after the prompt, one record per position (1 + n).
  std::vector<llama_token> extra;
  if (const char* ap = std::getenv("MTMD_APPEND_IDS")) {
    std::ifstream af(ap, std::ios::binary);
    std::ostringstream as; as << af.rdbuf();
    const std::string raw = as.str();
    extra.resize(raw.size() / 4);
    std::memcpy(extra.data(), raw.data(), extra.size() * 4);
  }
  std::vector<float> all;  // (1 + extra.size()) x V
  all.insert(all.end(), llama_get_logits_ith(lctx, -1), llama_get_logits_ith(lctx, -1) + V);
  for (auto t : extra) {
    llama_batch b = llama_batch_get_one(&t, 1);
    if (llama_decode(lctx, b) != 0) { std::fprintf(stderr, "append decode failed\n"); return 1; }
    const float* q = llama_get_logits_ith(lctx, -1);
    all.insert(all.end(), q, q + V);
  }
  const float* l = all.data();
  std::ofstream o(argv[5], std::ios::binary);
  const uint32_t version = 1, n_pos = static_cast<uint32_t>(1 + extra.size()), top = 64;
  o.write("I2OR", 4);
  o.write(reinterpret_cast<const char*>(&version), 4);
  o.write(reinterpret_cast<const char*>(&n_pos), 4);
  o.write(reinterpret_cast<const char*>(&top), 4);
  int32_t tid[64]; float tl[64];
  std::vector<int> order(V);
  for (size_t p = 0; p < n_pos; ++p) {
    const float* lp = l + p * V;
    double mx = -1e30;
    for (int k = 0; k < V; ++k) mx = std::max<double>(mx, lp[k]);
    double sum = 0;
    for (int k = 0; k < V; ++k) sum += std::exp(static_cast<double>(lp[k]) - mx);
    const float lse = static_cast<float>(mx + std::log(sum));
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + 64, order.end(), [&](int x, int y) { return lp[x] > lp[y] || (lp[x] == lp[y] && x < y); });
    for (int k = 0; k < 64; ++k) { tid[k] = order[k]; tl[k] = lp[order[k]]; }
    o.write(reinterpret_cast<const char*>(&lse), 4);
    o.write(reinterpret_cast<const char*>(tid), sizeof tid);
    o.write(reinterpret_cast<const char*>(tl), sizeof tl);
    o.write(reinterpret_cast<const char*>(tl), sizeof tl);
  }
  std::fprintf(stderr, "top-1 token %d (%s) logit %.3f\n", tid[0], llama_vocab_get_text(vocab, tid[0]), tl[0]);
  if (const char* g = std::getenv("MTMD_GEN")) {
    const int n_gen = std::atoi(g);
    std::string text;
    llama_token t = static_cast<llama_token>(std::max_element(all.data(), all.data() + V) - all.data());
    for (int i = 0; i < n_gen && !llama_vocab_is_eog(vocab, t); ++i) {
      char piece[256];
      const int n = llama_token_to_piece(vocab, t, piece, sizeof piece, 0, true);
      if (n > 0) text.append(piece, n);
      if (!std::getenv("MTMD_GEN_PAST") && std::string(llama_vocab_get_text(vocab, t)) == "<|end_message|>") break;
      llama_batch b = llama_batch_get_one(&t, 1);
      if (llama_decode(lctx, b) != 0) break;
      const float* q = llama_get_logits_ith(lctx, -1);
      t = static_cast<llama_token>(std::max_element(q, q + V) - q);
    }
    std::printf("GEN: %s\n", text.c_str());
  }
  return 0;
}
