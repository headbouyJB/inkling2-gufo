// Oracle: runs llama.cpp's mtmd library (CPU or GPU) on an image and dumps the image embeddings in
// the vision_test .bin layout. Also prints how mtmd tokenizes "<marker>" for this model (tokens,
// positions, causality). MIT; links libmtmd + libllama.
//   mtmd_dump MMPROJ TEXT_MODEL.gguf IMAGE OUT.bin [cpu|gpu] [threads]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <sys/stat.h>
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

// MTMD_TAP=dir: write the named intermediate tensors (patch_embd, embeddings, pre_ln, layer_out-N,
// encoder_out, projected) as vision_test .bin files (ne1 rows of ne0 floats) for layer-by-layer diffs.
static const char* g_tap_dir = nullptr;
static bool tap_cb(struct ggml_tensor* t, bool ask, void*) {
  const char* n = ggml_get_name(t);
  const bool want = !std::strcmp(n, "patch_embd") || !std::strcmp(n, "embeddings") || !std::strcmp(n, "pre_ln") ||
                    !std::strncmp(n, "layer_out-", 10) || !std::strcmp(n, "encoder_out") || !std::strcmp(n, "projected") ||
                    !std::strncmp(n, "attn_out-0", 10) || !std::strncmp(n, "ffn_out-0", 9);
  if (ask) return want;
  if (!want || t->type != GGML_TYPE_F32) return true;
  std::vector<float> buf(ggml_nelements(t));
  ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * 4);
  std::string path = std::string(g_tap_dir) + "/" + n + ".bin";
  FILE* f = std::fopen(path.c_str(), "wb");
  int hdr[4] = {(int)t->ne[1], (int)t->ne[0], 0, 0};
  std::fwrite(hdr, 4, 4, f);
  std::fwrite(buf.data(), 4, buf.size(), f);
  std::fclose(f);
  return true;
}

int main(int argc, char** argv) {
  if (argc < 5) { std::fprintf(stderr, "usage: mtmd_dump MMPROJ TEXT.gguf IMAGE OUT.bin [cpu|gpu] [threads]\n"); return 2; }
  const bool gpu = argc > 5 && !std::strcmp(argv[5], "gpu");
  const int threads = argc > 6 ? std::atoi(argv[6]) : 16;
  llama_backend_init();
  ggml_backend_load_all();
  llama_model_params mp = llama_model_default_params();
  mp.vocab_only = true;  // only the vocab / hparams: the text weights are never needed
  llama_model* model = llama_model_load_from_file(argv[2], mp);
  if (!model) { std::fprintf(stderr, "text model load failed\n"); return 1; }
  mtmd_context_params cp = mtmd_context_params_default();
  cp.use_gpu = gpu;
  cp.n_threads = threads;
  cp.print_timings = true;
  cp.warmup = false;
  if (std::getenv("MTMD_FA_OFF")) cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;  // f32 attention in the oracle
  if (std::getenv("MTMD_TAP")) { g_tap_dir = std::getenv("MTMD_TAP"); cp.cb_eval = tap_cb; }
  mtmd_context* ctx = mtmd_init_from_file(argv[1], model, cp);
  if (!ctx) { std::fprintf(stderr, "mtmd init failed\n"); return 1; }
  std::printf("n_embd_inp (text) = %d, vision=%d, use_non_causal=%d, use_mrope=%d, marker=%s\n", llama_model_n_embd_inp(model),
              mtmd_support_vision(ctx), mtmd_decode_use_non_causal(ctx, nullptr), mtmd_decode_use_mrope(ctx), mtmd_get_marker(ctx));
  mtmd_bitmap* bmp = mtmd_helper_bitmap_init_from_file(ctx, argv[3], false, mtmd_helper_init_opt_default()).bitmap;
  if (!bmp) { std::fprintf(stderr, "image load failed\n"); return 1; }
  mtmd_input_chunks* chunks = mtmd_input_chunks_init();
  std::string prompt = std::string("<|message_user|>") + mtmd_get_marker(ctx) + "<|end_message|>";
  mtmd_input_text txt{prompt.c_str(), prompt.size(), false, true};
  const mtmd_bitmap* bs[1] = {bmp};
  if (mtmd_tokenize(ctx, chunks, &txt, bs, 1) != 0) { std::fprintf(stderr, "tokenize failed\n"); return 1; }
  const llama_vocab* vocab = llama_model_get_vocab(model);
  const mtmd_input_chunk* img_chunk = nullptr;
  const size_t nc = mtmd_input_chunks_size(chunks);
  for (size_t i = 0; i < nc; i++) {
    const mtmd_input_chunk* c = mtmd_input_chunks_get(chunks, i);
    if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
      size_t n = 0;
      const llama_token* t = mtmd_input_chunk_get_tokens_text(c, &n);
      std::printf("chunk %zu: TEXT %zu tokens:", i, n);
      for (size_t j = 0; j < n; j++) std::printf(" %d(%s)", t[j], llama_vocab_get_text(vocab, t[j]));
      std::printf("\n");
    } else {
      img_chunk = c;
      const mtmd_image_tokens* it = mtmd_input_chunk_get_tokens_image(c);
      std::printf("chunk %zu: IMAGE n_tokens=%zu n_pos=%d\n", i, mtmd_image_tokens_get_n_tokens(it), (int)mtmd_image_tokens_get_n_pos(it));
      for (size_t j = 0; j < 3 && j < mtmd_image_tokens_get_n_tokens(it); j++) {
        mtmd_decoder_pos dp = mtmd_image_tokens_get_decoder_pos(it, 100, j);
        std::printf("  decoder pos of token %zu with pos_0=100: t=%d x=%d y=%d z=%d\n", j, (int)dp.t, (int)dp.x, (int)dp.y, (int)dp.z);
      }
    }
  }
  if (!img_chunk) { std::fprintf(stderr, "no image chunk\n"); return 1; }
  if (mtmd_encode_chunk(ctx, img_chunk) != 0) { std::fprintf(stderr, "encode failed\n"); return 1; }
  const mtmd_image_tokens* it = mtmd_input_chunk_get_tokens_image(img_chunk);
  const int n = (int)mtmd_image_tokens_get_n_tokens(it), d = llama_model_n_embd_inp(model) > 0 ? llama_model_n_embd_inp(model) : 2304;  // vocab_only models report 0
  const float* e = mtmd_get_output_embd(ctx);
  FILE* f = std::fopen(argv[4], "wb");
  int hdr[4] = {n, d, (int)mtmd_bitmap_get_nx(bmp), (int)mtmd_bitmap_get_ny(bmp)};
  std::fwrite(hdr, 4, 4, f);
  std::fwrite(e, 4, size_t(n) * d, f);
  std::fclose(f);
  std::printf("wrote %d x %d\n", n, d);
  mtmd_input_chunks_free(chunks);
  mtmd_bitmap_free(bmp);
  mtmd_free(ctx);
  llama_model_free(model);
  return 0;
}
