// Oracle side of the DSpark check (0.1.3 step 2): teacher-forces a token-id file through the pr390 fork of llama.cpp (the DSpark
// reference, MIT) and writes, for our inkling2_dspark_test (--mode oracle|oracle-own|taps):
//
//   ids.bin        int32 token ids (the file that was read)
//   feats.bin      "DSF1", int32 n_pos, int32 4 (taps), int32 2304; then float32 [n_pos][4][2304]: for every position the inputs of
//                  target layers 5, 23, 41 and the final normed hidden state (llama_get_embeddings_layer_inp; the draft model's
//                  target_layer_ids), exactly the rows pr390's drafter reads
//   probes.txt     one line per draft: `pos k tok1 ... tokk` (pos = position of the anchor ids[pos]; the context is [0, pos))
//   logits_p<pos>.bin  int32 n_vocab, then float32 [7][n_vocab]: the final logits of block rows 1..7 of the k = 7 draft (the
//                  graph's softcap(base + head . correction), as llama_get_logits_ith returns them)
//   hidden_p<pos>.bin  (best effort) int32 n_embd, then float32 [8][n_embd]: the draft block's hidden rows after the output norm,
//                  when llama_get_embeddings_ith returns them for the draft context
//
// Usage (a normal pr390 command line plus three options of its own):
//   dspark_dump -m TARGET.gguf -ngl 99 -c 8192 -md DSPARK.gguf -ngld 99 --spec-type draft-dspark --spec-draft-n-max 7
//               --dump-ids IDS.bin --dump-out DIR --dump-positions 64,200,512,1024
// The ids are int32 little endian (written by the driver from the tokenizer of the Inkling model; the same ids go to
// `inkling2_dspark_test --mode oracle-own`). Nothing here runs unless the driver runs it; build: tools/build-dspark-dump.sh.
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "sampling.h"
#include "speculative.h"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {


// Removes `--dump-*` options from argv (pr390's parser does not know them) and keeps their values.
std::string TakeOption(int& argc, char** argv, const char* name) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], name) == 0) {
      std::string v = argv[i + 1];
      for (int j = i; j + 2 < argc; ++j) argv[j] = argv[j + 2];
      argc -= 2;
      return v;
    }
  }
  return {};
}

template <class T>
void WriteVec(const std::string& path, const std::vector<T>& v, const void* header = nullptr, std::size_t header_bytes = 0) {
  std::ofstream f(path, std::ios::binary);
  if (header_bytes) f.write(static_cast<const char*>(header), static_cast<std::streamsize>(header_bytes));
  f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(T)));
}

}  // namespace

int main(int argc, char** argv) {
  std::setlocale(LC_NUMERIC, "C");
  const std::string ids_path = TakeOption(argc, argv, "--dump-ids");
  const std::string out_dir = TakeOption(argc, argv, "--dump-out");
  const std::string pos_arg = TakeOption(argc, argv, "--dump-positions");
  if (ids_path.empty() || out_dir.empty() || pos_arg.empty()) {
    std::fprintf(stderr, "dspark_dump: --dump-ids FILE --dump-out DIR --dump-positions P,P,... are required\n");
    return 2;
  }
  std::vector<int> positions;
  {
    std::string t;
    for (const char ch : pos_arg + ",") {
      if (ch == ',') {
        if (!t.empty()) positions.push_back(std::atoi(t.c_str()));
        t.clear();
      } else {
        t.push_back(ch);
      }
    }
    std::sort(positions.begin(), positions.end());
    positions.erase(std::unique(positions.begin(), positions.end()), positions.end());
  }
  std::vector<llama_token> ids;
  {
    std::ifstream f(ids_path, std::ios::binary | std::ios::ate);
    if (!f) { std::fprintf(stderr, "cannot read %s\n", ids_path.c_str()); return 2; }
    const auto size = static_cast<std::size_t>(f.tellg());
    f.seekg(0);
    ids.resize(size / sizeof(std::int32_t));
    f.read(reinterpret_cast<char*>(ids.data()), static_cast<std::streamsize>(ids.size() * sizeof(std::int32_t)));
  }

  common_params params;
  common_init();
  if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) return 1;
  const auto output_limits = common_speculative_get_output_limits(params.n_batch, params.n_parallel, common_speculative_n_max(&params.speculative));
  params.n_outputs_max = output_limits.total;
  params.n_outputs_max_per_seq = output_limits.per_seq;

  llama_backend_init();
  llama_numa_init(params.numa);
  auto llama_init_tgt = common_init_from_params(params);
  llama_model* model_tgt = llama_init_tgt->model();
  llama_context* ctx_tgt = llama_init_tgt->context();
  common_speculative_init_result_ptr spec_init;
  {
    common_params params_dft = common_base_params_to_speculative(params);
    spec_init = common_speculative_init_from_params(params_dft, model_tgt, ctx_tgt);
    params.speculative.draft.ctx_tgt = ctx_tgt;
    params.speculative.draft.ctx_dft = spec_init->context();
  }
  llama_context* ctx_dft = params.speculative.draft.ctx_dft;
  if (ctx_dft == nullptr) { std::fprintf(stderr, "no draft context (is -md a DSpark model and --spec-type draft-dspark set?)\n"); return 1; }
  common_speculative* spec = common_speculative_init(params.speculative, 1);
  if (spec == nullptr) { std::fprintf(stderr, "failed to initialise the speculative implementation\n"); return 1; }
  const llama_model* model_dft = llama_get_model(ctx_dft);
  const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model_dft));
  const int n_embd = llama_model_n_embd(model_tgt);
  const int32_t* layer_ids = llama_model_target_layer_ids(model_dft);
  const uint32_t n_layer_ids = llama_model_target_layer_ids_n(model_dft);
  if (n_layer_ids != 4) { std::fprintf(stderr, "expected 4 target layers, the draft model has %u\n", n_layer_ids); return 1; }
  std::printf("dspark_dump: %zu ids, %zu probes, n_vocab %d, target layers %d %d %d %d\n", ids.size(), positions.size(), n_vocab, layer_ids[0], layer_ids[1], layer_ids[2], layer_ids[3]);
  std::string mk = "mkdir -p '" + out_dir + "'";
  if (std::system(mk.c_str()) != 0) return 1;

  const llama_seq_id seq_id = 0;
  std::vector<float> feats;  // [n][4][n_embd]
  std::ofstream probes(out_dir + "/probes.txt");
  std::size_t cursor = 0;
  bool begun = false;
  for (std::size_t pi = 0; pi < positions.size(); ++pi) {
    const std::size_t p = static_cast<std::size_t>(positions[pi]);
    if (p >= ids.size()) { std::fprintf(stderr, "position %zu is outside the ids (%zu)\n", p, ids.size()); return 1; }
    // 1. teacher-force the tokens [cursor, p) through the target and feed them to the drafter, in ubatch-sized pieces
    const std::size_t step = std::max<std::size_t>(1, std::min<std::size_t>(llama_n_ubatch(ctx_tgt), 512));
    while (cursor < p) {
      const std::size_t n = std::min(step, p - cursor);
      common_batch batch(ctx_tgt);
      for (std::size_t i = 0; i < n; ++i) batch.add(ids[cursor + i], static_cast<llama_pos>(cursor + i), seq_id, false);
      if (llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) { std::fprintf(stderr, "target decode failed at %zu\n", cursor); return 1; }
      // the four taps of these n positions, as pr390's drafter reads them
      feats.resize((cursor + n) * 4 * static_cast<std::size_t>(n_embd));
      for (uint32_t t = 0; t < 4; ++t) {
        const float* rows = llama_get_embeddings_layer_inp(ctx_tgt, static_cast<uint32_t>(layer_ids[t]));
        if (rows == nullptr) { std::fprintf(stderr, "layer %d input was not extracted\n", layer_ids[t]); return 1; }
        for (std::size_t i = 0; i < n; ++i)
          std::memcpy(feats.data() + ((cursor + i) * 4 + t) * static_cast<std::size_t>(n_embd), rows + i * static_cast<std::size_t>(n_embd), static_cast<std::size_t>(n_embd) * 4);
      }
      if (!common_speculative_process(spec, batch)) { std::fprintf(stderr, "speculative process failed at %zu\n", cursor); return 1; }
      cursor += n;
    }
    llama_tokens prompt_tgt(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(p));
    if (!begun) {
      common_speculative_begin(spec, seq_id, prompt_tgt);
      begun = true;
    }
    // 2. drafts of 3 and 7 from the committed context [0, p) with the anchor ids[p]
    for (const int k : {3, 7}) {
      llama_tokens draft;
      common_speculative_get_draft_params(spec, seq_id) = {
          /* .drafting   = */ true,
          /* .n_max      = */ k,
          /* .pos0       = */ static_cast<llama_pos>(p),
          /* .id_last    = */ ids[p],
          /* .prompt     = */ &prompt_tgt,
          /* .result     = */ &draft,
      };
      common_speculative_draft(spec);
      probes << p << ' ' << k;
      for (const auto t : draft) probes << ' ' << t;
      probes << '\n';
      std::printf("  pos %zu k %d:", p, k);
      for (const auto t : draft) std::printf(" %d", t);
      std::printf("\n");
      if (k == 7) {
        // the block was the last decode on the draft context: rows 1..7 hold the final logits
        std::vector<float> logits;
        bool ok = true;
        for (int row = 1; row < 8 && ok; ++row) {
          const float* l = llama_get_logits_ith(ctx_dft, row);
          if (l == nullptr) { ok = false; break; }
          logits.insert(logits.end(), l, l + n_vocab);
        }
        if (ok) {
          const std::int32_t nv = n_vocab;
          WriteVec(out_dir + "/logits_p" + std::to_string(p) + ".bin", logits, &nv, sizeof nv);
        } else {
          std::printf("    (no logits for the block rows)\n");
        }
        std::vector<float> hidden;
        for (int row = 0; row < 8; ++row) {
          const float* h = llama_get_embeddings_ith(ctx_dft, row);
          if (h == nullptr) { hidden.clear(); break; }
          hidden.insert(hidden.end(), h, h + n_embd);
        }
        if (!hidden.empty()) {
          const std::int32_t ne = n_embd;
          WriteVec(out_dir + "/hidden_p" + std::to_string(p) + ".bin", hidden, &ne, sizeof ne);
        }
      }
      // forget the block rows the draft left in the draft context (the committed positions stay)
      llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, static_cast<llama_pos>(p), -1);
    }
  }
  // 3. the files
  {
    WriteVec(out_dir + "/ids.bin", ids);
    std::ofstream f(out_dir + "/feats.bin", std::ios::binary);
    f.write("DSF1", 4);
    const std::int32_t hdr[3] = {static_cast<std::int32_t>(cursor), 4, n_embd};
    f.write(reinterpret_cast<const char*>(hdr), sizeof hdr);
    f.write(reinterpret_cast<const char*>(feats.data()), static_cast<std::streamsize>(feats.size() * 4));
  }
  std::printf("dspark_dump: wrote %s (%zu positions of features)\n", out_dir.c_str(), cursor);
  common_speculative_free(spec);
  llama_backend_free();
  return 0;
}
