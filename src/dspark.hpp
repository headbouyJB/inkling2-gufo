// DSpark block drafter (`dflash` arch) — weights and config only.
//
// The drafter is a separate small artifact: 3 dense layers (block size 8, non-causal inside the
// block) plus a recurrent "markov" head that turns the block into chained draft tokens. It reads
// the *target's* embedding table and output matrix, so it is bound next to an Inkling2Weights and
// never on its own. See notes/B-forward-spec.md section 4 for the graph and the file's provenance.
//
// This file binds and validates; the forward pass lives in dspark_forward.cpp so a load problem
// never looks like a numerics problem.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>

#include <gufo/ext/gguf_reader.hpp>

#include "src/tensor_ref.hpp"

namespace gufo::models::inkling2 {

/// One drafter layer, in the shape the graph needs (ggml ne order: ne[0] is the input dim).
struct DsparkLayerWeights {
  TensorRef attn_norm;        // (n_embd)                    F32
  TensorRef attn_q;           // (n_embd, 3072)              Q8_0
  TensorRef attn_k;           // (n_embd, 1024)              Q8_0
  TensorRef attn_v;           // (n_embd, 1024)              Q8_0
  TensorRef attn_output;      // (3072, n_embd)              Q8_0
  TensorRef attn_q_norm;      // (head_dim)                  F32
  TensorRef attn_k_norm;      // (head_dim)                  F32
  TensorRef attn_k_ctx;       // (n_embd, 1024)  context injection
  TensorRef attn_v_ctx;       // (n_embd, 1024)
  TensorRef attn_k_ctx_norm;  // (head_dim)
  TensorRef dflare_norm;      // (n_embd)  RMS over the fused feature
  TensorRef ffn_norm;         // (n_embd)
  TensorRef ffn_gate;         // (n_embd, n_ff)
  TensorRef ffn_up;           // (n_embd, n_ff)
  TensorRef ffn_down;         // (n_ff, n_embd)
};

struct DsparkConfig {
  static constexpr int kNLayer = 3;
  static constexpr int kNTap = 4;   // the target visits the drafter sees
  static constexpr int kMaxBlock = 8;

  std::uint32_t n_embd = 0;   // 2304
  std::uint32_t n_ff = 0;     // 9216
  std::uint32_t n_head = 0;   // 24
  std::uint32_t n_kv = 0;     // 8
  std::uint32_t head_dim = 0; // 128
  std::uint32_t block_size = 0;   // 8 rows per draft block
  std::uint32_t markov_rank = 0;  // 256
  std::uint32_t n_vocab = 0;      // 201024 padded
  std::uint32_t n_vocab_text = 0; // 200058 real
  /// Target visit whose *input* hidden state is tapped; the last entry is the index just past the
  /// final visit, i.e. the post-output_norm hidden.
  std::array<std::uint32_t, kNTap> tap_visit{};
  std::array<bool, kNLayer> swa{};  // sliding_window_pattern [1,0,1]
  float rope_theta = 0.f;
  float softcap = 0.f;
  float rms_eps = 0.f;
  bool has_confidence_head = false;
  bool sample_from_anchor = false;

  [[nodiscard]] std::uint32_t q_width() const { return n_head * head_dim; }
  [[nodiscard]] std::uint32_t kv_width() const { return n_kv * head_dim; }
  /// Draft tokens one block can produce: row 0 is the anchor, not a prediction slot.
  [[nodiscard]] std::uint32_t max_draft() const {
    return sample_from_anchor ? block_size : block_size - 1;
  }
};

struct DsparkWeights {
  DsparkConfig config;
  std::array<DsparkLayerWeights, DsparkConfig::kNLayer> layers;
  TensorRef fuse;         // dspark_fuse.weight     (4, 4)   F32, rows 0-2 per layer, row 3 the head
  TensorRef mask_embd;    // mask_embd.weight       (n_embd) F32, rows 1..7 of a block
  TensorRef dspark_norm;  // dspark_norm.weight     (n_embd) F32, before the target lm_head
  TensorRef output_norm;  // output_norm.weight     (n_embd) F32
  TensorRef markov_w1;    // markov_w1.weight       (n_vocab, markov_rank)  token -> rank
  TensorRef joint_proj;   // joint_proj.weight      (4 * (rank + n_embd) / .., 3 * rank)
  TensorRef joint_bias;   // joint_proj.bias        (3 * markov_rank) F32
  TensorRef hidden_out;   // dspark_hidden_out.weight (markov_rank, n_embd)
  TensorRef hidden_bias;  // dspark_hidden_out.bias   (n_embd) F32

  /// Binds every tensor of a `dflash` artifact and checks it against the config it declares.
  /// Refuses unknown or unclaimed tensors, so a converter change cannot pass silently.
  [[nodiscard]] static std::shared_ptr<const DsparkWeights> Bind(
      const core::GgufReader& reader, std::string* error);

  /// True when the artifact can drive this target: same hidden size, same vocab, same head layout.
  [[nodiscard]] bool MatchesTarget(std::uint32_t n_embd, std::uint32_t n_vocab_text) const {
    return config.n_embd == n_embd && config.n_vocab_text == n_vocab_text;
  }
};

/// One-line-per-item human summary, for the bench and the serve banner.
[[nodiscard]] std::string DescribeDspark(const DsparkWeights& w);

}  // namespace gufo::models::inkling2
