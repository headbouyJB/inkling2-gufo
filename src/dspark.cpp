#include "src/dspark.hpp"

#include <algorithm>
#include <sstream>
#include <string_view>
#include <unordered_set>
#include <variant>
#include <vector>

namespace gufo::models::inkling2 {
namespace {

using Dims = std::vector<std::uint64_t>;

std::string ShapeText(const Dims& ne) {
  std::string text = "[";
  for (std::size_t i = 0; i < ne.size(); ++i) {
    if (i != 0) text += ", ";
    text += std::to_string(ne[i]);
  }
  return text + "]";
}

/// Binds the drafter's 54 tensors, refusing anything it does not recognise.
class Binder {
 public:
  explicit Binder(const core::GgufReader& reader) : reader_(reader) {}

  [[nodiscard]] const std::string& why() const { return why_; }

  bool Take(TensorRef& out, const std::string& name, const Dims& ne) {
    const auto* info = reader_.FindTensor(name);
    if (info == nullptr) return Fail("missing tensor " + name);
    if (info->dimensions != ne) {
      return Fail(name + " has shape " + ShapeText(info->dimensions) + ", expected " +
                  ShapeText(ne));
    }
    if (!IsSupportedType(info->type)) return Fail(name + " has an unsupported type");
    out.type = info->type;
    out.data = info->data;
    out.size_bytes = info->size_bytes;
    out.n_dims = static_cast<std::uint32_t>(ne.size());
    std::ranges::copy(ne, out.ne.begin());
    claimed_.insert(info->name);
    return true;
  }

  bool CheckAllClaimed() {
    for (const auto& tensor : reader_.GetTensors())
      if (!claimed_.contains(tensor.name)) return Fail("unexpected tensor " + std::string(tensor.name));
    return true;
  }

  bool Fail(std::string message) {
    if (why_.empty()) why_ = std::move(message);
    return false;
  }

 private:
  const core::GgufReader& reader_;
  std::unordered_set<std::string_view> claimed_;
  std::string why_;
};

bool U32(const core::GgufReader& r, std::string_view key, std::uint32_t* out, std::string* why) {
  const auto v = r.GetMetadataUint32(key);
  if (!v) {
    *why = "dspark: metadata " + std::string(key) + " missing";
    return false;
  }
  *out = *v;
  return true;
}

bool F32(const core::GgufReader& r, std::string_view key, float* out, std::string* why) {
  const auto v = r.GetMetadataFloat32(key);
  if (!v) {
    *why = "dspark: metadata " + std::string(key) + " missing";
    return false;
  }
  *out = *v;
  return true;
}

/// Reads a numeric array of exactly `n` entries into `out`. The reader keeps integer arrays as
/// uint64 or int64 vectors depending on the declaration width (target_layers is uint32, the
/// sliding-window pattern is uint8 in the file), so all three element types are accepted.
template <typename T>
bool NumArray(const core::GgufReader& r, std::string_view key, T* out, std::size_t n) {
  const core::GgufMetadataValue* v = r.FindMetadata(key);
  if (v == nullptr) return false;
  auto take = [&](const auto& vec) {
    if (vec.size() != n) return false;
    for (std::size_t i = 0; i < n; ++i) out[i] = static_cast<T>(vec[i]);
    return true;
  };
  if (const auto* p = std::get_if<std::vector<std::uint64_t>>(&v->value)) return take(*p);
  if (const auto* p = std::get_if<std::vector<std::int64_t>>(&v->value)) return take(*p);
  if (const auto* p = std::get_if<std::vector<double>>(&v->value)) return take(*p);
  return false;
}

}  // namespace

std::shared_ptr<const DsparkWeights> DsparkWeights::Bind(const core::GgufReader& reader,
                                                         std::string* error) {
  auto fail = [&](std::string why) -> std::shared_ptr<const DsparkWeights> {
    if (error) *error = why;
    return nullptr;
  };
  const auto arch = reader.GetMetadataString("general.architecture");
  if (!arch || *arch != "dflash")
    return fail("dspark: artifact architecture is not dflash");

  auto w = std::make_shared<DsparkWeights>();
  DsparkConfig& c = w->config;
  std::string why;
  std::uint32_t n_layer = 0;
  if (!U32(reader, "dflash.block_count", &n_layer, &why)) return fail(why);
  if (n_layer != static_cast<std::uint32_t>(DsparkConfig::kNLayer))
    return fail("dspark: expected " + std::to_string(DsparkConfig::kNLayer) + " layers, file has " +
                std::to_string(n_layer));
  if (!U32(reader, "dflash.block_size", &c.block_size, &why)) return fail(why);
  if (!U32(reader, "dflash.embedding_length", &c.n_embd, &why)) return fail(why);
  if (!U32(reader, "dflash.feed_forward_length", &c.n_ff, &why)) return fail(why);
  if (!U32(reader, "dflash.attention.head_count", &c.n_head, &why)) return fail(why);
  if (!U32(reader, "dflash.attention.head_count_kv", &c.n_kv, &why)) return fail(why);
  if (!U32(reader, "dflash.attention.key_length", &c.head_dim, &why)) return fail(why);
  if (!U32(reader, "dflash.dspark.markov_rank", &c.markov_rank, &why)) return fail(why);
  if (!U32(reader, "dflash.vocab_size", &c.n_vocab, &why)) return fail(why);
  if (!U32(reader, "dflash.vocab_size_unpadded", &c.n_vocab_text, &why)) return fail(why);
  if (!F32(reader, "dflash.rope.freq_base", &c.rope_theta, &why)) return fail(why);
  if (!F32(reader, "dflash.final_logit_softcapping", &c.softcap, &why)) return fail(why);
  if (!F32(reader, "dflash.attention.layer_norm_rms_epsilon", &c.rms_eps, &why)) return fail(why);
  if (const auto v = reader.GetMetadataBool("dflash.has_confidence_head")) c.has_confidence_head = *v;
  if (const auto v = reader.GetMetadataBool("dflash.sample_from_anchor")) c.sample_from_anchor = *v;
  if (const auto v = reader.GetMetadataBool("dflash.attention.causal")) {
    if (*v) return fail("dspark: the draft block must be non-causal inside the block");
  }
  if (c.has_confidence_head) return fail("dspark: a confidence head is not supported (file has one)");
  if (c.block_size != static_cast<std::uint32_t>(DsparkConfig::kMaxBlock))
    return fail("dspark: block_size " + std::to_string(c.block_size) + " is not 8");

  {
    std::uint32_t taps[DsparkConfig::kNTap] = {0, 0, 0, 0};
    if (!NumArray(reader, "dflash.target_layers", taps, DsparkConfig::kNTap))
      return fail("dspark: dflash.target_layers must hold 4 visit indices");
    for (int i = 0; i < DsparkConfig::kNTap; ++i) c.tap_visit[i] = taps[i];
    if (c.tap_visit[DsparkConfig::kNTap - 1] <= c.tap_visit[DsparkConfig::kNTap - 2])
      return fail("dspark: the last tap must be past the final visit");
  }
  {
    std::uint32_t pat[DsparkConfig::kNLayer] = {0, 0, 0};
    if (!NumArray(reader, "dflash.attention.sliding_window_pattern", pat, DsparkConfig::kNLayer))
      return fail("dspark: sliding_window_pattern must hold one flag per layer");
    for (int i = 0; i < DsparkConfig::kNLayer; ++i) c.swa[i] = pat[i] != 0;
  }

  Binder b(reader);
  const Dims emb{c.n_embd};
  const Dims q{c.n_embd, c.q_width()};
  const Dims kv{c.n_embd, c.kv_width()};
  const Dims o{c.q_width(), c.n_embd};
  const Dims hd{c.head_dim};
  const Dims ff{c.n_embd, c.n_ff};
  const Dims ffd{c.n_ff, c.n_embd};
  for (int l = 0; l < DsparkConfig::kNLayer; ++l) {
    const std::string p = "blk." + std::to_string(l) + ".";
    DsparkLayerWeights& L = w->layers[l];
    if (!b.Take(L.attn_norm, p + "attn_norm.weight", emb) ||
        !b.Take(L.attn_q, p + "attn_q.weight", q) ||
        !b.Take(L.attn_k, p + "attn_k.weight", kv) ||
        !b.Take(L.attn_v, p + "attn_v.weight", kv) ||
        !b.Take(L.attn_output, p + "attn_output.weight", o) ||
        !b.Take(L.attn_q_norm, p + "attn_q_norm.weight", hd) ||
        !b.Take(L.attn_k_norm, p + "attn_k_norm.weight", hd) ||
        !b.Take(L.attn_k_ctx, p + "attn_k_ctx.weight", kv) ||
        !b.Take(L.attn_v_ctx, p + "attn_v_ctx.weight", kv) ||
        !b.Take(L.attn_k_ctx_norm, p + "attn_k_ctx_norm.weight", hd) ||
        !b.Take(L.dflare_norm, p + "dflare_norm.weight", emb) ||
        !b.Take(L.ffn_norm, p + "ffn_norm.weight", emb) ||
        !b.Take(L.ffn_gate, p + "ffn_gate.weight", ff) ||
        !b.Take(L.ffn_up, p + "ffn_up.weight", ff) ||
        !b.Take(L.ffn_down, p + "ffn_down.weight", ffd)) {
      return fail(b.why());
    }
  }
  const Dims fuse{DsparkConfig::kNTap, DsparkConfig::kNTap};
  const Dims mw{c.markov_rank, c.n_vocab};
  const Dims jp{2 * c.n_embd + 2 * c.markov_rank, 3 * c.markov_rank};
  const Dims jb{3 * c.markov_rank};
  const Dims ho{c.markov_rank, c.n_embd};
  const Dims hb{c.n_embd};
  if (!b.Take(w->fuse, "dspark_fuse.weight", fuse) || !b.Take(w->mask_embd, "mask_embd.weight", emb) ||
      !b.Take(w->dspark_norm, "dspark_norm.weight", emb) ||
      !b.Take(w->output_norm, "output_norm.weight", emb) ||
      !b.Take(w->markov_w1, "markov_w1.weight", mw) ||
      !b.Take(w->joint_proj, "joint_proj.weight", jp) ||
      !b.Take(w->joint_bias, "joint_proj.bias", jb) ||
      !b.Take(w->hidden_out, "dspark_hidden_out.weight", ho) ||
      !b.Take(w->hidden_bias, "dspark_hidden_out.bias", hb)) {
    return fail(b.why());
  }
  if (!b.CheckAllClaimed()) return fail(b.why());
  return w;
}

std::string DescribeDspark(const DsparkWeights& w) {
  const DsparkConfig& c = w.config;
  std::ostringstream os;
  os << "dspark: layers=" << DsparkConfig::kNLayer << " block=" << c.block_size
     << " max_draft=" << c.max_draft() << " embd=" << c.n_embd << " ff=" << c.n_ff
     << " head=" << c.n_head << "/" << c.n_kv << "x" << c.head_dim << " rank=" << c.markov_rank
     << " vocab=" << c.n_vocab_text << "/" << c.n_vocab << " rope=" << c.rope_theta
     << " softcap=" << c.softcap << " eps=" << c.rms_eps
     << " swa=" << (c.swa[0] ? "1" : "0") << (c.swa[1] ? "1" : "0") << (c.swa[2] ? "1" : "0")
     << " taps=" << c.tap_visit[0] << "," << c.tap_visit[1] << "," << c.tap_visit[2] << ","
     << c.tap_visit[3] << " anchor=" << (c.sample_from_anchor ? 1 : 0);
  return os.str();
}

}  // namespace gufo::models::inkling2
