// LoRA-wrapped layer family — packed adapters, merged qkv/gate_up slices, the
// tensor-parallel slice_lora_a/slice_lora_b rules, embedding and logits LoRA.
//
// PORTED FROM (${VLLM_SOURCE} @ 555967922 / vLLM 0.26.0.dev0):
//   tests/lora/test_layers.py:116-142   get_random_id_to_index
//   tests/lora/test_layers.py:145-199   populate_loras (pack + per-slice optimize)
//   tests/lora/test_layers.py:202-244   create_random_inputs
//   tests/lora/test_layers.py:269-364   test_embeddings
//   tests/lora/test_layers.py:372-485   test_lm_head_logits_processor
//   tests/lora/test_layers.py:490-510   test_lm_head_logits_processor_invalid_vocab_size
//   tests/lora/test_layers.py:516-620   test_linear_replicated
//   tests/lora/test_layers.py:629-750   test_linear_parallel (row/column x fully_shard)
//   tests/lora/test_layers.py:762-917   test_column_parallel_packed (repeats 1/2/3)
//   tests/lora/utils.py:29-46           DummyLoRAManager.init_random_lora
//                                       (rank 8, alpha 1 => scaling 1/8, uniform [0,1))
//
// HARNESS ADAPTATIONS (documented per the port discipline):
//  * Upstream builds a real `ReplicatedLinear`/`QKVParallelLinear` and compares
//    `lora_linear(x)` against `linear(x) + delta`. Our wrapped layer is a DELTA
//    APPLIER over a base-linear output (the base GEMM stays behind
//    `LinearMethodBase`, exactly like `_apply_lora_to_output` receives an
//    already-computed `output`), so the base output is drawn directly instead of
//    being recomputed from a base weight. The delta — everything the layer owns —
//    is checked against an independent double-precision per-adapter reference.
//  * `torch.rand` under `set_random_seed(i)` becomes a deterministic LCG; the
//    role (reproducible uniform [0,1) inputs across NUM_RANDOM_SEEDS=2 seeds) is
//    preserved.
//  * Tolerances are upstream's `TOLERANCES[torch.float32] = (5e-3, 5e-3)`
//    (test_layers.py:53-57) applied as torch.testing.assert_close's
//    |a-b| <= atol + rtol*|b|. Our portable path stores float32, so the float32
//    row is the applicable one.
//  * The tensor-parallel cases run the slicing rules directly at tp_size 2/4
//    rather than under a real process group: `slice_lora_a`/`slice_lora_b` are
//    pure functions of (tp_rank, tp_size) and upstream's own tests only ever
//    exercise them at tp_size 1.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "vllm/lora/layers.h"
#include "vllm/lora/lora_weights.h"
#include "vllm/lora/punica.h"

using vllm::lora::ColumnParallelLinearWithLoRA;
using vllm::lora::ColumnParallelLinearWithShardedLoRA;
using vllm::lora::LogitsProcessorWithLoRA;
using vllm::lora::LoRALayerWeights;
using vllm::lora::LoRAMat;
using vllm::lora::MatList;
using vllm::lora::MergedColumnParallelLinearWithLoRA;
using vllm::lora::MergedColumnParallelLinearWithShardedLoRA;
using vllm::lora::MergedQKVParallelLinearWithLoRA;
using vllm::lora::MergedQKVParallelLinearWithShardedLoRA;
using vllm::lora::PackedLoRALayerWeights;
using vllm::lora::QKVParallelLinearWithLoRA;
using vllm::lora::ReplicatedLinearWithLoRA;
using vllm::lora::RowParallelLinearWithLoRA;
using vllm::lora::RowParallelLinearWithShardedLoRA;
using vllm::lora::VocabParallelEmbeddingWithLoRA;

namespace {

// ---------------------------------------------------------------------------
// Harness

// Deterministic uniform [0,1). Stands in for torch.rand under set_random_seed.
class Rng {
 public:
  explicit Rng(uint32_t seed) : s_(seed * 2654435761u + 12345u) {}
  float Next() {
    s_ = s_ * 1664525u + 1013904223u;
    return static_cast<float>(s_ >> 8) / 16777216.0f;
  }
  std::vector<float> Vec(int64_t n) {
    std::vector<float> v(static_cast<size_t>(n));
    for (auto& e : v) e = Next();
    return v;
  }
  int64_t Int(int64_t lo, int64_t hi) {  // [lo, hi)
    return lo + static_cast<int64_t>(Next() * static_cast<float>(hi - lo));
  }

 private:
  uint32_t s_;
};

LoRAMat RandMat(Rng& r, int64_t rows, int64_t cols) {
  LoRAMat m;
  m.rows = rows;
  m.cols = cols;
  m.data = r.Vec(rows * cols);
  return m;
}

// TOLERANCES[torch.float32] (test_layers.py:53-57) under torch.testing's
// |got - ref| <= atol + rtol * |ref|.
constexpr double kRtol = 5e-3;
constexpr double kAtol = 5e-3;

// One CHECK for the whole tensor (a per-element CHECK would emit millions of
// assertions); reports the worst violation so a failure is diagnosable.
void CheckAllClose(const std::vector<float>& got, const std::vector<double>& ref) {
  REQUIRE(got.size() == ref.size());
  double worst = 0.0;
  size_t worst_i = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double slack =
        std::abs(static_cast<double>(got[i]) - ref[i]) - (kAtol + kRtol * std::abs(ref[i]));
    if (slack > worst) {
      worst = slack;
      worst_i = i;
    }
  }
  if (worst > 0.0) {
    MESSAGE("worst element " << worst_i << ": got " << got[worst_i] << " ref "
                             << ref[worst_i]);
  }
  CHECK(worst <= 0.0);
}

// get_random_id_to_index (test_layers.py:116-142). Slot -> lora id (1-based),
// 0 == free slot (upstream's None).
std::vector<int> GetRandomIdToIndex(int num_loras, int num_slots, Rng& rng) {
  REQUIRE(num_loras <= num_slots);
  std::vector<int> slots(static_cast<size_t>(num_slots), 0);
  // Fisher-Yates over the slot indices == torch.randperm(num_slots)[:num_loras].
  std::vector<int> perm(static_cast<size_t>(num_slots));
  for (int i = 0; i < num_slots; ++i) perm[static_cast<size_t>(i)] = i;
  for (int i = num_slots - 1; i > 0; --i) {
    const int j = static_cast<int>(rng.Int(0, i + 1));
    std::swap(perm[static_cast<size_t>(i)], perm[static_cast<size_t>(j)]);
  }
  for (int lora_id = 1; lora_id <= num_loras; ++lora_id) {
    slots[static_cast<size_t>(perm[static_cast<size_t>(lora_id - 1)])] = lora_id;
  }
  return slots;
}

// One adapter as populate_loras builds it: `repeats` sub-LoRAs, each already
// optimize()d so its scaling is folded into lora_b (test_layers.py:174-186).
struct AdapterGroup {
  std::vector<LoRALayerWeights> subloras;
  MatList a;  // per sub-module lora_a  [rank, in]
  MatList b;  // per sub-module lora_b  [out_i, rank]
};

// DummyLoRAManager.init_random_lora (utils.py:29-46) + the populate_loras row
// slice + optimize(). `weight_rows`/`weight_cols` are the base layer weight
// shape upstream passes as `layer_weights`.
AdapterGroup MakeAdapterGroup(Rng& rng, int64_t weight_rows, int64_t weight_cols,
                              int repeats, int rank = 8) {
  AdapterGroup g;
  const int64_t sublora_len = weight_rows / repeats;
  for (int i = 0; i < repeats; ++i) {
    LoRAMat a = RandMat(rng, rank, weight_cols);
    LoRAMat full_b = RandMat(rng, weight_rows, rank);
    LoRAMat b;
    b.rows = sublora_len;
    b.cols = rank;
    b.data.assign(full_b.data.begin() + static_cast<long>(sublora_len * i * rank),
                  full_b.data.begin() + static_cast<long>(sublora_len * (i + 1) * rank));
    LoRALayerWeights w("fake_" + std::to_string(i), rank, /*alpha=*/1, a.data, b.data,
                       static_cast<int>(weight_cols), static_cast<int>(sublora_len));
    w.Optimize();  // scaling 1/rank folded into lora_b => scaling == 1
    g.subloras.push_back(w);
  }
  for (const auto& w : g.subloras) {
    LoRAMat a;
    a.rows = w.rank;
    a.cols = w.input_dim;
    a.data = w.lora_a;
    LoRAMat b;
    b.rows = w.output_dim;
    b.cols = w.rank;
    b.data = w.lora_b;
    g.a.push_back(a);
    g.b.push_back(b);
  }
  return g;
}

// create_random_inputs (test_layers.py:202-244): num_inputs rows, each assigned
// a random active lora id; index_mapping repeats the id per token.
struct Batch {
  std::vector<float> x;         // [T, input_size]
  std::vector<int32_t> slots;   // [T] slot index, -1 == no adapter
  std::vector<int> lora_ids;    // [T] lora id (0 == base model)
  int64_t T = 0;
};

Batch MakeBatch(Rng& rng, const std::vector<int>& active_lora_ids,
                const std::vector<int>& id_to_index, int64_t num_inputs,
                int64_t input_size) {
  Batch batch;
  batch.T = num_inputs;
  batch.x = rng.Vec(num_inputs * input_size);
  for (int64_t t = 0; t < num_inputs; ++t) {
    const int lora_id =
        active_lora_ids[static_cast<size_t>(rng.Int(0, static_cast<int64_t>(active_lora_ids.size())))];
    batch.lora_ids.push_back(lora_id);
    int32_t slot = -1;  // convert_mapping: id <= 0 -> -1 (utils.py:104-110)
    for (size_t s = 0; s < id_to_index.size(); ++s) {
      if (lora_id > 0 && id_to_index[s] == lora_id) slot = static_cast<int32_t>(s);
    }
    batch.slots.push_back(slot);
  }
  return batch;
}

// Independent double reference for one token/slice:
//   y[off : off+out_i] += x[t] @ a_i^T @ b_i^T * scaling
void RefAddSlice(std::vector<double>& ref, int64_t t, int64_t y_width, int64_t offset,
                 const std::vector<float>& x, int64_t in_dim, const LoRAMat& a,
                 const LoRAMat& b, double scaling) {
  std::vector<double> shrunk(static_cast<size_t>(a.rows), 0.0);
  for (int64_t r = 0; r < a.rows; ++r) {
    double acc = 0.0;
    for (int64_t i = 0; i < in_dim; ++i) {
      acc += static_cast<double>(x[static_cast<size_t>(t * in_dim + i)]) *
             static_cast<double>(a.data[static_cast<size_t>(r * a.cols + i)]);
    }
    shrunk[static_cast<size_t>(r)] = acc * scaling;
  }
  for (int64_t o = 0; o < b.rows; ++o) {
    double acc = 0.0;
    for (int64_t r = 0; r < b.cols; ++r) {
      acc += shrunk[static_cast<size_t>(r)] *
             static_cast<double>(b.data[static_cast<size_t>(o * b.cols + r)]);
    }
    ref[static_cast<size_t>(t * y_width + offset + o)] += acc;
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// PackedLoRALayerWeights (lora_weights.py:99-282)

TEST_CASE("PackedLoRALayerWeights::Pack folds scaling and keeps per-slice weights") {
  // lora_weights.py:126-152 — pack() optimize()s every sub-LoRA, takes rank and
  // module_name from the first present one, and sets every present slice's
  // scaling to 1 (absent slices stay absent).
  LoRALayerWeights q("qkv_proj", /*rank=*/2, /*alpha=*/4, /*a=*/{1.0f, 2.0f, 3.0f, 4.0f},
                     /*b=*/{1.0f, 1.0f}, /*in=*/2, /*out=*/1);  // scaling 2.0
  LoRALayerWeights v("qkv_proj", /*rank=*/2, /*alpha=*/2, /*a=*/{5.0f, 6.0f, 7.0f, 8.0f},
                     /*b=*/{2.0f, 3.0f}, /*in=*/2, /*out=*/1);  // scaling 1.0

  std::vector<std::optional<LoRALayerWeights>> loras = {q, std::nullopt, v};
  PackedLoRALayerWeights packed = PackedLoRALayerWeights::Pack(loras);

  CHECK(packed.is_packed());
  CHECK(packed.rank == 2);
  CHECK(packed.module_name == "qkv_proj");
  REQUIRE(packed.subloras.size() == 3);
  CHECK(packed.subloras[0].has_value());
  CHECK_FALSE(packed.subloras[1].has_value());  // the "None" sub-module
  CHECK(packed.subloras[2].has_value());

  // q's scaling (alpha/rank = 2) folded into lora_b; scaling now 1.
  CHECK(packed.subloras[0]->scaling == doctest::Approx(1.0));
  CHECK(packed.subloras[0]->lora_b[0] == doctest::Approx(2.0f));
  CHECK(packed.subloras[0]->lora_b[1] == doctest::Approx(2.0f));
  // v's scaling was already 1 => optimize() is a no-op.
  CHECK(packed.subloras[2]->lora_b[0] == doctest::Approx(2.0f));
  CHECK(packed.subloras[2]->lora_b[1] == doctest::Approx(3.0f));

  // Packed::Optimize (lora_weights.py:263-270) is idempotent after Pack.
  auto before = packed.subloras[0]->lora_b;
  packed.Optimize();
  CHECK(packed.subloras[0]->lora_b == before);
}

// ---------------------------------------------------------------------------
// test_linear_replicated (test_layers.py:516-620)

TEST_CASE("test_linear_replicated: batched apply then reset to base") {
  constexpr int64_t kInput = 512;
  constexpr int64_t kOutput = 512;
  constexpr int64_t kMaxLoras = 8;
  constexpr int64_t kMaxRank = 8;

  for (int num_loras : {1, 2, 4}) {
    for (int seed = 0; seed < 2; ++seed) {  // NUM_RANDOM_SEEDS
      Rng rng(static_cast<uint32_t>(seed * 31 + num_loras));
      auto id_to_index = GetRandomIdToIndex(num_loras, static_cast<int>(kMaxLoras), rng);

      ReplicatedLinearWithLoRA layer(kInput, kOutput);
      layer.CreateLoraWeights(kMaxLoras, kMaxRank, /*fully_sharded_loras=*/false);
      REQUIRE(layer.n_slices() == 1);
      REQUIRE(layer.output_slices().size() == 1);

      std::vector<AdapterGroup> groups;
      std::vector<int> active_ids;
      for (size_t slot = 0; slot < id_to_index.size(); ++slot) {
        if (id_to_index[slot] == 0) continue;
        AdapterGroup g = MakeAdapterGroup(rng, kOutput, kInput, /*repeats=*/1);
        layer.SetLora(static_cast<int64_t>(slot), g.a, g.b);
        groups.push_back(g);
        active_ids.push_back(id_to_index[slot]);
      }

      Batch batch = MakeBatch(rng, active_ids, id_to_index, 32 * num_loras, kInput);
      std::vector<float> base = rng.Vec(batch.T * kOutput);
      std::vector<float> y = base;
      layer.ApplyLoraToOutput(y.data(), batch.x.data(), batch.T, batch.slots.data());

      std::vector<double> ref(base.begin(), base.end());
      for (int64_t t = 0; t < batch.T; ++t) {
        const int lora_id = batch.lora_ids[static_cast<size_t>(t)];
        if (lora_id <= 0) continue;
        size_t gi = 0;
        for (size_t k = 0; k < active_ids.size(); ++k) {
          if (active_ids[k] == lora_id) gi = k;
        }
        const AdapterGroup& g = groups[gi];
        RefAddSlice(ref, t, kOutput, 0, batch.x, kInput, g.a[0], g.b[0],
                    g.subloras[0].scaling);
      }
      CheckAllClose(y, ref);

      // "Check that resetting the lora weights succeeds" (test_layers.py:596-620).
      for (int64_t slot = 0; slot < kMaxLoras; ++slot) layer.ResetLora(slot);
      std::vector<float> y2 = base;
      layer.ApplyLoraToOutput(y2.data(), batch.x.data(), batch.T, batch.slots.data());
      CHECK(y2 == base);
    }
  }
}

// ---------------------------------------------------------------------------
// test_linear_parallel (test_layers.py:629-750)

TEST_CASE("test_linear_parallel: row and column orientation, fully_shard on/off") {
  constexpr int64_t kSize = 512;  // upstream 4096; the shape structure is what matters
  constexpr int64_t kMaxLoras = 8;
  constexpr int64_t kMaxRank = 8;

  for (const std::string orientation : {"row", "column"}) {
    for (bool fully_shard : {false, true}) {
      for (int num_loras : {1, 2, 4}) {
        Rng rng(static_cast<uint32_t>(num_loras * 7 + (fully_shard ? 3 : 0) +
                                      (orientation == "row" ? 100u : 200u)));
        auto id_to_index = GetRandomIdToIndex(num_loras, static_cast<int>(kMaxLoras), rng);

        std::unique_ptr<vllm::lora::BaseLinearLayerWithLoRA> layer;
        if (orientation == "row") {
          if (fully_shard) {
            layer = std::make_unique<RowParallelLinearWithShardedLoRA>(kSize, kSize, 1, 0);
          } else {
            layer = std::make_unique<RowParallelLinearWithLoRA>(kSize, kSize, 1, 0);
          }
        } else {
          if (fully_shard) {
            layer = std::make_unique<ColumnParallelLinearWithShardedLoRA>(kSize, kSize, 1, 0);
          } else {
            layer = std::make_unique<ColumnParallelLinearWithLoRA>(kSize, kSize, 1, 0);
          }
        }
        layer->CreateLoraWeights(kMaxLoras, kMaxRank, fully_shard);
        // test_layers.py:677-682 — n_slices == len(a_stacked) == len(b_stacked) == 1
        REQUIRE(layer->n_slices() == 1);

        std::vector<AdapterGroup> groups;
        std::vector<int> active_ids;
        for (size_t slot = 0; slot < id_to_index.size(); ++slot) {
          if (id_to_index[slot] == 0) continue;
          AdapterGroup g = MakeAdapterGroup(rng, kSize, kSize, /*repeats=*/1);
          layer->SetLora(static_cast<int64_t>(slot), g.a, g.b);
          groups.push_back(g);
          active_ids.push_back(id_to_index[slot]);
        }

        Batch batch = MakeBatch(rng, active_ids, id_to_index, 32 * num_loras, kSize);
        std::vector<float> base = rng.Vec(batch.T * kSize);
        std::vector<float> y = base;
        layer->ApplyLoraToOutput(y.data(), batch.x.data(), batch.T, batch.slots.data());

        std::vector<double> ref(base.begin(), base.end());
        for (int64_t t = 0; t < batch.T; ++t) {
          const int lora_id = batch.lora_ids[static_cast<size_t>(t)];
          if (lora_id <= 0) continue;
          size_t gi = 0;
          for (size_t k = 0; k < active_ids.size(); ++k) {
            if (active_ids[k] == lora_id) gi = k;
          }
          RefAddSlice(ref, t, kSize, 0, batch.x, kSize, groups[gi].a[0], groups[gi].b[0],
                      groups[gi].subloras[0].scaling);
        }
        CheckAllClose(y, ref);

        for (int64_t slot = 0; slot < kMaxLoras; ++slot) layer->ResetLora(slot);
        std::vector<float> y2 = base;
        layer->ApplyLoraToOutput(y2.data(), batch.x.data(), batch.T, batch.slots.data());
        CHECK(y2 == base);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// test_column_parallel_packed (test_layers.py:762-917)

TEST_CASE("test_column_parallel_packed: repeats 1 (qkv), 2 (gate_up), 3 (merged qkv)") {
  constexpr int64_t kInput = 512;
  constexpr int64_t kMaxLoras = 8;
  constexpr int64_t kMaxRank = 8;
  // QKVParallelLinear(4096, head_size=64, total_num_heads=32) scaled down: the
  // three slices are q/k/v with head_size * num_heads each.
  constexpr int64_t kHeadSize = 16;
  constexpr int64_t kNumHeads = 8;

  for (int repeats : {1, 2, 3}) {
    for (int num_loras : {1, 2, 4}) {
      Rng rng(static_cast<uint32_t>(repeats * 101 + num_loras));
      auto id_to_index = GetRandomIdToIndex(num_loras, static_cast<int>(kMaxLoras), rng);

      std::unique_ptr<vllm::lora::BaseLinearLayerWithLoRA> layer;
      int64_t total_out = 0;
      if (repeats == 2) {
        // MergedColumnParallelLinear(input, [out] * 2) -> gate_up_proj.
        layer = std::make_unique<MergedColumnParallelLinearWithLoRA>(
            kInput, std::vector<int64_t>{kInput, kInput}, 1, 0);
        total_out = 2 * kInput;
      } else if (repeats == 3) {
        layer = std::make_unique<MergedQKVParallelLinearWithLoRA>(
            kInput, kHeadSize, kNumHeads, kNumHeads, kNumHeads, kNumHeads, 1, 0);
        total_out = 3 * kHeadSize * kNumHeads;
      } else {
        layer = std::make_unique<QKVParallelLinearWithLoRA>(
            kInput, kHeadSize, kNumHeads, kNumHeads, kNumHeads, kNumHeads, 1, 0);
        total_out = 3 * kHeadSize * kNumHeads;
      }
      layer->CreateLoraWeights(kMaxLoras, kMaxRank, /*fully_sharded_loras=*/false);
      REQUIRE(layer->n_slices() == repeats);
      REQUIRE(static_cast<int>(layer->output_slices().size()) == repeats);

      std::vector<AdapterGroup> groups;
      std::vector<int> active_ids;
      for (size_t slot = 0; slot < id_to_index.size(); ++slot) {
        if (id_to_index[slot] == 0) continue;
        AdapterGroup g = MakeAdapterGroup(rng, total_out, kInput, repeats);
        layer->SetLora(static_cast<int64_t>(slot), g.a, g.b);
        groups.push_back(g);
        active_ids.push_back(id_to_index[slot]);
      }

      Batch batch = MakeBatch(rng, active_ids, id_to_index, 32 * num_loras, kInput);
      std::vector<float> base = rng.Vec(batch.T * total_out);
      std::vector<float> y = base;
      layer->ApplyLoraToOutput(y.data(), batch.x.data(), batch.T, batch.slots.data());

      // test_layers.py:889-897 — each sublora writes its own output window.
      std::vector<double> ref(base.begin(), base.end());
      for (int64_t t = 0; t < batch.T; ++t) {
        const int lora_id = batch.lora_ids[static_cast<size_t>(t)];
        if (lora_id <= 0) continue;
        size_t gi = 0;
        for (size_t k = 0; k < active_ids.size(); ++k) {
          if (active_ids[k] == lora_id) gi = k;
        }
        const AdapterGroup& g = groups[gi];
        for (int i = 0; i < repeats; ++i) {
          const int64_t offset = g.b[static_cast<size_t>(i)].rows * i;
          RefAddSlice(ref, t, total_out, offset, batch.x, kInput, g.a[static_cast<size_t>(i)],
                      g.b[static_cast<size_t>(i)],
                      g.subloras[static_cast<size_t>(i)].scaling);
        }
      }
      CheckAllClose(y, ref);

      for (int64_t slot = 0; slot < kMaxLoras; ++slot) layer->ResetLora(slot);
      std::vector<float> y2 = base;
      layer->ApplyLoraToOutput(y2.data(), batch.x.data(), batch.T, batch.slots.data());
      CHECK(y2 == base);
    }
  }
}

// ---------------------------------------------------------------------------
// Tensor-parallel slicing (column_parallel_linear.py / row_parallel_linear.py)

TEST_CASE("slice_lora_a / slice_lora_b under tensor parallel") {
  constexpr int64_t kRank = 4;

  SUBCASE("RowParallelLinearWithLoRA slices lora_a along the input dim") {
    // row_parallel_linear.py:32-37 — a[:, rank*shard : (rank+1)*shard].
    Rng rng(1);
    const int64_t tp = 4, in_per_partition = 8;
    RowParallelLinearWithLoRA layer(in_per_partition, /*output_size=*/6, tp, /*tp_rank=*/2);
    LoRAMat a = RandMat(rng, kRank, in_per_partition * tp);
    MatList sliced = layer.SliceLoraA({a});
    REQUIRE(sliced.size() == 1);
    CHECK(sliced[0].rows == kRank);
    CHECK(sliced[0].cols == in_per_partition);
    for (int64_t r = 0; r < kRank; ++r) {
      for (int64_t c = 0; c < in_per_partition; ++c) {
        CHECK(sliced[0].At(r, c) == a.At(r, 2 * in_per_partition + c));
      }
    }
    // slice_lora_b is the identity for row-parallel (row_parallel_linear.py:39-40).
    LoRAMat b = RandMat(rng, 6, kRank);
    CHECK(layer.SliceLoraB({b})[0].data == b.data);
  }

  SUBCASE("ColumnParallelLinearWithLoRA slices lora_b along the output dim") {
    // column_parallel_linear.py:125-129.
    Rng rng(2);
    const int64_t tp = 4, out_per_partition = 5;
    ColumnParallelLinearWithLoRA layer(/*input_size=*/8, out_per_partition, tp, 3);
    LoRAMat b = RandMat(rng, out_per_partition * tp, kRank);
    MatList sliced = layer.SliceLoraB({b});
    CHECK(sliced[0].rows == out_per_partition);
    for (int64_t r = 0; r < out_per_partition; ++r) {
      for (int64_t c = 0; c < kRank; ++c) {
        CHECK(sliced[0].At(r, c) == b.At(3 * out_per_partition + r, c));
      }
    }
    // slice_lora_a is the identity (column_parallel_linear.py:104-105).
    LoRAMat a = RandMat(rng, kRank, 8);
    CHECK(layer.SliceLoraA({a})[0].data == a.data);
  }

  SUBCASE("ColumnParallelLinearWithLoRA takes both halves for a merged base layer") {
    // column_parallel_linear.py:110-122 — is_merged_col_linear concatenates the
    // rank's shard of the left half and of the right half.
    Rng rng(3);
    const int64_t tp = 2, out_per_partition = 6;  // shard_size = 3 per half
    ColumnParallelLinearWithLoRA layer(/*input_size=*/8, out_per_partition, tp, /*tp_rank=*/1,
                                       /*is_merged_col_linear=*/true);
    LoRAMat b = RandMat(rng, out_per_partition * tp, kRank);  // 12 rows: two halves of 6
    MatList sliced = layer.SliceLoraB({b});
    REQUIRE(sliced[0].rows == out_per_partition);
    const int64_t shard = out_per_partition / 2;  // 3
    for (int64_t r = 0; r < shard; ++r) {
      for (int64_t c = 0; c < kRank; ++c) {
        CHECK(sliced[0].At(r, c) == b.At(1 * shard + r, c));
        CHECK(sliced[0].At(shard + r, c) == b.At(6 + 1 * shard + r, c));
      }
    }
  }

  SUBCASE("QKVParallelLinearWithLoRA concatenates the q, k and v shards") {
    // column_parallel_linear.py:399-420.
    Rng rng(4);
    const int64_t tp = 2, head_size = 4, total_heads = 8, total_kv_heads = 8;
    const int64_t heads = total_heads / tp, kv_heads = total_kv_heads / tp;
    QKVParallelLinearWithLoRA layer(/*input_size=*/8, head_size, total_heads, heads,
                                    total_kv_heads, kv_heads, tp, /*tp_rank=*/1);
    const int64_t q_total = total_heads * head_size;      // 32
    const int64_t kv_total = total_kv_heads * head_size;  // 32
    LoRAMat b = RandMat(rng, q_total + 2 * kv_total, kRank);
    MatList sliced = layer.SliceLoraB({b});
    const int64_t q_shard = heads * head_size;      // 16
    const int64_t kv_shard = kv_heads * head_size;  // 16
    REQUIRE(sliced[0].rows == q_shard + 2 * kv_shard);
    for (int64_t r = 0; r < q_shard; ++r) {
      CHECK(sliced[0].At(r, 0) == b.At(q_shard + r, 0));
    }
    for (int64_t r = 0; r < kv_shard; ++r) {
      CHECK(sliced[0].At(q_shard + r, 0) == b.At(q_total + kv_shard + r, 0));
      CHECK(sliced[0].At(q_shard + kv_shard + r, 0) ==
            b.At(q_total + kv_total + kv_shard + r, 0));
    }
  }

  SUBCASE("MergedColumnParallelLinearWithLoRA slices every sub-module") {
    // column_parallel_linear.py:253-264.
    Rng rng(5);
    const int64_t tp = 2;
    MergedColumnParallelLinearWithLoRA layer(/*input_size=*/8,
                                             std::vector<int64_t>{8, 8}, tp, /*tp_rank=*/1);
    LoRAMat b0 = RandMat(rng, 8, kRank);
    LoRAMat b1 = RandMat(rng, 8, kRank);
    MatList sliced = layer.SliceLoraB({b0, b1});
    REQUIRE(sliced.size() == 2);
    for (size_t i = 0; i < 2; ++i) {
      CHECK(sliced[i].rows == 4);
      const LoRAMat& src = i == 0 ? b0 : b1;
      for (int64_t r = 0; r < 4; ++r) {
        CHECK(sliced[i].At(r, 0) == src.At(4 + r, 0));
      }
    }
  }

  SUBCASE("Fully-sharded (S-LoRA) variants shard lora_a along the rank dim") {
    // column_parallel_linear.py:516-520 (column), :553-563 (merged column),
    // :632-649 (merged qkv); row_parallel_linear.py:111-116 (b, not a).
    Rng rng(6);
    const int64_t tp = 2, max_rank = 8;
    ColumnParallelLinearWithShardedLoRA col(/*input_size=*/8, /*output_size=*/8, tp, 1);
    col.CreateLoraWeights(/*max_loras=*/2, max_rank, /*fully_sharded_loras=*/true);
    CHECK(col.lora_a_rows() == max_rank / tp);
    LoRAMat a = RandMat(rng, max_rank, 8);
    MatList sliced = col.SliceLoraA({a});
    CHECK(sliced[0].rows == max_rank / tp);
    for (int64_t r = 0; r < max_rank / tp; ++r) {
      CHECK(sliced[0].At(r, 0) == a.At(max_rank / tp + r, 0));
    }

    RowParallelLinearWithShardedLoRA row(/*input_size=*/8, /*output_size=*/8, tp, 1);
    row.CreateLoraWeights(/*max_loras=*/2, max_rank, /*fully_sharded_loras=*/true);
    // row_parallel_linear.py:120-125 — lora_b is sharded along the output dim.
    LoRAMat b = RandMat(rng, 8, max_rank);
    MatList rsliced = row.SliceLoraB({b});
    CHECK(rsliced[0].rows == 4);
    for (int64_t r = 0; r < 4; ++r) CHECK(rsliced[0].At(r, 0) == b.At(4 + r, 0));

    MergedColumnParallelLinearWithShardedLoRA mcol(
        /*input_size=*/8, std::vector<int64_t>{8, 8}, tp, 1);
    mcol.CreateLoraWeights(/*max_loras=*/2, max_rank, /*fully_sharded_loras=*/true);
    LoRAMat a0 = RandMat(rng, max_rank, 8);
    LoRAMat a1 = RandMat(rng, max_rank, 8);
    MatList msliced = mcol.SliceLoraA({a0, a1});
    REQUIRE(msliced.size() == 2);
    CHECK(msliced[0].rows == max_rank / tp);
    CHECK(msliced[1].rows == max_rank / tp);

    MergedQKVParallelLinearWithShardedLoRA mqkv(/*input_size=*/8, /*head_size=*/4,
                                                /*total_num_heads=*/8, /*num_heads=*/4,
                                                /*total_num_kv_heads=*/8, /*num_kv_heads=*/4,
                                                tp, 1);
    mqkv.CreateLoraWeights(/*max_loras=*/2, max_rank, /*fully_sharded_loras=*/true);
    LoRAMat qa = RandMat(rng, max_rank, 8);
    MatList qsliced = mqkv.SliceLoraA({qa, qa, qa});
    REQUIRE(qsliced.size() == 3);
    for (const auto& m : qsliced) CHECK(m.rows == max_rank / tp);
  }
}

// ---------------------------------------------------------------------------
// test_embeddings (test_layers.py:269-364)

TEST_CASE("test_embeddings: lora_a is an embedding lookup, lora_b an expand") {
  constexpr int64_t kEmbedDim = 64;  // upstream 256
  constexpr int64_t kMaxLoras = 8;
  constexpr int64_t kMaxRank = 8;

  for (int64_t vocab_size : {512, 32000}) {  // upstream also 64000/128000
    for (int num_loras : {1, 2, 4}) {
      Rng rng(static_cast<uint32_t>(vocab_size + num_loras));
      auto id_to_index = GetRandomIdToIndex(num_loras, static_cast<int>(kMaxLoras), rng);

      VocabParallelEmbeddingWithLoRA layer(vocab_size, kEmbedDim);
      layer.CreateLoraWeights(kMaxLoras, kMaxRank);

      // populate_loras(layer_weights=embedding.weight.T) => lora_a [rank, vocab],
      // lora_b [embedding_dim, rank] (test_layers.py:303-307).
      std::vector<LoRAMat> a_mats, b_mats;
      std::vector<double> scalings;
      std::vector<int> active_ids;
      for (size_t slot = 0; slot < id_to_index.size(); ++slot) {
        if (id_to_index[slot] == 0) continue;
        AdapterGroup g = MakeAdapterGroup(rng, kEmbedDim, vocab_size, /*repeats=*/1);
        layer.SetLora(static_cast<int64_t>(slot), g.a[0], g.b[0]);
        a_mats.push_back(g.a[0]);
        b_mats.push_back(g.b[0]);
        scalings.push_back(g.subloras[0].scaling);
        active_ids.push_back(id_to_index[slot]);
      }

      const int64_t T = 64;
      std::vector<int32_t> tokens(static_cast<size_t>(T));
      std::vector<int32_t> slots(static_cast<size_t>(T));
      std::vector<int> lora_ids(static_cast<size_t>(T));
      for (int64_t t = 0; t < T; ++t) {
        tokens[static_cast<size_t>(t)] = static_cast<int32_t>(rng.Int(1, vocab_size));
        const int lora_id =
            active_ids[static_cast<size_t>(rng.Int(0, static_cast<int64_t>(active_ids.size())))];
        lora_ids[static_cast<size_t>(t)] = lora_id;
        int32_t slot = -1;
        for (size_t s = 0; s < id_to_index.size(); ++s) {
          if (id_to_index[s] == lora_id) slot = static_cast<int32_t>(s);
        }
        slots[static_cast<size_t>(t)] = slot;
      }

      std::vector<float> base = rng.Vec(T * kEmbedDim);
      std::vector<float> y = base;
      layer.ApplyLoraToOutput(y.data(), tokens.data(), T, slots.data());

      // Reference (test_layers.py:325-332):
      //   after_a = F.embedding(input_, lora_a.T);  result += after_a @ lora_b.T
      std::vector<double> ref(base.begin(), base.end());
      for (int64_t t = 0; t < T; ++t) {
        size_t gi = 0;
        for (size_t k = 0; k < active_ids.size(); ++k) {
          if (active_ids[k] == lora_ids[static_cast<size_t>(t)]) gi = k;
        }
        const LoRAMat& a = a_mats[gi];
        const LoRAMat& b = b_mats[gi];
        const int64_t tok = tokens[static_cast<size_t>(t)];
        for (int64_t o = 0; o < kEmbedDim; ++o) {
          double acc = 0.0;
          for (int64_t r = 0; r < a.rows; ++r) {
            acc += static_cast<double>(a.At(r, tok)) *
                   static_cast<double>(b.At(o, r)) * scalings[gi];
          }
          ref[static_cast<size_t>(t * kEmbedDim + o)] += acc;
        }
      }
      CheckAllClose(y, ref);

      for (int64_t slot = 0; slot < kMaxLoras; ++slot) layer.ResetLora(slot);
      std::vector<float> y2 = base;
      layer.ApplyLoraToOutput(y2.data(), tokens.data(), T, slots.data());
      CHECK(y2 == base);
    }
  }
}

// ---------------------------------------------------------------------------
// test_lm_head_logits_processor (test_layers.py:372-510)

TEST_CASE("test_lm_head_logits_processor: sampler-indexed shrink/expand onto logits") {
  constexpr int64_t kHidden = 128;  // upstream 1024
  constexpr int64_t kMaxLoras = 8;
  constexpr int64_t kMaxRank = 8;

  for (int64_t vocab_size : {6400, 25856}) {  // upstream 64000/256512/258048
    for (int num_loras : {1, 2, 4}) {
      Rng rng(static_cast<uint32_t>(vocab_size + num_loras * 3));
      auto id_to_index = GetRandomIdToIndex(num_loras, static_cast<int>(kMaxLoras), rng);

      LogitsProcessorWithLoRA layer(vocab_size, kHidden);
      layer.CreateLoraWeights(kMaxLoras, kMaxRank);

      std::vector<LoRAMat> a_mats, b_mats;
      std::vector<double> scalings;
      std::vector<int> active_ids;
      for (size_t slot = 0; slot < id_to_index.size(); ++slot) {
        if (id_to_index[slot] == 0) continue;
        AdapterGroup g = MakeAdapterGroup(rng, vocab_size, kHidden, /*repeats=*/1);
        layer.SetLora(static_cast<int64_t>(slot), g.a[0], g.b[0]);
        a_mats.push_back(g.a[0]);
        b_mats.push_back(g.b[0]);
        scalings.push_back(g.subloras[0].scaling);
        active_ids.push_back(id_to_index[slot]);
      }

      Batch batch = MakeBatch(rng, active_ids, id_to_index, 8 * num_loras, kHidden);
      std::vector<float> base = rng.Vec(batch.T * vocab_size);
      std::vector<float> y = base;
      layer.ApplyLoraToLogits(y.data(), vocab_size, batch.x.data(), batch.T,
                              batch.slots.data());

      // result += input_ @ lora_a.T @ lora_b.T * lora.scaling (test_layers.py:446).
      std::vector<double> ref(base.begin(), base.end());
      for (int64_t t = 0; t < batch.T; ++t) {
        size_t gi = 0;
        for (size_t k = 0; k < active_ids.size(); ++k) {
          if (active_ids[k] == batch.lora_ids[static_cast<size_t>(t)]) gi = k;
        }
        RefAddSlice(ref, t, vocab_size, 0, batch.x, kHidden, a_mats[gi], b_mats[gi],
                    scalings[gi]);
      }
      CheckAllClose(y, ref);

      for (int64_t slot = 0; slot < kMaxLoras; ++slot) layer.ResetLora(slot);
      std::vector<float> y2 = base;
      layer.ApplyLoraToLogits(y2.data(), vocab_size, batch.x.data(), batch.T,
                              batch.slots.data());
      CHECK(y2 == base);
    }
  }
}

TEST_CASE("test_lm_head_logits_processor_invalid_vocab_size") {
  // logits_processor.py:90-92 — "When using LoRA, vocab size must be <= 258048".
  for (int64_t vocab_size : {258049, 300000}) {
    LogitsProcessorWithLoRA layer(vocab_size, /*hidden_size=*/128);
    CHECK_THROWS_WITH_AS(layer.CreateLoraWeights(/*max_loras=*/8, /*max_lora_rank=*/8),
                         "When using LoRA, vocab size must be <= 258048",
                         std::invalid_argument);
  }
  // 258048 itself is legal (the guard is strictly greater-than).
  LogitsProcessorWithLoRA ok(258048, /*hidden_size=*/8);
  CHECK_NOTHROW(ok.CreateLoraWeights(/*max_loras=*/1, /*max_lora_rank=*/1));
}
