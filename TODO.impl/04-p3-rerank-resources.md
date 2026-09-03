# 04 — P3 rerank + resources

Phase P3 of [66-kotoshu-core.md] (gem repo): embedding reranking and
resource handling. Includes the plan-68 B-items as named milestones (see
[68-sota-adoption.md], gem repo, adopted 2026-09-02: "B1–B3 fold into
plan 66 P3").

## Goal

Context-aware reranking over host-supplied embeddings, plus manifest/sha256/
tier parsing of the resource registry (`kotoshu://models/{lang}/{tier}`).

## Tasks

- `rerank.rs`: pure vector math + the `EmbeddingProvider` trait (inference is
  host-injected; rerank math works on vectors the host supplies).
- Optional dep: `onnx` → `ort = { version = "2", default-features = false,
  features = ["load-dynamic"] }` — hosts share the libonnxruntime they
  already ship; standalone builds use ort's bundled binaries.
- `resource/` module: registry manifest parse, sha256 verification, tier
  metadata.
- Rerank conformance vectors (rerank specs from the gem).
- **B1 milestone** — int4 group-128 quantized full tier (120 MB → ~15–20 MB
  near-lossless): core dequant math / ort support; possibly a `nano` tier
  (tier names are owner decisions).
- **B2 milestone** — hashed char n-gram OOV fallback (fastText buckets or
  collision-aware hashing, arxiv 1709.03933) so unseen words still embed;
  converter work stays in the models repo.
- **B3 milestone** — tiny cross-encoder reranker option (~22M distilled
  MiniLM class, ONNX) behind the same `EmbeddingProvider`/rerank trait; our
  ≤25-candidate scale keeps it cheap.

## Acceptance

- Rerank conformance vectors pass through the provider trait with the ort
  `load-dynamic` provider and a test double.
- B1/B2/B3 each land with conformance + eval gates (per plan 68 policy).

## Status

**Core implemented** (2026-09-03, branch `feat/p3-rerank`).

- `rerank/` (always compiled, zero deps): `EmbeddingProvider` trait
  (`embedding`/`dims`/`embedding_oov`), `cosine` mirroring the gem's
  `WordEmbedding#similarity` (0.0 on mismatch/zero norms), `Context`
  (before/current/after + surrounding words), and `CosineReranker`
  porting `rank_by_context`/`context_boost` (0.02 weight, cap at 1.0,
  descending sort, deterministic tie order). Two documented deviations
  from the gem: surrounding words come from the *neighbors* (the gem's
  `build_context` puts the OOV word itself in `current`, making its
  boost path a no-op), and tie order is stable (the gem's is
  MRI-version-dependent; no vector freezes it).
- `onnx` feature: `OrtProvider` (ort 2.0.0-rc.13, `load-dynamic`
  only, `KOTOSHU_ORT_DYLIB` → ort default search). Reads the int8
  tier graph + vocab.json; one `session.run` per embedding. ort's
  lazy-load `expect` panic is caught so absent dylibs produce `Err`,
  not a crash — tests skip cleanly.
- `resources` feature: registry parse (spec `kotoshu.resources/v1`),
  `(language, tier)` resolve, sha256 verify, cache under
  `KOTOSHU_CACHE_PATH`/XDG/HOME; primary-then-mirror download,
  checksum-before-write, atomic replace, corrupt-cache self-heal.
  Fetch is injectable (`ensure_model_with`); the default fetcher
  shells out to system `curl` — the feature's dependency budget is
  serde/serde_json/sha2, no HTTP stack.
- **B2 landed, honest scope**: `oov::fasttext_hash` (FNV-1a with the
  sign-extended `int8` byte cast, known-answer tested) is the bucket
  selector for future artifacts; over today's word-only artifacts the
  fallback is the L2-normalized sum of in-vocab character n-gram
  substrings (`SubwordFallback`). Full fastText bucket hashing needs
  re-converted artifacts — a models-repo converter change; the
  mechanism is trait-pluggable (`embedding_oov`).
- **B1 groundwork (dequant only)**: `dequant::RowFormat` parses the
  tier `quantization` metadata into documented format bytes
  (0x00 fp32 / 0x08 int8-per-row / 0x04 int4-per-row); int8 dequant
  mirrors `build_tiers.py`'s recipe (parity-tested against its own
  0.05 gate); the int4-per-row packing contract (signed nibbles, high
  first, `scale = max_abs/7`) is unit-tested with synthetic tensors.
  **The int4 artifacts themselves remain a models-repo task.**
- **B3 cross-encoder: not started** — needs a trained model; the
  natural extension is a second provider method (e.g.
  `score(pair) -> f64`) beside `EmbeddingProvider`, keeping the ≤25
  candidate scale cheap. Next P3+ item.
- Integration test (`tests/rerank_integration.rs`, `#[ignore]`):
  committed registry fixture (v1.0.1, verbatim from the models repo),
  real en/mini download sha-verified through the resource layer,
  golden cosines precomputed with python + onnxruntime (1e-4
  tolerance; regeneration snippet in the test), rerank of
  `suggest("helo")` keeps "hello" at rank 1, B2 fallback over the
  real vocabulary. New CI `onnx` job (pip onnxruntime →
  `KOTOSHU_ORT_DYLIB` → `cargo test --features onnx,resources` +
  `-- --ignored`) runs it; unit tests skip cleanly without the dylib.
- Conformance: 1315 + 1315 still green through engine and C ABI
  (rerank is an opt-in post-step; the default pipeline is untouched).

[68-sota-adoption.md]: https://github.com/kotoshu/kotoshu/blob/main/TODO.impl/68-sota-adoption.md
