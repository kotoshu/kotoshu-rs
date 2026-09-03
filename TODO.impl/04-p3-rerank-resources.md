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

_Planning._

[68-sota-adoption.md]: https://github.com/kotoshu/kotoshu/blob/main/TODO.impl/68-sota-adoption.md
