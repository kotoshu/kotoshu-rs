# kotoshu-rs

Kotoshu Rust core: **one engine, every language.**

Kotoshu 「言修」 is a semantic spell checker: Hunspell-style dictionaries and
affixes, ranked suggestions (edit distance, phonetic, keyboard proximity,
n-gram, frequency), and context-aware reranking with FastText-style word
embeddings. This repository is the pure-Rust engine behind every Kotoshu
surface — the Ruby gem, Python and JS packages, WASM, CLI, and LSP — exposed
through a stable C ABI plus feature-gated bindings.

## Parsanol blueprint

The approach is copied wholesale from [`parsanol-rs`][parsanol-rs] +
`parsanol-ruby` (Ribose, in production), because it demonstrably works:
**one pure-Rust core; all FFI feature-gated *inside* the core; per-language
packages as thin cdylib shims; a pure-Ruby fallback retained; a dual-backend
conformance suite.** The batch FFI wire format in `kotoshu/src/ffi/shared.rs`
(one serialization, all bindings; measured 3-5x over object-by-object FFI in
parsanol) is the load-bearing pattern: every binding and the conformance
vectors speak the same bytes.

[parsanol-rs]: https://github.com/parsanol/parsanol-rs

## Status: P3 (rerank + resources)

Everything from P2 plus context-aware reranking in `kotoshu/src/rerank/`
and resource handling in `kotoshu/src/resource/` (both P3 features;
see below):

- `rerank/` (always compiled, zero dependencies): the
  `EmbeddingProvider` trait (inference is host-injected — the rerank
  math works on vectors the host supplies), cosine similarity and the
  `CosineReranker` ported from the gem's `SemanticAnalyzer`
  (`rank_by_context` / `context_boost`: each suggestion's confidence
  gains `0.02 × Σ cosine(suggestion, context_word)`, capped at 1.0,
  then sorted descending). The gem's context wiring — which boosts
  candidates against the misspelling's own OOV form, a no-op — is
  fixed by building the surrounding words from the neighbors; the
  deviation is documented in the module docs.
- `rerank/dequant.rs` (B1 groundwork, dequant math only): the
  int8-per-row dequant the tier graphs run in-graph, plus a
  documented int4-per-row packing contract (format byte + metadata
  string) accepted ahead of the models-repo artifacts.
- `rerank/oov.rs` (B2): fastText's FNV-1a n-gram hash (with the
  sign-extended `int8` byte quirk) for the future bucket-table
  artifacts, and the honest OOV fallback over today's word-only
  artifacts — the L2-normalized sum of a word's in-vocab character
  n-gram substrings (`SubwordFallback` provider wrapper).
- `onnx` feature: `OrtProvider` over `ort` 2.0.0-rc.13 with
  `load-dynamic` only (a bundled onnxruntime is never linked); the
  library path comes from `KOTOSHU_ORT_DYLIB` or ort's default search.
  Tests skip cleanly with a clear message when no dylib is loadable.
- `resources` feature: the models registry (`kotoshu.resources/v1`)
  parsed with serde, `(language, tier)` resolution, sha256 verification
  (sha2), and a local cache honoring `KOTOSHU_CACHE_PATH`; downloads
  go primary-then-mirror through an injectable fetcher (system `curl`
  by default — no HTTP stack enters the dependency budget) and are
  checksum-verified before an atomic on-disk replace.
- The ignored integration test (`tests/rerank_integration.rs`) runs
  the real thing: the committed registry fixture (release v1.0.1), a
  real en/mini download (~3 MB) sha-verified through the resource
  layer, golden cosines precomputed against the artifact with
  python + onnxruntime (1e-4 tolerance, regeneration documented), and
  `suggest("helo")` reranked with "hello" kept at rank 1.

All 2630 conformance vectors remain green (1315 `correct` + 1315
`suggest`, engine and C ABI). Remaining phases — see
[`TODO.impl/`](TODO.impl/) and the authoritative plan in the gem repo,
[plan 66 — kotoshu-rs: the Rust core][plan-66].

[plan-66]: https://github.com/kotoshu/kotoshu/blob/main/TODO.impl/66-kotoshu-core.md

## Layout

```
kotoshu/            core crate (rlib): dict/{aff,dic,casing,encoding,lookup},
                    suggest/{edit_distance,phonetic,keyboard,ngram,frequency,rank,...},
                    rerank/{dequant,oov,onnx}, resource/,
                    ffi/{shared,registry,c,ruby,wasm}
tests/              conformance-vector runner + golden JSONL pack (+ synced fixtures, gitignored),
                    rerank_integration.rs (#[ignore]; real model, network + dylib) + registry.json
scripts/            sync_conformance.sh (vectors + fixture dictionaries from the gem repo)
.github/workflows/  ci.yml, ruby-ffi.yml, wasm.yml, release-plz.yml
```

## Features

The DEFAULT feature set stays empty (P0 policy): without features the
core has zero third-party dependencies. Optional deps attach per phase:

| Feature    | Attaches | Purpose |
|------------|----------|---------|
| `onnx`     | ort `load-dynamic` + serde/serde_json (P3) | embedding inference over the tier `.onnx` artifacts |
| `resources`| serde/serde_json + sha2 (P3) | registry parse, sha256 verify, model cache |
| `ruby`     | magnus (P4) | Ruby bindings inside the core |
| `wasm`     | wasm-bindgen & co. (P4) | `@kotoshu/wasm` browser/Node |
| `parallel` | rayon (P2) | parallel batch checking |
| `logging`  | log (P2, deferred from P1) | diagnostics |

`ort` is deliberately `load-dynamic`-only: the library never links or
downloads an onnxruntime; the host supplies `libonnxruntime`
(`KOTOSHU_ORT_DYLIB`, or ort's own `ORT_DYLIB_PATH`/default search).

## MSRV

Deliberately **not set**: the minimum supported Rust version is an owner
decision (see plan 67's owner-decision gates). `rust-toolchain.toml` pins
stable for development and CI.

## License

BSD-2-Clause — see [LICENSE](LICENSE).
