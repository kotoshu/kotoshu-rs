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

## Status: P2 (suggestions)

Everything from P1 plus the full suggestion pipeline in
`kotoshu/src/suggest/`: the gem's four default strategies — edit distance
(Damerau threshold DP with enhanced scoring: frequency bonus, keyboard
penalty, transposition and typo-pattern bonuses), phonetic (Soundex),
keyboard proximity (variant generation over a neighbor table), and n-gram
(multiset Jaccard) — composited into one ranked `SuggestionSet`. Ranking
reproduces the gem byte-for-byte, including Ruby's unstable sort tie
orders: MRI 3.4's `sort_by` introsort (`suggest/ruby_sort.rs`) and the
macOS libc `qsort_r` that `Array#sort!` delegates to on the export
platform (`suggest/macos_qsort.rs`) — 16 frozen vectors contain full-tie
suggestion pairs whose surviving `source` label is decided by that tie
order. The Kelly frequency tiers that shaped the export (frozen from the
gem's `FrequencyCache` state, embedded in `suggest/frequency_data.rs`)
drive the frequency bonus. All 2630 conformance vectors assert green:
1315/1315 `correct` and 1315/1315 `suggest` (ordered equality, exact f64
confidences) — through the engine AND through the KOSH-v1 batch wire on
the C ABI (`kotoshu_dict_load`/`kotoshu_dict_free` lifecycle +
`kotoshu_batch`). Differential fuzz against the live gem (5,948 suggests
across all 125 fixture dictionaries: mutations, junk, unicode, limits
1-10) reports zero mismatches. Remaining phases — see
[`TODO.impl/`](TODO.impl/) and the authoritative plan in the gem repo,
[plan 66 — kotoshu-rs: the Rust core][plan-66].

[plan-66]: https://github.com/kotoshu/kotoshu/blob/main/TODO.impl/66-kotoshu-core.md

## Layout

```
kotoshu/            core crate (rlib): dict/{aff,dic,casing,encoding,lookup},
                    suggest/{edit_distance,phonetic,keyboard,ngram,frequency,rank,...},
                    ffi/{shared,registry,c,ruby,wasm}
tests/              conformance-vector runner + golden JSONL pack (+ synced fixtures, gitignored)
scripts/            sync_conformance.sh (vectors + fixture dictionaries from the gem repo)
.github/workflows/  ci.yml, ruby-ffi.yml, wasm.yml, release-plz.yml
```

## Features

All features are declared with empty dependency sets at P0 (the workspace has
zero third-party dependencies by design); optional deps attach as each phase
lands:

| Feature   | Attaches    | Purpose                                    |
|-----------|-------------|--------------------------------------------|
| `ruby`    | magnus (P4) | Ruby bindings inside the core              |
| `wasm`    | wasm-bindgen & co. (P4) | `@kotoshu/wasm` browser/Node  |
| `onnx`    | ort `load-dynamic` (P3) | embedding inference          |
| `parallel`| rayon (P2)  | parallel batch checking                     |
| `logging` | log (P2, deferred from P1) | diagnostics                     |

## MSRV

Deliberately **not set**: the minimum supported Rust version is an owner
decision (see plan 67's owner-decision gates). `rust-toolchain.toml` pins
stable for development and CI.

## License

BSD-2-Clause — see [LICENSE](LICENSE).
