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

## Status: P1 (dictionary engine)

Everything from P0 plus the real Hunspell dictionary path: `.aff`/`.dic`
parsing (encodings, flag formats, `AF` aliases), affix expansion (two-fold
suffixes, cross-product prefixes, `COMPLEXPREFIXES`), compounds (flags and
rules), capitalization (standard/German/Turkic), break patterns, `ICONV`,
`IGNORE`, and the full `correct?` flag semantics (`KEEPCASE`, `NEEDAFFIX`,
`CIRCUMFIX`, `ONLYINCOMPOUND`, `FORBIDDENWORD`, …) in `kotoshu/src/dict/`.
The 2630-vector conformance pack from the gem is asserted for `correct`
(1315/1315) via `scripts/sync_conformance.sh` (fixtures synced from the gem
repo, never committed; the test skips gracefully without them). Suggest
vectors count but are not asserted until P2. Remaining phases — see
[`TODO.impl/`](TODO.impl/) and the authoritative plan in the gem repo,
[plan 66 — kotoshu-rs: the Rust core][plan-66].

[plan-66]: https://github.com/kotoshu/kotoshu/blob/main/TODO.impl/66-kotoshu-core.md

## Layout

```
kotoshu/            core crate (rlib): dict/{aff,dic,casing,encoding,lookup}, ffi/{shared,c,ruby,wasm}
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
