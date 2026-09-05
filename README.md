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

## Status: P4d (Python bindings, build half) on top of P4a–P4c/P3

Everything from P3 (reranking, resources, onnx) plus the first P4 shims,
the `ruby` and `wasm` features and the new `python` feature in
`kotoshu/src/ffi/python/`:

- `python` feature (P4d): the `kotoshu_native` module over pyo3 0.29
  (released crates.io; 0.29.2 verified by the smoke below) — `VERSION`,
  `available()`, and a `Dictionary` class (`load(aff_path, dic_path)`
  instance with `correct(word)` and `suggest(word, limit = 5)` returning
  dicts of the conformance `SUGGESTION_KEYS`
  (`word`/`distance`/`confidence`/`source`)); errors raise
  `KotoshuNativeError` carrying the Rust message. Engine calls run under
  `Python::detach` (pyo3 ≥ 0.26's rename of `allow_threads`) so the GIL
  is released for loads and lookups. The `kotoshu-python` workspace
  member (thin `#[pymodule]` cdylib re-export, its own opt-in `python`
  feature) is the maturin wheel: distribution `kotoshu-native`, module
  `kotoshu_native`, 0.1.0 LIVE on PyPI (owner-published). CI builds and
  smoke-tests the wheel matrix — cp310–cp313 on linux x86_64/aarch64
  (manylinux), macOS x86_64/arm64 and windows x64 — via
  `python-wheels.yml`, published keyless by `release-pypi.yml` behind the
  `kotoshu-native-v*` tag; the procedure and the owner-side PyPI
  trusted-publisher registration live in `kotoshu-python/RELEASING.md`,
  which also documents how the PyPI `kotoshu` package consumes the wheel.
  Conformance chain: `scripts/python_smoke.sh` +
  `scripts/python_smoke.py` (real fixtures, frozen `hlelo` row) in
  `python-ffi.yml` (Python 3.12); per-wheel install smoke:
  `scripts/python_wheel_smoke.py` in the matrix.
- `wasm` feature (P4c): the `KotoshuWasm` JS class in `ffi/wasm` over
  wasm-bindgen — `VERSION`, `new(affSrc, dicSrc)` taking source STRING
  contents (wasm has no fs; byte-symmetric with a path load via the new
  `Dictionary::load_from_sources`), `correct(word)`, `suggest(word,
  limit = 5)` returning plain objects of the conformance
  `SUGGESTION_KEYS` (`word`/`distance`/`confidence`/`source`),
  `JsError` rejections carrying the Rust message, and
  console_error_panic_hook at module start + in the constructor so
  panics are never swallowed. The `kotoshu-wasm` workspace member (thin
  cdylib re-export, its own opt-in `wasm` feature — default workspace
  builds stay dependency-free) is the wasm-pack package: `@kotoshu/wasm`,
  version 0.1.0 PLACEHOLDER (first release is an owner decision; publish
  blocked on npm credentials — `kotoshu-wasm/RELEASING.md`), built by
  `scripts/wasm_build.sh` (bundler default, `web` documented) and smoked
  by `scripts/wasm_node_smoke.mjs` (real fixtures, frozen `hlelo` row);
  CI runs the whole chain (`wasm.yml`).
- `ruby` feature: magnus 0.8 (released crates.io; parsanol's git-rev
  magnus/rb-sys patches are only needed for Ruby 4.0, and this matrix is
  3.3/3.4 — the released line is verified by the smoke below). The core
  stays an `rlib`; `ffi::ruby::init(&Ruby)` defines `Kotoshu::Native`
  (`VERSION`, `available?`, `Dictionary.load(aff, dic)` → instance with
  `correct?(word)` and `suggest(word, limit = 5)` returning hashes of the
  gem's Suggestion fields; errors raise `Kotoshu::Native::Error`). The
  per-language gem's future `ext/kotoshu_native` is a cdylib whose
  `#[magnus::init]` just forwards to that `init`.
- `tests/ruby_ext/` is that shim, exactly, kept as the smoke test: the
  reference extension builds against the workspace, loads into a real MRI
  (`scripts/ruby_ffi_smoke.sh`), and asserts the engine against the synced
  conformance fixtures (including the canonical `suggest("hlelo")` →
  `hello/1/1.0/edit_distance` vector). CI runs it on Ruby 3.3 and 3.4
  (`ruby-ffi.yml`). The gem-side scaffold (extconf.rb, `KOTOSHU_BACKEND`,
  `rake compat:*`) is the separate P4b PR in the gem repo.

Everything from P2/P3 remains:

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
                    ffi/{shared,registry,c,ruby,wasm,python}
kotoshu-wasm/       @kotoshu/wasm packaging member (thin cdylib over ffi/wasm,
                    own opt-in wasm feature — wasm-pack builds it)
kotoshu-python/     kotoshu-native packaging member (maturin #[pymodule] shim over
                    ffi/python, own opt-in python feature — provides kotoshu_native)
tests/              conformance-vector runner + golden JSONL pack (+ synced fixtures, gitignored),
                    rerank_integration.rs (#[ignore]; real model, network + dylib) + registry.json,
                    ruby_ext/ (reference gem-shim cdylib, workspace-excluded) + ruby_ffi_smoke.rb
scripts/            sync_conformance.sh (vectors + fixture dictionaries from the gem repo),
                    ruby_ffi_smoke.sh (build the shim, run the smoke under MRI),
wasm_build.sh (wasm-pack package build for @kotoshu/wasm),
wasm_node_smoke.mjs (Node smoke test over real fixtures),
python_smoke.sh + python_smoke.py (venv + maturin wheel + Python smoke)
.github/workflows/  ci.yml, ruby-ffi.yml, wasm.yml, python-ffi.yml,
                    python-wheels.yml (kotoshu-native wheel matrix),
                    release-plz.yml, release-crate.yml, release-npm.yml,
                    release-pypi.yml (keyless kotoshu-native publish)
```

## Features

The DEFAULT feature set stays empty (P0 policy): without features the
core has zero third-party dependencies. Optional deps attach per phase:

| Feature    | Attaches | Purpose |
|------------|----------|---------|
| `onnx`     | ort `load-dynamic` + serde/serde_json (P3) | embedding inference over the tier `.onnx` artifacts |
| `resources`| serde/serde_json + sha2 (P3) | registry parse, sha256 verify, model cache |
| `ruby`     | magnus 0.8 (P4) | `Kotoshu::Native` bindings inside the core; the gem's ext cdylib forwards to `ffi::ruby::init` |
| `wasm`     | wasm-bindgen/js-sys/console_error_panic_hook (P4) | `KotoshuWasm` JS class in `ffi/wasm` (in-memory sources, conformance-row shape); the `kotoshu-wasm` member packages it as `@kotoshu/wasm` (publish blocked on npm credentials) |
| `python`   | pyo3 0.29 (P4) | `kotoshu_native` module in `ffi/python` (`Dictionary.load`/`correct`/`suggest`, conformance-row dicts, GIL released for engine calls); the `kotoshu-python` member builds the `kotoshu-native` maturin wheel (0.1.0 live on PyPI; wheel matrix in `python-wheels.yml`, keyless publish in `release-pypi.yml`) |
| `parallel` | rayon (P2) | parallel batch checking |
| `logging`  | log (P2, deferred from P1) | diagnostics |

`ort` is deliberately `load-dynamic`-only: the library never links or
downloads an onnxruntime; the host supplies `libonnxruntime`
(`KOTOSHU_ORT_DYLIB`, or ort's own `ORT_DYLIB_PATH`/default search).

## JS: `@kotoshu/wasm`

```js
import init, { KotoshuWasm } from "@kotoshu/wasm";

await init(); // with a bundler; or await init(wasmBytes) anywhere

const aff = await (await fetch("/dictionaries/en.aff")).text();
const dic = await (await fetch("/dictionaries/en.dic")).text();
const dictionary = new KotoshuWasm(aff, dic); // source contents, not paths

dictionary.correct("hlelo"); // => false
dictionary.suggest("hlelo", 5);
// => [{ word: "hello", distance: 1, confidence: 1.0, source: "edit_distance" }, ...]
```

Same engine, same frozen conformance vectors, same suggestion-row shape as
the Ruby gem and the C ABI. Publishing is blocked on npm org credentials;
build the package locally with `scripts/wasm_build.sh` and run the same
smoke CI runs with `node scripts/wasm_node_smoke.mjs`.

## Python: `kotoshu_native`

```python
import kotoshu_native

kotoshu_native.VERSION  # "0.1.0" (kotoshu crate version)

dictionary = kotoshu_native.Dictionary.load("en.aff", "en.dic")

dictionary.correct("hlelo")  # False
dictionary.suggest("hlelo", 5)
# [{"word": "hello", "distance": 1, "confidence": 1.0, "source": "edit_distance"}, ...]
```

Same engine, same frozen conformance vectors, same suggestion-row shape as
the Ruby gem, the C ABI and the WASM build; every failure raises
`kotoshu_native.KotoshuNativeError` with the Rust message. The wheel
(distribution `kotoshu-native`, module `kotoshu_native`) is 0.1.0 LIVE on
PyPI; CI builds the cp310–cp313 matrix (linux x86_64/aarch64, macOS
x86_64/arm64, windows x64) and publishes keyless behind the
`kotoshu-native-v*` tag — see `kotoshu-python/RELEASING.md` for the
procedure and the one-time owner-side trusted-publisher registration.
Build and smoke locally with `scripts/python_smoke.sh` (venv + maturin +
the frozen `hlelo` row).

## MSRV

Deliberately **not set**: the minimum supported Rust version is an owner
decision (see plan 67's owner-decision gates). `rust-toolchain.toml` pins
stable for development and CI.

## License

BSD-2-Clause — see [LICENSE](LICENSE).
