# 05 — P4 shims

Phase P4 of [66-kotoshu-core.md] (gem repo), milestone M5 of
[67-kotoshu-rs-and-access-libraries.md]: language bindings over the finished
core.

## Goal

Every host consumes the same engine: Ruby gem ext, Python wheel, `@kotoshu/
wasm`, single-binary CLI/LSP.

## Tasks

- `ruby` feature: attach magnus (`ffi/ruby/init.rs`, engine init + batch
  calls over `ffi::shared`); `[patch.crates-io]` rb-sys revs per parsanol
  policy. Activate `ruby-ffi.yml` (one MRI to start; matrix to
  3.2/3.3/3.4/4.0-head at the gate).
- Gem side (kotoshu repo, PR-gated): `ext/kotoshu_native` cdylib,
  `extconf.rb` via `rb_sys/mkmf` + `create_rust_makefile`,
  `Kotoshu::Native.available?` guard, `KOTOSHU_BACKEND=native|ruby|auto`,
  `rake compat:ruby|native|compare`.
- `wasm` feature: attach wasm-bindgen/js-sys/console_error_panic_hook;
  activate `wasm.yml` (wasm32 build + wasm-pack, publish `@kotoshu/wasm` —
  npm credentials currently blocked per plan 67).
- Python: maturin wheel over the same core in `kotoshu-python` (PyPI token
  currently blocked; publish code regardless).
- Rust CLI/LSP: single-binary distribution (`release-binary.yml`).

## Acceptance

- Conformance vectors pass through every binding; `rake compat:compare`
  zero diffs.
- `gem install kotoshu` works with and without a Rust toolchain, identical
  results.
- Version lines independent per package (parsanol policy); all first
  releases and versions are owner decisions.

## Status

**P4a core-side done** (2026-09-03, branch `feat/p4a-ruby`) — the `ruby`
feature and the activated `ruby-ffi.yml`. The gem-side ext scaffold is
**pending (P4b, separate PR in the gem repo)**.

- `kotoshu/src/ffi/ruby/`: `ffi::ruby::init(&Ruby)` defines
  `Kotoshu::Native` — `VERSION` (crate version), `available?` → true,
  `Dictionary.load(aff, dic)` returning an instance with `correct?(word)`
  and `suggest(word, limit = 5)`. Suggestion rows are hashes of exactly
  the gem Suggestion / conformance `SUGGESTION_KEYS` fields
  (`word`/`distance`/`confidence`/`source`), so P4b materializes
  `Kotoshu::Suggestions::Suggestion` objects from them directly. Failures
  raise `Kotoshu::Native::Error` (RuntimeError subclass) carrying the
  Rust message. Magnus types stop at the ffi module (engine stays pure
  Rust). Instance surface, not the KOSH batch calls — the wire stays
  available via the C ABI for hosts that want it; the Ruby-idiomatic
  surface is what the gem needs.
- **Version policy**: magnus 0.8 from crates.io (rb-sys 0.9.130 resolves
  under it). Parsanol's git-rev magnus 0.9 + `[patch.crates-io]` rb-sys
  exist for Ruby 4.0 compatibility ("use magnus and rb-sys compatible
  with ruby 4.0"); on this repo's 3.3/3.4 matrix the released line is
  verified by the smoke test, and the patch's remaining rev in parsanol
  is unused in its own graph today. **Relax trigger**: when the matrix
  grows 4.0-head (P4 gate) and the released line misbehaves, mirror the
  parsanol revs then. No patch pins were added.
- `tests/ruby_ext/` (workspace-excluded cdylib) is the shim pattern
  itself — `#[magnus::init]` forwarding to `ffi::ruby::init` — kept as
  the smoke test: `scripts/ruby_ffi_smoke.sh` builds it, stages the
  bundle for `require`, and runs `tests/ruby_ffi_smoke.rb` (21
  assertions incl. the canonical `hlelo` → `hello/1/1.0/edit_distance`
  conformance row and the error surface). macOS needs
  `-undefined dynamic_lookup` in `tests/ruby_ext/.cargo/config.toml`
  (rb_sys/mkmf normally supplies it; this build is outside extconf).
- `ruby-ffi.yml` is real: clippy + `cargo test --features ruby` (full
  2630-vector conformance pack) on Ruby 3.3/3.4, then the smoke.
- **P4c build half done** (2026-09-04, branch `feat/p4c-wasm`) — the
  `wasm` feature is real; only the npm publish is left (credentials-
  blocked, below).
  - `kotoshu/src/ffi/wasm/`: the `KotoshuWasm` class (wasm-bindgen) —
    `VERSION` (crate version), `new(aff_src, dic_src)` over STRING
    contents (wasm has no fs; byte-symmetric with a path load through the
    new `Dictionary::load_from_sources` / `Lookuper::from_bytes`),
    `correct(word)`, `suggest(word, limit = 5)` returning plain objects
    of exactly the conformance `SUGGESTION_KEYS` fields,
    `Result<_, JsError>` for graceful errors, console_error_panic_hook at
    module start and in the constructor (idempotent) so panic messages
    are never swallowed. wasm-bindgen/js-sys types stop at `ffi/wasm`.
  - `kotoshu-wasm/` member: thin cdylib re-exporting the core surface —
    the `tests/ruby_ext` shim pattern promoted to a workspace member,
    with its OWN opt-in `wasm` feature so default workspace builds stay
    dependency-free (P0 policy; parsanol has no separate wasm member —
    deviation, it wasm-pack builds its single core crate). npm name
    `@kotoshu/wasm` (wasm-pack cannot express scopes; the build script
    rewrites package.json), version 0.1.0 PLACEHOLDER — owner decision.
    `pkg/` gitignored; `RELEASING.md` documents the blocked publish and
    the exact commands for when credentials exist.
  - `scripts/wasm_build.sh` (wasm-pack --target bundler default, `web`
    documented via WASM_PACK_TARGET) and `scripts/wasm_node_smoke.mjs`
    (real fixture strings via sync_conformance, frozen `hlelo` →
    `hello/1/1.0/edit_distance` row, OOV word, error surface — the Node
    twin of the Ruby smoke).
  - `wasm.yml` real: wasm32-unknown-unknown build + wasm-pack + the Node
    smoke on synced fixtures; no publish step (deliberately).
- **P4d build half done** (2026-09-04, branch `feat/p4d-python`) — the
  `python` feature is real; only the PyPI publish is left
  (credentials-blocked, below).
  - `kotoshu/src/ffi/python/`: `ffi::python::register(&Bound<PyModule>)`
    defines the `kotoshu_native` module — `VERSION` (crate version),
    `available()` → `True`, `Dictionary.load(aff_path, dic_path)`
    returning a `#[pyclass]` instance with `correct(word)` and
    `suggest(word, limit = 5)` returning dicts of exactly the conformance
    `SUGGESTION_KEYS` (`word`/`distance`/`confidence`/`source`), the
    same row shape `ffi::ruby` hashes. Failures raise
    `KotoshuNativeError` (Exception subclass) carrying the Rust message;
    `Dictionary()` itself raises (instances exist only through `load`,
    the ruby feature's undef-`allocate` stance). Engine calls run under
    `Python::detach` — pyo3 ≥ 0.26's rename of `allow_threads` — so the
    GIL is released for loads/lookups (engine is plain `Send + Sync`
    data; rows materialize back under the GIL). pyo3 is the released
    0.29.2 from crates.io (no git revs/patches, magnus policy); pyo3
    types stop at `ffi/python`.
  - `kotoshu-python/` member: thin `#[pymodule] kotoshu_native` cdylib
    forwarding to `ffi::python::register` (the `tests/ruby_ext` shim
    pattern promoted to a workspace member, the `kotoshu-wasm` twin),
    with its OWN opt-in `python` feature (default workspace builds stay
    dependency-free, P0 policy) and a direct pyo3 dep for the macro.
    maturin wheel: distribution `kotoshu-native`, module `kotoshu_native`
    (the `[lib]` name), version 0.1.0 PLACEHOLDER — owner decision, name
    included. `RELEASING.md` documents the blocked publish, the exact
    commands, and the integration plan: the PyPI `kotoshu` package
    (kotoshu-python repository, pure-Python today) will depend on
    `kotoshu-native` and import `kotoshu_native` behind a
    `KOTOSHU_BACKEND=native|http|auto` guard — this repo's counterpart
    of the gem-side P4b PR.
  - `scripts/python_smoke.sh` (venv + maturin + wheel install) and
    `scripts/python_smoke.py` (real fixture dictionaries via
    sync_conformance, frozen `hlelo` → `hello/1/1.0/edit_distance` row,
    OOV word, error surface, 22 assertions — the Python twin of the Ruby
    and Node smokes). Note: standalone test binaries link libpython;
    macOS dev needs `DYLD_LIBRARY_PATH` at the interpreter's libdir (CI
    sets the Linux `LD_LIBRARY_PATH` equivalent, like the ruby job's
    libruby).
  - `python-ffi.yml` real: clippy + `cargo test --features python` (full
    conformance pack) on Python 3.12, then the venv/wheel/smoke chain;
    no publish step (deliberately).
- Remaining in this phase: the `@kotoshu/wasm` npm publish and the
  `kotoshu-native` PyPI publish (owner credentials, plan 67 M5), the
  kotoshu-python repo integration above, CLI/LSP distribution — and P4b gem-side
  (`ext/kotoshu_native`, `extconf.rb`, `KOTOSHU_BACKEND`,
  `rake compat:*`), which needs from this PR only the
  `ffi::ruby::init` signature above and the suggestion-row shape.
