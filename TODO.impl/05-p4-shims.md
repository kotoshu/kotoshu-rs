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

_Planning._
