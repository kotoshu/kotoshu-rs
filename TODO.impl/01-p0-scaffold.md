# 01 — P0 scaffold

Implements P0 of [66-kotoshu-core.md] (gem repo), milestone M3 of
[67-kotoshu-rs-and-access-libraries.md].

## Goal

A compiling skeleton — workspace, core crate, batch wire format, C ABI,
conformance-vector runner, CI — not a port. Zero third-party dependencies:
`ruby`/`wasm`/`onnx`/`parallel`/`logging` features are declared with empty
dependency sets so CI stays green without the magnus/ort/wasm toolchains.

## Tasks

- Workspace: root `Cargo.toml` (parsanol profile: release `opt-level 3`,
  `lto = true`, `codegen-units = 1`; deps `opt-level "s"`), deny/typos/
  release-plz/pre-commit configs, `rust-toolchain.toml` (stable; MSRV
  deliberately unset — owner decision).
- `kotoshu` core crate (`crate-type = ["rlib"]`): `ffi/shared.rs` batch
  format (KOSH v1: check/suggest request+response, documented little-endian
  layout, manual encode/decode), `ffi/c.rs` (`kotoshu_version`,
  `kotoshu_batch`, `kotoshu_free`), cfg-gated empty `ffi/ruby`, `ffi/wasm`.
- `tests/conformance.rs` runner + placeholder JSONL pack (3 lines, gem API
  shapes `correct?`/`suggest`), to be replaced by the gem's
  `rake kotoshu:conformance:export` golden vectors.
- CI: `ci.yml` (fmt/clippy/test, Linux+macOS), `ruby-ffi.yml` and `wasm.yml`
  (P0 smoke checks; activate at P4), `release-plz.yml` (disabled — first
  release and versions are owner decisions).

## Acceptance

- `cargo build`, `cargo test`, `cargo build --features ruby` green locally.
- CI green on push to main.
- No deps, no published version, no tags.

## Status

**Implemented** (2026-09-03). Deferred to later phases: optional deps
(magnus P4, wasm-bindgen P4, ort P3, rayon P2, log P1); real engine results
(P1 check, P2 suggest); golden conformance vectors (blocked on the gem's
export rake task).

[66-kotoshu-core.md]: https://github.com/kotoshu/kotoshu/blob/main/TODO.impl/66-kotoshu-core.md
[67-kotoshu-rs-and-access-libraries.md]: https://github.com/kotoshu/kotoshu/blob/main/TODO.impl/67-kotoshu-rs-and-access-libraries.md
