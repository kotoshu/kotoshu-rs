//! The `@kotoshu/wasm` packaging shim (plan 66 P4c).
//!
//! A thin cdylib over the core's wasm surface, `kotoshu::ffi::wasm` — the
//! `ruby` feature's `tests/ruby_ext` reference-shim pattern promoted to a
//! workspace member, because wasm-pack builds a package, not a module. No
//! engine code and no wasm-bindgen annotations live here (P0 MECE policy:
//! wasm-bindgen types stop at the core's `ffi::wasm` boundary); the
//! wasm-bindgen link metadata ships inside the wasm binary, so re-exporting
//! the surface from the entry crate is the supported packaging shape.
//!
//! Default builds compile this member WITHOUT its `wasm` feature (P0
//! dependency policy); build the npm payload with `scripts/wasm_build.sh`
//! (wasm-pack, `--target bundler` by default). RELEASING.md carries the
//! publish procedure — blocked on npm org credentials (plan 67 M5).

/// The JS surface: the `KotoshuWasm` class (`VERSION`, constructor over
/// source strings, `correct`, `suggest`). Feature-gated like everything
/// wasm — see `kotoshu::ffi::wasm` for the API it exposes.
#[cfg(feature = "wasm")]
pub use kotoshu::ffi::wasm::KotoshuWasm;
