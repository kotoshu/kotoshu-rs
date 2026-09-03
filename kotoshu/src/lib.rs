//! Kotoshu Rust core: one engine, every language.
//!
//! Semantic spell-checking engine (Hunspell-style dictionaries and affixes,
//! ranked suggestions, embedding-based reranking) behind a stable C ABI and
//! feature-gated language bindings. The approach is copied wholesale from
//! `parsanol-rs` (Ribose, in production): one pure-Rust core, all FFI
//! feature-gated inside the core, per-language packages as thin shims.

pub mod ffi;

#[cfg(feature = "ruby")]
pub use ffi::ruby as ruby_ffi;
