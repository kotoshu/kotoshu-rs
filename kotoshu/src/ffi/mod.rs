//! FFI: one batch serialization, every binding.
//!
//! - [`shared`] — batch wire format shared by ALL bindings (always available)
//! - [`registry`] — process-wide language → dictionary registry backing the
//!   batch protocol and the C ABI lifecycle calls
//! - [`c`] — C ABI (always available)
//! - [`ruby`] — magnus bindings (feature `ruby`)
//! - [`wasm`] — wasm-bindgen bindings (feature `wasm`; lands P4)
//! - [`python`] — pyo3 bindings (feature `python`; lands P4)

pub mod c;
pub mod registry;
pub mod shared;

#[cfg(feature = "ruby")]
pub mod ruby;

#[cfg(feature = "wasm")]
pub mod wasm;

#[cfg(feature = "python")]
pub mod python;
