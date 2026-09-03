//! FFI: one batch serialization, every binding.
//!
//! - [`shared`] — batch wire format shared by ALL bindings (always available)
//! - [`c`] — C ABI (always available)
//! - [`ruby`] — magnus bindings (feature `ruby`; lands P4)
//! - [`wasm`] — wasm-bindgen bindings (feature `wasm`; lands P4)

pub mod c;
pub mod shared;

#[cfg(feature = "ruby")]
pub mod ruby;

#[cfg(feature = "wasm")]
pub mod wasm;
