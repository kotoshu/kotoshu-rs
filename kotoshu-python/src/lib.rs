//! The `kotoshu_native` extension module (plan 66 P4d).
//!
//! A thin cdylib over the core's pyo3 surface, `kotoshu::ffi::python` —
//! the `ruby` feature's `tests/ruby_ext` reference-shim pattern promoted
//! to a workspace member (the `kotoshu-wasm` member's twin), because
//! maturin builds a package, not a module. No engine code and no pyo3
//! annotations beyond this `#[pymodule]` live here (P0 MECE policy: pyo3
//! types stop at the core's `ffi::python` boundary).
//!
//! Default builds compile this member WITHOUT its `python` feature (P0
//! dependency policy); the wheel is built by maturin with it
//! (`scripts/python_smoke.sh`, or `maturin develop` in a venv).
//! RELEASING.md carries the publish procedure — blocked on PyPI
//! credentials (plan 67 M5) — and how the PyPI `kotoshu` package in the
//! kotoshu-python repository will consume this wheel.

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// The Python surface: `VERSION`, `available()`, `Dictionary.load` /
/// `correct` / `suggest`, `KotoshuNativeError`. Feature-gated like
/// everything python — see `kotoshu::ffi::python` for the API it exposes.
#[cfg(feature = "python")]
#[pymodule]
fn kotoshu_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    kotoshu::ffi::python::register(module)
}
