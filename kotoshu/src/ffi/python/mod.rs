//! Python bindings (feature `python`): the engine behind the
//! `kotoshu_native` extension module, built with [pyo3] and packaged as a
//! maturin wheel (`kotoshu-python` member — PyPI publication is blocked on
//! credentials, plan 67 M5; see `kotoshu-python/RELEASING.md`).
//!
//! [pyo3]: https://crates.io/crates/pyo3
//!
//! # Shim pattern (parsanol blueprint, plan 66)
//!
//! This crate is an `rlib`; it never defines the `PyInit_` entry point.
//! The Python package builds a tiny cdylib whose `#[pymodule]` does
//! nothing but forward:
//!
//! ```ignore
//! #[pymodule]
//! fn kotoshu_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
//!     kotoshu::ffi::python::register(m)
//! }
//! ```
//!
//! `kotoshu-python` in this repository is exactly that shim, built by
//! maturin and exercised as the smoke test (`scripts/python_smoke.sh`).
//!
//! # Exposed API
//!
//! ```python
//! import kotoshu_native
//!
//! kotoshu_native.VERSION                    # "0.1.0" (kotoshu crate version)
//! kotoshu_native.available()                # True
//!
//! dictionary = kotoshu_native.Dictionary.load(aff_path, dic_path)
//! dictionary.correct("hello")               # True / False
//! dictionary.suggest("hlelo", 5)            # [{"word": "hello", ...}]
//! dictionary.suggest("hlelo")               # limit defaults to 5
//! ```
//!
//! `suggest` returns one `dict` per suggestion with exactly the four keys
//! of the gem's `Kotoshu::Suggestions::Suggestion` / the conformance
//! `SUGGESTION_KEYS`: `"word"` (str), `"distance"` (int),
//! `"confidence"` (float in `[0, 1]`), `"source"` (str, one of
//! `edit_distance`, `phonetic`, `keyboard_proximity`, `ngram`) — the same
//! row shape `ffi::ruby` hashes, `ffi::wasm` objects and the frozen
//! vectors use. The Python wrapper package (PyPI `kotoshu`) materializes
//! its `Suggestion` dataclasses from these dicts.
//!
//! All engine failures surface as `kotoshu_native.KotoshuNativeError`
//! (an `Exception` subclass) carrying the Rust error message.
//!
//! # GIL handling
//!
//! Every engine call runs inside [`Python::detach`] — the
//! `Python::allow_threads` of pyo3 < 0.26, renamed for free-threading
//! terminology — so the interpreter lock is released for the duration of
//! every dictionary load, lookup and suggestion run: other Python threads
//! keep running while the engine works. The choice is safe because the
//! engine is plain owned data ([`Dictionary`] is `Send + Sync`, no
//! interior mutability) and touches no Python objects; the result rows
//! are materialized only after the call returns, back under the GIL. On
//! free-threaded builds (`detach` detaches the thread state) the same
//! holds: the engine never re-enters Python.
//!
//! pyo3 types stop at this module: the engine modules stay pure Rust
//! (P0 MECE policy), exactly like the C ABI in [`super::c`] and the
//! magnus types in [`super::ruby`].

use std::path::PathBuf;

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule, PyType};

use crate::dict::Dictionary;
use crate::suggest::Suggestion;

/// Default suggestion limit when `suggest` is called without one — the
/// gem's `Spellchecker#suggest` default (mirrors `ffi::ruby` and
/// `ffi::wasm`).
const DEFAULT_SUGGEST_LIMIT: usize = 5;

// The `kotoshu_native.KotoshuNativeError` exception raised for every
// engine failure crossing the boundary, carrying the Rust message (the
// Python twin of `Kotoshu::Native::Error` and `ffi::wasm`'s `JsError`
// rejections). Doc comment as a `//` comment: rustdoc does not document
// macro invocations.
create_exception!(
    kotoshu_native,
    KotoshuNativeError,
    PyException,
    "Engine failure from the kotoshu_native extension; the message is the Rust error."
);

/// One loaded dictionary wrapped as a Python object: the Python twin of
/// the `ruby` feature's `Kotoshu::Native::Dictionary` and the `wasm`
/// feature's `KotoshuWasm` — same engine, same suggestion-row shape.
/// Python's refcounting owns it once wrapped; dropping the Python object
/// drops the engine dictionary.
#[derive(Debug)]
#[pyclass(module = "kotoshu_native", name = "Dictionary")]
pub struct PythonDictionary {
    inner: Dictionary,
}

/// One suggestion row: a fresh `dict` with exactly the conformance
/// `SUGGESTION_KEYS`. Setting known keys on a fresh dict cannot fail; a
/// panic here would mean engine misuse, not bad input (the same stance as
/// `ffi::wasm`'s `Reflect::set`).
fn suggestion_row<'py>(py: Python<'py>, suggestion: &Suggestion) -> Bound<'py, PyDict> {
    let row = PyDict::new(py);
    row.set_item("word", suggestion.word.as_str())
        .expect("set_item on a fresh dict");
    row.set_item("distance", suggestion.distance)
        .expect("set_item on a fresh dict");
    row.set_item("confidence", suggestion.confidence)
        .expect("set_item on a fresh dict");
    row.set_item("source", suggestion.source.as_str())
        .expect("set_item on a fresh dict");
    row
}

#[pymethods]
impl PythonDictionary {
    /// `Dictionary` has no public constructor: instances exist only via
    /// [`Dictionary::load`] (the `ruby` feature undefines `allocate` for
    /// the same reason — a dictionary-less instance is meaningless).
    #[new]
    fn new() -> PyResult<Self> {
        Err(KotoshuNativeError::new_err(
            "Dictionary has no public constructor; use Dictionary.load(aff_path, dic_path)",
        ))
    }

    /// `Dictionary.load(aff_path, dic_path)` — loads the `.aff`/`.dic`
    /// pair with [`Dictionary::load`] and wraps the result, with the GIL
    /// released for the load (see the module docs). Failures raise
    /// [`KotoshuNativeError`] naming the paths and the Rust error.
    #[classmethod]
    fn load(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        aff_path: String,
        dic_path: String,
    ) -> PyResult<Self> {
        let (aff, dic) = (PathBuf::from(&aff_path), PathBuf::from(&dic_path));
        let dictionary = py
            .detach(move || Dictionary::load(&aff, &dic))
            .map_err(|error| {
                KotoshuNativeError::new_err(format!(
                    "failed to load dictionary ({aff_path}, {dic_path}): {error}"
                ))
            })?;
        Ok(Self { inner: dictionary })
    }

    /// `Dictionary.correct(word)` — [`Dictionary::correct`] (the gem's
    /// `correct?`). The GIL is released for the lookup.
    fn correct(&self, py: Python<'_>, word: String) -> bool {
        py.detach(|| self.inner.correct(&word))
    }

    /// `Dictionary.suggest(word, limit = 5)` — [`Dictionary::suggest`],
    /// one dict per suggestion (see the module docs for the row shape).
    /// The GIL is released for the engine call only; the rows are built
    /// back under it.
    #[pyo3(signature = (word, limit = None))]
    fn suggest<'py>(
        &self,
        py: Python<'py>,
        word: String,
        limit: Option<usize>,
    ) -> Vec<Bound<'py, PyDict>> {
        let limit = limit.unwrap_or(DEFAULT_SUGGEST_LIMIT);
        let dictionary = &self.inner;
        let suggestions = py.detach(|| dictionary.suggest(&word, limit));
        suggestions
            .into_iter()
            .map(|suggestion| suggestion_row(py, &suggestion))
            .collect()
    }
}

/// `kotoshu_native.available()` — the Python twin of the gem's
/// `Kotoshu::Native.available?` native-backend guard: `True` whenever
/// this extension is imported.
#[pyfunction]
fn available() -> bool {
    true
}

/// Define the `kotoshu_native` module surface (see the module docs).
/// Called by the package shim's `#[pymodule]`; see `kotoshu-python` for
/// the reference shim.
///
/// Idempotent at the Python level is not needed (CPython calls module
/// init exactly once per interpreter), so unlike `ffi::ruby::init` there
/// is no re-definition concern to document.
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("VERSION", env!("CARGO_PKG_VERSION"))?;
    module.add(
        "KotoshuNativeError",
        module.py().get_type::<KotoshuNativeError>(),
    )?;
    module.add_class::<PythonDictionary>()?;
    module.add_function(wrap_pyfunction!(available, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_dictionary_is_send_and_sync() {
        // `Python::detach` hands the engine to GIL-free threads; keep the
        // engine's plain-data guarantee visible in this module's own test
        // output (mirrors ffi/ruby's Send assertion).
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<PythonDictionary>();
        assert_send_sync::<Dictionary>();
    }

    #[test]
    fn version_is_the_crate_version() {
        assert!(
            env!("CARGO_PKG_VERSION")
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_digit())
        );
    }
}
