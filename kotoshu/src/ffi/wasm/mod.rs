//! WASM bindings (feature `wasm`): the engine behind a `KotoshuWasm` JS
//! class, built with wasm-bindgen and packaged as `@kotoshu/wasm` (build
//! half only — npm publication is blocked on org credentials, plan 67 M5;
//! see `kotoshu-wasm/RELEASING.md`).
//!
//! # Shim pattern (parsanol blueprint, plan 66)
//!
//! This module defines the surface; the `kotoshu-wasm` workspace member is
//! the thin cdylib wasm-pack builds — the same shape as the `ruby`
//! feature's `tests/ruby_ext` reference shim, minus the workspace
//! exclusion (wasm-pack needs a package to build), with its own opt-in
//! `wasm` feature so default workspace builds stay dependency-free (P0
//! policy). Parsanol keeps the surface inside its single core crate and
//! runs wasm-pack against it; kotoshu needs the member because the core
//! stays a pure `rlib` consumed by every other binding too.
//!
//! # Exposed API
//!
//! ```js
//! import init, { KotoshuWasm } from "@kotoshu/wasm";
//! await init(); // or: await init(wasmBytes)
//!
//! KotoshuWasm.VERSION; // => "0.1.0" (kotoshu crate version)
//!
//! const aff = "..."; // .aff file CONTENTS, not a path (wasm has no fs)
//! const dic = "..."; // .dic file contents
//! const dictionary = new KotoshuWasm(aff, dic);
//! dictionary.correct("hello");    // => true / false
//! dictionary.suggest("hlelo", 5); // => [{ word: "hello", distance: 1,
//!                                 //      confidence: 1.0,
//!                                 //      source: "edit_distance" }, ...]
//! ```
//!
//! `suggest` returns one plain object per suggestion with exactly the four
//! keys of the gem's `Kotoshu::Suggestions::Suggestion` / the conformance
//! `SUGGESTION_KEYS`: `word` (string), `distance` (number),
//! `confidence` (number in `[0, 1]`) and `source` (string, one of
//! `edit_distance`, `phonetic`, `keyboard_proximity`, `ngram`) — the same
//! row shape `ffi::ruby` hashes and the frozen vectors use. `limit` may be
//! omitted (defaults to 5, the gem's `Spellchecker#suggest` default).
//!
//! Engine failures reject the constructor with a `JsError` carrying the
//! Rust message. Panics surface on `console.error` verbatim via
//! console_error_panic_hook, installed at module start and again in the
//! constructor (`set_once` is idempotent) — panic messages are never
//! swallowed.
//!
//! wasm-bindgen/js-sys types stop at this module: the engine modules stay
//! pure Rust (P0 MECE policy), exactly like the C ABI in `ffi::c` and the
//! magnus types in `ffi::ruby`.

use js_sys::{Array, Object, Reflect};
use wasm_bindgen::{JsError, prelude::*};

use crate::dict::Dictionary;

/// Default suggestion limit when `suggest` is called without one — the
/// gem's `Spellchecker#suggest` default (mirrors `ffi::ruby`).
const DEFAULT_SUGGEST_LIMIT: usize = 5;

/// Route panics to `console.error` with the full message, at module start
/// (the generated bundler glue calls the start export at import time).
/// `private`: the hook must not leak into the public JS surface.
#[wasm_bindgen(start, private)]
fn install_panic_hook() {
    console_error_panic_hook::set_once();
}

/// One loaded dictionary: the JS twin of the `ruby` feature's
/// `Kotoshu::Native::Dictionary` — same engine, same suggestion-row shape.
#[wasm_bindgen]
pub struct KotoshuWasm {
    dictionary: Dictionary,
}

/// One suggestion row: a fresh plain object with `fields` set. Setting a
/// plain-named field on a fresh object cannot fail; a panic here (routed to
/// `console.error`) would mean engine misuse, not bad input.
fn suggestion_row(fields: &[(&str, JsValue)]) -> JsValue {
    let row = Object::new();
    for (key, value) in fields {
        Reflect::set(&row, &JsValue::from(*key), value).expect("Reflect::set on a fresh object");
    }
    row.into()
}

#[wasm_bindgen]
impl KotoshuWasm {
    /// The engine (kotoshu crate) version. The npm package version line is
    /// independent per package (parsanol policy) — its `package.json`
    /// governs the published version, not this constant.
    #[wasm_bindgen(js_name = "VERSION", getter)]
    pub fn version() -> String {
        env!("CARGO_PKG_VERSION").to_owned()
    }

    /// Load a dictionary from the string CONTENTS of its `.aff` and `.dic`
    /// sources (wasm has no filesystem). Byte-symmetric with a path load:
    /// the pair is decoded per the `.aff` `SET` line exactly like
    /// [`Dictionary::load`], so hosts holding UTF-8 sources pass them
    /// through verbatim; hosts holding legacy-encoded dictionaries decode
    /// them per `SET` first (and hand the `.aff` over as UTF-8).
    ///
    /// Failures reject with the Rust error message.
    #[wasm_bindgen(constructor)]
    pub fn new(aff_source: &str, dic_source: &str) -> Result<KotoshuWasm, JsError> {
        // Idempotent — also installed at module start; the constructor is
        // the first thing every caller runs, so the hook is in place
        // before any engine work even if the start function was skipped.
        console_error_panic_hook::set_once();
        Dictionary::load_from_sources(aff_source, dic_source)
            .map(|dictionary| Self { dictionary })
            .map_err(|error| JsError::new(&error.to_string()))
    }

    /// Whether `word` is spelled correctly per this dictionary (the gem's
    /// `correct?`).
    pub fn correct(&self, word: &str) -> bool {
        self.dictionary.correct(word)
    }

    /// Ranked suggestions for `word` (the gem's `suggest`): a plain array,
    /// one row per suggestion — see the module docs for the row shape.
    /// `limit` defaults to 5 when omitted.
    pub fn suggest(&self, word: &str, limit: Option<usize>) -> Array {
        let limit = limit.unwrap_or(DEFAULT_SUGGEST_LIMIT);
        self.dictionary
            .suggest(word, limit)
            .into_iter()
            .map(|suggestion| {
                suggestion_row(&[
                    ("word", JsValue::from(suggestion.word)),
                    ("distance", JsValue::from(suggestion.distance)),
                    ("confidence", JsValue::from(suggestion.confidence)),
                    ("source", JsValue::from(suggestion.source.as_str())),
                ])
            })
            .collect()
    }
}
