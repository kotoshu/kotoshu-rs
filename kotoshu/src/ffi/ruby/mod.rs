//! Ruby bindings (feature `ruby`): the engine behind a
//! `Kotoshu::Native` module, built with [magnus].
//!
//! [magnus]: https://crates.io/crates/magnus
//!
//! # Shim pattern (parsanol blueprint, plan 66)
//!
//! This crate is an `rlib`; it never defines a Ruby `Init` entry point.
//! The per-language gem builds a tiny `cdylib` whose own `#[magnus::init]`
//! does nothing but forward:
//!
//! ```ignore
//! #[magnus::init]
//! fn init(ruby: &magnus::Ruby) -> Result<(), magnus::Error> {
//!     kotoshu::ffi::ruby::init(ruby)
//! }
//! ```
//!
//! [`tests/ruby_ext`](../../../tests/ruby_ext) in this repository is exactly
//! that shim, exercised as a smoke test (`scripts/ruby_ffi_smoke.sh`).
//!
//! # Exposed API
//!
//! ```ruby
//! Kotoshu::Native::VERSION                  # => "0.1.0" (kotoshu crate version)
//! Kotoshu::Native.available?                # => true
//!
//! dictionary = Kotoshu::Native::Dictionary.load(aff_path, dic_path)
//! dictionary.correct?("hello")              # => true / false
//! dictionary.suggest("hlelo", 5)            # => [ { "word" => "hello", ... } ]
//! dictionary.suggest("hlelo")               # limit defaults to 5
//! ```
//!
//! `suggest` returns one [`magnus::RHash`] per suggestion with the four keys
//! of the gem's `Kotoshu::Suggestions::Suggestion` / the conformance
//! `SUGGESTION_KEYS`: `"word"` (String), `"distance"` (Integer),
//! `"confidence"` (Float in `[0, 1]`), `"source"` (String, one of
//! `edit_distance`, `phonetic`, `keyboard_proximity`, `ngram`). The gem-side
//! wrapper materializes its `Suggestion` objects from these hashes.
//!
//! All engine failures surface as `Kotoshu::Native::Error` (a
//! `RuntimeError` subclass) carrying the Rust error message.
//!
//! Magnus types stop at this module: the engine modules stay pure Rust
//! (P0 MECE policy), exactly like the C ABI in [`super::c`].

use std::path::Path;

use magnus::scan_args::scan_args;
use magnus::typed_data::Obj;
use magnus::value::Lazy;
use magnus::{
    Class, DataType, DataTypeFunctions, Error, ExceptionClass, Module, Object, RArray, RClass,
    RModule, Ruby, TypedData, data_type_builder, function, method,
};

use crate::dict::{Dictionary, LoadError};

/// Default suggestion limit when `suggest` is called without one — the
/// gem's `Spellchecker#suggest` default.
const DEFAULT_SUGGEST_LIMIT: usize = 5;

/// The `Kotoshu` module, defined (idempotently) on first access. The gem's
/// pure-Ruby `Kotoshu` module and this definition are the same constant:
/// `rb_define_module` returns the existing module when already defined.
fn kotoshu_module(ruby: &Ruby) -> RModule {
    static MODULE: Lazy<RModule> = Lazy::new(|ruby| {
        ruby.define_module("Kotoshu")
            .expect("cannot define Kotoshu module")
    });
    ruby.get_inner(&MODULE)
}

/// The `Kotoshu::Native` module hosting every binding this file defines.
fn native_module(ruby: &Ruby) -> RModule {
    static MODULE: Lazy<RModule> = Lazy::new(|ruby| {
        kotoshu_module(ruby)
            .define_module("Native")
            .expect("cannot define Kotoshu::Native module")
    });
    ruby.get_inner(&MODULE)
}

/// The `Kotoshu::Native::Error` exception class (RuntimeError subclass)
/// raised for every engine failure crossing the boundary.
fn error_class(ruby: &Ruby) -> ExceptionClass {
    static CLASS: Lazy<ExceptionClass> = Lazy::new(|ruby| {
        native_module(ruby)
            .define_error("Error", ruby.exception_runtime_error())
            .expect("cannot define Kotoshu::Native::Error")
    });
    ruby.get_inner(&CLASS)
}

/// A loaded dictionary wrapped as a Ruby object. Ruby's GC owns it once
/// wrapped; dropping the Ruby object drops the engine dictionary.
#[derive(Debug)]
pub struct RubyDictionary {
    inner: Dictionary,
}

// The engine dictionary is plain owned data (parsed aff/dic structures);
// Ruby objects are handed between threads only with the GVL held, which
// `TypedData`'s `Send` bound models.
impl DataTypeFunctions for RubyDictionary {}

unsafe impl TypedData for RubyDictionary {
    fn class(ruby: &Ruby) -> RClass {
        static CLASS: Lazy<RClass> = Lazy::new(|ruby| {
            let class = native_module(ruby)
                .define_class("Dictionary", ruby.class_object())
                .expect("cannot define Kotoshu::Native::Dictionary");
            // Instances exist only through `Dictionary.load`; `new`/`allocate`
            // would produce a dictionary-less object.
            class.undef_default_alloc_func();
            class
        });
        ruby.get_inner(&CLASS)
    }

    fn data_type() -> &'static DataType {
        static DATA_TYPE: DataType =
            data_type_builder!(RubyDictionary, "Kotoshu/Native/Dictionary").build();
        &DATA_TYPE
    }
}

fn load_error(ruby: &Ruby, aff_path: &str, dic_path: &str, error: LoadError) -> Error {
    Error::new(
        error_class(ruby),
        format!("failed to load dictionary ({aff_path}, {dic_path}): {error}"),
    )
}

/// `Kotoshu::Native::Dictionary.load(aff_path, dic_path)` — loads the
/// `.aff`/`.dic` pair with [`Dictionary::load`] and wraps the result.
fn dictionary_load(
    ruby: &Ruby,
    _class: RClass,
    aff_path: String,
    dic_path: String,
) -> Result<Obj<RubyDictionary>, Error> {
    let dictionary = Dictionary::load(Path::new(&aff_path), Path::new(&dic_path))
        .map_err(|error| load_error(ruby, &aff_path, &dic_path, error))?;
    Ok(ruby.obj_wrap(RubyDictionary { inner: dictionary }))
}

/// `Kotoshu::Native::Dictionary#correct?(word)` — [`Dictionary::correct`].
fn dictionary_correct(rb_self: &RubyDictionary, word: String) -> Result<bool, Error> {
    Ok(rb_self.inner.correct(&word))
}

/// `Kotoshu::Native::Dictionary#suggest(word, limit = 5)` —
/// [`Dictionary::suggest`], one hash per suggestion (see the module docs
/// for the row shape).
fn dictionary_suggest(
    ruby: &Ruby,
    rb_self: &RubyDictionary,
    args: &[magnus::Value],
) -> Result<RArray, Error> {
    let scanned = scan_args::<(String,), (Option<usize>,), (), (), (), ()>(args)?;
    let (word,) = scanned.required;
    let (limit,) = scanned.optional;
    let limit = limit.unwrap_or(DEFAULT_SUGGEST_LIMIT);

    let suggestions = rb_self.inner.suggest(&word, limit);
    let array = ruby.ary_new_capa(suggestions.len());
    for suggestion in suggestions {
        let hash = ruby.hash_new();
        hash.aset("word", suggestion.word.as_str())?;
        hash.aset("distance", i64::from(suggestion.distance))?;
        hash.aset("confidence", suggestion.confidence)?;
        hash.aset("source", suggestion.source.as_str())?;
        array.push(hash)?;
    }
    Ok(array)
}

/// `Kotoshu::Native.available?` — the gem's native-backend guard: true
/// whenever this extension is loaded.
fn is_available() -> bool {
    true
}

/// Define `Kotoshu::Native` with the full binding surface (see the module
/// docs). Called by the per-language gem shim's `#[magnus::init]`; see
/// [`tests/ruby_ext`](../../../tests/ruby_ext) for the reference shim.
///
/// Idempotent at the Ruby level (module/class definitions return the
/// existing constant), so a double `init` from a misbehaving host is not
/// fatal; method re-definition is likewise a no-op rebind.
pub fn init(ruby: &Ruby) -> Result<(), Error> {
    let native = native_module(ruby);
    native.const_set("VERSION", env!("CARGO_PKG_VERSION"))?;

    // Defined eagerly so the exception class exists even if the first
    // failure happens before any `load` call.
    error_class(ruby);

    let class = RubyDictionary::class(ruby);
    class.define_singleton_method("load", method!(dictionary_load, 2))?;
    class.define_method("correct?", method!(dictionary_correct, 1))?;
    class.define_method("suggest", method!(dictionary_suggest, -1))?;

    native.define_module_function("available?", function!(is_available, 0))?;

    #[cfg(feature = "model")]
    typo::init(ruby)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ruby_dictionary_is_send() {
        // TypedData requires Send; keep the engine's plain-data guarantee
        // visible in this module's own test output.
        fn assert_send<T: Send>() {}
        assert_send::<RubyDictionary>();
        assert_send::<Dictionary>();
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

// The typo-retrieval layer (plan 131), exposed when the `model`
// feature rides along. Three objects, mirroring the wasm surface:
//
// ```ruby
// typo = Kotoshu::Native::TypoModel.load(onnx_path, char_vocab_path)
// tier = Kotoshu::Native::TypoTier.load(tier_onnx_path, tier_vocab_path)
// engine = Kotoshu::Native::TypoEngine.new(typo, tier)
// engine.typo_suggest("helo", 5)  # => [{ "word" => ..., "score" => ... }]
// ```
//
// The derived vocabulary index builds lazily on the first suggest
// (one forward pass per tier word, parallel; ~15 s for the 100k full
// tier) and is reused for the engine's lifetime. `typo_suggest`
// returns `[]` for a typo outside the tier vocabulary — the hybrid's
// honest in-vocab rule.
#[cfg(feature = "model")]
pub mod typo {

    use magnus::typed_data::Obj;
    use magnus::value::Lazy;
    use magnus::{
        Class, DataType, DataTypeFunctions, Error, Module, Object, RArray, RClass, Ruby, TypedData,
        data_type_builder, method,
    };

    use crate::rerank::int8_model::Int8Model;
    use crate::typo::model::TypoModel as EngineTypoModel;
    use crate::typo::{TypoEngine, TypoModelError};

    fn native_module(ruby: &Ruby) -> magnus::RModule {
        static NATIVE: Lazy<magnus::RModule> = Lazy::new(|ruby| {
            ruby.define_module("Kotoshu")
                .expect("cannot define Kotoshu")
                .define_module("Native")
                .expect("cannot define Kotoshu::Native")
        });
        ruby.get_inner(&NATIVE)
    }

    fn error_class(ruby: &Ruby) -> magnus::ExceptionClass {
        static ERROR: Lazy<magnus::ExceptionClass> = Lazy::new(|ruby| {
            native_module(ruby)
                .define_error("Error", ruby.exception_runtime_error())
                .expect("cannot define Kotoshu::Native::Error")
        });
        ruby.get_inner(&ERROR)
    }

    fn model_error(ruby: &Ruby, error: TypoModelError) -> Error {
        Error::new(error_class(ruby), error.to_string())
    }

    /// Run a pure-Rust computation with the GVL released. The typo
    /// layer's index build takes seconds and its scan milliseconds;
    /// held under the GVL, one arming request would stop every other
    /// Ruby thread for the duration. The compute touches no Ruby
    /// objects — the payload box round-trips through
    /// `rb_thread_call_without_gvl`, panics are caught and re-raised
    /// as `Kotoshu::Native::Error` on re-entry, and the no-op
    /// unblock function keeps Ruby schedulable throughout.
    fn without_gvl<'f, T: Send>(
        ruby: &Ruby,
        f: impl FnOnce() -> T + Send + 'f,
    ) -> Result<T, Error> {
        use std::ffi::c_void;

        struct Payload<'f, T> {
            f: Option<Box<dyn FnOnce() -> T + Send + 'f>>,
            out: Option<T>,
            panic: Option<String>,
        }

        // 'f rides the payload pointer, not the fn signature (late-
        // bound lifetimes cannot be named in fn pointers)
        #[allow(clippy::extra_unused_lifetimes)]
        extern "C" fn trampoline<'f, T: Send>(data: *mut c_void) -> *mut c_void {
            unsafe {
                let payload = &mut *(data as *mut Payload<'f, T>);
                let job = payload.f.take().expect("payload runs once");
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(job)) {
                    Ok(value) => payload.out = Some(value),
                    Err(panic) => {
                        let message = panic
                            .downcast_ref::<String>()
                            .cloned()
                            .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_owned()))
                            .unwrap_or_else(|| "panic inside GVL-released compute".to_owned());
                        payload.panic = Some(message);
                    }
                }
                std::ptr::null_mut()
            }
        }

        extern "C" fn ubf(_data: *mut c_void) {
            // The compute is uninterruptible by design (a
            // deterministic index build); Ruby stays scheduled while
            // it runs.
        }

        let mut payload = Payload {
            f: Some(Box::new(f)),
            out: None,
            panic: None,
        };
        unsafe {
            ::rb_sys::rb_thread_call_without_gvl(
                Some(
                    trampoline::<T>
                        as unsafe extern "C" fn(*mut std::ffi::c_void) -> *mut std::ffi::c_void,
                ),
                (&mut payload as *mut Payload<'f, T>).cast(),
                Some(ubf),
                std::ptr::null_mut(),
            );
        }
        if let Some(message) = payload.panic {
            return Err(Error::new(
                error_class(ruby),
                format!("GVL-released compute panicked: {message}"),
            ));
        }
        Ok(payload.out.take().expect("payload ran"))
    }

    /// The loaded bi-encoder artifact pair.
    pub struct RubyTypoModel {
        inner: EngineTypoModel,
    }

    impl DataTypeFunctions for RubyTypoModel {}

    unsafe impl TypedData for RubyTypoModel {
        fn class(ruby: &Ruby) -> RClass {
            static CLASS: Lazy<RClass> = Lazy::new(|ruby| {
                let class = native_module(ruby)
                    .define_class("TypoModel", ruby.class_object())
                    .expect("cannot define Kotoshu::Native::TypoModel");
                class.undef_default_alloc_func();
                class
            });
            ruby.get_inner(&CLASS)
        }

        fn data_type() -> &'static DataType {
            static DATA_TYPE: DataType =
                data_type_builder!(RubyTypoModel, "Kotoshu/Native/TypoModel").build();
            &DATA_TYPE
        }
    }

    /// The fastText tier the hybrid rescores with (int8-per-row or the
    /// fp32 full tier — the rescore is fp32-exact by contract).
    #[derive(Debug)]
    pub struct RubyTypoTier {
        inner: Int8Model,
    }

    impl DataTypeFunctions for RubyTypoTier {}

    unsafe impl TypedData for RubyTypoTier {
        fn class(ruby: &Ruby) -> RClass {
            static CLASS: Lazy<RClass> = Lazy::new(|ruby| {
                let class = native_module(ruby)
                    .define_class("TypoTier", ruby.class_object())
                    .expect("cannot define Kotoshu::Native::TypoTier");
                class.undef_default_alloc_func();
                class
            });
            ruby.get_inner(&CLASS)
        }

        fn data_type() -> &'static DataType {
            static DATA_TYPE: DataType =
                data_type_builder!(RubyTypoTier, "Kotoshu/Native/TypoTier").build();
            &DATA_TYPE
        }
    }

    /// The compose engine: bi-encoder retrieval + tier rescore — the
    /// same `TypoEngine` the wasm surface holds (one compose
    /// implementation everywhere).
    pub struct RubyTypoEngine {
        engine: TypoEngine,
        tier: Int8Model,
    }

    impl DataTypeFunctions for RubyTypoEngine {}

    unsafe impl TypedData for RubyTypoEngine {
        fn class(ruby: &Ruby) -> RClass {
            static CLASS: Lazy<RClass> = Lazy::new(|ruby| {
                let class = native_module(ruby)
                    .define_class("TypoEngine", ruby.class_object())
                    .expect("cannot define Kotoshu::Native::TypoEngine");
                class.undef_default_alloc_func();
                class
            });
            ruby.get_inner(&CLASS)
        }

        fn data_type() -> &'static DataType {
            static DATA_TYPE: DataType =
                data_type_builder!(RubyTypoEngine, "Kotoshu/Native/TypoEngine").build();
            &DATA_TYPE
        }
    }

    fn typo_model_load(
        ruby: &Ruby,
        _class: RClass,
        onnx_path: String,
        vocab_path: String,
    ) -> Result<Obj<RubyTypoModel>, Error> {
        let onnx = std::fs::read(&onnx_path).map_err(|error| {
            Error::new(
                error_class(ruby),
                format!("cannot read {onnx_path}: {error}"),
            )
        })?;
        let vocab = std::fs::read(&vocab_path).map_err(|error| {
            Error::new(
                error_class(ruby),
                format!("cannot read {vocab_path}: {error}"),
            )
        })?;
        let inner =
            EngineTypoModel::parse(&onnx, &vocab).map_err(|error| model_error(ruby, error))?;
        Ok(ruby.obj_wrap(RubyTypoModel { inner }))
    }

    fn tier_load(
        ruby: &Ruby,
        _class: RClass,
        onnx_path: String,
        vocab_path: String,
    ) -> Result<Obj<RubyTypoTier>, Error> {
        let read = |path: &str| {
            std::fs::read(path).map_err(|error| {
                Error::new(error_class(ruby), format!("cannot read {path}: {error}"))
            })
        };
        let inner = Int8Model::parse(&read(&onnx_path)?, &read(&vocab_path)?).map_err(|error| {
            Error::new(
                error_class(ruby),
                format!("failed to load tier ({onnx_path}, {vocab_path}): {error}"),
            )
        })?;
        Ok(ruby.obj_wrap(RubyTypoTier { inner }))
    }

    fn typo_engine_new(
        ruby: &Ruby,
        _class: RClass,
        typo: Obj<RubyTypoModel>,
        tier: Obj<RubyTypoTier>,
    ) -> Result<Obj<RubyTypoEngine>, Error> {
        Ok(ruby.obj_wrap(RubyTypoEngine {
            engine: TypoEngine::new(typo.inner.clone()),
            tier: tier.inner.clone(),
        }))
    }

    fn typo_engine_suggest(
        ruby: &Ruby,
        engine: &RubyTypoEngine,
        word: String,
    ) -> Result<RArray, Error> {
        // capture the Sync engine fields, not the TypedData wrapper
        // (the compute runs on this thread with the GVL released)
        let (compose, tier) = (&engine.engine, &engine.tier);
        let rows = without_gvl(ruby, || compose.suggest(tier, &word, crate::typo::SLATE))?;
        let out = ruby.ary_new_capa(rows.as_ref().map_or(0, Vec::len) as _);
        if let Some(rows) = rows {
            for (word, score) in rows {
                let row = ruby.hash_new();
                row.aset("word", word.as_str())
                    .and_then(|()| row.aset("score", score))
                    .map_err(|error| Error::new(error_class(ruby), error.to_string()))?;
                out.push(row)
                    .map_err(|error| Error::new(error_class(ruby), error.to_string()))?;
            }
        }
        Ok(out)
    }

    /// Define the typo classes on `Kotoshu::Native` (idempotent).
    pub fn init(ruby: &Ruby) -> Result<(), Error> {
        let typo_class = RubyTypoModel::class(ruby);
        typo_class.define_singleton_method("load", method!(typo_model_load, 2))?;

        let tier_class = RubyTypoTier::class(ruby);
        tier_class.define_singleton_method("load", method!(tier_load, 2))?;

        let engine_class = RubyTypoEngine::class(ruby);
        engine_class.define_singleton_method("new", method!(typo_engine_new, 2))?;
        engine_class.define_method("typo_suggest", method!(typo_engine_suggest, 1))?;
        engine_class.define_method("build_index", method!(typo_engine_build_index, 1))?;
        Ok(())
    }

    /// `TypoEngine#build_index(tier)` — derive the per-language index
    /// NOW, with the GVL released: the cost lands at setup where it
    /// belongs, instead of stalling the first suggest (and the whole
    /// process under the GVL). The OnceLock inside `suggest` answers
    /// instantly afterwards; building again is a no-op.
    fn typo_engine_build_index(
        ruby: &Ruby,
        engine: &RubyTypoEngine,
        tier: Obj<RubyTypoTier>,
    ) -> Result<usize, Error> {
        let (compose, tier) = (&engine.engine, &tier.inner);
        let len = without_gvl(ruby, move || compose.derived_index(tier).len())?;
        Ok(len)
    }
}
