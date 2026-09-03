//! The `ort` [`EmbeddingProvider`] — feature `onnx`.
//!
//! Loads a tier artifact pair (`.onnx` graph + `.vocab.json` sibling)
//! and answers [`EmbeddingProvider::embedding`] with one `session.run`
//! per word. `ort` is built with `load-dynamic` only: this library
//! never links a bundled onnxruntime — the host supplies
//! `libonnxruntime` via the `KOTOSHU_ORT_DYLIB` environment variable
//! (full path to the shared library) or, failing that, ort's own
//! default search (which also honors `ORT_DYLIB_PATH`). CI installs
//! onnxruntime via pip and exports `KOTOSHU_ORT_DYLIB`.
//!
//! Graph shapes accepted (both produced by the models repo):
//!
//! - **full**: `Constant word_embeddings` fp32 `[V, d]` + `Gather` +
//!   `Squeeze`; no `quantization` metadata.
//! - **mini/fluency** (int8-per-row): `Constant q_embeddings` int8
//!   `[V, d]` + `row_scale` fp32 `[V]` + `Gather + Gather + Reshape +
//!   `Cast + Mul + Squeeze`; metadata `quantization = "int8-per-row"`.
//!
//! Both share input `word_index` int64 `[1]` and output `embedding`
//! fp32 `[d]`, and write `embedding_dimension` metadata. A future
//! int4-per-row variant is accepted by its metadata string via
//! [`crate::rerank::dequant::RowFormat`] (B1 groundwork; artifacts do
//! not exist yet).

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use crate::rerank::dequant::RowFormat;
use crate::rerank::{EmbeddingProvider, oov};

/// Environment variable holding the full path to `libonnxruntime`
/// (checked before ort's own default search, which honors
/// `ORT_DYLIB_PATH`).
pub const DYLIB_ENV: &str = "KOTOSHU_ORT_DYLIB";

/// Errors of the ort provider.
#[derive(Debug)]
pub enum OrtError {
    /// The onnxruntime shared library could not be loaded (missing or
    /// unloadable `KOTOSHU_ORT_DYLIB`, or the default search found
    /// nothing).
    Dylib(String),
    /// The `.onnx` file could not be opened.
    OpenModel { path: PathBuf, source: String },
    /// The `.vocab.json` sibling could not be read or parsed.
    Vocab { path: PathBuf, source: String },
    /// Required graph metadata is missing or malformed.
    Metadata(String),
    /// The graph's quantization descriptor is unknown.
    UnsupportedQuantization(String),
    /// A session run failed.
    Inference(String),
}

impl fmt::Display for OrtError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dylib(source) => write!(
                f,
                "onnxruntime unavailable: {source}. Set {DYLIB_ENV} to the full path of \
                 libonnxruntime (e.g. the onnxruntime pip package's capi/libonnxruntime), \
                 or make it findable by ort's default search (ORT_DYLIB_PATH)"
            ),
            Self::OpenModel { path, source } => {
                write!(f, "cannot open model {}: {source}", path.display())
            }
            Self::Vocab { path, source } => {
                write!(f, "cannot parse vocabulary {}: {source}", path.display())
            }
            Self::Metadata(source) => write!(f, "model metadata: {source}"),
            Self::UnsupportedQuantization(descriptor) => write!(
                f,
                "unsupported quantization {descriptor:?} (known: absent, \
                 int8-per-row, int4-per-row)"
            ),
            Self::Inference(source) => write!(f, "inference failed: {source}"),
        }
    }
}

impl std::error::Error for OrtError {}

/// The ort-backed embedding provider over one tier artifact.
pub struct OrtProvider {
    /// `Session::run` takes `&mut self`; the provider trait hands out
    /// `&self`, so the session sits behind a mutex (providers are
    /// shareable across threads).
    session: Mutex<ort::session::Session>,
    vocab: HashMap<String, i64>,
    dims: usize,
    format: RowFormat,
    vocab_size: usize,
}

/// One-time per-process ort initialization (the dylib is loaded here,
/// not per session). Resolution order: `KOTOSHU_ORT_DYLIB` (anchored
/// against the working directory when relative), then ort's own default
/// search (`ORT_DYLIB_PATH`, then the platform library name).
///
/// ort panics — via an internal `expect` — when its *default* search
/// finds no onnxruntime; the panic is caught here so callers get an
/// `Err` and tests can skip cleanly. An explicit path (either env var)
/// fails with a proper error instead.
fn init_dylib() -> Result<(), OrtError> {
    static INIT: OnceLock<Result<(), String>> = OnceLock::new();
    INIT.get_or_init(|| {
        let explicit = std::env::var_os(DYLIB_ENV)
            .filter(|value| !value.is_empty())
            .map(|path| {
                std::path::absolute(&path).map_err(|error| format!("{DYLIB_ENV}={path:?}: {error}"))
            })
            .transpose();

        let commit = match explicit {
            Err(error) => return Err(error),
            // Explicit KOTOSHU_ORT_DYLIB: init_from fails with an error
            // (not a panic) when the library cannot load.
            Ok(Some(absolute)) => ort::init_from(&absolute)
                .map_err(|error| error.to_string())?
                .with_name("kotoshu")
                .commit(),
            // Default search: silence the panic hook for the duration of
            // the probe (ort's internal expect), then restore it. ort
            // defers the dylib load to the first API call, so the probe
            // must force environment creation — `Environment::current`
            // — inside the window, not just commit the builder.
            Ok(None) => {
                let previous_hook = std::panic::take_hook();
                std::panic::set_hook(Box::new(|_| {}));
                let probe = std::panic::catch_unwind(|| {
                    ort::init().with_name("kotoshu").commit();
                    ort::environment::Environment::current().map(|_| ())
                });
                std::panic::set_hook(previous_hook);
                match probe {
                    // A proper error (dylib loaded but env creation
                    // failed) still beats a panic.
                    Ok(Ok(())) => true,
                    Ok(Err(error)) => return Err(error.to_string()),
                    Err(payload) => {
                        let message = payload
                            .downcast_ref::<&str>()
                            .map(|s| (*s).to_owned())
                            .or_else(|| payload.downcast_ref::<String>().cloned())
                            .unwrap_or_else(|| "ort could not load libonnxruntime".to_owned());
                        return Err(format!(
                            "default library search failed ({message}); expected \
                             libonnxruntime on the loader path, ORT_DYLIB_PATH, or {DYLIB_ENV}"
                        ));
                    }
                }
            }
        };
        // `false` only means an environment was already configured
        // (by the host or an earlier call) — not an error.
        let _ = commit;
        Ok(())
    })
    .clone()
    .map_err(OrtError::Dylib)
}

/// Probe whether the onnxruntime shared library can be loaded — the
/// clean-skip gate for tests: `Err(message)` explains why not.
pub fn dylib_available() -> Result<(), String> {
    if let Some(path) = std::env::var_os(DYLIB_ENV).filter(|value| !value.is_empty()) {
        let path = PathBuf::from(&path);
        if !path.is_file() {
            return Err(format!("{DYLIB_ENV}={} does not exist", path.display()));
        }
    }
    match init_dylib() {
        Ok(()) => Ok(()),
        Err(error) => Err(error.to_string()),
    }
}

/// The `.vocab.json` file as written by the models-repo converters
/// (`{"vocab_size": N, "word_to_idx": {word: row}}`). A flat
/// `word → row` map (the historical format the gem's
/// `OnnxModel.from_file` indexed directly) is accepted too.
#[derive(serde::Deserialize)]
struct VocabFile {
    #[serde(default)]
    vocab_size: Option<usize>,
    #[serde(default)]
    word_to_idx: Option<HashMap<String, u32>>,
}

impl OrtProvider {
    /// Load a tier artifact: `onnx_path` (the graph) and `vocab_path`
    /// (its `.vocab.json` sibling, as distributed with the release).
    pub fn load(
        onnx_path: impl AsRef<Path>,
        vocab_path: impl AsRef<Path>,
    ) -> Result<Self, OrtError> {
        let onnx_path = onnx_path.as_ref();
        let vocab_path = vocab_path.as_ref();

        init_dylib()?;
        let session = ort::session::Session::builder()
            .and_then(|mut builder| builder.commit_from_file(onnx_path))
            .map_err(|error| OrtError::OpenModel {
                path: onnx_path.to_owned(),
                source: error.to_string(),
            })?;

        // Metadata is borrowed from the session; lift the values out
        // before the session moves into the provider.
        let (format, dims) = {
            let metadata = session
                .metadata()
                .map_err(|error| OrtError::Metadata(error.to_string()))?;
            let model_type = metadata.custom("model_type");
            if model_type
                .as_deref()
                .is_some_and(|value| value != "fasttext_embedding")
            {
                return Err(OrtError::Metadata(format!(
                    "model_type {model_type:?} is not a fasttext embedding graph"
                )));
            }
            let quantization = metadata.custom("quantization");
            let format = RowFormat::from_metadata(quantization.as_deref()).ok_or_else(|| {
                OrtError::UnsupportedQuantization(quantization.unwrap_or_default())
            })?;
            let dims = metadata
                .custom("embedding_dimension")
                .and_then(|value| value.parse::<usize>().ok())
                .ok_or_else(|| {
                    OrtError::Metadata("embedding_dimension missing or malformed".to_owned())
                })?;
            (format, dims)
        };

        let vocab_bytes = std::fs::read(vocab_path).map_err(|error| OrtError::Vocab {
            path: vocab_path.to_owned(),
            source: error.to_string(),
        })?;
        let vocab_file: VocabFile =
            serde_json::from_slice(&vocab_bytes).map_err(|error| OrtError::Vocab {
                path: vocab_path.to_owned(),
                source: error.to_string(),
            })?;
        let raw = vocab_file.word_to_idx.unwrap_or_else(|| {
            // Flat historical format: reparse the same bytes as the map.
            serde_json::from_slice::<HashMap<String, u32>>(&vocab_bytes).unwrap_or_default()
        });
        let vocab: HashMap<String, i64> = raw
            .into_iter()
            .map(|(word, row)| (word, i64::from(row)))
            .collect();
        if vocab.is_empty() {
            return Err(OrtError::Vocab {
                path: vocab_path.to_owned(),
                source: "empty vocabulary".to_owned(),
            });
        }
        if let Some(declared) = vocab_file.vocab_size
            && declared != vocab.len()
        {
            return Err(OrtError::Vocab {
                path: vocab_path.to_owned(),
                source: format!(
                    "vocab_size {declared} disagrees with {} entries",
                    vocab.len()
                ),
            });
        }

        let vocab_size = vocab.len();
        Ok(Self {
            session: Mutex::new(session),
            vocab,
            dims,
            format,
            vocab_size,
        })
    }

    /// The row format parsed from the graph metadata (B1 groundwork:
    /// int4 artifacts will report [`RowFormat::Int4PerRow`]).
    pub fn format(&self) -> RowFormat {
        self.format
    }

    /// Vocabulary entry count.
    pub fn vocab_len(&self) -> usize {
        self.vocab_size
    }

    /// The row index of `word`, if it is in vocabulary.
    pub fn word_index(&self, word: &str) -> Option<i64> {
        self.vocab.get(word).copied()
    }
}

impl EmbeddingProvider for OrtProvider {
    fn embedding(&self, word: &str) -> Option<Vec<f32>> {
        let index = *self.vocab.get(word)?;
        let input = ort::value::Tensor::<i64>::from_array(([1i64], vec![index].into_boxed_slice()))
            .map_err(|error| OrtError::Inference(error.to_string()))
            .ok()?;
        let mut session = self.session.lock().expect("ort session mutex poisoned");
        let outputs = session
            .run(ort::inputs!["word_index" => input])
            .map_err(|error| OrtError::Inference(error.to_string()))
            .ok()?;
        let (_, values) = outputs["embedding"]
            .try_extract_tensor::<f32>()
            .map_err(|error| OrtError::Inference(error.to_string()))
            .ok()?;
        let vector = values.to_vec();
        debug_assert_eq!(vector.len(), self.dims);
        Some(vector)
    }

    fn dims(&self) -> usize {
        self.dims
    }

    fn embedding_oov(&self, word: &str) -> Option<Vec<f32>> {
        // B2, honest over the current artifacts: in-vocab character
        // n-gram substrings only (see `rerank::oov`).
        oov::substring_ngram_embedding(word, self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dylib_probe_reports_a_clear_reason_without_the_library() {
        // Without KOTOSHU_ORT_DYLIB set this exercises ort's default
        // search; with it set (the CI onnx job, or a developer
        // exporting it), the file existence check runs first. Either
        // way the probe must produce a message, never panic.
        match dylib_available() {
            Ok(()) => eprintln!("dylib probe: onnxruntime available"),
            Err(message) => eprintln!("dylib probe: {message}"),
        }
    }
}
