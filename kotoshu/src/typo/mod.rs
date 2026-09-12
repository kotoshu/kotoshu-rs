//! The hybrid typo-retrieval layer (plan 131) — the productized
//! PR-34/plan-128 verdict: a frozen char-BiGRU bi-encoder retrieves
//! top-20 dictionary candidates for a misspelling, the fastText full
//! tier rescores the slate, and the result merges into the suggestion
//! pipeline ahead of frequency ranking. Opt-in per language, exactly
//! like the semantic tier; never frozen into the conformance vectors.
//!
//! Three concerns, one module each:
//!
//! - [`model`] — the bi-encoder artifact: wire-format reader for the
//!   frozen ONNX graph + the eval-identical forward pass.
//! - [`index`] — the derived per-language candidate matrix (the
//!   registry never ships one; it is built by encoding the loaded
//!   fastText vocabulary through the model, int8-quantized per row)
//!   and the top-k retrieval over it.

pub mod index;
#[cfg(feature = "model")]
pub mod model;
#[cfg(feature = "model")]
pub mod suggest;

pub use index::{TypoIndex, TypoSuggestion};
pub use model::{CHAR_DIM, GRU_DIM, MAX_WORD_LEN, OUT_DIM, TypoModel, TypoModelError};
pub use suggest::{SLATE, TypoEngine};
