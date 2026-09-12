//! The hybrid compose (plan 131): C-retrieve top-20 by the bi-encoder,
//! rescore the slate by fastText full-tier cosine to the typo, exactly
//! the measured PR-34 semantics. The slate size is frozen at 20 — the
//! number the verdict priced — and the typo must sit in the fastText
//! vocabulary, the bench's in-vocab rule; otherwise there is nothing
//! the hybrid can say and [`None`] is the honest answer.

use std::sync::OnceLock;

use super::index::TypoIndex;
use super::model::TypoModel;
use crate::rerank::int8_model::Int8Model;

/// The frozen hybrid slate size (plan 115's pricing, plan 128's verdict).
pub const SLATE: usize = 20;

/// A loaded bi-encoder plus the per-language index derived lazily from
/// the fastText vocabulary it will be paired with. The index costs one
/// forward pass per vocabulary word (the matrix is derived at load by
/// design — the registry ships only the encoder), so it is built once
/// per engine and reused for every suggestion.
pub struct TypoEngine {
    model: TypoModel,
    index: OnceLock<TypoIndex>,
}

impl TypoEngine {
    pub fn new(model: TypoModel) -> Self {
        Self {
            model,
            index: OnceLock::new(),
        }
    }

    /// The underlying encoder (parity probes, pack building).
    pub fn model(&self) -> &TypoModel {
        &self.model
    }

    /// The derived index, building it over the fastText vocabulary on
    /// first use.
    pub fn derived_index(&self, ft: &Int8Model) -> &TypoIndex {
        self.index
            .get_or_init(|| TypoIndex::build(&self.model, ft.vocab()))
    }

    /// The hybrid suggestion slate for `word`, rescored and sorted:
    /// `(candidate, fastText cosine to the typo)`. [`None`] when the
    /// word is empty or outside the fastText vocabulary (the hybrid's
    /// in-vocab rule) or the encoder rejects it.
    pub fn suggest(&self, ft: &Int8Model, word: &str, k: usize) -> Option<Vec<(String, f64)>> {
        let typo_ft = ft.embedding(word)?;
        let typo_row = usize::try_from(ft.word_index(word)?).ok()?;
        let query = self.model.embed(word).ok()?;
        let index = self.derived_index(ft);
        let slate = index.top_k(&query, SLATE.max(k), Some(typo_row));

        let mut rescored: Vec<(String, f64)> = slate
            .into_iter()
            .map(|hit| {
                let candidate = &index.vocab()[hit.index];
                let score = ft
                    .vocab_embedding(hit.index)
                    .map(|row| crate::rerank::cosine(&row, &typo_ft))
                    .unwrap_or_default();
                (candidate.clone(), score)
            })
            .collect();
        // ties keep slate order — the deterministic-ranking house rule
        rescored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap()
                .then(std::cmp::Ordering::Equal)
        });
        rescored.truncate(k);
        Some(rescored)
    }
}
