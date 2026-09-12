//! The derived per-language candidate matrix and top-k retrieval.
//!
//! The registry ships only the bi-encoder; the matrix is *derived at
//! load* (the plan-115 pricing: building one row costs one forward
//! pass, ~5 µs amortized on M1 Max for 100k words — the shipped
//! engines pay it once per language, lazily). Rows are int8 with one
//! f32 scale per row (`max|v| / 127`), the same per-row scheme the
//! tiers use ([`crate::rerank::dequant`]) — plan 115 measured ranking
//! equivalence with the fp32 brute-force sweep on every frozen
//! C-benchmark component.

use super::model::{OUT_DIM, TypoModel};

/// The derived matrix over one language's vocabulary.
pub struct TypoIndex {
    /// Row-major int8 embeddings: `V × OUT_DIM`.
    rows: Vec<i8>,
    /// One scale per row.
    scales: Vec<f32>,
    /// The encoded vocabulary, index-parallel to the rows.
    vocab: Vec<String>,
}

impl TypoIndex {
    /// Build the index by encoding every vocabulary word through the
    /// model. Words the model cannot encode (empty after truncation —
    /// impossible for dictionary vocabularies, but the API does not
    /// assume it) are skipped.
    pub fn build<'a, I, S>(model: &TypoModel, vocab: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str> + 'a,
    {
        let mut rows = Vec::new();
        let mut scales = Vec::new();
        let mut kept = Vec::new();
        let mut scratch = [0f32; OUT_DIM];
        for word in vocab {
            let Ok(embedding) = model.embed(word.as_ref()) else {
                continue;
            };
            let mut max_abs = 0f32;
            for (dst, v) in scratch.iter_mut().zip(embedding) {
                *dst = v;
                max_abs = max_abs.max(v.abs());
            }
            let scale = max_abs / 127.0;
            scales.push(scale);
            for v in scratch {
                let q = if scale > 0.0 {
                    (v / scale).round().clamp(-127.0, 127.0) as i8
                } else {
                    0
                };
                rows.push(q);
            }
            kept.push(word.as_ref().to_owned());
        }
        Self {
            rows,
            scales,
            vocab: kept,
        }
    }

    /// Number of indexed words.
    pub fn len(&self) -> usize {
        self.vocab.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.vocab.is_empty()
    }

    /// The indexed vocabulary, index-parallel to the matrix rows.
    pub fn vocab(&self) -> &[String] {
        &self.vocab
    }

    /// Retrieve the top-k candidates by cosine to the query embedding,
    /// excluding `exclude` (the typo's own row — the eval rule). Ties
    /// keep ascending index order (a stable sort), matching the
    /// deterministic-ranking house rule.
    pub fn top_k(
        &self,
        query: &[f32; OUT_DIM],
        k: usize,
        exclude: Option<usize>,
    ) -> Vec<TypoSuggestion> {
        let mut best: Vec<TypoSuggestion> = Vec::with_capacity(k + 1);
        let mut scratch = [0f32; OUT_DIM];
        for idx in 0..self.vocab.len() {
            if Some(idx) == exclude {
                continue;
            }
            let row = &self.rows[idx * OUT_DIM..(idx + 1) * OUT_DIM];
            let scale = self.scales[idx];
            for (dst, q) in scratch.iter_mut().zip(row) {
                *dst = f32::from(*q) * scale;
            }
            let mut dot = 0f32;
            for (a, b) in query.iter().zip(scratch) {
                dot += a * b;
            }
            if best.len() < k || dot > best[k - 1].score {
                let hit = TypoSuggestion {
                    index: idx,
                    score: dot,
                };
                let pos = best
                    .binary_search_by(|probe| probe.score.partial_cmp(&dot).unwrap().reverse())
                    .unwrap_or_else(|p| p);
                best.insert(pos.min(k), hit);
                best.truncate(k);
            }
        }
        best
    }
}

/// One retrieved candidate: vocabulary index and cosine score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TypoSuggestion {
    pub index: usize,
    pub score: f32,
}
