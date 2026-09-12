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
        // One forward pass per word is independent, so the build
        // parallelizes across the vocabulary (std::thread::scope, no
        // new dependency); chunks stay in vocabulary order, keeping
        // rows index-parallel deterministically.
        let words: Vec<String> = vocab.into_iter().map(|w| w.as_ref().to_owned()).collect();
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .clamp(1, 16);
        type Chunk = (Vec<i8>, Vec<f32>, Vec<String>);
        let chunk_len = words.len().div_ceil(threads);
        let mut chunks: Vec<Option<Chunk>> = (0..threads).map(|_| None).collect();
        std::thread::scope(|scope| {
            let handles: Vec<_> = chunks
                .iter_mut()
                .zip(words.chunks(chunk_len))
                .map(|(slot, chunk)| {
                    scope.spawn(move || {
                        let mut rows = Vec::with_capacity(chunk.len() * OUT_DIM);
                        let mut scales = Vec::with_capacity(chunk.len());
                        let mut kept = Vec::with_capacity(chunk.len());
                        let mut scratch = [0f32; OUT_DIM];
                        for word in chunk {
                            let Ok(embedding) = model.embed(word) else {
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
                            kept.push(word.clone());
                        }
                        *slot = Some((rows, scales, kept));
                    })
                })
                .collect();
            for handle in handles {
                handle.join().expect("index build thread");
            }
        });
        let mut rows = Vec::with_capacity(words.len() * OUT_DIM);
        let mut scales = Vec::with_capacity(words.len());
        let mut kept = Vec::with_capacity(words.len());
        for slot in chunks.into_iter().flatten() {
            let (r, s, w) = slot;
            rows.extend(r);
            scales.extend(s);
            kept.extend(w);
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
    /// keep ascending index order (a stable insert), matching the
    /// deterministic-ranking house rule.
    ///
    /// The scan is int8×int8 (plan 115's pricing path): the query is
    /// quantized per-tensor symmetric (`max|v| / 127`), each row dot
    /// accumulates in i32, and the scales multiply back at the end —
    /// measured identical in hit@1/5/20 to the fp32 sweep on every
    /// frozen component, and single-digit milliseconds per query over
    /// a 100k vocabulary.
    pub fn top_k(
        &self,
        query: &[f32; OUT_DIM],
        k: usize,
        exclude: Option<usize>,
    ) -> Vec<TypoSuggestion> {
        let qmax = query.iter().fold(0f32, |m, v| m.max(v.abs()));
        let qscale = if qmax > 0.0 { qmax / 127.0 } else { 1.0 };
        let q8: [i8; OUT_DIM] =
            std::array::from_fn(|i| (query[i] / qscale).round().clamp(-127.0, 127.0) as i8);

        let mut best: Vec<TypoSuggestion> = Vec::with_capacity(k + 1);
        for idx in 0..self.vocab.len() {
            if Some(idx) == exclude {
                continue;
            }
            let row = &self.rows[idx * OUT_DIM..(idx + 1) * OUT_DIM];
            // iterator form: rustc auto-vectorizes this widening dot
            let acc: i32 = q8
                .iter()
                .zip(row)
                .map(|(q, r)| i32::from(*q) * i32::from(*r))
                .sum();
            let dot = acc as f32 * qscale * self.scales[idx];
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
