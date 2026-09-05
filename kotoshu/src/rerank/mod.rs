//! Embedding-based reranking: pure vector math + the
//! [`EmbeddingProvider`] trait (inference is host-injected; the rerank
//! math works on vectors the host supplies).
//!
//! Port of the gem's context scoring (`lib/kotoshu/analyzers/
//! semantic_analyzer.rb` — `rank_by_context` + `context_boost` — over
//! `models/context.rb` and `models/suggestion.rb`). Where the gem's
//! semantic path is entangled with its ONNX runtime (nearest-neighbor
//! sweeps over a Numo matrix, session lifecycle, error classification),
//! only the pure math was kept here and the vectors arrive through the
//! provider trait:
//!
//! - [`cosine`] mirrors `WordEmbedding#similarity` — dot over norms,
//!   `0.0` for mismatched dimensions or zero norms (gem quirks kept).
//! - [`CosineReranker::rerank`] mirrors `rank_by_context`: each
//!   suggestion's confidence gains `boost_weight *` the sum of cosines
//!   against the surrounding context words, capped at `1.0`, then the
//!   list is sorted descending.
//!
//! Two deliberate deviations from the gem, recorded here so the port is
//! honest:
//!
//! 1. The gem's `build_context` puts the error word itself in
//!    `Context#current` and slices ±32 chars into `before`/`after`;
//!    `surrounding_words` then splits `current` — the OOV misspelling —
//!    so `context_boost` sums cosines of a word against *itself's* OOV
//!    form, which is always `nil → 0.0`: the boost path is effectively a
//!    no-op as wired. This port builds the surrounding words from the
//!    *neighbors* (before/after text around the misspelling), which is
//!    what the scoring was always meant to do.
//! 2. Tie order is deterministic (descending score, ties keep the
//!    incoming order — a stable sort). The gem sorts through
//!    `NearestNeighbor#<=>` + `.reverse`, whose tie order is
//!    MRI-version-dependent; no conformance vector freezes it.

pub mod dequant;
pub mod oov;

#[cfg(feature = "onnx")]
pub mod onnx;

#[cfg(feature = "model")]
pub mod int8_model;

use crate::suggest::Suggestion;

/// Host-supplied word vectors (the inference seam; plan 66: one trait,
/// pluggable providers — ort behind the `onnx` feature, wasm-side
/// candidates later).
pub trait EmbeddingProvider {
    /// The word's vector, or `None` when the word is out of vocabulary.
    fn embedding(&self, word: &str) -> Option<Vec<f32>>;

    /// Vector dimensionality of this provider.
    fn dims(&self) -> usize;

    /// Fallback vector for an out-of-vocabulary word (B2, plan 68).
    ///
    /// The default is `None` (no fallback). The honest implementation
    /// over the current tier artifacts is the character-n-gram
    /// substring sum in [`oov::substring_ngram_embedding`]; full
    /// fastText bucket hashing needs re-converted artifacts (a models
    /// repo change) and stays out until they exist.
    fn embedding_oov(&self, word: &str) -> Option<Vec<f32>> {
        let _ = word;
        None
    }
}

/// Any provider is usable through a shared reference (so wrappers can
/// borrow an owned provider, e.g. `SubwordFallback<&OrtProvider>`).
impl<P: EmbeddingProvider + ?Sized> EmbeddingProvider for &P {
    fn embedding(&self, word: &str) -> Option<Vec<f32>> {
        (**self).embedding(word)
    }

    fn dims(&self) -> usize {
        (**self).dims()
    }

    fn embedding_oov(&self, word: &str) -> Option<Vec<f32>> {
        (**self).embedding_oov(word)
    }
}

/// Cosine similarity of two vectors — the gem's `WordEmbedding#similarity`:
/// dot product over the product of magnitudes, `0.0` on length mismatch
/// (the gem returns `0.0` when dimensions differ) or a zero norm.
pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        dot += f64::from(*x) * f64::from(*y);
        norm_a += f64::from(x * x);
        norm_b += f64::from(y * y);
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}

/// Cosine similarity of two words through a provider — the gem's
/// `EmbeddingModel#similarity`: `None` when either word is OOV.
pub fn similarity(provider: &dyn EmbeddingProvider, a: &str, b: &str) -> Option<f64> {
    let vec_a = provider.embedding(a)?;
    let vec_b = provider.embedding(b)?;
    Some(cosine(&vec_a, &vec_b))
}

/// Text around a misspelled word: the gem's `Models::Context` value
/// object (`before` / `current` / `after`), with `current` being the
/// misspelled word.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Context {
    /// Text before the misspelled word.
    pub before: String,
    /// The misspelled word itself.
    pub current: String,
    /// Text after the misspelled word.
    pub after: String,
}

impl Context {
    /// Build a context from its three slices.
    pub fn new(before: &str, current: &str, after: &str) -> Self {
        Self {
            before: before.to_owned(),
            current: current.to_owned(),
            after: after.to_owned(),
        }
    }

    /// Up to `window` words on each side of the misspelled word, taken
    /// from the before/after text and lowercased for vocabulary lookup
    /// (fastText vocabularies are predominantly lowercase; the gem
    /// downcases at tokenize time).
    ///
    /// See the module docs for why this reads the *neighbors* rather
    /// than the gem's `current`.
    pub fn surrounding_words(&self, window: usize) -> Vec<String> {
        let before: Vec<&str> = tokenize(&self.before).collect();
        let after: Vec<&str> = tokenize(&self.after).collect();
        before
            .into_iter()
            .rev()
            .take(window)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .chain(after.into_iter().take(window))
            .map(|word| word.to_lowercase())
            .collect()
    }
}

/// Split text into word tokens: runs of unicode alphanumeric characters
/// joined by `'`, `-` or `’` (the gem's
/// `/[a-z]+(?:['’-][a-z]+)*/i` tokenizer, generalized past ASCII).
fn tokenize(text: &str) -> impl Iterator<Item = &str> {
    let mut rest = text;
    std::iter::from_fn(move || {
        loop {
            let trimmed = rest.trim_start_matches(|c: char| !is_word_char(c, false));
            if trimmed.is_empty() {
                return None;
            }
            let end = trimmed
                .char_indices()
                .find(|(_, c)| !is_word_char(*c, true))
                .map_or(trimmed.len(), |(i, _)| i);
            let (word, tail) = trimmed.split_at(end);
            if word.chars().all(|c| !c.is_alphanumeric()) {
                rest = tail;
                continue; // separator-only run (e.g. "--")
            }
            rest = tail;
            return Some(word.trim_matches(|c: char| !c.is_alphanumeric()));
        }
    })
}

/// Character class for [`tokenize`]: word characters are alphanumeric or
/// (inside a word) the joining apostrophes/hyphens.
fn is_word_char(c: char, inside: bool) -> bool {
    c.is_alphanumeric() || (inside && (c == '\'' || c == '’' || c == '-'))
}

/// Context window in words on each side (gem `Context#surrounding_words`
/// default 3).
pub const CONTEXT_WINDOW: usize = 3;

/// Boost per unit of context cosine (gem `context_boost`: `sim * 0.02`).
pub const CONTEXT_BOOST_WEIGHT: f64 = 0.02;

/// The cosine reranker: reorders suggestions by contextual relevance
/// (gem `SemanticAnalyzer#rank_by_context` / `#context_boost`).
///
/// Each suggestion's confidence is raised by
/// `boost_weight * Σ cosine(suggestion, context_word)` over the
/// surrounding words that have vectors (OOV context words contribute
/// nothing — the gem's `(sim || 0.0)`), capped at `1.0`; suggestions
/// whose word has no vector keep their confidence unchanged. The list is
/// then sorted descending by the adjusted confidence, ties keeping the
/// incoming order.
#[derive(Debug, Clone)]
pub struct CosineReranker {
    window: usize,
    boost_weight: f64,
}

impl Default for CosineReranker {
    fn default() -> Self {
        Self {
            window: CONTEXT_WINDOW,
            boost_weight: CONTEXT_BOOST_WEIGHT,
        }
    }
}

impl CosineReranker {
    /// The gem's default configuration (window 3, boost weight 0.02).
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the context window (words each side).
    pub fn with_window(mut self, window: usize) -> Self {
        self.window = window;
        self
    }

    /// Override the boost weight per unit cosine.
    pub fn with_boost_weight(mut self, boost_weight: f64) -> Self {
        self.boost_weight = boost_weight;
        self
    }

    /// Boost for one candidate word against the surrounding words —
    /// the gem's `context_boost`, verbatim except that it takes vectors
    /// through the provider instead of a model's full similarity sweep.
    pub fn context_boost(
        &self,
        provider: &dyn EmbeddingProvider,
        word: &str,
        surrounding: &[String],
    ) -> f64 {
        let Some(vector) = lookup(provider, word) else {
            return 0.0;
        };
        surrounding
            .iter()
            .filter_map(|context_word| lookup(provider, context_word))
            .map(|context_vector| cosine(&vector, &context_vector) * self.boost_weight)
            .sum()
    }

    /// Rerank `suggestions` for the misspelled word `context.current`
    /// against the surrounding words of `context`.
    pub fn rerank(
        &self,
        provider: &dyn EmbeddingProvider,
        context: &Context,
        suggestions: Vec<Suggestion>,
    ) -> Vec<Suggestion> {
        let surrounding = context.surrounding_words(self.window);
        if surrounding.is_empty() {
            return suggestions;
        }

        let mut scored: Vec<(f64, usize, Suggestion)> = suggestions
            .into_iter()
            .enumerate()
            .map(|(index, mut suggestion)| {
                let boost = self.context_boost(provider, &suggestion.word, &surrounding);
                // Gem: boosted_similarity = [similarity + boost, 1.0].min
                suggestion.confidence = (suggestion.confidence + boost).min(1.0);
                (suggestion.confidence, index, suggestion)
            })
            .collect();
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.cmp(&b.1))
        });
        scored
            .into_iter()
            .map(|(_, _, suggestion)| suggestion)
            .collect()
    }
}

/// Lowercased word tokens of free text, up to `cap` — the same
/// normalization [`Context::surrounding_words`] applies, for callers
/// that hold one context string instead of a before/current/after
/// split (the wasm context score, [`crate::rerank::int8_model`]).
pub fn context_tokens(text: &str, cap: usize) -> Vec<String> {
    tokenize(text)
        .take(cap)
        .map(|word| word.to_lowercase())
        .collect()
}

/// Vocabulary lookup with a lowercase fallback: exact form first (the
/// vocabularies keep cased entries such as "NASA"), then the lowercased
/// form (the gem downcases at tokenize time).
pub(crate) fn lookup(provider: &dyn EmbeddingProvider, word: &str) -> Option<Vec<f32>> {
    provider
        .embedding(word)
        .or_else(|| provider.embedding(&word.to_lowercase()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Real in-memory provider (trait implementation, not a mock):
    /// 3-dim unit-ish vectors from a fixed table.
    struct MapProvider {
        map: HashMap<String, Vec<f32>>,
    }

    impl MapProvider {
        fn new(entries: &[(&str, [f32; 3])]) -> Self {
            Self {
                map: entries
                    .iter()
                    .map(|(word, vector)| ((*word).to_owned(), vector.to_vec()))
                    .collect(),
            }
        }
    }

    impl EmbeddingProvider for MapProvider {
        fn embedding(&self, word: &str) -> Option<Vec<f32>> {
            self.map.get(word).cloned()
        }

        fn dims(&self) -> usize {
            3
        }
    }

    fn suggestion(word: &str, confidence: f64) -> Suggestion {
        Suggestion {
            word: word.to_owned(),
            distance: 1,
            confidence,
            source: crate::suggest::SuggestionSource::EditDistance,
        }
    }

    fn words(suggestions: &[Suggestion]) -> Vec<&str> {
        suggestions.iter().map(|s| s.word.as_str()).collect()
    }

    #[test]
    fn cosine_matches_the_gem_semantics() {
        let a = [1.0f32, 0.0, 0.0];
        let b = [2.0f32, 0.0, 0.0];
        let orthogonal = [0.0f32, 5.0, 0.0];
        let opposite = [-3.0f32, 0.0, 0.0];
        assert!((cosine(&a, &b) - 1.0).abs() < 1e-12);
        assert!((cosine(&a, &orthogonal) - 0.0).abs() < 1e-12);
        assert!((cosine(&a, &opposite) - -1.0).abs() < 1e-12);
        // 45 degrees: dot 1, norms 1 and sqrt(2)
        let diag = [1.0f32, 1.0, 0.0];
        assert!((cosine(&a, &diag) - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12);
        // Mismatched dimensions and zero norms are 0.0 (gem quirks).
        assert_eq!(cosine(&a, &[1.0, 0.0]), 0.0);
        assert_eq!(cosine(&[0.0, 0.0, 0.0], &b), 0.0);
    }

    #[test]
    fn similarity_is_none_when_either_word_is_oov() {
        let provider = MapProvider::new(&[("hello", [1.0, 0.0, 0.0])]);
        assert!(similarity(&provider, "hello", "hello").is_some());
        assert!(similarity(&provider, "hello", "zzz").is_none());
        assert!(similarity(&provider, "zzz", "hello").is_none());
    }

    #[test]
    fn surrounding_words_window_and_downcasing() {
        let context = Context::new("He SAID", "helo", "to the World again");
        assert_eq!(
            context.surrounding_words(3),
            ["he", "said", "to", "the", "world"]
        );
        assert_eq!(context.surrounding_words(1), ["said", "to"]);
        assert_eq!(
            Context::new("", "x", "").surrounding_words(3),
            Vec::<String>::new()
        );
    }

    #[test]
    fn tokenizer_handles_gem_word_shapes() {
        let context = Context::new("can't - don't", "x", "co-operate 42");
        assert_eq!(
            context.surrounding_words(10),
            ["can't", "don't", "co-operate", "42"]
        );
    }

    #[test]
    fn context_tokens_normalizes_and_caps_free_text() {
        assert_eq!(
            context_tokens("He SAID, quite loudly!", 8),
            ["he", "said", "quite", "loudly"]
        );
        assert_eq!(context_tokens("a b c d", 2), ["a", "b"]);
        assert_eq!(context_tokens("!!!", 4), Vec::<String>::new());
    }

    #[test]
    fn context_boost_sums_weighted_cosines() {
        // boost = 0.02 * (cos(x,a) + cos(x,b)); here +0.5 and -0.5 cancel.
        let provider = MapProvider::new(&[
            ("x", [1.0, 0.0, 0.0]),
            ("a", [1.0, 0.0, 0.0]),
            ("b", [-1.0, 0.0, 0.0]),
        ]);
        let reranker = CosineReranker::new();
        let surrounding = vec!["a".to_owned(), "b".to_owned()];
        assert!(reranker.context_boost(&provider, "x", &surrounding).abs() < 1e-12);
        // Only the in-vocab context words count; OOV candidate → 0.
        assert!(
            reranker
                .context_boost(&provider, "x", &["zzz".to_owned()])
                .abs()
                < 1e-12
        );
        assert_eq!(
            reranker.context_boost(&provider, "zzz", &["a".to_owned()]),
            0.0
        );
        // Exact arithmetic: parallel vectors, weight 0.02.
        assert!((reranker.context_boost(&provider, "x", &["a".to_owned()]) - 0.02).abs() < 1e-12);
    }

    #[test]
    fn rerank_reorders_ties_by_context() {
        // Two candidates with equal confidence; context cosine decides.
        let provider = MapProvider::new(&[
            ("cat", [1.0, 0.0, 0.0]),
            ("dog", [0.0, 1.0, 0.0]),
            ("meow", [1.0, 0.0, 0.0]),
            ("bark", [0.0, 1.0, 0.0]),
        ]);
        let suggestions = vec![suggestion("cat", 0.5), suggestion("dog", 0.5)];

        // Favoring the runner-up flips the order.
        let favor_dog = Context::new("the", "x", "bark");
        let reranked = CosineReranker::new().rerank(&provider, &favor_dog, suggestions.clone());
        assert_eq!(words(&reranked), ["dog", "cat"]);

        // Favoring the leader keeps it.
        let favor_cat = Context::new("the", "x", "meow");
        let reranked = CosineReranker::new().rerank(&provider, &favor_cat, suggestions.clone());
        assert_eq!(words(&reranked), ["cat", "dog"]);

        // Adjusted confidence = 0.5 + 0.02 * 1.0.
        assert!((reranked[0].confidence - 0.52).abs() < 1e-12);
        assert_eq!(
            reranked[0].source,
            crate::suggest::SuggestionSource::EditDistance
        );
    }

    #[test]
    fn rerank_caps_confidence_at_one() {
        let provider = MapProvider::new(&[("hello", [1.0, 0.0, 0.0]), ("world", [1.0, 0.0, 0.0])]);
        let reranked = CosineReranker::new().rerank(
            &provider,
            &Context::new("", "x", "world"),
            vec![suggestion("hello", 1.0)],
        );
        assert_eq!(reranked[0].confidence, 1.0);
    }

    #[test]
    fn rerank_without_context_words_is_a_stable_noop() {
        let provider = MapProvider::new(&[("hello", [1.0, 0.0, 0.0])]);
        let suggestions = vec![suggestion("hello", 0.9), suggestion("zzz", 0.8)];
        let reranked = CosineReranker::new().rerank(
            &provider,
            &Context::new("", "x", ""),
            suggestions.clone(),
        );
        assert_eq!(words(&reranked), ["hello", "zzz"]);
        assert_eq!(reranked[0].confidence, 0.9);

        // Context words that are all OOV leave confidences untouched.
        let reranked =
            CosineReranker::new().rerank(&provider, &Context::new("qqq", "x", "www"), suggestions);
        assert_eq!(reranked[0].confidence, 0.9);
        assert_eq!(reranked[1].confidence, 0.8);
    }

    #[test]
    fn rerank_looks_up_lowercased_context() {
        // Cased context words resolve via the lowercase fallback.
        let provider = MapProvider::new(&[("hello", [1.0, 0.0, 0.0]), ("world", [1.0, 0.0, 0.0])]);
        let reranked = CosineReranker::new().rerank(
            &provider,
            &Context::new("", "x", "World"),
            vec![suggestion("hello", 0.5)],
        );
        assert!((reranked[0].confidence - 0.52).abs() < 1e-12);
    }

    #[test]
    fn rerank_breaks_full_ties_by_incoming_order() {
        let provider = MapProvider::new(&[("a", [1.0, 0.0, 0.0])]);
        let suggestions = vec![
            suggestion("a", 0.5),
            suggestion("zzz", 0.5),
            suggestion("a", 0.5),
        ];
        let reranked =
            CosineReranker::new().rerank(&provider, &Context::new("", "x", "a"), suggestions);
        // "a" gains the boost; the two equal-"a" entries keep their
        // relative order, and OOV "zzz" (unchanged 0.5) sorts below.
        assert_eq!(words(&reranked), ["a", "a", "zzz"]);
    }
}
