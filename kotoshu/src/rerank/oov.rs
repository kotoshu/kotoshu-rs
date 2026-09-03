//! OOV embedding fallback — B2 (plan 68): unseen words still embed.
//!
//! fastText embeds OOV words through hashed character n-grams: each
//! n-gram of `<word>` (minn..maxn = 3..=6 by default) is mapped by a
//! deterministic FNV-1a hash into a bucket table (`2M` rows by default)
//! that lives *alongside* the word matrix, and the OOV vector is the
//! (normalized) sum over word + bucket rows ([arxiv 1709.03933]).
//!
//! **Our tier artifacts do not carry the bucket table** — the models
//! repo converts plain `.vec` files, which have word vectors only. The
//! honest fallback over the current artifacts is therefore narrower: a
//! word's character n-grams contribute only when the n-gram *is itself
//! an in-vocabulary word* (e.g. the "hello" inside "qqhelloqq"). Full
//! bucket hashing requires re-converted artifacts with a subword matrix
//! — a models-repo converter change (named there as B2's sibling) — and
//! remains trait-pluggable: implement
//! [`EmbeddingProvider::embedding_oov`] (or compose
//! [`SubwordFallback`]) however the vectors become available.

use std::collections::HashSet;

use super::EmbeddingProvider;

/// Minimum n-gram length (fastText `minn` default).
pub const NGRAM_MIN: usize = 3;

/// Maximum n-gram length (fastText `maxn` default).
pub const NGRAM_MAX: usize = 6;

/// FNV-1a offset basis — the constant in fastText's `Dictionary::hash`.
pub const FASTTEXT_HASH_BASIS: u32 = 2166136261;

/// FNV-1a prime — the multiplier in fastText's `Dictionary::hash`.
pub const FASTTEXT_HASH_PRIME: u32 = 16777619;

/// The bucket count fastText uses for n-gram rows (`bucket`, default
/// 2,000,000). Recorded for the future bucket-table artifacts; the
/// current fallback does not use it.
pub const FASTTEXT_BUCKET_COUNT: u32 = 2_000_000;

/// fastText's n-gram hash — FNV-1a 32 over the bytes, with each byte
/// first cast to *signed* `i8` and then zero-extended to `u32`
/// (`uint32_t(int8_t(c))` in `fasttext/src/dictionary.cc`; bytes ≥ 0x80
/// sign-extend). Reproduced here so the future bucket artifacts select
/// exactly the rows the reference implementation would.
pub fn fasttext_hash(s: &str) -> u32 {
    let mut hash = FASTTEXT_HASH_BASIS;
    for byte in s.as_bytes() {
        hash ^= i32::from(*byte as i8) as u32;
        hash = hash.wrapping_mul(FASTTEXT_HASH_PRIME);
    }
    hash
}

/// The bucket row an n-gram would occupy in a fastText artifact
/// (`hash % bucket`). Unused by the current substring fallback.
pub fn fasttext_bucket(s: &str) -> u32 {
    fasttext_hash(s) % FASTTEXT_BUCKET_COUNT
}

/// The distinct character n-grams (length `NGRAM_MIN..=NGRAM_MAX`) of
/// the lowercased word, as contiguous substrings. Boundary markers
/// (`<`/`>`) are deliberately omitted: this vocabulary contains no
/// marked tokens, so marked n-grams could never resolve.
pub fn substring_ngrams(word: &str) -> Vec<String> {
    let chars: Vec<char> = word.to_lowercase().chars().collect();
    let mut ngrams = HashSet::new();
    for length in NGRAM_MIN..=NGRAM_MAX.min(chars.len()) {
        for start in 0..=chars.len() - length {
            ngrams.insert(chars[start..start + length].iter().collect::<String>());
        }
    }
    ngrams.into_iter().collect()
}

/// The honest OOV embedding over word-only artifacts: the L2-normalized
/// sum of the vectors of every character n-gram of `word` that is
/// itself an in-vocabulary word. `None` when no n-gram resolves (or the
/// sum degenerates to zero norm).
pub fn substring_ngram_embedding(word: &str, provider: &dyn EmbeddingProvider) -> Option<Vec<f32>> {
    let dims = provider.dims();
    let mut sum = vec![0.0f32; dims];
    let mut resolved = 0usize;
    for ngram in substring_ngrams(word) {
        if let Some(vector) = provider.embedding(&ngram)
            && vector.len() == dims
        {
            for (accumulator, value) in sum.iter_mut().zip(vector) {
                *accumulator += value;
            }
            resolved += 1;
        }
    }
    if resolved == 0 {
        return None;
    }
    let norm = sum.iter().map(|v| f64::from(v * v)).sum::<f64>().sqrt();
    if norm == 0.0 {
        return None;
    }
    let norm = norm as f32;
    Some(sum.iter().map(|v| *v / norm).collect())
}

/// A provider wrapper wiring the substring fallback into
/// [`EmbeddingProvider::embedding`] (B2): in-vocabulary lookups go to
/// the inner provider, misses fall back to
/// [`substring_ngram_embedding`].
#[derive(Debug, Clone)]
pub struct SubwordFallback<P> {
    inner: P,
}

impl<P> SubwordFallback<P> {
    /// Wrap `inner` with the substring-n-gram OOV fallback.
    pub fn new(inner: P) -> Self {
        Self { inner }
    }

    /// The wrapped provider.
    pub fn inner(&self) -> &P {
        &self.inner
    }
}

impl<P: EmbeddingProvider> EmbeddingProvider for SubwordFallback<P> {
    fn embedding(&self, word: &str) -> Option<Vec<f32>> {
        self.inner
            .embedding(word)
            .or_else(|| self.embedding_oov(word))
    }

    fn dims(&self) -> usize {
        self.inner.dims()
    }

    fn embedding_oov(&self, word: &str) -> Option<Vec<f32>> {
        substring_ngram_embedding(word, &self.inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    struct MapProvider {
        dims: usize,
        map: HashMap<String, Vec<f32>>,
    }

    impl MapProvider {
        fn new(dims: usize, entries: &[(&str, Vec<f32>)]) -> Self {
            Self {
                dims,
                map: entries
                    .iter()
                    .map(|(word, vector)| ((*word).to_owned(), vector.clone()))
                    .collect(),
            }
        }
    }

    impl EmbeddingProvider for MapProvider {
        fn embedding(&self, word: &str) -> Option<Vec<f32>> {
            self.map.get(word).cloned()
        }

        fn dims(&self) -> usize {
            self.dims
        }
    }

    #[test]
    fn fasttext_hash_known_answers() {
        // FNV-1a 32 reference vectors (ASCII — no sign extension).
        assert_eq!(fasttext_hash(""), FASTTEXT_HASH_BASIS);
        assert_eq!(fasttext_hash("hello"), 0x4f9f2cab);
        assert_eq!(fasttext_hash("h"), 0xed0c3757);
        // "<hello" — the fastText subword form of "hello".
        assert_eq!(fasttext_hash("<hello"), 0x51e1ed89);
        // Non-ASCII exercises the int8 sign extension:
        // 'é' = [0xc3, 0xa9], 'ü' = [0xc3, 0xbc], 'ß' = [0xc3, 0x9f],
        // ' ' = [0x20], '日' = [0xe6, 0x97, 0xa5], '本' = [0xe6, 0x9c, 0xac].
        assert_eq!(fasttext_hash("é"), 0x3cfa68c1);
        assert_eq!(fasttext_hash("üß"), 0x2bdb26dc);
        assert_eq!(fasttext_hash(" 日本"), 0x057d70ff);
    }

    #[test]
    fn bucket_is_hash_mod_bucket_count() {
        assert_eq!(fasttext_bucket("hello"), 0x4f9f2cab % FASTTEXT_BUCKET_COUNT);
        assert_eq!(fasttext_bucket("hello"), 1335831723 % 2_000_000);
    }

    #[test]
    fn substring_ngrams_are_lowercased_distinct_substrings() {
        let ngrams = substring_ngrams("Hello");
        // 3..=5 of "hello": hel ell llo hell ello hello
        assert_eq!(
            ngrams.iter().map(String::as_str).collect::<HashSet<_>>(),
            ["hel", "ell", "llo", "hell", "ello", "hello"]
                .into_iter()
                .collect::<HashSet<_>>()
        );
        // Short words produce no n-grams.
        assert!(substring_ngrams("ab").is_empty());
        // A 3-char word contributes exactly itself.
        assert_eq!(
            substring_ngrams("abc")
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            ["abc"]
        );
    }

    #[test]
    fn substring_embedding_sums_and_normalizes_in_vocab_ngrams() {
        let provider = MapProvider::new(
            2,
            &[
                ("hel", vec![3.0, 4.0]), // norm 5
                ("ell", vec![-1.0, 0.0]),
                ("zzz", vec![9.0, 9.0]), // never a substring below
            ],
        );
        // "unhel": n-grams unh/nhe/hel/unhe/nhel/unhel — only "hel" is
        // in vocab → the fallback is the unit vector of (3, 4).
        let embedding = substring_ngram_embedding("unhel", &provider).unwrap();
        assert_eq!(embedding.len(), 2);
        let norm = (embedding[0] as f64).hypot(embedding[1] as f64);
        assert!((norm - 1.0).abs() < 1e-6);
        assert!((embedding[0] - 0.6).abs() < 1e-6);
        assert!((embedding[1] - 0.8).abs() < 1e-6);

        // Sum over multiple n-grams: "hellish" over {"hel", "ell"}.
        let provider = MapProvider::new(2, &[("hel", vec![1.0, 0.0]), ("ell", vec![0.0, 1.0])]);
        let embedding = substring_ngram_embedding("hellish", &provider).unwrap();
        assert!((embedding[0] - std::f64::consts::FRAC_1_SQRT_2 as f32).abs() < 1e-6);
        assert!((embedding[1] - std::f64::consts::FRAC_1_SQRT_2 as f32).abs() < 1e-6);
    }

    #[test]
    fn substring_embedding_is_none_without_resolvable_ngrams() {
        let provider = MapProvider::new(2, &[("zzz", vec![1.0, 1.0])]);
        assert!(substring_ngram_embedding("unhel", &provider).is_none());
        assert!(substring_ngram_embedding("", &provider).is_none());
        // Canceling vectors degenerate to zero norm → None (not NaN).
        let provider = MapProvider::new(2, &[("hel", vec![1.0, 0.0]), ("ell", vec![-1.0, 0.0])]);
        assert!(substring_ngram_embedding("hellish", &provider).is_none());
    }

    #[test]
    fn subword_fallback_wraps_the_inner_provider() {
        let inner = MapProvider::new(2, &[("hello", vec![3.0, 4.0]), ("hel", vec![3.0, 4.0])]);
        let provider = SubwordFallback::new(inner);

        // In-vocabulary words go straight to the inner provider.
        let direct = provider.embedding("hello").unwrap();
        assert_eq!(direct, vec![3.0, 4.0]);
        assert_eq!(provider.dims(), 2);

        // OOV words fall back to the normalized substring sum; the raw
        // inner lookup stays a miss.
        assert!(provider.inner().embedding("qqhelloqq").is_none());
        let fallback = provider.embedding("qqhelloqq").unwrap();
        assert!((fallback[0] - 0.6).abs() < 1e-6);
        assert!((fallback[1] - 0.8).abs() < 1e-6);
        // embedding_oov is exposed directly, too.
        assert!(provider.embedding_oov("qqhelloqq").is_some());
        // And a word with no in-vocab n-gram stays None end to end.
        assert!(provider.embedding("zzqqzz").is_none());
    }
}
