//! N-gram and similarity metrics, ported from the gem's
//! `NgramStrategy` and `BaseStrategy#calculate_ngram_similarity`.

use std::collections::HashMap;

/// `NgramStrategy#extract_ngrams` — n-gram → count map (raw case, exactly
/// as the gem slices the input word; dictionary words are compared
/// verbatim).
pub fn extract_ngrams(word: &[char], n: usize) -> HashMap<String, usize> {
    let mut ngrams = HashMap::new();
    if word.len() < n {
        return ngrams;
    }
    for i in 0..=(word.len() - n) {
        let ngram: String = word[i..i + n].iter().collect();
        *ngrams.entry(ngram).or_insert(0) += 1;
    }
    ngrams
}

/// `NgramStrategy#ngram_similarity` — multiset Jaccard coefficient
/// between the input word's n-grams and another word's.
pub fn ngram_similarity(
    word_ngrams: &HashMap<String, usize>,
    other_word: &[char],
    n: usize,
) -> f64 {
    let other_ngrams = extract_ngrams(other_word, n);

    let mut intersection = 0usize;
    for (ngram, count) in word_ngrams {
        if let Some(other_count) = other_ngrams.get(ngram) {
            intersection += (*count).min(*other_count);
        }
    }

    let mut union = 0usize;
    // Ruby: `word_ngrams.keys | other_ngrams.keys` — each distinct n-gram
    // once.
    for (ngram, count) in word_ngrams {
        union += (*count).max(other_ngrams.get(ngram).copied().unwrap_or(0));
    }
    for (ngram, count) in &other_ngrams {
        if !word_ngrams.contains_key(ngram) {
            union += *count;
        }
    }

    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

/// `BaseStrategy#calculate_ngram_similarity` — the gem's typo-correction
/// similarity (character overlap + prefix weight + suffix weight − length
/// penalty), clamped to `[0.0, 1.0]`. Despite the name it is NOT the
/// n-gram Jaccard above; the gem's ranking mixes both.
pub fn typo_similarity(word1: &str, word2: &str) -> f64 {
    if word1.is_empty() || word2.is_empty() {
        return 0.0;
    }

    let w1: Vec<char> = word1.to_lowercase().chars().collect();
    let w2: Vec<char> = word2.to_lowercase().chars().collect();
    if w1 == w2 {
        return 1.0;
    }

    let len1 = w1.len();
    let len2 = w2.len();
    let max_len = len1.max(len2);

    // Common prefix length (up to 4 characters).
    let mut prefix_len = 0usize;
    for i in 0..len1.min(len2).min(4) {
        if w1[i] != w2[i] {
            break;
        }
        prefix_len += 1;
    }

    // Common suffix length (up to 4 characters).
    let mut suffix_len = 0usize;
    for i in 1..=len1.min(len2).min(4) {
        if w1[len1 - i] != w2[len2 - i] {
            break;
        }
        suffix_len += 1;
    }

    // Character overlap: each w1 character present in w2 counts.
    let overlap = w1.iter().filter(|c| w2.contains(c)).count();

    let similarity = overlap as f64 / max_len as f64;
    let prefix_bonus = prefix_len as f64 * 0.15;
    let suffix_bonus = suffix_len as f64 * 0.05;
    let length_penalty = len1.abs_diff(len2) as f64 * 0.1;

    // Ruby: similarity + prefix_bonus + suffix_bonus - length_penalty
    // (left-associated), then clamp [0.0, 1.0] — `[[s, 1.0].min, 0.0].max`,
    // which equals clamp for the finite values produced here.
    let combined = similarity + prefix_bonus + suffix_bonus - length_penalty;
    combined.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chars(s: &str) -> Vec<char> {
        s.chars().collect()
    }

    #[test]
    fn typo_similarity_identity_and_case() {
        assert_eq!(typo_similarity("hello", "hello"), 1.0);
        assert_eq!(typo_similarity("Hello", "hello"), 1.0);
        assert_eq!(typo_similarity("", "hello"), 0.0);
    }

    #[test]
    fn typo_similarity_shape() {
        // "helo" vs "hello": overlap 4/5, prefix 3, suffix 1, length diff 1
        // — 0.8 + 0.45 + 0.05 - 0.1 = 1.2, clamped to 1.0.
        assert_eq!(typo_similarity("helo", "hello"), 1.0);
        // "xelo" vs "hello": overlap 3/5, prefix 0, suffix 2, length
        // diff 1 — 0.6 + 0.1 - 0.1 = 0.6 (values frozen from the gem).
        assert_eq!(typo_similarity("xelo", "hello"), 0.6);
    }

    #[test]
    fn ngram_jaccard_counts_multiplicities() {
        let grams = extract_ngrams(&chars("aaa"), 2); // {"aa": 2}
        assert_eq!(grams.get("aa"), Some(&2));
        // "aaa" vs "aaaa": intersection 2, union 3.
        let s = ngram_similarity(&grams, &chars("aaaa"), 2);
        assert!((s - 2.0 / 3.0).abs() < 1e-12);
    }
}
