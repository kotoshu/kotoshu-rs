//! Per-dictionary sweep invariants, built once at the first suggestion
//! sweep and kept for the dictionary's lifetime.
//!
//! Every strategy used to walk the whole word list per sweep and decode
//! each word before its gate — the length window, the Soundex code, the
//! n-gram length bound — so a full dictionary paid for decoding words a
//! sweep was about to skip. Those per-word values never change with the
//! query: they are computed here one time, and the sweeps touch only the
//! slices their gates admit.
//!
//! The n-gram Soundex codes pack four ASCII code points into a `u32`
//! (`u32::MAX` is the empty code); lengths are char counts, and
//! `by_length` groups word indices by length in original order so the
//! edit-distance sweep can iterate only its ±2 window.

use std::collections::HashMap;

use super::phonetic;

/// `soundex_key`'s empty code, packed.
const NO_SOUNDEX: u32 = u32::MAX;

/// Pack a 4-char Soundex key (ASCII letters/digits) into a `u32`.
pub(crate) fn pack_key(key: [char; 4]) -> u32 {
    let mut packed = 0u32;
    for c in key {
        packed = (packed << 8) | (c as u32 & 0xff);
    }
    packed
}

#[derive(Debug)]
pub(crate) struct SweepIndex {
    lengths: Vec<u32>,
    soundex: Vec<u32>,
    by_length: HashMap<u32, Vec<u32>>,
}

impl SweepIndex {
    pub(crate) fn build(words: &[String]) -> Self {
        let mut lengths = Vec::with_capacity(words.len());
        let mut soundex = Vec::with_capacity(words.len());
        let mut by_length: HashMap<u32, Vec<u32>> = HashMap::new();
        for (idx, word) in words.iter().enumerate() {
            lengths.push(word.chars().count() as u32);
            soundex.push(phonetic::soundex_key(word).map_or(NO_SOUNDEX, pack_key));
            by_length
                .entry(word.chars().count() as u32)
                .or_default()
                .push(idx as u32);
        }
        Self {
            lengths,
            soundex,
            by_length,
        }
    }

    /// Char length of word `idx`.
    pub(crate) fn length(&self, idx: usize) -> u32 {
        self.lengths[idx]
    }

    /// Packed Soundex code of word `idx` — equal codes compare equal,
    /// including two empty codes.
    pub(crate) fn soundex(&self, idx: usize) -> u32 {
        self.soundex[idx]
    }

    /// Word indices whose length is within `min..=max`, every word in
    /// exactly one bucket, each bucket in original order.
    pub(crate) fn indices_in_length_window(&self, min: u32, max: u32) -> Vec<u32> {
        let mut hits = Vec::new();
        for length in min..=max {
            if let Some(bucket) = self.by_length.get(&length) {
                hits.extend_from_slice(bucket);
            }
        }
        // Restore the word-list order the strategies' candidate arrays
        // were built in before this index existed — ties in the ranking
        // sort are decided by input order, so this must not drift.
        hits.sort_unstable();
        hits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_cover_every_word_exactly_once() {
        let words: Vec<String> = ["one", "two", "three", "four", "five", "seventeen"]
            .iter()
            .map(|w| w.to_string())
            .collect();
        let index = SweepIndex::build(&words);
        let mut all = index.indices_in_length_window(0, 100);
        all.sort_unstable();
        assert_eq!(all, (0..words.len() as u32).collect::<Vec<u32>>());
    }

    #[test]
    fn window_selects_by_char_length_in_original_order() {
        let words: Vec<String> = ["ab", "cd", "xyz", "abcd", "ef"]
            .iter()
            .map(|w| w.to_string())
            .collect();
        let index = SweepIndex::build(&words);
        assert_eq!(index.indices_in_length_window(2, 2), vec![0, 1, 4]);
        assert_eq!(index.indices_in_length_window(3, 4), vec![2, 3]);
    }

    #[test]
    fn soundex_codes_match_the_key_form() {
        let words: Vec<String> = ["Robert", "Rupert", "Ashcraft", "é"]
            .iter()
            .map(|w| w.to_string())
            .collect();
        let index = SweepIndex::build(&words);
        assert_eq!(index.soundex(0), index.soundex(1));
        assert_ne!(index.soundex(0), index.soundex(2));
        // The non-ASCII-only word has the empty code.
        assert_eq!(index.soundex(3), NO_SOUNDEX);
        assert_eq!(index.length(3), 1);
    }
}
