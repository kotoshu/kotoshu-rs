//! Ranking: the gem's `Suggestions::Suggestion` ordering and
//! `SuggestionSet` sort/dedup/limit pipeline.
//!
//! `Suggestion#<=>` is a total order over distinct lowercased words
//! (combined score desc, distance asc, length-difference asc, n-gram score
//! desc, lowercased word asc). Elements that tie everywhere can still
//! differ in `source`, and `SuggestionSet#sort!` — MRI `Array#sort!` —
//! resolves those ties with the platform libc quicksort's deterministic
//! but unstable order; [`super::macos_qsort`] reproduces the exact
//! permutation the vectors were exported with, and the subsequent
//! `uniq!` keeps whichever duplicate landed first.

use std::cmp::Ordering;

use super::SuggestionSource;

/// A ranked suggestion carrying the metadata the gem's comparator needs
/// (`original_length`, `ngram_score`).
#[derive(Debug, Clone)]
pub(crate) struct Candidate {
    pub word: String,
    pub distance: u8,
    pub confidence: f64,
    pub source: SuggestionSource,
    pub original_length: usize,
    pub ngram_score: f64,
}

impl Candidate {
    /// `Suggestion#combined_score` — distance score (0.3) plus confidence
    /// (0.7). The arithmetic sequence matches Ruby Float evaluation.
    fn combined_score(&self) -> f64 {
        let normalized_distance = self.distance.min(5) as f64 / 5.0;
        let distance_score = 1.0 - normalized_distance;
        (distance_score * 0.3) + (self.confidence * 0.7)
    }

    /// `Suggestion#<=>` (fully total except for same-lowercase-word ties).
    fn compare(&self, other: &Candidate) -> Ordering {
        // Combined score, higher first.
        let score_cmp = other
            .combined_score()
            .partial_cmp(&self.combined_score())
            .unwrap_or(Ordering::Equal);
        if score_cmp != Ordering::Equal {
            return score_cmp;
        }
        // Distance, lower first.
        let distance_cmp = self.distance.cmp(&other.distance);
        if distance_cmp != Ordering::Equal {
            return distance_cmp;
        }
        // Length similarity: |word.length - original_length|, lower first.
        let my_len_diff = self.word.chars().count().abs_diff(self.original_length);
        let other_len_diff = other.word.chars().count().abs_diff(other.original_length);
        let len_cmp = my_len_diff.cmp(&other_len_diff);
        if len_cmp != Ordering::Equal {
            return len_cmp;
        }
        // N-gram score, higher first.
        let ngram_cmp = other
            .ngram_score
            .partial_cmp(&self.ngram_score)
            .unwrap_or(Ordering::Equal);
        if ngram_cmp != Ordering::Equal {
            return ngram_cmp;
        }
        // Lowercased word, ascending (Ruby String#<=> is byte-wise).
        self.word.to_lowercase().cmp(&other.word.to_lowercase())
    }
}

/// `SuggestionSet.new(suggestions, max_size:)` — sort (MRI `sort!`),
/// dedup by lowercased word (`uniq!`, first occurrence survives), limit.
pub(crate) fn suggestion_set(mut suggestions: Vec<Candidate>, max_size: usize) -> Vec<Candidate> {
    super::macos_qsort::sort_by(&mut suggestions, |a, b| a.compare(b));
    dedup_and_limit(suggestions, max_size)
}

/// `sort_and_limit!`'s dedup + limit tail, for callers that already hold a
/// sorted list.
pub(crate) fn dedup_and_limit(suggestions: Vec<Candidate>, max_size: usize) -> Vec<Candidate> {
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    suggestions
        .into_iter()
        .filter(|candidate| seen.insert(candidate.word.to_lowercase()))
        .take(max_size)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(word: &str, distance: u8, confidence: f64, original_length: usize) -> Candidate {
        Candidate {
            word: word.to_owned(),
            distance,
            confidence,
            source: SuggestionSource::EditDistance,
            original_length,
            ngram_score: 0.0,
        }
    }

    #[test]
    fn combined_score_matches_ruby_arithmetic() {
        // distance 1, confidence 0.5: (1 - 1/5) * 0.3 + 0.5 * 0.7
        let c = candidate("w", 1, 0.5, 1);
        assert_eq!(c.combined_score(), (1.0 - 1.0 / 5.0) * 0.3 + 0.5 * 0.7);
    }

    #[test]
    fn ranks_by_score_then_distance() {
        let better = candidate("aaa", 1, 0.9, 3);
        let worse = candidate("bbb", 1, 0.5, 3);
        assert_eq!(better.compare(&worse), Ordering::Less); // sorts first

        let nearer = candidate("aa", 1, 0.5, 3);
        let farther = candidate("bb", 2, 0.5, 3);
        assert_eq!(nearer.compare(&farther), Ordering::Less);
    }

    #[test]
    fn dedup_keeps_first_after_sort() {
        let a = candidate("hello", 1, 0.9, 5);
        let b = candidate("hello", 1, 0.5, 5); // lower confidence sorts later
        let sorted = suggestion_set(vec![b.clone(), a.clone()], 10);
        assert_eq!(sorted.len(), 1);
        assert_eq!(sorted[0].confidence, 0.9);
    }

    #[test]
    fn limit_truncates_to_max_size() {
        let items: Vec<Candidate> = (0..8)
            .map(|i| candidate(&format!("w{i}"), 1, 0.5, 2))
            .collect();
        assert_eq!(suggestion_set(items, 5).len(), 5);
    }
}
