//! SymSpell channel (TODO.sota/2 port): the gem's SymSpellStrategy —
//! the composite's primary when the language has a frequency full_list.
//!
//! A line-faithful port of `generate` + `precompute!`: the deletion
//! index carries single deletions of each word AND of its folded form
//! AND its adjacent transpositions; the lookup unions the input's
//! verbatim bucket, the folded form's single-deletion buckets, and two
//! rounds of input-side deletion expansion (with `@words` membership
//! promoting the variant itself); candidates are a set (duplicates in
//! a bucket collapse); scoring is fold-normalized Damerau (NFD
//! combining-mark strip + sharp-s expansion, fold-equal substitutions
//! cost zero — plan C9) with cutoff `max_dist + 1`; the sort key is
//! (distance, double-letter pattern, frequency rank) — en's confusion
//! set is the gem's NULL_SET so that slot is a constant there. The
//! slate is emitted verbatim (`ranked: true` — no re-sort, no dedup,
//! cap `min(caller limit, 10)`).

#[path = "symspell_data.rs"]
mod symspell_data;

use self::symspell_data::{FOLD_TABLE, FULL_LIST};
use super::SuggestionSource;
use super::rank::Candidate;
use std::collections::{HashMap, HashSet};

const MAX_DISTANCE: usize = 2;
/// `BaseStrategy` config default `max_results` — the slate's own cap,
/// independent of the caller-facing limit.
const STRATEGY_MAX_RESULTS: usize = 10;

/// Fold per the gem's `fold_word`: lowercased, sharp-s expands to
/// "ss", everything else through the generated NFD-strip table
/// (FOLD_TABLE — produced by MRI's `unicode_normalize(:nfd)`).
///
/// Returns FOLD UNITS, not chars: the gem builds the array with
/// `flat_map`, and `flat_map` does not split Strings — the sharp-s
/// expansion stays ONE element ("ss" among single-char elements).
/// Deletion keys and distances are element-wise over these units, so
/// a doubled unit must never be split.
pub fn fold_word(word: &str) -> Vec<String> {
    let lower = word.to_lowercase();
    let mut out: Vec<String> = Vec::with_capacity(lower.len());
    for ch in lower.chars() {
        if ch == 'ß' {
            out.push("ss".to_string());
            continue;
        }
        match fold_table_lookup(ch) {
            Some(folded) => out.extend(folded.chars().map(String::from)),
            None => out.push(ch.to_string()),
        }
    }
    out
}

fn units_join(units: &[String]) -> String {
    units.concat()
}

/// Element-wise single deletions of a fold-unit array (the gem's
/// `generate_single_deletions_from_array` over the fold array).
fn single_deletions_units(units: &[String]) -> Vec<String> {
    let mut out = Vec::with_capacity(units.len());
    for i in 0..units.len() {
        let v = units_join(&[&units[..i], &units[i + 1..]].concat());
        if !v.is_empty() {
            out.push(v);
        }
    }
    out
}

fn fold_table_lookup(ch: char) -> Option<&'static str> {
    FOLD_TABLE
        .binary_search_by_key(&(ch as u32), |&(cp, _)| cp)
        .ok()
        .map(|i| FOLD_TABLE[i].1)
}

/// Bounded Damerau-Levenshtein (OSA) over two fold-unit slices — the
/// gem's `folded_distance` (and `bounded_edit_distance`, same body):
/// element-wise equality, nil once a row minimum proves the distance
/// exceeds `max`.
fn damerau(a: &[String], b: &[String], max: usize) -> Option<usize> {
    if a == b {
        return Some(0);
    }
    let (la, lb) = (a.len(), b.len());
    if la.abs_diff(lb) > max {
        return None;
    }
    let mut prev2: Option<Vec<usize>> = None;
    let mut prev: Vec<usize> = (0..=lb).collect();
    for i in 1..=la {
        let mut cur = vec![i];
        cur.resize(lb + 1, 0);
        let mut row_min = i;
        for j in 1..=lb {
            let sub = prev[j - 1] + usize::from(a[i - 1] != b[j - 1]);
            let mut v = (prev[j] + 1).min(cur[j - 1] + 1).min(sub);
            if i > 1
                && j > 1
                && a[i - 1] == b[j - 2]
                && a[i - 2] == b[j - 1]
                && let Some(p2) = &prev2
            {
                v = v.min(p2[j - 2] + 1);
            }
            cur[j] = v;
            row_min = row_min.min(v);
        }
        if row_min > max {
            return None;
        }
        prev2 = Some(prev);
        prev = cur;
    }
    let d = prev[lb];
    (d <= max).then_some(d)
}

/// The gem's `double_letter_pattern?` — `word` is the caller's
/// original (case preserved); char lengths, not byte lengths.
fn double_letter_pattern(word: &str, candidate: &str) -> bool {
    let w: Vec<char> = word.chars().collect();
    let c: Vec<char> = candidate.chars().collect();
    let doubled_removal = |long: &[char], short: &[char]| -> bool {
        if long.len() != short.len() + 1 {
            return false;
        }
        for i in 0..long.len() - 1 {
            if long[i] == long[i + 1] {
                let mut removed: Vec<char> = long[..i].to_vec();
                removed.extend_from_slice(&long[i + 1..]);
                if removed == short {
                    return true;
                }
            }
        }
        false
    };
    if c.len() == w.len() + 1 {
        w.windows(2).any(|p| p[0] == p[1]) || doubled_removal(&c, &w)
    } else if w.len() == c.len() + 1 {
        doubled_removal(&w, &c)
    } else {
        false
    }
}

/// The gem's `calculate_ngram_similarity` (downcased, prefix/suffix
/// bonuses capped at 4, overlap over the longer length, clamped).
fn ngram_similarity(word1: &str, word2: &str) -> f64 {
    let w1: Vec<char> = word1.to_lowercase().chars().collect();
    let w2: Vec<char> = word2.to_lowercase().chars().collect();
    if w1.is_empty() || w2.is_empty() {
        return 0.0;
    }
    if w1 == w2 {
        return 1.0;
    }
    let (len1, len2) = (w1.len(), w2.len());
    let max_len = len1.max(len2);
    let mut prefix_len = 0;
    for i in 0..len1.min(len2).min(4) {
        if w1[i] != w2[i] {
            break;
        }
        prefix_len += 1;
    }
    let mut suffix_len = 0;
    for i in 1..=len1.min(len2).min(4) {
        if w1[len1 - i] != w2[len2 - i] {
            break;
        }
        suffix_len += 1;
    }
    let overlap = w1.iter().filter(|c| w2.contains(c)).count();
    let similarity =
        overlap as f64 / max_len as f64 + prefix_len as f64 * 0.15 + suffix_len as f64 * 0.05
            - len1.abs_diff(len2) as f64 * 0.1;
    similarity.clamp(0.0, 1.0)
}

struct Index {
    words: HashSet<String>,
    ranks: HashMap<String, u32>,
    deletes: HashMap<String, Vec<String>>,
}

static INDEX: std::sync::OnceLock<Index> = std::sync::OnceLock::new();

fn index() -> &'static Index {
    INDEX.get_or_init(|| {
        let mut words = HashSet::new();
        let mut ranks = HashMap::new();
        let mut deletes: HashMap<String, Vec<String>> = HashMap::new();
        for (word, rank) in FULL_LIST {
            let lower = word.to_lowercase();
            // Last occurrence wins — the gem builds `ranks` as a Hash
            // assignment over the raw list, so duplicated entries take
            // the final rank (next@165→4800, bar@989→5825 in en).
            ranks.insert(lower.clone(), *rank);
            if words.insert(lower.clone()) {
                for variant in single_deletions(&lower) {
                    deletes.entry(variant).or_default().push(lower.clone());
                }
                let folded = fold_word(&lower);
                let lower_units: Vec<String> = lower.chars().map(String::from).collect();
                if folded != lower_units {
                    for variant in single_deletions_units(&folded) {
                        deletes.entry(variant).or_default().push(lower.clone());
                    }
                }
                // `generate_transpositions` — adjacent-swap keys ("boar"
                // is discoverable from "obar").
                let chars: Vec<char> = lower.chars().collect();
                for i in 0..chars.len().saturating_sub(1) {
                    let mut swapped = chars.clone();
                    swapped.swap(i, i + 1);
                    let variant: String = swapped.iter().collect();
                    if variant != lower {
                        deletes.entry(variant).or_default().push(lower.clone());
                    }
                }
            }
        }
        Index {
            words,
            ranks,
            deletes,
        }
    })
}

fn single_deletions(word: &str) -> Vec<String> {
    single_deletions_chars(&word.chars().collect::<Vec<_>>())
}

fn single_deletions_chars(chars: &[char]) -> Vec<String> {
    let mut out = Vec::with_capacity(chars.len());
    for i in 0..chars.len() {
        let v: String = chars[..i].iter().chain(chars[i + 1..].iter()).collect();
        if !v.is_empty() {
            out.push(v);
        }
    }
    out
}

/// True when the engine is in ranked mode: the embedded frequency
/// list is present, so SymSpell is the composite's primary (the gem's
/// `frequency_ranked?`). The embedded en list is always present.
pub fn ranked() -> bool {
    !FULL_LIST.is_empty()
}

/// The SymSpell slate for `word` — empty when the word is a known
/// frequency word (the gem's in-dictionary guard) or no candidates
/// match.
pub fn slate(word: &str, caller_limit: usize) -> Vec<Candidate> {
    let index = index();
    let limit = caller_limit.min(STRATEGY_MAX_RESULTS);
    let lower = word.to_lowercase();
    if index.words.contains(&lower) {
        return Vec::new();
    }
    let lower_units: Vec<String> = lower.chars().map(String::from).collect();
    let bucket = |key: &str| -> Vec<String> { index.deletes.get(key).cloned().unwrap_or_default() };

    let mut candidates: HashSet<String> = HashSet::new();
    candidates.extend(bucket(&lower));

    let folded = fold_word(&lower);
    if folded != lower_units {
        for v in single_deletions_units(&folded) {
            candidates.extend(bucket(&v));
        }
    }

    // `generate_deletions_from_set` rounds: each round expands the
    // previously-new variants (the already-checked ones dedupe via
    // `checked`, matching the gem's membership filter).
    let mut checked: HashSet<String> = HashSet::new();
    checked.insert(lower.clone());
    let mut frontier: Vec<String> = vec![lower.clone()];
    for _ in 0..MAX_DISTANCE {
        let mut next: Vec<String> = Vec::new();
        for variant in &frontier {
            for del in single_deletions(variant) {
                if checked.insert(del.clone()) {
                    if index.words.contains(&del) {
                        candidates.insert(del.clone());
                    }
                    candidates.extend(bucket(&del));
                    next.push(del);
                }
            }
        }
        frontier = next;
    }
    candidates.remove(&lower);

    let mut scored: Vec<(String, usize)> = Vec::with_capacity(candidates.len());
    for cand in &candidates {
        let cand_fold = fold_word(cand);
        if let Some(dist) = damerau(&folded, &cand_fold, MAX_DISTANCE + 1) {
            scored.push((cand.clone(), dist));
        }
    }
    scored.sort_by(|a, b| {
        let key = |w: &str, dist: usize| {
            let pattern = u8::from(!double_letter_pattern(word, w));
            let rank = index.ranks.get(w).copied().unwrap_or(1_000_000_000);
            (dist, pattern, rank)
        };
        key(&a.0, a.1).cmp(&key(&b.0, b.1))
    });

    scored
        .into_iter()
        .take(limit)
        .map(|(cand, dist)| Candidate {
            confidence: if dist == 0 {
                1.0
            } else {
                1.0 / (1.0 + dist as f64)
            },
            distance: dist as u8,
            ngram_score: ngram_similarity(word, &cand),
            original_length: word.chars().count(),
            source: SuggestionSource::SymSpell,
            word: cand,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn units(words: &[&str]) -> Vec<String> {
        words.iter().map(|u| u.to_string()).collect()
    }

    #[test]
    fn fold_strips_latin1_marks() {
        assert_eq!(fold_word("á"), units(&["a"]));
        assert_eq!(fold_word("é"), units(&["e"]));
        assert_eq!(fold_word("ó"), units(&["o"]));
        assert_eq!(fold_word("ç"), units(&["c"]));
        assert_eq!(fold_word("Iç"), units(&["i", "c"]));
        assert_eq!(fold_word("foó'"), units(&["f", "o", "o", "'"]));
    }

    #[test]
    fn fold_expands_sharp_s() {
        assert_eq!(fold_word("ß"), units(&["ss"]));
        assert_eq!(fold_word("söße"), units(&["s", "o", "ss", "e"]));
    }

    #[test]
    fn sharp_s_is_one_fold_unit() {
        // The gem's fold_word builds its array with flat_map, which does
        // NOT split the "ss" String — the expansion stays one element, so
        // element-wise deletions of the fold never produce "musig" from
        // "müßig" (the runtime check: slate("MÜßIG") is empty while
        // slate("mussig") finds "musing").
        let folded_units = fold_word("müßig");
        assert_eq!(folded_units, units(&["m", "u", "ss", "i", "g"]));
        assert!(!single_deletions_units(&folded_units).contains(&"musig".to_string()));
    }

    #[test]
    fn fold_table_is_sorted() {
        assert!(FOLD_TABLE.windows(2).all(|w| w[0].0 < w[1].0));
    }

    #[test]
    fn embedded_slate_finds_fold_distance_zero() {
        // "ic" is rank 12021 in the published en list; the folded
        // deletion of "iç" reaches it at distance 0.
        let slate = slate("Iç", 5);
        assert_eq!(slate.first().map(|c| c.word.as_str()), Some("ic"));
        assert_eq!(slate.first().map(|c| c.distance), Some(0));
    }

    #[test]
    fn embedded_slate_sugesst() {
        let slate = slate("sugesst", 5);
        let words: Vec<&str> = slate.iter().map(|c| c.word.as_str()).collect();
        assert_eq!(words, vec!["suggest", "surest", "guest", "guess", "surges"]);
    }
}
