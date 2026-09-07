//! Suggestion pipeline: a port of the gem's `Suggestions::Generator`
//! default algorithms — `EditDistanceStrategy`, `PhoneticStrategy`,
//! `KeyboardProximityStrategy`, `NgramStrategy` — composed through
//! `CompositeStrategy` into one `SuggestionSet` (sort, dedup, limit).
//!
//! The conformance vectors freeze the gem's actual output, quirks
//! included: per-strategy result caps of 10 (the strategies' own config
//! default, independent of the caller-facing limit), the phonetic
//! strategy's hardcoded `distance = 1` from `create_suggestion_set`'s
//! empty distance map, the keyboard strategy's dead "extra double letter"
//! branch, and MRI/macOS-libc sort tie orders (see the private `ruby_sort`
//! and `macos_qsort` submodules). `EditDistanceStrategy` additionally
//! sweeps the distance-1 edits of the misspelling (see `edit_sweep` and
//! the private `permutations` submodule), so dictionary forms — affixed
//! or capitalization variants — surface alongside raw stems. Behavioral
//! reference: `lib/kotoshu/suggestions/` in the gem.

mod edit_distance;
mod frequency;
mod keyboard;
mod macos_qsort;
mod ngram;
mod permutations;
mod phonetic;
mod rank;
mod ruby_sort;
mod sweep_index;

pub(crate) use sweep_index::SweepIndex;

use crate::dict::Dictionary;

use rank::Candidate;

/// Default per-strategy result cap (`BaseStrategy` config default 10 —
/// independent of the caller-facing limit).
const STRATEGY_MAX_RESULTS: usize = 10;

/// `EditDistanceStrategy` config defaults.
const EDIT_MAX_DISTANCE: usize = 2;
const EDIT_MIN_CONFIDENCE: f64 = 0.75;
const EDIT_MIN_SIMILARITY: f64 = 0.70;
const EDIT_MIN_RESULTS: usize = 3;

/// `PhoneticStrategy` / `KeyboardProximityStrategy` distance cap.
const STRATEGY_MAX_DISTANCE: usize = 2;

/// `KeyboardProximityStrategy#min_similarity`.
const KEYBOARD_MIN_SIMILARITY: f64 = 0.70;

/// `NgramStrategy` config defaults.
const NGRAM_N: usize = 3;
const NGRAM_MIN_SIMILARITY: f64 = 0.3;
const NGRAM_MIN_TYPO_SIMILARITY: f64 = 0.70;

/// One ranked suggestion (the public projection of the gem's
/// `Suggestions::Suggestion` minus its metadata).
#[derive(Debug, Clone, PartialEq)]
pub struct Suggestion {
    /// The suggested dictionary word.
    pub word: String,
    /// Edit distance (per-strategy semantics; see the gem).
    pub distance: u8,
    /// Confidence in `[0, 1]`.
    pub confidence: f64,
    /// Which strategy produced the suggestion.
    pub source: SuggestionSource,
}

/// Which strategy produced a suggestion (gem `Suggestions::Strategies::*`,
/// wire `source` strings in the conformance vectors).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuggestionSource {
    /// `EditDistanceStrategy`.
    EditDistance,
    /// `PhoneticStrategy`.
    Phonetic,
    /// `KeyboardProximityStrategy`.
    KeyboardProximity,
    /// `NgramStrategy`.
    Ngram,
}

impl SuggestionSource {
    /// The gem's strategy name (`Suggestion#source`, a Symbol rendered to
    /// its string form by the exporter).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::EditDistance => "edit_distance",
            Self::Phonetic => "phonetic",
            Self::KeyboardProximity => "keyboard_proximity",
            Self::Ngram => "ngram",
        }
    }
}

/// Generate ranked suggestions for `word` (the gem's
/// `Spellchecker#suggest` → `Generator#generate` over the default
/// algorithms).
///
/// Mirrors the gem exactly: empty words yield nothing, and words the
/// dictionary accepts yield nothing (every default strategy's `handles?`
/// is `!dictionary.lookup(word)`).
pub fn suggest(dictionary: &Dictionary, word: &str, limit: usize) -> Vec<Suggestion> {
    if word.is_empty() || dictionary.correct(word) {
        return Vec::new();
    }

    let words = dictionary.words();
    // Per-dictionary invariants shared by every sweep: lengths, Soundex
    // codes, length buckets — built once, kept for the dictionary's life.
    let sweep = dictionary.sweep_index();
    // Built once per sweep: the keyboard strategy looks up one variant at
    // a time and its two-round variant set reaches tens of thousands of
    // lookups, each of which used to scan the whole dictionary.
    let index = WordIndex::build(words);
    let mut pool: Vec<Candidate> = Vec::new();
    pool.extend(edit_distance_strategy(dictionary, word, words, sweep));
    pool.extend(phonetic_strategy(word, words, sweep));
    pool.extend(keyboard_proximity_strategy(word, &index));
    pool.extend(ngram_strategy(word, words, sweep));

    rank::suggestion_set(pool, limit)
        .into_iter()
        .map(|candidate| Suggestion {
            word: candidate.word,
            distance: candidate.distance,
            confidence: candidate.confidence,
            source: candidate.source,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// EditDistanceStrategy
// ---------------------------------------------------------------------------

fn edit_distance_strategy(
    dictionary: &Dictionary,
    word: &str,
    words: &[String],
    sweep: &SweepIndex,
) -> Vec<Candidate> {
    let word_chars: Vec<char> = word.chars().collect();
    let target_length = word_chars.len();
    let length_min = target_length.saturating_sub(EDIT_MAX_DISTANCE);
    let length_max = target_length + EDIT_MAX_DISTANCE;

    let mut candidates: Vec<(String, usize, f64)> = Vec::new();
    let mut dict_chars: Vec<char> = Vec::with_capacity(32);
    // Only the ±2 length buckets — the index restores word-list order, so
    // the candidate array (and every tie in the ranking sort) is exactly
    // what the whole-list walk produced.
    for idx in sweep.indices_in_length_window(length_min as u32, length_max as u32) {
        let dict_word = &words[idx as usize];
        if dict_word == word {
            continue;
        }
        dict_chars.clear();
        dict_chars.extend(dict_word.chars());
        let Some(distance) =
            edit_distance::damerau_with_threshold(&word_chars, &dict_chars, EDIT_MAX_DISTANCE)
        else {
            continue;
        };
        if distance == 0 {
            continue;
        }
        let score = enhanced_score(&word_chars, &dict_chars, distance);
        candidates.push((dict_word.clone(), distance, score));
    }

    // Single-edit sweep (the gem's `edit_sweep_candidates`): the
    // distance-1 edits of the misspelling itself — adjacent
    // transposition (restricted Damerau, cost 1), substitution and
    // insertion over the dictionary TRY string, deletion — validated
    // against the full affix-aware lookup. The stem loop above can only
    // ever surface raw stems; without the sweep, dictionary FORMS
    // (affixed like "definately" → "definitely", or capitalization
    // variants like "Teh" → "The") can never be suggested.
    for (candidate_word, distance) in edit_sweep(dictionary, word, &word_chars) {
        let candidate_chars: Vec<char> = candidate_word.chars().collect();
        let score = enhanced_score(&word_chars, &candidate_chars, distance);
        candidates.push((candidate_word, distance, score));
    }

    if candidates.is_empty() {
        return Vec::new();
    }

    // `candidates.sort_by { |_, _, score| score }` — Float keys, MRI's
    // uniform introsort tie order.
    ruby_sort::sort_by(&mut candidates, |candidate| candidate.2);

    // Case-variant dedup (the gem's `seen_words`): keep the
    // best-scoring form, first after the sort, per lowercased word.
    let mut seen_words = std::collections::HashSet::new();
    candidates.retain(|(candidate_word, _, _)| seen_words.insert(candidate_word.to_lowercase()));

    let max_score = candidates
        .iter()
        .map(|candidate| candidate.2)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_score = candidates
        .iter()
        .map(|candidate| candidate.2)
        .fold(f64::INFINITY, f64::min);
    let score_range = (max_score - min_score).abs();

    let mut suggestions: Vec<Candidate> = Vec::new();
    for (dict_word, distance, score) in &candidates {
        let confidence = if score_range > 0.0 {
            1.0 - ((score - min_score) / score_range)
        } else {
            1.0
        };
        let jaro = ngram::typo_similarity(word, dict_word);

        if (confidence < EDIT_MIN_CONFIDENCE || jaro < EDIT_MIN_SIMILARITY)
            && suggestions.len() >= EDIT_MIN_RESULTS
        {
            continue;
        }

        suggestions.push(Candidate {
            word: dict_word.clone(),
            distance: *distance as u8,
            confidence,
            source: SuggestionSource::EditDistance,
            original_length: target_length,
            ngram_score: jaro,
        });

        if suggestions.len() >= STRATEGY_MAX_RESULTS {
            break;
        }
    }

    rank::suggestion_set(suggestions, STRATEGY_MAX_RESULTS)
}

/// `EditDistanceStrategy#edit_sweep_candidates` — every single-edit
/// variant of the word (canonical generator order: transposition,
/// substitution, insertion, deletion), deduplicated by exact string in
/// emission order, kept when the dictionary accepts it, with its true
/// Damerau distance.
fn edit_sweep(dictionary: &Dictionary, word: &str, word_chars: &[char]) -> Vec<(String, usize)> {
    let try_string = dictionary.try_string();
    let mut variants = permutations::swapchar(word);
    variants.extend(permutations::badchar(word, try_string));
    variants.extend(permutations::forgotchar(word, try_string));
    variants.extend(permutations::extrachar(word));

    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for candidate in variants {
        if candidate == word {
            continue;
        }
        if !seen.insert(candidate.clone()) {
            continue;
        }
        if !dictionary.correct(&candidate) {
            continue;
        }
        let candidate_chars: Vec<char> = candidate.chars().collect();
        let Some(distance) =
            edit_distance::damerau_with_threshold(word_chars, &candidate_chars, EDIT_MAX_DISTANCE)
        else {
            continue;
        };
        if distance == 0 {
            continue;
        }
        out.push((candidate, distance));
    }
    out
}

/// `EditDistanceStrategy#calculate_enhanced_score` — lower is better.
fn enhanced_score(original: &[char], suggestion: &[char], distance: usize) -> f64 {
    let mut score = distance as f64 * 1000.0;
    let suggestion_word: String = suggestion.iter().collect();
    score -= frequency::bonus(&suggestion_word) as f64;
    score += keyboard_penalty(original, suggestion) as f64;
    score -= transposition_bonus(original, suggestion) as f64;
    score -= typo_pattern_bonus(original, suggestion) as f64;
    let length_diff = original.len().abs_diff(suggestion.len()) as f64;
    score + length_diff * 50.0
}

/// `EditDistanceStrategy#keyboard_penalty` — Manhattan-distance-weighted
/// substitution penalty on the QWERTY grid (equal-length words only).
fn keyboard_penalty(original: &[char], suggestion: &[char]) -> u32 {
    if original.len() != suggestion.len() {
        return 0;
    }
    let layout = keyboard::Layout::qwerty();
    let mut penalty = 0u32;
    for (c1, c2) in original.iter().zip(suggestion) {
        if c1 == c2 {
            continue;
        }
        penalty += match layout.distance(*c1, *c2) {
            None => 50,     // unknown key — medium penalty
            Some(1) => 10,  // adjacent keys — very likely typo
            Some(2) => 30,  // somewhat likely
            Some(_) => 100, // far keys (0 included) — unlikely
        };
    }
    penalty
}

/// `EditDistanceStrategy#transposition_bonus` — adjacent-swap detection.
fn transposition_bonus(original: &[char], suggestion: &[char]) -> u32 {
    if original.len() != suggestion.len() {
        return 0;
    }
    let o: Vec<char> = original.iter().flat_map(|c| c.to_lowercase()).collect();
    let s: Vec<char> = suggestion.iter().flat_map(|c| c.to_lowercase()).collect();

    let mut transpositions = 0usize;
    for i in 0..o.len() {
        // Downcasing can change lengths (e.g. "İ" → "i̇"); Ruby's `o[i] ==
        // s[i]` reads past the end as nil, which never equals a char.
        let Some(o_char) = o.get(i) else { continue };
        let Some(s_char) = s.get(i) else { break };
        if o_char == s_char {
            continue;
        }
        // s.index(o[i], i + 1): first matching position at or after i+1.
        if let Some(match_idx) = (i + 1..s.len()).find(|&j| s[j] == *o_char) {
            // Ruby cross-indexes s[i] against o[match_idx] (nil reads are
            // never equal).
            let cross = o.get(match_idx).is_some_and(|oc| s.get(i) == Some(oc));
            if match_idx == i + 1 || (match_idx > i + 1 && cross) {
                transpositions += 1;
            }
        }
    }

    if transpositions == 1 {
        200
    } else {
        (transpositions * 100) as u32
    }
}

/// `EditDistanceStrategy#typo_pattern_bonus`.
fn typo_pattern_bonus(original: &[char], suggestion: &[char]) -> u32 {
    let mut bonus = 0u32;
    let original_word: String = original.iter().collect();

    // Pattern 1: missing double letter ("helo" → "hello").
    if suggestion.len() == original.len() + 1 {
        for i in 0..suggestion.len().saturating_sub(1) {
            if suggestion[i] == suggestion[i + 1] {
                // Removing the second of the pair must give the original.
                let expected: String = suggestion[..=i]
                    .iter()
                    .chain(&suggestion[i + 2..])
                    .collect();
                if expected == original_word {
                    bonus += 300;
                    break;
                }
            }
        }
    }

    // Pattern 2: extra double letter — dead in the gem: it reconstructs
    // `original` from `original` (identity) and compares against the
    // shorter `suggestion`, which can never match. Not ported as live
    // logic; recorded here so the omission is a documented choice.

    // Pattern 3: suggestion extends a shared 3-character prefix.
    if suggestion.len() > original.len() {
        let prefix: String = suggestion[..suggestion.len().min(3)].iter().collect();
        if original_word.starts_with(&prefix) {
            bonus += 30;
        }
    }

    bonus
}

// ---------------------------------------------------------------------------
// PhoneticStrategy
// ---------------------------------------------------------------------------

fn phonetic_strategy(word: &str, words: &[String], sweep: &SweepIndex) -> Vec<Candidate> {
    let word_code = phonetic::soundex_key(word).map_or(u32::MAX, sweep_index::pack_key);
    let word_chars: Vec<char> = word.chars().collect();

    let mut results: Vec<(&str, usize)> = Vec::new();
    let mut dict_chars: Vec<char> = Vec::with_capacity(32);
    for (idx, dict_word) in words.iter().enumerate() {
        if dict_word == word {
            continue;
        }
        // Packed codes compare without decoding the word — u32::MAX on
        // both sides is two empty codes, equal as before.
        if sweep.soundex(idx) != word_code {
            continue;
        }
        dict_chars.clear();
        dict_chars.extend(dict_word.chars());
        // Damerau-Levenshtein (the gem delegates to
        // `Algorithms::EditDistance`): an adjacent transposition costs 1.
        let Some(distance) =
            edit_distance::damerau_with_threshold(&word_chars, &dict_chars, STRATEGY_MAX_DISTANCE)
        else {
            continue;
        };
        if distance == 0 {
            continue;
        }
        results.push((dict_word, distance));
    }

    // `results.sort_by { |_, dist| dist }` — Integer keys.
    ruby_sort::sort_by(&mut results, |result| result.1 as i64);

    // `create_suggestion_set(sorted_words)` with no distances: every
    // suggestion gets distance 1 (the map-miss default) — gem quirk kept.
    let suggestions = results
        .into_iter()
        .map(|(dict_word, _)| Candidate {
            word: dict_word.to_owned(),
            distance: 1,
            confidence: 1.0 / (1.0 + 1.0),
            source: SuggestionSource::Phonetic,
            original_length: dict_word.chars().count(),
            ngram_score: 0.0,
        })
        .collect();

    rank::suggestion_set(suggestions, STRATEGY_MAX_RESULTS)
}

// ---------------------------------------------------------------------------
// KeyboardProximityStrategy
// ---------------------------------------------------------------------------

fn keyboard_proximity_strategy(word: &str, index: &WordIndex<'_>) -> Vec<Candidate> {
    let word_chars: Vec<char> = word.chars().collect();
    let variants = keyboard_variants(word, STRATEGY_MAX_DISTANCE);

    // Insertion-ordered `results_with_distances` (min distance per word).
    let mut order: Vec<String> = Vec::new();
    let mut distances: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for variant in &variants {
        let Some(dict_word) = index.find(variant) else {
            continue;
        };
        if dict_word == word {
            continue;
        }
        let dict_chars: Vec<char> = dict_word.chars().collect();
        // Damerau-Levenshtein (the gem delegates to
        // `Algorithms::EditDistance`): an adjacent transposition costs 1.
        let Some(distance) =
            edit_distance::damerau_with_threshold(&word_chars, &dict_chars, STRATEGY_MAX_DISTANCE)
        else {
            continue;
        };
        let similarity = ngram::typo_similarity(word, dict_word);
        if similarity < KEYBOARD_MIN_SIMILARITY {
            continue;
        }
        match distances.get_mut(dict_word) {
            Some(existing) => {
                if distance < *existing {
                    *existing = distance;
                }
            }
            None => {
                distances.insert(dict_word.to_owned(), distance);
                order.push(dict_word.to_owned());
            }
        }
    }

    let mut pairs: Vec<(String, usize)> = order
        .into_iter()
        .map(|word| {
            let distance = distances[&word];
            (word, distance)
        })
        .collect();
    ruby_sort::sort_by(&mut pairs, |pair| pair.1 as i64);

    // `create_suggestion_set(..., distances:, original_word: word)`.
    let suggestions = pairs
        .into_iter()
        .map(|(dict_word, distance)| Candidate {
            ngram_score: ngram::typo_similarity(word, &dict_word),
            word: dict_word,
            confidence: 1.0 / (1.0 + distance as f64),
            distance: distance as u8,
            source: SuggestionSource::KeyboardProximity,
            original_length: word_chars.len(),
        })
        .collect();

    rank::suggestion_set(suggestions, STRATEGY_MAX_RESULTS)
}

/// `KeyboardProximityStrategy#keyboard_variants` — exactly
/// `max_distance` rounds of substitution/deletion/insertion over the
/// neighbor table; each round REPLACES the variant set (so two rounds
/// yields exactly-two-operation variants). Ruby `Set` order is
/// first-insertion; replicated with a Vec + HashSet.
fn keyboard_variants(word: &str, max_distance: usize) -> Vec<String> {
    if word.is_empty() {
        return Vec::new();
    }
    let lowered: Vec<char> = word.to_lowercase().chars().collect();

    let mut variants: Vec<Vec<char>> = vec![lowered];
    for _ in 0..max_distance {
        let mut next: Vec<Vec<char>> = Vec::new();
        let mut next_seen: std::collections::HashSet<Vec<char>> = std::collections::HashSet::new();
        for variant in &variants {
            for i in 0..variant.len() {
                for neighbor in keyboard::proximity_neighbors(variant[i]) {
                    let neighbor: Vec<char> = neighbor.chars().collect();
                    let mutations: [Vec<char>; 3] = [
                        {
                            // substitution
                            let mut m = variant[..i].to_vec();
                            m.extend_from_slice(&neighbor);
                            m.extend_from_slice(&variant[i + 1..]);
                            m
                        },
                        {
                            // deletion
                            let mut m = variant[..i].to_vec();
                            m.extend_from_slice(&variant[i + 1..]);
                            m
                        },
                        {
                            // insertion
                            let mut m = variant[..i].to_vec();
                            m.extend_from_slice(&neighbor);
                            m.extend_from_slice(&variant[i..]);
                            m
                        },
                    ];
                    for mutation in mutations {
                        if next_seen.insert(mutation.clone()) {
                            next.push(mutation);
                        }
                    }
                }
            }

            // Adjacent transposition: swap each neighboring character
            // pair — a single 1-cost edit (the restricted Damerau
            // operation), enumerated alongside the neighbor mutations.
            if variant.len() >= 2 {
                for i in 0..variant.len() - 1 {
                    let mut swapped = variant[..i].to_vec();
                    swapped.push(variant[i + 1]);
                    swapped.push(variant[i]);
                    swapped.extend_from_slice(&variant[i + 2..]);
                    if next_seen.insert(swapped.clone()) {
                        next.push(swapped);
                    }
                }
            }
        }
        variants = next;
    }

    variants
        .into_iter()
        .map(|chars| chars.into_iter().collect())
        .collect()
}

/// `KeyboardProximityStrategy#find_word` — exact match first, then
/// case-insensitive (the word list is lowercased, so the fallback is a
/// formality — kept for shape).
/// Exact and case-insensitive word lookup over the dictionary word list,
/// built once per sweep — `find_word` used to scan the whole list (with a
/// `to_lowercase` allocation per word on the case-insensitive branch) per
/// keyboard variant, and the variant set reaches tens of thousands.
///
/// Semantics are `find_word` exactly: the exact match wins over the
/// case-insensitive one, and among words sharing a lowercase form the
/// FIRST in list order wins.
struct WordIndex<'a> {
    exact: std::collections::HashSet<&'a str>,
    /// Keys are the lowercase forms: borrowed when a word already is its
    /// own lowercase form (the overwhelming case — nothing is allocated),
    /// owned otherwise.
    lowered: std::collections::HashMap<std::borrow::Cow<'a, str>, &'a str>,
}

impl<'a> WordIndex<'a> {
    fn build(words: &'a [String]) -> Self {
        let mut index = Self {
            exact: std::collections::HashSet::with_capacity(words.len()),
            lowered: std::collections::HashMap::with_capacity(words.len()),
        };
        for word in words {
            index.exact.insert(word.as_str());
            let key = if is_lowercase_identity(word) {
                std::borrow::Cow::Borrowed(word.as_str())
            } else {
                std::borrow::Cow::<str>::Owned(word.to_lowercase())
            };
            index.lowered.entry(key).or_insert(word.as_str());
        }
        index
    }

    fn find(&self, word: &str) -> Option<&'a str> {
        if word.is_empty() {
            return None;
        }
        if let Some(exact) = self.exact.get(word) {
            return Some(exact);
        }
        let lowered = word.to_lowercase();
        self.lowered.get(lowered.as_str()).copied()
    }
}

/// True when `to_lowercase` would return the word unchanged — then the
/// slice itself is the map key and nothing is allocated.
fn is_lowercase_identity(word: &str) -> bool {
    word.chars().all(|c| {
        let mut lowered = c.to_lowercase();
        lowered.next() == Some(c) && lowered.next().is_none()
    })
}

// ---------------------------------------------------------------------------
// NgramStrategy
// ---------------------------------------------------------------------------

fn ngram_strategy(word: &str, words: &[String], sweep: &SweepIndex) -> Vec<Candidate> {
    let word_chars: Vec<char> = word.chars().collect();
    if word_chars.len() < NGRAM_N {
        return Vec::new();
    }
    let mut word_ngrams: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
    ngram::extract_ngram_keys(&word_chars, &mut word_ngrams);
    let mut scratch: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();

    let mut order: Vec<String> = Vec::new();
    let mut distances: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut dict_chars: Vec<char> = Vec::with_capacity(32);
    for (idx, dict_word) in words.iter().enumerate() {
        if dict_word == word {
            continue;
        }
        // The indexed length gates before the word is ever decoded.
        let dict_len = sweep.length(idx);
        if dict_len < NGRAM_N as u32 {
            continue;
        }
        dict_chars.clear();
        dict_chars.extend(dict_word.chars());
        // Pre-gate, mathematically equivalent to the threshold test:
        // similarity = intersection/union ≤ min_ngrams/max_ngrams, so a
        // word whose length-bound is already under NGRAM_MIN_SIMILARITY
        // can never pass the real test. Skips most of the dictionary
        // before any key is extracted.
        let lo = dict_len.min(word_chars.len() as u32) - (NGRAM_N as u32 - 1);
        let hi = dict_len.max(word_chars.len() as u32) - (NGRAM_N as u32 - 1);
        if (lo as f64) / (hi as f64) < NGRAM_MIN_SIMILARITY {
            continue;
        }
        ngram::extract_ngram_keys(&dict_chars, &mut scratch);
        let similarity = ngram::ngram_similarity_keys(&word_ngrams, &scratch);
        if similarity < NGRAM_MIN_SIMILARITY {
            continue;
        }
        let typo_sim = ngram::typo_similarity(word, dict_word);
        if typo_sim < NGRAM_MIN_TYPO_SIMILARITY {
            continue;
        }
        // Ruby Float#to_i truncates toward zero.
        let distance = ((1.0 - similarity) * 10.0) as usize;
        if distance == 0 {
            continue;
        }
        if !distances.contains_key(dict_word) {
            distances.insert(dict_word.clone(), distance);
            order.push(dict_word.clone());
        } else if distance < distances[dict_word] {
            distances.insert(dict_word.clone(), distance);
        }
    }

    let mut pairs: Vec<(String, usize)> = order
        .into_iter()
        .map(|word| {
            let distance = distances[&word];
            (word, distance)
        })
        .collect();
    ruby_sort::sort_by(&mut pairs, |pair| pair.1 as i64);

    let suggestions = pairs
        .into_iter()
        .map(|(dict_word, distance)| Candidate {
            ngram_score: ngram::typo_similarity(word, &dict_word),
            word: dict_word,
            confidence: 1.0 / (1.0 + distance as f64),
            distance: distance as u8,
            source: SuggestionSource::Ngram,
            original_length: word_chars.len(),
        })
        .collect();

    rank::suggestion_set(suggestions, STRATEGY_MAX_RESULTS)
}
