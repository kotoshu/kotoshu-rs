//! Conformance-vector runner.
//!
//! Reads `tests/fixtures/vectors.jsonl` (synced from the gem by
//! `scripts/sync_conformance.sh`) or, when the fixtures were never synced,
//! the committed pack in `tests/conformance/*.jsonl`. Each line is one
//! vector in the gem's documented API shape:
//!
//! ```json
//! {"kind":"correct","language":"en","dictionary":"spec/integrational/fixtures/base",
//!  "input":"created","expected":true}
//! {"kind":"suggest","language":"en","dictionary":"spec/integrational/fixtures/base",
//!  "input":"hlelo","limit":5,"expected":[{"word":"hello","distance":1,
//!  "confidence":1.0,"source":"edit_distance"}]}
//! ```
//!
//! The live pack is the gem's golden vectors (2630 lines, 125 fixture
//! dictionaries), exported by `rake kotoshu:conformance:export` in the gem
//! repo (plan 67 M3) — one source of truth, three enforcement points (C ABI,
//! `ruby` feature, wasm32). `expected` freezes the Ruby engine's ACTUAL
//! behavior; kotoshu-rs must reproduce it byte-for-byte. v0-placeholders.jsonl
//! is the original hand-written trio, kept for history.
//!
//! P2 policy: `correct` vectors are asserted through the engine
//! ([`kotoshu::dict::Dictionary`]); `suggest` vectors are asserted through
//! `Dictionary::suggest` — ordered equality of (word, distance, source)
//! with exact f64 confidence comparison (the exporter wrote Ruby `Float`s,
//! and the Rust pipeline performs the same IEEE-754 operations in the same
//! order, so equality is exact). When the fixture dictionaries are absent
//! (local checkout without the gem) the assertions skip gracefully; a
//! partial fixture set is an error.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use kotoshu::dict::Dictionary;
use serde::Deserialize;

const FIXTURES_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../tests/fixtures");
const CONFORMANCE_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../tests/conformance");

/// One conformance vector (superset of both kinds' fields).
#[derive(Deserialize)]
struct Vector {
    kind: String,
    #[serde(default)]
    dictionary: String,
    input: String,
    #[serde(default)]
    limit: Option<usize>,
    expected: serde_json::Value,
}

/// One expected suggestion entry of a `suggest` vector.
#[derive(Deserialize)]
struct ExpectedSuggestion {
    word: String,
    distance: u8,
    confidence: f64,
    source: String,
}

#[test]
fn conformance_vectors_are_well_formed() {
    let dir = Path::new(CONFORMANCE_DIR);
    if !dir.is_dir() {
        return;
    }

    let mut files = 0;
    let mut vectors = 0;

    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(error) => panic!("cannot read conformance dir {CONFORMANCE_DIR}: {error}"),
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_none_or(|ext| ext != "jsonl") {
            continue;
        }
        files += 1;
        for (line_no, line) in fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()))
            .lines()
            .enumerate()
        {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            serde_json::from_str::<serde_json::Value>(line).unwrap_or_else(|error| {
                panic!("{}:{}: invalid JSON: {error}", path.display(), line_no + 1)
            });
            vectors += 1;
        }
    }

    if files > 0 {
        assert!(vectors > 0, "conformance files present but empty: {dir:?}");
    }
    eprintln!("conformance: {files} files, {vectors} vectors");
}

/// Locate the vector pack: the synced copy wins (it is paired with the
/// synced fixtures), the committed copy is the fallback.
fn vectors_file() -> Option<PathBuf> {
    let synced = Path::new(FIXTURES_DIR).join("vectors.jsonl");
    if synced.is_file() {
        return Some(synced);
    }
    let dir = Path::new(CONFORMANCE_DIR);
    let entries = fs::read_dir(dir).ok()?;
    let mut files: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "jsonl"))
        .collect();
    files.sort();
    files.pop()
}

fn read_vectors(path: &Path) -> Vec<Vector> {
    let text = fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
    text.lines()
        .enumerate()
        .filter_map(|(line_no, line)| {
            let line = line.trim();
            if line.is_empty() {
                return None;
            }
            serde_json::from_str(line).unwrap_or_else(|error| {
                panic!(
                    "{}:{}: invalid vector: {error}",
                    path.display(),
                    line_no + 1
                )
            })
        })
        .collect()
}

fn fixture_dictionary_paths(dictionary: &str) -> (PathBuf, PathBuf) {
    let base = Path::new(FIXTURES_DIR).join(dictionary);
    (base.with_extension("aff"), base.with_extension("dic"))
}

#[test]
fn conformance_correct_vectors() {
    let Some(path) = vectors_file() else {
        eprintln!("conformance: no vector pack found; skipping");
        return;
    };
    let vectors = read_vectors(&path);

    let mut correct_total = 0;
    let mut correct_asserted = 0;
    let mut correct_skipped = 0;
    let mut suggest_counted = 0;
    let mut loaded: HashMap<String, Dictionary> = HashMap::new();
    let mut failures: Vec<String> = Vec::new();
    const REPORT_LIMIT: usize = 40;

    for vector in &vectors {
        match vector.kind.as_str() {
            "suggest" => {
                suggest_counted += 1;
                let (aff_path, dic_path) = fixture_dictionary_paths(&vector.dictionary);
                if !aff_path.is_file() || !dic_path.is_file() {
                    correct_skipped += 1;
                    continue;
                }
                let dictionary = match loaded.entry(vector.dictionary.clone()) {
                    std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        match Dictionary::load(&aff_path, &dic_path) {
                            Ok(dictionary) => entry.insert(dictionary),
                            Err(error) => {
                                failures.push(format!(
                                    "{}: dictionary {} failed to load: {error}",
                                    vector.input, vector.dictionary
                                ));
                                continue;
                            }
                        }
                    }
                };
                let expected_list: Vec<ExpectedSuggestion> =
                    serde_json::from_value(vector.expected.clone()).unwrap_or_else(|error| {
                        panic!(
                            "suggest vector with malformed expected list {:?}: {error}",
                            vector.input
                        )
                    });
                let got = dictionary.suggest(&vector.input, vector.limit.unwrap_or(5));
                if let Some(divergence) = first_divergence(&got, &expected_list)
                    && failures.len() < REPORT_LIMIT
                {
                    failures.push(format!(
                        "{}: {} input {:?}: {divergence}",
                        vector.dictionary,
                        path.display(),
                        vector.input
                    ));
                }
            }
            "correct" => {
                correct_total += 1;
                let (aff_path, dic_path) = fixture_dictionary_paths(&vector.dictionary);
                if !aff_path.is_file() || !dic_path.is_file() {
                    correct_skipped += 1;
                    continue;
                }
                let dictionary = match loaded.entry(vector.dictionary.clone()) {
                    std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        match Dictionary::load(&aff_path, &dic_path) {
                            Ok(dictionary) => entry.insert(dictionary),
                            Err(error) => {
                                failures.push(format!(
                                    "{}: dictionary {} failed to load: {error}",
                                    vector.input, vector.dictionary
                                ));
                                continue;
                            }
                        }
                    }
                };
                let expected = vector.expected.as_bool().unwrap_or_else(|| {
                    panic!("correct vector with non-boolean expected: {}", vector.input)
                });
                let got = dictionary.correct(&vector.input);
                if got == expected {
                    correct_asserted += 1;
                } else if failures.len() < REPORT_LIMIT {
                    failures.push(format!(
                        "{}: {} input {:?}: expected {expected}, got {got}",
                        vector.dictionary,
                        path.display(),
                        vector.input
                    ));
                }
            }
            other => panic!("unknown vector kind: {other}"),
        }
    }

    if correct_asserted == 0 && correct_skipped == correct_total {
        eprintln!(
            "conformance: fixtures absent; skipped {} correct vectors, counted {} suggest vectors",
            correct_total, suggest_counted
        );
        return;
    }
    assert_eq!(
        correct_skipped, 0,
        "partial fixture sync: {correct_skipped} vectors had no dictionary fixture"
    );
    eprintln!(
        "conformance: {} correct vectors, {correct_asserted}/{correct_total} asserted (all passing), {} suggest vectors asserted",
        path.display(),
        suggest_counted
    );
    assert!(
        failures.is_empty(),
        "{} conformance failures (of {} correct + {} suggest vectors):\n{}",
        failures.len(),
        correct_total,
        suggest_counted,
        failures.join("\n")
    );
}

/// Ordered-equality check with a first-divergence diagnostic.
fn first_divergence(
    got: &[kotoshu::suggest::Suggestion],
    expected: &[ExpectedSuggestion],
) -> Option<String> {
    for (index, (g, e)) in got.iter().zip(expected).enumerate() {
        if g.word != e.word {
            return Some(format!(
                "[{index}] word: expected {:?}, got {:?} (full expected {:?}, got {:?})",
                e.word,
                g.word,
                expected.iter().map(|s| s.word.as_str()).collect::<Vec<_>>(),
                got.iter().map(|s| s.word.as_str()).collect::<Vec<_>>(),
            ));
        }
        if g.distance != e.distance {
            return Some(format!(
                "[{index}] {:?}: distance: expected {}, got {}",
                e.word, e.distance, g.distance
            ));
        }
        if g.source.as_str() != e.source {
            return Some(format!(
                "[{index}] {:?}: source: expected {:?}, got {:?}",
                e.word,
                e.source,
                g.source.as_str()
            ));
        }
        if g.confidence != e.confidence {
            return Some(format!(
                "[{index}] {:?}: confidence: expected {}, got {} (delta {})",
                e.word,
                e.confidence,
                g.confidence,
                (e.confidence - g.confidence).abs()
            ));
        }
    }
    if got.len() != expected.len() {
        return Some(format!(
            "length: expected {} suggestions, got {} (expected {:?}, got {:?})",
            expected.len(),
            got.len(),
            expected.iter().map(|s| s.word.as_str()).collect::<Vec<_>>(),
            got.iter().map(|s| s.word.as_str()).collect::<Vec<_>>(),
        ));
    }
    None
}
