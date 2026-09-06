//! Acceptance tests for the `EditDistanceStrategy` edit sweep — the
//! distance-1 candidate enumeration (adjacent transposition, TRY
//! substitution/insertion, deletion) validated against the full
//! affix-aware lookup, mirroring the gem's
//! `spec/kotoshu/suggestions/strategies/edit_distance_sweep_spec.rb`.
//!
//! The dictionary mirrors the gem fixture: a TRY string, a productive
//! `-ly` suffix (`SFX Y`), and stems chosen so the acceptance answers
//! are reachable ONLY through the sweep ("definitely" is a suffix form,
//! never a stem; "The" is the INITCAP form of the stem "the").

use kotoshu::dict::Dictionary;

const AFF: &str = concat!(
    "SET UTF-8\n",
    "TRY esianrtolcdugmphbyfvkwz\n",
    "SFX Y Y 1\n",
    "SFX Y 0 ly .\n"
);

const DIC: &str = concat!(
    "5\n",
    "the\n",
    "definite/Y\n",
    "receive\n",
    "world\n",
    "te\n"
);

fn dictionary() -> Dictionary {
    Dictionary::load_from_sources(AFF, DIC).expect("fixture dictionary loads")
}

fn suggestions(word: &str) -> Vec<String> {
    dictionary()
        .suggest(word, 5)
        .into_iter()
        .map(|s| s.word)
        .collect()
}

#[test]
fn try_string_is_exposed() {
    assert_eq!(dictionary().try_string(), Some("esianrtolcdugmphbyfvkwz"));
}

#[test]
fn transposition_suggests_capitalization_form_first() {
    // "The" is the INITCAP form of the stem "the"; the stem comparison
    // alone charges the case difference plus the transposition.
    let words = suggestions("Teh");
    assert_eq!(words.first().map(String::as_str), Some("The"));
    // The lowercased stem must not appear alongside its form.
    assert!(!words.iter().any(|w| w == "the"));
}

#[test]
fn substitution_suggests_suffix_form_first() {
    // "definitely" = definite + -ly: a valid lookup, never a stem.
    assert!(!dictionary().words().contains(&"definitely".to_string()));
    assert!(dictionary().correct("definitely"));
    assert_eq!(
        suggestions("definately").first().map(String::as_str),
        Some("definitely")
    );
}

#[test]
fn insertion_suggests_suffix_form_first() {
    assert_eq!(
        suggestions("definitly").first().map(String::as_str),
        Some("definitely")
    );
}

#[test]
fn deletion_suggests_suffix_form_first() {
    assert_eq!(
        suggestions("definitelyy").first().map(String::as_str),
        Some("definitely")
    );
}

#[test]
fn transposition_regressions_hold() {
    assert_eq!(
        suggestions("recieve").first().map(String::as_str),
        Some("receive")
    );
    assert_eq!(
        suggestions("wrold").first().map(String::as_str),
        Some("world")
    );
}
