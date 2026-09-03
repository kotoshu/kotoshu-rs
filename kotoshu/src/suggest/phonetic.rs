//! Phonetic coding: Soundex, ported from the gem's
//! `Suggestions::Strategies::PhoneticStrategy` (the gem also carries a
//! simplified Metaphone, unused on the default `algorithm: :soundex`
//! path and therefore not ported).

/// Soundex code (letter + three digits). Ported quirk-for-quirk: the
/// first letter is kept verbatim, `H`/`W` (code `"0"`) never reset the
/// previous-code memory, and non-ASCII letters are stripped after
/// upcasing.
pub fn soundex(word: &str) -> String {
    if word.is_empty() {
        return String::new();
    }
    // Ruby: word.upcase.gsub(/[^A-Z]/, "")
    let letters: Vec<char> = word
        .to_uppercase()
        .chars()
        .filter(|c| c.is_ascii_uppercase())
        .collect();
    if letters.is_empty() {
        return String::new();
    }

    let first_letter = letters[0];
    let rest = &letters[1..];

    let mut code = String::from(first_letter);
    let mut prev_code = soundex_encode(first_letter);
    let mut i = 0;
    while code.chars().count() < 4 && i < rest.len() {
        let encoded = soundex_encode(rest[i]);
        if encoded != '0' && encoded != prev_code {
            code.push(encoded);
        }
        if encoded != '0' {
            prev_code = encoded;
        }
        i += 1;
    }

    // code.ljust(4, "0")[0...4]
    while code.chars().count() < 4 {
        code.push('0');
    }
    code.chars().take(4).collect()
}

/// Soundex digit for one letter (`"0"` = uncoded).
fn soundex_encode(c: char) -> char {
    match c.to_ascii_uppercase() {
        'B' | 'P' | 'F' | 'V' => '1',
        'C' | 'S' | 'K' | 'G' | 'J' | 'Q' | 'X' | 'Z' => '2',
        'D' | 'T' => '3',
        'L' => '4',
        'M' | 'N' => '5',
        'R' => '6',
        _ => '0',
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classic_soundex_examples() {
        // Frozen from the gem's implementation (differs from textbook
        // Soundex on H/W handling — see `hw_never_resets_previous_code`).
        assert_eq!(soundex("Robert"), "R163");
        assert_eq!(soundex("Rupert"), "R163");
        assert_eq!(soundex("Ashcraft"), "A261");
    }

    #[test]
    fn hw_never_resets_previous_code() {
        // Ashcraft: S,H(0),C — H doesn't reset the previous code, so the
        // C is suppressed against the S (=> A26, not A22).
        assert_eq!(soundex("Tymczak"), "T520");
        assert_eq!(soundex("Pfister"), "P236");
        assert_eq!(soundex("hello"), "H400");
    }

    #[test]
    fn empty_and_non_ascii() {
        assert_eq!(soundex(""), "");
        assert_eq!(soundex("é"), "");
        assert_eq!(soundex("a"), "A000");
    }
}
