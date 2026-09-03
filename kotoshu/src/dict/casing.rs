//! Capitalization handling, ported from the gem's
//! `Algorithms::Capitalization` (itself from Spylls).
//!
//! Three casings exist: the standard one, German (`CHECKSHARPS` or
//! `LANG de*`, where uppercased `SS` lowercases to both `ss` and `ß`) and
//! Turkic (`LANG tr*`/`az*`/…, where `I` lowercases to `ı` and `i`
//! uppercases to `İ`).

/// Capitalization type of a word, from [`Casing::guess`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapType {
    /// All lowercase: "foo".
    No,
    /// Only the first letter capitalized: "Foo".
    Init,
    /// All uppercase: "FOO".
    All,
    /// Mixed capitalization: "fooBar".
    Huh,
    /// Mixed capitalization with a capitalized first letter: "FooBar".
    HuhInit,
}

/// Which language-specific casing rules apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Standard,
    German,
    Turkic,
}

/// Casing algorithms for one dictionary (see [`Kind`]).
#[derive(Debug, Clone, Copy)]
pub struct Casing {
    kind: Kind,
}

impl Casing {
    /// The standard casing.
    pub fn standard() -> Self {
        Self {
            kind: Kind::Standard,
        }
    }

    /// German casing (`ß`/`SS` lowercased two ways).
    pub fn german() -> Self {
        Self { kind: Kind::German }
    }

    /// Turkic casing (dotted/dotless i).
    pub fn turkic() -> Self {
        Self { kind: Kind::Turkic }
    }

    /// Select the casing the gem's `LookupBuilder#select_casing` picks for
    /// the given `LANG` value and `CHECKSHARPS` setting.
    pub fn select(lang: Option<&str>, checksharps: bool) -> Self {
        let lang = lang.unwrap_or_default().to_lowercase();
        if checksharps {
            Self::german()
        } else if ["tr", "az", "crh", "tt", "krc", "kaa"]
            .iter()
            .any(|prefix| lang.starts_with(prefix))
        {
            Self::turkic()
        } else if lang.starts_with("de") {
            Self::german()
        } else {
            Self::standard()
        }
    }

    /// Guess the word's capitalization type.
    pub fn guess(&self, word: &str) -> CapType {
        let result = self.guess_standard(word);
        if self.kind == Kind::German && word.contains('ß') {
            let without = word.replace('ß', "");
            if self.guess_standard(&without) == CapType::All {
                return CapType::All;
            }
        }
        result
    }

    fn guess_standard(&self, word: &str) -> CapType {
        if lowercase(word) == word {
            return CapType::No;
        }
        if uppercase(word) == word {
            return CapType::All;
        }
        let mut chars = word.chars();
        match chars.next() {
            None => CapType::No,
            Some(first) => {
                let rest = chars.as_str();
                let first_upper = upper_char(first) == first.to_string();
                if first_upper && lowercase(rest) == rest {
                    return CapType::Init;
                }
                if first_upper {
                    CapType::HuhInit
                } else {
                    CapType::Huh
                }
            }
        }
    }

    /// All possible lowercasings of the word (German `SS` yields both `ss`
    /// and `ß` variants; empty for words starting with `İ`).
    pub fn lower(&self, word: &str) -> Vec<String> {
        match self.kind {
            Kind::Turkic => {
                let translated: String = word
                    .chars()
                    .map(|c| match c {
                        'İ' => 'i',
                        'I' => 'ı',
                        other => other,
                    })
                    .collect();
                self.lower_standard(&translated)
            }
            _ => self.lower_standard(word),
        }
    }

    fn lower_standard(&self, word: &str) -> Vec<String> {
        // Mirrors the gem: empty words and words starting with İ cannot be
        // lowercased at all.
        if word.is_empty() || word.starts_with('İ') {
            return Vec::new();
        }
        let lowered = lowercase(word).replace(LOWERED_DOTLESS_I, "i");
        match self.kind {
            Kind::German if word.contains("SS") => {
                let mut variants = sharp_s_variants(&lowered, 0);
                variants.push(lowered);
                variants
            }
            _ => vec![lowered],
        }
    }

    /// Uppercase the word (Turkic `i` → `İ`, `ı` → `I`).
    pub fn upper(&self, word: &str) -> String {
        match self.kind {
            Kind::Turkic => {
                let translated: String = word
                    .chars()
                    .map(|c| match c {
                        'i' => 'İ',
                        'ı' => 'I',
                        other => other,
                    })
                    .collect();
                uppercase(&translated)
            }
            _ => uppercase(word),
        }
    }

    /// All capitalizations (first letter uppercased, rest lowercased) of
    /// the word.
    pub fn capitalize(&self, word: &str) -> Vec<String> {
        let mut chars = word.chars();
        match chars.next() {
            None => vec![],
            Some(first) if chars.as_str().is_empty() => vec![self.upper(&first.to_string())],
            Some(first) => {
                let upper_first = self.upper(&first.to_string());
                self.lower(chars.as_str())
                    .into_iter()
                    .map(|lowered| format!("{upper_first}{lowered}"))
                    .collect()
            }
        }
    }

    /// Variants with only the first letter lowercased.
    pub fn lowerfirst(&self, word: &str) -> Vec<String> {
        let mut chars = word.chars();
        match chars.next() {
            None => vec![],
            Some(first) => self
                .lower(&first.to_string())
                .into_iter()
                .map(|lowered| format!("{lowered}{}", chars.as_str()))
                .collect(),
        }
    }

    /// Hypotheses of how a correctly spelled word might be cased in the
    /// dictionary, paired with the original word's capitalization type.
    pub fn variants(&self, word: &str) -> (CapType, Vec<String>) {
        let captype = self.guess(word);
        let mut variants = vec![word.to_owned()];
        match captype {
            CapType::No | CapType::Huh => {}
            CapType::Init => variants.extend(self.lower(word)),
            CapType::HuhInit => variants.extend(self.lowerfirst(word)),
            CapType::All => {
                variants.extend(self.lower(word));
                variants.extend(self.capitalize(word));
            }
        }
        (captype, variants)
    }
}

/// `i` followed by U+0307 (combining dot above) — what `İ` lowercases to.
const LOWERED_DOTLESS_I: &str = "i\u{0307}";

/// Full Unicode lowercase of a string (Ruby `String#downcase`).
pub(crate) fn unicode_lowercase(word: &str) -> String {
    lowercase(word)
}

fn lowercase(word: &str) -> String {
    let mut out = String::with_capacity(word.len());
    for c in word.chars() {
        for lowered in c.to_lowercase() {
            out.push(lowered);
        }
    }
    out
}

/// Full Unicode uppercase of a string (Ruby `String#upcase`).
fn uppercase(word: &str) -> String {
    let mut out = String::with_capacity(word.len());
    for c in word.chars() {
        for uppered in c.to_uppercase() {
            out.push(uppered);
        }
    }
    out
}

/// Uppercase of a single character as a string (may be multi-char).
fn upper_char(c: char) -> String {
    let mut out = String::new();
    for uppered in c.to_uppercase() {
        out.push(uppered);
    }
    out
}

/// All variants with `ss` replaced by `ß`, recursively, mirroring the gem's
/// `GermanCasing#sharp_s_variants` (start offset included so each
/// replacement site is enumerated exactly once). Offsets are character
/// positions, as in Ruby.
fn sharp_s_variants(text: &str, start: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let Some(pos) =
        (start..chars.len().saturating_sub(1)).find(|&i| chars[i] == 's' && chars[i + 1] == 's')
    else {
        return Vec::new();
    };
    let replaced: String = chars[..pos]
        .iter()
        .copied()
        .chain(std::iter::once('ß'))
        .chain(chars[pos + 2..].iter().copied())
        .collect();

    let mut variants = vec![replaced.clone()];
    variants.extend(sharp_s_variants(&replaced, pos + 1));
    variants.extend(sharp_s_variants(text, pos + 2));
    variants
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guesses_capitalization_types() {
        let casing = Casing::standard();
        assert_eq!(casing.guess("foo"), CapType::No);
        assert_eq!(casing.guess("Foo"), CapType::Init);
        assert_eq!(casing.guess("FOO"), CapType::All);
        assert_eq!(casing.guess("fooBar"), CapType::Huh);
        assert_eq!(casing.guess("FooBar"), CapType::HuhInit);
    }

    #[test]
    fn standard_variants() {
        let casing = Casing::standard();
        assert_eq!(
            casing.variants("Kitten"),
            (
                CapType::Init,
                vec!["Kitten".to_owned(), "kitten".to_owned()]
            )
        );
        assert_eq!(
            casing.variants("FOO"),
            (
                CapType::All,
                vec!["FOO".to_owned(), "foo".to_owned(), "Foo".to_owned()]
            )
        );
        assert_eq!(
            casing.variants("foo"),
            (CapType::No, vec!["foo".to_owned()])
        );
    }

    #[test]
    fn german_casing() {
        let casing = Casing::german();
        assert_eq!(casing.guess("STRAßE"), CapType::All);
        assert_eq!(
            casing.lower("STRASSE"),
            vec!["straße".to_owned(), "strasse".to_owned()]
        );
    }

    #[test]
    fn turkic_casing() {
        let casing = Casing::turkic();
        assert_eq!(casing.lower("Izmir"), vec!["ızmir".to_owned()]);
        assert_eq!(casing.upper("Izmir"), "IZMİR");
    }

    #[test]
    fn select_casing_mirrors_the_gem() {
        assert_eq!(Casing::select(Some("de_DE"), false).kind, Kind::German);
        assert_eq!(Casing::select(None, true).kind, Kind::German);
        assert_eq!(Casing::select(Some("tr_TR"), false).kind, Kind::Turkic);
        assert_eq!(Casing::select(Some("en_US"), false).kind, Kind::Standard);
    }
}
