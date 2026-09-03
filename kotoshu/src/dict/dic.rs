//! `.dic` file loading, ported from the gem's `Readers::DicReader` (+ the
//! index building of `LookupBuilder#build_dic_structure`).
//!
//! Quirks preserved: morphological data splits at the first tab or before
//! the first `key:value` token; only the character immediately before a
//! `/` counts for the escape check; `\/` becomes `/` in stems (except for
//! words that start with a slash); `AF` aliases apply only to digit-only
//! flag strings; `IGNORE` characters are stripped from stems at read time;
//! only stems whose casing is not `NO` feed the lowercase index; and `ph:`
//! morph data is converted into REP entries.

use std::collections::{BTreeSet, HashMap};

use super::aff::{FlagFormat, RepEntry, strip_ignore};
use super::casing::{CapType, Casing};

/// One dictionary entry.
#[derive(Debug, Clone)]
pub struct DicEntry {
    /// Stem with `IGNORE` characters already removed.
    pub stem: String,
    /// The entry's flags.
    pub flags: BTreeSet<String>,
}

/// The indexed dictionary: entries plus the exact-stem and lowercase
/// indexes used by lookup.
#[derive(Debug)]
pub struct Dic {
    /// Entries in file order.
    pub entries: Vec<DicEntry>,
    /// Exact stems → entry indexes (homonyms preserved).
    word_index: HashMap<String, Vec<usize>>,
    /// Lowercased stems → entry indexes, for the ALLCAPS fallback.
    lowercase_index: HashMap<String, Vec<usize>>,
    /// The suggestion pipeline's word list (the gem's
    /// `Dictionary::Hunspell#words`, i.e. `@word_index.keys`): lowercased
    /// RAW stems — before `IGNORE` stripping — in first-occurrence order.
    suggest_words: Vec<String>,
}

impl Dic {
    /// Parse the (already decoded, stripped, non-empty) `.dic` lines — the
    /// first line is the declared word count and is skipped — and build the
    /// indexes. Returns the dictionary and the `ph:`-derived REP entries
    /// (the gem appends those to the aff REP table).
    pub fn parse(
        lines: &[String],
        flag_format: FlagFormat,
        aliases: &HashMap<String, BTreeSet<String>>,
        casing: &Casing,
        ignore: &[char],
    ) -> (Self, Vec<RepEntry>) {
        let mut entries = Vec::new();
        let mut word_index: HashMap<String, Vec<usize>> = HashMap::new();
        let mut lowercase_index: HashMap<String, Vec<usize>> = HashMap::new();
        let mut ph_reps = Vec::new();
        let mut suggest_words: Vec<String> = Vec::new();
        let mut suggest_seen: std::collections::HashSet<String> = std::collections::HashSet::new();

        for line in lines.iter().skip(1) {
            let (head, morph) = split_stem_and_morph(line);
            let head = head.trim();

            let (stem, flags): (String, BTreeSet<String>) = if head.starts_with('/') {
                // A word starting with `/` is the word itself, not an empty
                // stem plus flags (and its `\/` escapes stay untouched).
                (head.to_owned(), BTreeSet::new())
            } else {
                let (stem, flags_str) = match unescaped_slash_index(head) {
                    Some(idx) => (head[..idx].to_owned(), Some(&head[idx + 1..])),
                    None => (head.to_owned(), None),
                };
                let stem = if stem.contains("\\/") {
                    stem.replace("\\/", "/")
                } else {
                    stem
                };
                let flags = flags_str
                    .filter(|s| !s.is_empty())
                    .map(|s| parse_dic_flags(s, flag_format, aliases))
                    .unwrap_or_default();
                (stem, flags)
            };

            // Capture the pre-`IGNORE` stem for the suggest word list.
            let raw_stem = stem.clone();
            let stem = strip_ignore(&stem, ignore);
            // Suggest-word key: the RAW stem lowercased (pre-IGNORE), as
            // the gem's `Dictionary::Hunspell#build_word_index` does.
            let raw_lower = raw_stem.to_lowercase();
            if suggest_seen.insert(raw_lower.clone()) {
                suggest_words.push(raw_lower);
            }
            let ph_tokens: Vec<&str> = morph
                .split_whitespace()
                .filter_map(|token| token.strip_prefix("ph:"))
                .collect();
            ph_reps.extend(ph_rep_entries(&stem, &ph_tokens));

            let idx = entries.len();
            entries.push(DicEntry {
                stem: stem.clone(),
                flags,
            });
            word_index.entry(stem.clone()).or_default().push(idx);

            if casing.guess(&stem) != CapType::No {
                for lowered in casing.lower(&stem) {
                    lowercase_index.entry(lowered).or_default().push(idx);
                }
            }
        }

        (
            Self {
                entries,
                word_index,
                lowercase_index,
                suggest_words,
            },
            ph_reps,
        )
    }

    /// The suggestion pipeline's word list (first-occurrence lowercased
    /// raw stems, file order).
    pub fn suggest_words(&self) -> &[String] {
        &self.suggest_words
    }

    /// Exact-stem homonyms (empty when the stem is unknown).
    pub fn homonyms(&self, word: &str) -> &[usize] {
        self.word_index.get(word).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Lowercase-index homonyms (ALLCAPS fallback).
    pub fn homonyms_ignorecase(&self, word: &str) -> &[usize] {
        self.lowercase_index
            .get(word)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Whether `word`'s exact-stem homonyms carry `flag` (all of them when
    /// `for_all`). Vacuously false when there are no homonyms.
    pub fn has_flag(&self, word: &str, flag: &str, for_all: bool) -> bool {
        let homonyms = self.word_index.get(word).map(Vec::as_slice).unwrap_or(&[]);
        if homonyms.is_empty() {
            return false;
        }
        if for_all {
            homonyms
                .iter()
                .all(|&idx| self.entries[idx].flags.contains(flag))
        } else {
            homonyms
                .iter()
                .any(|&idx| self.entries[idx].flags.contains(flag))
        }
    }
}

/// Parse dic flags. Unlike the aff side, `AF` aliases apply only when the
/// string is entirely digits (mirroring `DicReader::Word.parse_flags`).
fn parse_dic_flags(
    string: &str,
    format: FlagFormat,
    aliases: &HashMap<String, BTreeSet<String>>,
) -> BTreeSet<String> {
    if !aliases.is_empty() && !string.is_empty() && string.chars().all(|c| c.is_ascii_digit()) {
        if let Some(set) = aliases.get(string) {
            return set.clone();
        }
        return BTreeSet::new();
    }
    parse_flags_format(string, format)
}

/// Split flags per the `FLAG` format (shared by aff and dic parsing).
fn parse_flags_format(string: &str, format: FlagFormat) -> BTreeSet<String> {
    let mut flags = BTreeSet::new();
    match format {
        FlagFormat::Short | FlagFormat::Utf8 => {
            flags.extend(string.chars().map(|c| c.to_string()));
        }
        FlagFormat::Long => {
            let mut chars = string.chars();
            while let (Some(a), Some(b)) = (chars.next(), chars.next()) {
                flags.insert(format!("{a}{b}"));
            }
        }
        FlagFormat::Num => {
            let mut current = String::new();
            for c in string.chars() {
                if c.is_ascii_digit() {
                    current.push(c);
                } else if !current.is_empty() {
                    flags.insert(std::mem::take(&mut current));
                }
            }
            if !current.is_empty() {
                flags.insert(current);
            }
        }
    }
    flags
}

/// Split a dic line into its stem portion and morphological-data portion:
/// at the first tab, or before the first `key:value` token (the gem's
/// `split_stem_and_morph`); the morph portion keeps its leading
/// whitespace.
fn split_stem_and_morph(line: &str) -> (&str, &str) {
    if let Some(tab) = line.find('\t') {
        return (&line[..tab], &line[tab + 1..]);
    }
    // Mirror /(.*?)(\s+[a-zA-Z]+:[^\s].*)$/: the earliest split point where
    // the remainder starts with whitespace, letters, ':' and a non-space
    // character, all the way to the end.
    for (start, _) in line.char_indices() {
        let rest = &line[start..];
        let mut chars = rest.chars();
        if !chars.next().is_some_and(|c| c.is_whitespace()) {
            continue;
        }
        let mut letters = chars.by_ref().peekable();
        while letters.peek().is_some_and(|c| c.is_ascii_alphabetic()) {
            letters.next();
        }
        if letters.next_if(|&c| c == ':') != Some(':') {
            continue;
        }
        match letters.next() {
            Some(c) if !c.is_whitespace() => {}
            _ => continue,
        }
        return (&line[..start], rest);
    }
    (line, "")
}

/// Byte index of the first `/` whose immediately preceding character is
/// not a backslash (the gem checks only the one preceding character).
fn unescaped_slash_index(head: &str) -> Option<usize> {
    let mut previous: Option<char> = None;
    for (idx, c) in head.char_indices() {
        if c == '/' && !matches!(previous, Some('\\')) {
            return Some(idx);
        }
        previous = Some(c);
    }
    None
}

/// Convert `ph:` morph payloads into REP entries (the gem's
/// `PhRepExtractor`): simple (`ph:wich` → REP(wich, which)), star
/// (`ph:prity*` → REP(prit, prett)) and arrow (`ph:hepi->happi` →
/// REP(hepi, happi)) forms.
fn ph_rep_entries(stem: &str, ph_tokens: &[&str]) -> Vec<RepEntry> {
    ph_tokens
        .iter()
        .filter_map(|token| {
            if let Some(pattern) = token.strip_suffix('*') {
                // Drop the trailing char of the pattern (already done by
                // strip_suffix) and of the stem.
                let pattern = pattern.strip_suffix(&pattern.chars().next_back()?.to_string())?;
                let stem_chars: Vec<char> = stem.chars().collect();
                if stem_chars.len() < 2 {
                    return None;
                }
                let replacement: String = stem_chars[..stem_chars.len() - 1].iter().collect();
                if pattern.is_empty() || replacement.is_empty() {
                    return None;
                }
                Some(RepEntry {
                    pattern: pattern.to_owned(),
                    replacement,
                })
            } else if let Some((from, to)) = token.split_once("->") {
                (!from.is_empty() && !to.is_empty()).then_some(RepEntry {
                    pattern: from.to_owned(),
                    replacement: to.to_owned(),
                })
            } else if token.is_empty() {
                None
            } else {
                Some(RepEntry {
                    pattern: (*token).to_owned(),
                    replacement: stem.to_owned(),
                })
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_line(line: &str) -> (String, BTreeSet<String>) {
        let (head, _) = split_stem_and_morph(line);
        let head = head.trim();
        if head.starts_with('/') {
            return (head.to_owned(), BTreeSet::new());
        }
        let (stem, flags) = match unescaped_slash_index(head) {
            Some(idx) => (head[..idx].to_owned(), Some(&head[idx + 1..])),
            None => (head.to_owned(), None),
        };
        let stem = stem.replace("\\/", "/");
        let flags = flags
            .filter(|s| !s.is_empty())
            .map(|s| parse_dic_flags(s, FlagFormat::Short, &HashMap::new()))
            .unwrap_or_default();
        (stem, flags)
    }

    #[test]
    fn parses_stem_and_flags() {
        assert_eq!(parse_line("foo").0, "foo");
        assert_eq!(
            parse_line("foo/AB").1,
            BTreeSet::from(["A".to_owned(), "B".to_owned()])
        );
        assert_eq!(parse_line("a\\/b").0, "a/b");
        // Only the character immediately before the slash counts.
        assert_eq!(parse_line("a\\\\/b").0, "a\\/b");
    }

    #[test]
    fn splits_morph_data() {
        assert_eq!(
            split_stem_and_morph("wich/GR ph:wich"),
            ("wich/GR", " ph:wich")
        );
        assert_eq!(split_stem_and_morph("foo\tbar:baz"), ("foo", "bar:baz"));
        assert_eq!(split_stem_and_morph("plain"), ("plain", ""));
    }

    #[test]
    fn leading_slash_is_the_word() {
        assert_eq!(parse_line("/AB").0, "/AB");
    }

    #[test]
    fn ph_tokens_become_rep_entries() {
        assert_eq!(
            ph_rep_entries("which", &["wich"]),
            vec![RepEntry {
                pattern: "wich".to_owned(),
                replacement: "which".to_owned()
            }]
        );
        assert_eq!(
            ph_rep_entries("pretty", &["prity*"]),
            vec![RepEntry {
                pattern: "prit".to_owned(),
                replacement: "prett".to_owned()
            }]
        );
        assert_eq!(
            ph_rep_entries("stem", &["hepi->happi"]),
            vec![RepEntry {
                pattern: "hepi".to_owned(),
                replacement: "happi".to_owned()
            }]
        );
    }
}
