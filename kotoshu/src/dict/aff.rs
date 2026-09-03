//! `.aff` file parsing, ported from the gem's `Readers::AffReader` +
//! `Readers::AffData` (which derive from Spylls).
//!
//! Behavior is reproduced exactly, quirks included: boolean directives are
//! `true` regardless of their payload, counted blocks (`REP`, `SFX`, …)
//! consume that many following lines, unknown directives are ignored (but
//! still consume nothing), `-` inside condition classes is a literal (the
//! gem escapes it, so ranges never form), and so on.

use std::collections::{BTreeSet, HashMap};

use super::casing::Casing;

/// Flag format declared by `FLAG` (defaults to short).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FlagFormat {
    /// One flag per character (`FLAG short`, the default).
    #[default]
    Short,
    /// Two characters per flag (`FLAG long`).
    Long,
    /// Decimal numbers separated by commas (`FLAG num`).
    Num,
    /// One flag per Unicode character (`FLAG UTF-8`).
    Utf8,
}

impl FlagFormat {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "short" => Ok(Self::Short),
            "long" => Ok(Self::Long),
            "num" => Ok(Self::Num),
            "UTF-8" => Ok(Self::Utf8),
            other => Err(format!("unknown flag format: {other}")),
        }
    }
}

/// Parse a flag string in the given format, with `AF` alias resolution.
///
/// Mirrors `AffReader#parse_flags`: aliases apply when the whole string is
/// an `AF` key (the aff side has no digits-only guard; the dic side does —
/// see [`super::dic`]).
pub fn parse_aff_flags(
    string: &str,
    format: FlagFormat,
    aliases: &HashMap<String, BTreeSet<String>>,
) -> BTreeSet<String> {
    if let Some(set) = aliases.get(string) {
        return set.clone();
    }
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
            flags.extend(split_digit_runs(string));
        }
    }
    flags
}

fn split_digit_runs(string: &str) -> Vec<String> {
    let mut runs = Vec::new();
    let mut current = String::new();
    for c in string.chars() {
        if c.is_ascii_digit() {
            current.push(c);
        } else if !current.is_empty() {
            runs.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        runs.push(current);
    }
    runs
}

/// Ruby `String#to_i`: leading integer prefix, 0 when there is none.
fn ruby_to_i(value: Option<&str>) -> i64 {
    let Some(value) = value else { return 0 };
    let mut chars = value.chars().peekable();
    while chars
        .peek()
        .is_some_and(|c| c.is_ascii_whitespace() || *c == '\0')
    {
        chars.next();
    }
    let mut digits = String::new();
    if let Some(sign) = chars.peek()
        && (*sign == '+' || *sign == '-')
    {
        digits.push(*sign);
        chars.next();
    }
    while chars.peek().is_some_and(|c| c.is_ascii_digit()) {
        digits.push(chars.next().unwrap());
    }
    if digits == "-" || digits == "+" {
        return 0;
    }
    digits.parse().unwrap_or(0)
}

/// One condition token of an affix condition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CondToken {
    /// `.` — any character.
    Any,
    /// A literal character.
    Char(char),
    /// A `[...]` class; `true` when negated (`[^...]`).
    Class(Vec<char>, bool),
}

/// A compiled affix condition, anchored at the stem's start (prefix rules)
/// or end (suffix rules).
///
/// The gem embeds the condition into a regex (`^cond` / `cond$`) and
/// searches; because every token consumes exactly one character, that is
/// equivalent to matching the first/last *n* characters positionally, which
/// is what this port implements. `-` is always a literal (the gem escapes
/// it), so character ranges never form.
#[derive(Debug, Clone)]
pub struct Condition {
    tokens: Vec<CondToken>,
    prefix: bool,
}

impl Condition {
    /// Compile a condition for a prefix (`:prefix`) or suffix (`:suffix`)
    /// affix rule.
    pub fn compile(condition: &str, prefix: bool) -> Self {
        let mut tokens = Vec::new();
        let mut chars = condition.chars().peekable();
        while let Some(&c) = chars.peek() {
            match c {
                '.' => {
                    chars.next();
                    tokens.push(CondToken::Any);
                }
                '[' => {
                    chars.next();
                    let mut negated = false;
                    if chars.peek() == Some(&'^') {
                        chars.next();
                        negated = true;
                    }
                    let mut class = Vec::new();
                    for c in chars.by_ref() {
                        if c == ']' {
                            break;
                        }
                        class.push(c);
                    }
                    tokens.push(CondToken::Class(class, negated));
                }
                other => {
                    chars.next();
                    tokens.push(CondToken::Char(other));
                }
            }
        }
        Self { tokens, prefix }
    }

    /// Whether `stem` satisfies the condition.
    pub fn matches(&self, stem: &str) -> bool {
        if self.tokens.len() > stem.chars().count() {
            return false;
        }
        let chars: Vec<char> = stem.chars().collect();
        let window: &[char] = if self.prefix {
            &chars[..self.tokens.len()]
        } else {
            &chars[chars.len() - self.tokens.len()..]
        };
        self.tokens
            .iter()
            .zip(window)
            .all(|(token, c)| token.matches(*c))
    }
}

impl CondToken {
    fn matches(&self, c: char) -> bool {
        match self {
            CondToken::Any => true,
            CondToken::Char(expected) => *expected == c,
            CondToken::Class(class, negated) => class.contains(&c) != *negated,
        }
    }
}

/// One prefix or suffix rule line.
#[derive(Debug, Clone)]
pub struct Affix {
    /// The rule's flag (dictionary entries listing this flag may use the
    /// rule).
    pub flag: String,
    /// Whether the rule participates in cross-product combinations
    /// (`Y` in the header line).
    pub crossproduct: bool,
    /// Characters stripped from the word before `add` is attached.
    pub strip: String,
    /// Characters attached to the stem (already `IGNORE`-stripped).
    pub add: String,
    /// Compiled condition, `None` when the rule has none.
    pub condition: Option<Condition>,
    /// Continuation flags carried by the rule (`able/CD` syntax).
    pub flags: BTreeSet<String>,
}

/// One `COMPOUNDRULE` item: a set of alternative flag strings plus a
/// quantifier.
#[derive(Debug, Clone)]
pub struct RuleItem {
    /// Accepted flag strings for this position (single characters, or
    /// parenthesized groups like `(1001)` for long/num flags).
    pub alts: BTreeSet<String>,
    /// How often the item may occur.
    pub quant: RuleQuant,
}

/// Quantifier of a [`RuleItem`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuleQuant {
    /// Exactly once.
    One,
    /// Zero or once (`?`).
    Optional,
    /// Zero or more (`*`).
    Star,
}

/// A `COMPOUNDRULE` pattern such as `A*B?CD` or `(nn)*(11)(tt)`.
///
/// Matching works like the gem's regex over the flag-combination string:
/// each item consumes one flag string from its alternatives (or none, for
/// `?`/`*`); `full_match` requires the whole string to be consumed,
/// `partial_match` allows stopping after any prefix of items.
#[derive(Debug, Clone)]
pub struct CompoundRule {
    items: Vec<RuleItem>,
    /// All flag strings appearing in the rule (for intersection with word
    /// flags).
    pub flags: BTreeSet<String>,
}

impl CompoundRule {
    /// Parse a rule's text.
    pub fn new(text: &str) -> Self {
        let mut items = Vec::new();
        let mut flags = BTreeSet::new();
        let mut chars = text.chars().peekable();
        while let Some(&c) = chars.peek() {
            if c == '(' {
                chars.next();
                let mut group = String::new();
                for gc in chars.by_ref() {
                    if gc == ')' {
                        break;
                    }
                    group.push(gc);
                }
                let quant = match chars.peek() {
                    Some('*') => {
                        chars.next();
                        RuleQuant::Star
                    }
                    Some('?') => {
                        chars.next();
                        RuleQuant::Optional
                    }
                    _ => RuleQuant::One,
                };
                // A parenthesized group is ONE alternative (possibly a
                // multi-character flag string, e.g. `(nn)` under FLAG
                // long/num) — the gem keeps the group content whole.
                let alt = group;
                flags.insert(alt.clone());
                items.push(RuleItem {
                    alts: BTreeSet::from([alt]),
                    quant,
                });
            } else if c != '*' && c != '?' {
                chars.next();
                let quant = match chars.peek() {
                    Some('*') => {
                        chars.next();
                        RuleQuant::Star
                    }
                    Some('?') => {
                        chars.next();
                        RuleQuant::Optional
                    }
                    _ => RuleQuant::One,
                };
                let alt = c.to_string();
                flags.insert(alt.clone());
                items.push(RuleItem {
                    alts: BTreeSet::from([alt]),
                    quant,
                });
            } else {
                chars.next();
            }
        }
        Self { items, flags }
    }

    /// Whether some combination of the relevant flags of every part
    /// satisfies the whole rule.
    pub fn full_match(&self, flag_sets: &[BTreeSet<String>]) -> bool {
        if flag_sets.is_empty() {
            return false;
        }
        let relevant: Vec<Vec<String>> = flag_sets
            .iter()
            .map(|set| {
                let mut intersection: Vec<String> =
                    set.intersection(&self.flags).cloned().collect();
                intersection.sort();
                intersection
            })
            .collect();
        if relevant.is_empty() || relevant.iter().any(|r| r.is_empty()) {
            return false;
        }
        let mut combination = Vec::with_capacity(relevant.len());
        full_product(&relevant, 0, &mut combination, &mut |combination| {
            self.matches(combination, true)
        })
    }

    /// Whether some combination matches a prefix of the rule (used to prune
    /// the compounds-by-rules recursion).
    pub fn partial_match(&self, flag_sets: &[BTreeSet<String>]) -> bool {
        if flag_sets.is_empty() {
            return false;
        }
        let relevant: Vec<Vec<String>> = flag_sets
            .iter()
            .map(|set| {
                let mut intersection: Vec<String> =
                    set.intersection(&self.flags).cloned().collect();
                intersection.sort();
                intersection
            })
            .collect();
        if relevant.is_empty() || relevant.iter().any(|r| r.is_empty()) {
            return false;
        }
        let mut combination = Vec::with_capacity(relevant.len());
        full_product(&relevant, 0, &mut combination, &mut |combination| {
            self.matches(combination, false)
        })
    }

    /// Backtracking matcher over the concatenated flag string.
    fn matches(&self, combination: &[String], full: bool) -> bool {
        let tokens: Vec<&str> = combination.iter().map(String::as_str).collect();
        self.match_items(&tokens, 0, 0, full)
    }

    fn match_items(&self, tokens: &[&str], item: usize, pos: usize, full: bool) -> bool {
        // Partial match: the gem makes every trailing item optional, so
        // any point where the string is fully consumed is a match.
        if pos == tokens.len() && !full {
            return true;
        }
        if item == self.items.len() {
            return pos == tokens.len();
        }
        let rule_item = &self.items[item];
        // Zero occurrences.
        if rule_item.quant != RuleQuant::One && self.match_items(tokens, item + 1, pos, full) {
            return true;
        }
        // One occurrence.
        if pos < tokens.len() && rule_item.alts.contains(tokens[pos]) {
            match rule_item.quant {
                RuleQuant::Star => {
                    if self.match_items(tokens, item + 1, pos + 1, full) {
                        return true;
                    }
                    // `*` may consume several in a row.
                    if self.match_items(tokens, item, pos + 1, full) {
                        return true;
                    }
                }
                _ => {
                    if self.match_items(tokens, item + 1, pos + 1, full) {
                        return true;
                    }
                }
            }
        }
        false
    }
}

/// Enumerate the cartesian product of `sets`, calling `f` on each
/// combination; short-circuits on the first `true`.
fn full_product(
    sets: &[Vec<String>],
    depth: usize,
    combination: &mut Vec<String>,
    f: &mut dyn FnMut(&[String]) -> bool,
) -> bool {
    if depth == sets.len() {
        return f(combination);
    }
    for value in &sets[depth] {
        combination.push(value.clone());
        if full_product(sets, depth + 1, combination, f) {
            combination.pop();
            return true;
        }
        combination.pop();
    }
    false
}

/// A `CHECKCOMPOUNDPATTERN` entry.
#[derive(Debug, Clone)]
pub struct CompoundPattern {
    /// Left stem suffix that triggers the pattern (`0` means empty).
    pub left_stem: String,
    /// Optional flag the left part must carry.
    pub left_flag: Option<String>,
    /// Whether the left part must NOT be a bare stem (`0/...` form).
    pub left_no_affix: bool,
    /// Right stem prefix that triggers the pattern.
    pub right_stem: String,
    /// Optional flag the right part must carry.
    pub right_flag: Option<String>,
    /// Whether the right part must NOT be a bare stem.
    pub right_no_affix: bool,
}

impl CompoundPattern {
    /// Parse a `CHECKCOMPOUNDPATTERN` row (left, right, optional
    /// replacement).
    pub fn new(left: &str, right: &str, _replacement: Option<&str>) -> Self {
        let (left_stem, left_flag) = partition_flag(left);
        let (right_stem, right_flag) = partition_flag(right);
        Self {
            left_no_affix: left_stem.is_empty() && left.starts_with('0'),
            left_stem,
            left_flag,
            right_no_affix: right_stem.is_empty() && right.starts_with('0'),
            right_stem,
            right_flag,
        }
    }

    /// Whether the pattern rejects this compound pair (stems are the
    /// dictionary stems, flags the combined part flags, `is_base` whether
    /// the part carries no affixes).
    pub fn matches(
        &self,
        left_stem: &str,
        left_flags: &BTreeSet<String>,
        left_is_base: bool,
        right_stem: &str,
        right_flags: &BTreeSet<String>,
        right_is_base: bool,
    ) -> bool {
        if !left_stem.ends_with(self.left_stem.as_str()) {
            return false;
        }
        if !right_stem.starts_with(self.right_stem.as_str()) {
            return false;
        }
        if self.left_no_affix && left_is_base {
            return false;
        }
        if self.right_no_affix && right_is_base {
            return false;
        }
        if let Some(flag) = &self.left_flag
            && !left_flags.contains(flag)
        {
            return false;
        }
        if let Some(flag) = &self.right_flag
            && !right_flags.contains(flag)
        {
            return false;
        }
        true
    }
}

/// Ruby `String#partition('/')` semantics: `nil` flag when there is no
/// slash, `""`-stem special-cased for `0`.
fn partition_flag(part: &str) -> (String, Option<String>) {
    match part.split_once('/') {
        None => (
            if part == "0" {
                String::new()
            } else {
                part.to_owned()
            },
            None,
        ),
        Some((stem, flag)) => (
            if stem == "0" {
                String::new()
            } else {
                stem.to_owned()
            },
            Some(flag.to_owned()),
        ),
    }
}

/// A `BREAK` pattern: literal text, optionally anchored with `^`/`$`.
#[derive(Debug, Clone)]
pub struct BreakPattern {
    /// The pattern without its anchors.
    pub pattern: String,
    /// Whether the pattern is anchored at the start.
    pub anchor_start: bool,
    /// Whether the pattern is anchored at the end.
    pub anchor_end: bool,
}

impl BreakPattern {
    /// Parse a `BREAK` pattern line.
    pub fn new(pattern: &str) -> Self {
        let anchor_start = pattern.starts_with('^');
        let anchor_end = pattern.ends_with('$');
        let trimmed = pattern
            .trim_start_matches('^')
            .trim_end_matches('$')
            .to_owned();
        Self {
            pattern: trimmed,
            anchor_start,
            anchor_end,
        }
    }
}

/// One `ICONV`/`OCONV` table row.
#[derive(Debug, Clone)]
pub struct ConvRow {
    /// Pattern with `_` removed (the gem strips underscores to form the
    /// search key).
    pub search: String,
    /// Anchored at word start (leading `_`).
    pub anchored_start: bool,
    /// Anchored at word end (trailing `_`).
    pub anchored_end: bool,
    /// Replacement, with `_` turned into spaces.
    pub replacement: String,
}

/// An `ICONV`/`OCONV` conversion table, sorted by search length (stable, so
/// declaration order breaks ties — mirroring the gem's index-based
/// tie-break).
#[derive(Debug, Clone, Default)]
pub struct ConvTable {
    rows: Vec<ConvRow>,
}

impl ConvTable {
    /// Compile the raw `[pattern, replacement]` pairs.
    pub fn new(pairs: &[(String, String)]) -> Self {
        let mut rows: Vec<ConvRow> = pairs
            .iter()
            .map(|(pat1, pat2)| {
                let anchored_start = pat1.starts_with('_');
                let anchored_end = pat1.ends_with('_');
                ConvRow {
                    search: pat1.replace('_', ""),
                    anchored_start,
                    anchored_end,
                    replacement: pat2.replace('_', " "),
                }
            })
            .collect();
        rows.sort_by_key(|row| row.search.chars().count());
        Self { rows }
    }

    /// Whether the row matches `word` at byte offset `pos`.
    fn row_matches_at(row: &ConvRow, word: &str, pos: usize) -> bool {
        if row.anchored_start && pos != 0 {
            return false;
        }
        let end = pos + row.search.len();
        if row.anchored_end && end != word.len() {
            return false;
        }
        word.is_char_boundary(end) && word[pos..].starts_with(&row.search)
    }

    /// Apply the conversions left to right (longest search wins at each
    /// position; earlier rows break length ties).
    pub fn apply(&self, word: &str) -> String {
        let mut result = String::with_capacity(word.len());
        let mut pos = 0;
        while pos < word.len() {
            let mut best: Option<(usize, usize)> = None; // (search len, row idx)
            for (idx, row) in self.rows.iter().enumerate() {
                if !Self::row_matches_at(row, word, pos) {
                    continue;
                }
                let len = row.search.chars().count();
                match best {
                    Some((best_len, _)) if best_len >= len => {}
                    _ => best = Some((len, idx)),
                }
            }
            if let Some((_, idx)) = best {
                result.push_str(&self.rows[idx].replacement);
                pos += self.rows[idx].search.len();
            } else {
                let ch = word[pos..].chars().next().unwrap();
                result.push(ch);
                pos += ch.len_utf8();
            }
        }
        result
    }
}

/// A `REP` entry (typical misspelling pair), literal in this port (the
/// corpus contains no regex metacharacters in REP patterns).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepEntry {
    /// Misspelling pattern.
    pub pattern: String,
    /// Correction (may contain `_`, meaning a space when applied).
    pub replacement: String,
}

/// Everything the lookup path needs from the `.aff` file, mirroring the
/// gem's `LookupBuilder#build_aff_structure`.
#[derive(Debug)]
pub struct Aff {
    /// Language-specific casing.
    pub casing: Casing,
    /// `IGNORE` characters (stripped from stems, affixes and inputs).
    pub ignore: Vec<char>,
    /// Suffix rules indexed by the last character of their `add` (`""`
    /// bucket for empty adds), values indexing [`Aff::suffixes`].
    pub suffixes_index: HashMap<String, Vec<usize>>,
    /// All suffix rules.
    pub suffixes: Vec<Affix>,
    /// Prefix rules indexed by the first character of their `add`.
    pub prefixes_index: HashMap<String, Vec<usize>>,
    /// All prefix rules.
    pub prefixes: Vec<Affix>,
    /// `COMPOUNDMIN` (Hunspell default 3 applied at use site).
    pub compoundmin: Option<i64>,
    /// `COMPOUNDWORDMAX`.
    pub compoundwordmax: Option<i64>,
    /// `COMPOUNDBEGIN` flag.
    pub compoundbegin: Option<String>,
    /// `COMPOUNDMIDDLE` flag.
    pub compoundmiddle: Option<String>,
    /// `COMPOUNDEND` flag.
    pub compoundend: Option<String>,
    /// `COMPOUNDFLAG` flag.
    pub compoundflag: Option<String>,
    /// `COMPOUNDPERMITFLAG`.
    pub compoundpermitflag: Option<String>,
    /// `COMPOUNDFORBIDFLAG`.
    pub compoundforbidflag: Option<String>,
    /// `COMPOUNDRULE` patterns.
    pub compoundrules: Vec<CompoundRule>,
    /// `ONLYINCOMPOUND` flag.
    pub onlyincompound: Option<String>,
    /// `COMPLEXPREFIXES` (two prefixes may stack).
    pub complexprefixes: bool,
    /// `FORCEUCASE` flag.
    pub forceucase: Option<String>,
    /// `FORBIDDENWORD` flag.
    pub forbiddenword: Option<String>,
    /// `NOSUGGEST` flag.
    pub nosuggest: Option<String>,
    /// `KEEPCASE` flag.
    pub keepcase: Option<String>,
    /// `NEEDAFFIX` flag.
    pub needaffix: Option<String>,
    /// `CIRCUMFIX` flag.
    pub circumfix: Option<String>,
    /// `WARN` flag. Parsed for fidelity with the gem's aff data; the
    /// lookup path never consults it (suggest-side concern, P2).
    #[allow(dead_code)]
    pub warn: Option<String>,
    /// `CHECKCOMPOUNDCASE`.
    pub checkcompoundcase: bool,
    /// `CHECKCOMPOUNDDUP`.
    pub checkcompounddup: bool,
    /// `CHECKCOMPOUNDREP`.
    pub checkcompoundrep: bool,
    /// `CHECKCOMPOUNDTRIPLE`.
    pub checkcompoundtriple: bool,
    /// `CHECKCOMPOUNDPATTERN` entries.
    pub checkcompoundpatterns: Vec<CompoundPattern>,
    /// `SIMPLIFIEDTRIPLE`.
    pub simplifiedtriple: bool,
    /// `BREAK` patterns (defaults to `-`, `^-`, `-$` when absent).
    pub break_patterns: Vec<BreakPattern>,
    /// `ICONV` table.
    pub iconv: Option<ConvTable>,
    /// `REP` table (plus dictionary `ph:`-derived entries).
    pub rep: Vec<RepEntry>,
    /// `CHECKSHARPS`.
    pub checksharps: bool,
    /// Final `FLAG` format (needed to interpret the `.dic` file).
    pub flag_format: FlagFormat,
    /// `AF` aliases (needed to interpret the `.dic` file).
    pub af_aliases: HashMap<String, BTreeSet<String>>,
}

/// Directive parse state accumulated line by line, mirroring the gem's
/// `data` hash + `@flag_format`/`@flag_synonyms` timing.
#[derive(Default)]
struct RawAff {
    sfx: HashMap<String, Vec<AffixLine>>,
    pfx: HashMap<String, Vec<AffixLine>>,
    lang: Option<String>,
    flag_format: FlagFormat,
    af: HashMap<String, BTreeSet<String>>,
    keepcase: Option<String>,
    circumfix: Option<String>,
    needaffix: Option<String>,
    forbiddenword: Option<String>,
    nosuggest: Option<String>,
    warn: Option<String>,
    compoundflag: Option<String>,
    compoundbegin: Option<String>,
    compoundmiddle: Option<String>,
    compoundend: Option<String>,
    onlyincompound: Option<String>,
    compoundpermitflag: Option<String>,
    compoundforbidflag: Option<String>,
    forceucase: Option<String>,
    compoundmin: Option<i64>,
    compoundwordmax: Option<i64>,
    complexprefixes: bool,
    checksharps: bool,
    checkcompoundcase: bool,
    checkcompounddup: bool,
    checkcompoundrep: bool,
    checkcompoundtriple: bool,
    simplifiedtriple: bool,
    ignore: Option<String>,
    break_patterns: Option<Vec<BreakPattern>>,
    compoundrules: Vec<CompoundRule>,
    iconv: Option<ConvTable>,
    rep: Vec<RepEntry>,
    checkcompoundpatterns: Vec<CompoundPattern>,
}

/// An affix rule as read from the file (condition not yet compiled).
#[derive(Debug, Clone)]
struct AffixLine {
    flag: String,
    crossproduct: bool,
    strip: String,
    add: String,
    condition: String,
    flags: BTreeSet<String>,
}

/// Parse the (already decoded, stripped, non-empty) `.aff` lines.
pub fn parse_lines(lines: Vec<String>) -> Result<Aff, String> {
    let mut raw = RawAff {
        flag_format: FlagFormat::Short,
        ..RawAff::default()
    };
    let mut lines = lines.into_iter();
    while let Some(line) = lines.next() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        let Some(&name) = parts.first() else {
            continue;
        };
        if name.is_empty() || !name.chars().all(|c| c.is_ascii_uppercase()) {
            continue;
        }
        let name = match name {
            "PSEUDOROOT" => "NEEDAFFIX",
            "COMPOUNDLAST" => "COMPOUNDEND",
            other => other,
        };
        let values = &parts[1..];

        match name {
            "SET" | "KEY" | "TRY" | "WORDCHARS" => {}
            "FLAG" => {
                if let Some(value) = values.first() {
                    raw.flag_format = FlagFormat::parse(value)?;
                }
            }
            "LANG" => raw.lang = values.first().copied().map(String::from),
            "MAXDIFF" | "MAXNGRAMSUGS" | "MAXCPDSUGS" => {}
            "COMPOUNDMIN" => raw.compoundmin = values.first().map(|v| ruby_to_i(Some(v))),
            "COMPOUNDWORDMAX" => raw.compoundwordmax = values.first().map(|v| ruby_to_i(Some(v))),
            "COMPLEXPREFIXES"
            | "FULLSTRIP"
            | "NOSPLITSUGS"
            | "CHECKSHARPS"
            | "CHECKCOMPOUNDCASE"
            | "CHECKCOMPOUNDDUP"
            | "CHECKCOMPOUNDREP"
            | "CHECKCOMPOUNDTRIPLE"
            | "SIMPLIFIEDTRIPLE"
            | "ONLYMAXDIFF"
            | "COMPOUNDMORESUFFIXES" => {
                // The gem stores `true` regardless of the payload.
                match name {
                    "COMPLEXPREFIXES" => raw.complexprefixes = true,
                    "CHECKSHARPS" => raw.checksharps = true,
                    "CHECKCOMPOUNDCASE" => raw.checkcompoundcase = true,
                    "CHECKCOMPOUNDDUP" => raw.checkcompounddup = true,
                    "CHECKCOMPOUNDREP" => raw.checkcompoundrep = true,
                    "CHECKCOMPOUNDTRIPLE" => raw.checkcompoundtriple = true,
                    "SIMPLIFIEDTRIPLE" => raw.simplifiedtriple = true,
                    _ => {}
                }
            }
            "NOSUGGEST" => raw.nosuggest = parse_flag(values.first().copied(), &raw),
            "KEEPCASE" => raw.keepcase = parse_flag(values.first().copied(), &raw),
            "CIRCUMFIX" => raw.circumfix = parse_flag(values.first().copied(), &raw),
            "NEEDAFFIX" => raw.needaffix = parse_flag(values.first().copied(), &raw),
            "FORBIDDENWORD" => raw.forbiddenword = parse_flag(values.first().copied(), &raw),
            "WARN" => raw.warn = parse_flag(values.first().copied(), &raw),
            "COMPOUNDFLAG" => raw.compoundflag = parse_flag(values.first().copied(), &raw),
            "COMPOUNDBEGIN" => raw.compoundbegin = parse_flag(values.first().copied(), &raw),
            "COMPOUNDMIDDLE" => raw.compoundmiddle = parse_flag(values.first().copied(), &raw),
            "COMPOUNDEND" => raw.compoundend = parse_flag(values.first().copied(), &raw),
            "ONLYINCOMPOUND" => raw.onlyincompound = parse_flag(values.first().copied(), &raw),
            "COMPOUNDPERMITFLAG" => {
                raw.compoundpermitflag = parse_flag(values.first().copied(), &raw)
            }
            "COMPOUNDFORBIDFLAG" => {
                raw.compoundforbidflag = parse_flag(values.first().copied(), &raw)
            }
            "FORCEUCASE" => raw.forceucase = parse_flag(values.first().copied(), &raw),
            "SUBSTANDARD" | "SYLLABLENUM" | "COMPOUNDROOT" => {}
            "IGNORE" => raw.ignore = Some(values.first().copied().unwrap_or("").to_owned()),
            "BREAK" => {
                let rows = read_array(&mut lines, ruby_to_i(values.first().copied()) as usize)?;
                raw.break_patterns = Some(
                    rows.iter()
                        .filter_map(|row| row.first())
                        .map(|pattern| BreakPattern::new(pattern))
                        .collect(),
                );
            }
            "COMPOUNDRULE" => {
                let rows = read_array(&mut lines, ruby_to_i(values.first().copied()) as usize)?;
                raw.compoundrules = rows
                    .iter()
                    .filter_map(|row| row.first())
                    .map(|text| CompoundRule::new(text))
                    .collect();
            }
            "ICONV" | "OCONV" => {
                let rows = read_array(&mut lines, ruby_to_i(values.first().copied()) as usize)?;
                let pairs: Vec<(String, String)> = rows
                    .iter()
                    .map(|row| {
                        (
                            row.first().cloned().unwrap_or_default(),
                            row.get(1).cloned().unwrap_or_default(),
                        )
                    })
                    .collect();
                if name == "ICONV" {
                    raw.iconv = Some(ConvTable::new(&pairs));
                }
            }
            "REP" => {
                let rows = read_array(&mut lines, ruby_to_i(values.first().copied()) as usize)?;
                raw.rep = rows
                    .iter()
                    .map(|row| RepEntry {
                        pattern: row.first().cloned().unwrap_or_default(),
                        replacement: row.get(1).cloned().unwrap_or_default(),
                    })
                    .collect();
            }
            "MAP" | "PHONE" | "AM" => {
                let _ = read_array(&mut lines, ruby_to_i(values.first().copied()) as usize)?;
            }
            "SFX" | "PFX" => {
                let flag = values.first().copied().unwrap_or_default().to_owned();
                let crossproduct = values.get(1) == Some(&"Y");
                let count = ruby_to_i(values.get(2).copied()) as usize;
                let rows = read_array(&mut lines, count)?;
                let prefix = name == "PFX";
                let affixes: Vec<AffixLine> = rows
                    .iter()
                    .map(|row| {
                        // Row shape after read_array dropped the directive:
                        // [flag,] strip, add, condition[, morph…].
                        let strip = match row.get(1).map(String::as_str) {
                            None | Some("0") => String::new(),
                            Some(s) => s.to_owned(),
                        };
                        let add = row.get(2).cloned().unwrap_or_default();
                        let condition = row.get(3).cloned().unwrap_or_else(|| ".".to_owned());
                        let (add, flags) = match add.rsplit_once('/') {
                            Some((add_str, flags_str)) => (
                                if add_str == "0" {
                                    String::new()
                                } else {
                                    add_str.to_owned()
                                },
                                if flags_str.is_empty() {
                                    BTreeSet::new()
                                } else {
                                    parse_aff_flags(flags_str, raw.flag_format, &raw.af)
                                },
                            ),
                            None => (
                                if add == "0" { String::new() } else { add },
                                BTreeSet::new(),
                            ),
                        };
                        AffixLine {
                            flag: flag.clone(),
                            crossproduct,
                            strip,
                            add,
                            condition,
                            flags,
                        }
                    })
                    .collect();
                if prefix {
                    raw.pfx.insert(flag, affixes);
                } else {
                    raw.sfx.insert(flag, affixes);
                }
            }
            "CHECKCOMPOUNDPATTERN" => {
                let rows = read_array(&mut lines, ruby_to_i(values.first().copied()) as usize)?;
                raw.checkcompoundpatterns = rows
                    .iter()
                    .map(|row| {
                        CompoundPattern::new(
                            row.first().map(String::as_str).unwrap_or(""),
                            row.get(1).map(String::as_str).unwrap_or(""),
                            row.get(2).map(String::as_str),
                        )
                    })
                    .collect();
            }
            "AF" => {
                let rows = read_array(&mut lines, ruby_to_i(values.first().copied()) as usize)?;
                let mut aliases = HashMap::new();
                for (idx, row) in rows.iter().enumerate() {
                    let flags: BTreeSet<String> = row
                        .first()
                        .map(|first| first.chars().map(|c| c.to_string()).collect())
                        .unwrap_or_default();
                    aliases.insert((idx + 1).to_string(), flags);
                }
                raw.af = aliases;
            }
            "COMPOUNDSYLLABLE" => {
                // Parsed by the gem as [count, flag]; unused by lookup.
            }
            _ => {
                // Unknown directive: ignored.
            }
        }
    }
    build_aff(raw)
}

fn parse_flag(value: Option<&str>, raw: &RawAff) -> Option<String> {
    let value = value?;
    let flags = parse_aff_flags(value, raw.flag_format, &raw.af);
    flags.into_iter().next()
}

/// Consume `count` lines, dropping each line's directive word (the gem's
/// `read_array`).
fn read_array(
    lines: &mut dyn Iterator<Item = String>,
    count: usize,
) -> Result<Vec<Vec<String>>, String> {
    let mut rows = Vec::with_capacity(count);
    for _ in 0..count {
        let Some(line) = lines.next() else {
            return Err("counted directive block truncated".to_owned());
        };
        // The gem keeps only rows that had a value after the directive.
        let all: Vec<&str> = line.split_whitespace().collect();
        if all.len() > 1 {
            rows.push(all[1..].iter().map(|s| (*s).to_owned()).collect());
        }
    }
    Ok(rows)
}

/// Turn the parsed directives into the lookup-ready [`Aff`] (the gem's
/// `build_aff_structure`).
fn build_aff(raw: RawAff) -> Result<Aff, String> {
    let casing = Casing::select(raw.lang.as_deref(), raw.checksharps);
    let ignore: Vec<char> = raw.ignore.map(|s| s.chars().collect()).unwrap_or_default();

    let mut aff = Aff {
        casing,
        ignore: ignore.clone(),
        suffixes_index: HashMap::new(),
        suffixes: Vec::new(),
        prefixes_index: HashMap::new(),
        prefixes: Vec::new(),
        compoundmin: raw.compoundmin,
        compoundwordmax: raw.compoundwordmax,
        compoundbegin: raw.compoundbegin,
        compoundmiddle: raw.compoundmiddle,
        compoundend: raw.compoundend,
        compoundflag: raw.compoundflag,
        compoundpermitflag: raw.compoundpermitflag,
        compoundforbidflag: raw.compoundforbidflag,
        compoundrules: raw.compoundrules,
        onlyincompound: raw.onlyincompound,
        complexprefixes: raw.complexprefixes,
        forceucase: raw.forceucase,
        forbiddenword: raw.forbiddenword,
        nosuggest: raw.nosuggest,
        keepcase: raw.keepcase,
        needaffix: raw.needaffix,
        circumfix: raw.circumfix,
        warn: raw.warn,
        checkcompoundcase: raw.checkcompoundcase,
        checkcompounddup: raw.checkcompounddup,
        checkcompoundrep: raw.checkcompoundrep,
        checkcompoundtriple: raw.checkcompoundtriple,
        checkcompoundpatterns: raw.checkcompoundpatterns,
        simplifiedtriple: raw.simplifiedtriple,
        break_patterns: raw.break_patterns.unwrap_or_else(|| {
            vec![
                BreakPattern::new("-"),
                BreakPattern::new("^-"),
                BreakPattern::new("-$"),
            ]
        }),
        iconv: raw.iconv,
        rep: raw.rep,
        checksharps: raw.checksharps,
        flag_format: raw.flag_format,
        af_aliases: raw.af,
    };

    for affixes in raw.sfx.into_values() {
        for line in affixes {
            let key = index_key_suffix(&line.add, &ignore);
            let idx = aff.suffixes.len();
            aff.suffixes.push(Affix {
                flag: line.flag,
                crossproduct: line.crossproduct,
                strip: line.strip,
                add: strip_ignore(&line.add, &ignore),
                condition: compile_condition(&line.condition, false),
                flags: line.flags,
            });
            aff.suffixes_index.entry(key).or_default().push(idx);
        }
    }
    for affixes in raw.pfx.into_values() {
        for line in affixes {
            let stripped_add = strip_ignore(&line.add, &ignore);
            let key = stripped_add
                .chars()
                .next()
                .map(|c| c.to_string())
                .unwrap_or_default();
            let idx = aff.prefixes.len();
            aff.prefixes.push(Affix {
                flag: line.flag,
                crossproduct: line.crossproduct,
                strip: line.strip,
                add: stripped_add,
                condition: compile_condition(&line.condition, true),
                flags: line.flags,
            });
            aff.prefixes_index.entry(key).or_default().push(idx);
        }
    }
    Ok(aff)
}

fn compile_condition(condition: &str, prefix: bool) -> Option<Condition> {
    if condition.is_empty() {
        None
    } else {
        Some(Condition::compile(condition, prefix))
    }
}

fn index_key_suffix(add: &str, ignore: &[char]) -> String {
    let stripped = strip_ignore(add, ignore);
    stripped
        .chars()
        .next_back()
        .map(|c| c.to_string())
        .unwrap_or_default()
}

/// Remove `IGNORE` characters from a string.
pub fn strip_ignore(text: &str, ignore: &[char]) -> String {
    if ignore.is_empty() {
        return text.to_owned();
    }
    text.chars().filter(|c| !ignore.contains(c)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conditions_match_stem_edges() {
        // The gem's regex is `[^y]$` searched in the stem: it matches when
        // the stem does NOT end with "y" (its docstring example is stale).
        let suffix = Condition::compile("[^y]", false);
        assert!(!suffix.matches("try"));
        assert!(suffix.matches("trj"));
        let prefix = Condition::compile("ij", true);
        assert!(prefix.matches("ijs"));
        assert!(!prefix.matches("jis"));
        assert!(!Condition::compile("[aeiou]y", false).matches("try"));
        assert!(Condition::compile(".", false).matches("anything"));
    }

    #[test]
    fn dash_is_literal_in_classes() {
        // The gem escapes '-', so [a-z] is the set {a, -, z}.
        let condition = Condition::compile("[a-z]", false);
        assert!(condition.matches("x-"));
        assert!(!condition.matches("xy"));
    }

    #[test]
    fn compound_rules_match_flag_combinations() {
        let rule = CompoundRule::new("A*B?C?");
        let a = BTreeSet::from(["A".to_owned()]);
        let b = BTreeSet::from(["B".to_owned()]);
        assert!(rule.full_match(std::slice::from_ref(&a)));
        assert!(rule.full_match(&[a.clone(), a.clone(), b.clone()]));
        // `A*B?C?` (regex \AA*B?C?\z) accepts a lone B.
        assert!(rule.full_match(std::slice::from_ref(&b)));
        assert!(rule.partial_match(std::slice::from_ref(&a)));
        assert!(rule.partial_match(std::slice::from_ref(&b)));

        let strict = CompoundRule::new("AB");
        assert!(!strict.full_match(std::slice::from_ref(&b)));
        assert!(strict.full_match(&[a.clone(), b.clone()]));
        assert!(strict.partial_match(std::slice::from_ref(&a)));
        assert!(!strict.partial_match(std::slice::from_ref(&b)));

        let grouped = CompoundRule::new("(nn)*(11)(tt)");
        let n1 = BTreeSet::from(["nn".to_owned()]);
        let t1 = BTreeSet::from(["11".to_owned(), "xx".to_owned()]);
        let t2 = BTreeSet::from(["tt".to_owned()]);
        assert!(grouped.full_match(&[n1.clone(), t1.clone(), t2.clone()]));
    }

    #[test]
    fn conv_table_prefers_longest_match() {
        let table = ConvTable::new(&[
            ("Da".to_owned(), "DA".to_owned()),
            ("Gag".to_owned(), "GAG".to_owned()),
            ("Gagg".to_owned(), "GAGG".to_owned()),
        ]);
        assert_eq!(table.apply("Gagg"), "GAGG");
        assert_eq!(table.apply("Da"), "DA");
    }
}
