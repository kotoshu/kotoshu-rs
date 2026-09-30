//! The sentence-level checker: segment text, tokenize with offsets,
//! tag, run every rule, emit character-span errors. Rust twin of
//! Kotoshu::Grammar::Checker.

use super::pattern_rule::{MatcherKind, PatternRule};
use super::pos_tagger::{tag, tokenize_with_offsets, Pos};

#[derive(Debug, Clone, PartialEq)]
pub struct TokenView {
    pub word: String,
    pub pos: Pos,
    pub start_offset: usize,
    pub end_offset: usize,
    pub index: usize,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct GrammarError {
    pub rule_id: String,
    pub start_offset: usize,
    pub end_offset: usize,
    pub message: String,
    pub suggestions: Vec<String>,
}

pub struct Checker {
    pub rules: Vec<PatternRule>,
}

fn sentence_spans(text: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let bytes = text.as_bytes();
    let mut start = 0;
    for (i, b) in bytes.iter().enumerate() {
        if *b == b'.' || *b == b'!' || *b == b'?' {
            spans.push((start, i + 1));
            start = i + 1;
        }
    }
    if start < text.len() {
        spans.push((start, text.len()));
    }
    spans
}

impl Checker {
    pub fn new(rules: Vec<PatternRule>) -> Checker {
        Checker { rules }
    }

    pub fn check(&self, text: &str) -> Vec<GrammarError> {
        let mut errors = Vec::new();
        for (s, e) in sentence_spans(text) {
            let sentence = &text[s..e];
            let words = tokenize_with_offsets(sentence, s);
            if words.is_empty() {
                continue;
            }
            let mut tokens: Vec<TokenView> = Vec::with_capacity(words.len());
            let poses = tag(
                &words
                    .iter()
                    .map(|(w, _, _)| w.clone())
                    .collect::<Vec<_>>(),
            );
            for (i, ((word, start, end), pos)) in words.iter().zip(poses).enumerate() {
                tokens.push(TokenView {
                    word: word.clone(),
                    pos,
                    start_offset: *start,
                    end_offset: *end,
                    index: i,
                });
            }
            for rule in &self.rules {
                if !rule.enabled {
                    continue;
                }
                if let MatcherKind::PosSequence = rule.matcher {
                    for (start_index, end_index) in seq_match_spans(rule, &tokens) {
                        let window = &tokens[start_index..=end_index];
                        errors.push(GrammarError {
                            rule_id: rule.id.clone(),
                            start_offset: window.first().map(|t| t.start_offset).unwrap_or(s),
                            end_offset: window.last().map(|t| t.end_offset).unwrap_or(e),
                            message: rule.message.clone(),
                            suggestions: rule.render_suggestions(window),
                        });
                    }
                } else {
                    let hits = rule.word_list_hits(&tokens);
                    for (start_index, end_index) in hits {
                        let window = &tokens[start_index..=end_index];
                        errors.push(GrammarError {
                            rule_id: rule.id.clone(),
                            start_offset: window.first().map(|t| t.start_offset).unwrap_or(s),
                            end_offset: window.last().map(|t| t.end_offset).unwrap_or(e),
                            message: rule.message.clone(),
                            suggestions: rule.render_suggestions(window),
                        });
                    }
                }
            }
        }
        errors
    }
}

/// WordList hits exposed for the checker (the rule method is private
/// to the matcher path).
impl PatternRule {
    pub fn word_list_hits(&self, tokens: &[TokenView]) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for parts in &self.words {
            if parts.is_empty() || tokens.len() < parts.len() {
                continue;
            }
            for start in 0..=(tokens.len() - parts.len()) {
                let window = &tokens[start..start + parts.len()];
                let hit = window.iter().zip(parts).all(|(t, p)| {
                    if self.case_sensitive_words {
                        t.word == *p
                    } else {
                        t.word.eq_ignore_ascii_case(p)
                    }
                });
                if hit {
                    out.push((start, start + parts.len() - 1));
                }
            }
        }
        out
    }
}

/// Find every match; returns (first_token_index, last_token_index).
pub fn seq_match_spans(rule: &PatternRule, tokens: &[TokenView]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    if rule.pattern.is_empty() || rule.pattern.len() > 12 || tokens.is_empty() {
        return out;
    }
    for start in 0..tokens.len() {
        if let Some(consumed) = seq_match(rule, &rule.pattern, 0, start, tokens, &[]) {
            if !consumed.is_empty() {
                out.push((*consumed.first().unwrap(), *consumed.last().unwrap()));
            }
        }
    }
    out
}

/// Entry used by PatternRule's pos_sequence path (the Ruby side calls
/// the same matcher through PatternRule#check).
pub fn seq_match_all(rule: &PatternRule, tokens: &[TokenView]) -> Vec<String> {
    seq_match_spans(rule, tokens)
        .into_iter()
        .map(|(f, l)| {
            tokens[f..=l]
                .iter()
                .map(|t| t.word.clone())
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect()
}

fn token_matches(constraint: &super::pattern_rule::Constraint, token: &TokenView) -> bool {
    if let Some(words) = &constraint.exception {
        if words.iter().any(|w| w.eq_ignore_ascii_case(&token.word)) {
            return false;
        }
    }
    if let Some(pos) = &constraint.exception_pos {
        if pos.iter().any(|p| token.pos.matches(p)) {
            return false;
        }
    }

    let mut matched = true;
    if let Some(pos_values) = &constraint.pos {
        if !pos_values.iter().any(|p| token.pos.matches(p)) {
            matched = false;
        }
    }
    if matched {
        if let Some(words) = &constraint.word {
            let hit = words.iter().any(|w| {
                if constraint.case_sensitive {
                    *w == token.word
                } else {
                    w.eq_ignore_ascii_case(&token.word)
                }
            });
            if !hit {
                matched = false;
            }
        }
    }
    if matched {
        if let Some(regex) = &constraint.regex {
            if !regex_token_match(regex, &token.word, constraint.case_sensitive) {
                matched = false;
            }
        }
    }
    if constraint.negated {
        !matched
    } else {
        matched
    }
}

/// Whole-token regex with support for the constructs the rule files
/// actually use: literals, `[class]` with `*`/`?`, and `.*`.
fn regex_token_match(pattern: &str, word: &str, case_sensitive: bool) -> bool {
    let w = if case_sensitive {
        word.to_string()
    } else {
        word.to_lowercase()
    };
    let chars: Vec<char> = w.chars().collect();
    for alt in pattern.split('|') {
        let trimmed = alt.trim().trim_start_matches("(?i)");
        let trimmed = trimmed
            .trim_start_matches("\\A(?:")
            .trim_start_matches("\\A");
        let trimmed = trimmed.trim_end_matches(")\\z").trim_end_matches("\\z");
        if regex_seq(trimmed.trim(), &chars, 0, 0) {
            return true;
        }
    }
    false
}

fn regex_seq(pat: &str, word: &[char], pi: usize, wi: usize) -> bool {
    let pchars: Vec<char> = pat.chars().collect();
    if pi >= pchars.len() {
        return wi == word.len();
    }
    if pi + 1 < pchars.len() && pchars[pi] == '.' && pchars[pi + 1] == '*' {
        return (wi..=word.len()).any(|next| regex_seq(pat, word, pi + 2, next));
    }
    if pchars[pi] == '[' {
        if let Some(close) = pat[pi..].find(']') {
            let class = &pat[pi + 1..pi + close];
            let rest = &pat[pi + close + 1..];
            let (quant, rest) = if let Some(r) = rest.strip_prefix('*') {
                ("*", r)
            } else if let Some(r) = rest.strip_prefix('?') {
                ("?", r)
            } else {
                ("", rest)
            };
            let (min, max) = match quant {
                "*" => (0usize, usize::MAX),
                "?" => (0, 1),
                _ => (1, 1),
            };
            for count in (min..=max.min(word.len() - wi)).rev() {
                let slice: Vec<char> = word[wi..wi + count].to_vec();
                if slice.iter().all(|c| class_matches(class, *c))
                    && regex_seq(rest, word, 0, wi + count)
                {
                    return true;
                }
            }
            return false;
        }
    }
    if wi < word.len() && pchars[pi] == word[wi] {
        return regex_seq(pat, word, pi + 1, wi + 1);
    }
    false
}

fn seq_match(
    rule: &PatternRule,
    pattern: &[super::pattern_rule::Constraint],
    p_i: usize,
    t_i: usize,
    tokens: &[TokenView],
    consumed: &[usize],
) -> Option<Vec<usize>> {
    if p_i == pattern.len() {
        return if consumed.is_empty() {
            None
        } else {
            Some(consumed.to_vec())
        };
    }
    let constraint = &pattern[p_i];

    if let Some(pos_values) = &constraint.pos {
        if pos_values.contains(&Pos::SentStart) {
            if t_i != 0 {
                return None;
            }
            return seq_match(rule, pattern, p_i + 1, t_i, tokens, consumed);
        }
    }

    if constraint.optional {
        if let Some(hit) = try_take(rule, pattern, p_i, t_i, tokens, consumed) {
            return Some(hit);
        }
        return seq_match(rule, pattern, p_i + 1, t_i, tokens, consumed);
    }

    try_take(rule, pattern, p_i, t_i, tokens, consumed)
}

fn try_take(
    rule: &PatternRule,
    pattern: &[super::pattern_rule::Constraint],
    p_i: usize,
    t_i: usize,
    tokens: &[TokenView],
    consumed: &[usize],
) -> Option<Vec<usize>> {
    let constraint = &pattern[p_i];
    if t_i >= tokens.len() {
        return None;
    }
    let token = &tokens[t_i];
    if !token_matches(constraint, token) {
        return None;
    }
    let mut taken = consumed.to_vec();
    taken.push(t_i);
    for gap in 0..=constraint.skip {
        if let Some(result) = seq_match(rule, pattern, p_i + 1, t_i + 1 + gap, tokens, &taken) {
            return Some(result);
        }
    }
    None
}

/// Class membership with range support ("a-z", "aeiouAEIOU").
fn class_matches(class: &str, c: char) -> bool {
    let chars: Vec<char> = class.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if i + 2 < chars.len() && chars[i + 1] == '-' {
            if c >= chars[i] && c <= chars[i + 2] {
                return true;
            }
            i += 3;
        } else {
            if chars[i] == c {
                return true;
            }
            i += 1;
        }
    }
    false
}
