//! PatternRule: the Rust twin of Kotoshu::Grammar::PatternRule.
//! Deserializes the SAME YAML rule files; dispatches to the
//! pos_sequence or word_list matcher; renders template/replace/
//! replace_span suggestions with 3sg/base morphology.

use super::pos_tagger::{split_clitic, Pos, IRREGULAR_VERBS};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct RuleFile(pub Vec<RuleDef>);

#[derive(Debug, Deserialize)]
pub struct RuleDef {
    pub id: String,
    #[serde(default)]
    pub category: Option<String>,
    #[serde(default = "default_language")]
    pub language: String,
    #[serde(default)]
    pub matcher: Option<String>,
    pub pattern: serde_yaml::Value,
    pub message: String,
    #[serde(default)]
    pub suggestions: Vec<Suggestion>,
    #[serde(default)]
    pub examples: Vec<Example>,
    #[serde(default)]
    pub default: Option<String>,
}

fn default_language() -> String {
    "en".to_string()
}

#[derive(Debug, Deserialize)]
pub struct Example {
    #[serde(default)]
    pub bad: Option<String>,
    #[serde(default)]
    pub good: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum Suggestion {
    Template { template: String },
    Replace { replace: ReplaceDef },
    ReplaceSpan { replace_span: ReplaceSpanDef },
    Plain(String),
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ReplaceDef {
    pub word: String,
    pub with: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ReplaceSpanDef {
    pub with: String,
}

/// One pattern slot. Word values may be a scalar or a list; `no` and
/// `not` arrive as YAML booleans and are normalized to strings.
#[derive(Debug, Clone, PartialEq)]
pub struct Constraint {
    pub word: Option<Vec<String>>,
    pub regex: Option<String>,
    pub pos: Option<Vec<Pos>>,
    pub case_sensitive: bool,
    pub negated: bool,
    pub optional: bool,
    pub skip: usize,
    pub exception: Option<Vec<String>>,
    pub exception_pos: Option<Vec<Pos>>,
}

fn yaml_scalar_to_string(v: &serde_yaml::Value) -> Option<String> {
    match v {
        serde_yaml::Value::String(s) => Some(s.clone()),
        serde_yaml::Value::Bool(b) => Some(b.to_string()),
        serde_yaml::Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

fn word_list(v: &serde_yaml::Value) -> Option<Vec<String>> {
    match v {
        serde_yaml::Value::Sequence(seq) => Some(
            seq.iter()
                .filter_map(yaml_scalar_to_string)
                .collect(),
        ),
        v => yaml_scalar_to_string(v).map(|s| vec![s]),
    }
}

fn pos_list(v: &serde_yaml::Value) -> Option<Vec<Pos>> {
    match v {
        serde_yaml::Value::Sequence(seq) => Some(
            seq.iter()
                .filter_map(yaml_scalar_to_string)
                .map(|s| Pos::parse(&s))
                .collect(),
        ),
        v => yaml_scalar_to_string(v).map(|s| vec![Pos::parse(&s)]),
    }
}

impl Constraint {
    fn parse(v: &serde_yaml::Value) -> Constraint {
        let get = |key: &str| v.get(key);
        Constraint {
            word: get("word").and_then(word_list),
            regex: get("regex").and_then(yaml_scalar_to_string),
            pos: get("pos").and_then(pos_list),
            case_sensitive: get("case_sensitive")
                .and_then(|x| x.as_str())
                .is_some_and(|s| s == "yes" || s == "true")
                || get("case_sensitive")
                    .and_then(|x| x.as_bool())
                    .unwrap_or(false),
            negated: get("negated")
                .and_then(|x| x.as_str())
                .is_some_and(|s| s == "yes" || s == "true")
                || get("negated")
                    .and_then(|x| x.as_bool())
                    .unwrap_or(false),
            optional: get("optional")
                .and_then(|x| x.as_bool())
                .unwrap_or(false),
            skip: get("skip")
                .and_then(serde_yaml::Value::as_i64)
                .unwrap_or(0)
                .max(0) as usize,
            exception: get("exception").and_then(word_list),
            exception_pos: get("exception_pos").and_then(pos_list),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PatternRule {
    pub id: String,
    pub category: String,
    pub matcher: MatcherKind,
    pub pattern: Vec<Constraint>,
    pub words: Vec<Vec<String>>, // word_list phrases, split
    pub case_sensitive_words: bool,
    pub message: String,
    suggestions: Vec<Suggestion>,
    pub enabled: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MatcherKind {
    PosSequence,
    WordList,
}

impl PatternRule {
    pub fn from_def(def: &RuleDef) -> PatternRule {
        let matcher = match def.matcher.as_deref() {
            Some("word_list") => MatcherKind::WordList,
            _ => MatcherKind::PosSequence,
        };
        let (pattern, words, case_sensitive_words) = match matcher {
            MatcherKind::PosSequence => (
                def.pattern
                    .as_sequence()
                    .map(|seq| seq.iter().map(Constraint::parse).collect())
                    .unwrap_or_default(),
                Vec::new(),
                false,
            ),
            MatcherKind::WordList => {
                let map = def.pattern.as_mapping();
                let words = map
                    .and_then(|m| m.get(serde_yaml::Value::String("words".into())))
                    .and_then(word_list)
                    .unwrap_or_default()
                    .iter()
                    .map(|phrase| phrase.split_whitespace().map(String::from).collect())
                    .collect();
                let sensitive = map
                    .and_then(|m| m.get(serde_yaml::Value::String("case_sensitive".into())))
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                (Vec::new(), words, sensitive)
            }
        };
        PatternRule {
            id: def.id.clone(),
            category: def.category.clone().unwrap_or_else(|| "grammar".into()),
            matcher,
            pattern,
            words,
            case_sensitive_words,
            message: def.message.clone(),
            suggestions: def.suggestions.clone(),
            enabled: def.default.as_deref() != Some("off"),
        }
    }

    /// Render suggestions for a matched window.
    pub fn render_suggestions(&self, window: &[super::checker::TokenView]) -> Vec<String> {
        self.suggestions
            .iter()
            .filter_map(|s| match s {
                Suggestion::Replace { replace } => Some(
                    window
                        .iter()
                        .map(|t| {
                            if t.word.eq_ignore_ascii_case(&replace.word) {
                                replace.with.clone()
                            } else {
                                t.word.clone()
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(" ")
                        .split_whitespace()
                        .collect::<Vec<_>>()
                        .join(" "),
                ),
                Suggestion::ReplaceSpan { replace_span } => Some(replace_span.with.clone()),
                Suggestion::Template { template } => Some(expand_template(template, window)),
                Suggestion::Plain(text) => Some(expand_template(text, window)),
            })
            .collect()
    }
}

fn expand_template(template: &str, window: &[super::checker::TokenView]) -> String {
    // {{N}}, {{N|3sg}}, {{N|base}}
    let mut out = String::new();
    let bytes = template.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if template[i..].starts_with("{{") {
            if let Some(end) = template[i + 2..].find("}}") {
                let inner = &template[i + 2..i + 2 + end];
                let mut parts = inner.split('|');
                let idx: Option<usize> = parts.next().and_then(|p| p.trim().parse().ok());
                let verb = parts.next();
                if let Some(n) = idx {
                    let word = window.get(n).map(|t| t.word.as_str()).unwrap_or("");
                    match verb {
                        Some("3sg") => out.push_str(&inflect_3sg(word)),
                        Some("base") => out.push_str(&uninflect_3sg(word)),
                        _ => out.push_str(word),
                    }
                    i += 2 + end + 2;
                    continue;
                }
            }
        }
        out.push(template[i..].chars().next().unwrap());
        i += template[i..].chars().next().unwrap().len_utf8();
    }
    out
}

pub fn inflect_3sg(word: &str) -> String {
    let base = word.to_lowercase();
    if let Some((_, three_sg, _, _)) = IRREGULAR_VERBS.iter().find(|(b, _, _, _)| *b == base) {
        return three_sg.to_string();
    }
    let ends_es = ["s", "sh", "ch", "x", "z"]
        .iter()
        .any(|s| base.ends_with(s));
    let consonant_y = base.len() >= 2
        && base.ends_with('y')
        && !"aeiou".contains(base.chars().nth(base.len() - 2).unwrap_or('a'));
    if ends_es {
        format!("{base}es")
    } else if consonant_y {
        format!("{}ies", &base[..base.len() - 1])
    } else {
        format!("{base}s")
    }
}

pub fn uninflect_3sg(word: &str) -> String {
    let base = word.to_lowercase();
    if let Some((stem, _, _, _)) = IRREGULAR_VERBS.iter().find(|(_, sg, _, _)| *sg == base) {
        return stem.to_string();
    }
    if base.len() >= 4 && base.ends_with("ies") {
        return format!("{}y", &base[..base.len() - 3]);
    }
    for suffix in ["ches", "shes", "xes", "zes", "oes"] {
        if base.ends_with(suffix) {
            return base[..base.len() - suffix.len() + if suffix == "oes" { 1 } else { 2 }].to_string();
        }
    }
    if base.len() >= 4 && base.ends_with('s') && !base.ends_with("ss") {
        return base[..base.len() - 1].to_string();
    }
    word.to_string()
}

/// Split a word into clitic pieces for tokenization checks.
pub fn clitic_split(word: &str) -> Option<(String, String)> {
    split_clitic(word)
}
