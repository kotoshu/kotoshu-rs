//! Rule-based English POS tagger — the Rust twin of
//! Kotoshu::Grammar::PosTagger. Word lists, suffix rules, context
//! disambiguation; the tag set and decision order match exactly so
//! conformance vectors replay identically.

/// The reduced tag set. The `Other` arm exists for extension tags the
/// Ruby side may emit (WH, PUNCT, ANY); comparisons are by name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pos {
    Noun,
    SingNoun,
    PlurNoun,
    ProperNoun,
    Verb,
    VerbBase,
    Verb3sg,
    VerbPast,
    VerbIng,
    VerbParticiple,
    Adj,
    Adv,
    Prep,
    Det,
    Pron,
    Pron1sg,
    Pron3sg,
    PronPlural,
    Modal,
    Aux,
    Conj,
    Interj,
    Number,
    Wh,
    Punct,
    SentStart,
    SentEnd,
    Any,
    Unknown,
}

impl Pos {
    pub fn parse(name: &str) -> Pos {
        match name {
            "NOUN" => Pos::Noun,
            "SING_NOUN" => Pos::SingNoun,
            "PLUR_NOUN" => Pos::PlurNoun,
            "PROPER_NOUN" => Pos::ProperNoun,
            "VERB" => Pos::Verb,
            "VERB_BASE" => Pos::VerbBase,
            "VERB_3SG" => Pos::Verb3sg,
            "VERB_PAST" => Pos::VerbPast,
            "VERB_ING" => Pos::VerbIng,
            "VERB_PARTICIPLE" => Pos::VerbParticiple,
            "ADJ" => Pos::Adj,
            "ADV" => Pos::Adv,
            "PREP" => Pos::Prep,
            "DET" => Pos::Det,
            "PRON" => Pos::Pron,
            "PRON_1SG" => Pos::Pron1sg,
            "PRON_3SG" => Pos::Pron3sg,
            "PRON_PLURAL" => Pos::PronPlural,
            "MODAL" => Pos::Modal,
            "AUX" => Pos::Aux,
            "CONJ" => Pos::Conj,
            "INTERJ" => Pos::Interj,
            "NUMBER" => Pos::Number,
            "WH" => Pos::Wh,
            "PUNCT" => Pos::Punct,
            "SENT_START" => Pos::SentStart,
            "SENT_END" => Pos::SentEnd,
            "ANY" => Pos::Any,
            _ => Pos::Unknown,
        }
    }

    /// Hierarchical matching (pos_match? in the Ruby matcher).
    pub fn matches(&self, expected: &Pos) -> bool {
        if self == expected {
            return true;
        }
        use Pos::*;
        match expected {
            Verb => matches!(
                self,
                Verb | VerbBase | Verb3sg | VerbPast | VerbIng | VerbParticiple
            ),
            Noun => matches!(self, Noun | SingNoun | PlurNoun | ProperNoun),
            Pron => matches!(self, Pron | Pron1sg | Pron3sg | PronPlural),
            Any => true,
            _ => false,
        }
    }
}

const PRONOUNS_1SG: &[&str] = &["i"];
const PRONOUNS_OBJ: &[&str] = &["me", "him", "her", "us", "them"];
const WH_WORDS: &[&str] = &[
    "who",
    "whom",
    "whose",
    "which",
    "what",
    "where",
    "why",
    "how",
    "whoever",
    "whomever",
    "whatever",
    "whichever",
    "wherever",
    "however",
    "whenever",
];
const PRONOUNS_3SG: &[&str] = &["he", "she", "it", "this", "that"];
const PRONOUNS_PLURAL: &[&str] = &["they", "we", "you", "these", "those"];
const MODALS: &[&str] = &[
    "can", "could", "may", "might", "must", "shall", "should", "will", "would", "ought", "need",
    "dare",
];
const AUXILIARIES: &[&str] = &[
    "be", "am", "is", "are", "was", "were", "been", "being", "have", "has", "had", "do", "does",
    "did",
];
const DETERMINERS: &[&str] = &[
    "a", "an", "the", "this", "that", "these", "those", "my", "your", "his", "her", "its", "our",
    "their", "every", "each", "some", "any", "no", "all", "both", "half",
];
const PREPOSITIONS: &[&str] = &[
    "in", "on", "at", "by", "for", "with", "from", "to", "of", "about", "over", "under", "between",
    "among", "through", "during", "before", "after", "above", "below", "near", "against",
    "without", "within", "into", "onto", "upon", "across", "behind", "beyond", "despite", "except",
    "inside", "outside", "toward", "towards",
];
const CONJUNCTIONS: &[&str] = &[
    "and", "but", "or", "nor", "so", "yet", "for", "if", "then", "than", "when", "while",
    "although", "because", "since", "unless", "until", "whereas", "whether",
];
const INTERJECTIONS: &[&str] = &["oh", "ah", "eh", "um", "uh", "hey", "wow", "ouch", "oops"];
const NUMBERS: &[&str] = &[
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "hundred",
    "thousand", "million", "billion",
];
const COMMON_ADJECTIVES: &[&str] = &[
    "good",
    "bad",
    "new",
    "old",
    "young",
    "big",
    "small",
    "large",
    "great",
    "little",
    "long",
    "short",
    "high",
    "low",
    "early",
    "late",
    "easy",
    "hard",
    "simple",
    "difficult",
    "wrong",
    "right",
    "strong",
    "weak",
    "hot",
    "cold",
    "warm",
    "cool",
    "clean",
    "dirty",
    "fast",
    "slow",
    "open",
    "closed",
    "rich",
    "poor",
    "happy",
    "sad",
    "ready",
    "true",
    "false",
    "real",
    "many",
    "few",
    "several",
    "most",
    "more",
    "enough",
    "important",
    "interesting",
    "available",
    "possible",
    "able",
    "similar",
    "various",
    "popular",
    "expensive",
    "cheap",
    "beautiful",
    "famous",
    "comfortable",
    "dangerous",
    "different",
    "common",
    "special",
    "modern",
    "national",
    "public",
    "political",
    "social",
    "economic",
    "international",
    "necessary",
    "responsible",
    "serious",
    "careful",
    "useful",
    "useless",
    "helpful",
    "proud",
    "afraid",
    "alive",
    "alone",
    "aware",
];
const IRREGULAR_PLURALS: &[&str] = &[
    "people", "children", "men", "women", "feet", "teeth", "mice", "geese", "oxen", "sheep",
    "deer", "fish",
];

/// Irregular verbs: base → [3sg, past, participle].
pub const IRREGULAR_VERBS: &[(&str, &str, &str, &str)] = &[
    ("be", "is", "was", "been"),
    ("have", "has", "had", "had"),
    ("do", "does", "did", "done"),
    ("go", "goes", "went", "gone"),
    ("say", "says", "said", "said"),
    ("make", "makes", "made", "made"),
    ("know", "knows", "knew", "known"),
    ("take", "takes", "took", "taken"),
    ("see", "sees", "saw", "seen"),
    ("come", "comes", "came", "come"),
    ("get", "gets", "got", "gotten"),
    ("give", "gives", "gave", "given"),
    ("find", "finds", "found", "found"),
    ("tell", "tells", "told", "told"),
    ("write", "writes", "wrote", "written"),
    ("run", "runs", "ran", "run"),
    ("put", "puts", "put", "put"),
    ("set", "sets", "set", "set"),
];

/// Suffix → POS candidates, checked in order (SUFFIX_RULES order).
fn suffix_pos(lower: &str) -> Option<Pos> {
    const RULES: &[(&str, Pos)] = &[
        ("ing", Pos::VerbIng),
        ("edly", Pos::Adv),
        ("ly", Pos::Adv),
        ("tion", Pos::Noun),
        ("sion", Pos::Noun),
        ("ment", Pos::Noun),
        ("ness", Pos::Noun),
        ("ity", Pos::Noun),
        ("ance", Pos::Noun),
        ("ence", Pos::Noun),
        ("able", Pos::Adj),
        ("ible", Pos::Adj),
        ("ous", Pos::Adj),
        ("ful", Pos::Adj),
        ("less", Pos::Adj),
        ("ive", Pos::Adj),
        ("ish", Pos::Adj),
        ("al", Pos::Adj),
        ("ic", Pos::Adj),
        ("ed", Pos::VerbPast),
        ("er", Pos::Noun),
        ("est", Pos::Adj),
        ("s", Pos::PlurNoun),
    ];
    RULES
        .iter()
        .find(|(suffix, _)| lower.ends_with(suffix))
        .map(|(_, pos)| *pos)
}

fn subject_pronoun(word: &str) -> bool {
    matches!(
        word,
        "he" | "she" | "it" | "i" | "they" | "we" | "you" | "this" | "that" | "these" | "those"
    )
}

fn is_punct(word: &str) -> bool {
    !word.is_empty() && word.chars().all(|c| c.is_ascii_punctuation())
}

/// Detach a trailing English clitic ("Valentine's" → base + clitic),
/// matching the Ruby split_clitic (including the lazy-base "ca" from
/// "can't").
pub fn split_clitic(word: &str) -> Option<(String, String)> {
    const CLITICS: &[&str] = &["n't", "'s", "'t", "'re", "'ve", "'ll", "'d", "'m"];
    let lower = word.to_lowercase();
    for clitic in CLITICS {
        if lower.len() > clitic.len() && lower.ends_with(clitic) {
            let split_at = word.len() - clitic.len();
            if split_at >= 2 {
                return Some((word[..split_at].to_string(), clitic.to_string()));
            }
        }
    }
    None
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub word: String,
    pub pos: Pos,
    pub start_offset: usize,
    pub end_offset: usize,
}

pub fn tokenize_with_offsets(sentence: &str, base: usize) -> Vec<(String, usize, usize)> {
    let mut out = Vec::new();
    let bytes = sentence.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        // Unicode word characters: alphanumeric in any script, plus
        // _ ' - (ASCII \w drops accents, mangling fr/de/es text).
        let _ch_len = utf8_char_len(&sentence[i..]);
        let ch = sentence[i..].chars().next().unwrap();
        if ch.is_alphanumeric() || ch == '_' || ch == '\'' || ch == '-' {
            let start = i;
            while i < bytes.len() {
                let c = sentence[i..].chars().next().unwrap();
                if c.is_alphanumeric() || c == '_' || c == '\'' || c == '-' {
                    i += utf8_char_len(&sentence[i..]);
                } else {
                    break;
                }
            }
            push_word(&mut out, &sentence[start..i], start + base);
        } else if b.is_ascii_punctuation() {
            let start = i;
            i += 1;
            push_word(&mut out, &sentence[start..i], start + base);
        } else {
            i += 1;
        }
    }
    out
}

fn push_word(out: &mut Vec<(String, usize, usize)>, word: &str, start: usize) {
    match split_clitic(word) {
        Some((base_part, clitic)) => {
            out.push((base_part.clone(), start, start + base_part.len() - 1));
            out.push((clitic, start + base_part.len(), start + word.len() - 1));
        }
        None => out.push((word.to_string(), start, start + word.len() - 1)),
    }
}

fn pos_for(word: &str, index: usize, all_words: &[String]) -> Pos {
    let lower = word.to_lowercase();

    if is_punct(word) {
        return Pos::Punct;
    }
    if COMMON_ADJECTIVES.contains(&lower.as_str()) {
        return Pos::Adj;
    }
    if IRREGULAR_PLURALS.contains(&lower.as_str()) {
        return Pos::PlurNoun;
    }
    if WH_WORDS.contains(&lower.as_str()) {
        return Pos::Wh;
    }
    if PRONOUNS_OBJ.contains(&lower.as_str()) {
        return Pos::Pron;
    }
    if PRONOUNS_1SG.contains(&lower.as_str()) {
        return Pos::Pron1sg;
    }
    if PRONOUNS_3SG.contains(&lower.as_str()) {
        return Pos::Pron3sg;
    }
    if PRONOUNS_PLURAL.contains(&lower.as_str()) {
        return Pos::PronPlural;
    }
    if MODALS.contains(&lower.as_str()) {
        return Pos::Modal;
    }
    if AUXILIARIES.contains(&lower.as_str()) {
        return Pos::Aux;
    }
    if DETERMINERS.contains(&lower.as_str()) {
        return Pos::Det;
    }
    if PREPOSITIONS.contains(&lower.as_str()) {
        return Pos::Prep;
    }
    if CONJUNCTIONS.contains(&lower.as_str()) {
        return Pos::Conj;
    }
    if INTERJECTIONS.contains(&lower.as_str()) {
        return Pos::Interj;
    }
    if NUMBERS.contains(&lower.as_str()) || lower.chars().next().is_some_and(|c| c.is_ascii_digit())
    {
        return Pos::Number;
    }
    if lower == "is" || lower == "has" {
        return Pos::Verb3sg;
    }
    if lower == "was" || lower == "had" {
        return Pos::VerbPast;
    }
    if IRREGULAR_VERBS.iter().any(|(base, _, _, _)| *base == lower) {
        return Pos::VerbBase;
    }
    for (_, three_sg, past, participle) in IRREGULAR_VERBS {
        if lower == *three_sg {
            return Pos::Verb3sg;
        }
        if lower == *past {
            return Pos::VerbPast;
        }
        if lower == *participle {
            return Pos::VerbParticiple;
        }
    }
    if let Some(pos) = suffix_pos(&lower) {
        return pos;
    }
    if index > 0 {
        let prev = all_words[index - 1].to_lowercase();
        if MODALS.contains(&prev.as_str()) {
            return Pos::VerbBase;
        }
        let next_raw = all_words.get(index + 1).map(|w| w.as_str()).unwrap_or("");
        let clitic_next = next_raw.starts_with('\'') || next_raw == "n't";
        if subject_pronoun(&prev) && !clitic_next && suffix_pos(&lower).is_none() {
            return Pos::VerbBase;
        }
    }
    if index > 0 && word.chars().next().is_some_and(|c| c.is_ascii_uppercase()) {
        return Pos::ProperNoun;
    }
    Pos::Noun
}

/// Tag words into tokens (word + POS; offsets are filled by the
/// checker, which owns them).
pub fn tag(words: &[String]) -> Vec<Pos> {
    (0..words.len())
        .map(|i| pos_for(&words[i], i, words))
        .collect()
}

/// Byte length of the UTF-8 character starting at the head of `s`.
fn utf8_char_len(s: &str) -> usize {
    s.chars().next().map(|c| c.len_utf8()).unwrap_or(1)
}
