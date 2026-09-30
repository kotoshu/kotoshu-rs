//! Grammar rules engine (TODO.grammar): YAML rules, a rule-based POS
//! tagger, and a recursive sequence matcher — the Rust twin of the
//! Ruby engine, driven by the SAME rule files.

pub mod checker;
pub mod pattern_rule;
pub mod pos_tagger;

pub use checker::{Checker, GrammarError};
