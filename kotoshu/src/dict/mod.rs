//! Hunspell dictionary engine: `.aff`/`.dic` parsing, affix expansion and
//! `correct?` lookup.
//!
//! Behavioral reference: the Ruby gem's readers (`AffReader`, `DicReader`,
//! `LookupBuilder`) and `Algorithms::Lookup` / `Algorithms::Capitalization`.
//! The port reproduces the gem's behavior exactly, quirks included — the
//! conformance vectors freeze that behavior.

mod aff;
mod casing;
mod dic;
mod encoding;
mod lookup;

use std::fmt;
use std::path::Path;

/// A loaded Hunspell dictionary answering `correct?` queries.
///
/// Created with [`Dictionary::load`]; each instance owns its parsed `.aff`
/// configuration and indexed `.dic` entries.
#[derive(Debug)]
pub struct Dictionary {
    lookup: lookup::Lookuper,
}

impl Dictionary {
    /// Load a dictionary from its `.aff` and `.dic` files.
    ///
    /// The `.dic` file is interpreted with the flag format and `AF` aliases
    /// declared by the `.aff` file, mirroring the gem's `LookupBuilder`.
    pub fn load(aff_path: &Path, dic_path: &Path) -> Result<Self, LoadError> {
        lookup::Lookuper::load(aff_path, dic_path).map(|lookup| Self { lookup })
    }

    /// Whether `word` is spelled correctly per this dictionary.
    ///
    /// Covers the gem's full lookup path: capitalization variants, affix
    /// stripping (two-fold suffixes, `COMPLEXPREFIXES` double prefixes,
    /// cross-product prefix+suffix), compounds (flag- and rule-based),
    /// break patterns, `ICONV`, `IGNORE`, `KEEPCASE`/`NEEDAFFIX`/
    /// `CIRCUMFIX`/`ONLYINCOMPOUND` and friends.
    pub fn correct(&self, word: &str) -> bool {
        self.lookup.call(word)
    }
}

/// Failure to load a dictionary.
#[derive(Debug)]
pub enum LoadError {
    /// An `.aff`/`.dic` file could not be read.
    Io(std::io::Error),
    /// The `.aff` file is malformed (bad directive payload, truncated
    /// counted block, unknown flag format).
    Aff(String),
    /// The `.dic` file is malformed.
    Dic(String),
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::Io(error) => write!(f, "i/o error: {error}"),
            LoadError::Aff(message) => write!(f, "aff parse error: {message}"),
            LoadError::Dic(message) => write!(f, "dic parse error: {message}"),
        }
    }
}

impl From<std::io::Error> for LoadError {
    fn from(error: std::io::Error) -> Self {
        LoadError::Io(error)
    }
}

impl std::error::Error for LoadError {}
