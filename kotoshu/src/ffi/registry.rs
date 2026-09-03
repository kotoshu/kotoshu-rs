//! Process-wide dictionary registry: the KOSH batch protocol routes by
//! language, so hosts load dictionaries once
//! (C ABI: [`crate::ffi::c::kotoshu_dict_load`]) and reference them from
//! batch requests. Overwriting a language replaces its dictionary
//! atomically; `free` drops it.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

use crate::dict::Dictionary;

fn registry() -> &'static Mutex<HashMap<String, Arc<Dictionary>>> {
    static REGISTRY: OnceLock<Mutex<HashMap<String, Arc<Dictionary>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Load an `.aff`/`.dic` pair and register it under `language`,
/// replacing any dictionary already registered for that language.
///
/// # Errors
///
/// Propagates [`crate::dict::LoadError`] (the previous registration, if
/// any, is preserved on failure).
pub fn register(
    language: &str,
    aff_path: &Path,
    dic_path: &Path,
) -> Result<(), crate::dict::LoadError> {
    let dictionary = Arc::new(Dictionary::load(aff_path, dic_path)?);
    registry()
        .lock()
        .expect("kotoshu dictionary registry poisoned")
        .insert(language.to_owned(), dictionary);
    Ok(())
}

/// Drop the dictionary registered under `language`. Returns whether one
/// was registered.
pub fn unregister(language: &str) -> bool {
    registry()
        .lock()
        .expect("kotoshu dictionary registry poisoned")
        .remove(language)
        .is_some()
}

/// The dictionary registered under `language`, if any.
pub fn lookup(language: &str) -> Option<Arc<Dictionary>> {
    registry()
        .lock()
        .expect("kotoshu dictionary registry poisoned")
        .get(language)
        .cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    const AFF: &str = "SET UTF-8\n";
    const DIC: &str = "2\nhello\nworld\n";

    fn temp_dictionary(tag: &str, dic_body: &str) -> (std::path::PathBuf, std::path::PathBuf) {
        let aff = std::env::temp_dir().join(format!("kotoshu-registry-{tag}.aff"));
        let dic = std::env::temp_dir().join(format!("kotoshu-registry-{tag}.dic"));
        std::fs::write(&aff, AFF).unwrap();
        std::fs::write(&dic, dic_body).unwrap();
        (aff, dic)
    }

    #[test]
    fn register_lookup_unregister_cycle() {
        let (aff, dic) = temp_dictionary("cycle", DIC);
        let language = "zz-cycle";
        assert!(lookup(language).is_none());
        register(language, &aff, &dic).unwrap();
        assert!(lookup(language).is_some());
        assert!(lookup(language).unwrap().correct("hello"));
        assert!(unregister(language));
        assert!(!unregister(language));
        assert!(lookup(language).is_none());
    }

    #[test]
    fn reregistration_replaces_and_failure_preserves() {
        let (aff, dic) = temp_dictionary("replace", DIC);
        let (aff2, dic2) = temp_dictionary("replace2", "1\nkotoshu\n");
        let language = "zz-replace";
        register(language, &aff, &dic).unwrap();
        register(language, &aff2, &dic2).unwrap();
        let dictionary = lookup(language).unwrap();
        assert!(!dictionary.correct("hello"));
        assert!(dictionary.correct("kotoshu"));

        let missing = std::env::temp_dir().join("kotoshu-registry-missing.aff");
        std::fs::remove_file(&missing).ok();
        assert!(register(language, &missing, &dic).is_err());
        assert!(lookup(language).is_some());
        assert!(lookup(language).unwrap().correct("kotoshu"));
        unregister(language);
    }
}
