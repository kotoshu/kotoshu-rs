//! Language packs (feature `model`) — plan 113: one fetch for a whole
//! language.
//!
//! A language load is normally five transfers from two repos (`.aff` +
//! `.dic` from `kotoshu/dictionaries`; mini model + vocab + buckets from
//! `kotoshu/models-fasttext-onnx`). A pack (`kotoshu://packs/{lang}` in
//! the models registry) concatenates those artifacts into ONE file, so
//! a cold load pays one round trip and one progress stream.
//!
//! The `KPK1` framing is a length-prefixed section stream, pinned by
//! the builder (`scripts/build_packs.py` in the models repo) and this
//! reader together through a shared golden test (same synthetic
//! sections, same whole-file sha256 — the test module documents it):
//!
//! ```text
//! offset 0   magic "KPK1"              4 bytes
//! offset 4   u32 LE section_count
//! then one framing per section, in fixed order:
//!             u32 LE payload_length    4 bytes
//!             u8 tag                   1 byte (1 aff, 2 dic, 3 model,
//!                                      4 vocab, 5 buckets)
//!             payload                  payload_length bytes
//!             sha256(payload)          32 bytes (per-section footer)
//! ```
//!
//! Sections carry the EXACT bytes of their parts — no compression, no
//! transformation — so a pack verifies by checksum alone:
//! [`parse`] checks every section footer before returning anything, and
//! a mismatch aborts the load (a truncated or corrupt fetch can never
//! construct half a language). `buckets` is the one optional section
//! (languages whose bucket table failed the plan 103 gates ship
//! without); everything else is required.
//!
//! [`load`] then hands the sections to the ordinary per-artifact
//! loaders — [`crate::dict::Dictionary::load_from_sources`] and
//! [`crate::rerank::int8_model::Int8Model::parse`] +
//! [`Int8Model::attach_buckets`] — which is the parity contract: the
//! pack path constructs the same handles the per-artifact path
//! constructs, from the same bytes, so behavior is identical by
//! construction (the tests prove it on the checked-in real-derived
//! fixtures).
//!
//! The aff/dic sections must be UTF-8: the wasm surface hands
//! dictionary sources to the engine as strings, exactly like a
//! per-artifact text fetch does (the registry-pack builder refuses to
//! cut a pack from non-UTF-8 dictionary sources for the same reason).

use std::fmt;

use sha2::{Digest, Sha256};

use crate::dict::Dictionary;
use crate::rerank::int8_model::Int8Model;
use crate::typo::model::TypoModel;

/// The pack magic — `KPK1`, the framing version (`KPK2` would be a new
/// format, cut as new artifacts, never parsed leniently).
pub const MAGIC: [u8; 4] = *b"KPK1";

/// Header size: magic + u32 LE section count.
const HEADER_LEN: usize = 4 + 4;

/// One pack section. Tags are the wire contract (`1..=5`, fixed order).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SectionTag {
    /// The dictionary `.aff` source (UTF-8 text).
    Aff = 1,
    /// The dictionary `.dic` source (UTF-8 text).
    Dic = 2,
    /// The embedding tier `.onnx` artifact (mini by default).
    Model = 3,
    /// The tier `.vocab.json` sibling.
    Vocab = 4,
    /// The bucket-table sibling (plan 103; optional).
    Buckets = 5,
    /// The typo bi-encoder `.onnx` artifact (plan 131; optional, and
    /// only meaningful beside [`SectionTag::TypoVocab`]).
    TypoModel = 6,
    /// The typo bi-encoder char-vocab sibling (plan 131; optional).
    TypoVocab = 7,
}

impl SectionTag {
    fn from_u8(byte: u8) -> Option<Self> {
        match byte {
            1 => Some(Self::Aff),
            2 => Some(Self::Dic),
            3 => Some(Self::Model),
            4 => Some(Self::Vocab),
            5 => Some(Self::Buckets),
            6 => Some(Self::TypoModel),
            7 => Some(Self::TypoVocab),
            _ => None,
        }
    }

    /// The canonical section name (also the registry `contents` key).
    pub fn name(self) -> &'static str {
        match self {
            Self::Aff => "aff",
            Self::Dic => "dic",
            Self::Model => "model",
            Self::Vocab => "vocab",
            Self::Buckets => "buckets",
            Self::TypoModel => "typo-model",
            Self::TypoVocab => "typo-vocab",
        }
    }
}

/// Errors of the pack reader.
#[derive(Debug)]
pub enum PackError {
    /// The bytes are not a pack (bad magic, truncated header, bad
    /// count) or the framing walk fails (truncation, trailing bytes,
    /// duplicate or unknown tags).
    Format(String),
    /// A section footer does not match the sha256 of its payload.
    Checksum { section: &'static str },
    /// A required section is missing.
    Missing(&'static str),
    /// The aff/dic section is not valid UTF-8.
    Utf8 { section: &'static str },
    /// The dictionary sections failed to load as a dictionary.
    Dictionary(crate::dict::LoadError),
    /// The model/vocab sections failed to load as a tier.
    Model(crate::rerank::int8_model::Int8ModelError),
}

impl fmt::Display for PackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Format(source) => write!(f, "pack: {source}"),
            Self::Checksum { section } => {
                write!(f, "pack section {section:?}: sha256 footer mismatch")
            }
            Self::Missing(section) => write!(f, "pack: required section {section:?} missing"),
            Self::Utf8 { section } => {
                write!(f, "pack section {section:?}: not valid UTF-8")
            }
            Self::Dictionary(source) => write!(f, "pack dictionary: {source}"),
            Self::Model(source) => write!(f, "pack model: {source}"),
        }
    }
}

impl std::error::Error for PackError {}

/// The borrowed sections of one parsed pack — zero-copy views into the
/// pack bytes, every footer already verified.
#[derive(Debug)]
pub struct Sections<'a> {
    /// The `.aff` source text bytes.
    pub aff: &'a [u8],
    /// The `.dic` source text bytes.
    pub dic: &'a [u8],
    /// The tier `.onnx` artifact bytes.
    pub model: &'a [u8],
    /// The tier `.vocab.json` bytes.
    pub vocab: &'a [u8],
    /// The bucket-table artifact bytes, when the pack carries one.
    pub buckets: Option<&'a [u8]>,
    /// The typo bi-encoder artifact bytes, when the pack carries the
    /// plan-131 pair.
    pub typo_model: Option<&'a [u8]>,
    /// The typo bi-encoder char-vocab bytes, when the pack carries the
    /// plan-131 pair.
    pub typo_vocab: Option<&'a [u8]>,
}

/// One fully loaded language pack: the dictionary and the embedding
/// model a per-artifact load would produce, from one byte stream.
pub struct LoadedPack {
    /// The dictionary built from the aff/dic sections.
    pub dictionary: Dictionary,
    /// The tier model built from the model/vocab sections, with the
    /// bucket table attached when present.
    pub model: Int8Model,
    /// The typo bi-encoder built from the typo-model/typo-vocab
    /// sections, when the pack carries the plan-131 pair. The derived
    /// per-language index is NOT part of the pack (it derives from the
    /// tier vocabulary at load, by design).
    pub typo: Option<TypoModel>,
}

/// Hand-written: the handles carry multi-megabyte tables whose
/// `Debug` would be noise; tests need the type name and whether the
/// typo half is present.
impl std::fmt::Debug for LoadedPack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadedPack")
            .field("dictionary", &"..")
            .field("model", &"..")
            .field("typo", &self.typo.as_ref().map(|_| ".."))
            .finish()
    }
}

/// Walk the framing and verify every section footer.
///
/// Returns the sections in fixed order. Errors on: bad magic, header
/// truncation, count mismatch, truncated section headers or payloads,
/// unknown or duplicate tags, out-of-order tags, sha256 footer
/// mismatch, and trailing bytes.
pub fn parse(bytes: &[u8]) -> Result<Sections<'_>, PackError> {
    if bytes.len() < HEADER_LEN {
        return Err(PackError::Format(format!(
            "too short for a header ({} bytes)",
            bytes.len()
        )));
    }
    if bytes[..4] != MAGIC {
        return Err(PackError::Format(format!(
            "bad magic {:?} (expected {:?})",
            &bytes[..4],
            MAGIC
        )));
    }
    let count = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;

    let mut aff: Option<&[u8]> = None;
    let mut dic: Option<&[u8]> = None;
    let mut model: Option<&[u8]> = None;
    let mut vocab: Option<&[u8]> = None;
    let mut buckets: Option<&[u8]> = None;
    let mut typo_model: Option<&[u8]> = None;
    let mut typo_vocab: Option<&[u8]> = None;
    let mut pos = HEADER_LEN;
    let mut expected: &[SectionTag] = &[SectionTag::Aff];

    for index in 0..count {
        if pos + 5 > bytes.len() {
            return Err(PackError::Format(format!(
                "truncated section header #{index} at offset {pos}"
            )));
        }
        let length =
            u32::from_le_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]])
                as usize;
        let tag = SectionTag::from_u8(bytes[pos + 4]).ok_or_else(|| {
            PackError::Format(format!(
                "unknown section tag {} at offset {}",
                bytes[pos + 4],
                pos
            ))
        })?;
        // The order is part of the format: the builder writes aff, dic,
        // model, vocab, then the optional buckets and typo pair.
        // Accepting any order would make the offsets the registry
        // declares ambiguous. Optionals may be skipped, so the
        // expectation is the set of tags that may legally follow.
        if !expected.contains(&tag) {
            let wants = if expected.is_empty() {
                "<end>".to_owned()
            } else {
                expected
                    .iter()
                    .map(|tag| tag.name())
                    .collect::<Vec<_>>()
                    .join(" or ")
            };
            return Err(PackError::Format(format!(
                "section #{index} is {} but the format order expects {}",
                tag.name(),
                wants
            )));
        }
        pos += 5;
        let payload_end = pos
            .checked_add(length)
            .filter(|end| *end + 32 <= bytes.len())
            .ok_or_else(|| {
                PackError::Format(format!(
                    "section {} truncated at offset {pos} (length {length})",
                    tag.name()
                ))
            })?;
        let payload = &bytes[pos..payload_end];
        let footer = &bytes[payload_end..payload_end + 32];
        if Sha256::digest(payload)[..] != footer[..] {
            return Err(PackError::Checksum {
                section: tag.name(),
            });
        }
        match tag {
            SectionTag::Aff => aff = Some(payload),
            SectionTag::Dic => dic = Some(payload),
            SectionTag::Model => model = Some(payload),
            SectionTag::Vocab => vocab = Some(payload),
            SectionTag::Buckets => buckets = Some(payload),
            SectionTag::TypoModel => typo_model = Some(payload),
            SectionTag::TypoVocab => typo_vocab = Some(payload),
        }
        pos = payload_end + 32;
        expected = match tag {
            SectionTag::Aff => &[SectionTag::Dic],
            SectionTag::Dic => &[SectionTag::Model],
            SectionTag::Model => &[SectionTag::Vocab],
            // buckets and the typo pair are independently optional:
            // after vocab comes either, or the end
            SectionTag::Vocab => &[SectionTag::Buckets, SectionTag::TypoModel],
            SectionTag::Buckets => &[SectionTag::TypoModel],
            SectionTag::TypoModel => &[SectionTag::TypoVocab],
            SectionTag::TypoVocab => &[],
        };
    }
    if pos != bytes.len() {
        return Err(PackError::Format(format!(
            "{} trailing bytes after {count} sections",
            bytes.len() - pos
        )));
    }
    let aff = aff.ok_or(PackError::Missing("aff"))?;
    let dic = dic.ok_or(PackError::Missing("dic"))?;
    let model = model.ok_or(PackError::Missing("model"))?;
    let vocab = vocab.ok_or(PackError::Missing("vocab"))?;
    Ok(Sections {
        aff,
        dic,
        model,
        vocab,
        buckets,
        typo_model,
        typo_vocab,
    })
}

/// Parse and load a whole language pack: the dictionary from its aff/dic
/// sections, the tier model from its model/vocab sections, the bucket
/// table attached when present — the same handles a per-artifact load
/// constructs from the same bytes.
pub fn load(bytes: &[u8]) -> Result<LoadedPack, PackError> {
    let sections = parse(bytes)?;
    let aff = std::str::from_utf8(sections.aff).map_err(|_| PackError::Utf8 { section: "aff" })?;
    let dic = std::str::from_utf8(sections.dic).map_err(|_| PackError::Utf8 { section: "dic" })?;
    let dictionary = Dictionary::load_from_sources(aff, dic).map_err(PackError::Dictionary)?;
    let mut model = Int8Model::parse(sections.model, sections.vocab).map_err(PackError::Model)?;
    if let Some(buckets) = sections.buckets {
        model.attach_buckets(buckets).map_err(PackError::Model)?;
    }
    let typo = match (sections.typo_model, sections.typo_vocab) {
        (Some(onnx), Some(vocab)) => Some(
            TypoModel::parse(onnx, vocab)
                .map_err(|error| PackError::Format(format!("typo bi-encoder section: {error}")))?,
        ),
        (None, None) => None,
        (found, missing) => {
            let found = if found.is_some() {
                "typo-model"
            } else {
                "typo-vocab"
            };
            let missing = if missing.is_some() {
                "typo-model"
            } else {
                "typo-vocab"
            };
            return Err(PackError::Format(format!(
                "pack carries {found} without {missing} — the typo pair travels together"
            )));
        }
    };
    Ok(LoadedPack {
        dictionary,
        model,
        typo,
    })
}

#[cfg(test)]
mod typo_section_tests {
    use super::*;

    fn synced_typo() -> Option<(Vec<u8>, Vec<u8>)> {
        let base = concat!(env!("CARGO_MANIFEST_DIR"), "/../tests/fixtures/models");
        let onnx = std::fs::read(format!("{base}/typo.biencoder.onnx")).ok()?;
        let vocab = std::fs::read(format!("{base}/typo.biencoder.vocab.json")).ok()?;
        Some((onnx, vocab))
    }

    fn sections_pack(typo: Option<(&[u8], &[u8])>) -> Vec<u8> {
        // the test builder below (build) writes the five frozen tags;
        // this local builder appends the optional pair the same way
        use sha2::Digest;
        let aff: &[u8] = b"SET UTF-8\nTRY esianrtolcdugmphbyfvkwz\n";
        let dic: &[u8] = b"3\nhello\nworld\nkotoshu\n";
        let model: &[u8] = include_bytes!("../tests/fixtures/models/en-mini-truncated.onnx");
        let vocab: &[u8] = include_bytes!("../tests/fixtures/models/en-mini-truncated.vocab.json");
        let mut parts: Vec<(u8, &[u8])> = vec![
            (SectionTag::Aff as u8, aff),
            (SectionTag::Dic as u8, dic),
            (SectionTag::Model as u8, model),
            (SectionTag::Vocab as u8, vocab),
        ];
        if let Some((onnx, chars)) = typo {
            parts.push((SectionTag::TypoModel as u8, onnx));
            parts.push((SectionTag::TypoVocab as u8, chars));
        }
        let mut out = Vec::new();
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&(parts.len() as u32).to_le_bytes());
        for (tag, payload) in parts {
            out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
            out.push(tag);
            out.extend_from_slice(payload);
            out.extend_from_slice(&Sha256::digest(payload));
        }
        out
    }

    #[test]
    fn typo_sections_load_and_half_pairs_error() {
        let Some((onnx, chars)) = synced_typo() else {
            eprintln!("typo fixtures absent (skipped)");
            return;
        };
        // full pair loads with the encoder present
        let loaded = load(&sections_pack(Some((&onnx, &chars)))).expect("load with typo");
        assert!(loaded.typo.is_some());
        let model = loaded.typo.unwrap();
        let embed = model.embed("hello").expect("embed through the pack");
        assert!((embed.iter().map(|v| v * v).sum::<f32>().sqrt() - 1.0).abs() < 1e-5);

        // packs without the pair are unchanged (None, no error)
        let plain = load(&sections_pack(None)).expect("load without typo");
        assert!(plain.typo.is_none());

        // a half pair rejects: the pair travels together
        let mut half = sections_pack(None);
        // append only typo-model (tag 6) after the frozen five
        use sha2::Digest as _;
        let count = u32::from_le_bytes([half[4], half[5], half[6], half[7]]) as usize + 1;
        half[4..8].copy_from_slice(&(count as u32).to_le_bytes());
        half.extend_from_slice(&(onnx.len() as u32).to_le_bytes());
        half.push(SectionTag::TypoModel as u8);
        half.extend_from_slice(&onnx);
        half.extend_from_slice(&Sha256::digest(&onnx));
        let error = load(&half).unwrap_err();
        assert!(error.to_string().contains("travels together"), "{error}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The checked-in real-derived fixtures (plan 85 / plan 103): a
    // 40-word truncation of the REAL registry en/mini tier and the
    // bucket rows the teh/catt/hotcold queries address — bytes onnx
    // actually wrote, not synthetic ones.
    const FIXTURE_MODEL: &[u8] = include_bytes!("../tests/fixtures/models/en-mini-truncated.onnx");
    const FIXTURE_VOCAB: &[u8] =
        include_bytes!("../tests/fixtures/models/en-mini-truncated.vocab.json");
    const FIXTURE_BUCKETS: &[u8] =
        include_bytes!("../tests/fixtures/models/en-buckets-truncated.onnx");

    /// A minimal real dictionary the engine accepts (the ffi tests use
    /// the same shape).
    const AFF: &[u8] = b"SET UTF-8\nTRY esianrtolcdugmphbyfvkwz\n";
    const DIC: &[u8] = b"3\nhello\nworld\nkotoshu\n";

    // The synthetic sections BOTH suites pin — this suite here and
    // tests/test_packs.py in kotoshu/models-fasttext-onnx. The whole-
    // file sha256 below is the framing contract: the Python builder and
    // this reader must produce/consume byte-identical packs.
    /// Per-section framing overhead: u32 LE length + tag byte + the
    /// sha256 footer.
    const SECTION_OVERHEAD: usize = 4 + 1 + 32;

    const GOLDEN_AFF: &[u8] = b"SET UTF-8\nTRY esianrtolcdugmphbyfvkwzESIANRTOLCDUGMPHBYFVKWZ\n";
    const GOLDEN_DIC: &[u8] = b"3\nhello/M\nworld\nkotoshu\n";
    const GOLDEN_VOCAB: &[u8] =
        b"{\"vocab_size\":3,\"word_to_idx\":{\"hello\":0,\"kotoshu\":1,\"world\":2}}";
    const GOLDEN_SHA256: &str = "5f124baa3fcae30625abb0e3297479ae5d05dad411c97d08e0e83f106f09843b";
    const GOLDEN_SIZE: usize = 898;

    fn golden_model() -> Vec<u8> {
        (0..64u8).cycle().take(256).collect()
    }

    fn golden_buckets() -> Vec<u8> {
        (0..300u16).map(|i| (i * 7) as u8).collect()
    }

    fn frame(tag: SectionTag, payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(payload.len() + SECTION_OVERHEAD);
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.push(tag as u8);
        out.extend_from_slice(payload);
        out.extend_from_slice(&Sha256::digest(payload));
        out
    }

    fn build(
        aff: &[u8],
        dic: &[u8],
        model: &[u8],
        vocab: &[u8],
        buckets: Option<&[u8]>,
    ) -> Vec<u8> {
        let sections = [
            (SectionTag::Aff, aff),
            (SectionTag::Dic, dic),
            (SectionTag::Model, model),
            (SectionTag::Vocab, vocab),
        ];
        let mut out = Vec::new();
        out.extend_from_slice(&MAGIC);
        let count = sections.len() + usize::from(buckets.is_some());
        out.extend_from_slice(&(count as u32).to_le_bytes());
        for (tag, payload) in sections {
            out.extend_from_slice(&frame(tag, payload));
        }
        if let Some(payload) = buckets {
            out.extend_from_slice(&frame(SectionTag::Buckets, payload));
        }
        out
    }

    fn fixture_pack() -> Vec<u8> {
        build(
            AFF,
            DIC,
            FIXTURE_MODEL,
            FIXTURE_VOCAB,
            Some(FIXTURE_BUCKETS),
        )
    }

    fn golden_pack() -> Vec<u8> {
        let model = golden_model();
        let buckets = golden_buckets();
        build(GOLDEN_AFF, GOLDEN_DIC, &model, GOLDEN_VOCAB, Some(&buckets))
    }

    #[test]
    fn golden_bytes_match_the_models_repo_builder() {
        let pack = golden_pack();
        assert_eq!(pack.len(), GOLDEN_SIZE);
        let hex = Sha256::digest(&pack);
        let mut string = String::with_capacity(64);
        for byte in hex {
            string.push_str(&format!("{byte:02x}"));
        }
        assert_eq!(string, GOLDEN_SHA256);
    }

    #[test]
    fn parses_fixture_pack_sections() {
        let pack = fixture_pack();
        let sections = parse(&pack).unwrap();
        assert_eq!(sections.aff, AFF);
        assert_eq!(sections.dic, DIC);
        assert_eq!(sections.model, FIXTURE_MODEL);
        assert_eq!(sections.vocab, FIXTURE_VOCAB);
        assert_eq!(sections.buckets, Some(FIXTURE_BUCKETS));
    }

    #[test]
    fn optional_buckets_absent() {
        let pack = build(AFF, DIC, FIXTURE_MODEL, FIXTURE_VOCAB, None);
        let sections = parse(&pack).unwrap();
        assert!(sections.buckets.is_none());
    }

    // - parity: pack handles == per-artifact handles -------------------

    /// The parity contract (plan 113): every handle and answer the pack
    /// path produces equals the per-artifact path over the same bytes.
    #[test]
    fn pack_loads_match_per_artifact_loads() {
        let pack = load(&fixture_pack()).unwrap();

        let dictionary = Dictionary::load_from_sources(
            std::str::from_utf8(AFF).unwrap(),
            std::str::from_utf8(DIC).unwrap(),
        )
        .unwrap();
        let mut model = Int8Model::parse(FIXTURE_MODEL, FIXTURE_VOCAB).unwrap();
        model.attach_buckets(FIXTURE_BUCKETS).unwrap();

        // Dictionary parity: the word list is the index; correctness and
        // suggestions follow from identical sources.
        assert_eq!(pack.dictionary.words(), dictionary.words());
        for word in ["hello", "world", "kotoshu", "helo", "wrld", ""] {
            assert_eq!(
                pack.dictionary.correct(word),
                dictionary.correct(word),
                "correct({word:?})"
            );
            assert_eq!(
                pack.dictionary.suggest(word, 5),
                dictionary.suggest(word, 5),
                "suggest({word:?})"
            );
        }

        // Model parity: structural shape, then behavior including the
        // bucket-backed OOV path the fixture exists to exercise.
        assert_eq!(pack.model.vocab_len(), model.vocab_len());
        assert_eq!(pack.model.dims(), model.dims());
        assert_eq!(
            pack.model.buckets().map(|t| t.len()),
            model.buckets().map(|t| t.len())
        );
        let vocab_words: Vec<String> = [
            "hello", "world", "cat", "dog", "computer", "teh", "catt", "hotcold",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        for word in &vocab_words {
            assert_eq!(
                pack.model.word_index(word),
                model.word_index(word),
                "word_index({word:?})"
            );
            assert_eq!(
                pack.model.embedding(word),
                model.embedding(word),
                "embedding({word:?})"
            );
            assert_eq!(
                pack.model.semantic_neighbors(word, 4),
                model.semantic_neighbors(word, 4),
                "semantic_neighbors({word:?})"
            );
        }
        for (word, context) in [
            ("cat", "the dog and the cat"),
            ("hello", "world"),
            ("teh", "hello teh world"),
        ] {
            assert_eq!(
                pack.model.context_score(word, context),
                model.context_score(word, context),
                "context_score({word:?}, {context:?})"
            );
        }
    }

    // - corruption: a bad pack can never half-load ---------------------

    #[test]
    fn rejects_bad_magic() {
        let mut pack = fixture_pack();
        pack[0] = b'X';
        let error = parse(&pack).unwrap_err();
        assert!(error.to_string().contains("magic"), "{error}");
    }

    #[test]
    fn rejects_truncated_header() {
        let error = parse(&fixture_pack()[..7]).unwrap_err();
        assert!(error.to_string().contains("header"), "{error}");
    }

    #[test]
    fn rejects_count_mismatch() {
        let mut pack = fixture_pack();
        pack[4] = 6; // six declared, five framed
        let error = parse(&pack).unwrap_err();
        assert!(error.to_string().contains("truncated"), "{error}");
    }

    #[test]
    fn rejects_out_of_order_sections() {
        // dic framed first: the fixed order is part of the format.
        let mut pack = Vec::new();
        pack.extend_from_slice(&MAGIC);
        pack.extend_from_slice(&1u32.to_le_bytes());
        pack.extend_from_slice(&frame(SectionTag::Dic, DIC));
        let error = parse(&pack).unwrap_err();
        assert!(error.to_string().contains("order"), "{error}");
    }

    #[test]
    fn rejects_tampered_payload() {
        let mut pack = fixture_pack();
        // Inside the model section payload: header(8) + aff + dic
        // framings, then 5 bytes into the model payload.
        let model_payload_at = HEADER_LEN
            + SECTION_OVERHEAD
            + AFF.len()
            + SECTION_OVERHEAD
            + DIC.len()
            + SECTION_OVERHEAD
            + 64;
        pack[model_payload_at] ^= 0xFF;
        let error = parse(&pack).unwrap_err();
        assert!(
            matches!(error, PackError::Checksum { section: "model" }),
            "{error}"
        );
    }

    #[test]
    fn rejects_truncated_payload() {
        let error = parse(&fixture_pack()[..fixture_pack().len() - 1]).unwrap_err();
        assert!(error.to_string().contains("truncated"), "{error}");
    }

    #[test]
    fn rejects_trailing_bytes() {
        let mut pack = fixture_pack();
        pack.push(0);
        let error = parse(&pack).unwrap_err();
        assert!(error.to_string().contains("trailing"), "{error}");
    }

    #[test]
    fn rejects_unknown_tag() {
        let mut pack = build(AFF, DIC, FIXTURE_MODEL, FIXTURE_VOCAB, None);
        // The first section framing starts at 8; its tag byte at 12.
        pack[12] = 9;
        let error = parse(&pack).unwrap_err();
        assert!(error.to_string().contains("unknown section tag"), "{error}");
    }

    #[test]
    fn rejects_missing_required_section() {
        // A pack of only the dic section (count 1) — aff is missing.
        let mut pack = Vec::new();
        pack.extend_from_slice(&MAGIC);
        pack.extend_from_slice(&1u32.to_le_bytes());
        pack.extend_from_slice(&frame(SectionTag::Dic, DIC));
        let error = parse(&pack).unwrap_err();
        assert!(matches!(error, PackError::Format(_)), "{error}");
    }

    #[test]
    fn rejects_non_utf8_dictionary_section() {
        let bad_aff: &[u8] = b"SET UTF-8\n\xff\xfe\n";
        let pack = build(bad_aff, DIC, FIXTURE_MODEL, FIXTURE_VOCAB, None);
        let error = load(&pack).unwrap_err();
        assert!(
            matches!(error, PackError::Utf8 { section: "aff" }),
            "{error}"
        );
    }

    #[test]
    fn rejects_broken_dictionary_sections() {
        // REP announces 2 entries but supplies 1 — a genuine LoadError
        // surfacing through the pack path.
        let pack = build(b"REP 2\nX\n", b"0\n", FIXTURE_MODEL, FIXTURE_VOCAB, None);
        let error = load(&pack).unwrap_err();
        assert!(matches!(error, PackError::Dictionary(_)), "{error}");
    }

    #[test]
    fn rejects_broken_model_sections() {
        // A model section that is not an ONNX protobuf at all.
        let pack = build(AFF, DIC, b"not an onnx model", FIXTURE_VOCAB, None);
        let error = load(&pack).unwrap_err();
        assert!(matches!(error, PackError::Model(_)), "{error}");
    }
}
