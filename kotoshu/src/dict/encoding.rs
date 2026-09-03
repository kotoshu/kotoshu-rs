//! Encoding detection and line decoding for `.aff`/`.dic` files,
//! mirroring the gem's `AffReader#detect_encoding` + `FileReader`.
//!
//! Supported encodings are the three that appear in the fixture corpus:
//! UTF-8, ISO-8859-1 and ISO-8859-15 (the gem delegates to Ruby's encoding
//! machinery; everything else would fail there and the corpus would have
//! been skipped at export time).

/// File encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    /// `SET UTF-8`, or valid UTF-8 bytes with no `SET` line.
    Utf8,
    /// `SET ISO-8859-1` (or `ISO8859-1`), or the Latin-1 fallback for
    /// invalid UTF-8.
    Latin1,
    /// `SET ISO-8859-15` (or `ISO8859-15`).
    Latin15,
    /// `SET ISO-8859-2` (or `ISO8859-2`) — Central European.
    Latin2,
}

/// Detect the encoding of an `.aff` file's bytes: the `SET` directive wins
/// (line-anchored, first match), then UTF-8 validity, then Latin-1.
pub fn detect(aff_bytes: &[u8]) -> Result<Encoding, String> {
    for line in aff_bytes.split(|&b| b == b'\n') {
        if let Some(name) = set_directive_value(line) {
            return normalize(name);
        }
    }
    if std::str::from_utf8(aff_bytes).is_ok() {
        Ok(Encoding::Utf8)
    } else {
        Ok(Encoding::Latin1)
    }
}

/// Extract the value of a `SET` directive from one raw line
/// (Ruby `/^SET\s+(\S+)/`).
fn set_directive_value(line: &[u8]) -> Option<&[u8]> {
    let rest = line.strip_prefix(b"SET")?;
    let mut idx = 0;
    while idx < rest.len() && is_ruby_space(rest[idx]) {
        idx += 1;
    }
    if idx == 0 {
        return None; // `\s+` requires at least one space.
    }
    let value = &rest[idx..];
    let end = value
        .iter()
        .position(|&b| is_ruby_space(b))
        .unwrap_or(value.len());
    if end == 0 {
        return None;
    }
    Some(&value[..end])
}

fn is_ruby_space(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\r' | b'\x0B' | b'\x0C')
}

/// Normalize a `SET` value (the gem's `normalize_encoding_name`).
fn normalize(name: &[u8]) -> Result<Encoding, String> {
    let name = std::str::from_utf8(name).map_err(|_| "SET value is not UTF-8".to_owned())?;
    if name.to_uppercase() == "UTF-8" {
        return Ok(Encoding::Utf8);
    }
    let normalized: String = name.to_uppercase().replace('-', "");
    if let Some(rest) = normalized.strip_prefix("ISO8859") {
        return match rest {
            "1" => Ok(Encoding::Latin1),
            "2" => Ok(Encoding::Latin2),
            "15" => Ok(Encoding::Latin15),
            other => Err(format!("unsupported SET encoding: ISO-8859-{other}")),
        };
    }
    Err(format!("unsupported SET encoding: {name}"))
}

/// Decode the file into its lines: each line stripped (Ruby `String#strip`
/// — ASCII whitespace and NUL), empty lines skipped, a leading UTF-8 BOM
/// removed from the first line. Lines invalid in the declared encoding
/// fall back to Latin-1, as the gem does per line.
pub fn decode_lines(bytes: &[u8], encoding: Encoding) -> Vec<String> {
    let mut lines = Vec::new();
    for (raw_idx, raw) in bytes.split(|&b| b == b'\n').enumerate() {
        let mut line = decode_line(raw, encoding);
        if raw_idx == 0 {
            // The gem compares against the raw BOM bytes, so a BOM is
            // stripped even when the surrounding encoding is Latin-1
            // (where it decodes to `ï»¿`).
            if let Some(stripped) = line.strip_prefix('\u{FEFF}') {
                line = stripped.to_owned();
            } else if let Some(stripped) = line.strip_prefix("ï»¿") {
                line = stripped.to_owned();
            }
        }
        let line = ruby_strip(&line);
        if line.is_empty() {
            continue;
        }
        lines.push(line.to_owned());
    }
    lines
}

/// Decode one raw line; invalid UTF-8 falls back to Latin-1 (every byte is
/// a valid Latin-1 codepoint, so this never fails).
fn decode_line(raw: &[u8], encoding: Encoding) -> String {
    match encoding {
        Encoding::Utf8 => match std::str::from_utf8(raw) {
            Ok(line) => line.to_owned(),
            Err(_) => decode_latin(raw, Encoding::Latin1),
        },
        other => decode_latin(raw, other),
    }
}

fn decode_latin(raw: &[u8], encoding: Encoding) -> String {
    raw.iter()
        .map(|&b| match (encoding, b) {
            (Encoding::Latin15, 0xA4) => '\u{20AC}', // €
            (Encoding::Latin15, 0xA6) => '\u{0160}', // Š
            (Encoding::Latin15, 0xA8) => '\u{0161}', // š
            (Encoding::Latin15, 0xB4) => '\u{017D}', // Ž
            (Encoding::Latin15, 0xB8) => '\u{017E}', // ž
            (Encoding::Latin15, 0xBC) => '\u{0152}', // Œ
            (Encoding::Latin15, 0xBD) => '\u{0153}', // œ
            (Encoding::Latin15, 0xBE) => '\u{0178}', // Ÿ
            (Encoding::Latin2, 0xA0) => char::from_u32(0x00A0).unwrap(),
            (Encoding::Latin2, 0xA1) => char::from_u32(0x0104).unwrap(),
            (Encoding::Latin2, 0xA2) => char::from_u32(0x02D8).unwrap(),
            (Encoding::Latin2, 0xA3) => char::from_u32(0x0141).unwrap(),
            (Encoding::Latin2, 0xA4) => char::from_u32(0x00A4).unwrap(),
            (Encoding::Latin2, 0xA5) => char::from_u32(0x013D).unwrap(),
            (Encoding::Latin2, 0xA6) => char::from_u32(0x015A).unwrap(),
            (Encoding::Latin2, 0xA7) => char::from_u32(0x00A7).unwrap(),
            (Encoding::Latin2, 0xA8) => char::from_u32(0x00A8).unwrap(),
            (Encoding::Latin2, 0xA9) => char::from_u32(0x0160).unwrap(),
            (Encoding::Latin2, 0xAA) => char::from_u32(0x015E).unwrap(),
            (Encoding::Latin2, 0xAB) => char::from_u32(0x0164).unwrap(),
            (Encoding::Latin2, 0xAC) => char::from_u32(0x0179).unwrap(),
            (Encoding::Latin2, 0xAD) => char::from_u32(0x00AD).unwrap(),
            (Encoding::Latin2, 0xAE) => char::from_u32(0x017D).unwrap(),
            (Encoding::Latin2, 0xAF) => char::from_u32(0x017B).unwrap(),
            (Encoding::Latin2, 0xB0) => char::from_u32(0x00B0).unwrap(),
            (Encoding::Latin2, 0xB1) => char::from_u32(0x0105).unwrap(),
            (Encoding::Latin2, 0xB2) => char::from_u32(0x02DB).unwrap(),
            (Encoding::Latin2, 0xB3) => char::from_u32(0x0142).unwrap(),
            (Encoding::Latin2, 0xB4) => char::from_u32(0x00B4).unwrap(),
            (Encoding::Latin2, 0xB5) => char::from_u32(0x013E).unwrap(),
            (Encoding::Latin2, 0xB6) => char::from_u32(0x015B).unwrap(),
            (Encoding::Latin2, 0xB7) => char::from_u32(0x02C7).unwrap(),
            (Encoding::Latin2, 0xB8) => char::from_u32(0x00B8).unwrap(),
            (Encoding::Latin2, 0xB9) => char::from_u32(0x0161).unwrap(),
            (Encoding::Latin2, 0xBA) => char::from_u32(0x015F).unwrap(),
            (Encoding::Latin2, 0xBB) => char::from_u32(0x0165).unwrap(),
            (Encoding::Latin2, 0xBC) => char::from_u32(0x017A).unwrap(),
            (Encoding::Latin2, 0xBD) => char::from_u32(0x02DD).unwrap(),
            (Encoding::Latin2, 0xBE) => char::from_u32(0x017E).unwrap(),
            (Encoding::Latin2, 0xBF) => char::from_u32(0x017C).unwrap(),
            (Encoding::Latin2, 0xC0) => char::from_u32(0x0154).unwrap(),
            (Encoding::Latin2, 0xC1) => char::from_u32(0x00C1).unwrap(),
            (Encoding::Latin2, 0xC2) => char::from_u32(0x00C2).unwrap(),
            (Encoding::Latin2, 0xC3) => char::from_u32(0x0102).unwrap(),
            (Encoding::Latin2, 0xC4) => char::from_u32(0x00C4).unwrap(),
            (Encoding::Latin2, 0xC5) => char::from_u32(0x0139).unwrap(),
            (Encoding::Latin2, 0xC6) => char::from_u32(0x0106).unwrap(),
            (Encoding::Latin2, 0xC7) => char::from_u32(0x00C7).unwrap(),
            (Encoding::Latin2, 0xC8) => char::from_u32(0x010C).unwrap(),
            (Encoding::Latin2, 0xC9) => char::from_u32(0x00C9).unwrap(),
            (Encoding::Latin2, 0xCA) => char::from_u32(0x0118).unwrap(),
            (Encoding::Latin2, 0xCB) => char::from_u32(0x00CB).unwrap(),
            (Encoding::Latin2, 0xCC) => char::from_u32(0x011A).unwrap(),
            (Encoding::Latin2, 0xCD) => char::from_u32(0x00CD).unwrap(),
            (Encoding::Latin2, 0xCE) => char::from_u32(0x00CE).unwrap(),
            (Encoding::Latin2, 0xCF) => char::from_u32(0x010E).unwrap(),
            (Encoding::Latin2, 0xD0) => char::from_u32(0x0110).unwrap(),
            (Encoding::Latin2, 0xD1) => char::from_u32(0x0143).unwrap(),
            (Encoding::Latin2, 0xD2) => char::from_u32(0x0147).unwrap(),
            (Encoding::Latin2, 0xD3) => char::from_u32(0x00D3).unwrap(),
            (Encoding::Latin2, 0xD4) => char::from_u32(0x00D4).unwrap(),
            (Encoding::Latin2, 0xD5) => char::from_u32(0x0150).unwrap(),
            (Encoding::Latin2, 0xD6) => char::from_u32(0x00D6).unwrap(),
            (Encoding::Latin2, 0xD7) => char::from_u32(0x00D7).unwrap(),
            (Encoding::Latin2, 0xD8) => char::from_u32(0x0158).unwrap(),
            (Encoding::Latin2, 0xD9) => char::from_u32(0x016E).unwrap(),
            (Encoding::Latin2, 0xDA) => char::from_u32(0x00DA).unwrap(),
            (Encoding::Latin2, 0xDB) => char::from_u32(0x0170).unwrap(),
            (Encoding::Latin2, 0xDC) => char::from_u32(0x00DC).unwrap(),
            (Encoding::Latin2, 0xDD) => char::from_u32(0x00DD).unwrap(),
            (Encoding::Latin2, 0xDE) => char::from_u32(0x0162).unwrap(),
            (Encoding::Latin2, 0xDF) => char::from_u32(0x00DF).unwrap(),
            (Encoding::Latin2, 0xE0) => char::from_u32(0x0155).unwrap(),
            (Encoding::Latin2, 0xE1) => char::from_u32(0x00E1).unwrap(),
            (Encoding::Latin2, 0xE2) => char::from_u32(0x00E2).unwrap(),
            (Encoding::Latin2, 0xE3) => char::from_u32(0x0103).unwrap(),
            (Encoding::Latin2, 0xE4) => char::from_u32(0x00E4).unwrap(),
            (Encoding::Latin2, 0xE5) => char::from_u32(0x013A).unwrap(),
            (Encoding::Latin2, 0xE6) => char::from_u32(0x0107).unwrap(),
            (Encoding::Latin2, 0xE7) => char::from_u32(0x00E7).unwrap(),
            (Encoding::Latin2, 0xE8) => char::from_u32(0x010D).unwrap(),
            (Encoding::Latin2, 0xE9) => char::from_u32(0x00E9).unwrap(),
            (Encoding::Latin2, 0xEA) => char::from_u32(0x0119).unwrap(),
            (Encoding::Latin2, 0xEB) => char::from_u32(0x00EB).unwrap(),
            (Encoding::Latin2, 0xEC) => char::from_u32(0x011B).unwrap(),
            (Encoding::Latin2, 0xED) => char::from_u32(0x00ED).unwrap(),
            (Encoding::Latin2, 0xEE) => char::from_u32(0x00EE).unwrap(),
            (Encoding::Latin2, 0xEF) => char::from_u32(0x010F).unwrap(),
            (Encoding::Latin2, 0xF0) => char::from_u32(0x0111).unwrap(),
            (Encoding::Latin2, 0xF1) => char::from_u32(0x0144).unwrap(),
            (Encoding::Latin2, 0xF2) => char::from_u32(0x0148).unwrap(),
            (Encoding::Latin2, 0xF3) => char::from_u32(0x00F3).unwrap(),
            (Encoding::Latin2, 0xF4) => char::from_u32(0x00F4).unwrap(),
            (Encoding::Latin2, 0xF5) => char::from_u32(0x0151).unwrap(),
            (Encoding::Latin2, 0xF6) => char::from_u32(0x00F6).unwrap(),
            (Encoding::Latin2, 0xF7) => char::from_u32(0x00F7).unwrap(),
            (Encoding::Latin2, 0xF8) => char::from_u32(0x0159).unwrap(),
            (Encoding::Latin2, 0xF9) => char::from_u32(0x016F).unwrap(),
            (Encoding::Latin2, 0xFA) => char::from_u32(0x00FA).unwrap(),
            (Encoding::Latin2, 0xFB) => char::from_u32(0x0171).unwrap(),
            (Encoding::Latin2, 0xFC) => char::from_u32(0x00FC).unwrap(),
            (Encoding::Latin2, 0xFD) => char::from_u32(0x00FD).unwrap(),
            (Encoding::Latin2, 0xFE) => char::from_u32(0x0163).unwrap(),
            (Encoding::Latin2, 0xFF) => char::from_u32(0x02D9).unwrap(),
            _ => b as char,
        })
        .collect()
}

/// Ruby `String#strip`: removes leading/trailing ASCII whitespace and NUL.
fn ruby_strip(line: &str) -> &str {
    line.trim_matches(|c: char| matches!(c, '\0' | '\t' | '\n' | '\x0B' | '\x0C' | '\r' | ' '))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_set_utf8() {
        assert_eq!(detect(b"TRY abc\nSET UTF-8\n"), Ok(Encoding::Utf8));
    }

    #[test]
    fn detects_set_latin15_aliases() {
        assert_eq!(detect(b"SET ISO8859-15\n"), Ok(Encoding::Latin15));
        assert_eq!(detect(b"SET ISO-8859-15\n"), Ok(Encoding::Latin15));
    }

    #[test]
    fn falls_back_to_latin1_for_invalid_utf8() {
        assert_eq!(
            detect(b"SFX A 0 x .\nSFX A \xE9\xE9 x \xE9\n"),
            Ok(Encoding::Latin1)
        );
    }

    #[test]
    fn decodes_latin15_euro() {
        // The word-count line is a dic-parse concern, not the decoder's.
        let lines = decode_lines(b"1\nfu\xA4r\n", Encoding::Latin15);
        assert_eq!(lines, vec!["1".to_owned(), "fu\u{20AC}r".to_owned()]);
    }

    #[test]
    fn strips_bom_from_first_line_only() {
        let lines = decode_lines("\u{FEFF}SET UTF-8\n".as_bytes(), Encoding::Utf8);
        assert_eq!(lines, vec!["SET UTF-8".to_owned()]);
    }
}
