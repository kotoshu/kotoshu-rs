//! Batch serialization shared by ALL bindings (C ABI, Ruby, WASM, Python).
//!
//! Parsanol pattern: one serialization, every binding — measured 3-5x faster
//! than object-by-object FFI. Every host produces and consumes the same byte
//! format, so conformance vectors hold every port to identical outputs.
//!
//! # Wire format (version 1)
//!
//! All integers are little-endian. Strings are UTF-8, length-prefixed with a
//! `u32` byte count. Every buffer starts with the 4-byte magic `KOSH` and a
//! `u32` format version.
//!
//! Request:
//!
//! ```text
//! magic "KOSH", u32 version, u16 kind
//!   kind 1 (check):   str language, u32 n, n x str word
//!   kind 2 (suggest): str language, str word, u8 limit
//! ```
//!
//! Response:
//!
//! ```text
//! magic "KOSH", u32 version, u16 kind        (echoes the request kind)
//!   kind 1 (check):   u32 n, n x u8          (0 = miss, 1 = correct)
//!   kind 2 (suggest): u32 n, n x suggestion
//!     suggestion := str word, u8 distance, f32 confidence (LE bits),
//!                   u8 source (SuggestionSource discriminant)
//! ```

/// Magic prefix of every batch buffer.
pub const MAGIC: [u8; 4] = *b"KOSH";

/// Batch wire-format version. Bumped on any incompatible layout change.
pub const FORMAT_VERSION: u32 = 1;

/// Request/response kind: batch `correct?` lookup.
pub const KIND_CHECK: u16 = 1;
/// Request/response kind: ranked suggestions for one word.
pub const KIND_SUGGEST: u16 = 2;

/// Batch request. Mirrors the gem's `Kotoshu.correct?` / `Kotoshu.suggest`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Request {
    Check {
        language: String,
        words: Vec<String>,
    },
    Suggest {
        language: String,
        word: String,
        limit: u8,
    },
}

/// One ranked suggestion. Mirrors the gem's
/// `Kotoshu::Suggestions::Suggestion(word:, distance:, confidence:, source:)`.
#[derive(Debug, Clone, PartialEq)]
pub struct Suggestion {
    pub word: String,
    pub distance: u8,
    pub confidence: f32,
    pub source: SuggestionSource,
}

/// Which strategy produced a suggestion (gem `Suggestions::Strategies::*`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SuggestionSource {
    EditDistance = 0,
    Phonetic = 1,
    KeyboardProximity = 2,
    Ngram = 3,
    Semantic = 4,
}

impl SuggestionSource {
    fn from_discriminant(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::EditDistance),
            1 => Some(Self::Phonetic),
            2 => Some(Self::KeyboardProximity),
            3 => Some(Self::Ngram),
            4 => Some(Self::Semantic),
            _ => None,
        }
    }
}

/// Batch response.
#[derive(Debug, Clone, PartialEq)]
pub enum Response {
    Check { correct: Vec<bool> },
    Suggest { suggestions: Vec<Suggestion> },
}

/// Status codes shared with the C ABI ([`crate::ffi::c`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum Status {
    Ok = 0,
    NullPointer = 1,
    Truncated = 2,
    UnsupportedVersion = 3,
    UnknownKind = 4,
    InvalidUtf8 = 5,
}

/// Placeholder response: shape-correct, engine-empty.
///
/// TODO(P2): route `Check` through [`crate::dict::Dictionary`] (needs the
/// dictionary lifecycle on the C ABI — load/register/free calls keyed by
/// language or path — which lands with the `parallel` batch feature) and
/// `Suggest` through the P2 suggester. The P1 engine is exercised directly
/// by the conformance suite via `kotoshu::dict::Dictionary::correct`.
pub fn stub_response(request: &Request) -> Response {
    match request {
        Request::Check { words, .. } => Response::Check {
            correct: vec![false; words.len()],
        },
        Request::Suggest { .. } => Response::Suggest {
            suggestions: Vec::new(),
        },
    }
}

struct Writer(Vec<u8>);

impl Writer {
    fn new() -> Self {
        Self(Vec::new())
    }

    fn u8(&mut self, value: u8) {
        self.0.push(value);
    }

    fn u16(&mut self, value: u16) {
        self.0.extend_from_slice(&value.to_le_bytes());
    }

    fn u32(&mut self, value: u32) {
        self.0.extend_from_slice(&value.to_le_bytes());
    }

    fn f32(&mut self, value: f32) {
        self.0.extend_from_slice(&value.to_bits().to_le_bytes());
    }

    fn str(&mut self, value: &str) {
        self.u32(value.len() as u32);
        self.0.extend_from_slice(value.as_bytes());
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl Reader<'_> {
    fn new(bytes: &[u8]) -> Reader<'_> {
        Reader { bytes, pos: 0 }
    }

    fn take(&mut self, len: usize) -> Result<&[u8], Status> {
        let end = self.pos.checked_add(len).ok_or(Status::Truncated)?;
        let slice = self.bytes.get(self.pos..end).ok_or(Status::Truncated)?;
        self.pos = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8, Status> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, Status> {
        let bytes = self.take(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn u32(&mut self) -> Result<u32, Status> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn f32(&mut self) -> Result<f32, Status> {
        let bytes = self.take(4)?;
        Ok(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn str(&mut self) -> Result<String, Status> {
        let len = self.u32()? as usize;
        let bytes = self.take(len)?;
        std::str::from_utf8(bytes)
            .map(|s| s.to_owned())
            .map_err(|_| Status::InvalidUtf8)
    }

    fn header(&mut self) -> Result<u16, Status> {
        if self.take(4)? != MAGIC {
            return Err(Status::Truncated);
        }
        let version = self.u32()?;
        if version != FORMAT_VERSION {
            return Err(Status::UnsupportedVersion);
        }
        self.u16()
    }
}

fn header(kind: u16) -> Writer {
    let mut writer = Writer::new();
    writer.0.extend_from_slice(&MAGIC);
    writer.u32(FORMAT_VERSION);
    writer.u16(kind);
    writer
}

/// Serialize a request to the batch wire format.
pub fn encode_request(request: &Request) -> Vec<u8> {
    let writer = match request {
        Request::Check { language, words } => {
            let mut writer = header(KIND_CHECK);
            writer.str(language);
            writer.u32(words.len() as u32);
            for word in words {
                writer.str(word);
            }
            writer
        }
        Request::Suggest {
            language,
            word,
            limit,
        } => {
            let mut writer = header(KIND_SUGGEST);
            writer.str(language);
            writer.str(word);
            writer.u8(*limit);
            writer
        }
    };
    writer.0
}

/// Deserialize a request from the batch wire format.
pub fn decode_request(bytes: &[u8]) -> Result<Request, Status> {
    let mut reader = Reader::new(bytes);
    let kind = reader.header()?;
    let request = match kind {
        KIND_CHECK => {
            let language = reader.str()?;
            let count = reader.u32()? as usize;
            let mut words = Vec::with_capacity(count);
            for _ in 0..count {
                words.push(reader.str()?);
            }
            Request::Check { language, words }
        }
        KIND_SUGGEST => {
            let language = reader.str()?;
            let word = reader.str()?;
            let limit = reader.u8()?;
            Request::Suggest {
                language,
                word,
                limit,
            }
        }
        _ => return Err(Status::UnknownKind),
    };
    Ok(request)
}

/// Serialize a response to the batch wire format.
pub fn encode_response(response: &Response) -> Vec<u8> {
    let writer = match response {
        Response::Check { correct } => {
            let mut writer = header(KIND_CHECK);
            writer.u32(correct.len() as u32);
            for is_correct in correct {
                writer.u8(u8::from(*is_correct));
            }
            writer
        }
        Response::Suggest { suggestions } => {
            let mut writer = header(KIND_SUGGEST);
            writer.u32(suggestions.len() as u32);
            for suggestion in suggestions {
                writer.str(&suggestion.word);
                writer.u8(suggestion.distance);
                writer.f32(suggestion.confidence);
                writer.u8(suggestion.source as u8);
            }
            writer
        }
    };
    writer.0
}

/// Deserialize a response from the batch wire format.
pub fn decode_response(bytes: &[u8]) -> Result<Response, Status> {
    let mut reader = Reader::new(bytes);
    let kind = reader.header()?;
    let response = match kind {
        KIND_CHECK => {
            let count = reader.u32()? as usize;
            let mut correct = Vec::with_capacity(count);
            for _ in 0..count {
                correct.push(reader.u8()? != 0);
            }
            Response::Check { correct }
        }
        KIND_SUGGEST => {
            let count = reader.u32()? as usize;
            let mut suggestions = Vec::with_capacity(count);
            for _ in 0..count {
                let word = reader.str()?;
                let distance = reader.u8()?;
                let confidence = reader.f32()?;
                let source =
                    SuggestionSource::from_discriminant(reader.u8()?).ok_or(Status::UnknownKind)?;
                suggestions.push(Suggestion {
                    word,
                    distance,
                    confidence,
                    source,
                });
            }
            Response::Suggest { suggestions }
        }
        _ => return Err(Status::UnknownKind),
    };
    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_request() -> Request {
        Request::Check {
            language: "en".to_owned(),
            words: vec!["hello".to_owned(), "recieve".to_owned()],
        }
    }

    fn suggest_request() -> Request {
        Request::Suggest {
            language: "en".to_owned(),
            word: "recieve".to_owned(),
            limit: 5,
        }
    }

    #[test]
    fn request_check_round_trips() {
        let request = check_request();
        assert_eq!(decode_request(&encode_request(&request)).unwrap(), request);
    }

    #[test]
    fn request_suggest_round_trips() {
        let request = suggest_request();
        assert_eq!(decode_request(&encode_request(&request)).unwrap(), request);
    }

    #[test]
    fn response_check_round_trips() {
        let response = Response::Check {
            correct: vec![true, false],
        };
        assert_eq!(
            decode_response(&encode_response(&response)).unwrap(),
            response
        );
    }

    #[test]
    fn response_suggest_round_trips() {
        let response = Response::Suggest {
            suggestions: vec![Suggestion {
                word: "receive".to_owned(),
                distance: 1,
                confidence: 0.95,
                source: SuggestionSource::EditDistance,
            }],
        };
        assert_eq!(
            decode_response(&encode_response(&response)).unwrap(),
            response
        );
    }

    #[test]
    fn stub_response_round_trips_with_matching_shape() {
        let request = check_request();
        let stub = stub_response(&request);
        let Response::Check { correct } = decode_response(&encode_response(&stub)).unwrap() else {
            panic!("expected check response");
        };
        assert_eq!(correct, vec![false, false]);
    }

    #[test]
    fn rejects_bad_magic() {
        assert_eq!(decode_request(b"NOPE"), Err(Status::Truncated));
    }

    #[test]
    fn rejects_unsupported_version() {
        let mut bytes = encode_request(&check_request());
        bytes[4] = 9;
        assert_eq!(decode_request(&bytes), Err(Status::UnsupportedVersion));
    }

    #[test]
    fn rejects_truncated_buffer() {
        let bytes = encode_request(&check_request());
        assert_eq!(
            decode_request(&bytes[..bytes.len() - 1]),
            Err(Status::Truncated)
        );
    }
}
