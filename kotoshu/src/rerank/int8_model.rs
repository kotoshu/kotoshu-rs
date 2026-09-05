//! Pure-Rust reader + scorer for the int8-per-row embedding tiers
//! (feature `model`) — plan 85, the wasm-side sibling of the `onnx`
//! feature's `rerank::onnx` provider.
//!
//! The `onnx` feature's provider runs inference through `ort`, which is
//! `load-dynamic` by policy — a host must supply `libonnxruntime`, which
//! no browser can. But the quantized tiers (`kotoshu://models/{lang}/
//! mini|fluency`) do not need a runtime at all: the graph is two
//! constants (`q_embeddings` int8 `[V, d]`, `row_scale` fp32 `[V]`) and
//! fixed gather/dequant plumbing. This module walks the ONNX protobuf
//! directly — a minimal wire-format reader (varints, length-delimited
//! fields, packed scalars; no schema, no third-party parser) — copies
//! the two tensors plus the metadata out, and scores words with the
//! same math the graph performs: `value = q as f32 * row_scale`
//! ([`super::dequant`]) then [`super::cosine`].
//!
//! The artifacts' exact serialization (verified against the registry
//! v1.0.1 `en/mini` release asset): the tensors ride **Constant nodes**
//! (`op_type = "Constant"`, attribute `value` → TensorProto), with the
//! payloads in `raw_data` — `onnx.numpy_helper.from_array` does not use
//! the typed repeated fields. Both storages are accepted here, plus
//! graph **initializers**, so a re-serialized-but-equivalent artifact
//! still loads.
//!
//! # Memory
//!
//! `Int8Model` owns `V*d` bytes of int8 + `V` f32 scales + the vocab
//! map — approximately the tier file's own size (mini ≈ 3 MB, fluency
//! ≈ 15 MB). Rows dequantize on demand into a `d`-sized `Vec<f32>`
//! (300 dims ≈ 1.2 KB); the full fp32 matrix is never materialized.
//! The struct frees by ordinary `Drop` (its `Vec`s), so a wasm handle
//! releases the whole footprint deterministically on `.free()` and on
//! garbage collection.

use std::collections::HashMap;
use std::fmt;

use super::dequant::{RowFormat, dequant_row_int8};
use super::{EmbeddingProvider, cosine, oov};

/// Free-text context cap for [`Int8Model::context_score`] — the gem
/// windows at ±3 words around the misspelling; free text has no anchor,
/// so the first 32 tokens bound the work instead (playground sentences
/// are far shorter).
pub const CONTEXT_TOKEN_CAP: usize = 32;

/// Errors of the int8 tier reader.
#[derive(Debug)]
pub enum Int8ModelError {
    /// The `.onnx` bytes are not a parsable ONNX protobuf, or are
    /// structurally unexpected (missing tensors, bad shapes, unsafe
    /// vocabulary rows).
    Onnx(String),
    /// The `.vocab.json` bytes could not be parsed or disagree with the
    /// matrix.
    Vocab(String),
    /// The graph metadata is missing or malformed.
    Metadata(String),
    /// The tier's quantization descriptor is one this reader does not
    /// handle (the fp32 `full` tier, the future int4 artifacts).
    UnsupportedQuantization(String),
}

impl fmt::Display for Int8ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Onnx(source) => write!(f, "int8 tier: {source}"),
            Self::Vocab(source) => write!(f, "int8 tier vocabulary: {source}"),
            Self::Metadata(source) => write!(f, "int8 tier metadata: {source}"),
            Self::UnsupportedQuantization(descriptor) => write!(
                f,
                "int8 tier quantization {descriptor:?} not readable without a \
                 runtime (known: int8-per-row)"
            ),
        }
    }
}

impl std::error::Error for Int8ModelError {}

/// One loaded int8-per-row tier: quantized matrix, row scales, and the
/// vocabulary — everything the dequantizing graph would need, with the
/// session omitted.
#[derive(Debug)]
pub struct Int8Model {
    /// Quantized rows, row-major `[V, d]`.
    q: Vec<i8>,
    /// Per-row dequantization scale `[V]` (the graph's `row_scale`).
    scales: Vec<f32>,
    /// Word → row index (validated `< V` at load; a bad row would read
    /// out of bounds at score time, so it is a load error, not a score
    /// error).
    vocab: HashMap<String, u32>,
    dims: usize,
}

/// The `.vocab.json` shape the models repo writes (`{"vocab_size": N,
/// "word_to_idx": {word: row}}`) — the same contract as the `onnx`
/// feature's reader; kept local so the two features stay independently
/// attachable.
#[derive(serde::Deserialize)]
struct VocabFile {
    #[serde(default)]
    vocab_size: Option<usize>,
    #[serde(default)]
    word_to_idx: Option<HashMap<String, u32>>,
}

impl Int8Model {
    /// Parse an int8-per-row tier: `onnx_bytes` is the `.onnx` artifact,
    /// `vocab_bytes` its `.vocab.json` sibling, exactly as distributed.
    pub fn parse(onnx_bytes: &[u8], vocab_bytes: &[u8]) -> Result<Self, Int8ModelError> {
        let onnx = parse_onnx(onnx_bytes)?;
        let dims = onnx.dims;

        let vocab_file: VocabFile = serde_json::from_slice(vocab_bytes)
            .map_err(|error| Int8ModelError::Vocab(error.to_string()))?;
        let raw = vocab_file.word_to_idx.unwrap_or_default();
        if raw.is_empty() {
            return Err(Int8ModelError::Vocab("empty vocabulary".to_owned()));
        }
        if let Some(declared) = vocab_file.vocab_size
            && declared != raw.len()
        {
            return Err(Int8ModelError::Vocab(format!(
                "vocab_size {declared} disagrees with {} entries",
                raw.len()
            )));
        }
        let rows = onnx.rows;
        if let Some((word, row)) = raw.iter().find(|(_, row)| {
            let row = usize::try_from(**row).unwrap_or(usize::MAX);
            row >= rows
        }) {
            return Err(Int8ModelError::Vocab(format!(
                "word {word:?} maps to row {row}, out of range for {rows} rows"
            )));
        }

        Ok(Self {
            q: onnx.q,
            scales: onnx.scales,
            vocab: raw,
            dims,
        })
    }

    /// Vector dimensionality.
    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Vocabulary entry count.
    pub fn vocab_len(&self) -> usize {
        self.vocab.len()
    }

    /// The row index of `word`, if it is in vocabulary.
    pub fn word_index(&self, word: &str) -> Option<u32> {
        self.vocab.get(word).copied()
    }

    /// The dequantized vector of `word` (the graph's `Cast + Mul`),
    /// or `None` when the word is out of vocabulary.
    pub fn embedding(&self, word: &str) -> Option<Vec<f32>> {
        let row = usize::try_from(self.word_index(word)?).ok()?;
        let start = row.checked_mul(self.dims)?;
        let end = start.checked_add(self.dims)?;
        let scale = *self.scales.get(row)?;
        Some(dequant_row_int8(self.q.get(start..end)?, scale))
    }

    /// Mean cosine of `word` against the in-vocabulary tokens of
    /// `context` — the wasm rerank score.
    ///
    /// Shape: the gem's `context_boost` is `0.02 × Σ cosine` over a ±3
    /// word window, added to a suggestion's confidence; per fixed
    /// context that is a positive multiple of the mean, so ranking by
    /// this score orders candidates exactly as the gem's boost does.
    /// The mean (not the sum) keeps scores comparable across contexts
    /// of different length and bounded in `[-1, 1]`. Out-of-vocabulary
    /// words — either side — contribute nothing (`0.0` overall when the
    /// word or every context token is OOV), mirroring the gem's
    /// `(sim || 0.0)`.
    pub fn context_score(&self, word: &str, context: &str) -> f32 {
        let Some(vector) = super::lookup(self, word) else {
            return 0.0;
        };
        let mut sum = 0.0f64;
        let mut count = 0u32;
        for token in super::context_tokens(context, CONTEXT_TOKEN_CAP) {
            if let Some(context_vector) = super::lookup(self, &token) {
                sum += cosine(&vector, &context_vector);
                count += 1;
            }
        }
        if count == 0 {
            0.0
        } else {
            (sum / f64::from(count)) as f32
        }
    }
}

impl EmbeddingProvider for Int8Model {
    fn embedding(&self, word: &str) -> Option<Vec<f32>> {
        Int8Model::embedding(self, word)
    }

    fn dims(&self) -> usize {
        self.dims
    }

    fn embedding_oov(&self, word: &str) -> Option<Vec<f32>> {
        // Same honest fallback as the ort provider (B2, plan 68).
        oov::substring_ngram_embedding(word, self)
    }
}

// --- The minimal ONNX protobuf reader -----------------------------------
//
// Protobuf wire format, the parts ONNX uses: each field is a varint tag
// `(number << 3) | wire_type` followed by its payload — varint (wire 0),
// 64-bit (1), length-delimited (2), 32-bit (5). `int32_data`/`dims` pack
// into a length-delimited run of varints; `float_data` packs into a run
// of little-endian fixed32s. Groups (wire 3/4) are long-obsolete in
// ONNX and rejected.

/// ONNX `TensorProto.DataType` values this reader names.
const DATA_TYPE_FLOAT: i32 = 1;
const DATA_TYPE_INT8: i32 = 3;

/// Field numbers of the protos walked (onnx/onnx.proto).
mod field {
    pub const MODEL_GRAPH: u32 = 7;
    pub const MODEL_METADATA: u32 = 14;

    pub const GRAPH_NODE: u32 = 1;
    pub const GRAPH_INITIALIZER: u32 = 5;

    pub const NODE_OP_TYPE: u32 = 4;
    pub const NODE_ATTRIBUTE: u32 = 5;

    pub const ATTRIBUTE_NAME: u32 = 1;
    pub const ATTRIBUTE_TENSOR: u32 = 5;

    pub const TENSOR_DIMS: u32 = 1;
    pub const TENSOR_DATA_TYPE: u32 = 2;
    pub const TENSOR_FLOAT_DATA: u32 = 4;
    pub const TENSOR_INT32_DATA: u32 = 5;
    pub const TENSOR_NAME: u32 = 8;
    pub const TENSOR_RAW_DATA: u32 = 9;

    pub const METADATA_KEY: u32 = 1;
    pub const METADATA_VALUE: u32 = 2;
}

/// Forward-only protobuf cursor over one message body.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn done(&self) -> bool {
        self.pos >= self.buf.len()
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], String> {
        let end = self.pos.checked_add(len).ok_or("field length overflow")?;
        let slice = self
            .buf
            .get(self.pos..end)
            .ok_or_else(|| format!("message truncated at byte {}", self.buf.len()))?;
        self.pos = end;
        Ok(slice)
    }

    fn varint(&mut self) -> Result<u64, String> {
        let mut value = 0u64;
        let mut shift = 0u32;
        loop {
            let byte = *self
                .buf
                .get(self.pos)
                .ok_or_else(|| format!("varint truncated at byte {}", self.buf.len()))?;
            self.pos += 1;
            value |= u64::from(byte & 0x7f).checked_shl(shift).unwrap_or(0);
            if byte & 0x80 == 0 {
                return Ok(value);
            }
            shift += 7;
            if shift >= 64 {
                return Err("varint longer than 10 bytes".to_owned());
            }
        }
    }

    /// Next field tag: `(number, wire_type)`.
    fn tag(&mut self) -> Result<(u32, u8), String> {
        let tag = self.varint()?;
        let number = u32::try_from(tag >> 3).map_err(|_| "field number overflow")?;
        let wire = (tag & 0x7) as u8;
        Ok((number, wire))
    }

    /// Length-delimited payload (wire 2).
    fn bytes(&mut self) -> Result<&'a [u8], String> {
        let len = usize::try_from(self.varint()?).map_err(|_| "length-delimited size overflow")?;
        self.take(len)
    }

    /// Skip a field of the given wire type (groups are rejected).
    fn skip(&mut self, wire: u8) -> Result<(), String> {
        match wire {
            0 => {
                self.varint()?;
            }
            1 => {
                self.take(8)?;
            }
            2 => {
                self.bytes()?;
            }
            5 => {
                self.take(4)?;
            }
            other => return Err(format!("unsupported wire type {other}")),
        }
        Ok(())
    }
}

/// A borrowed UTF-8 field.
fn text(bytes: &[u8]) -> Result<&str, String> {
    std::str::from_utf8(bytes).map_err(|_| "string field is not UTF-8".to_owned())
}

/// One decoded `StringStringEntryProto`.
fn metadata_entry(buf: &[u8]) -> Result<(String, String), String> {
    let mut reader = Reader::new(buf);
    let mut key = None;
    let mut value = None;
    while !reader.done() {
        let (number, wire) = reader.tag()?;
        match (number, wire) {
            (field::METADATA_KEY, 2) => key = Some(text(reader.bytes()?)?.to_owned()),
            (field::METADATA_VALUE, 2) => value = Some(text(reader.bytes()?)?.to_owned()),
            _ => reader.skip(wire)?,
        }
    }
    match (key, value) {
        (Some(key), Some(value)) => Ok((key, value)),
        _ => Err("metadata entry missing key or value".to_owned()),
    }
}

/// One decoded `TensorProto`, borrowed — payloads stay slices of the
/// input until [`ParsedOnnx`] copies the two it needs.
struct Tensor<'a> {
    name: String,
    dims: Vec<i64>,
    data_type: i32,
    raw_data: &'a [u8],
    float_data: Vec<f32>,
    int32_data: Vec<i32>,
}

fn parse_tensor(buf: &[u8]) -> Result<Tensor<'_>, String> {
    let mut reader = Reader::new(buf);
    let mut tensor = Tensor {
        name: String::new(),
        dims: Vec::new(),
        data_type: 0,
        raw_data: &[],
        float_data: Vec::new(),
        int32_data: Vec::new(),
    };
    while !reader.done() {
        let (number, wire) = reader.tag()?;
        match (number, wire) {
            (field::TENSOR_NAME, 2) => tensor.name = text(reader.bytes()?)?.to_owned(),
            (field::TENSOR_DATA_TYPE, 0) => {
                tensor.data_type = i32::try_from(reader.varint()?).unwrap_or(0);
            }
            // dims: packed varints (wire 2) or one unpacked varint.
            (field::TENSOR_DIMS, 2) => {
                let packed = reader.bytes()?;
                let mut packed_reader = Reader::new(packed);
                while !packed_reader.done() {
                    tensor.dims.push(packed_reader.varint()? as i64);
                }
            }
            (field::TENSOR_DIMS, 0) => tensor.dims.push(reader.varint()? as i64),
            // float_data: packed little-endian fixed32s (or one fixed32).
            (field::TENSOR_FLOAT_DATA, 2) => {
                let packed = reader.bytes()?;
                if packed.len() % 4 != 0 {
                    return Err("packed float_data is not a multiple of 4 bytes".to_owned());
                }
                tensor.float_data = packed
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|chunk| f32::from_le_bytes(*chunk))
                    .collect();
            }
            (field::TENSOR_FLOAT_DATA, 5) => {
                let bytes = reader.take(4)?;
                tensor
                    .float_data
                    .push(f32::from_le_bytes(bytes.try_into().expect("4 bytes")));
            }
            // int32_data: packed varints (negative values sign-extend to
            // 10-byte varints; int8 payloads stay well inside ±127).
            (field::TENSOR_INT32_DATA, 2) => {
                let packed = reader.bytes()?;
                let mut packed_reader = Reader::new(packed);
                while !packed_reader.done() {
                    tensor.int32_data.push(packed_reader.varint()? as i32);
                }
            }
            (field::TENSOR_INT32_DATA, 0) => tensor.int32_data.push(reader.varint()? as i32),
            (field::TENSOR_RAW_DATA, 2) => tensor.raw_data = reader.bytes()?,
            _ => reader.skip(wire)?,
        }
    }
    Ok(tensor)
}

/// The two tensors + metadata of a parsed tier file, with payloads
/// copied into owned storage.
struct ParsedOnnx {
    q: Vec<i8>,
    scales: Vec<f32>,
    rows: usize,
    dims: usize,
}

/// Walk a `ModelProto` and lift the int8-per-row tier out of it.
fn parse_onnx(bytes: &[u8]) -> Result<ParsedOnnx, Int8ModelError> {
    let bad = |source: String| Int8ModelError::Onnx(source);

    let mut metadata: HashMap<String, String> = HashMap::new();
    let mut tensors: Vec<Tensor<'_>> = Vec::new();

    let mut reader = Reader::new(bytes);
    while !reader.done() {
        let (number, wire) = reader.tag().map_err(bad)?;
        match (number, wire) {
            (field::MODEL_METADATA, 2) => {
                let (key, value) = metadata_entry(reader.bytes().map_err(bad)?).map_err(bad)?;
                metadata.insert(key, value);
            }
            (field::MODEL_GRAPH, 2) => {
                parse_graph(reader.bytes().map_err(bad)?, &mut tensors).map_err(bad)?;
            }
            _ => reader.skip(wire).map_err(bad)?,
        }
    }

    // Metadata gates, mirroring the ort provider's checks.
    if let Some(model_type) = metadata.get("model_type")
        && model_type != "fasttext_embedding"
    {
        return Err(Int8ModelError::Metadata(format!(
            "model_type {model_type:?} is not a fasttext embedding graph"
        )));
    }
    let quantization = metadata.get("quantization").map(String::as_str);
    match RowFormat::from_metadata(quantization) {
        Some(RowFormat::Int8PerRow) => {}
        Some(_) => {
            return Err(Int8ModelError::UnsupportedQuantization(
                quantization.unwrap_or_default().to_owned(),
            ));
        }
        None => {
            return Err(Int8ModelError::UnsupportedQuantization(
                quantization.unwrap_or_default().to_owned(),
            ));
        }
    }
    let dims = metadata
        .get("embedding_dimension")
        .and_then(|value| value.parse::<usize>().ok())
        .ok_or_else(|| {
            Int8ModelError::Metadata("embedding_dimension missing or malformed".to_owned())
        })?;

    // The two constants (Constant-node attribute values, or
    // initializers — the graph shape accepts both storages).
    let find = |name: &str| tensors.iter().find(|tensor| tensor.name == name);
    let q =
        find("q_embeddings").ok_or_else(|| bad("graph has no q_embeddings tensor".to_owned()))?;
    let scale = find("row_scale").ok_or_else(|| bad("graph has no row_scale tensor".to_owned()))?;

    if q.data_type != DATA_TYPE_INT8 {
        return Err(bad(format!(
            "q_embeddings is not int8 (data_type {})",
            q.data_type
        )));
    }
    if scale.data_type != DATA_TYPE_FLOAT {
        return Err(bad(format!(
            "row_scale is not float (data_type {})",
            scale.data_type
        )));
    }
    let [rows, q_dims] = match q.dims.as_slice() {
        [rows, dims] if *rows > 0 && *dims > 0 => [*rows as usize, *dims as usize],
        shape => return Err(bad(format!("q_embeddings is not 2-D: {shape:?}"))),
    };
    if q_dims != dims {
        return Err(bad(format!(
            "q_embeddings width {q_dims} disagrees with embedding_dimension {dims}"
        )));
    }
    match scale.dims.as_slice() {
        [scale_rows] if usize::try_from(*scale_rows) == Ok(rows) => {}
        shape => {
            return Err(bad(format!(
                "row_scale shape {shape:?} disagrees with {rows} rows"
            )));
        }
    }

    // Payloads: raw_data (what onnx.numpy_helper writes) or the typed
    // packed fields.
    let q_values: Vec<i8> = if !q.raw_data.is_empty() {
        if q.raw_data.len() != rows * q_dims {
            return Err(bad(format!(
                "q_embeddings raw_data is {} bytes, expected {}",
                q.raw_data.len(),
                rows * q_dims
            )));
        }
        q.raw_data.iter().map(|byte| *byte as i8).collect()
    } else if q.int32_data.len() == rows * q_dims {
        q.int32_data.iter().map(|value| *value as i8).collect()
    } else {
        return Err(bad(
            "q_embeddings has neither raw_data nor int32_data".to_owned()
        ));
    };
    let scales: Vec<f32> = if !scale.raw_data.is_empty() {
        if scale.raw_data.len() != rows * 4 {
            return Err(bad(format!(
                "row_scale raw_data is {} bytes, expected {}",
                scale.raw_data.len(),
                rows * 4
            )));
        }
        scale
            .raw_data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect()
    } else if scale.float_data.len() == rows {
        scale.float_data.clone()
    } else {
        return Err(bad(
            "row_scale has neither raw_data nor float_data".to_owned()
        ));
    };

    Ok(ParsedOnnx {
        q: q_values,
        scales,
        rows,
        dims,
    })
}

/// Walk a `GraphProto`, appending every tensor found — Constant-node
/// `value` attributes and initializers alike.
fn parse_graph<'a>(buf: &'a [u8], tensors: &mut Vec<Tensor<'a>>) -> Result<(), String> {
    let mut reader = Reader::new(buf);
    while !reader.done() {
        let (number, wire) = reader.tag()?;
        match (number, wire) {
            (field::GRAPH_INITIALIZER, 2) => {
                tensors.push(parse_tensor(reader.bytes()?)?);
            }
            (field::GRAPH_NODE, 2) => {
                let node = reader.bytes()?;
                let mut node_reader = Reader::new(node);
                let mut op_type = String::new();
                while !node_reader.done() {
                    let (node_number, node_wire) = node_reader.tag()?;
                    match (node_number, node_wire) {
                        (field::NODE_OP_TYPE, 2) => {
                            op_type = text(node_reader.bytes()?)?.to_owned()
                        }
                        (field::NODE_ATTRIBUTE, 2) => {
                            let attribute = node_reader.bytes()?;
                            let mut attribute_reader = Reader::new(attribute);
                            let mut name = String::new();
                            while !attribute_reader.done() {
                                let (attribute_number, attribute_wire) = attribute_reader.tag()?;
                                match (attribute_number, attribute_wire) {
                                    (field::ATTRIBUTE_NAME, 2) => {
                                        name = text(attribute_reader.bytes()?)?.to_owned()
                                    }
                                    (field::ATTRIBUTE_TENSOR, 2) => {
                                        // Only lift `value` tensors of
                                        // Constant nodes; other
                                        // attributes are skipped (the
                                        // payload is read here, so the
                                        // borrow checker forces the
                                        // take-now shape).
                                        let tensor = attribute_reader.bytes()?;
                                        if op_type == "Constant" && name == "value" {
                                            tensors.push(parse_tensor(tensor)?);
                                        }
                                    }
                                    _ => attribute_reader.skip(attribute_wire)?,
                                }
                            }
                        }
                        _ => node_reader.skip(node_wire)?,
                    }
                }
            }
            _ => reader.skip(wire)?,
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- A hand-rolled protobuf writer for synthetic fixtures ----------
    //
    // The tests build miniature ModelProtos field by field, so every
    // expected number below is computed from the bytes by hand.

    fn varint(mut value: u64) -> Vec<u8> {
        let mut out = Vec::new();
        loop {
            let byte = (value & 0x7f) as u8;
            value >>= 7;
            if value == 0 {
                out.push(byte);
                return out;
            }
            out.push(byte | 0x80);
        }
    }

    /// Tagged length-delimited field.
    fn bytes_field(number: u32, payload: &[u8]) -> Vec<u8> {
        let mut out = varint(u64::from(number) << 3 | 2);
        out.extend(varint(payload.len() as u64));
        out.extend_from_slice(payload);
        out
    }

    /// Tagged varint field.
    fn varint_field(number: u32, value: u64) -> Vec<u8> {
        let mut out = varint(u64::from(number) << 3);
        out.extend(varint(value));
        out
    }

    fn metadata(key: &str, value: &str) -> Vec<u8> {
        let mut entry = bytes_field(1, key.as_bytes());
        entry.extend(bytes_field(2, value.as_bytes()));
        bytes_field(field::MODEL_METADATA, &entry)
    }

    fn tensor_bytes(name: &str, dims: &[i64], data_type: i32, raw: &[u8]) -> Vec<u8> {
        let mut tensor = bytes_field(8, name.as_bytes());
        for dim in dims {
            tensor.extend(varint_field(1, *dim as u64));
        }
        tensor.extend(varint_field(2, data_type as u64));
        tensor.extend(bytes_field(9, raw));
        tensor
    }

    /// A Constant node carrying `tensor` as its `value` attribute — the
    /// storage shape the real tier artifacts use.
    fn constant_node(tensor: &[u8]) -> Vec<u8> {
        let mut attribute = bytes_field(1, b"value");
        attribute.extend(bytes_field(5, tensor));
        let mut node = bytes_field(4, b"Constant");
        node.extend(bytes_field(5, &attribute));
        bytes_field(field::GRAPH_NODE, &node)
    }

    fn graph(tensors: &[u8]) -> Vec<u8> {
        bytes_field(field::MODEL_GRAPH, tensors)
    }

    /// The hand-computed tier: vocab `{"a": 0, "b": 1}`, dims 2,
    /// row a = [3, -1] with scale 0.5 → [1.5, -0.5],
    /// row b = [2, 0] with scale 2.0 → [4.0, 0.0].
    ///
    /// cos(a, b) = 6 / (sqrt(2.5) * 4) = 0.9486832980505138…
    fn hand_tier() -> (Vec<u8>, Vec<u8>) {
        let mut model = metadata("model_type", "fasttext_embedding");
        model.extend(metadata("quantization", "int8-per-row"));
        model.extend(metadata("embedding_dimension", "2"));

        let mut graph_body = constant_node(&tensor_bytes(
            "q_embeddings",
            &[2, 2],
            DATA_TYPE_INT8,
            &[3, 0xff, 2, 0], // [3, -1, 2, 0] int8
        ));
        let scale_raw: Vec<u8> = [0.5f32, 2.0]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        graph_body.extend(constant_node(&tensor_bytes(
            "row_scale",
            &[2],
            DATA_TYPE_FLOAT,
            &scale_raw,
        )));
        model.extend(graph(&graph_body));

        let vocab = br#"{"vocab_size": 2, "word_to_idx": {"a": 0, "b": 1}}"#;
        (model, vocab.to_vec())
    }

    #[test]
    fn parses_the_hand_computed_tier() {
        let (onnx, vocab) = hand_tier();
        let model = Int8Model::parse(&onnx, &vocab).expect("hand tier parses");
        assert_eq!(model.dims(), 2);
        assert_eq!(model.vocab_len(), 2);
        assert_eq!(model.word_index("a"), Some(0));
        // Dequantization is the graph's Cast + Mul.
        assert_eq!(model.embedding("a").unwrap(), [1.5, -0.5]);
        assert_eq!(model.embedding("b").unwrap(), [4.0, 0.0]);
        assert!(model.embedding("zzz").is_none());
    }

    #[test]
    fn context_score_matches_hand_computed_cosines() {
        let (onnx, vocab) = hand_tier();
        let model = Int8Model::parse(&onnx, &vocab).expect("hand tier parses");

        let cos_ab = 6.0 / ((2.5f64).sqrt() * 4.0); // 0.9486832980505138
        // Single-token context: the score IS the cosine.
        assert!((model.context_score("a", "b") as f64 - cos_ab).abs() < 1e-6);
        // Mean over both tokens: (cos(a,a) + cos(a,b)) / 2.
        let mean = (1.0 + cos_ab) / 2.0;
        assert!((model.context_score("a", "a b") as f64 - mean).abs() < 1e-6);
        // The tokenizer strips punctuation and downcases ("B" → "b",
        // "The" is OOV and drops out of the mean).
        assert!((model.context_score("a", "The, B!") as f64 - cos_ab).abs() < 1e-6);
        // OOV on either side scores 0.0 (the gem's `(sim || 0.0)`).
        assert_eq!(model.context_score("zzz", "a b"), 0.0);
        assert_eq!(model.context_score("a", "zzz qqq"), 0.0);
        // Free-text cap: the first CONTEXT_TOKEN_CAP tokens only — here
        // a long context whose only in-vocab token sits far beyond the
        // first "a".
        let far = format!("{} a", "zzz ".repeat(CONTEXT_TOKEN_CAP + 4));
        assert_eq!(model.context_score("b", &far), 0.0);
        // Same-side sanity: a word against itself is 1.0.
        assert_eq!(model.context_score("a", "a"), 1.0);
    }

    #[test]
    fn typed_packed_fields_are_accepted_too() {
        // Same hand tier, but q rides int32_data (packed varints, with
        // -1 sign-extended to a 10-byte varint) and row_scale rides
        // float_data (packed fixed32).
        let mut model = metadata("quantization", "int8-per-row");
        model.extend(metadata("embedding_dimension", "2"));

        let mut q_tensor = bytes_field(8, b"q_embeddings");
        q_tensor.extend(varint_field(1, 2));
        q_tensor.extend(varint_field(1, 2));
        q_tensor.extend(varint_field(2, DATA_TYPE_INT8 as u64));
        let mut packed = Vec::new();
        for value in [3i32, -1, 2, 0] {
            packed.extend(varint(value as i64 as u64));
        }
        q_tensor.extend(bytes_field(5, &packed));

        let mut scale_tensor = bytes_field(8, b"row_scale");
        scale_tensor.extend(varint_field(1, 2));
        scale_tensor.extend(varint_field(2, DATA_TYPE_FLOAT as u64));
        let float_packed: Vec<u8> = [0.5f32, 2.0]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        scale_tensor.extend(bytes_field(4, &float_packed));

        // Deliver these two as graph initializers instead of Constant
        // nodes — that storage shape must load too.
        let mut graph_body = bytes_field(field::GRAPH_INITIALIZER, &q_tensor);
        graph_body.extend(bytes_field(field::GRAPH_INITIALIZER, &scale_tensor));
        model.extend(graph(&graph_body));

        let vocab = br#"{"word_to_idx": {"a": 0, "b": 1}}"#;
        let model = Int8Model::parse(&model, vocab).expect("typed-field tier parses");
        assert_eq!(model.embedding("a").unwrap(), [1.5, -0.5]);
        assert_eq!(model.embedding("b").unwrap(), [4.0, 0.0]);
        assert_eq!(model.vocab_len(), 2); // no declared vocab_size is fine
    }

    #[test]
    fn rejects_structurally_wrong_tiers() {
        let (onnx, vocab) = hand_tier();

        // Not a protobuf at all / truncated.
        assert!(Int8Model::parse(b"", &vocab).is_err());
        assert!(Int8Model::parse(&onnx[..onnx.len() / 2], &vocab).is_err());

        // The fp32 full tier (quantization metadata absent).
        let mut fp32 = metadata("model_type", "fasttext_embedding");
        fp32.extend(metadata("embedding_dimension", "2"));
        fp32.extend(graph(&constant_node(&tensor_bytes(
            "word_embeddings",
            &[2, 2],
            DATA_TYPE_FLOAT,
            &[0; 16],
        ))));
        let error = Int8Model::parse(&fp32, &vocab).unwrap_err();
        assert!(matches!(error, Int8ModelError::UnsupportedQuantization(_)));

        // Wrong model_type.
        let mut wrong_type = metadata("model_type", "something_else");
        wrong_type.extend(metadata("quantization", "int8-per-row"));
        wrong_type.extend(metadata("embedding_dimension", "2"));
        wrong_type.extend(graph(&onnx[onnx.len()..]));
        assert!(matches!(
            Int8Model::parse(&wrong_type, &vocab),
            Err(Int8ModelError::Metadata(_))
        ));

        // Missing row_scale constant.
        let mut no_scale = metadata("quantization", "int8-per-row");
        no_scale.extend(metadata("embedding_dimension", "2"));
        no_scale.extend(graph(&constant_node(&tensor_bytes(
            "q_embeddings",
            &[2, 2],
            DATA_TYPE_INT8,
            &[3, 0xff, 2, 0],
        ))));
        assert!(matches!(
            Int8Model::parse(&no_scale, &vocab),
            Err(Int8ModelError::Onnx(_))
        ));

        // Matrix width disagrees with embedding_dimension.
        let mut bad_dims = metadata("quantization", "int8-per-row");
        bad_dims.extend(metadata("embedding_dimension", "3"));
        let mut graph_body = constant_node(&tensor_bytes(
            "q_embeddings",
            &[2, 2],
            DATA_TYPE_INT8,
            &[3, 0xff, 2, 0],
        ));
        graph_body.extend(constant_node(&tensor_bytes(
            "row_scale",
            &[2],
            DATA_TYPE_FLOAT,
            &[0; 8],
        )));
        bad_dims.extend(graph(&graph_body));
        assert!(Int8Model::parse(&bad_dims, &vocab).is_err());

        // Vocabulary: empty, disagreeing vocab_size, out-of-range row.
        assert!(Int8Model::parse(&onnx, br#"{"word_to_idx": {}}"#).is_err());
        assert!(
            Int8Model::parse(
                &onnx,
                br#"{"vocab_size": 3, "word_to_idx": {"a": 0, "b": 1}}"#
            )
            .is_err()
        );
        assert!(Int8Model::parse(&onnx, br#"{"word_to_idx": {"a": 0, "b": 9}}"#).is_err());
        assert!(Int8Model::parse(&onnx, b"not json").is_err());
    }

    #[test]
    fn provider_surface_reuses_the_rerank_math() {
        let (onnx, vocab) = hand_tier();
        let model = Int8Model::parse(&onnx, &vocab).expect("hand tier parses");
        let provider: &dyn EmbeddingProvider = &model;
        assert_eq!(provider.dims(), 2);
        assert_eq!(provider.embedding("a").unwrap(), [1.5, -0.5]);
        // similarity() through the provider trait agrees with the
        // hand-computed cosine.
        let cos_ab = 6.0 / ((2.5f64).sqrt() * 4.0);
        assert!((super::super::similarity(provider, "a", "b").unwrap() - cos_ab).abs() < 1e-12);
    }

    // --- The real-artifact fixture --------------------------------------
    //
    // Checked in at tests/fixtures/models/ — a 40-word truncation of
    // the registry v1.0.1 en/mini tier, built by
    // scripts/make_model_fixture.py (see the fixture dir's LICENSE
    // note for provenance). Running the real bytes through the parser
    // guards the wire reader against drift in what onnx actually
    // writes (Constant nodes, raw_data payloads).

    const FIXTURE_ONNX: &[u8] =
        include_bytes!("../../tests/fixtures/models/en-mini-truncated.onnx");
    const FIXTURE_VOCAB: &[u8] =
        include_bytes!("../../tests/fixtures/models/en-mini-truncated.vocab.json");

    fn fixture() -> Int8Model {
        Int8Model::parse(FIXTURE_ONNX, FIXTURE_VOCAB).expect("real-derived fixture parses")
    }

    #[test]
    fn parses_the_real_derived_fixture() {
        let model = fixture();
        assert_eq!(model.dims(), 300);
        assert_eq!(model.vocab_len(), 40);
        // The vectors are real fastText (frozen at generation time by
        // scripts/make_model_fixture.py): dog is a much nearer
        // neighbor of cat than computer is, and the mean-cosine score
        // preserves that ordering (the gem's boost is a positive
        // multiple).
        let dog = model.context_score("cat", "dog");
        assert!(
            (dog - 0.707432).abs() < 1e-4,
            "cat-dog cosine drifted: {dog}"
        );
        let computer = model.context_score("cat", "computer");
        assert!(
            (computer - 0.187228).abs() < 1e-4,
            "cat-computer drifted: {computer}"
        );
    }

    #[test]
    fn fixture_scores_a_sentence_context_sensibly() {
        let model = fixture();
        // Frozen at generation: mean cos over in-vocab tokens.
        let animal = model.context_score("puppy", "the dog and the cat");
        let machine = model.context_score("puppy", "the computer and the keyboard");
        assert!(
            (animal - 0.340374).abs() < 1e-4,
            "animal context drifted: {animal}"
        );
        assert!(
            (machine - 0.126196).abs() < 1e-4,
            "machine context drifted: {machine}"
        );
        assert!(
            animal > machine,
            "puppy should fit the animal context ({animal}) better than the machine one ({machine})"
        );
        // OOV words on either side are honest zeros.
        assert_eq!(model.context_score("puppy", "florbington blorble"), 0.0);
    }
}
