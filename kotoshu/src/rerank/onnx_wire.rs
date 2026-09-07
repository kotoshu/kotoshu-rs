//! The minimal ONNX protobuf walker shared by the pure-Rust artifact
//! readers (feature `model`): a forward-only wire-format cursor
//! (`Reader`), the `TensorProto` / graph / metadata decoding the tier
//! and LID artifacts need, and nothing else — no schema, no third-party
//! parser. Extracted from `int8_model` when the LID reader (plan 102)
//! needed the same walk; both readers keep their artifact-specific
//! tensor extraction and validation in their own modules (MECE: the
//! wire format is one concern, in one place).

use std::collections::HashMap;

// --- The minimal ONNX protobuf reader -----------------------------------
//
// Protobuf wire format, the parts ONNX uses: each field is a varint tag
// `(number << 3) | wire_type` followed by its payload — varint (wire 0),
// 64-bit (1), length-delimited (2), 32-bit (5). `int32_data`/`dims` pack
// into a length-delimited run of varints; `float_data` packs into a run
// of little-endian fixed32s. Groups (wire 3/4) are long-obsolete in
// ONNX and rejected.

/// ONNX `TensorProto.DataType` values this reader names.
pub(crate) const DATA_TYPE_FLOAT: i32 = 1;
pub(crate) const DATA_TYPE_INT8: i32 = 3;
pub(crate) const DATA_TYPE_INT64: i32 = 7;

/// Field numbers of the protos walked (onnx/onnx.proto).
pub(crate) mod field {
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
    pub const TENSOR_INT64_DATA: u32 = 7;
    pub const TENSOR_NAME: u32 = 8;
    pub const TENSOR_RAW_DATA: u32 = 9;

    pub const METADATA_KEY: u32 = 1;
    pub const METADATA_VALUE: u32 = 2;
}

/// Forward-only protobuf cursor over one message body.
pub(crate) struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub(crate) fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    pub(crate) fn done(&self) -> bool {
        self.pos >= self.buf.len()
    }

    pub(crate) fn take(&mut self, len: usize) -> Result<&'a [u8], String> {
        let end = self.pos.checked_add(len).ok_or("field length overflow")?;
        let slice = self
            .buf
            .get(self.pos..end)
            .ok_or_else(|| format!("message truncated at byte {}", self.buf.len()))?;
        self.pos = end;
        Ok(slice)
    }

    pub(crate) fn varint(&mut self) -> Result<u64, String> {
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
    pub(crate) fn tag(&mut self) -> Result<(u32, u8), String> {
        let tag = self.varint()?;
        let number = u32::try_from(tag >> 3).map_err(|_| "field number overflow")?;
        let wire = (tag & 0x7) as u8;
        Ok((number, wire))
    }

    /// Length-delimited payload (wire 2).
    pub(crate) fn bytes(&mut self) -> Result<&'a [u8], String> {
        let len = usize::try_from(self.varint()?).map_err(|_| "length-delimited size overflow")?;
        self.take(len)
    }

    /// Skip a field of the given wire type (groups are rejected).
    pub(crate) fn skip(&mut self, wire: u8) -> Result<(), String> {
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
pub(crate) fn text(bytes: &[u8]) -> Result<&str, String> {
    std::str::from_utf8(bytes).map_err(|_| "string field is not UTF-8".to_owned())
}

/// One decoded `StringStringEntryProto`.
pub(crate) fn metadata_entry(buf: &[u8]) -> Result<(String, String), String> {
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
pub(crate) struct Tensor<'a> {
    pub(crate) name: String,
    pub(crate) dims: Vec<i64>,
    pub(crate) data_type: i32,
    pub(crate) raw_data: &'a [u8],
    pub(crate) float_data: Vec<f32>,
    pub(crate) int32_data: Vec<i32>,
    pub(crate) int64_data: Vec<i64>,
}

pub(crate) fn parse_tensor(buf: &[u8]) -> Result<Tensor<'_>, String> {
    let mut reader = Reader::new(buf);
    let mut tensor = Tensor {
        name: String::new(),
        dims: Vec::new(),
        data_type: 0,
        raw_data: &[],
        float_data: Vec::new(),
        int32_data: Vec::new(),
        int64_data: Vec::new(),
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
            // int64_data: packed varints (the bucket_ids tensor).
            (field::TENSOR_INT64_DATA, 2) => {
                let packed = reader.bytes()?;
                let mut packed_reader = Reader::new(packed);
                while !packed_reader.done() {
                    tensor.int64_data.push(packed_reader.varint()? as i64);
                }
            }
            (field::TENSOR_INT64_DATA, 0) => tensor.int64_data.push(reader.varint()? as i64),
            (field::TENSOR_RAW_DATA, 2) => tensor.raw_data = reader.bytes()?,
            _ => reader.skip(wire)?,
        }
    }
    Ok(tensor)
}

/// Walk a `GraphProto`, appending every tensor found — Constant-node
/// `value` attributes and initializers alike.
pub(crate) fn parse_graph<'a>(buf: &'a [u8], tensors: &mut Vec<Tensor<'a>>) -> Result<(), String> {
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

/// A walked `ModelProto`: its metadata entries and every tensor found.
pub(crate) struct WireModel<'a> {
    pub(crate) metadata: HashMap<String, String>,
    pub(crate) tensors: Vec<Tensor<'a>>,
}

impl WireModel<'_> {
    pub(crate) fn tensor(&self, name: &str) -> Option<&Tensor<'_>> {
        self.tensors.iter().find(|tensor| tensor.name == name)
    }
}

/// Walk a `ModelProto`, lifting metadata entries and tensors.
pub(crate) fn parse_model(bytes: &[u8]) -> Result<WireModel<'_>, String> {
    let mut metadata: HashMap<String, String> = HashMap::new();
    let mut tensors: Vec<Tensor<'_>> = Vec::new();

    let mut reader = Reader::new(bytes);
    while !reader.done() {
        let (number, wire) = reader.tag()?;
        match (number, wire) {
            (field::MODEL_METADATA, 2) => {
                let (key, value) = metadata_entry(reader.bytes()?)?;
                metadata.insert(key, value);
            }
            (field::MODEL_GRAPH, 2) => {
                parse_graph(reader.bytes()?, &mut tensors)?;
            }
            _ => reader.skip(wire)?,
        }
    }
    Ok(WireModel { metadata, tensors })
}

/// Decode a float32 payload (raw_data or float_data) of `rows` values.
pub(crate) fn float_payload(tensor: &Tensor<'_>, rows: usize) -> Result<Vec<f32>, String> {
    if !tensor.raw_data.is_empty() {
        if tensor.raw_data.len() != rows * 4 {
            return Err(format!(
                "{} raw_data is {} bytes, expected {}",
                tensor.name,
                tensor.raw_data.len(),
                rows * 4
            ));
        }
        Ok(tensor
            .raw_data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect())
    } else if tensor.float_data.len() == rows {
        Ok(tensor.float_data.clone())
    } else {
        Err(format!(
            "{} has neither raw_data nor float_data",
            tensor.name
        ))
    }
}

/// Decode an int8 payload (raw_data or int32_data) of `rows * width` values.
pub(crate) fn int8_payload(
    tensor: &Tensor<'_>,
    rows: usize,
    width: usize,
) -> Result<Vec<i8>, String> {
    if !tensor.raw_data.is_empty() {
        if tensor.raw_data.len() != rows * width {
            return Err(format!(
                "{} raw_data is {} bytes, expected {}",
                tensor.name,
                tensor.raw_data.len(),
                rows * width
            ));
        }
        Ok(tensor.raw_data.iter().map(|byte| *byte as i8).collect())
    } else if tensor.int32_data.len() == rows * width {
        Ok(tensor.int32_data.iter().map(|value| *value as i8).collect())
    } else {
        Err(format!(
            "{} has neither raw_data nor int32_data",
            tensor.name
        ))
    }
}
