//! Bucket-table sibling artifact — the OOV half of fastText's input
//! matrix (feature `model`) — plan 103.
//!
//! The tier artifacts carry VOCAB word vectors only (the models repo
//! converts plain `.vec` files); an OOV word whose character n-grams are
//! not themselves vocabulary words embeds to nothing ([`super::oov`]'s
//! documented gap). fastText's real OOV path hashes character n-grams
//! into `bucket` rows of the training binary's input matrix — rows the
//! upstream `.vec` never contained. This module reads the models repo's
//! exported sibling artifact (`kotoshu://models/{lang}/buckets`,
//! `scripts/export_buckets.py` there) so [`super::int8_model`]'s OOV
//! fallback can consult exactly those rows.
//!
//! # Artifact shape
//!
//! The tier graph shape with one extra constant — `onnx.Constant` nodes
//! carrying `q_embeddings` int8 `[K, d]`, `row_scale` fp32 `[K]`, and
//! `bucket_ids` int64 `[K]` ASCENDING (the original hash-bucket index of
//! each kept row) — plus metadata `model_type=fasttext_buckets`,
//! `quantization=int8-per-row`, `embedding_dimension=d`,
//! `bucket_count=B`, `minn`/`maxn` (the n-gram lengths the training
//! wrote; the crawl models were trained with minn=maxn=5, discovered
//! when reading the upstream binary), and `buckets=K`. Row lookup is a
//! binary search over `bucket_ids`; row dequantization is the same
//! `q as f32 * row_scale` the tiers use.
//!
//! The exported table is a top-K subset of the full `2M x d` table
//! (the full int8 table would be ~624 MB per language): the models repo
//! keeps the rows by training usage plus typo-corpus demand, with the
//! measured quality delta in `eval/reports/{lang}.buckets.json`. A
//! lookup whose bucket was not kept is an honest miss (`None`).

use std::fmt;

use super::dequant::{RowFormat, dequant_row_int8};
use super::onnx_wire;
use super::oov;

/// Errors of the bucket-table reader.
#[derive(Debug)]
pub enum BucketError {
    /// The `.onnx` bytes are not a parsable ONNX protobuf, or are
    /// structurally unexpected (missing tensors, bad shapes, unsafe
    /// bucket ids).
    Onnx(String),
    /// The graph metadata is missing or malformed.
    Metadata(String),
    /// The quantization descriptor is not one this reader handles.
    UnsupportedQuantization(String),
}

impl fmt::Display for BucketError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Onnx(source) => write!(f, "bucket table: {source}"),
            Self::Metadata(source) => write!(f, "bucket table metadata: {source}"),
            Self::UnsupportedQuantization(descriptor) => write!(
                f,
                "bucket table quantization {descriptor:?} not readable (known: int8-per-row)"
            ),
        }
    }
}

impl std::error::Error for BucketError {}

/// One loaded bucket table: quantized rows, row scales, the kept bucket
/// ids (ascending, binary-searchable), and the hash modulus + n-gram
/// range the export used.
#[derive(Clone, Debug)]
pub struct BucketTable {
    /// Original bucket indices of the kept rows, ascending.
    ids: Vec<u32>,
    /// Quantized rows, row-major `[K, d]`, parallel to `ids`.
    q: Vec<i8>,
    /// Per-row dequantization scale `[K]`.
    scales: Vec<f32>,
    /// The hash modulus (`fastText args.bucket`, 2,000,000 for the
    /// crawl models).
    bucket_count: u32,
    /// The n-gram lengths the training wrote (`args.minn`/`args.maxn`;
    /// 5..=5 for the crawl models).
    minn: usize,
    maxn: usize,
    dims: usize,
}

impl BucketTable {
    /// Parse a bucket-table artifact from the byte CONTENTS of its
    /// `.onnx` file, exactly as distributed.
    pub fn parse(onnx_bytes: &[u8]) -> Result<Self, BucketError> {
        let bad = |source: String| BucketError::Onnx(source);
        let wire = onnx_wire::parse_model(onnx_bytes).map_err(bad)?;

        if let Some(model_type) = wire.metadata.get("model_type")
            && model_type != "fasttext_buckets"
        {
            return Err(BucketError::Metadata(format!(
                "model_type {model_type:?} is not a fastText bucket table"
            )));
        }
        let quantization = wire.metadata.get("quantization").map(String::as_str);
        match RowFormat::from_metadata(quantization) {
            Some(RowFormat::Int8PerRow) => {}
            _ => {
                return Err(BucketError::UnsupportedQuantization(
                    quantization.unwrap_or_default().to_owned(),
                ));
            }
        }
        let dims = wire
            .metadata
            .get("embedding_dimension")
            .and_then(|value| value.parse::<usize>().ok())
            .ok_or_else(|| {
                BucketError::Metadata("embedding_dimension missing or malformed".to_owned())
            })?;
        let bucket_count = wire
            .metadata
            .get("bucket_count")
            .and_then(|value| value.parse::<u32>().ok())
            .ok_or_else(|| BucketError::Metadata("bucket_count missing or malformed".to_owned()))?;
        let minn = wire
            .metadata
            .get("minn")
            .and_then(|value| value.parse::<usize>().ok())
            .ok_or_else(|| BucketError::Metadata("minn missing or malformed".to_owned()))?;
        let maxn = wire
            .metadata
            .get("maxn")
            .and_then(|value| value.parse::<usize>().ok())
            .ok_or_else(|| BucketError::Metadata("maxn missing or malformed".to_owned()))?;
        if minn == 0 || minn > maxn {
            return Err(BucketError::Metadata(format!(
                "n-gram range {minn}..={maxn} is empty or invalid"
            )));
        }

        let q_tensor = wire
            .tensor("q_embeddings")
            .ok_or_else(|| bad("graph has no q_embeddings tensor".to_owned()))?;
        let scale_tensor = wire
            .tensor("row_scale")
            .ok_or_else(|| bad("graph has no row_scale tensor".to_owned()))?;
        let ids_tensor = wire
            .tensor("bucket_ids")
            .ok_or_else(|| bad("graph has no bucket_ids tensor".to_owned()))?;

        if q_tensor.data_type != onnx_wire::DATA_TYPE_INT8 {
            return Err(bad(format!(
                "q_embeddings is not int8 (data_type {})",
                q_tensor.data_type
            )));
        }
        if scale_tensor.data_type != onnx_wire::DATA_TYPE_FLOAT {
            return Err(bad(format!(
                "row_scale is not float (data_type {})",
                scale_tensor.data_type
            )));
        }
        if ids_tensor.data_type != onnx_wire::DATA_TYPE_INT64 {
            return Err(bad(format!(
                "bucket_ids is not int64 (data_type {})",
                ids_tensor.data_type
            )));
        }
        let [rows, q_dims] = match q_tensor.dims.as_slice() {
            [rows, dims] if *rows > 0 && *dims > 0 => [*rows as usize, *dims as usize],
            shape => return Err(bad(format!("q_embeddings is not 2-D: {shape:?}"))),
        };
        if q_dims != dims {
            return Err(bad(format!(
                "q_embeddings width {q_dims} disagrees with embedding_dimension {dims}"
            )));
        }
        match scale_tensor.dims.as_slice() {
            [scale_rows] if usize::try_from(*scale_rows) == Ok(rows) => {}
            shape => {
                return Err(bad(format!(
                    "row_scale shape {shape:?} disagrees with {rows} rows"
                )));
            }
        }
        match ids_tensor.dims.as_slice() {
            [ids_rows] if usize::try_from(*ids_rows) == Ok(rows) => {}
            shape => {
                return Err(bad(format!(
                    "bucket_ids shape {shape:?} disagrees with {rows} rows"
                )));
            }
        }

        // Payloads: raw_data (what onnx.numpy_helper writes) or the
        // typed packed fields.
        let q = onnx_wire::int8_payload(q_tensor, rows, dims).map_err(bad)?;
        let scales = onnx_wire::float_payload(scale_tensor, rows).map_err(bad)?;
        let id_values: Vec<u32> = if !ids_tensor.raw_data.is_empty() {
            if ids_tensor.raw_data.len() != rows * 8 {
                return Err(bad(format!(
                    "bucket_ids raw_data is {} bytes, expected {}",
                    ids_tensor.raw_data.len(),
                    rows * 8
                )));
            }
            ids_tensor
                .raw_data
                .as_chunks::<8>()
                .0
                .iter()
                .map(|chunk| u64::from_le_bytes(*chunk))
                .map(u32::try_from)
                .collect::<Result<Vec<u32>, _>>()
                .map_err(|_| bad("bucket id does not fit u32".to_owned()))?
        } else if ids_tensor.int64_data.len() == rows {
            ids_tensor
                .int64_data
                .iter()
                .map(|value| u32::try_from(*value))
                .collect::<Result<Vec<u32>, _>>()
                .map_err(|_| bad("bucket id does not fit u32".to_owned()))?
        } else {
            return Err(bad(
                "bucket_ids has neither raw_data nor int64_data".to_owned()
            ));
        };

        // Ascending, in-range ids: the binary-search contract and the
        // hash modulus agree (a violation would misdirect lookups, so
        // it is a load error, not a lookup-time concern).
        for pair in id_values.windows(2) {
            if pair[0] >= pair[1] {
                return Err(bad(format!(
                    "bucket_ids not strictly ascending at {}",
                    pair[0]
                )));
            }
        }
        if let Some(id) = id_values.iter().find(|id| **id >= bucket_count) {
            return Err(bad(format!(
                "bucket id {id} out of range for bucket_count {bucket_count}"
            )));
        }

        Ok(Self {
            ids: id_values,
            q,
            scales,
            bucket_count,
            minn,
            maxn,
            dims,
        })
    }

    /// The hash modulus this table was exported for.
    pub fn bucket_count(&self) -> u32 {
        self.bucket_count
    }

    /// The n-gram lengths (in characters, of the MARKED `<word>` form)
    /// whose bucket rows training wrote.
    pub fn ngram_range(&self) -> (usize, usize) {
        (self.minn, self.maxn)
    }

    /// Vector dimensionality.
    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Kept-row count.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether the table keeps no rows at all.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// The kept bucket ids, ascending.
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }

    /// Whether the table keeps `bucket`'s row.
    pub fn contains(&self, bucket: u32) -> bool {
        self.ids.binary_search(&bucket).is_ok()
    }

    /// The dequantized row of `bucket` (binary search over `ids`), or
    /// `None` when the table does not keep it.
    pub fn row(&self, bucket: u32) -> Option<Vec<f32>> {
        let index = self.ids.binary_search(&bucket).ok()?;
        let start = index.checked_mul(self.dims)?;
        let end = start.checked_add(self.dims)?;
        let scale = *self.scales.get(index)?;
        Some(dequant_row_int8(self.q.get(start..end)?, scale))
    }

    /// The dequantized row a MARKED n-gram of a word hashes to — the
    /// exact row fastText training wrote for that n-gram — or `None`
    /// when the n-gram's length is outside the trained range or the
    /// table does not keep its bucket.
    pub fn ngram_row(&self, ngram: &str) -> Option<Vec<f32>> {
        let chars = ngram.chars().count();
        if !(self.minn..=self.maxn).contains(&chars) {
            return None;
        }
        self.row(oov::fasttext_hash_mod(ngram, self.bucket_count))
    }

    /// The distinct marked n-grams (of `<word>`, lengths
    /// [`BucketTable::ngram_range`]) whose buckets this table keeps —
    /// the bucket-backed half of the OOV composition.
    pub fn marked_ngrams_kept(&self, word: &str) -> Vec<String> {
        let mut ngrams: Vec<String> = Vec::new();
        let marked: Vec<char> = format!("<{word}>").chars().collect();
        for length in self.minn..=self.maxn.min(marked.len()) {
            for start in 0..=marked.len() - length {
                let ngram: String = marked[start..start + length].iter().collect();
                if self.ngram_row(&ngram).is_some() && !ngrams.contains(&ngram) {
                    ngrams.push(ngram);
                }
            }
        }
        ngrams
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- The hand-rolled protobuf writer (same shape as int8_model's) --

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

    fn bytes_field(number: u32, payload: &[u8]) -> Vec<u8> {
        let mut out = varint(u64::from(number) << 3 | 2);
        out.extend(varint(payload.len() as u64));
        out.extend_from_slice(payload);
        out
    }

    fn varint_field(number: u32, value: u64) -> Vec<u8> {
        let mut out = varint(u64::from(number) << 3);
        out.extend(varint(value));
        out
    }

    fn metadata(key: &str, value: &str) -> Vec<u8> {
        let mut entry = bytes_field(1, key.as_bytes());
        entry.extend(bytes_field(2, value.as_bytes()));
        bytes_field(14, &entry)
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

    fn constant_node(tensor: &[u8]) -> Vec<u8> {
        let mut attribute = bytes_field(1, b"value");
        attribute.extend(bytes_field(5, tensor));
        let mut node = bytes_field(4, b"Constant");
        node.extend(bytes_field(5, &attribute));
        bytes_field(1, &node)
    }

    /// The graph body wrapped in the ModelProto's `graph` field (7).
    fn graph_field(graph_body: &[u8]) -> Vec<u8> {
        bytes_field(7, graph_body)
    }

    /// A tiny hand table: bucket_count 100, n-grams 3..=3, two kept
    /// buckets 7 and 42 with hand-computed rows:
    /// row 7 = [4, -2] scale 0.5 → [2.0, -1.0] (norm sqrt 5),
    /// row 42 = [127, 0] scale 2.0 → [254.0, 0.0].
    fn hand_table() -> Vec<u8> {
        let mut model = metadata("model_type", "fasttext_buckets");
        model.extend(metadata("quantization", "int8-per-row"));
        model.extend(metadata("embedding_dimension", "2"));
        model.extend(metadata("bucket_count", "100"));
        model.extend(metadata("minn", "3"));
        model.extend(metadata("maxn", "3"));
        model.extend(metadata("buckets", "2"));

        let mut graph = constant_node(&tensor_bytes(
            "q_embeddings",
            &[2, 2],
            onnx_wire::DATA_TYPE_INT8,
            &[4, 0xfe, 127, 0],
        ));
        let scale_raw: Vec<u8> = [0.5f32, 2.0]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        graph.extend(constant_node(&tensor_bytes(
            "row_scale",
            &[2],
            onnx_wire::DATA_TYPE_FLOAT,
            &scale_raw,
        )));
        let ids_raw: Vec<u8> = [7u64, 42]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        graph.extend(constant_node(&tensor_bytes(
            "bucket_ids",
            &[2],
            onnx_wire::DATA_TYPE_INT64,
            &ids_raw,
        )));
        model.extend(graph_field(&graph));
        model
    }

    #[test]
    fn parses_the_hand_table() {
        let table = BucketTable::parse(&hand_table()).expect("hand table parses");
        assert_eq!(table.dims(), 2);
        assert_eq!(table.len(), 2);
        assert_eq!(table.bucket_count(), 100);
        assert_eq!(table.ngram_range(), (3, 3));
        assert_eq!(table.ids(), &[7, 42]);
        assert!(table.contains(7));
        assert!(table.contains(42));
        assert!(!table.contains(8));

        // Row dequantization is the graph's Cast + Mul.
        assert_eq!(table.row(7).unwrap(), [2.0, -1.0]);
        assert_eq!(table.row(42).unwrap(), [254.0, 0.0]);
        assert!(table.row(8).is_none());

        // The hash lookup: "<te" in a bucket_count-100 table —
        // oov::fasttext_hash_mod is the shared FNV-1a.
        let bucket = oov::fasttext_hash_mod("<te", 100);
        let expected = table.contains(bucket);
        assert_eq!(
            table.ngram_row("<te").is_some(),
            expected,
            "ngram_row must agree with the hash lookup"
        );
        // Lengths outside the trained range are honest misses.
        assert!(table.ngram_row("<teh").is_none()); // 4 chars > maxn 3
        assert!(table.ngram_row("<t").is_none()); // 2 chars < minn 3
    }

    #[test]
    fn marked_ngrams_kept_reports_the_resolvable_set() {
        let table = BucketTable::parse(&hand_table()).expect("hand table parses");
        // minn=maxn=3: the marked 3-grams of "teh" are <te, teh, eh>.
        let kept = table.marked_ngrams_kept("teh");
        for ngram in ["<te", "teh", "eh>"] {
            assert_eq!(
                kept.contains(&ngram.to_owned()),
                table.ngram_row(ngram).is_some(),
                "{ngram} membership must agree with ngram_row"
            );
        }
        // Deduplicated (the marked form of "aaa" yields <aa twice).
        let kept = table.marked_ngrams_kept("aaa");
        assert!(kept.iter().all(|id| id.len() == kept[0].len()));
    }

    #[test]
    fn rejects_structurally_wrong_tables() {
        // Not a protobuf at all.
        assert!(BucketTable::parse(b"").is_err());

        // Wrong model_type (a tier artifact, not a bucket table).
        let mut tier = metadata("model_type", "fasttext_embedding");
        tier.extend(metadata("quantization", "int8-per-row"));
        tier.extend(metadata("embedding_dimension", "2"));
        tier.extend(graph_field(&[]));
        let error = BucketTable::parse(&tier).unwrap_err();
        assert!(matches!(error, BucketError::Metadata(_)));

        // Missing bucket_count / minn / maxn metadata.
        let mut no_count = metadata("model_type", "fasttext_buckets");
        no_count.extend(metadata("quantization", "int8-per-row"));
        no_count.extend(metadata("embedding_dimension", "2"));
        no_count.extend(graph_field(&[]));
        let error = BucketTable::parse(&no_count).unwrap_err();
        assert!(matches!(error, BucketError::Metadata(_)));

        // Missing bucket_ids tensor.
        let mut no_ids = metadata("model_type", "fasttext_buckets");
        no_ids.extend(metadata("quantization", "int8-per-row"));
        no_ids.extend(metadata("embedding_dimension", "2"));
        no_ids.extend(metadata("bucket_count", "100"));
        no_ids.extend(metadata("minn", "3"));
        no_ids.extend(metadata("maxn", "3"));
        no_ids.extend(constant_node(&tensor_bytes(
            "q_embeddings",
            &[1, 2],
            onnx_wire::DATA_TYPE_INT8,
            &[4, 0xfe],
        )));
        no_ids.extend(graph_field(&[]));
        let error = BucketTable::parse(&no_ids).unwrap_err();
        assert!(matches!(error, BucketError::Onnx(_)));

        // Non-ascending bucket ids (binary search would misdirect).
        let mut unsorted = metadata("model_type", "fasttext_buckets");
        unsorted.extend(metadata("quantization", "int8-per-row"));
        unsorted.extend(metadata("embedding_dimension", "2"));
        unsorted.extend(metadata("bucket_count", "100"));
        unsorted.extend(metadata("minn", "3"));
        unsorted.extend(metadata("maxn", "3"));
        let mut graph = constant_node(&tensor_bytes(
            "q_embeddings",
            &[2, 2],
            onnx_wire::DATA_TYPE_INT8,
            &[4, 0xfe, 127, 0],
        ));
        let scale_raw: Vec<u8> = [0.5f32, 2.0]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        graph.extend(constant_node(&tensor_bytes(
            "row_scale",
            &[2],
            onnx_wire::DATA_TYPE_FLOAT,
            &scale_raw,
        )));
        let ids_raw: Vec<u8> = [42u64, 7]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        graph.extend(constant_node(&tensor_bytes(
            "bucket_ids",
            &[2],
            onnx_wire::DATA_TYPE_INT64,
            &ids_raw,
        )));
        let error = BucketTable::parse(&[graph_field(&graph), unsorted].concat()).unwrap_err();
        assert!(matches!(error, BucketError::Onnx(_)));

        // Id out of range for the declared bucket_count.
        let ids_raw: Vec<u8> = [7u64, 999]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let mut out_of_range = metadata("model_type", "fasttext_buckets");
        out_of_range.extend(metadata("quantization", "int8-per-row"));
        out_of_range.extend(metadata("embedding_dimension", "2"));
        out_of_range.extend(metadata("bucket_count", "100"));
        out_of_range.extend(metadata("minn", "3"));
        out_of_range.extend(metadata("maxn", "3"));
        let mut graph = constant_node(&tensor_bytes(
            "q_embeddings",
            &[2, 2],
            onnx_wire::DATA_TYPE_INT8,
            &[4, 0xfe, 127, 0],
        ));
        graph.extend(constant_node(&tensor_bytes(
            "row_scale",
            &[2],
            onnx_wire::DATA_TYPE_FLOAT,
            &scale_raw,
        )));
        graph.extend(constant_node(&tensor_bytes(
            "bucket_ids",
            &[2],
            onnx_wire::DATA_TYPE_INT64,
            &ids_raw,
        )));
        let error = BucketTable::parse(&[graph_field(&graph), out_of_range].concat()).unwrap_err();
        assert!(matches!(error, BucketError::Onnx(_)));

        // fp32 storage (quantization metadata absent) is unsupported.
        let mut fp32 = metadata("model_type", "fasttext_buckets");
        fp32.extend(metadata("embedding_dimension", "2"));
        fp32.extend(graph_field(&[]));
        let error = BucketTable::parse(&fp32).unwrap_err();
        assert!(matches!(error, BucketError::UnsupportedQuantization(_)));
    }
}
