//! Pure-Rust reader + scorer for the language-identification model
//! (feature `model`) — plan 102, the LID sibling of
//! [`crate::rerank::int8_model`].
//!
//! The gem detects document language by shelling out to the fastText
//! bindings over upstream `lid.176.ftz` (176 labels, dim 16, char 2–4
//! grams, hierarchical softmax). The registry converts that model into
//! the same artifact shape the embedding tiers use — an ONNX container
//! (`kotoshu://models/lid/lid-176`: int8-per-row input matrix, row
//! scales, output matrix, Constant nodes, `raw_data`) plus a
//! `lid.176.vocab.json` sidecar (labels, Huffman tree counts, word
//! dictionary, pruned-ngram index map, arguments) — and this module
//! walks those bytes with no runtime, exactly how `int8_model` reads
//! its tiers.
//!
//! # Inference, bit-faithful to the bindings
//!
//! Everything the C++ does at predict time is reproduced, including
//! the quirks the parity test freezes:
//!
//! - **Hashing**: FNV-1a over SIGN-EXTENDED bytes (`uint32_t(int8_t)`)
//!   — every byte `>= 0x80` sign-extends, which moves the hash for all
//!   non-ASCII scripts.
//! - **Features**: whitespace tokens; an in-vocabulary token adds its
//!   word row plus its char n-gram rows (`minn..maxn` of `<word>`,
//!   byte-walked so UTF-8 continuation bytes extend, not start, an
//!   n-gram); an out-of-vocabulary token contributes n-grams only; a
//!   trailing EOS token (`</s>`) contributes its word row. Duplicate
//!   rows are kept (the mean over the multiset is fastText's hidden).
//! - **Pruned dictionary**: hashed n-grams map through the sidecar
//!   `pruneidx` table; unmapped n-grams are dropped entirely
//!   (`Dictionary::pushHash`).
//! - **Hidden**: the f32 sum of the input rows scaled by `1/N`.
//! - **Scoring**: hierarchical softmax — the Huffman tree rebuilt from
//!   label counts (same construction, same tie behavior), sigmoid
//!   computed through f64 and returned as f32 (the shipped bindings
//!   use the exact formula, not the 0.9.2 lookup table), `log(x + 1e-5)`
//!   per tree step, `exp` of the f32 log-probability.
//!
//! Parity: labels equal and scores within 5e-4 of the gem's
//! `LanguageIdentifier` output on the frozen corpus (the int8
//! re-quantization drift measured by the models repo gate; the
//! pre-quantization math matches to 1 ulp).
//!
//! # Memory
//!
//! A `LidModel` owns ~1 MB of int8 rows + scales + ~1.8 MB of sidecar
//! tables — approximately the artifact pair's own size. Detection
//! dequantizes one row at a time into a 16-element scratch vector; the
//! full fp32 matrix is never materialized.

use std::collections::HashMap;
use std::fmt;

use crate::rerank::dequant::dequant_row_int8;
use crate::rerank::onnx_wire::{
    DATA_TYPE_FLOAT, DATA_TYPE_INT8, Reader, Tensor, field, metadata_entry, parse_graph,
};

/// Errors of the LID reader.
#[derive(Debug)]
pub enum LidModelError {
    /// The `.onnx` bytes are not a parsable ONNX protobuf or miss the
    /// expected tensors (`q_input`, `row_scale`, `output_weights`).
    Onnx(String),
    /// The `.vocab.json` sidecar could not be parsed or disagrees with
    /// the matrix.
    Vocab(String),
    /// The graph metadata is missing or malformed.
    Metadata(String),
    /// A quantization descriptor this reader cannot handle.
    UnsupportedQuantization(String),
}

impl fmt::Display for LidModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Onnx(source) => write!(f, "lid model: {source}"),
            Self::Vocab(source) => write!(f, "lid vocabulary: {source}"),
            Self::Metadata(source) => write!(f, "lid metadata: {source}"),
            Self::UnsupportedQuantization(descriptor) => write!(
                f,
                "lid quantization {descriptor:?} not readable (known: int8-per-row)"
            ),
        }
    }
}

impl std::error::Error for LidModelError {}

/// One detection: the top label code (ISO 639-1 or 639-3 as the model
/// labels it) and its probability, the gem's
/// `LanguageIdentifier::DetectionResult` pair.
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    /// Label code without the `__label__` prefix (`"en"`, `"zh"`).
    pub code: String,
    /// Probability of the top label (fastText predict scores sum to
    /// 1 over the tree — plus the model's `1e-5` log epsilon).
    pub score: f32,
}

/// fastText `Dictionary::hash` — FNV-1a over sign-extended bytes.
fn fnv1a(bytes: &[u8]) -> u32 {
    let mut hash = 2_166_136_261u32;
    for &byte in bytes {
        // uint32_t(int8_t(c)): bytes >= 0x80 become 0xFFFFFF00 | c.
        let extended = (byte as i8) as u32; // uint32_t(int8_t(c))
        hash ^= extended;
        hash = hash.wrapping_mul(16_777_619);
    }
    hash
}

/// The `.vocab.json` sidecar shape the models repo writes
/// (`scripts/build_lid.py`).
#[derive(serde::Deserialize)]
struct VocabFile {
    labels: Vec<String>,
    label_counts: Vec<i64>,
    words: Vec<String>,
    pruneidx: Vec<(i32, i32)>,
    args: SidecarArgs,
}

#[derive(serde::Deserialize)]
struct SidecarArgs {
    dim: usize,
    bucket: u32,
    minn: usize,
    maxn: usize,
    #[serde(default = "one")]
    word_ngrams: u32,
    nwords: u32,
    #[serde(default = "default_eos")]
    eos: String,
}

fn one() -> u32 {
    1
}

fn default_eos() -> String {
    "</s>".to_owned()
}

/// One label's route through the Huffman tree: `path[k]` is the output
/// row of the k-th internal node from the leaf toward the root, and
/// `code[k]` says which child was taken (`true` = right = the sigmoid
/// is added, `false` = left = the complement is added).
#[derive(Debug)]
struct LabelRoute {
    path: Vec<u16>,
    code: Vec<bool>,
}

/// One loaded language-identification model: the int8 input matrix and
/// scales, the output matrix, the label routes, and the dictionary
/// tables — everything fastText needs at predict time.
#[derive(Debug)]
pub struct LidModel {
    /// Quantized input rows, row-major `[rows, dim]` (empty for the
    /// float32 fallback artifact).
    q: Vec<i8>,
    /// Per-row dequantization scales `[rows]`.
    scales: Vec<f32>,
    /// Float32 fallback rows, row-major `[rows, dim]` (empty for the
    /// int8 artifact).
    dense: Vec<f32>,
    /// Output (internal-node) weights, row-major `[labels - 1, dim]`.
    wo: Vec<f32>,
    dims: usize,
    labels: Vec<String>,
    routes: Vec<LabelRoute>,
    words: HashMap<String, u32>,
    pruneidx: HashMap<u32, u32>,
    /// Word-row count: hashed-ngram rows start at this offset.
    nwords: u32,
    bucket: u32,
    minn: usize,
    maxn: usize,
    word_ngrams: u32,
    eos: String,
}

impl LidModel {
    /// Parse the artifact pair exactly as distributed: `onnx_bytes` the
    /// `.onnx` container, `vocab_bytes` its `.vocab.json` sidecar.
    pub fn parse(onnx_bytes: &[u8], vocab_bytes: &[u8]) -> Result<Self, LidModelError> {
        let vocab: VocabFile = serde_json::from_slice(vocab_bytes)
            .map_err(|error| LidModelError::Vocab(error.to_string()))?;
        if vocab.labels.len() != vocab.label_counts.len() {
            return Err(LidModelError::Vocab(format!(
                "{} labels but {} counts",
                vocab.labels.len(),
                vocab.label_counts.len()
            )));
        }
        let LidOnnxData {
            q,
            scales,
            dense,
            wo,
            rows,
            dims,
        } = walk_onnx(onnx_bytes)?;
        if dims != vocab.args.dim {
            return Err(LidModelError::Vocab(format!(
                "sidecar dim {} disagrees with graph {}",
                vocab.args.dim, dims
            )));
        }
        if rows != vocab.args.nwords as usize + vocab.pruneidx.len() {
            return Err(LidModelError::Vocab(format!(
                "sidecar nwords {} + {} pruned ngrams disagree with {rows} matrix rows",
                vocab.args.nwords,
                vocab.pruneidx.len()
            )));
        }
        if wo.len() != vocab.labels.len() * dims {
            return Err(LidModelError::Vocab(format!(
                "output matrix holds {} rows, labels say {}",
                wo.len() / dims,
                vocab.labels.len()
            )));
        }

        let routes = build_routes(&vocab.label_counts);
        let words = vocab
            .words
            .iter()
            .enumerate()
            .map(|(index, word)| (word.clone(), u32::try_from(index).unwrap_or(u32::MAX)))
            .collect();

        Ok(Self {
            q,
            scales,
            dense,
            wo,
            dims,
            labels: vocab.labels,
            routes,
            words,
            pruneidx: vocab
                .pruneidx
                .into_iter()
                .map(|(key, value)| {
                    (
                        u32::try_from(key).unwrap_or(u32::MAX),
                        u32::try_from(value).unwrap_or(u32::MAX),
                    )
                })
                .collect(),
            nwords: vocab.args.nwords,
            bucket: vocab.args.bucket,
            minn: vocab.args.minn,
            maxn: vocab.args.maxn,
            word_ngrams: vocab.args.word_ngrams,
            eos: vocab.args.eos,
        })
    }

    /// The number of input rows (word rows + pruned-ngram rows).
    pub fn rows(&self) -> usize {
        self.q.len() / self.dims.max(1) + self.dense.len() / self.dims.max(1)
    }

    /// The label codes, in tree-leaf (dictionary) order.
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Dequantized input row `index` (int8 artifact) or the float32
    /// row (fallback artifact) — the graph's `Cast + Mul` path.
    fn row(&self, index: usize) -> Vec<f32> {
        debug_assert!(self.dims > 0);
        if self.dense.is_empty() {
            let start = index * self.dims;
            dequant_row_int8(&self.q[start..start + self.dims], self.scales[index])
        } else {
            let start = index * self.dims;
            self.dense[start..start + self.dims].to_vec()
        }
    }

    /// `Dictionary::pushHash`: a hashed n-gram row, or `None` when the
    /// quantized dictionary pruned that n-gram away.
    fn ngram_row(&self, hashed: u32) -> Option<u32> {
        if self.pruneidx.is_empty() {
            return self.nwords.checked_add(hashed);
        }
        let mapped = *self.pruneidx.get(&hashed)?;
        self.nwords.checked_add(mapped)
    }

    /// `Dictionary::computeSubwords`: the hashed n-gram rows of one
    /// `<word>` bracketed token, byte-walked so UTF-8 continuation
    /// bytes extend an n-gram instead of starting one.
    fn compute_subwords(&self, word: &[u8], out: &mut Vec<u32>) {
        let length = word.len();
        for i in 0..length {
            if word[i] & 0xC0 == 0x80 {
                continue;
            }
            let mut j = i;
            let mut n = 1usize;
            while j < length && n <= self.maxn {
                j += 1;
                while j < length && word[j] & 0xC0 == 0x80 {
                    j += 1;
                }
                n += 1;
                let char_len = n - 1;
                if char_len >= self.minn
                    && !(char_len == 1 && (i == 0 || j == length))
                    && let Some(row) = self.ngram_row(fnv1a(&word[i..j]) % self.bucket)
                {
                    out.push(row);
                }
            }
        }
    }

    /// `Dictionary::addSubwords` + `getLine`: the feature rows of one
    /// token.
    fn add_subwords(&self, token: &str, out: &mut Vec<u32>) {
        let word_id = self.words.get(token).copied();
        if let Some(id) = word_id {
            out.push(id);
            if token != self.eos {
                let bracketed = format!("<{token}>");
                self.compute_subwords(bracketed.as_bytes(), out);
            }
        } else if token != self.eos {
            let bracketed = format!("<{token}>");
            self.compute_subwords(bracketed.as_bytes(), out);
        }
    }

    /// The feature rows of `text` as fastText's `getLine` produces them
    /// for one predict line: whitespace tokens plus the trailing EOS.
    fn features(&self, text: &str) -> Vec<u32> {
        let mut features = Vec::new();
        let mut word_hashes: Vec<u32> = Vec::new();
        let bytes = text.as_bytes();
        let mut start = 0;
        for (index, &byte) in bytes.iter().enumerate() {
            if byte.is_ascii_whitespace() || byte == 0 || byte == 0x0B {
                if index > start {
                    let token = &text[start..index];
                    self.add_subwords(token, &mut features);
                    word_hashes.push(fnv1a(&bytes[start..index]));
                }
                start = index + 1;
            }
        }
        if start < bytes.len() {
            let token = &text[start..];
            self.add_subwords(token, &mut features);
            word_hashes.push(fnv1a(&bytes[start..]));
        }
        self.add_subwords(&self.eos, &mut features);
        self.add_word_ngrams(&mut features, &word_hashes);
        features
    }

    /// `Dictionary::addWordNgrams`: word-order n-gram hashes (unused by
    /// lid.176, which trains with wordNgrams=1, but part of the
    /// semantics the sidecar args carry).
    fn add_word_ngrams(&self, line: &mut Vec<u32>, hashes: &[u32]) {
        if self.word_ngrams <= 1 {
            return;
        }
        for i in 0..hashes.len() {
            let mut hash = u64::from(hashes[i]);
            for &next in hashes
                .iter()
                .skip(i + 1)
                .take(self.word_ngrams as usize - 1)
            {
                hash = hash.wrapping_mul(116_049_371).wrapping_add(u64::from(next));
                if let Some(row) = self.ngram_row((hash % u64::from(self.bucket)) as u32) {
                    line.push(row);
                }
            }
        }
    }

    /// Detect the language of `text`: the top label and its
    /// probability, matching the gem's `LanguageIdentifier#detect`
    /// (its python fasttext path) exactly up to the int8 drift.
    pub fn detect(&self, text: &str) -> Detection {
        let features = self.features(text);
        // fastText predict on an empty line still scores the EOS token;
        // a token stream with no features at all cannot happen (EOS is
        // in the dictionary), but an artifact whose EOS row vanished
        // would divide by zero — guard honestly instead.
        if features.is_empty() {
            return Detection {
                code: String::new(),
                score: 0.0,
            };
        }

        // computeHidden: f32 row sum, then scale by 1/N.
        let mut hidden = vec![0.0f32; self.dims];
        for &feature in &features {
            let row = self.row(feature as usize);
            for (slot, value) in hidden.iter_mut().zip(row) {
                *slot += value;
            }
        }
        let scale = (1.0 / features.len() as f64) as f32;
        for slot in &mut hidden {
            *slot *= scale;
        }

        // Hierarchical softmax: score every leaf, keep the best.
        let mut best = (0usize, f32::NEG_INFINITY);
        for (index, route) in self.routes.iter().enumerate() {
            let mut logprob = 0.0f32;
            for (&node, &bit) in route.path.iter().zip(&route.code) {
                let row =
                    &self.wo[node as usize * self.dims..node as usize * self.dims + self.dims];
                let mut dot = 0.0f32;
                for (weight, value) in row.iter().zip(&hidden) {
                    dot += weight * value;
                }
                let probability = (1.0 / (1.0 + (-(dot as f64)).exp())) as f32;
                let x = if bit { probability } else { 1.0 - probability };
                logprob += ((f64::from(x) + 1e-5).ln()) as f32;
            }
            let score = (f64::from(logprob).exp()) as f32;
            if score > best.1 {
                best = (index, score);
            }
        }

        Detection {
            code: self.labels[best.0].clone(),
            score: best.1,
        }
    }
}

/// Walk the ONNX container: metadata gates plus the three tensors.
/// The pieces an LID ONNX file yields: the (int8, f32 scales) or
/// (f32 rows) input matrix, the (labels, dim) output matrix, row count,
/// and dimensionality.
struct LidOnnxData {
    q: Vec<i8>,
    scales: Vec<f32>,
    dense: Vec<f32>,
    wo: Vec<f32>,
    rows: usize,
    dims: usize,
}

fn walk_onnx(bytes: &[u8]) -> Result<LidOnnxData, LidModelError> {
    let bad = LidModelError::Onnx;
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

    if let Some(model_type) = metadata.get("model_type")
        && model_type != "fasttext_lid"
    {
        return Err(LidModelError::Metadata(format!(
            "model_type {model_type:?} is not the fasttext LID graph"
        )));
    }
    match metadata.get("quantization").map(String::as_str) {
        Some("int8-per-row") | None => {}
        Some(other) => {
            return Err(LidModelError::UnsupportedQuantization(other.to_owned()));
        }
    }
    let dims = metadata
        .get("embedding_dimension")
        .and_then(|value| value.parse::<usize>().ok())
        .ok_or_else(|| {
            LidModelError::Metadata("embedding_dimension missing or malformed".to_owned())
        })?;

    let find = |name: &str| {
        tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .ok_or_else(|| bad(format!("graph has no {name} tensor")))
    };
    let q_input = find("q_input")?;
    let row_scale = find("row_scale")?;
    let output = find("output_weights")?;

    let expected_rows =
        |tensor: &Tensor| -> usize { tensor.dims.first().copied().unwrap_or(0) as usize };
    let rows = expected_rows(q_input);
    if rows == 0 || rows != expected_rows(row_scale) {
        return Err(bad(format!(
            "q_input/row_scale row counts disagree ({rows} vs {})",
            expected_rows(row_scale)
        )));
    }

    let float_payload = |tensor: &Tensor, expected: usize| -> Result<Vec<f32>, LidModelError> {
        if !tensor.raw_data.is_empty() {
            if tensor.raw_data.len() != expected * 4 {
                return Err(bad(format!(
                    "{} raw_data is {} bytes, expected {}",
                    tensor.name,
                    tensor.raw_data.len(),
                    expected * 4
                )));
            }
            return Ok(tensor
                .raw_data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| f32::from_le_bytes(*chunk))
                .collect());
        }
        if tensor.float_data.len() == expected {
            return Ok(tensor.float_data.clone());
        }
        Err(bad(format!(
            "{} has neither raw_data nor float_data with {expected} values",
            tensor.name
        )))
    };

    let output_dims = output.dims.first().copied().unwrap_or(0) as usize;
    let output_values = float_payload(output, output_dims * dims)?;

    let (q, scales, dense) = if q_input.data_type == DATA_TYPE_INT8 {
        if q_input.raw_data.len() != rows * dims {
            return Err(bad(format!(
                "q_input raw_data is {} bytes, expected {}",
                q_input.raw_data.len(),
                rows * dims
            )));
        }
        let scales = float_payload(row_scale, rows)?;
        if scales.len() != rows {
            return Err(bad("row_scale payload disagrees with row count".to_owned()));
        }
        (
            q_input.raw_data.iter().map(|byte| *byte as i8).collect(),
            scales,
            Vec::new(),
        )
    } else if q_input.data_type == DATA_TYPE_FLOAT {
        let values = float_payload(q_input, rows * dims)?;
        (Vec::new(), Vec::new(), values)
    } else {
        return Err(bad(format!(
            "q_input is neither int8 nor float (data_type {})",
            q_input.data_type
        )));
    };

    Ok(LidOnnxData {
        q,
        scales,
        dense,
        wo: output_values,
        rows,
        dims,
    })
}

/// The Huffman routes of every label — `HierarchicalSoftmaxLoss::
/// buildTree` verbatim, including its tie behavior (equal counts merge
/// with the newest internal node, not the leaf).
fn build_routes(counts: &[i64]) -> Vec<LabelRoute> {
    let leaves = counts.len();
    let nodes = 2 * leaves - 1;
    let mut parent = vec![-1i32; nodes];
    let mut binary = vec![false; nodes];
    let mut total = vec![i64::MAX; nodes];
    for (i, &count) in counts.iter().enumerate() {
        total[i] = count;
    }
    let mut leaf = leaves as i32 - 1;
    let mut node = leaves as i32;
    for i in leaves..nodes {
        let mut mini = [0i32; 2];
        for slot in &mut mini {
            if leaf >= 0 && total[leaf as usize] < total[node as usize] {
                *slot = leaf;
                leaf -= 1;
            } else {
                *slot = node;
                node += 1;
            }
        }
        total[i] = total[mini[0] as usize].saturating_add(total[mini[1] as usize]);
        parent[mini[0] as usize] = i as i32;
        parent[mini[1] as usize] = i as i32;
        binary[mini[1] as usize] = true;
    }

    let mut routes = Vec::with_capacity(leaves);
    for i in 0..leaves {
        let mut path = Vec::new();
        let mut code = Vec::new();
        let mut j = i as i32;
        while parent[j as usize] != -1 {
            let p = parent[j as usize];
            path.push((p - leaves as i32) as u16);
            code.push(binary[j as usize]);
            j = p;
        }
        routes.push(LabelRoute { path, code });
    }
    routes
}

#[cfg(test)]
mod tests {
    use super::*;

    /// fnv1a values frozen against fastText's Dictionary::hash.
    #[test]
    fn fnv1a_matches_fasttext_including_sign_extension() {
        assert_eq!(fnv1a(b"the"), 3_020_861_980);
        assert_eq!(fnv1a(b"hello"), 1_335_831_723);
        // 0xC3 0xBC is "ü" in UTF-8: both bytes sign-extend.
        assert_eq!(fnv1a("ü".as_bytes()), 804_934_730);
        assert_eq!(fnv1a("今日".as_bytes()), 1_974_294_176);
    }

    /// A toy Huffman tree: counts [5, 2, 1] merge to the same routes
    /// the C++ buildTree produces (leaf ties lose to internal nodes).
    #[test]
    fn huffman_routes_match_the_cpp_construction() {
        let routes = build_routes(&[5, 2, 1]);
        // Tree: leaves 0(5) 1(2) 2(1); merge leaves 1+2 -> node 3(3);
        // merge leaf 0 + node 3 -> root 4. The leaf side of every
        // merge is the right child (binary=true). Leaf 0 walks up via
        // node 3, then root 4. Frozen from the reference.
        assert_eq!(routes[0].path, [1]);
        assert_eq!(routes[0].code, [true]);
        assert_eq!(routes[1].path, [0, 1]);
        assert_eq!(routes[1].code, [true, false]);
        assert_eq!(routes[2].path, [0, 1]);
        assert_eq!(routes[2].code, [false, false]);
    }

    /// The real-artifact fixture: kotoshu/tests/fixtures/lid/ holds the
    /// committed registry artifact pair (the models repo output, MIT
    /// license, see the fixture dir LICENSE-NOTE.md) and the gem-frozen
    /// parity corpus. Derived by scripts/make_lid_fixture.py.
    const FIXTURE_ONNX: &[u8] = include_bytes!("../../tests/fixtures/lid/lid.176.onnx");
    const FIXTURE_VOCAB: &[u8] = include_bytes!("../../tests/fixtures/lid/lid.176.vocab.json");

    fn fixture() -> LidModel {
        LidModel::parse(FIXTURE_ONNX, FIXTURE_VOCAB).expect("lid fixture parses")
    }

    #[test]
    fn fixture_parses_with_expected_shape() {
        let model = fixture();
        assert_eq!(model.labels().len(), 176);
        assert_eq!(model.rows(), 50_000);
        assert_eq!(model.labels()[0], "en");
        // The pruned dictionary keeps 7235 word rows.
        assert!(model.words.contains_key("de"));
        // "the" did not survive the upstream cutoff - it embeds through
        // its n-grams, exactly like the bindings.
        assert!(!model.words.contains_key("the"));
        assert_eq!(model.words.get("</s>"), Some(&0));
    }

    #[test]
    fn subword_rows_match_the_validated_reference() {
        let model = fixture();
        // ngrams of "<the>": only "<the>" survived the prune (row 34687
        // = nwords 7235 + pruneidx 27452); frozen from the reference.
        let mut rows = Vec::new();
        model.compute_subwords(b"<the>", &mut rows);
        assert_eq!(rows, [34_687]);

        // "ü" -> "<ü>": all three byte n-grams survived the prune;
        // rows frozen from the reference (nwords 7235 + pruneidx).
        let mut umlaut = Vec::new();
        model.compute_subwords("<ü>".as_bytes(), &mut umlaut);
        assert_eq!(umlaut, [7_235 + 1_558, 7_235 + 16_840, 7_235 + 3_618]);
    }

    #[test]
    fn detect_matches_the_gem_frozen_scores() {
        let model = fixture();
        // Frozen from the gem's LanguageIdentifier on lid.176.ftz
        // (scripts/make_lid_fixture.py); full corpus in the integration
        // test. Tolerance = the models-repo int8 gate drift (4.8e-4).
        let cases = [
            (
                "the quick brown fox jumps over the lazy dog",
                "en",
                0.751_221_8_f32,
            ),
            (
                "今日はとても良い天気ですね、散歩に行きましょう",
                "ja",
                0.997_666_2_f32,
            ),
            (
                "Быстрая бурая лиса прыгает через ленивую собаку",
                "ru",
                0.973_018_7_f32,
            ),
            (
                "Esta é uma frase de teste para verificar a deteção de idioma",
                "pt",
                0.999_211_9_f32,
            ),
        ];
        for (text, code, score) in cases {
            let detection = model.detect(text);
            assert_eq!(detection.code, code, "{text:?}");
            assert!(
                (detection.score - score).abs() < 1e-3,
                "{text:?}: {} vs {score}",
                detection.score
            );
        }
    }

    #[test]
    fn empty_text_scores_the_eos_row_like_the_gem() {
        // Empty input predicts through the EOS token alone; the gem
        // returns en 0.12450418 there. Frozen.
        let model = fixture();
        let detection = model.detect("");
        assert_eq!(detection.code, "en");
        assert!((detection.score - 0.124_504_18_f32).abs() < 1e-3);
    }

    #[test]
    fn rejected_artifacts_fail_loudly() {
        // Not ONNX bytes.
        assert!(matches!(
            LidModel::parse(b"nonsense", FIXTURE_VOCAB),
            Err(LidModelError::Onnx(_))
        ));
        // Sidecar that disagrees with the matrix.
        let mut vocab: serde_json::Value =
            serde_json::from_slice(FIXTURE_VOCAB).expect("fixture vocab is json");
        vocab["args"]["dim"] = serde_json::json!(32);
        let bytes = serde_json::to_vec(&vocab).expect("serialize");
        assert!(matches!(
            LidModel::parse(FIXTURE_ONNX, &bytes),
            Err(LidModelError::Vocab(_))
        ));
    }
}
