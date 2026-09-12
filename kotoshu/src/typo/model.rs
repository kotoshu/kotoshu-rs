//! The frozen char-BiGRU typo bi-encoder (plan 131): a pure-Rust reader
//! and forward pass for the `typo.biencoder.onnx` artifact trained in
//! the models repo (plan 114, priced plan 115, verdict plan 128). The
//! graph is not interpreted generatively — the artifact is ours, its
//! architecture frozen and sha-pinned in the registry, so this reader
//! walks the ONNX wire format (the shared [`onnx_wire`] walker),
//! validates the graph against that frozen shape, and runs the forward
//! pass by hand. That keeps inference dependency-free and wasm-clean
//! (the `ort` provider needs a host `libonnxruntime`, which no browser
//! has) — the same philosophy as the int8 tier reader.
//!
//! # The frozen graph (verified against the shipped artifact)
//!
//! ```text
//! char_ids int64 [B, S<=24]
//!   → Gather(m.emb.weight_quantized uint8 [1823, 48])          (ids: pad=0,
//!     chars 1..=1821, unk = vocab_len+1 = 1822)
//!   → DequantizeLinear(per-tensor: (q − zp) · scale)
//!   → Transpose to [S, B, 48]
//!   → GRU(hidden 96, bidirectional, linear_before_reset=1,
//!        W [2,288,48], R [2,288,96], B [2,576], h₀ = 0)
//!   → Transpose/Reshape/Transpose to [B, S, 192]
//!   → masked mean over valid steps (ids ≠ 0), denominator clamped ≥ 1
//!   → DynamicQuantizeLinear → MatMulInteger with m.proj.weight_quantized
//!     int8 [192, 256] → dequantize → Add(m.proj.bias)            (the
//!     runtime-quantized projection the dynamic quantizer produced)
//!   → L2 normalize (norm clamped ≥ 1e-12) → embedding [256]
//! ```
//!
//! Gate order in the GRU tensors is ONNX's (z, r, n); ONNX
//! `linear_before_reset=1` is exactly PyTorch's GRU recurrence, which
//! the trainer exported.

use std::collections::HashMap;
use std::fmt;

use crate::rerank::onnx_wire;

/// Input truncation — the trainer's `MAX_WORD_LEN`.
pub const MAX_WORD_LEN: usize = 24;
/// Char-embedding width, GRU width, output width — frozen architecture.
pub const CHAR_DIM: usize = 48;
pub const GRU_DIM: usize = 96;
pub const OUT_DIM: usize = 256;
/// Input feature width of the projection: both GRU directions.
const PROJ_IN: usize = 2 * GRU_DIM;

/// Errors of the typo bi-encoder reader and encoder.
#[derive(Debug)]
pub enum TypoModelError {
    /// The `.onnx` bytes are not a parsable model, or the graph is not
    /// the frozen shape this reader accepts.
    Graph(String),
    /// The `.vocab.json` sibling is not the frozen shape (`char → id`).
    Vocab(String),
    /// `embed` on a word with no characters after truncation.
    EmptyWord,
}

impl fmt::Display for TypoModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Graph(s) => write!(f, "typo bi-encoder graph: {s}"),
            Self::Vocab(s) => write!(f, "typo bi-encoder char vocab: {s}"),
            Self::EmptyWord => write!(f, "typo bi-encoder: empty word"),
        }
    }
}

impl std::error::Error for TypoModelError {}

/// The frozen bi-encoder, weights loaded and ready.
pub struct TypoModel {
    char_to_id: HashMap<char, u32>,
    /// unk id = vocab_len + 1 (pad occupies 0; chars 1..=vocab_len).
    unk_id: u32,
    emb_rows: usize,
    emb_q: Vec<u8>,
    emb_scale: f32,
    emb_zero_point: u8,
    /// GRU weights, per direction: W [3H, I], R [3H, H], bias [6H].
    gru: [GruDir; 2],
    proj_q: Vec<i8>,
    proj_scale: f32,
    proj_zero_point: i8,
    proj_bias: Vec<f32>,
}

struct GruDir {
    /// z/r/n gate rows stacked: [3H, I].
    w: Vec<f32>,
    /// [3H, H].
    r: Vec<f32>,
    /// [Wb; Rb] — [6H].
    b: Vec<f32>,
}

impl TypoModel {
    /// Parse the frozen artifact pair exactly as distributed: the
    /// `.onnx` graph and its `.vocab.json` char-vocab sibling.
    pub fn parse(onnx_bytes: &[u8], char_vocab_json: &[u8]) -> Result<Self, TypoModelError> {
        let wire = onnx_wire::parse_model(onnx_bytes).map_err(TypoModelError::Graph)?;
        let graph = wire
            .graph
            .ok_or_else(|| TypoModelError::Graph("model carries no graph".to_owned()))?;
        let nodes = onnx_wire::parse_nodes(graph).map_err(TypoModelError::Graph)?;

        // Structural validation: the graph must be the frozen one.
        let gru_node = nodes
            .iter()
            .find(|n| n.op_type == "GRU")
            .ok_or_else(|| TypoModelError::Graph("no GRU node".to_owned()))?;
        if gru_node.str_attr("direction") != Some("bidirectional") {
            return Err(TypoModelError::Graph(
                "GRU is not bidirectional — not the frozen artifact".to_owned(),
            ));
        }
        if gru_node.int_attr("hidden_size") != Some(GRU_DIM as i64) {
            return Err(TypoModelError::Graph(
                "GRU hidden size is not 96".to_owned(),
            ));
        }
        if gru_node.int_attr("linear_before_reset") != Some(1) {
            return Err(TypoModelError::Graph(
                "GRU is not linear_before_reset=1 — not the frozen artifact".to_owned(),
            ));
        }

        fn tensor<'m>(
            wire: &'m onnx_wire::WireModel<'_>,
            name: &str,
        ) -> Result<&'m onnx_wire::Tensor<'m>, TypoModelError> {
            wire.tensor(name)
                .ok_or_else(|| TypoModelError::Graph(format!("missing initializer {name}")))
        }
        fn expected_dims(
            wire: &onnx_wire::WireModel<'_>,
            name: &str,
            want: &[i64],
        ) -> Result<(), TypoModelError> {
            let got = tensor(&wire, name)?.dims.clone();
            if got != want {
                return Err(TypoModelError::Graph(format!(
                    "{name} dims {got:?} != {want:?}"
                )));
            }
            Ok(())
        }
        fn floats(wire: &onnx_wire::WireModel<'_>, name: &str) -> Result<Vec<f32>, TypoModelError> {
            let t = tensor(&wire, name)?;
            let n = t
                .dims
                .iter()
                .try_fold(1usize, |acc, d| acc.checked_mul(*d as usize))
                .ok_or_else(|| TypoModelError::Graph(format!("{name} dims overflow")))?;
            onnx_wire::float_payload(t, n).map_err(TypoModelError::Graph)
        }
        fn scalar_f32(wire: &onnx_wire::WireModel<'_>, name: &str) -> Result<f32, TypoModelError> {
            floats(&wire, name)?
                .first()
                .copied()
                .ok_or_else(|| TypoModelError::Graph(format!("{name} is empty")))
        }
        fn byte_scalar(wire: &onnx_wire::WireModel<'_>, name: &str) -> Result<u8, TypoModelError> {
            let t = tensor(&wire, name)?;
            if !t.raw_data.is_empty() {
                return Ok(t.raw_data[0]);
            }
            // the ONNX helper serializes uint8/int8 scalars into
            // int32_data with empty raw_data — the shipped artifact does
            t.int32_data
                .first()
                .map(|v| *v as u8)
                .ok_or_else(|| TypoModelError::Graph(format!("{name} is empty")))
        }

        // char embedding table: uint8 [rows, 48]
        let emb_dims = tensor(&wire, "m.emb.weight_quantized")?.dims.clone();
        if emb_dims.len() != 2 || emb_dims[1] != CHAR_DIM as i64 || emb_dims[0] < 3 {
            return Err(TypoModelError::Graph(
                "embedding table is not [rows>=3, 48]".to_owned(),
            ));
        }
        let emb_rows = emb_dims[0] as usize;
        let emb_q = onnx_wire::uint8_payload(tensor(&wire, "m.emb.weight_quantized")?)
            .map_err(TypoModelError::Graph)?;
        if emb_q.len() != emb_rows * CHAR_DIM {
            return Err(TypoModelError::Graph(
                "embedding payload size mismatch".to_owned(),
            ));
        }
        let emb_scale = scalar_f32(&wire, "m.emb.weight_scale")?;
        let emb_zero_point = byte_scalar(&wire, "m.emb.weight_zero_point")?;

        // GRU tensors: W [2, 3H, I], R [2, 3H, H], B [2, 6H]
        expected_dims(
            &wire,
            "onnx::GRU_223",
            &[2, (3 * GRU_DIM) as i64, CHAR_DIM as i64],
        )?;
        expected_dims(
            &wire,
            "onnx::GRU_224",
            &[2, (3 * GRU_DIM) as i64, GRU_DIM as i64],
        )?;
        expected_dims(&wire, "onnx::GRU_222", &[2, (6 * GRU_DIM) as i64])?;
        let w = floats(&wire, "onnx::GRU_223")?;
        let r = floats(&wire, "onnx::GRU_224")?;
        let b = floats(&wire, "onnx::GRU_222")?;
        let per_w = 3 * GRU_DIM * CHAR_DIM;
        let per_r = 3 * GRU_DIM * GRU_DIM;
        let per_b = 6 * GRU_DIM;
        let mut w_fwd = w;
        let w_bwd = w_fwd.split_off(per_w);
        let mut r_fwd = r;
        let r_bwd = r_fwd.split_off(per_r);
        let mut b_fwd = b;
        let b_bwd = b_fwd.split_off(per_b);
        let gru = [
            GruDir {
                w: w_fwd,
                r: r_fwd,
                b: b_fwd,
            },
            GruDir {
                w: w_bwd,
                r: r_bwd,
                b: b_bwd,
            },
        ];

        // projection: int8 [192, 256] + per-tensor scale/zero + fp32 bias
        expected_dims(
            &wire,
            "m.proj.weight_quantized",
            &[PROJ_IN as i64, OUT_DIM as i64],
        )?;
        let proj_q =
            onnx_wire::int8_payload(tensor(&wire, "m.proj.weight_quantized")?, PROJ_IN, OUT_DIM)
                .map_err(TypoModelError::Graph)?;
        let proj_scale = scalar_f32(&wire, "m.proj.weight_scale")?;
        let proj_zero_point = byte_scalar(&wire, "m.proj.weight_zero_point")? as i8;
        expected_dims(&wire, "m.proj.bias", &[OUT_DIM as i64])?;
        let proj_bias = floats(&wire, "m.proj.bias")?;

        // char vocab: char → id (1-based); ids must be unique.
        let char_to_id: HashMap<char, u32> = serde_json::from_str::<HashMap<String, u32>>(
            std::str::from_utf8(char_vocab_json)
                .map_err(|e| TypoModelError::Vocab(e.to_string()))?,
        )
        .map_err(|e| TypoModelError::Vocab(e.to_string()))?
        .into_iter()
        .filter_map(|(k, v)| k.chars().next().map(|c| (c, v)))
        .collect();
        if char_to_id.is_empty() {
            return Err(TypoModelError::Vocab("empty char vocab".to_owned()));
        }
        let unk_id = char_to_id.len() as u32 + 1;
        if unk_id as usize >= emb_rows {
            return Err(TypoModelError::Graph(
                "embedding table too small for the unk id".to_owned(),
            ));
        }

        Ok(Self {
            char_to_id,
            unk_id,
            emb_rows,
            emb_q,
            emb_scale,
            emb_zero_point,
            gru,
            proj_q,
            proj_scale,
            proj_zero_point,
            proj_bias,
        })
    }

    /// Number of rows in the frozen char-embedding table.
    pub fn char_table_rows(&self) -> usize {
        self.emb_rows
    }

    /// Encode one word to its L2-normalized 256-d embedding, the
    /// eval-identical forward pass (the eval used unk = vocab_len + 1).
    pub fn embed(&self, word: &str) -> Result<[f32; OUT_DIM], TypoModelError> {
        // char ids, pad 0, truncate 24
        let mut ids = [0u32; MAX_WORD_LEN];
        let mut len = 0usize;
        for ch in word.chars().take(MAX_WORD_LEN) {
            ids[len] = *self.char_to_id.get(&ch).unwrap_or(&self.unk_id);
            len += 1;
        }
        if len == 0 {
            return Err(TypoModelError::EmptyWord);
        }

        // The graph has NO sequence-lengths input: the recurrence runs
        // over the full padded MAX_WORD_LEN window (pad id 0 → the
        // table's row 0) and only the POOL is masked. Running the valid
        // prefix alone would diverge — the backward direction starts
        // at the padding, and its state at real timesteps depends on
        // the trailing pad steps.
        let mut x = vec![0f32; MAX_WORD_LEN * CHAR_DIM];
        for (t, &id) in ids.iter().enumerate() {
            let row = id as usize * CHAR_DIM;
            for (j, xv) in x[t * CHAR_DIM..(t + 1) * CHAR_DIM].iter_mut().enumerate() {
                *xv = (i32::from(self.emb_q[row + j]) - i32::from(self.emb_zero_point)) as f32
                    * self.emb_scale;
            }
        }

        // BiGRU → [S, 192] (fwd; bwd concatenated at the feature level)
        let fwd = self.gru[0].run(&x, MAX_WORD_LEN, false);
        let bwd = self.gru[1].run(&x, MAX_WORD_LEN, true);
        let mut pooled = [0f32; PROJ_IN];
        for t in 0..len {
            for h in 0..GRU_DIM {
                pooled[h] += fwd[t * GRU_DIM + h];
                pooled[GRU_DIM + h] += bwd[t * GRU_DIM + h];
            }
        }
        let denom = f32::max(len as f32, 1.0);
        for v in pooled.iter_mut() {
            *v /= denom;
        }

        // runtime-quantized projection: DQL(pooled) → MatMulInteger →
        // dequantize → add bias
        let (q, a_scale, a_zp) = dynamic_quantize_linear(&pooled);
        let total_scale = a_scale * self.proj_scale;
        let mut out = [0f32; OUT_DIM];
        for j in 0..OUT_DIM {
            let mut acc: i32 = 0;
            for i in 0..PROJ_IN {
                acc += (i32::from(q[i]) - i32::from(a_zp))
                    * (i32::from(self.proj_q[i * OUT_DIM + j]) - i32::from(self.proj_zero_point));
            }
            out[j] = acc as f32 * total_scale + self.proj_bias[j];
        }

        // L2 normalize, norm clamped ≥ 1e-12 (the graph's Clip)
        let norm = out.iter().map(|v| v * v).sum::<f32>().sqrt();
        let norm = f32::max(norm, 1e-12);
        for v in out.iter_mut() {
            *v /= norm;
        }
        Ok(out)
    }
}

impl GruDir {
    /// One direction's forward pass over the embedded sequence:
    /// returns per-step outputs [S, 96] for this direction.
    fn run(&self, x: &[f32], len: usize, reverse: bool) -> Vec<f32> {
        let mut h = vec![0f32; GRU_DIM];
        let mut out = vec![0f32; len * GRU_DIM];
        let (wb, rb) = self.b.split_at(3 * GRU_DIM);
        for step in 0..len {
            let t = if reverse { len - 1 - step } else { step };
            let xt = &x[t * CHAR_DIM..(t + 1) * CHAR_DIM];
            // pre-activations: W·x + Wb  and  R·h + Rb (lbr=1 biases
            // the reset branch before the gate multiplies)
            let mut gates_w = [0f32; 3 * GRU_DIM];
            let mut gates_h = [0f32; 3 * GRU_DIM];
            for g in 0..3 * GRU_DIM {
                let mut s = wb[g];
                for (k, xv) in xt.iter().enumerate() {
                    s += self.w[g * CHAR_DIM + k] * xv;
                }
                gates_w[g] = s;
                let mut s = rb[g];
                for k in 0..GRU_DIM {
                    s += self.r[g * GRU_DIM + k] * h[k];
                }
                gates_h[g] = s;
            }
            // ONNX gate order: z, r, n
            for k in 0..GRU_DIM {
                let z = sigmoid(gates_w[k] + gates_h[k]);
                let r = sigmoid(gates_w[GRU_DIM + k] + gates_h[GRU_DIM + k]);
                let n = (gates_w[2 * GRU_DIM + k] + r * gates_h[2 * GRU_DIM + k]).tanh();
                h[k] = (1.0 - z) * n + z * h[k];
            }
            out[t * GRU_DIM..(t + 1) * GRU_DIM].copy_from_slice(&h);
        }
        out
    }
}

/// ONNX `DynamicQuantizeLinear` (int8): symmetric-range scale over the
/// adjusted min/max, zero point clamped into [-128, 127].
fn dynamic_quantize_linear(x: &[f32]) -> (Vec<i8>, f32, i8) {
    const QMIN: i32 = -128;
    const QMAX: i32 = 127;
    let mut amin = 0f32;
    let mut amax = 0f32;
    for &v in x {
        amin = amin.min(v);
        amax = amax.max(v);
    }
    let scale = (amax - amin) / (QMAX - QMIN) as f32;
    if scale == 0.0 {
        return (vec![0i8; x.len()], 1.0, 0);
    }
    let zp_unclamped = QMIN as f32 - (amin / scale).round();
    let zp = zp_unclamped.clamp(QMIN as f32, QMAX as f32).round() as i32;
    let q = x
        .iter()
        .map(|&v| {
            ((v / scale).round() as i32)
                .saturating_add(zp)
                .clamp(QMIN, QMAX) as i8
        })
        .collect();
    (q, scale, zp as i8)
}

fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}
