//! Row-quantized embedding dequantization — B1 groundwork (plan 68),
//! dequant math only.
//!
//! The current tier artifacts (`kotoshu://models/{lang}/mini|fluency`,
//! `scripts/build_tiers.py` in the models repo) store the embedding
//! matrix as an int8-per-row graph: `Constant q_embeddings` int8 `[V, d]`
//! + `row_scale` fp32 `[V]`, with `Gather + Gather + Reshape + Cast +
//! Mul + Squeeze` dequantizing inside the ONNX session, and metadata
//!   `quantization = "int8-per-row"`. The full tier carries a float32
//!   matrix with no `quantization` metadata.
//!
//! The future int4 tier (B1: int4 group-128 of the full tier, 120 MB →
//! ~15–20 MB near-lossless per [arxiv 2501.10534]) is accepted *now* by
//! its format byte so the reading side never needs a breaking change
//! when the models repo ships the artifacts. The artifacts themselves
//! are a models-repo task and do not exist yet; until then the int4 path
//! is exercised by the synthetic-tensor unit tests below.
//!
//! ## Format bytes
//!
//! A one-byte row-storage tag derived from the ONNX metadata
//! `quantization` string:
//!
//! | metadata                    | byte | row storage                         |
//! |-----------------------------|------|-------------------------------------|
//! | _absent_                    | 0x00 | float32 (full tier)                 |
//! | `"int8-per-row"`            | 0x08 | signed i8 × fp32 row scale          |
//! | `"int4-per-row"` (future B1)| 0x04 | packed signed nibbles × row scale   |
//!
//! ## int4-per-row packing (the future artifact contract)
//!
//! Mirrors the int8 recipe (`scale = max_abs / 7.0`, values clipped to
//! the signed-nibble range `[-8, 7]` — the symmetric-positive divisor
//! 7, not 8, so `+max_abs` round-trips exactly, matching the int8
//! tier's `max_abs / 127.0` convention in `quantize_int8_per_row`):
//! each row of `d` elements occupies `ceil(d / 2)` bytes; element `i`
//! lives in byte `i / 2` — even `i` in the HIGH nibble, odd `i` in the
//! LOW nibble — as two's-complement signed nibbles; the dequantized
//! value is `nibble as f32 * row_scale`.

/// Format byte: float32 rows (no quantization; the `full` tier).
pub const FORMAT_BYTE_FP32: u8 = 0x00;

/// Format byte: `int8-per-row` (the current mini/fluency tiers).
pub const FORMAT_BYTE_INT8_PER_ROW: u8 = 0x08;

/// Format byte: `int4-per-row` (the future B1 full-tier artifacts; not
/// yet produced by the models repo).
pub const FORMAT_BYTE_INT4_PER_ROW: u8 = 0x04;

/// Metadata value of the current quantized tiers (models repo
/// `build_tiers.py`).
pub const QUANT_INT8_PER_ROW: &str = "int8-per-row";

/// Metadata value of the future B1 artifacts. The models-repo converter
/// must emit exactly this string for [`RowFormat::from_metadata`] to
/// accept them.
pub const QUANT_INT4_PER_ROW: &str = "int4-per-row";

/// The row-storage format of an embedding artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowFormat {
    /// float32 rows (no quantization metadata; the `full` tier).
    Fp32,
    /// Signed int8 rows scaled by one fp32 factor per row.
    Int8PerRow,
    /// Packed signed-int4 rows scaled by one fp32 factor per row
    /// (future B1 artifacts).
    Int4PerRow,
}

impl RowFormat {
    /// The documented one-byte tag for this format.
    pub fn format_byte(self) -> u8 {
        match self {
            Self::Fp32 => FORMAT_BYTE_FP32,
            Self::Int8PerRow => FORMAT_BYTE_INT8_PER_ROW,
            Self::Int4PerRow => FORMAT_BYTE_INT4_PER_ROW,
        }
    }

    /// Parse the ONNX metadata `quantization` string (as written by the
    /// models-repo converters). `None` (metadata absent) is the float32
    /// full tier; unknown strings are unsupported.
    pub fn from_metadata(quantization: Option<&str>) -> Option<Self> {
        match quantization {
            None => Some(Self::Fp32),
            Some(QUANT_INT8_PER_ROW) => Some(Self::Int8PerRow),
            Some(QUANT_INT4_PER_ROW) => Some(Self::Int4PerRow),
            Some(other) => {
                let _ = other;
                None
            }
        }
    }

    /// Parse a format byte (the inverse of [`RowFormat::format_byte`]).
    pub fn from_format_byte(byte: u8) -> Option<Self> {
        match byte {
            FORMAT_BYTE_FP32 => Some(Self::Fp32),
            FORMAT_BYTE_INT8_PER_ROW => Some(Self::Int8PerRow),
            FORMAT_BYTE_INT4_PER_ROW => Some(Self::Int4PerRow),
            _ => None,
        }
    }
}

/// Dequantize one int8-per-row row: `value = q as f32 * scale` (the
/// graph's `Cast + Mul`, reproduced host-side).
pub fn dequant_row_int8(q: &[i8], scale: f32) -> Vec<f32> {
    q.iter().map(|q| f32::from(*q) * scale).collect()
}

/// Quantize one row the way the models repo does
/// (`quantize_int8_per_row`: `scale = max_abs / 127`, `0 → 1` scale
/// guard, `rint` and clip) — used by the parity test below so the Rust
/// dequant is checked against the real converter recipe.
#[cfg(test)]
pub(crate) fn quantize_row_int8(y: &[f32]) -> (Vec<i8>, f32) {
    let max_abs = y.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let mut scale = max_abs / 127.0;
    if scale == 0.0 {
        scale = 1.0;
    }
    let q = y
        .iter()
        .map(|v| (v / scale).round().clamp(-127.0, 127.0) as i8)
        .collect();
    (q, scale)
}

/// Dequantize one int4-per-row row (see the module docs for the packing
/// contract). `packed` holds `ceil(d / 2)` bytes; an odd trailing low
/// nibble (a padding nibble in odd-width rows) is ignored.
pub fn dequant_row_int4(packed: &[u8], scale: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(packed.len() * 2);
    for &byte in packed {
        for nibble in [byte >> 4, byte & 0x0f] {
            // Two's-complement sign extension from 4 bits to i8.
            let value = ((nibble << 4) as i8) >> 4;
            out.push(f32::from(value) * scale);
        }
    }
    out
}

/// Pack one int4-per-row row (the inverse of [`dequant_row_int4`]);
/// values must already be in `[-8, 7]`. Test/helper side of the future
/// artifact contract.
pub fn pack_row_int4(values: &[i8]) -> Vec<u8> {
    debug_assert!(values.iter().all(|v| (-8..=7).contains(v)));
    values
        .chunks(2)
        .map(|pair| {
            let high = (pair[0] & 0x0f) as u8;
            let low = pair.get(1).map_or(0, |v| (v & 0x0f) as u8);
            (high << 4) | low
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_byte_round_trip() {
        for format in [
            RowFormat::Fp32,
            RowFormat::Int8PerRow,
            RowFormat::Int4PerRow,
        ] {
            assert_eq!(
                RowFormat::from_format_byte(format.format_byte()),
                Some(format)
            );
        }
        assert_eq!(RowFormat::from_format_byte(0x7f), None);
    }

    #[test]
    fn metadata_parsing_matches_the_converters() {
        assert_eq!(RowFormat::from_metadata(None), Some(RowFormat::Fp32));
        assert_eq!(
            RowFormat::from_metadata(Some(QUANT_INT8_PER_ROW)),
            Some(RowFormat::Int8PerRow)
        );
        assert_eq!(
            RowFormat::from_metadata(Some(QUANT_INT4_PER_ROW)),
            Some(RowFormat::Int4PerRow)
        );
        assert_eq!(RowFormat::from_metadata(Some("int4-group-128")), None);
        // The documented constants agree with the tier metadata strings.
        assert_eq!(QUANT_INT8_PER_ROW, "int8-per-row");
        assert_eq!(QUANT_INT4_PER_ROW, "int4-per-row");
    }

    #[test]
    fn int8_dequant_matches_the_converter_recipe() {
        // A row quantized exactly like build_tiers.py must dequantize
        // within the converter's own gate (QUANT_MAX_ABS_TOL = 0.05).
        let row = [0.51, -1.2, 3.33, 0.0, -0.007, 2.0];
        let (q, scale) = quantize_row_int8(&row);
        assert!(q.iter().all(|v| (-127..=127).contains(v)));
        let dequant = dequant_row_int8(&q, scale);
        for (y, d) in row.iter().zip(&dequant) {
            assert!((y - d).abs() < 0.05, "{y} vs {d}");
        }
        // Zero rows keep the scale guard (scale 1.0), not a zero scale.
        let (q0, s0) = quantize_row_int8(&[0.0, 0.0]);
        assert_eq!((q0.as_slice(), s0), (&[0i8, 0][..], 1.0));
    }

    #[test]
    fn int8_dequant_is_exact_for_representable_values() {
        let dequant = dequant_row_int8(&[3, -127, 0, 127], 0.1);
        let expected = [0.3f32, -12.7, 0.0, 12.7];
        for (got, want) in dequant.iter().zip(expected) {
            assert!((got - want).abs() < 1e-6);
        }
    }

    #[test]
    fn int4_synthetic_row_round_trips() {
        // Every representable nibble, including the asymmetric -8.
        let values: Vec<i8> = (-8..=7).collect();
        let packed = pack_row_int4(&values);
        // 16 values → 8 bytes, two nibbles each.
        assert_eq!(packed.len(), 8);
        let dequant = dequant_row_int4(&packed, 0.5);
        assert_eq!(dequant.len(), values.len());
        for (got, want) in dequant.iter().zip(&values) {
            assert_eq!(*got, f32::from(*want) * 0.5);
        }
    }

    #[test]
    fn int4_odd_width_pads_the_low_nibble() {
        // Odd width: the trailing low nibble is padding and must be
        // ignored (d = 3 → 2 bytes, 4 nibble slots, one padding).
        let packed = pack_row_int4(&[1, -2, 7, 0]);
        assert_eq!(packed.len(), 2);
        let dequant = dequant_row_int4(&packed, 1.0);
        assert_eq!(dequant[..3], [1.0, -2.0, 7.0]);
        assert_eq!(dequant.len(), 4); // caller slices to d
    }

    #[test]
    fn int4_negative_values_use_twos_complement() {
        // -8 = 0b1000 in both nibbles of 0x88; -1 = 0b1111 → 0xff.
        assert_eq!(dequant_row_int4(&[0x88], 1.0), [-8.0, -8.0]);
        assert_eq!(dequant_row_int4(&[0xff], 1.0), [-1.0, -1.0]);
        assert_eq!(dequant_row_int4(&[0x01], 2.0), [0.0, 2.0]);
    }
}
