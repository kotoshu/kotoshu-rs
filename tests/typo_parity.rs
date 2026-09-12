//! Parity of the pure-Rust typo bi-encoder against onnxruntime on the
//! frozen artifact (plan 131's correctness gate): the reference
//! embeddings were produced by the models repo's eval path
//! (onnxruntime CPU, eval unk id, pad/truncate 24) and checked in
//! beside the artifact. The bar is ranking-equivalence, not bit
//! equality — the runtime-quantized projection rounds differently —
//! measured as cosine and per-element distance.

use kotoshu::typo::{MAX_WORD_LEN, OUT_DIM, TypoIndex, TypoModel};
use serde_json::Value;

// Synced from the models repo by scripts/sync_conformance.sh
// (KOTOSHU_MODELS_DIR); like the conformance fixtures, the artifact is
// never committed here and the parity assertions skip gracefully
// without it.
const FIXTURES_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../tests/fixtures");

fn artifacts() -> Option<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let read = |name: &str| -> Option<Vec<u8>> {
        std::fs::read(format!("{FIXTURES_DIR}/models/{name}")).ok()
    };
    Some((
        read("typo.biencoder.onnx")?,
        read("typo.biencoder.vocab.json")?,
        read("typo-embeddings-reference.json")?,
    ))
}

fn reference() -> Option<Vec<(String, [f32; OUT_DIM])>> {
    let value: Value = serde_json::from_slice(&artifacts()?.2).expect("reference json");
    Some(
        value
            .get("embeddings")
            .and_then(Value::as_object)
            .expect("reference carries an embeddings object")
            .iter()
            .map(|(word, arr)| {
                let mut out = [0f32; OUT_DIM];
                for (dst, v) in out.iter_mut().zip(arr.as_array().expect("row")) {
                    *dst = v.as_f64().expect("f64") as f32;
                }
                (word.clone(), out)
            })
            .collect(),
    )
}

#[test]
fn rust_encoder_matches_onnxruntime_on_the_frozen_artifact() {
    let Some((onnx, char_vocab, _)) = artifacts() else {
        eprintln!("typo fixtures absent - sync via KOTOSHU_MODELS_DIR (skipped)");
        return;
    };
    let model = TypoModel::parse(&onnx, &char_vocab).expect("parse frozen artifact");
    assert_eq!(model.char_table_rows(), 1823);

    let mut worst_cos = 1.0f32;
    let mut worst_abs = 0f32;
    let Some(reference) = reference() else {
        eprintln!("typo fixtures absent (skipped)");
        return;
    };
    for (word, expected) in reference {
        let got = model.embed(&word).expect("embed");
        let mut dot = 0f32;
        let mut na = 0f32;
        let mut nn = 0f32;
        let mut ng = 0f32;
        for (a, b) in got.iter().zip(expected) {
            dot += a * b;
            na = na.max((a - b).abs());
            nn += b * b;
            ng += a * a;
        }
        let cos = dot / (nn.sqrt() * ng.sqrt());
        worst_cos = worst_cos.min(cos);
        worst_abs = worst_abs.max(na);
    }
    // Plan 115 priced ranking equivalence at far looser tolerance than
    // this; both bounds leave an order of magnitude of slack.
    assert!(worst_cos > 0.9999, "worst cosine {worst_cos}");
    assert!(worst_abs < 5e-3, "worst element distance {worst_abs}");
}

#[test]
fn empty_word_is_an_error_and_long_words_truncate() {
    let Some((onnx, char_vocab, _)) = artifacts() else {
        eprintln!("typo fixtures absent (skipped)");
        return;
    };
    let model = TypoModel::parse(&onnx, &char_vocab).expect("parse");
    assert!(model.embed("").is_err());

    let long = "a".repeat(MAX_WORD_LEN + 8);
    let truncated = "a".repeat(MAX_WORD_LEN);
    let a = model.embed(&long).expect("embed long");
    let b = model.embed(&truncated).expect("embed truncated");
    // identical inputs after truncation → identical outputs
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < 1e-7);
    }
}

#[test]
fn index_retrieval_matches_brute_force_and_excludes_self() {
    let Some((onnx, char_vocab, _)) = artifacts() else {
        eprintln!("typo fixtures absent (skipped)");
        return;
    };
    let model = TypoModel::parse(&onnx, &char_vocab).expect("parse");
    let vocab: Vec<&str> = vec![
        "hello",
        "helo",
        "hell",
        "help",
        "shell",
        "halo",
        "yellow",
        "cello",
        "definately",
        "definitely",
        "define",
        "finite",
        "infinite",
    ];
    let index = TypoIndex::build(&model, &vocab);
    assert_eq!(index.len(), vocab.len());

    let query = model.embed("helo").expect("embed");
    let typo_row = index.vocab().iter().position(|w| w == "helo").expect("row");
    let top = index.top_k(&query, 3, Some(typo_row));
    assert_eq!(top.len(), 3);
    assert!(!top.iter().any(|s| s.index == typo_row), "self excluded");
    assert!(
        top[0].score >= top[1].score && top[1].score >= top[2].score,
        "sorted"
    );

    // brute force over dequantized rows, same rule
    let mut expected: Vec<(usize, f32)> = (0..vocab.len())
        .filter(|i| *i != typo_row)
        .map(|i| {
            let q = model.embed(vocab[i]).expect("embed");
            let dot = q.iter().zip(query).map(|(a, b)| a * b).sum::<f32>();
            (i, dot)
        })
        .collect();
    expected.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
    for (rank, (exp, got)) in expected.iter().zip(&top).enumerate() {
        assert!(
            (exp.1 - got.score).abs() < 2e-3,
            "rank {rank}: expected {} got {}",
            exp.1,
            got.score
        );
        assert_eq!(exp.0, got.index, "rank {rank} index");
    }
}
