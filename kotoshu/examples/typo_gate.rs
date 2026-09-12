//! Plan 131's productization gate (run by hand / CI eval, not a test):
//! replay the frozen C-benchmark real component through the SHIPPED
//! engine path — TypoModel + TypoIndex + the fastText full tier — and
//! report the hybrid's hit rates plus timings. The frozen reference
//! (models repo, eval/reports/hybrid-pricing.json, int8 variant, en
//! real n=2509): top1 0.118772, top5 0.269829, top20 0.405341; the
//! gate bar is agreement within noise (≤ 0.5 pp absolute per rate).
//!
//! Usage:
//! ```text
//! cargo run -p kotoshu --features model --release --example typo_gate -- \
//!     fasttext.en.onnx fasttext.en.vocab.json \
//!     typo.biencoder.onnx typo.biencoder.vocab.json cbench.en.json
//! ```

use std::time::Instant;

use kotoshu::rerank::int8_model::Int8Model;
use kotoshu::typo::{TypoEngine, TypoModel};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 6 {
        eprintln!(
            "usage: typo_gate <tier.onnx> <tier.vocab.json> <typo.onnx> <typo.vocab.json> <cbench.json>"
        );
        std::process::exit(2);
    }
    let read = |i: usize| std::fs::read(&args[i]).expect("readable input");
    let tier = Int8Model::parse(&read(1), &read(2)).expect("parse tier");
    let typo_model = TypoModel::parse(&read(3), &read(4)).expect("parse typo bi-encoder");
    let engine = TypoEngine::new(typo_model);

    let bench: serde_json::Value = serde_json::from_slice(&read(5)).expect("parse cbench");
    let pairs: Vec<(String, String)> = bench["real"]
        .as_array()
        .expect("real pairs array")
        .iter()
        .map(|pair| {
            (
                pair[0].as_str().expect("typo").to_owned(),
                pair[1].as_str().expect("correction").to_owned(),
            )
        })
        .collect();

    // the derived index: built once, timed separately from per-suggest
    let t0 = Instant::now();
    let index = engine.derived_index(&tier);
    let build_ms = t0.elapsed().as_secs_f64() * 1e3;

    let mut n = 0usize;
    let mut hits = [0usize; 3];
    let mut oov = 0usize;
    let t1 = Instant::now();
    for (typo, correction) in &pairs {
        let Some(slate) = engine.suggest(&tier, typo, 20) else {
            oov += 1;
            continue;
        };
        n += 1;
        let corr_score = slate
            .iter()
            .find(|(word, _)| word == correction)
            .map(|(_, score)| *score);
        let Some(corr_score) = corr_score else {
            continue; // correction outside the slate: miss for every k
        };
        let rank0 = slate
            .iter()
            .filter(|(_, score)| *score > corr_score)
            .count();
        // hit@1/@5/@20, ties optimistic (strictly-greater count)
        if rank0 < 1 {
            hits[0] += 1;
        }
        if rank0 < 5 {
            hits[1] += 1;
        }
        if rank0 < 20 {
            hits[2] += 1;
        }
    }
    let suggest_ms = t1.elapsed().as_secs_f64() * 1e3;

    println!(
        "{}",
        serde_json::json!({
            "pairs_total": pairs.len(),
            "pairs_evaluated": n,
            "typo_ft_oov": oov,
            "top1": hits[0] as f64 / n as f64,
            "top5": hits[1] as f64 / n as f64,
            "top20": hits[2] as f64 / n as f64,
            "index_vocab": index.len(),
            "index_build_ms": build_ms,
            "index_build_ms_amortized_per_suggest": build_ms / n as f64,
            "mean_suggest_ms": suggest_ms / n as f64,
        })
    );
}
