//! Plan 102 LID parity: the Rust engine vs the gem's frozen detection
//! outputs on the fixed multilingual corpus.
//!
//! The frozen expectations live in `parity.json` next to the fixture
//! artifact pair (kotoshu/tests/fixtures/lid/). They were produced by
//! `scripts/make_lid_fixture.py`, which drives
//! `Kotoshu::Language::LanguageIdentifier` (the gem shells out to the
//! python `fasttext` bindings over upstream `lid.176.ftz`). Every
//! sample asserts code equality and a score within the models-repo
//! int8 gate drift (4.8e-4 + headroom = 1e-3).
//!
//! Ignored by default: this is the long-form parity run (the unit
//! tests in `kotoshu/src/lid/mod.rs` exercise the engine on a
//! representative subset). The CI "lid" job re-enables it with
//! `--features model -- --include-ignored`. Locally:

use kotoshu::lid::LidModel;

const PARITY: &str = include_str!("../kotoshu/tests/fixtures/lid/parity.json");

#[test]
#[ignore = "full corpus parity run; the unit tests in kotoshu/src/lid cover representative samples"]
fn lid_detect_matches_the_frozen_gem_outputs_on_every_sample() {
    let fixture: serde_json::Value =
        serde_json::from_str(PARITY).expect("parity.json must be valid JSON");
    let tolerance = fixture["tolerance"].as_f64().expect("tolerance") as f32;
    let samples = fixture["samples"].as_array().expect("samples array");
    let model_bytes = include_bytes!("../kotoshu/tests/fixtures/lid/lid.176.onnx");
    let vocab_bytes = include_bytes!("../kotoshu/tests/fixtures/lid/lid.176.vocab.json");
    let model = LidModel::parse(model_bytes, vocab_bytes).expect("fixture parses");

    let mut mismatches = Vec::new();
    for sample in samples {
        let id = sample["id"].as_str().expect("sample id");
        let text = sample["text"].as_str().expect("sample text");
        let frozen_code = sample["code"].as_str().expect("frozen code");
        let frozen_score = sample["score"].as_f64().expect("frozen score") as f32;
        let detection = model.detect(text);
        if detection.code != frozen_code {
            mismatches.push(format!(
                "{id}: code {detection:?} != frozen {frozen_code:?}"
            ));
        }
        if (detection.score - frozen_score).abs() > tolerance {
            mismatches.push(format!(
                "{id}: score {} drifts from frozen {} by {} (> {tolerance})",
                detection.score,
                frozen_score,
                (detection.score - frozen_score).abs()
            ));
        }
    }

    assert!(
        mismatches.is_empty(),
        "{} parity mismatches:\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
}
