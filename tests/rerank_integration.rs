//! P3 rerank integration test: the real registry, the real en/mini
//! model, the real ort runtime.
//!
//! **Ignored by default** — it needs network access (GitHub release
//! download, ~3 MB) and a loadable `libonnxruntime`. Run it with:
//!
//! ```sh
//! # locate the dylib the python onnxruntime package ships:
//! python3 -c "import onnxruntime, glob, os; print(glob.glob(os.path.join(
//!     os.path.dirname(onnxruntime.__file__), 'capi', 'libonnxruntime.*'))[0])"
//! export KOTOSHU_ORT_DYLIB=/path/to/libonnxruntime.dylib
//! cargo test --features onnx,resources --test rerank_integration -- --ignored --nocapture
//! ```
//!
//! The test skips cleanly (with a clear message) when the dylib is
//! absent; the CI `onnx` job installs onnxruntime via pip, exports
//! `KOTOSHU_ORT_DYLIB`, and runs exactly the command above.
//!
//! ## Fixtures and golden values
//!
//! - `tests/registry.json` is a verbatim copy of the models repo's
//!   `registry.json` at release `v1.0.1` (spec `kotoshu.resources/v1`).
//!   Regenerate with:
//!   `cp ../models-fasttext-onnx/registry.json tests/registry.json`.
//! - The golden cosines below were computed ONCE against the real
//!   en/mini artifact (sha256 `d81f36c5…`, the value asserted from the
//!   registry) with python + onnxruntime 1.23.2, and are asserted with
//!   a 1e-4 tolerance. Regenerate with:
//!
//! ```python
//! import json, onnxruntime, numpy as np
//! v = json.load(open("fasttext.en.mini.vocab.json"))["word_to_idx"]
//! s = onnxruntime.InferenceSession("fasttext.en.mini.onnx")
//! e = lambda w: s.run(["embedding"], {"word_index": np.array([v[w]], np.int64)})[0]
//! cos = lambda a, b: float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
//! for a, b in [("hello","world"),("hello","computer"),("cat","dog"),("cat","computer")]:
//!     print((a, b), repr(cos(e(a), e(b))))
//! ```

use kotoshu::dict::Dictionary;
use kotoshu::rerank::dequant::RowFormat;
use kotoshu::rerank::onnx::OrtProvider;
use kotoshu::rerank::oov::SubwordFallback;
use kotoshu::rerank::{Context, CosineReranker, EmbeddingProvider, cosine};
use kotoshu::resource::{Registry, ResourceCache};

/// sha256 of `fasttext.en.mini.onnx` at registry release v1.0.1.
const EN_MINI_SHA256: &str = "d81f36c5e0097414db95d48406ce615161dd07c697996fe973297186279d5e2f";

/// Golden cosines from the real en/mini model (see the module docs for
/// the regeneration snippet): `(word_a, word_b, expected_cosine)`.
const GOLDEN_COSINES: [(&str, &str, f64); 4] = [
    ("hello", "world", 0.10845591872930527),
    ("hello", "computer", 0.11251780390739441),
    ("cat", "dog", 0.7074322700500488),
    ("cat", "computer", 0.18722765147686005),
];

/// Tolerance for the golden cosines (int8-per-row dequantization and
/// f32 arithmetic make them stable far below this).
const COSINE_TOLERANCE: f64 = 1e-4;

#[test]
#[ignore = "requires network + libonnxruntime; see the module docs for the exact invocation"]
fn real_model_registry_download_and_rerank() {
    // 0. Dylib gate — skip cleanly with a clear message when the
    // onnxruntime shared library cannot be loaded.
    if let Err(message) = kotoshu::rerank::onnx::dylib_available() {
        eprintln!("SKIP: onnxruntime dylib unavailable: {message}");
        return;
    }

    // 1. The real registry fixture: parse, resolve en/mini, cross-check
    //    the sha256 this test pins against the registry entry itself.
    let registry = Registry::parse(include_str!("registry.json"))
        .expect("committed registry fixture must parse");
    let resource = registry
        .resource("en", "mini")
        .expect("registry fixture must contain kotoshu://models/en/mini");
    assert_eq!(resource.sha256, EN_MINI_SHA256);
    assert_eq!(resource.tier.quantization.as_deref(), Some("int8-per-row"));
    assert_eq!(resource.tier.dims, 300);

    // 2. Download through the resource layer into a scratch cache
    //    (never the user's), sha-verified against the registry entry.
    let scratch = std::env::temp_dir().join(format!(
        "kotoshu-rs-rerank-integration-{}",
        std::process::id()
    ));
    let cache = ResourceCache::with_root(&scratch);
    let paths = cache
        .ensure_model(&registry, "en", "mini")
        .expect("en/mini must download + verify from the registry URLs");
    let size = std::fs::metadata(&paths.onnx).expect("downloaded model must exist");
    assert_eq!(size.len(), resource.size_bytes);

    // 3. Load the provider off the cached artifact pair.
    let provider = OrtProvider::load(&paths.onnx, &paths.vocab).expect("model must load");
    assert_eq!(provider.dims(), 300);
    assert_eq!(provider.format(), RowFormat::Int8PerRow);
    assert!(provider.vocab_len() >= 10_000);
    assert!(provider.embedding("hello").is_some());
    assert_eq!(provider.embedding("hello").map(|v| v.len()), Some(300));

    // 4. Golden cosines (regeneration: see the module docs).
    for (a, b, expected) in GOLDEN_COSINES {
        let (vec_a, vec_b) = (
            provider.embedding(a).unwrap(),
            provider.embedding(b).unwrap(),
        );
        let actual = cosine(&vec_a, &vec_b);
        assert!(
            (actual - expected).abs() < COSINE_TOLERANCE,
            "cos({a}, {b}) = {actual}, expected {expected}"
        );
    }

    // 5. The scope assertion: suggest("helo") reranked by the provider
    //    keeps "hello" at rank 1. The base fixture dictionary (synced
    //    from the gem repo) yields exactly [hello(1.0, edit_distance)]
    //    for "helo"; the context words around it are all in-vocab and
    //    cosine-positive, so the boost cannot displace it.
    let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../tests/fixtures/spec/integrational/fixtures/base");
    let (aff, dic) = (base.with_extension("aff"), base.with_extension("dic"));
    if !aff.is_file() || !dic.is_file() {
        eprintln!(
            "SKIP: fixture dictionaries not synced (run scripts/sync_conformance.sh); \
             rerank-over-suggest assertions skipped"
        );
        cleanup(&scratch);
        return;
    }
    let dictionary = Dictionary::load(&aff, &dic).expect("base fixture dictionary must load");
    let suggestions = dictionary.suggest("helo", 10);
    assert!(
        suggestions.iter().any(|s| s.word == "hello"),
        "engine must suggest hello for helo: {suggestions:?}"
    );

    let context = Context::new("said ", "helo", " to the world");
    let reranked = CosineReranker::new().rerank(&provider, &context, suggestions);
    assert_eq!(reranked[0].word, "hello");
    assert_eq!(reranked[0].confidence, 1.0); // capped boost over 1.0
    eprintln!(
        "rerank: {} -> {:?}",
        context.current,
        reranked
            .iter()
            .map(|s| (s.word.as_str(), s.confidence))
            .collect::<Vec<_>>()
    );

    // 6. B2 over the real vocabulary: "qqhelloqq" is OOV, but its
    //    in-vocab n-grams resolve through the substring fallback. In the
    //    en/mini top-10k those are exactly "hell" (4) and "hello" (5),
    //    so the fallback is the normalized sum of two vectors — unit
    //    norm, strongly aligned with "hello" (measured 0.7488 on this
    //    artifact), and far more aligned with it than with "computer".
    assert!(
        provider.embedding("qqhelloqq").is_none(),
        "qqhelloqq must be OOV"
    );
    let fallback = SubwordFallback::new(&provider);
    let oov = fallback
        .embedding("qqhelloqq")
        .expect("in-vocab substrings (hell, hello) must resolve through the fallback");
    assert_eq!(oov.len(), 300);
    let norm = oov.iter().map(|v| f64::from(v * v)).sum::<f64>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-4,
        "fallback vector must be L2-normalized, norm {norm}"
    );
    let to_hello = cosine(&oov, &provider.embedding("hello").unwrap());
    let to_computer = cosine(&oov, &provider.embedding("computer").unwrap());
    assert!(
        to_hello > 0.6,
        "fallback must align with the inner word, got {to_hello}"
    );
    assert!(
        to_hello - to_computer > 0.3,
        "fallback must prefer hello over computer: {to_hello} vs {to_computer}"
    );

    cleanup(&scratch);
}

/// Remove the scratch cache (best effort — CI temp dirs are wiped
/// anyway; nothing here is source).
fn cleanup(scratch: &std::path::Path) {
    let _ = std::fs::remove_dir_all(scratch);
}
