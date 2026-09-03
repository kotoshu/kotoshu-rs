//! Conformance-vector runner.
//!
//! Reads `tests/conformance/*.jsonl` when present. Each line is one vector
//! in the gem's documented API shape:
//!
//! ```json
//! {"kind":"correct","language":"en","input":"hello","expected":true}
//! {"kind":"suggest","language":"en","input":"recieve",
//!  "expected":[{"word":"receive","distance":1,"confidence":0.95,"source":"edit_distance"}]}
//! ```
//!
//! The pack in this repository is a PLACEHOLDER (three hand-written lines
//! covering the `Kotoshu.correct?` / `Kotoshu.suggest` shapes). It will be
//! replaced by the gem's `rake kotoshu:conformance:export` golden vectors
//! (plan 67 M3) — one source of truth, three enforcement points (C ABI,
//! `ruby` feature, wasm32).
//!
//! P0 policy: with no engine yet, this test only validates pack shape and
//! counts vectors; real assertions activate with P1 (`correct?`) and P2
//! (`suggest`). An absent or empty pack asserts nothing.

use std::fs;
use std::path::Path;

const CONFORMANCE_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../tests/conformance");

#[test]
fn conformance_vectors_are_well_formed() {
    let dir = Path::new(CONFORMANCE_DIR);
    if !dir.is_dir() {
        return;
    }

    let mut files = 0;
    let mut vectors = 0;

    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(error) => panic!("cannot read conformance dir {CONFORMANCE_DIR}: {error}"),
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_none_or(|ext| ext != "jsonl") {
            continue;
        }
        files += 1;
        for (line_no, line) in fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()))
            .lines()
            .enumerate()
        {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            // P0 shape check only; a JSON parser lands with the P1 engine deps.
            assert!(
                line.contains("\"kind\"") && line.contains("\"input\""),
                "{}:{}: vector lacks kind/input",
                path.display(),
                line_no + 1
            );
            vectors += 1;
        }
    }

    if files > 0 {
        assert!(vectors > 0, "conformance files present but empty: {dir:?}");
    }
    eprintln!("conformance: {files} files, {vectors} vectors");
}
