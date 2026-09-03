# 03 — P2 suggest

Phase P2 of [66-kotoshu-core.md] (gem repo): suggestion strategies and
composite ranking.

## Goal

`Request::Suggest` returns ranked suggestions matching the gem's
`Kotoshu.suggest` output: banded edit distance with threshold (port the
gem's optimized DP), phonetic, keyboard proximity, n-gram, composite
ranking, frequency bonus.

## Tasks

- `suggest/` module: banded edit distance, phonet, keyboard layouts
  (qwerty/qwertz/azerty/jcuken/dvorak), n-gram, composite ranking with
  frequency bonus (Kelly tiers).
- Replace `shared::stub_response` for `Suggest` with ranked results
  (word/distance/confidence/source per `ffi::shared::Suggestion`).
- Activate conformance assertions for `kind: "suggest"` vectors (ranked
  lists from the gem's suggestion specs).
- Optional dep: `parallel` → `rayon` for batch checking.

## Acceptance

- Suggestion conformance vectors (ranked lists) pass on the C ABI —
  identical output to the gem backend.
- `rake compat:compare` in the gem shows zero diffs Ruby vs native.

## Status

**Implemented** (2026-09-03, branch `feat/p2-suggest`). The gem's entire
default suggestion path is ported in `kotoshu/src/suggest/` (edit distance
with enhanced scoring, phonetic/Soundex, keyboard proximity, n-gram,
composite ranking) with `Dictionary::suggest(word, limit)`; the KOSH-v1
batch path runs end-to-end on the C ABI (`kotoshu_dict_load`/`_free`
lifecycle + `kotoshu_batch` routing Check/Suggest by language via
`ffi/registry.rs`). All 1315 `suggest` vectors assert green — ordered
equality with exact f64 confidences — through the engine AND the C ABI
wire (second enforcement point in `tests/conformance.rs`); `correct`
stays 1315/1315. Differential fuzz against the live gem (5,948 suggests
over all 125 fixture dictionaries, limits 1-10, mutated/unicode/junk
inputs): zero mismatches after one fix (the gem's transposition check
cross-indexes `s[i] == o[match_idx]`).

Notes, honest and load-bearing:

- **Sort tie orders are frozen behavior.** The gem ranks through
  `sort_by` (MRI 3.4's uniform introsort — ported byte-exactly in
  `ruby_sort.rs`) and `Array#sort!` (on macOS, Apple libc `qsort_r` —
  ported in `macos_qsort.rs` because 16 vectors contain full-tie pairs
  whose surviving `source` label depends on that tie order). The vectors
  freeze the macOS/arm64 export platform; a Ruby process on Linux would
  order those 16 ties differently (and so, before this port, would any
  stable-sort implementation).
- **Frequency data is embedded, frozen.** The export ran with the Kelly
  `en` tiers from the exporter's `FrequencyCache`
  (`~/.cache/kotoshu/frequency-lists/en/frequency.json`, sha256
  97535823f4…); the cumulative top-50/200/1000 word sets are committed as
  `suggest/frequency_data.rs` (33 fixture words hit the tiers —
  "hello"/"help" among them — so the bonus demonstrably shapes the
  vectors). The capitalized Kelly entries ("London") can never match the
  gem's downcased lookup — quirk preserved.
- **Wire confidence is f64.** The P0 draft sketched f32; v1 never
  shipped outside this repo, so the layout was corrected to f64 before
  any binding consumed it (Ruby `Float` round-trips exactly). The
  conformance runner needs serde_json's `float_roundtrip` dev feature —
  its default float parse is 1-ulp lossy and false-fails exact compares.
- **`parallel` → rayon and `logging` → log remain deferred** (owner call
  on when the core takes its first third-party dependency; the features
  stay empty). Metaphone (`algorithm: :metaphone`) is not on the gem's
  default path and was not ported; the gem's simplified metaphone exists
  in `PhoneticStrategy` if ever needed.
