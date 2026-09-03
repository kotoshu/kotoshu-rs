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

_Planning._
