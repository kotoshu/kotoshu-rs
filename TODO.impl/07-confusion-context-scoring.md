# Plan 07: native real-word confusion scoring

## Status: proposed (blocked on models plan 15 calibration + gem plan 146 format freeze)

## Problem

The native check path mirrors the gem: in-vocab tokens are accepted
unquestioned, so real-word errors pass even though every ingredient
of context scoring already exists here (Context, CosineReranker,
int8 tier reader, typo matrix). The gem cannot stay the only engine
with the feature — wasm and server run the rs engine.

## What

- `suggest`-sibling module `confuse`: loads a confusion-pair table
  artifact (format frozen by gem plan 146), answers
  `confusions_for(word)`, and decides `is_real_word_error(word,
  context)` via the same margin math as `CosineReranker` (context
  neighbor vectors through the existing embedding provider trait).
- FFI/wasm: `check` gains the opt-in flag; the decision is a
  distinct error kind so clients can render real-word errors
  differently from misspellings.
- Thresholds ride the confusion artifact's metadata (frozen from the
  models calibration), not engine code — per-language numbers
  without engine rebuilds.
- Parity: ctest vectors shared with the gem specs (same fixture
  words, contexts, thresholds ⇒ same margins), plus a rejection test
  (clean in-vocab words below threshold are NOT flagged).

## Consumers

Gem plan 146 (Ruby parity), models plan 15 (artifact + thresholds).
CJK homophone confusion sources (pinyin/kana tables) plug in as
table generators here — no engine change.
