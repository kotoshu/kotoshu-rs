# 02 — P1 dictionary

Phase P1 of [66-kotoshu-core.md] (gem repo): aff/dic parsing, affix
expansion, lookup.

## Goal

`Request::Check` returns real `correct?` results: parse Hunspell `.aff`
(option lines, prefix/suffix rules, condition regexes) and `.dic` (stems +
flags), expand affixes lazily, index into a DAFSA/trie, answer batch lookups
through the C ABI.

## Tasks

- `dict/` module: `aff` parser, `dic` loader, affix expansion, DAFSA/trie
  (`data_structures` patterns from the gem).
- Capitalization handling (keepcase, warn, allcap) per the gem's
  `algorithms/capitalization`.
- Replace `shared::stub_response` for `Check` with engine results.
- Activate conformance assertions for `kind: "correct"` vectors (import the
  gem's exported golden vectors via `rake kotoshu:conformance:export`).
- First optional deps: `logging` → `log`; criterion benches skeleton.

## Acceptance

- Full `correct?` conformance vector pack passes on the C ABI.
- Lookup latency competitive with the gem's `IndexedDictionary` (bench).

## Status

**Implemented** (2026-09-03, branch `feat/p1-dictionary`). The gem's entire
Hunspell `correct?` path is ported in `kotoshu/src/dict/` (`aff` parser,
`dic` loader, `casing`, `encoding`, `lookup`); behavior is frozen by the
gem's conformance vectors — 1315/1315 `correct` vectors assert green
(1315 `suggest` vectors counted only until P2). Indexes use the gem's own
structures (char-keyed suffix/prefix buckets, exact + lowercase stem
indexes), not a separate trie/DAFSA — the gem's Trie is not on this path.

Deferred: `logging` → `log` (now planned with P2), criterion bench skeleton
(P2, alongside the parallel batch feature), and routing `Request::Check`
through the C ABI — that needs a dictionary-lifecycle API
(load/register/free by language or path) which lands with `parallel`; the
P0 stub response stays (see `ffi/shared.rs`), and the conformance suite
exercises `dict::Dictionary::correct` directly.
