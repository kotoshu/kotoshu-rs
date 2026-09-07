# License note: model fixtures

`en-mini-truncated.onnx` and `en-mini-truncated.vocab.json` are derived
from the Kotoshu `en/mini` embedding tier, registry v1.0.1 release
assets of https://github.com/kotoshu/models-fasttext-onnx:

- source `.onnx` sha256
  `d81f36c5e0097414db95d48406ce615161dd07c697996fe973297186279d5e2f`
- source `.vocab.json` sha256
  `d9173eee9958f5f29a60776c1ec8d76e752ab5fc64d939fb4b22dbf6522bf0c9`

`en-buckets-truncated.onnx` is derived from the Kotoshu `en/buckets`
bucket-table sibling artifact (plan 103), from the same registry:

- source `.onnx` sha256
  `1bd7825b8eb92b47701e483fb4804f09ae2064b147fe34be63ebfc7272570ff7`

The derivations (`scripts/make_model_fixture.py`, which refuses any
source whose sha256 differs) keep a 40-word tier truncation verbatim
(int8 rows, row scales, reindexed 0..39) and a 4-row bucket truncation
(the marked 5-gram rows the OOV queries `teh`/`catt`/`hotcold`
address, bucket ids verbatim) — no vector values are altered or
invented.

The source tiers are **CC-BY-SA-3.0** (as declared in the models
registry and the models repo LICENSE): they derive from FastText
pretrained vectors trained on Common Crawl by Facebook AI Research
(https://fasttext.cc/docs/en/crawl-vectors.html). Under
Attribution-ShareAlike these derived fixtures carry the same license:
https://creativecommons.org/licenses/by-sa/3.0/

This note travels with the fixtures; do not redistribute them without it.
