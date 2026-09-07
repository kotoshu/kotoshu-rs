# LID parity fixture

The committed artifact pair (lid.176.onnx, lid.176.vocab.json) is the
plan-102 registry resource kotoshu://models/lid/lid-176, mirrored to the
LFS media host (ACAO:*) per the tier-mirror pattern. It is produced by
`scripts/build_lid.py` in the models repo (kotoshu/models-fasttext-onnx)
from upstream `lid.176.ftz` and SHA-256-pinned in
`scripts/upstream_versions.json` (entry `lid`, sha256
`8f3472cfe8738a7b6099e8e999c3cbfae0dcd15696aac7d7738a8039db603e83`).

Source: https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz

License: MIT (fastText, https://github.com/facebookresearch/fastText/blob/main/LICENSE).

The committed binary is the EXACT mirror of the artifact the models
release ships; nothing is added, nothing is derived, the int8
re-quantization was applied upstream by `scripts/build_lid.py` and
gated against the float32 reconstruction (top-1 agreement 1.0000 on
the probe corpus, max score drift 4.8e-4).

parity.json holds the gem detection outputs for the fixed multilingual
corpus (`scripts/make_lid_fixture.py`). The Rust integration test
(`tests/lid_parity.rs`) loads this fixture and asserts the pure-Rust
reader reproduces every code exactly and every score within 1e-3
(the int8 gate drift, with headroom).
