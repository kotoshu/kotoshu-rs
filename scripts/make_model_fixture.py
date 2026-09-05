#!/usr/bin/env python3
"""Build the wasm model-rerank fixture (plan 85).

Derives a tiny int8-per-row tier from the REAL registry v1.0.1 `en/mini`
artifact: a 40-word vocabulary slice, its rows reindexed 0..N-1, written
as the same graph shape `scripts/build_tiers.py` produces in the models
repo (Constant nodes, raw_data payloads, identical metadata keys). The
checked-in fixture then exercises the pure-Rust ONNX reader against
bytes onnx actually writes — a synthetic byte fixture could not.

Provenance and license live beside the output
(LICENSE-NOTE.md): the source tier is CC-BY-SA-3.0 (FastText crawl
vectors), so the derived fixture carries the same license.

Usage:
  scripts/make_model_fixture.py \\
      --source-onnx  /path/to/fasttext.en.mini.onnx \\
      --source-vocab /path/to/fasttext.en.mini.vocab.json \\
      --out-dir kotoshu/tests/fixtures/models

The source artifacts are the registry v1.0.1 release assets
(https://github.com/kotoshu/models-fasttext-onnx/releases/tag/v1.0.1);
their sha256 values are pinned below and the script refuses to run on
anything else.
"""

from __future__ import annotations

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path

import numpy as np
import onnx
from onnx import StringStringEntryProto, TensorProto, helper, numpy_helper

# registry.json @ v1.0.1, kotoshu://models/en/mini
SOURCE_ONNX_SHA256 = "d81f36c5e0097414db95d48406ce615161dd07c697996fe973297186279d5e2f"
SOURCE_VOCAB_SHA256 = "d9173eee9958f5f29a60776c1ec8d76e752ab5fc64d939fb4b22dbf6522bf0c9"

# The fixture vocabulary: common English words spanning clearly
# separated semantic clusters (animals, royalty, technology, weather,
# emotions, ...), all verified in the en/mini 10k vocabulary.
WORDS = [
    "the", "and", "he", "they", "she",
    "home", "house", "water", "food", "book",
    "read", "cat", "dog", "puppy", "mouse",
    "king", "queen", "man", "woman", "cheese",
    "computer", "keyboard", "garden", "flower", "run",
    "walk", "eat", "car", "road", "fire",
    "hot", "cold", "big", "small", "love",
    "hate", "happy", "sad", "summer", "winter",
]


def sha256_of(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def constant_array(model: onnx.ModelProto, name: str) -> np.ndarray:
    # Mirrors build_tiers.constant_array (initializers or Constant nodes).
    for init in model.graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value" and (attr.t.name == name or name in node.output):
                return numpy_helper.to_array(attr.t)
    raise KeyError(f"array {name!r} not found in model")


def make_fixture_model(q: np.ndarray, scale: np.ndarray) -> onnx.ModelProto:
    # The build_tiers.make_tier_model graph, verbatim (minus the ops no
    # host-side reader executes, which the original keeps anyway).
    vocab_size, dims = q.shape
    input_tensor = helper.make_tensor_value_info("word_index", TensorProto.INT64, [1])
    output_tensor = helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [dims])
    nodes = [
        helper.make_node(
            "Constant", [], ["q_embeddings"],
            value=numpy_helper.from_array(q, name="q_embeddings"),
        ),
        helper.make_node(
            "Constant", [], ["row_scale"],
            value=numpy_helper.from_array(scale, name="row_scale"),
        ),
        helper.make_node(
            "Constant", [], ["scale_shape"],
            value=numpy_helper.from_array(np.array([1, 1], dtype=np.int64), name="scale_shape"),
        ),
        helper.make_node("Gather", ["q_embeddings", "word_index"], ["emb_i8"], axis=0),
        helper.make_node("Gather", ["row_scale", "word_index"], ["row_scale_i"], axis=0),
        helper.make_node("Reshape", ["row_scale_i", "scale_shape"], ["s_1x1"]),
        helper.make_node("Cast", ["emb_i8"], ["emb_f"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["emb_f", "s_1x1"], ["embedding_flat"]),
        helper.make_node("Squeeze", ["embedding_flat"], ["embedding"], axes=[0]),
    ]
    graph = helper.make_graph(nodes, "fasttext_mini_embedding", [input_tensor], [output_tensor])
    model = helper.make_model(
        graph,
        producer_name="kotoshu-rs-fixture",
        producer_version="1",
        opset_imports=[helper.make_operatorsetid("", 11)],
        ir_version=11,
    )
    model.metadata_props.append(StringStringEntryProto(key="vocabulary_size", value=str(vocab_size)))
    model.metadata_props.append(StringStringEntryProto(key="embedding_dimension", value=str(dims)))
    model.metadata_props.append(StringStringEntryProto(key="model_type", value="fasttext_embedding"))
    model.metadata_props.append(StringStringEntryProto(key="quantization", value="int8-per-row"))
    model.metadata_props.append(StringStringEntryProto(key="tier", value="mini"))
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-onnx", type=Path, required=True)
    parser.add_argument("--source-vocab", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    for path, expected in (
        (args.source_onnx, SOURCE_ONNX_SHA256),
        (args.source_vocab, SOURCE_VOCAB_SHA256),
    ):
        actual = sha256_of(path)
        if actual != expected:
            print(f"{path}: sha256 {actual} != pinned {expected}", file=sys.stderr)
            return 1

    vocab = json.loads(args.source_vocab.read_text(encoding="utf-8"))["word_to_idx"]
    missing = [word for word in WORDS if word not in vocab]
    if missing:
        print(f"source vocabulary lacks: {missing}", file=sys.stderr)
        return 1

    source = onnx.load(str(args.source_onnx))
    q_all = constant_array(source, "q_embeddings").astype(np.int8)
    scale_all = constant_array(source, "row_scale").astype(np.float32)

    # Original row order of the chosen words — a true truncation.
    ordered = sorted(WORDS, key=lambda word: vocab[word])
    rows = np.stack([q_all[vocab[word]] for word in ordered])
    scales = np.array([scale_all[vocab[word]] for word in ordered], dtype=np.float32)
    word_to_idx = {word: index for index, word in enumerate(ordered)}

    onnx_path = args.out_dir / "en-mini-truncated.onnx"
    vocab_path = args.out_dir / "en-mini-truncated.vocab.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    onnx.save(make_fixture_model(rows, scales), str(onnx_path))
    vocab_path.write_text(
        json.dumps({"vocab_size": len(word_to_idx), "word_to_idx": word_to_idx},
                   ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    onnx.checker.check_model(onnx.load(str(onnx_path)))

    # Dequantized rows for reporting the cosines the tests can freeze.
    dequant = rows.astype(np.float32) * scales[:, None]
    unit = dequant / np.linalg.norm(dequant, axis=1, keepdims=True)

    def cos(a: str, b: str) -> float:
        return float(unit[word_to_idx[a]] @ unit[word_to_idx[b]])

    def mean_cos(word: str, context_words: list[str]) -> float:
        return sum(cos(word, w) for w in context_words) / len(context_words)

    print(f"wrote {onnx_path} ({onnx_path.stat().st_size} bytes) and {vocab_path}")
    print(f"vocab_size: {len(word_to_idx)}, dims: {dequant.shape[1]}")
    print("reference cosines for assertions:")
    print(f"  cos(cat, dog)      = {cos('cat', 'dog'):.6f}")
    print(f"  cos(cat, computer) = {cos('cat', 'computer'):.6f}")
    print(f"  mean cos(puppy; the dog and the cat)          = "
          f"{mean_cos('puppy', ['the', 'dog', 'and', 'the', 'cat']):.6f}")
    print(f"  mean cos(puppy; the computer and the keyboard) = "
          f"{mean_cos('puppy', ['the', 'computer', 'and', 'the', 'keyboard']):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
