#!/usr/bin/env python3
"""Build the wasm model-rerank fixtures (plan 85 + plan 103 buckets).

Derives a tiny int8-per-row tier from the REAL registry v1.0.1 `en/mini`
artifact: a 40-word vocabulary slice, its rows reindexed 0..N-1, written
as the same graph shape `scripts/build_tiers.py` produces in the models
repo (Constant nodes, raw_data payloads, identical metadata keys). The
checked-in fixture then exercises the pure-Rust ONNX reader against
bytes onnx actually writes — a synthetic byte fixture could not.

With --source-buckets (plan 103), also derives a bucket-table fixture
from the REAL `en` buckets artifact (`scripts/export_buckets.py` in the
models repo): the marked-5-gram rows the OOV queries teh/catt/hotcold
address, ids verbatim, written in the bucket artifact's graph shape —
so the reader is exercised against real fastText bucket-row bytes too.

Provenance and license live beside the output
(LICENSE-NOTE.md): the source tiers are CC-BY-SA-3.0 (FastText crawl
vectors), so the derived fixture carries the same license.

Usage:
  scripts/make_model_fixture.py \\
      --source-onnx  /path/to/fasttext.en.mini.onnx \\
      --source-vocab /path/to/fasttext.en.mini.vocab.json \\
      [--source-buckets /path/to/fasttext.en.buckets.onnx] \\
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

# The bucket queries the bucket fixture must support (the kotoshu-rs
# specs freeze their neighbors); their marked 5-grams select the rows.
BUCKET_QUERY_WORDS = ["teh", "catt", "hotcold"]

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

FNV_BASIS = 2166136261
FNV_PRIME = 16777619


def fasttext_hash(ngram: str) -> int:
    """FNV-1a 32 with fastText's int8 sign-extension (dictionary.cc)."""
    h = FNV_BASIS
    for byte in ngram.encode("utf-8"):
        if byte >= 0x80:
            byte -= 256
        h = ((h ^ (byte & 0xFFFFFFFF)) * FNV_PRIME) & 0xFFFFFFFF
    return h


def marked_ngrams(word: str, minn: int, maxn: int) -> list[str]:
    marked = f"<{word}>"
    chars = list(marked)
    out: dict[str, None] = {}
    for n in range(minn, maxn + 1):
        if n > len(chars):
            break
        for start in range(0, len(chars) - n + 1):
            out["".join(chars[start : start + n])] = None
    return list(out)


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


def make_buckets_fixture_model(q: np.ndarray, scale: np.ndarray,
                              bucket_ids: np.ndarray, bucket_count: int,
                              dims: int, rows: int, minn: int, maxn: int) -> onnx.ModelProto:
    """The bucket-table graph shape (plan 103): the tier constant-node
    shape plus a `bucket_ids` int64 tensor (the original hash-bucket
    index of each kept row, ascending) and `bucket_count`/`minn`/`maxn`
    metadata that the reader requires."""
    input_tensor = helper.make_tensor_value_info("bucket_row", TensorProto.INT64, [1])
    output_tensor = helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [dims])
    nodes = [
        helper.make_node("Constant", [], ["q_embeddings"],
                         value=numpy_helper.from_array(q, name="q_embeddings")),
        helper.make_node("Constant", [], ["row_scale"],
                         value=numpy_helper.from_array(scale, name="row_scale")),
        helper.make_node("Constant", [], ["bucket_ids"],
                         value=numpy_helper.from_array(bucket_ids, name="bucket_ids")),
        helper.make_node("Constant", [], ["scale_shape"],
                         value=numpy_helper.from_array(np.array([1, 1], dtype=np.int64),
                                                        name="scale_shape")),
        helper.make_node("Gather", ["q_embeddings", "bucket_row"], ["emb_i8"], axis=0),
        helper.make_node("Gather", ["row_scale", "bucket_row"], ["row_scale_i"], axis=0),
        helper.make_node("Reshape", ["row_scale_i", "scale_shape"], ["s_1x1"]),
        helper.make_node("Cast", ["emb_i8"], ["emb_f"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["emb_f", "s_1x1"], ["embedding_flat"]),
        helper.make_node("Squeeze", ["embedding_flat"], ["embedding"], axes=[0]),
    ]
    graph = helper.make_graph(nodes, "fasttext_buckets_embedding",
                              [input_tensor], [output_tensor])
    model = helper.make_model(
        graph,
        producer_name="kotoshu-rs-fixture",
        producer_version="1",
        opset_imports=[helper.make_operatorsetid("", 11)],
        ir_version=11,
    )
    for key, value in (
        ("model_type", "fasttext_buckets"),
        ("quantization", "int8-per-row"),
        ("embedding_dimension", str(dims)),
        ("bucket_count", str(bucket_count)),
        ("buckets", str(rows)),
        ("minn", str(minn)),
        ("maxn", str(maxn)),
        ("tier", "buckets"),
    ):
        model.metadata_props.append(StringStringEntryProto(key=key, value=value))
    return model


def build_buckets_fixture(source_buckets: Path, out_dir: Path, tier_vocab: dict) -> Path:
    """Select the bucket rows the OOV queries `teh`/`catt`/`hotcold`
    address (plus the rows those queries' unmarked n-grams could
    reach), reindex them to ascending ids, and write the bucket fixture
    in the artifact's own shape. Prints the cosines the bucket-backed
    neighbors spec freezes (they are the fixture's own numbers, computed
    against the fixture's 40-word tier vocab)."""
    source = onnx.load(str(source_buckets))
    src_q = constant_array(source, "q_embeddings").astype(np.int8)
    src_scale = constant_array(source, "row_scale").astype(np.float32)
    src_ids = constant_array(source, "bucket_ids").astype(np.int64)
    bucket_count = int(source.metadata_props.get(("bucket_count",) if False else "bucket_count")
                      if False else 0)
    # onnx ModelProto's metadata_props is a list of (key, value) pairs;
    # read each key by iterating.
    metadata = {prop.key: prop.value for prop in source.metadata_props}
    bucket_count = int(metadata["bucket_count"])
    minn = int(metadata["minn"])
    maxn = int(metadata["maxn"])

    bucket_to_row: dict[int, int] = {int(b): i for i, b in enumerate(src_ids)}
    bucket_to_vector: dict[int, np.ndarray] = {}
    for i, b in enumerate(src_ids):
        bucket_to_vector[int(b)] = src_q[i].astype(np.float32) * src_scale[i]

    wanted_buckets: set[int] = set()
    for word in BUCKET_QUERY_WORDS:
        for ngram in marked_ngrams(word, minn, maxn):
            bucket = fasttext_hash(ngram) % bucket_count
            if bucket in bucket_to_vector:
                wanted_buckets.add(bucket)
        # Also keep the bucket rows the unmarked-n-gram fallback could
        # resolve through (substring_ngrams of the query word): each
        # unmarked n-gram's row, if it sits in the source table. This
        # keeps the fixture consistent with the reader's union contract
        # for queries whose n-grams are also real fastText buckets.
        chars = list(word)
        for n in range(3, 7):
            if n > len(chars):
                break
            for start in range(0, len(chars) - n + 1):
                ngram = "".join(chars[start:start + n])
                bucket = fasttext_hash(ngram) % bucket_count
                if bucket in bucket_to_vector:
                    wanted_buckets.add(bucket)

    sorted_buckets = sorted(wanted_buckets)
    rows = np.stack([bucket_to_vector[b] / src_scale[bucket_to_row[b]]
                     for b in sorted_buckets]).astype(np.int8)
    scales = np.array([src_scale[bucket_to_row[b]] for b in sorted_buckets],
                      dtype=np.float32)
    bucket_ids_out = np.array(sorted_buckets, dtype=np.int64)
    rows_q, dims = rows.shape

    fixture_path = out_dir / "en-buckets-truncated.onnx"
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx.save(make_buckets_fixture_model(rows, scales, bucket_ids_out,
                                          bucket_count, dims, rows_q,
                                          minn, maxn),
              str(fixture_path))
    onnx.checker.check_model(onnx.load(str(fixture_path)))

    # Reference cosines for the bucket-backed neighbors the rs spec
    # freezes. Uses the fixture's own rows + the same tier fixture
    # vocab so the numbers are reproducible from the checked-in bytes.
    dequant = rows.astype(np.float32) * scales[:, None]
    unit = dequant / np.linalg.norm(dequant, axis=1, keepdims=True)

    # The tier fixture vocab lives in en-mini-truncated.vocab.json
    # (already written by the tier path above); load it for the
    # dequantized tier rows.
    tier_onnx_path = out_dir / "en-mini-truncated.onnx"
    tier_model = onnx.load(str(tier_onnx_path))
    tier_q = constant_array(tier_model, "q_embeddings").astype(np.int8)
    tier_scale = constant_array(tier_model, "row_scale").astype(np.float32)
    tier_rows = tier_q.astype(np.float32) * tier_scale[:, None]
    tier_unit = tier_rows / np.linalg.norm(tier_rows, axis=1, keepdims=True)

    def neighbor(query: str, k: int = 4) -> list[tuple[str, float]]:
        """The bucket-backed embedding (union contract) for `query`,
        then the top-k tier vocab words by cosine."""
        # 1. Unmarked n-gram hits in tier vocab (fixture-local row
        # indices — the 40-word tier fixture was reindexed 0..39, NOT
        # the source mini vocab's 0..9999 ordering).
        chars = list(query)
        sum_vec = np.zeros(300, dtype=np.float32)
        fixture_vocab = json.loads((out_dir / "en-mini-truncated.vocab.json")
                                   .read_text(encoding="utf-8"))["word_to_idx"]
        vocab_words = sorted(fixture_vocab, key=lambda w: fixture_vocab[w])
        for n in range(3, 7):
            if n > len(chars):
                break
            for start in range(0, len(chars) - n + 1):
                ng = "".join(chars[start:start + n])
                if ng in fixture_vocab:
                    sum_vec += tier_rows[fixture_vocab[ng]]
        # 2. Marked 5-grams (minn=maxn for crawl) via bucket rows.
        for ngram in marked_ngrams(query, minn, maxn):
            bucket = fasttext_hash(ngram) % bucket_count
            if bucket in bucket_to_vector:
                sum_vec += bucket_to_vector[bucket]
        norm = float(np.linalg.norm(sum_vec))
        if norm == 0.0:
            return []
        query_vec = sum_vec / norm
        sims = tier_unit @ query_vec
        # Deterministic: (-score, word) ascending.
        order = sorted(((i, float(sims[i])) for i in range(len(sims))),
                       key=lambda t: (-t[1], vocab_words[t[0]]))
        return [(vocab_words[i], round(s, 4)) for i, s in order[:k]]

    print(f"wrote {fixture_path} ({fixture_path.stat().st_size} bytes, "
          f"{rows_q} rows, bucket_count {bucket_count})")
    print("reference neighbors (for the kotoshu-rs bucket spec):")
    for word in BUCKET_QUERY_WORDS:
        print(f"  semantic_neighbors({word!r}, 4) -> {neighbor(word, 4)}")
    return fixture_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-onnx", type=Path, required=True)
    parser.add_argument("--source-vocab", type=Path, required=True)
    parser.add_argument("--source-buckets", type=Path, default=None,
                        help="Real en/buckets artifact (plan 103); when "
                             "supplied, also emits en-buckets-truncated.onnx")
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

    if args.source_buckets is not None:
        build_buckets_fixture(args.source_buckets, args.out_dir, vocab)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
