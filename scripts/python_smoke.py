#!/usr/bin/env python3
"""Smoke test for the `python` feature (P4d): imports the kotoshu_native
extension (built and installed by scripts/python_smoke.sh) into a REAL
Python interpreter and drives the REAL engine over REAL fixture
dictionaries — no mocks.

Usage: scripts/python_smoke.sh (builds + installs the wheel into a venv
first). Expectations marked "conformance vector" are frozen by the gem's
exported vectors (tests/fixtures/vectors.jsonl), not hand-written — this
is the Python twin of tests/ruby_ffi_smoke.rb and
scripts/wasm_node_smoke.mjs.
"""

import os
import sys

FIXTURES_DIR = os.environ.get(
    "KOTOSHU_FIXTURES_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "fixtures"),
)

if not os.path.isdir(FIXTURES_DIR):
    sys.exit(
        f"python_smoke: fixtures not found at {FIXTURES_DIR} — "
        "run scripts/sync_conformance.sh first (KOTOSHU_GEM_DIR=... if needed)"
    )

import kotoshu_native

failures = 0
assertions = 0


def check(label, condition):
    global failures, assertions
    assertions += 1
    print(("PASS " if condition else "FAIL ") + label)
    if not condition:
        failures += 1


def check_equal(label, expected, actual):
    check(f"{label} (expected {expected!r}, got {actual!r})", expected == actual)


# --- Module surface -------------------------------------------------------

check("kotoshu_native.Dictionary is defined", hasattr(kotoshu_native, "Dictionary"))
check(
    "VERSION is a dotted triple str",
    isinstance(kotoshu_native.VERSION, str)
    and len(kotoshu_native.VERSION.split(".")) == 3
    and all(part.isdigit() for part in kotoshu_native.VERSION.split(".")),
)
check_equal("available() is True", True, kotoshu_native.available())
check(
    "KotoshuNativeError is an Exception subclass",
    issubclass(kotoshu_native.KotoshuNativeError, Exception),
)

# --- The gem's `en` test dictionary ---------------------------------------
# Every vector on this dictionary is frozen in vectors.jsonl; "helo" IS a
# word here, so its suggest list is empty (words the dictionary accepts get
# no suggestions — gem behavior).

test_dic = os.path.join(FIXTURES_DIR, "spec/fixtures/dictionaries/hunspell/test")
dictionary = kotoshu_native.Dictionary.load(f"{test_dic}.aff", f"{test_dic}.dic")
check("Dictionary instance", isinstance(dictionary, kotoshu_native.Dictionary))
check_equal("correct('hello') — conformance vector", True, dictionary.correct("hello"))
check_equal("correct('ruby') — conformance vector", False, dictionary.correct("ruby"))
check_equal("correct('helo') is a listed word", True, dictionary.correct("helo"))
check_equal(
    "suggest('helo', 5) — conformance vector (accepted words suggest nothing)",
    [],
    dictionary.suggest("helo", 5),
)
check_equal("suggest default limit path", [], dictionary.suggest("helo"))

# --- The `base` integrational dictionary ----------------------------------
# The dictionary behind the canonical "hlelo" -> "hello" conformance example.

base_dic = os.path.join(FIXTURES_DIR, "spec/integrational/fixtures/base")
base = kotoshu_native.Dictionary.load(f"{base_dic}.aff", f"{base_dic}.dic")
check_equal("base correct('hello')", True, base.correct("hello"))
check_equal("base correct('hlelo') — conformance vector", False, base.correct("hlelo"))
check_equal(
    "base suggest('hlelo', 5) — conformance vector",
    [
        {
            "word": "hello",
            "distance": 1,
            "confidence": 1.0,
            "source": "edit_distance",
        }
    ],
    base.suggest("hlelo", 5),
)

suggestions = base.suggest("helo", 5)
check("base suggest('helo', 5) includes 'hello'",
      any(suggestion["word"] == "hello" for suggestion in suggestions))
if not suggestions:
    check("suggestion row shape", False)
else:
    row = suggestions[0]
    check(
        "row is a dict of exactly the gem Suggestion fields",
        isinstance(row, dict) and sorted(row) == ["confidence", "distance", "source", "word"],
    )
    check("distance is an int", type(row["distance"]) is int)
    check(
        "confidence is a float in [0, 1]",
        type(row["confidence"]) is float and 0.0 <= row["confidence"] <= 1.0,
    )
    check(
        "source is a strategy str",
        isinstance(row["source"], str)
        and row["source"]
        in ("edit_distance", "phonetic", "keyboard_proximity", "ngram"),
    )
check("default-limit suggest stays within the limit", len(base.suggest("hlelo")) <= 5)

# --- Error surface --------------------------------------------------------

try:
    kotoshu_native.Dictionary.load(
        "/nonexistent-kotoshu.aff", "/nonexistent-kotoshu.dic"
    )
    check("missing dictionary raises KotoshuNativeError", False)
except kotoshu_native.KotoshuNativeError as error:
    check("missing dictionary raises KotoshuNativeError", True)
    check(
        "error message names the Rust failure and the path",
        "/nonexistent-kotoshu.aff" in str(error) and "failed to load" in str(error),
    )

try:
    kotoshu_native.Dictionary()
    check("Dictionary() has no public constructor", False)
except kotoshu_native.KotoshuNativeError as error:
    check(
        "Dictionary() has no public constructor",
        "Dictionary.load" in str(error),
    )

print(
    f"python ffi smoke: {assertions} assertions, {failures} failures"
    f" (kotoshu-rs {kotoshu_native.VERSION}, python {sys.version.split()[0]})"
)
sys.exit(0 if failures == 0 else 1)
