#!/usr/bin/env python3
"""Wheel smoke: install-and-import verification for built kotoshu_native wheels.

The matrix twin of scripts/python_smoke.py: that one drives the REAL gem
conformance fixtures (a bash-only sync away — run on ubuntu by
python-ffi.yml), while this one is self-contained so it can run on every
platform of the wheel matrix (linux x86_64/aarch64, macOS x86_64/arm64,
windows) right after `pip install` of the built wheel. It writes a minimal
Hunspell dictionary into a temp dir and exercises the same module surface:
VERSION, available(), Dictionary.load/correct/suggest (row shape), and the
KotoshuNativeError error path. The frozen gem conformance rows stay
covered by scripts/python_smoke.py.
"""

import os
import sys
import tempfile

MINIMAL_AFF = "\n".join(
    [
        "# Minimal dictionary for the wheel smoke (modelled on the gem's",
        "# spec/fixtures/dictionaries/hunspell/test.aff)",
        "SET UTF-8",
        "TRY esianrtolcdugmphbyfvkwz",
        "",
    ]
)

MINIMAL_DIC = "\n".join(
    [
        "3",
        "hello",
        "shell",
        "world",
        "",
    ]
)

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


try:
    import kotoshu_native
except ImportError as error:  # the wheel was not installed into this interpreter
    sys.exit(f"python_wheel_smoke: kotoshu_native not importable: {error}")

# --- Module surface ---------------------------------------------------------

check(
    "kotoshu_native.Dictionary is defined",
    hasattr(kotoshu_native, "Dictionary"),
)
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

# --- The minimal dictionary --------------------------------------------------

with tempfile.TemporaryDirectory(prefix="kotoshu-wheel-smoke-") as tmp:
    aff_path = os.path.join(tmp, "smoke.aff")
    dic_path = os.path.join(tmp, "smoke.dic")
    with open(aff_path, "w", encoding="utf-8") as aff:
        aff.write(MINIMAL_AFF)
    with open(dic_path, "w", encoding="utf-8") as dic:
        dic.write(MINIMAL_DIC)

    dictionary = kotoshu_native.Dictionary.load(aff_path, dic_path)
    check("Dictionary instance", isinstance(dictionary, kotoshu_native.Dictionary))
    check_equal("correct('hello')", True, dictionary.correct("hello"))
    check_equal("correct('world')", True, dictionary.correct("world"))
    check_equal("correct('hlelo')", False, dictionary.correct("hlelo"))
    check_equal("correct('ruby')", False, dictionary.correct("ruby"))

    rows = dictionary.suggest("hlelo", 5)
    check("suggest('hlelo', 5) is non-empty", len(rows) > 0)
    check(
        "suggest('hlelo', 5) leads with 'hello'",
        len(rows) > 0 and rows[0]["word"] == "hello",
    )
    if rows:
        row = rows[0]
        check(
            "row is a dict of exactly the gem Suggestion fields",
            isinstance(row, dict)
            and sorted(row) == ["confidence", "distance", "source", "word"],
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
    check("default-limit suggest stays within the limit", len(dictionary.suggest("hlelo")) <= 5)

# --- Error surface -----------------------------------------------------------

try:
    kotoshu_native.Dictionary.load(
        "/nonexistent-kotoshu.aff", "/nonexistent-kotoshu.dic"
    )
    check("missing dictionary raises KotoshuNativeError", False)
except kotoshu_native.KotoshuNativeError:
    check("missing dictionary raises KotoshuNativeError", True)

print(
    f"python wheel smoke: {assertions} assertions, {failures} failures"
    f" (kotoshu-rs {kotoshu_native.VERSION}, python {sys.version.split()[0]},"
    f" {sys.platform})"
)
sys.exit(0 if failures == 0 else 1)
