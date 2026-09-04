#!/usr/bin/env bash
# Build the kotoshu_native extension wheel (kotoshu-python member, plan 66
# P4d) with maturin and run the Python smoke test (scripts/python_smoke.py)
# against it in a dedicated venv, with the `python3` found on PATH.
# Requires the conformance fixtures to be synced first
# (scripts/sync_conformance.sh).
#
#   PYTHON=python3.12 scripts/python_smoke.sh        # interpreter override
#   KOTOSHU_PYTHON_VENV=/path/to/venv ...            # reuse a venv
#
# Publishing the wheel is BLOCKED on PyPI credentials (plan 67 M5) — see
# kotoshu-python/RELEASING.md; this script never publishes.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MEMBER="$ROOT/kotoshu-python"
PYTHON="${PYTHON:-python3}"
VENV="${KOTOSHU_PYTHON_VENV:-$ROOT/target/python-venv}"
WHEELS_DIR="$ROOT/target/wheels"

"$PYTHON" --version

# The venv the wheel is built against and installed into (under target/,
# never committed). Reused across runs; recreate by deleting it.
if [ ! -x "$VENV/bin/python" ]; then
  "$PYTHON" -m venv "$VENV"
fi
VENV_PY="$VENV/bin/python"
"$VENV_PY" -m pip install --quiet --upgrade pip

# maturin from the venv when absent (a system maturin would also do — this
# just pins the tool to the interpreter under test).
if [ ! -x "$VENV/bin/maturin" ]; then
  "$VENV_PY" -m pip install --quiet 'maturin>=1.5,<2.0'
fi
"$VENV/bin/maturin" --version

# Build the wheel against THIS venv's interpreter (the [tool.maturin]
# table in pyproject.toml supplies the python feature). The dev loop
# equivalent is `maturin develop` inside an activated venv.
mkdir -p "$WHEELS_DIR"
( cd "$MEMBER" && "$VENV/bin/maturin" build --interpreter "$VENV_PY" --out "$WHEELS_DIR" )

wheel="$(ls -1 "$WHEELS_DIR"/kotoshu_native-*.whl | tail -n 1)"
ls -lh "$wheel"
"$VENV_PY" -m pip install --quiet --force-reinstall "$wheel"

"$VENV_PY" "$ROOT/scripts/python_smoke.py"
