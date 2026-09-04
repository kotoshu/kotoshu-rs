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
# never committed). Reused across runs; recreate by deleting it. A venv
# restored from the CI cargo cache can lose its pyvenv.cfg — the python
# then silently behaves as the system interpreter (pip "succeeds" while
# installing maturin into /opt/hostedtoolcache, and $VENV/bin/maturin
# never appears) — so validate that it is still a venv by checking
# sys.prefix, and that maturin still runs, rather than file existence.
if ! "$VENV/bin/python" -c 'import sys; sys.exit(sys.prefix == sys.base_prefix)' >/dev/null 2>&1; then
  rm -rf "$VENV"
  "$PYTHON" -m venv "$VENV"
fi
VENV_PY="$VENV/bin/python"
"$VENV_PY" -m pip install --quiet --upgrade pip

# maturin from the venv when absent or no longer runnable (a system
# maturin would also do — this just pins the tool to the interpreter
# under test). --force-reinstall because a stale dist-info makes a plain
# install a no-op that would leave the broken launcher in place.
if ! "$VENV/bin/maturin" --version >/dev/null 2>&1; then
  "$VENV_PY" -m pip install --force-reinstall 'maturin>=1.5,<2.0'
fi
if ! "$VENV/bin/maturin" --version >/dev/null 2>&1; then
  echo "python_smoke: $VENV/bin/maturin still not runnable after install:" >&2
  ls -la "$VENV/bin" >&2 || true
  stat "$VENV/bin/maturin" >&2 || true
  head -c 16 "$VENV/bin/maturin" 2>/dev/null | od -An -tx1 >&2 || true
  "$VENV_PY" -m pip show -f maturin >&2 || true
  exit 1
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
