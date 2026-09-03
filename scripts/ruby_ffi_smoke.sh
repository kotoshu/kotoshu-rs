#!/usr/bin/env bash
# Build the `ruby` feature's reference shim extension (tests/ruby_ext — the
# exact cdylib shape the kotoshu gem's ext/kotoshu_native will use) and run
# the Ruby smoke test (tests/ruby_ffi_smoke.rb) against it with the `ruby`
# found on PATH. Requires the conformance fixtures to be synced first
# (scripts/sync_conformance.sh).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXT_DIR="$ROOT/tests/ruby_ext"
PROFILE="${CARGO_PROFILE:-debug}"
LIB_DIR="$ROOT/target/ruby-ext"

ruby --version

# Build from the extension's own directory: its .cargo/config.toml carries
# the macOS `-undefined dynamic_lookup` link argument, and cargo reads
# config from the working directory, not the manifest directory.
# CARGO_PROFILE=release builds optimized (artifacts under target/release).
PROFILE_DIR="debug"
PROFILE_ARGS=()
if [ "${CARGO_PROFILE:-}" = "release" ]; then
  PROFILE_DIR="release"
  PROFILE_ARGS=(--release)
fi
( cd "$EXT_DIR" && cargo build "${PROFILE_ARGS[@]}" --target-dir "$ROOT/target" )

lib=""
for candidate in "$ROOT/target/$PROFILE_DIR/libkotoshu_ruby_ext.dylib" \
                 "$ROOT/target/$PROFILE_DIR/libkotoshu_ruby_ext.so"; do
  if [ -f "$candidate" ]; then lib="$candidate"; break; fi
done
if [ -z "$lib" ]; then
  echo "ruby_ffi_smoke: built cdylib not found under $ROOT/target/$PROFILE" >&2
  exit 1
fi

# `require "kotoshu_ruby_ext"` needs the platform's extension suffix
# (.bundle on macOS, .so elsewhere) and the bare library name.
case "$(uname -s)" in
  Darwin) dlext=bundle ;;
  *) dlext=so ;;
esac
mkdir -p "$LIB_DIR"
cp -f "$lib" "$LIB_DIR/kotoshu_ruby_ext.$dlext"

exec ruby -I"$LIB_DIR" "$ROOT/tests/ruby_ffi_smoke.rb"
