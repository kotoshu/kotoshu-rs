#!/usr/bin/env bash
# Build the @kotoshu/wasm npm package payload (plan 66 P4c) with wasm-pack.
#
#   scripts/wasm_build.sh                        # --target bundler -> pkg/
#   WASM_PACK_TARGET=web scripts/wasm_build.sh   # fetch()-based ES module
#   WASM_PACK_OUT_DIR=pkg-web ...                # alternate output dir
#
# The generated package.json name is rewritten to @kotoshu/wasm: wasm-pack
# derives it from the crate name ("kotoshu-wasm") and cannot express npm
# scopes. pkg/ is a gitignored build artifact; publishing is BLOCKED on npm
# org credentials (plan 67 M5) — see kotoshu-wasm/RELEASING.md.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CRATE="$ROOT/kotoshu-wasm"
TARGET="${WASM_PACK_TARGET:-bundler}"
OUT_DIR="${WASM_PACK_OUT_DIR:-pkg}"
wasm-pack build "$CRATE" --target "$TARGET" --out-dir "$OUT_DIR" --features wasm
node -e '
  const [pkgPath] = process.argv.slice(1);
  const fs = require("fs");
  const pkg = JSON.parse(fs.readFileSync(pkgPath, "utf8"));
  pkg.name = "@kotoshu/wasm";
  fs.writeFileSync(pkgPath, JSON.stringify(pkg, null, 2) + "\n");
  console.log("wasm_build: " + pkgPath + " name -> @kotoshu/wasm, version " + pkg.version);
' "$CRATE/$OUT_DIR/package.json"
