#!/usr/bin/env bash
# Build the @kotoshu/worker npm package payload (plan 112) — the mirror
# of scripts/wasm_build.sh for the pure-JS worker package.
#
#   scripts/worker_build.sh    # assemble kotoshu-worker/pkg/ from src/
#
# The sources are plain ESM JavaScript with hand-written .d.ts (no JS
# toolchain is committed), so the build is an assembly: copy the src
# modules flat into pkg/, copy the metadata package.json + README, and
# validate the npm name the way wasm_build.sh rewrites the wasm-pack
# one (wasm-pack derives it from the crate name; here the source file
# already carries @kotoshu/worker, and a drift fails loudly instead of
# shipping kotoshu-worker unscoped). pkg/ is a gitignored build
# artifact; publishing is BLOCKED on the owner-side npm registration
# and version decision — see kotoshu-worker/RELEASING.md.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PKG_SRC="$ROOT/kotoshu-worker"
OUT_DIR="${WORKER_PKG_OUT_DIR:-pkg}"

rm -rf "$PKG_SRC/$OUT_DIR"
mkdir -p "$PKG_SRC/$OUT_DIR"
cp "$PKG_SRC"/src/*.js "$PKG_SRC"/src/*.d.ts "$PKG_SRC/$OUT_DIR/"
cp "$PKG_SRC/README.md" "$PKG_SRC/$OUT_DIR/"
cp "$PKG_SRC/package.json" "$PKG_SRC/$OUT_DIR/package.json"
node -e '
  const [pkgPath] = process.argv.slice(1);
  const fs = require("fs");
  const pkg = JSON.parse(fs.readFileSync(pkgPath, "utf8"));
  if (pkg.name !== "@kotoshu/worker") {
    throw new Error("worker_build: package.json name drifted to " + pkg.name);
  }
  console.log("worker_build: " + pkgPath + " name -> @kotoshu/worker, version " + pkg.version);
' "$PKG_SRC/$OUT_DIR/package.json"
