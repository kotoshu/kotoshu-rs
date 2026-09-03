#!/usr/bin/env bash
# Sync the gem's conformance vectors + Hunspell fixture dictionaries.
#
# Copies, from the Ruby gem repository (KOTOSHU_GEM_DIR, default ../kotoshu
# relative to this repo's root):
#
#   * conformance/vectors.jsonl        -> tests/fixtures/vectors.jsonl
#   * every distinct `dictionary` the
#     vectors reference (<path>.aff/.dic, gem-relative) -> tests/fixtures/<path>.aff/.dic
#
# The gem-relative layout is preserved so vector `dictionary` fields resolve
# directly against tests/fixtures/. Fixture files are never committed here
# (tests/fixtures/ is gitignored): they stay in the gem repo, which documents
# their provenance and keeps this repository's licensing clean.
#
# Idempotent: re-running refreshes the copies in place. Exits non-zero when
# the gem directory or the vectors file is missing, so CI fails loudly
# instead of silently skipping the conformance assertions.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GEM_DIR="${KOTOSHU_GEM_DIR:-$ROOT/../kotoshu}"
SRC_VECTORS="$GEM_DIR/conformance/vectors.jsonl"
DEST="$ROOT/tests/fixtures"

if [ ! -d "$GEM_DIR" ]; then
  echo "sync_conformance: gem directory not found: $GEM_DIR" >&2
  echo "sync_conformance: set KOTOSHU_GEM_DIR or clone kotoshu/kotoshu next to this repo" >&2
  exit 1
fi

if [ ! -f "$SRC_VECTORS" ]; then
  echo "sync_conformance: $SRC_VECTORS not found (run \`rake kotoshu:conformance:export\` in the gem)" >&2
  exit 1
fi

mkdir -p "$DEST"
cp -f "$SRC_VECTORS" "$DEST/vectors.jsonl"

dicts=$(grep -o '"dictionary":"[^"]*"' "$SRC_VECTORS" | sed 's/"dictionary":"//; s/"$//' | sort -u)
count=0
for dict in $dicts; do
  aff_src="$GEM_DIR/$dict.aff"
  dic_src="$GEM_DIR/$dict.dic"
  if [ ! -f "$aff_src" ] || [ ! -f "$dic_src" ]; then
    echo "sync_conformance: missing fixture for $dict (need $aff_src and $dic_src)" >&2
    exit 1
  fi
  dir="$DEST/$(dirname "$dict")"
  mkdir -p "$dir"
  cp -f "$aff_src" "$DEST/$dict.aff"
  cp -f "$dic_src" "$DEST/$dict.dic"
  count=$((count + 1))
done

vectors=$(grep -cve '^[[:space:]]*$' "$SRC_VECTORS")
echo "sync_conformance: $vectors vectors, $count dictionaries synced into $DEST"
