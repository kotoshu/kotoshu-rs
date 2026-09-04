# 06 — Publishing: version ledger and channels

Records the publish wave (plan 67 M5 / gem PR #110 era), so versions
and channel states have one home. Version numbers are the owner's
standing decision; the ledger records what IS.

## Version ledger (2026-09-04)

| Package | Version | Channel | State |
|---|---|---|---|
| `kotoshu` (pure Python client) | 0.1.0 | PyPI | **LIVE** — pypi.org/project/kotoshu/0.1.0/ |
| `kotoshu-native` (maturin wheel, module `kotoshu_native`) | 0.1.0 | PyPI | **LIVE** — pypi.org/project/kotoshu-native/0.1.0/ (sdist + cp310 macOS arm64 wheel; more platforms via CI later) |
| `@kotoshu/wasm` | 0.1.0 | npm | **BUILT, BLOCKED** — kotoshu-wasm/pkg/ ready; npm token invalid (E401) + org `kotoshu` not yet created |
| `@kotoshu/client` (JS) | 0.1.0 | npm | **BUILT, BLOCKED** — dist/ ready, smoke 7/7 |
| `kotoshu` gem | 0.6.x | RubyGems | live (native ext lands in the NEXT gem release — version the owner's call) |
| kotoshu-rs crates | 0.1.0 | crates.io | **NOT PUBLISHED** — release-plz gated on `RELEASE_PLZ_ENABLED` (owner) |

## npm unblock (owner, two steps)

1. Create the org: npmjs.com/org/create → `kotoshu`; then `npm login`
   (the current token returns E401).
2. Publish (or hand back):
   - `cd kotoshu-rs/kotoshu-wasm/pkg && npm publish --access public`
   - `cd ../kotoshu-js && npm publish --access public`

## Version policy

First releases are 0.1.0 across the ecosystem (matches kotoshu-lsp,
kotoshu-server, kotoshu-go precedent). Semver from here; breaking
changes bump minor while 0.x. The gem's native-ext release and any
crate publishing remain explicit owner actions.

## Status

**In progress** — PyPI complete; npm blocked on credentials; crates.io
deliberately owner-gated.
