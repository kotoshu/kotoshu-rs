# 06 — Publishing: version ledger and channels

Records the publish wave (plan 67 M5 / gem PR #110 era), so versions
and channel states have one home. Version numbers are the owner's
standing decision; the ledger records what IS.

## Version ledger (2026-09-04)

| Package | Version | Channel | State |
|---|---|---|---|
| `kotoshu` (pure Python client) | 0.1.0 | PyPI | **LIVE** — pypi.org/project/kotoshu/0.1.0/ |
| `kotoshu-native` (maturin wheel, module `kotoshu_native`) | 0.1.0 | PyPI | **LIVE** — pypi.org/project/kotoshu-native/0.1.0/ (sdist + cp310 macOS arm64 wheel; more platforms via CI later) |
| `@kotoshu/wasm` | 0.1.0 | npm | **LIVE** (2026-09-05, owner-published). Keyless CI publish wired: release-npm.yml, tag `@kotoshu/wasm-v*` |
| `@kotoshu/client` (JS) | 0.1.0 | npm | **LIVE** (2026-09-05, owner-published). Keyless CI publish wired: kotoshu-js release.yml |
| `kotoshu` gem | **0.7.0** | RubyGems | **LIVE** (2026-09-05) — rubygems.org/gems/kotoshu/versions/0.7.0; the universal-kotoshu cut: tiers+registry, native ext, conformance export, cascade, correctness wave. Git tag not pushed (owner action). Trusted publishing wired both sides (gem repo release.yml + rubygems.org registration). |
| kotoshu-rs `kotoshu` crate | 0.1.0 | crates.io | **LIVE** (2026-09-05, owner-published from ~/.cargo/credentials.toml). Keyless CI publish wired: release-crate.yml, tag `kotoshu-v*` |

## Trusted publishing (keyless OIDC) registrations

| Registry | Package | Repo/workflow | Environment | Owner-side |
|---|---|---|---|---|
| npm | `@kotoshu/wasm` | kotoshu/kotoshu-rs · release-npm.yml | none | done |
| npm | `@kotoshu/client` | kotoshu/kotoshu-js · release.yml | none | done |
| RubyGems | `kotoshu` gem | kotoshu/kotoshu · release.yml | per gem registration | done |
| crates.io | `kotoshu` crate | kotoshu/kotoshu-rs · release-crate.yml | none | **pending** — crate Settings → Trusted Publishing → Add: GitHub, owner `kotoshu`, repo `kotoshu-rs`, workflow `release-crate.yml`, no environment |

crates.io trusted publishing (RFC 3691): CI exchanges the GitHub OIDC
token via `rust-lang/crates-io-auth-action@v1` for a 30-minute
`CARGO_REGISTRY_TOKEN`; no long-lived token stored anywhere. First
publish of a crate must use a token (done for 0.1.0) — keyless covers
0.1.1+.

## Version policy

First releases are 0.1.0 across the ecosystem (matches kotoshu-lsp,
kotoshu-server, kotoshu-go precedent). Semver from here; breaking
changes bump minor while 0.x. The gem's native-ext release and any
crate publishing remain explicit owner actions.

## Status

**Published** — all six packages live on their registries; keyless CI
publish paths wired for npm, RubyGems, and crates.io. Remaining
trusted-publishing migrations if wanted: PyPI
(pypa/gh-action-pypi-publish + owner registration at
pypi.org/manage/account/publishing).
