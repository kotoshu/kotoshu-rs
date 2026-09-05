# Releasing `kotoshu-native`

**State: 0.1.0 is LIVE on PyPI** (pypi.org/project/kotoshu-native/0.1.0/
— the owner's first publish, sdist + one cp310 macOS arm64 wheel built
locally). CI now owns everything after that: the wheel matrix builds and
smoke-tests wheels for every supported platform, and the publish is
keyless (PyPI trusted publishing) behind an owner-pushed tag. No PyPI
token is stored anywhere; none is needed.

Version lines are independent per package (parsanol policy): the version
in this directory's `Cargo.toml` and `pyproject.toml` (kept in sync —
maturin reads the pyproject) is set by the owner for every release. The
same goes for the distribution name (`kotoshu-native`).

## What this wheel is

The `kotoshu_native` extension module over the same Rust engine as the
Ruby gem and `@kotoshu/wasm`: `VERSION`, `available()`, and a
`Dictionary` class (`load(aff_path, dic_path)`, `correct(word)`,
`suggest(word, limit = 5)` returning dicts of the conformance
`SUGGESTION_KEYS`), with every failure raising `KotoshuNativeError`
carrying the Rust message. The surface lives in `kotoshu/src/ffi/python`;
this member is only the maturin packaging shim.

Standalone-usable without the client package: `pip install
kotoshu-native` and `import kotoshu_native`. The PyPI `kotoshu` client
(kotoshu-python repository) consumes it via its `native` extra and the
`KOTOSHU_BACKEND=native|http|auto` guard.

## The wheel matrix (`.github/workflows/python-wheels.yml`)

| Platform | Runner | Build | Pythons |
|---|---|---|---|
| linux x86_64 (manylinux_2_28) | `ubuntu-latest` | container, one job | 3.10–3.13 |
| linux aarch64 (manylinux_2_28) | `ubuntu-24.04-arm` (native arm64) | container, one job | 3.10–3.13 |
| macOS x86_64 | `macos-15-intel` | host, one job each | 3.10–3.13 |
| macOS arm64 | `macos-15` | host, one job each | 3.10–3.13 |
| windows x64 | `windows-latest` | host, one job each | 3.10–3.13 |

Not shipped: Windows/Windows-on-ARM (`windows-arm64` runners cannot build
the MSVC x64 toolchain target and maturin cross-support there is
immature) and 32-bit anything — file an issue if a user shows up needing
them. Wheels are not abi3, so each CPython minor version gets its own
wheel file on every platform.

Every wheel is smoke-tested in-matrix before upload (`pip install` the
wheel, `import kotoshu_native`, real `Dictionary.load`/`correct`/`suggest`
over a minimal dictionary — `scripts/python_wheel_smoke.py`, the
fixture-free twin of `scripts/python_smoke.py`). The deep conformance
smoke over the gem fixtures stays in `python-ffi.yml`.

## Build and verify locally (publish-free)

```sh
scripts/python_smoke.sh                 # venv + maturin build + conformance smoke
# or, per-wheel:
python3 -m venv .venv && . .venv/bin/activate
pip install maturin
maturin develop                         # dev loop
maturin build --release --out dist      # wheel(s) into dist/
python -m pip install dist/*.whl && python ../scripts/python_wheel_smoke.py
```

## Publish (keyless, tag-gated)

1. Owner sets the version in BOTH this directory's `pyproject.toml` and
   `Cargo.toml` — every bump is an owner decision.
2. PR, merge to main (the release workflow must exist on the default
   branch for trusted publishing to authenticate).
3. Owner pushes the tag `kotoshu-native-v<version>` (the crates
   `kotoshu-v*` / npm `@kotoshu/wasm-v*` convention: distribution name
   plus `-v`). Tagging is always an owner action.
4. `release-pypi.yml` builds the full matrix + sdist (each wheel
   smoke-tested first), then publishes everything with
   `pypa/gh-action-pypi-publish` under `id-token: write`, attestations
   on. No token secret exists to leak.

Never publish a version that has ever been published (PyPI forbids
re-uploads even after yanking); never publish outside the tagged
workflow without the owner saying so.

## Owner-side trusted-publisher registration (one-time)

Both PyPI projects already exist, so registration is per-project (the
account-level page, pypi.org/manage/account/publishing/, only hosts
PENDING publishers for never-published names):

1. Merge the PR carrying `.github/workflows/release-pypi.yml` to main
   (PyPI authenticates the workflow file from the default branch).
2. Open `https://pypi.org/manage/project/kotoshu-native/settings/publishing/`
   and add a publisher:
   - Owner: `kotoshu`
   - Repository: `kotoshu-rs`
   - Workflow filename: `release-pypi.yml`
   - Environment name: leave blank (none — matches the crates.io and npm
     registrations; the workflow requests no GitHub environment)
3. First tagged release publishes keyless from then on.

(The sibling `kotoshu` distribution, published from the kotoshu-python
repository, registers the same way against `kotoshu/kotoshu-python` —
see that repository's README.)
