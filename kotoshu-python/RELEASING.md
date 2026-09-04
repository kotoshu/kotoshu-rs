# Releasing `kotoshu-native`

**Publishing is BLOCKED.** The PyPI token for this project does not exist
yet (plan 67 M5). Nothing has ever been published from this repository;
run no publish step until the owner supplies the credentials AND states
the first version. Version lines are independent per package (parsanol
policy): the `0.1.0` in this directory's `Cargo.toml` and
`pyproject.toml` (kept in sync — maturin reads the pyproject) is a
PLACEHOLDER, an owner decision. The distribution NAME is equally an owner
decision; `kotoshu-native` is the working name this plan expects.

## What this wheel is

The `kotoshu_native` extension module over the same Rust engine as the
Ruby gem and `@kotoshu/wasm`: `VERSION`, `available()`, and a
`Dictionary` class (`load(aff_path, dic_path)`, `correct(word)`,
`suggest(word, limit = 5)` returning dicts of the conformance
`SUGGESTION_KEYS`), with every failure raising `KotoshuNativeError`
carrying the Rust message. The surface lives in `kotoshu/src/ffi/python`;
this member is only the maturin packaging shim.

## How the PyPI `kotoshu` package will consume it

The `kotoshu` distribution is owned by the separate kotoshu-python
repository (a pure-Python HTTP client today). Its integration (this
repository's counterpart of the gem-side P4b PR) will add
`kotoshu-native` to that package's dependencies and import
`kotoshu_native` behind a backend guard (`KOTOSHU_BACKEND=native|http|auto`,
mirroring the gem's `native|ruby|auto`), materializing its own
`Suggestion` types from the row dicts. Until then the wheel is fully
usable standalone: `pip install` it and `import kotoshu_native`.

## Build (works today, publish-free)

```sh
scripts/python_smoke.sh    # venv + maturin build + wheel install + smoke
```

or manually:

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install maturin
maturin develop                          # dev loop: build into the venv
maturin build --release --out dist       # wheel(s) into dist/
```

Wheel artifacts land in `target/wheels/` (smoke script) or `dist/`;
neither is committed. The smoke test (`scripts/python_smoke.py`, run by
the shell wrapper) asserts the frozen conformance row
(`suggest("hlelo")[0] == {"word": "hello", "distance": 1,
"confidence": 1.0, "source": "edit_distance"}`) over the synced gem
fixtures — run `scripts/sync_conformance.sh` first.

## Publish (ONLY when credentials exist and the owner names the version)

1. Set the owner-stated version in BOTH this directory's
   `pyproject.toml` and `Cargo.toml` (the first release and every bump
   are owner decisions).
2. `maturin build --release --out dist` (and `maturin sdist --out dist`
   if a source distribution is wanted).
3. `scripts/python_smoke.sh` (must pass).
4. `maturin publish` (or `twine upload dist/*`) — never before the
   credentials exist and the owner has confirmed the name and version.
