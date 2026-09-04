# Releasing `@kotoshu/wasm`

**Publishing is BLOCKED.** The npm org credentials for the `kotoshu` scope
do not exist yet (plan 67 M5). Nothing has ever been published from this
repository; run no publish step until the owner supplies the credentials
AND states the first version. Version lines are independent per package
(parsanol policy): the `0.1.0` in this crate's `Cargo.toml` (and in the
generated `pkg/package.json`) is a PLACEHOLDER, an owner decision.

## Build (works today, publish-free)

```sh
scripts/wasm_build.sh                          # wasm-pack --target bundler -> pkg/
WASM_PACK_TARGET=web scripts/wasm_build.sh     # fetch()-based ES module variant
```

The script rewrites the generated `pkg/package.json` name to
`@kotoshu/wasm` — wasm-pack derives the name from the crate name
("kotoshu-wasm") and cannot express npm scopes. `pkg/` is gitignored: it
is a build artifact, never committed.

## Verify (what CI runs on every PR)

```sh
scripts/sync_conformance.sh                    # fixtures from the gem repo
node scripts/wasm_node_smoke.mjs               # Node >= 24 against the bundler pkg
                                               # (older Node: WASM_PACK_TARGET=web first)
cargo build -p kotoshu-wasm --features wasm --target wasm32-unknown-unknown --release
```

## Publish (ONLY when credentials exist and the owner names the version)

1. Set this crate's `version` in `Cargo.toml` to the owner-stated number
   (the first release and every bump are owner decisions).
2. `scripts/wasm_build.sh`
3. `node scripts/wasm_node_smoke.mjs` (must pass)
4. From this directory: `wasm-pack publish` (equivalent to
   `cd pkg && npm publish --access public` — a scoped package needs the
   public access flag on first publish).

Optional, not wired into CI: `wasm-pack test --node --features wasm` runs
the Rust tests under Node; the JS smoke test above already exercises the
exported surface end-to-end.
