# Releasing `@kotoshu/worker`

**Publishing is BLOCKED.** The attempted manual 0.1.0 publish did not land (registry 404 as of 2026-09-09); retry the manual publish, verify with `npm view @kotoshu/worker version`, then register the trusted publisher. Before that registration: The npm trusted-publisher registration on
npmjs.com exists for `@kotoshu/wasm` only; `@kotoshu/worker` needs the
same owner-side registration (Repository `kotoshu/kotoshu-rs`, Workflow
`release-npm-worker.yml`, no environment) before the keyless publish in
that workflow can succeed. Nothing has been published under this name;
run no publish step until the owner registers it AND states the first
version. Version lines are independent per package (parsanol policy):
the `0.1.0` in `kotoshu-worker/package.json` is a PLACEHOLDER, an owner
decision. The package version tracks the `@kotoshu/wasm` surface it
speaks (bump `DEFAULT_WASM_VERSION` in `src/engine-core.js` with it),
but the numbers themselves are owner decisions.

## Build (works today, publish-free)

```sh
scripts/worker_build.sh    # assemble kotoshu-worker/pkg/ from src/
```

The sources are plain ESM JavaScript with hand-written `.d.ts` — no JS
toolchain is committed — so the build assembles and validates (the npm
name must already be `@kotoshu/worker`). `pkg/` is gitignored: a build
artifact, never committed.

## Verify (what CI runs on every PR, in wasm.yml)

```sh
scripts/wasm_build.sh                      # the engine under test
scripts/worker_build.sh                    # the package payload
node scripts/worker_node_smoke.mjs         # full protocol over the local pkg
```

## Publish (ONLY when registered and the owner names the version)

1. Set `version` in `kotoshu-worker/package.json` to the owner-stated
   number (the first release and every bump are owner decisions).
2. `scripts/worker_build.sh`
3. `node scripts/worker_node_smoke.mjs` (must pass)
4. Tag `@kotoshu/worker-v<version>` and push it: `release-npm-worker.yml`
   builds, smoke-tests, and publishes from `kotoshu-worker/pkg` with
   OIDC trusted publishing and provenance (a scoped package needs the
   public access flag, which the workflow passes). Tags are the owner
   decision, every time.
