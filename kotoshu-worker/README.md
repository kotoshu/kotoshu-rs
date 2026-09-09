# @kotoshu/worker

The kotoshu engine worker, published: the playground worker as a
package, so JS users embed the engine without reimplementing it. One
module keeps every engine call off the main thread — dictionary load,
check, the priority suggest queue, the semantic tier, language
detection — resolving the same pinned sources the live playground
resolves: `@kotoshu/wasm` from jsDelivr, dictionaries from the pinned
`kotoshu/dictionaries` commit (both on-disk layouts), models through
the tag-pinned `kotoshu/models-fasttext-onnx` registry, all persisted
in Cache Storage on repeat visits.

## The worker

```js
import EngineWorker from '@kotoshu/worker/engine-worker'

const worker = new EngineWorker()   // bundlers inline the new URL(...) pair
worker.onmessage = (event) => { /* one switch over event.type */ }
worker.postMessage({ type: 'load', lang: 'en' })
```

## The protocol core (no Worker required)

```js
import { createEngine } from '@kotoshu/worker'

const engine = createEngine((event) => {
  if (event.type === 'checked') render(event.words)
  if (event.type === 'suggested') render(event.suggestions)
})

await engine.handle({ type: 'load', lang: 'en' })
await engine.handle({ type: 'check', text: 'I recieved teh leters' })
await engine.handle({ type: 'suggest', word: 'recieved', context: 'the letter' })
```

Under Node (no Cache Storage, no https dynamic import) point
`engineSource` at a local `@kotoshu/wasm` package directory and hand the
engine a `readFile` for its file: sources — exactly what
`scripts/worker_node_smoke.mjs` in kotoshu-rs does:

```js
import { readFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'

const engine = createEngine(onMessage, {
  engineSource: {
    glue: new URL('./node_modules/@kotoshu/wasm/kotoshu_wasm_bg.js', import.meta.url),
    wasm: new URL('./node_modules/@kotoshu/wasm/kotoshu_wasm_bg.wasm', import.meta.url),
  },
  readFile: (url) => readFile(fileURLToPath(url)),
})
```

## Protocol

In (messages): `load {lang}`, `check {text}`, `suggest {word, context?}`
(priority single), `suggest-batch {items: [{word, context}]}` (drops the
pending non-priority words), `semantic {enable, lang?}`, `detect {text}`.

Out (events): `load-progress {phase, kind, loaded, total}`, `loaded
{lang, engineVersion, wasmVersion, loadMs, ...}`, `load-error`, `checked
{words, ms}`, `suggested {word, suggestions, sweepMs, semanticMs,
rerankMs, semantic}`, `semantic-status {lang, state, modelBytes,
bucketsBytes, ...}`, `detected {code, score, ms, modelBytes}`,
`detect-error`. `Suggestion` rows carry the engine conformance
SUGGESTION_KEYS (`word` / `distance` / `confidence` / `source`).

The semantic layer is opt-in per language and resolves
`kotoshu://models/{lang}/mini` plus the `buckets` sibling through the
registry mirror; a missing or failed tier degrades to dictionary-only
suggestions, never an error. One model is resident at a time and is
freed deterministically on disable or language switch.

## Pins

`wasmVersion` (default `0.4.0`), `dictPin`, `registryTag` and
`cacheName` are `createEngine` options; the defaults are the pins the
playground ships with, and the wasm pin tracks the published
`@kotoshu/wasm` version line. Optional engine members (`loadModel`,
`rerank`, `semanticSuggest`, `loadLid`, `detectLanguage`) are probed at
call time, so an engine build without them degrades to
dictionary-only instead of throwing.

License: BSD-2-Clause (see the repository root).
