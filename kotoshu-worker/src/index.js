// @kotoshu/worker — the playground engine worker, published (plan 112).
//
// The package root re-exports the protocol core (createEngine + the
// semantic-merge half); the worker FILE lives at
// @kotoshu/worker/engine-worker for new Worker() wiring. See README.md
// for the full protocol.

export { createEngine } from './engine-core.js'
export {
  DEFAULT_CACHE_NAME,
  DEFAULT_DICT_PIN,
  DEFAULT_REGISTRY_TAG,
  DEFAULT_WASM_VERSION,
} from './engine-core.js'
export { SEMANTIC_SUGGEST_K, mergeSemanticCandidates } from './semantic-merge.js'
