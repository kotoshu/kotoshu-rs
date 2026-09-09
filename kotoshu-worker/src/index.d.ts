/**
 * @file Type surface of @kotoshu/worker (plan 112 port of the playground
 * engine worker). The implementation is plain ESM JavaScript; these
 * declarations are the contract consumers and the d.ts-consuming editors
 * see. They must stay in lockstep with engine-core.js / semantic-merge.js.
 */

/** A suggestion row — the engine conformance SUGGESTION_KEYS shape. */
export interface Suggestion {
  word: string
  distance: number
  confidence: number
  source: string
}

/** A semanticSuggest row — the wasm pair of word + cosine. */
export interface SemanticNeighbor {
  word: string
  score: number
}

/** How many neighbors the semantic layer asks the model for. */
export const SEMANTIC_SUGGEST_K: 4

/**
 * Fold model-generated neighbors into dictionary candidates (see
 * semantic-merge.js): case-insensitive dedupe with dictionary rows
 * winning, generated rows labeled source "semantic" with the cosine
 * clamped to [0, 1] as confidence and distance 0.
 */
export function mergeSemanticCandidates(
  dictionary: Suggestion[],
  neighbors: SemanticNeighbor[] | null,
): Suggestion[]

/** Default pins — the exact sources the live playground resolves. */
export const DEFAULT_WASM_VERSION: string
export const DEFAULT_DICT_PIN: string
export const DEFAULT_REGISTRY_TAG: string
export const DEFAULT_CACHE_NAME: string

/** Engine source override for hosts that cannot https-import a URL. */
export interface EngineSource {
  /** ES module exporting the wasm-bindgen glue (kotoshu_wasm_bg.js). */
  glue: string | URL
  /** The raw .wasm binary the glue imports. */
  wasm: string | URL
}

export interface EngineOptions {
  /** @kotoshu/wasm npm version pin (default 0.4.0). */
  wasmVersion?: string
  /** kotoshu/dictionaries commit pin (jsDelivr /gh serves SHAs). */
  dictPin?: string
  /** kotoshu/models-fasttext-onnx registry tag pin. */
  registryTag?: string
  /**
   * Override the registry URL outright (a https:/file: URL or path).
   * Default resolves raw.githubusercontent .../{registryTag}/registry.json;
   * a file: URL needs options.readFile, like a file: engineSource.
   */
  registryUrl?: string
  /**
   * Opt into language packs (plan 113): when the pinned registry carries
   * kotoshu://packs/{lang} and the engine exposes loadPack, a load
   * fetches ONE artifact instead of aff+dic (and semantic-enable reuses
   * the pack model instead of fetching the tier trio). Any miss — no
   * entry, old engine, failed fetch — degrades to the per-resource path,
   * never an error. Default false.
   */
  pack?: boolean
  /** Cache Storage bucket name (browsers only; absent caches degrade to fetch). */
  cacheName?: string
  /** Override both engine URLs — e.g. a local @kotoshu/wasm pkg under Node. */
  engineSource?: EngineSource
  /**
   * How file: engine sources are materialized (Node fs/promises readFile).
   * The package imports nothing from node itself; a file: engineSource
   * without this rejects with a pointer to it.
   */
  readFile?: (url: string) => Promise<ArrayBuffer | Uint8Array>
}

/** Messages the engine handles (the worker wires these to onmessage). */
export type EngineMessage =
  | { type: 'load'; lang: string }
  | { type: 'check'; text: string }
  | { type: 'suggest'; word: string; context?: string }
  | { type: 'suggest-batch'; items: Array<{ word: string; context: string }> }
  | { type: 'semantic'; enable: boolean; lang?: string }
  | { type: 'detect'; text: string }

/** Events the engine emits (the worker postMessages these). */
export type EngineEvent =
  | {
      type: 'load-progress'
      phase: 'engine' | 'pack' | 'dictionary' | 'semantic' | 'detect'
      kind: string
      loaded: number
      total: number
    }
  | {
      type: 'loaded'
      engineCached: boolean
      dictCached: boolean
      lang: string | undefined
      engineBytes: number
      dictionaryBytes: number
      engineVersion: string
      wasmVersion: string
      loadMs: number
      /** True when the language arrived as one pack artifact (plan 113). */
      packed: boolean
    }
  | { type: 'load-error'; lang: string | undefined; message: string }
  | { type: 'checked'; words: string[]; ms: number }
  | {
      type: 'suggested'
      word: string
      suggestions: Suggestion[]
      sweepMs: number
      semanticMs: number
      rerankMs: number
      semantic: boolean
    }
  | {
      type: 'semantic-status'
      lang: string | null
      state: 'off' | 'loading' | 'ready' | 'unavailable'
      detail?: string
      modelBytes: number
      bucketsBytes: number
      modelCached: boolean
    }
  | {
      type: 'detected'
      code: string
      score: number
      ms: number
      modelBytes: number
      modelCached: boolean
    }
  | { type: 'detect-error'; message: string }

/** The protocol core — one engine instance, one alive dictionary at a time. */
export interface Engine {
  /** Handle one protocol message. Never rejects; failures arrive as events. */
  handle(message: EngineMessage): Promise<void>
}

/**
 * Build an engine speaking the playground worker protocol.
 * @param onMessage receives every event the worker would postMessage.
 */
export function createEngine(
  onMessage: (message: EngineEvent) => void,
  options?: EngineOptions,
): Engine
