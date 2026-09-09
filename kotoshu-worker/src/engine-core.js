// The engine worker core (plan 112 port of the playground
// engine-worker.ts, published as @kotoshu/worker).
//
// createEngine(onMessage, options) returns a message handler speaking
// the playground worker protocol — load / check / suggest /
// suggest-batch / semantic / detect in, typed events out — so JS users
// embed the engine without reimplementing the worker. The protocol and
// every policy below are the live playground behavior:
//
// - the engine loads @kotoshu/wasm from a version-pinned CDN (raw
//   jsDelivr npm files, not esm.sh — the package is a wasm-bindgen
//   bundler-target build whose entry imports the .wasm binary
//   directly; both transforms break on it);
// - with options.pack (plan 113, opt-in) a load first tries ONE pack
//   artifact per language (kotoshu://packs/{lang} through the pinned
//   registry: dict aff+dic + mini model + vocab + buckets, sha256-
//   verified per section inside loadPack); any miss — no entry, an
//   engine without loadPack, a failed fetch — degrades to the
//   per-resource paths below, never an error. The pack model rides
//   along resident, so semantic-enable then fetches nothing;
// - dictionaries load from the pinned kotoshu/dictionaries commit,
//   trying the wired layout ({lang}/spelling/index.*) first and the
//   flat one ({lang}/index.*) second — both exist at the pin;
// - the semantic tier (mini + buckets) and the LID model resolve
//   through the tag-pinned models registry;
// - repeat visits load from Cache Storage, not the network (when the
//   host has one — Node has none, and the worker degrades to plain
//   fetch instead of dying);
// - suggestion work runs through the priority queue: one batch message
//   for the pane, priority singles for the popover, a yield between
//   words so a click mid-batch jumps ahead.
//
// The wasm pin tracks the published @kotoshu/wasm version line; bump
// it together with the engine surface it speaks (loadModel/rerank,
// semanticSuggest, loadLid/detectLanguage are all optional members so
// an engine build without them degrades to dictionary-only instead of
// throwing at the call site).

import { SEMANTIC_SUGGEST_K, mergeSemanticCandidates } from './semantic-merge.js'

/** Default pins: the exact sources the live playground resolves. */
export const DEFAULT_WASM_VERSION = '0.4.0'
export const DEFAULT_DICT_PIN = '1829a3e2e67dc7ffb38f8dcd2d3d2294b6a8580d'
export const DEFAULT_REGISTRY_TAG = 'v1.5.0'
export const DEFAULT_CACHE_NAME = 'kotoshu-worker-v1'

/**
 * @typedef {object} EngineOptions
 * @property {string} [wasmVersion]    @kotoshu/wasm npm version pin.
 * @property {string} [dictPin]        kotoshu/dictionaries commit pin.
 * @property {string} [registryTag]    kotoshu/models-fasttext-onnx tag pin.
 * @property {string} [cacheName]      Cache Storage bucket name.
 * @property {{ glue: string | URL, wasm: string | URL }} [engineSource]
 *   Override both engine URLs — the ESM glue module and the .wasm
 *   binary. Non-browser embeds (Node, worker_threads) cannot https
 *   import a URL, so they point this at a local @kotoshu/wasm pkg.
 * @property {(url: string) => Promise<ArrayBuffer | Uint8Array>} [readFile]
 *   How file: engine sources are materialized (Node fs/promises
 *   readFile). The package imports nothing from node itself; a file:
 *   engineSource without this rejects with a pointer to it.
 */

function defaultEngineUrls(wasmVersion) {
  // Raw jsDelivr npm files: the package is a wasm-bindgen bundler-target
  // build whose entry imports the .wasm binary directly, so esm.sh /
  // esm.run transforms break it. Loading the two published files and
  // instantiating them here is the delivery both were trying to produce.
  const base = `https://cdn.jsdelivr.net/npm/@kotoshu/wasm@${wasmVersion}`
  return { glue: `${base}/kotoshu_wasm_bg.js`, wasm: `${base}/kotoshu_wasm_bg.wasm` }
}

/**
 * Build an engine speaking the playground worker protocol.
 *
 * @param {(message: Record<string, unknown>) => void} onMessage receives
 *   every event the worker would postMessage (loaded, checked,
 *   suggested, load-progress, semantic-status, detected, ...).
 * @param {EngineOptions} [options]
 * @returns {{ handle(message: Record<string, unknown>): Promise<void> }} the
 *   message handler a worker entry wires to self.onmessage.
 */
export function createEngine(onMessage, options = {}) {
  const wasmVersion = options.wasmVersion ?? DEFAULT_WASM_VERSION
  const readFileOption = options.readFile
  const engineUrls = options.engineSource
    ? {
        glue: String(options.engineSource.glue),
        wasm: String(options.engineSource.wasm),
      }
    : defaultEngineUrls(wasmVersion)

  // jsDelivr /gh serves tags and commit SHAs — branch pins 404. This
  // commit is the head of the dictionaries repo v1 branch, the same pin
  // the gem resolves at runtime.
  const dictBase = `https://cdn.jsdelivr.net/gh/kotoshu/dictionaries@${options.dictPin ?? DEFAULT_DICT_PIN}`
  // The models registry, tag-pinned (plan 92): raw.githubusercontent
  // serves the registry JSON with ACAO:* — it is a plain git file, never
  // an LFS pointer — and entry.urls.mirror points at the media host, the
  // only source that serves the tier bytes themselves with ACAO:* to
  // browsers (release assets and jsDelivr /gh do not). registryUrl
  // overrides the resolved URL outright (tests and self-hosted embeds).
  const registryUrl = options.registryUrl ??
    `https://raw.githubusercontent.com/kotoshu/models-fasttext-onnx/${options.registryTag ?? DEFAULT_REGISTRY_TAG}/registry.json`
  // Language packs (plan 113) are opt-in: one fetch for dict + mini
  // tier + buckets when the registry carries kotoshu://packs/{lang} and
  // the engine exposes loadPack. Every miss degrades to the
  // per-resource path below, never an error.
  const packMode = options.pack === true
  const cacheName = options.cacheName ?? DEFAULT_CACHE_NAME

  function post(type, payload) {
    onMessage({ type, ...payload })
  }

  // - engine state --------------------------------------------------

  /** @typedef {{ free(): void }} KotoshuModelHandle */
  /** @typedef {{ correct(word: string): boolean, suggest(word: string, limit?: number): Array<{word: string, distance: number, confidence: number, source: string}> }} KotoshuWasmInstance */

  let glue = null
  let engineBytes = 0
  let engineCached = false
  let dictCached = false
  let modelCached = false

  // Module-scope caches: one dictionary instance alive at a time (the
  // indexed dictionary dominates wasm memory; pt is 4.4 MB of source),
  // sources cached per language so switching back is reconstruct-only.
  const sourceCache = new Map()
  const suggestCache = new Map()
  let active = null
  let activeLang = null

  // The phase tells the UI WHERE the load is (engine, dictionary,
  // semantic); the kind names the artifact within it. total 0 means
  // indeterminate — the transfer carries no Content-Length.
  function postProgress(phase, kind, loaded, total) {
    post('load-progress', { phase, kind, loaded, total })
  }

  // Repeat visits load from Cache Storage, not the network: engine
  // binary, dictionaries, registry, and model tiers persist under
  // version-pinned URLs (a new pin is a new URL, so entries never go
  // stale; old-pin entries are evicted opportunistically on open).
  // Hosts without Cache Storage (Node) get the same fetches uncached.
  async function openCache() {
    if (typeof caches === 'undefined') return null
    try {
      return await caches.open(cacheName)
    } catch {
      return null
    }
  }

  function sameArtifact(a, b) {
    const tail = (u) => u.split('/').slice(2).join('/').replace(/@[^/]+/, '')
    return tail(a) === tail(b) && a !== b
  }

  /** cachedFetch with byte-level progress on the network path: the body
      streams through a reader so the UI can show how much arrived;
      cache hits report complete instantly. file: URLs (engineSource
      pointed at a local pkg) have no fetch and no cache — hosts that
      use them inject readFile, the bytes are read directly and reported
      complete at once (the package itself imports nothing from node). */
  async function cachedFetchProgress(url, phase, kind) {
    if (url.startsWith('file:')) {
      if (typeof readFileOption !== 'function') {
        throw new Error('file: engine sources need options.readFile (a Node fs/promises readFile)')
      }
      const bytes = new Uint8Array(await readFileOption(url))
      postProgress(phase, kind, bytes.byteLength, bytes.byteLength)
      return {
        res: new Response(bytes, {
          status: 200,
          headers: { 'content-length': String(bytes.byteLength) },
        }),
        cached: false,
      }
    }
    const cache = await openCache()
    const hit = cache ? await cache.match(url) : undefined
    if (hit) {
      const size = Number(hit.headers.get('content-length') ?? 0)
      postProgress(phase, kind, size, size)
      return { res: hit, cached: true }
    }
    const res = await fetch(url)
    if (!res.ok) return { res, cached: false }
    const total = Number(res.headers.get('content-length') ?? 0)
    const reader = res.body?.getReader()
    if (!reader) {
      postProgress(phase, kind, total, total)
      return { res, cached: false }
    }
    const chunks = []
    let loaded = 0
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      chunks.push(value)
      loaded += value.byteLength
      postProgress(phase, kind, loaded, total)
    }
    const rebuilt = new Response(new Blob(chunks), { status: 200, headers: res.headers })
    if (cache) {
      try {
        await cache.put(url, rebuilt.clone())
        const keys = await cache.keys()
        for (const key of keys) {
          if (key.url !== url && sameArtifact(key.url, url)) await cache.delete(key)
        }
      } catch {
        /* quota — the network result still returns */
      }
    }
    return { res: rebuilt, cached: false }
  }

  // wasm-bindgen names the wasm import module after the original relative
  // specifier; provide both spellings so a regenerated package still loads.
  function importObject(mod) {
    return {
      './kotoshu_wasm_bg.js': mod,
      wbg: mod,
    }
  }

  async function importGlue(url) {
    // A bundler-target pkg is an ES module; when its package.json lacks
    // "type": "module" (an alternate wasm-pack target), Node needs the
    // module attribute to parse .js as ESM — same fallback as
    // scripts/wasm_node_smoke.mjs.
    try {
      return await import(/* @vite-ignore */ url)
    } catch {
      return await import(/* @vite-ignore */ url, { with: { type: 'module' } })
    }
  }

  async function ensureEngine() {
    if (glue) return

    // The glue module import streams through the host module loader —
    // no byte count is observable there, so report the phase as live but
    // indeterminate while it loads.
    postProgress('engine', 'glue', 0, 0)
    glue = await importGlue(engineUrls.glue)

    const { res, cached } = await cachedFetchProgress(engineUrls.wasm, 'engine', 'wasm')
    if (!res.ok) throw new Error(`engine download failed: HTTP ${res.status}`)
    engineCached = cached
    const bytes = new Uint8Array(await res.arrayBuffer())
    const { instance } = await WebAssembly.instantiate(bytes, importObject(glue))
    glue.__wbg_set_wasm(instance.exports)
    const start = instance.exports.__wbindgen_start
    if (typeof start === 'function') start()
    engineBytes = bytes.byteLength
  }

  // Two layouts exist at the pin: the original six full-feature languages
  // keep their dictionaries under {lang}/spelling/, everything else —
  // including the thirteen new full-feature languages — sits flat at
  // {lang}/index.aff. Try the wired layout first, fall back to flat.
  async function fetchDictFile(lang, ext) {
    const paths = [`${lang}/spelling/index.${ext}`, `${lang}/index.${ext}`]
    let lastStatus = 0
    for (const path of paths) {
      const { res, cached } = await cachedFetchProgress(
        `${dictBase}/${path}`,
        'dictionary',
        `${lang} ${ext}`,
      )
      if (res.ok) {
        dictCached = cached
        return res.text()
      }
      lastStatus = res.status
    }
    throw new Error(
      `dictionary download failed: no index.${ext} for ${lang} at the pin (HTTP ${lastStatus})`,
    )
  }

  // - language packs (plan 113, optional) -------------------------------
  // ONE artifact per language: dict aff+dic + mini model + vocab +
  // buckets as a length-prefixed section stream, sha256-verified per
  // section by the engine (a corrupt fetch rejects inside loadPack; the
  // catch in loadLanguage degrades to the per-resource path). The pack
  // model rides along resident, so a later semantic-enable is free.
  async function loadLanguageFromPack(lang) {
    if (!packMode || typeof glue?.loadPack !== 'function') return null
    const registry = await ensureRegistry()
    const entry = registry.resources?.[`kotoshu://packs/${lang}`]
    const mirror = entry?.urls?.mirror
    if (!mirror) return null
    const { res, cached } = await cachedFetchProgress(mirror, 'pack', `${lang} pack`)
    if (!res.ok) return null
    const bytes = new Uint8Array(await res.arrayBuffer())
    const t0 = performance.now()
    const pack = glue.loadPack(bytes) // { dictionary, model }
    const contents = entry.contents ?? {}
    dropModel()
    model = {
      handle: pack.model,
      lang,
      sizeBytes: (contents.model?.length ?? 0) + (contents.vocab?.length ?? 0),
      bucketsBytes: contents.buckets?.length ?? 0,
      fromPack: true,
    }
    modelCached = cached
    dictCached = cached
    return {
      dictionary: pack.dictionary,
      sizeBytes: bytes.byteLength,
      loadMs: performance.now() - t0,
      packed: true,
    }
  }

  async function loadLanguage(lang) {
    if (active && activeLang === lang) return active
    await ensureEngine()

    let loaded = null
    try {
      loaded = await loadLanguageFromPack(lang)
    } catch {
      // A pack that rejects (corrupt bytes, engine mismatch) must never
      // take the language down — the per-resource path is complete.
      loaded = null
    }
    if (!loaded) {
      let sources = sourceCache.get(lang)
      if (!sources) {
        const [aff, dic] = await Promise.all([fetchDictFile(lang, 'aff'), fetchDictFile(lang, 'dic')])
        sources = { aff, dic, sizeBytes: aff.length + dic.length }
        sourceCache.set(lang, sources)
      }
      const t0 = performance.now()
      loaded = {
        dictionary: new glue.KotoshuWasm(sources.aff, sources.dic),
        sizeBytes: sources.sizeBytes,
        loadMs: performance.now() - t0,
        packed: false,
      }
    }

    active = {
      dictionary: loaded.dictionary,
      sizeBytes: loaded.sizeBytes,
      loadMs: loaded.loadMs,
      packed: loaded.packed,
    }
    activeLang = lang
    suggestCache.clear()
    return active
  }

  /** Letter runs with optional internal apostrophes — matches the spans the UI underlines. */
  const WORD_RE = /[\p{L}\p{M}]+(?:['’][\p{L}\p{M}]+)*/gu

  function check(text) {
    if (!active) return { words: [], ms: 0 }
    const t0 = performance.now()
    const misspelled = new Set()
    const seen = new Set()
    for (const match of text.matchAll(WORD_RE)) {
      const word = match[0]
      const key = word.toLowerCase()
      if (key.length < 2 || seen.has(key)) continue
      seen.add(key)
      // A capitalized word may be a proper noun — only flag it when the
      // lowercase form fails too, mirroring how the gem treats names.
      if (!active.dictionary.correct(word) && !active.dictionary.correct(key)) {
        misspelled.add(word)
      }
    }
    return { words: [...misspelled], ms: Math.round(performance.now() - t0) }
  }

  // - semantic layer --------------------------------------------------
  // Opt-in per language: the embedder sends semantic-enable, the engine
  // resolves the language mini tier through the pinned registry and
  // loads it into wasm memory (~3 MB model + ~0.25 MB vocab). One model
  // is alive at a time; a language switch frees it with .free().
  //
  // The gem context-boost weight (Kotoshu::Analyzers::SemanticAnalyzer
  // #context_boost: similarity * 0.02 per surrounding word; wasm twin:
  // kotoshu::rerank::CONTEXT_BOOST_WEIGHT).
  const CONTEXT_BOOST_WEIGHT = 0.02
  let semanticState = 'off'
  let model = null
  let lid = null
  let lidCached = false
  let registryPromise = null

  function postSemantic(lang, detail) {
    post('semantic-status', {
      lang,
      state: semanticState,
      detail,
      modelBytes: model?.sizeBytes ?? 0,
      bucketsBytes: model?.bucketsBytes ?? 0,
      modelCached,
    })
  }

  function dropModel() {
    if (!model) return
    // Deterministic release, ahead of GC — the mini tier holds ~3 MB of
    // wasm linear memory for as long as the handle lives.
    model.handle.free()
    model = null
  }

  async function ensureRegistry() {
    registryPromise ??= cachedFetchProgress(registryUrl, 'semantic', 'registry').then(
      ({ res }) => {
        if (!res.ok) throw new Error(`registry download failed: HTTP ${res.status}`)
        return res.json()
      },
    )
    // A failed attempt must not poison the cache — clear for the next opt-in.
    registryPromise.catch(() => {
      registryPromise = null
    })
    return registryPromise
  }

  async function enableSemantic(lang) {
    if (model && model.lang === lang && semanticState === 'ready') {
      postSemantic(lang)
      return
    }
    // A pack load (plan 113) already made the tier resident — semantic
    // opt-in is free, nothing to fetch.
    if (model && model.lang === lang && model.fromPack) {
      semanticState = 'ready'
      postSemantic(lang)
      return
    }
    dropModel()
    semanticState = 'loading'
    postSemantic(lang)
    try {
      await ensureEngine()
      const { loadModel, rerank: rerankFn } = glue
      if (typeof loadModel !== 'function' || typeof rerankFn !== 'function') {
        throw new Error(`engine ${wasmVersion} exposes no model API`)
      }
      const registry = await ensureRegistry()
      const entry = registry.resources?.[`kotoshu://models/${lang}/mini`]
      const mirror = entry?.urls?.mirror
      if (!mirror) {
        throw new Error(`no mini tier for ${lang} in registry ${options.registryTag ?? DEFAULT_REGISTRY_TAG}`)
      }
      // The registry vocab_url points at a release asset, which sends no
      // CORS headers — the mirror sibling is the browser-usable vocab.
      const [modelPair, vocabPair] = await Promise.all([
        cachedFetchProgress(mirror, 'semantic', `${lang} model`),
        cachedFetchProgress(mirror.replace(/\.onnx$/, '.vocab.json'), 'semantic', `${lang} vocab`),
      ])
      modelCached = modelPair.cached && vocabPair.cached
      if (!modelPair.res.ok) throw new Error(`model download failed: HTTP ${modelPair.res.status}`)
      if (!vocabPair.res.ok) throw new Error(`vocab download failed: HTTP ${vocabPair.res.status}`)
      const [modelBytes, vocabBytes] = await Promise.all([
        modelPair.res.arrayBuffer().then((buf) => new Uint8Array(buf)),
        vocabPair.res.arrayBuffer().then((buf) => new Uint8Array(buf)),
      ])
      // Bucket sibling (plan 103): when the registry carries
      // kotoshu://models/{lang}/buckets, its rows let semanticSuggest
      // embed OOV n-grams the vocab lacks. Absent or failed fetch
      // degrades to the two-arg call — the layer works exactly as before.
      let bucketsBytes
      let bucketsBytesCount = 0
      const bucketMirror = registry.resources?.[`kotoshu://models/${lang}/buckets`]?.urls?.mirror
      if (bucketMirror) {
        const bucketPair = await cachedFetchProgress(bucketMirror, 'semantic', `${lang} buckets`)
        if (bucketPair.res.ok) {
          bucketsBytes = new Uint8Array(await bucketPair.res.arrayBuffer())
          bucketsBytesCount = bucketsBytes.byteLength
          modelCached = modelCached && bucketPair.cached
        }
      }
      model = {
        handle: loadModel(modelBytes, vocabBytes, bucketsBytes),
        lang,
        sizeBytes: modelBytes.byteLength + vocabBytes.byteLength,
        bucketsBytes: bucketsBytesCount,
      }
      semanticState = 'ready'
      postSemantic(lang)
    } catch (error) {
      dropModel()
      semanticState = 'unavailable'
      postSemantic(lang, error?.message)
    }
  }

  function disableSemantic() {
    dropModel()
    semanticState = 'off'
    postSemantic(activeLang)
  }

  // - language detection (plan 102 surface) ---------------------------
  async function ensureLid() {
    if (lid) return
    await ensureEngine()
    const { loadLid } = glue
    if (typeof loadLid !== 'function') {
      throw new Error(`engine ${wasmVersion} exposes no LID API`)
    }
    const entry = (await ensureRegistry()).resources?.['kotoshu://models/lid/lid-176']
    const mirror = entry?.urls?.mirror
    if (!mirror) {
      throw new Error(`no lid-176 in registry ${options.registryTag ?? DEFAULT_REGISTRY_TAG}`)
    }
    // Prefer the media-host sibling of the model mirror. The registry
    // vocab_url points at a release asset that may 404 or lack CORS.
    const vocabUrl = mirror.replace(/\.onnx$/, '.vocab.json')
    const [modelPair, vocabPair] = await Promise.all([
      cachedFetchProgress(mirror, 'detect', 'lid model'),
      cachedFetchProgress(vocabUrl, 'detect', 'lid vocab'),
    ])
    lidCached = modelPair.cached && vocabPair.cached
    if (!modelPair.res.ok) throw new Error(`lid model download failed: HTTP ${modelPair.res.status}`)
    if (!vocabPair.res.ok) throw new Error(`lid vocab download failed: HTTP ${vocabPair.res.status}`)
    const [modelBytes, vocabBytes] = await Promise.all([
      modelPair.res.arrayBuffer().then((buf) => new Uint8Array(buf)),
      vocabPair.res.arrayBuffer().then((buf) => new Uint8Array(buf)),
    ])
    lid = {
      handle: loadLid(modelBytes, vocabBytes),
      sizeBytes: modelBytes.byteLength + vocabBytes.byteLength,
    }
  }

  async function detectLanguage(text) {
    try {
      await ensureLid()
      const detect = glue?.detectLanguage
      if (!detect || !lid) throw new Error('LID not ready')
      const t0 = performance.now()
      const result = detect(lid.handle, text)
      post('detected', {
        code: result.code,
        score: result.score,
        ms: Math.round(performance.now() - t0),
        modelBytes: lid.sizeBytes,
        modelCached: lidCached,
      })
    } catch (error) {
      post('detect-error', { message: error?.message })
    }
  }

  // - suggest pipeline -------------------------------------------------
  function suggest(word, context) {
    const engine = active
    const t0 = performance.now()
    const key = `${activeLang}:${word.toLowerCase()}`
    let suggestions = suggestCache.get(key)
    if (!suggestions) {
      suggestions = engine.dictionary.suggest(word, 5)
      if (suggestCache.size > 300) suggestCache.clear()
      suggestCache.set(key, suggestions)
    }
    const sweepMs = Math.round(performance.now() - t0)
    // The cache holds DICTIONARY rows only; the semantic merge runs after
    // retrieval so it never poisons the cache and switching the layer off
    // returns to dictionary-only on the next request. Since wasm 0.3.0 the
    // engine sweep itself enumerates transpositions and substitutions with
    // frequency-aware ranking, so no client-side candidate patching remains.
    const t2 = performance.now()
    const withSemantic = generateSemantic(word, suggestions)
    const semanticMs = Math.round(performance.now() - t2)
    const t3 = performance.now()
    const suggestions2 = rerankSuggestions(withSemantic, context)
    const rerankMs = Math.round(performance.now() - t3)
    return { suggestions: suggestions2, sweepMs, semanticMs, rerankMs }
  }

  // Candidate generation: when the semantic layer is resident and the
  // engine exports semanticSuggest, the model nearest vocabulary words
  // join the dictionary candidates — the intended word for an OOV typo
  // often is one of them, something no edit-distance sweep can produce.
  // mergeSemanticCandidates dedupes and labels; rerankSuggestions above
  // then orders the merged list by context fit as before.
  function generateSemantic(word, dictionary) {
    const semanticSuggest = glue?.semanticSuggest
    if (!model || semanticState !== 'ready' || !semanticSuggest) {
      return dictionary
    }
    try {
      return mergeSemanticCandidates(
        dictionary,
        semanticSuggest(model.handle, word, SEMANTIC_SUGGEST_K),
      )
    } catch {
      // A generation failure must never take the popover down with it —
      // dictionary candidates are still complete and correct.
      return dictionary
    }
  }

  // The gem cascade default never skips (threshold 1.0 = always rerank),
  // so with a model loaded every candidate row is adjusted — mirroring
  // Kotoshu::Analyzers::SemanticAnalyzer#rank_by_context, which boosts
  // each candidate by 0.02 x sum of cosines over a +/-3-word window and
  // clamps at 1.0 before sorting. rerank() returns the MEAN cosine over
  // the in-vocab context tokens, so n x mean reconstructs the gem sum;
  // n is recovered with a self-cosine probe (a token scored against
  // itself returns ~1.0 in-vocab, exactly 0.0 out-of-vocab). The probe is
  // candidate-independent and runs once per popover. Divergence from the
  // gem, line by line: the gem calls model.similarity(word, token) per
  // token; here the wasm boundary batches them into one mean — the same
  // arithmetic with a single boundary crossing.
  function rerankSuggestions(suggestions, context) {
    const rerank = glue?.rerank
    if (!model || semanticState !== 'ready' || !rerank || suggestions.length === 0) {
      return suggestions
    }
    const handle = model.handle
    const tokens = context.match(WORD_RE)?.map((token) => token.toLowerCase()) ?? []
    if (tokens.length === 0) return suggestions
    const inVocab = tokens.filter((token) => rerank(handle, token, token) > 0.5).length
    if (inVocab === 0) return suggestions
    return suggestions
      .map((row, index) => {
        const score = rerank(handle, row.word, context)
        const boosted = Math.min(row.confidence + CONTEXT_BOOST_WEIGHT * inVocab * score, 1)
        return { row: { ...row, confidence: boosted }, index }
      })
      .sort((a, b) => b.row.confidence - a.row.confidence || a.index - b.index)
      .map((entry) => entry.row)
  }

  // - suggest queue ————————————————————————————
  // The pane sends ONE batch message for all its flagged words; the
  // popover sends priority singles. Words are swept one at a time here
  // (the engine is single-threaded) but with a yield between words, so a
  // click mid-batch jumps ahead of the remaining pane words. A fresh
  // batch drops the pending pane words it supersedes — the worker cache
  // makes any repeated word instant, so nothing is recomputed twice.
  let suggestQueue = []
  let draining = false

  function enqueueSuggest(tasks, replaceBatch) {
    if (replaceBatch) suggestQueue = suggestQueue.filter((task) => task.priority)
    suggestQueue.push(...tasks)
    void drainSuggest()
  }

  async function drainSuggest() {
    if (draining) return
    draining = true
    try {
      while (suggestQueue.length > 0 && active) {
        let next = suggestQueue.findIndex((task) => task.priority)
        if (next < 0) next = 0
        const task = suggestQueue.splice(next, 1)[0]
        const result = suggest(task.word, task.context)
        post('suggested', {
          word: task.word,
          suggestions: result.suggestions,
          sweepMs: result.sweepMs,
          semanticMs: result.semanticMs,
          rerankMs: result.rerankMs,
          semantic: semanticState === 'ready',
        })
        // Let newly arrived messages (a click, a fresh batch) reorder the
        // queue before the next word.
        await new Promise((resolve) => setTimeout(resolve, 0))
      }
    } finally {
      draining = false
    }
  }

  return {
    /** Handle one protocol message (the worker entry wires this to onmessage). */
    async handle(data) {
      if (data.type === 'load') {
        // Words queued for the previous language are stale — the pane
        // re-requests after loaded anyway.
        suggestQueue = []
        // One model at a time: switching language frees the previous tier
        // ~3 MB of wasm memory before the new dictionary even loads. The
        // embedder re-requests the new language tier on loaded.
        if (model && model.lang !== data.lang) {
          dropModel()
          semanticState = 'off'
          postSemantic(data.lang ?? null)
        }
        try {
          await ensureEngine()
          const loaded = await loadLanguage(data.lang)
          post('loaded', {
            engineCached,
            dictCached,
            lang: data.lang,
            engineBytes,
            dictionaryBytes: loaded.sizeBytes,
            engineVersion: glue.KotoshuWasm.VERSION ?? wasmVersion,
            wasmVersion,
            loadMs: Math.round(loaded.loadMs),
            packed: loaded.packed === true,
          })
        } catch (error) {
          post('load-error', { lang: data.lang, message: error?.message })
        }
      } else if (data.type === 'check') {
        const { words, ms } = check(data.text ?? '')
        post('checked', { words, ms })
      } else if (data.type === 'suggest') {
        if (!active) return
        enqueueSuggest([{ word: data.word, context: data.context ?? '', priority: true }], false)
      } else if (data.type === 'suggest-batch') {
        if (!active) return
        const items = (data.items ?? []).map((item) => ({ ...item, priority: false }))
        enqueueSuggest(items, true)
      } else if (data.type === 'semantic') {
        if (data.enable) await enableSemantic(data.lang ?? activeLang ?? '')
        else disableSemantic()
      } else if (data.type === 'detect') {
        await detectLanguage(data.text ?? '')
      }
    },
  }
}
