#!/usr/bin/env node
// Node smoke test for @kotoshu/worker (plan 112): drives the REAL
// package payload (kotoshu-worker/pkg, assembled by
// scripts/worker_build.sh) through the REAL protocol (createEngine —
// the same code the worker entry wires to onmessage) against the REAL
// engine (the wasm-pack pkg built by scripts/wasm_build.sh) and the
// REAL pinned sources (dictionaries + models registry from the CDN
// pins the worker ships with). No mocks: Node has no Worker and no
// Cache Storage, which is itself a code path under test — the engine
// degrades to uncached fetch instead of dying.
//
// Usage:
//   scripts/wasm_build.sh && scripts/worker_build.sh && \
//   node scripts/worker_node_smoke.mjs
// (KOTOSHU_WORKER_PKG / KOTOSHU_WASM_PKG override the pkg dirs.)

import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const workerPkg = process.env.KOTOSHU_WORKER_PKG ?? path.join(root, "kotoshu-worker", "pkg");
const wasmPkg = process.env.KOTOSHU_WASM_PKG ?? path.join(root, "kotoshu-wasm", "pkg");

let failures = 0;
let assertions = 0;

function assert(label, condition) {
  assertions += 1;
  console.log(`${condition ? "PASS" : "FAIL"} ${label}`);
  if (!condition) failures += 1;
}

const { createEngine, SEMANTIC_SUGGEST_K, mergeSemanticCandidates } = await import(
  pathToFileURL(path.join(workerPkg, "index.js")).href
);

// --- pure half first (no network) --------------------------------------
const merged = mergeSemanticCandidates(
  [{ word: "hello", distance: 1, confidence: 1.0, source: "edit_distance" }],
  [
    { word: "Hello", score: 0.9 },
    { word: "cat", score: 1.7 },
  ],
);
assert(
  "mergeSemanticCandidates dedupes case-insensitively and clamps",
  merged.length === 2 &&
    merged[0].word === "hello" &&
    merged[1].word === "cat" &&
    merged[1].source === "semantic" &&
    merged[1].distance === 0 &&
    merged[1].confidence === 1,
);
assert("SEMANTIC_SUGGEST_K is 4", SEMANTIC_SUGGEST_K === 4);
assert(
  "null neighbors return the dictionary verbatim",
  mergeSemanticCandidates(merged, null) === merged,
);

// --- the engine over the local pkg + the pinned CDN sources -------------
const events = [];
const engine = createEngine(
  (event) => {
    events.push(event);
  },
  {
    // Node cannot https-import a URL: point the engine at the locally
    // built @kotoshu/wasm pkg (the code under test, not the published
    // artifact). Dictionaries and models still resolve through the
    // default pins — the smoke doubles as a liveness check on them.
    engineSource: {
      glue: pathToFileURL(path.join(wasmPkg, "kotoshu_wasm_bg.js")).href,
      wasm: pathToFileURL(path.join(wasmPkg, "kotoshu_wasm_bg.wasm")).href,
    },
    readFile: (url) => readFile(fileURLToPath(url)),
  },
);

async function waitFor(type, predicate = () => true, timeoutMs = 120_000) {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const index = events.findIndex((event) => event.type === type && predicate(event));
    if (index >= 0) return events.splice(index, 1)[0];
    // While waiting for a loaded event, a load failure for the language
    // being loaded is a hard stop — report it instead of timing out on
    // the missing success event. (Other waits leave load-error events
    // alone: the zz probe below posts one deliberately.)
    if (type === "loaded") {
      const failure = events.find((event) => event.type === "load-error");
      if (failure) throw new Error(`load-error for ${failure.lang}: ${failure.message}`);
    }
    if (Date.now() > deadline) throw new Error(`timeout waiting for ${type}`);
    await new Promise((resolve) => setTimeout(resolve, 20));
  }
}

// load: engine (local wasm, bytes > 0) + en dictionary (CDN, dual layout)
await engine.handle({ type: "load", lang: "en" });
const loaded = await waitFor("loaded");
assert("loaded carries the language", loaded.lang === "en");
assert("loaded carries engine bytes", loaded.engineBytes > 400_000);
assert("loaded carries dictionary bytes", loaded.dictionaryBytes > 100_000);
assert("loaded carries engine + wasm versions", /^\d+\.\d+\.\d+$/.test(loaded.engineVersion) && /^\d+\.\d+\.\d+$/.test(loaded.wasmVersion));
const progress = await waitFor("load-progress", (event) => event.phase === "dictionary");
assert("load-progress flows for the dictionary phase", progress.loaded > 0);

// check: OOV words flagged, dictionary words not, proper-noun rule applied
await engine.handle({ type: "check", text: "I recieved teh seperate leters" });
const checked = await waitFor("checked");
assert(
  "checked flags the misspellings",
  ["recieved", "teh", "seperate", "leters"].every((word) => checked.words.includes(word)),
);
assert("checked reports a duration", typeof checked.ms === "number" && checked.ms >= 0);

// suggest (dictionary-only): the priority single path
await engine.handle({ type: "suggest", word: "teh", context: "" });
const plain = await waitFor("suggested", (event) => event.word === "teh");
assert("suggest returns rows", plain.suggestions.length > 0);
assert(
  "every row carries exactly the four SUGGESTION_KEYS",
  plain.suggestions.every((row) => {
    const keys = Object.keys(row).sort();
    return keys.length === 4 && keys.join() === "confidence,distance,source,word";
  }),
);
assert("dictionary-only suggest reports semantic false", plain.semantic === false);

// semantic enable: mini tier + buckets sibling through the registry mirror
await engine.handle({ type: "semantic", enable: true, lang: "en" });
const ready = await waitFor("semantic-status", (event) => event.state === "ready");
assert("semantic-status reaches ready", ready.state === "ready" && ready.lang === "en");
assert("the mini tier is resident (~3 MB + vocab)", ready.modelBytes > 3_000_000);
assert("the buckets sibling is resident", ready.bucketsBytes > 1_000_000);

// suggest with the layer on: generated neighbors join the dictionary rows
await engine.handle({ type: "suggest", word: "teh", context: "read teh letter" });
const semantic = await waitFor(
  "suggested",
  (event) => event.word === "teh" && event.semantic === true,
);
assert("semantic suggest reports semantic true", semantic.semantic === true);
assert(
  "generated neighbors join the dictionary rows (source: semantic)",
  semantic.suggestions.some((row) => row.source === "semantic"),
);

// suggest-batch: one batch message, one event per word, in order. Each
// word carries a real context so the rerank half runs — the merged list
// is then confidence-sorted end to end (with an empty context the merge
// appends semantic neighbors after the dictionary rows, unranked).
await engine.handle({
  type: "suggest-batch",
  items: [
    { word: "recieved", context: "i recieved the letter yesterday" },
    { word: "definately", context: "that is definately the answer" },
    { word: "accomodate", context: "they accomodate every request" },
  ],
});
const batch = [];
for (const word of ["recieved", "definately", "accomodate"]) {
  batch.push(await waitFor("suggested", (event) => event.word === word));
}
assert("the batch drains one event per word", batch.length === 3);
assert(
  "batch rows are confidence-sorted",
  batch.every(
    (event) =>
      event.suggestions.every(
        (row, index) => index === 0 || event.suggestions[index - 1].confidence >= row.confidence,
      ),
  ),
);

// detect: the LID model through the registry, on a Portuguese sentence
await engine.handle({
  type: "detect",
  text: "Esta é uma frase de teste para verificar a deteção de idioma",
});
const detected = await waitFor("detected");
assert("detectLanguage identifies pt", detected.code === "pt");
assert(
  "detected carries score, ms, and model bytes",
  detected.score >= 0 &&
    detected.score <= 1 &&
    detected.ms >= 0 &&
    detected.modelBytes > 1_000_000,
);

// language switch: the en tier is dropped (semantic-status off) and es loads
await engine.handle({ type: "load", lang: "es" });
const switched = await waitFor("loaded", (event) => event.lang === "es");
const off = await waitFor("semantic-status", (event) => event.state === "off");
assert("switching language loads es", switched.lang === "es");
assert("switching language frees the previous tier", off.state === "off");

// failure path: an unknown language posts load-error, nothing throws
await engine.handle({ type: "load", lang: "zz" });
const error = await waitFor("load-error");
assert(
  "load-error carries the language and message",
  error.lang === "zz" && typeof error.message === "string" && error.message.length > 0,
);
await engine.handle({ type: "check", text: "still alive" });
await waitFor("checked");
assert("the engine survives a failed load", true);

console.log(`${assertions} assertions, ${failures} failures`);
process.exit(failures === 0 ? 0 : 1);
