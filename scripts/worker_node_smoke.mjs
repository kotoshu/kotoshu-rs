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

// --- language pack mode (plan 113) ------------------------------------
// A LOCAL fixture pack (the conformance en dictionary + the checked-in
// model fixtures, framed exactly as scripts/build_packs.py in the
// models repo frames them) served through a LOCAL fixture registry: no
// network on the pack path at all — one fetch for the whole language,
// the tier model resident straight out of the pack, semantic-enable
// free. Then the same engine shape against a registry WITHOUT the pack
// entry: pack mode must degrade to the per-resource path silently.
{
  const { createHash } = await import("node:crypto");
  const { mkdtemp, writeFile } = await import("node:fs/promises");
  const os = await import("node:os");

  const sha = (buf) => createHash("sha256").update(buf).digest("hex");
  const frameSection = (tag, payload) => {
    const framed = Buffer.alloc(5 + payload.length + 32);
    framed.writeUInt32LE(payload.length, 0);
    framed[4] = tag;
    payload.copy(framed, 5);
    framed.set(createHash("sha256").update(payload).digest(), 5 + payload.length);
    return framed;
  };

  const modelsDir = path.join(root, "kotoshu", "tests", "fixtures", "models");
  const dicBase = path.join(
    root, "tests", "fixtures", "spec/fixtures/dictionaries/hunspell/test",
  );
  const aff = await readFile(`${dicBase}.aff`);
  const dic = await readFile(`${dicBase}.dic`);
  const model = await readFile(path.join(modelsDir, "en-mini-truncated.onnx"));
  const vocab = await readFile(path.join(modelsDir, "en-mini-truncated.vocab.json"));
  const buckets = await readFile(path.join(modelsDir, "en-buckets-truncated.onnx"));

  const sections = [
    [1, aff], [2, dic], [3, model], [4, vocab], [5, buckets],
  ];
  const countBuf = Buffer.alloc(4);
  countBuf.writeUInt32LE(sections.length, 0);
  const framed = sections.map(([tag, payload]) => frameSection(tag, payload));
  const pack = Buffer.concat([Buffer.from("KPK1"), countBuf, ...framed]);

  // Walk the framing to record the offsets the registry entry declares.
  const contents = {};
  let cursor = 8;
  for (const [index, [tag, payload]] of sections.entries()) {
    contents[["aff", "dic", "model", "vocab", "buckets"][index]] = {
      tag,
      offset: cursor + 5,
      length: payload.length,
      sha256: sha(payload),
    };
    cursor += 5 + payload.length + 32;
  }

  const tmp = await mkdtemp(path.join(os.tmpdir(), "kotoshu-pack-"));
  const packPath = path.join(tmp, "en-0.0.0-fixture.bin");
  await writeFile(packPath, pack);
  const registry = {
    spec: "kotoshu.resources/v1",
    registry_version: 1,
    release_tag: null,
    resources: {
      "kotoshu://packs/en": {
        type: "pack",
        language: "en",
        version: "0.0.0-fixture",
        tier: "mini",
        dictionary_pin: "0".repeat(40),
        urls: { primary: null, mirror: pathToFileURL(packPath).href },
        contents,
        sha256: sha(pack),
        size_bytes: pack.length,
        licenses: { embeddings: "CC-BY-SA-3.0", dictionary: "fixture" },
        min_engine_version: "0.5",
        eval_ref: null,
      },
    },
  };
  const registryPath = path.join(tmp, "registry.json");
  await writeFile(registryPath, JSON.stringify(registry));
  const bareRegistryPath = path.join(tmp, "registry-bare.json");
  await writeFile(bareRegistryPath, JSON.stringify({ ...registry, resources: {} }));

  // The pack engines run in CHILD processes, one engine each: a
  // wasm-bindgen glue module carries ONE wasm binding per JS realm, so
  // a second engine in this process would rebind it and a GC finalizer
  // of the first engine's handles would free through the wrong
  // instance (memory access out of bounds). One engine per realm is
  // the worker's own deployment shape — the children mirror it.
  const childPath = path.join(tmp, "pack-child.mjs");
  await writeFile(
    childPath,
    `
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const { createEngine } = await import(process.env.WORKER_ENTRY);
const events = [];
const engine = createEngine((event) => events.push(event), {
  pack: true,
  registryUrl: process.env.REGISTRY_URL,
  engineSource: {
    glue: process.env.GLUE_URL,
    wasm: process.env.WASM_URL,
  },
  readFile: (url) => readFile(fileURLToPath(url)),
});
const waitFor = async (type, predicate = () => true) => {
  const deadline = Date.now() + 120_000;
  for (;;) {
    const index = events.findIndex((event) => event.type === type && predicate(event));
    if (index >= 0) return events.splice(index, 1)[0];
    if (type === "loaded") {
      const failure = events.find((event) => event.type === "load-error");
      if (failure) throw new Error("load-error: " + failure.message);
    }
    if (Date.now() > deadline) throw new Error("timeout waiting for " + type);
    await new Promise((resolve) => setTimeout(resolve, 20));
  }
};
const pass = (label) => console.log("PASS " + label);
const fail = (label, detail) => {
  console.log("FAIL " + label + (detail === undefined ? "" : " (" + detail + ")"));
  process.exitCode = 1;
};

await engine.handle({ type: "load", lang: "en" });
const loaded = await waitFor("loaded");
if (process.env.PACK_MODE === "pack") {
  loaded.packed === true
    ? pass("packed load reports packed: true")
    : fail("packed load reports packed: true", loaded.packed);
  loaded.dictionaryBytes === Number(process.env.PACK_SIZE)
    ? pass("the pack IS the payload (one artifact, byte-counted)")
    : fail("the pack IS the payload", loaded.dictionaryBytes);
  const progress = await waitFor("load-progress", (event) => event.phase === "pack");
  progress.total === Number(process.env.PACK_SIZE)
    ? pass("pack progress flows as one stream")
    : fail("pack progress total", progress.total);
  await engine.handle({ type: "check", text: "I recieved teh leters" });
  const checked = await waitFor("checked");
  ["recieved", "teh", "leters"].every((word) => checked.words.includes(word))
    ? pass("the pack dictionary checks identically")
    : fail("the pack dictionary checks identically", checked.words);
  await engine.handle({ type: "semantic", enable: true, lang: "en" });
  const ready = await waitFor("semantic-status", (event) => event.state === "ready");
  ready.modelBytes === Number(process.env.MODEL_BYTES) &&
  ready.bucketsBytes === Number(process.env.BUCKETS_BYTES)
    ? pass("the pack model is resident with the registry-declared sizes")
    : fail("pack model resident sizes", ready.modelBytes + "/" + ready.bucketsBytes);
  await engine.handle({ type: "suggest", word: "teh", context: "read teh letter" });
  const suggested = await waitFor("suggested");
  suggested.semantic === true &&
  suggested.suggestions.some((row) => row.source === "semantic")
    ? pass("pack suggest merges semantic neighbors")
    : fail("pack suggest merges semantic neighbors", suggested.semantic);
} else {
  loaded.packed === false
    ? pass("pack mode without a pack entry degrades (packed: false)")
    : fail("pack mode without a pack entry degrades", loaded.packed);
  loaded.dictionaryBytes > 100_000
    ? pass("the degraded load still produced a working dictionary")
    : fail("degraded dictionary bytes", loaded.dictionaryBytes);
}
`,
  );

  const { spawn } = await import("node:child_process");
  const runChild = (mode) =>
    new Promise((resolve) => {
      const child = spawn(process.execPath, [childPath], {
        env: {
          ...process.env,
          PACK_MODE: mode,
          PACK_SIZE: String(pack.length),
          MODEL_BYTES: String(model.length + vocab.length),
          BUCKETS_BYTES: String(buckets.length),
          REGISTRY_URL: pathToFileURL(mode === "pack" ? registryPath : bareRegistryPath).href,
          WORKER_ENTRY: pathToFileURL(path.join(workerPkg, "index.js")).href,
          GLUE_URL: pathToFileURL(path.join(wasmPkg, "kotoshu_wasm_bg.js")).href,
          WASM_URL: pathToFileURL(path.join(wasmPkg, "kotoshu_wasm_bg.wasm")).href,
        },
        stdio: ["ignore", "inherit", "inherit"],
      });
      child.on("exit", (code) => resolve(code));
    });

  const packCode = await runChild("pack");
  assert("pack mode end to end (child process)", packCode === 0);
  const fallbackCode = await runChild("fallback");
  assert("pack fallback end to end (child process)", fallbackCode === 0);
}

console.log(`${assertions} assertions, ${failures} failures`);
process.exit(failures === 0 ? 0 : 1);
