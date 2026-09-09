#!/usr/bin/env node
// Wasm memory-ceiling test (plan 112): the playground residency pattern —
// dictionary + mini tier + buckets sibling loaded and ALIVE together in
// one wasm32 instance — must stay under a stated budget per language
// class, so the worker never quietly grows past what a browser tab can
// spare. Loading uses the exact site-worker dance (manual instantiate of
// the bundler glue, holding instance.exports.memory), the same sources
// the worker resolves (pinned dictionary CDN, pinned models registry
// mirror), and asserts residency after a real semanticSuggest proves the
// tier is usable, not merely allocated.
//
// Each class runs in its own process (wasm memory is monotonic within an
// instance; two classes in one process would measure each other).
//
// Usage: node scripts/wasm_memory_ceiling.mjs [en|pt]   (default en)
// Env:   KOTOSHU_WASM_PKG  wasm pkg dir (default kotoshu-wasm/pkg)
//        KOTOSHU_MEMORY_BUDGET_MB  budget override (defaults below)
//
// Budgets (MB, wasm32 linear memory, measured + ~1.3x headroom for
// engine growth and tier drift within a registry tag):
//   small class (en, 542 KB dict source):   64 MB   (measured 46.9)
//   large class (pt, 5.2 MB dict source):  192 MB  (measured 165.3)
// Measured on the dev machine (2026-09-09, wasm 0.4.0, registry v1.5.0):
//   en: engine 1.1 MB -> +dict 19.8 MB -> +tier 46.9 MB -> 46.9 MB peak
//   pt: engine 1.1 MB -> +dict 135.5 MB -> +tier 154.8 MB -> 165.3 MB peak
// (the dictionary index dominates: the SweepIndex structures expand the
// pt 5.2 MB source ~26x in linear memory, and a suggest sweep over it
// allocates another ~10 MB of candidates; the bucket ONNX — 9.8 MB on
// disk — loads into the same arena. The en/pt pair brackets both
// classes: every other language at the pin sits between them.)

import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const MB = 1024 * 1024;
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const pkgDir = process.env.KOTOSHU_WASM_PKG ?? path.join(root, "kotoshu-wasm", "pkg");
const lang = process.argv[2] ?? "en";

const CLASSES = {
  // small dictionaries: en, es, de, it (dict source < 1.5 MB)
  en: { budget: Number(process.env.KOTOSHU_MEMORY_BUDGET_MB ?? 64) },
  // large dictionaries: pt, nb, ru, fr (dict source up to 5.2 MB)
  pt: { budget: Number(process.env.KOTOSHU_MEMORY_BUDGET_MB ?? 192) },
};
const config = CLASSES[lang];
if (!config) {
  console.error(`unknown language class "${lang}" (known: ${Object.keys(CLASSES).join(", ")})`);
  process.exit(2);
}

// Frozen residency probes — an OOV typo whose intended word the tier
// returns top-1 through the bucket rows (the same gate the wasm smoke
// pins on its fixtures). "csaa"-style arbitrary probes can honestly
// resolve to nothing; these were verified against the real artifacts.
const PROBES = {
  en: { oov: "teh", expected: "the" },
  pt: { oov: "bibliotéca", expected: "biblioteca" },
};
const { oov, expected } = PROBES[lang];

const DICT_PIN = "1829a3e2e67dc7ffb38f8dcd2d3d2294b6a8580d";
const REGISTRY_TAG = "v1.5.0";

// The site-worker instantiation: import the bundler glue, fetch the raw
// bytes, instantiate with both import-object spellings, hand the exports
// to the glue, run the start function.
async function instantiateEngine() {
  const glue = await import(pathToFileURL(path.join(pkgDir, "kotoshu_wasm_bg.js")).href);
  const bytes = new Uint8Array(await readFile(path.join(pkgDir, "kotoshu_wasm_bg.wasm")));
  const { instance } = await WebAssembly.instantiate(bytes, {
    "./kotoshu_wasm_bg.js": glue,
    wbg: glue,
  });
  glue.__wbg_set_wasm(instance.exports);
  const start = instance.exports.__wbindgen_start;
  if (typeof start === "function") start();
  return { glue, memory: instance.exports.memory };
}

async function fetchDictFile(ext) {
  const paths = [`${lang}/spelling/index.${ext}`, `${lang}/index.${ext}`];
  for (const p of paths) {
    const res = await fetch(`https://cdn.jsdelivr.net/gh/kotoshu/dictionaries@${DICT_PIN}/${p}`);
    if (res.ok) return res.text();
  }
  throw new Error(`no index.${ext} for ${lang} at the dictionary pin`);
}

async function fetchBytes(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`download failed: HTTP ${res.status} for ${url}`);
  return new Uint8Array(await res.arrayBuffer());
}

const report = (label) => console.log(`${(memory.buffer.byteLength / MB).toFixed(1).padStart(6)} MB  ${label}`);

const { glue, memory } = await instantiateEngine();
report(`engine instantiated (${lang})`);

const [aff, dic] = await Promise.all([fetchDictFile("aff"), fetchDictFile("dic")]);
const dictionary = new glue.KotoshuWasm(aff, dic);
report(`+ dictionary (${lang}, ${Math.round((aff.length + dic.length) / 1024)} KB source)`);

const registryRes = await fetch(
  `https://raw.githubusercontent.com/kotoshu/models-fasttext-onnx/${REGISTRY_TAG}/registry.json`,
);
if (!registryRes.ok) throw new Error(`registry download failed: HTTP ${registryRes.status}`);
const registry = await registryRes.json();
const resources = registry.resources ?? {};
const mini = resources[`kotoshu://models/${lang}/mini`]?.urls?.mirror;
const buckets = resources[`kotoshu://models/${lang}/buckets`]?.urls?.mirror;
if (!mini) throw new Error(`no mini tier for ${lang} in registry ${REGISTRY_TAG}`);
if (!buckets) throw new Error(`no buckets sibling for ${lang} in registry ${REGISTRY_TAG}`);

const [modelBytes, vocabBytes, bucketsBytes] = await Promise.all([
  fetchBytes(mini),
  fetchBytes(mini.replace(/\.onnx$/, ".vocab.json")),
  fetchBytes(buckets),
]);
const model = glue.loadModel(modelBytes, vocabBytes, bucketsBytes);
report(`+ mini tier (${(modelBytes.byteLength / MB).toFixed(1)} MB model + ${(vocabBytes.byteLength / MB).toFixed(1)} MB vocab + ${(bucketsBytes.byteLength / MB).toFixed(1)} MB buckets)`);

// Residency must be usable, not merely allocated: the frozen OOV probe
// embeds through the bucket rows and returns its intended word top-1.
const neighbors = glue.semanticSuggest(model, oov, 4);
if (!Array.isArray(neighbors) || neighbors.length === 0) {
  console.error(`FAIL semanticSuggest returned nothing for ${oov} with the tier resident`);
  process.exit(1);
}
if (neighbors[0].word !== expected) {
  console.error(
    `FAIL semanticSuggest("${oov}")[0] is "${neighbors[0].word}", expected "${expected}"`,
  );
  process.exit(1);
}
report(`  semanticSuggest resident (${oov} -> ${neighbors[0].word} at ${neighbors[0].score.toFixed(3)})`);

const check = dictionary.correct(lang === "en" ? "hello" : "casa");
const suggestRows = dictionary.suggest(lang === "en" ? "hlelo" : oov, 5);
model.free();
report(
  `peak (after correct=${check} + suggest=${suggestRows.length} rows, tier freed)`,
);

const peak = memory.buffer.byteLength / MB;
const verdict = peak <= config.budget ? "PASS" : "FAIL";
console.log(`${verdict} ${lang}: peak ${peak.toFixed(1)} MB <= budget ${config.budget} MB`);
process.exit(verdict === "PASS" ? 0 : 1);
