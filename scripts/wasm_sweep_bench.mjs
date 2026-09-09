#!/usr/bin/env node
// Multi-language sweep bench over the built wasm (plan 112, the CI
// latency gate): the same harness shape the sweep-index work measured
// with (kotoshu-rs PR #23) — construct the REAL dictionary for each
// language from the pinned dictionary CDN, warm one sweep (the first
// sweep builds the SweepIndex; steady-state per-word latency is what
// the budgets gate), then sweep a FIXED committed wordlist of typos and
// report avg / p50 / p95 / worst per language.
//
// Budgets gate the AVERAGE (spec: en < 120 ms, pt < 700 ms), tuned to
// measured p50s with headroom — see the budget table below for the
// measurements they were tuned against (local dev machine, plus the
// PR #23 live-build numbers). A budget is exceeded -> exit 1 (CI red).
//
// Usage:
//   scripts/wasm_build.sh && node scripts/wasm_sweep_bench.mjs
// Env:
//   KOTOSHU_WASM_PKG     wasm pkg dir (default kotoshu-wasm/pkg)
//   KOTOSHU_BENCH_LANGS  comma list (default: all eight)
//   KOTOSHU_BENCH_EN_MS / KOTOSHU_BENCH_PT_MS  budget overrides

import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const pkgDir = process.env.KOTOSHU_WASM_PKG ?? path.join(root, "kotoshu-wasm", "pkg");

// The same dictionary pin the gem and the worker resolve at runtime.
const DICT_PIN = "1829a3e2e67dc7ffb38f8dcd2d3d2294b6a8580d";
const DICT_BASE = `https://cdn.jsdelivr.net/gh/kotoshu/dictionaries@${DICT_PIN}`;

// Fixed typo wordlists — committed, so every run sweeps the same words
// (deterministic input; only timings vary). Realistic misspellings:
// transpositions, deletions, doublings over common words.
const WORDLISTS = {
  en: [
    "teh", "hlelo", "recieved", "seperate", "definately", "accomodate",
    "occuring", "wich", "becuase", "thier", "adress", "enviroment",
    "goverment", "tommorow", "wierd", "sucessful",
  ],
  es: [
    "eprsona", "copmutadora", "trabjo", "bibloiteca", "conosimiento",
    "empesa", "aveinda", "qeu", "porqeu", "futbolo", "gverno", "ciuded",
  ],
  de: [
    "bibilothek", "entwiklung", "wisseschaft", "arbeitsplaz",
    "veranstaltugn", "geschihte", "freunlich", "schreiebn", "mrogan",
    "hsau", "strase", "zuknft",
  ],
  it: [
    "persoan", "biblioteac", "lavroo", "calcolatrce", "conosecnza",
    "citta", "gioranta", "svoglimento", "amicziia", "otobre",
    "mercoledi", "universita",
  ],
  fr: [
    "bonojur", "oridnateur", "bibliothéque", "dévelopement",
    "conaisance", "entrepise", "aujoudhui", "beacoup", "maiosn",
    "gouvernment", "travial", "languea",
  ],
  ru: [
    "привте", "спаисбо", "работта", "библитека", "госудраство",
    "развите", "компьюетр", "человке", "врмея", "яызк", "здраствуйте",
    "учиба",
  ],
  nb: [
    "biblotek", "hjlep", "kjøken", "univeristet", "utviklig",
    "regjreing", "spårk", "hvodran", "fremitd", "vernde", "samfudn",
    "daligg",
  ],
  pt: [
    // A representative popover mix: mostly 4-8 letter typos of frequent
    // words (transpositions, deletions, doublings) with a few long ones —
    // the shape PR #23 measured (avg 423 ms), not a long-word gauntlet.
    "esat", "cmo", "mias", "miuto", "cosia", "tepmo", "gentee", "garnde",
    "mudno", "fazre", "poedr", "sitema", "cidadae", "pesssoa", "obrigdao",
    "futeboll",
  ],
};

// Budgets (average per steady-state sweep, milliseconds):
//
//   en 120  — measured avg 54.6-75.9 / p50 58-74 / worst 86-134 over
//             repeated runs on the dev machine (the spread is background
//             load, not the engine); PR #23 measured avg 45 on the live
//             playground. 120 keeps ~2x over quiet-machine numbers.
//   pt 700  — the heaviest dictionary at the pin (5.2 MB dic source):
//             measured avg 267.8 / p50 299 / worst 458 on the
//             representative wordlist (PR #23 playground avg was 423);
//             700 keeps ~2.3x for CI-runner cores.
//
// The other six languages are measured and reported ungated (one quiet-
// machine run, wasm 0.4.0): es avg 109, de 233, it 195, fr 632 (the
// French worst case is the known dense-slice floor, PR #23), ru 233,
// nb 375 — all under pt by construction (smaller dictionaries).
const BUDGETS = {
  en: Number(process.env.KOTOSHU_BENCH_EN_MS ?? 120),
  pt: Number(process.env.KOTOSHU_BENCH_PT_MS ?? 700),
};

const langs = (process.env.KOTOSHU_BENCH_LANGS ?? Object.keys(WORDLISTS).join(","))
  .split(",")
  .map((lang) => lang.trim())
  .filter(Boolean);

// Two layouts exist at the pin: the six original full-feature languages
// under {lang}/spelling/, everything else flat at {lang}/index.*.
async function fetchDictFile(lang, ext) {
  const paths = [`${lang}/spelling/index.${ext}`, `${lang}/index.${ext}`];
  for (const p of paths) {
    const res = await fetch(`${DICT_BASE}/${p}`);
    if (res.ok) return res.text();
  }
  throw new Error(`no index.${ext} for ${lang} at the dictionary pin`);
}

function stats(samples) {
  const sorted = [...samples].sort((a, b) => a - b);
  const avg = sorted.reduce((sum, value) => sum + value, 0) / sorted.length;
  const p50 = sorted[Math.floor(sorted.length / 2)];
  const p95 = sorted[Math.ceil(sorted.length * 0.95) - 1];
  return { avg, p50, p95, worst: sorted[sorted.length - 1] };
}

const fmt = (ms) => ms.toFixed(1).padStart(8);
const { KotoshuWasm } = await import(pathToFileURL(path.join(pkgDir, "kotoshu_wasm.js")).href);

let red = false;
const rows = [];
for (const lang of langs) {
  const [aff, dic] = await Promise.all([fetchDictFile(lang, "aff"), fetchDictFile(lang, "dic")]);
  const dictionary = new KotoshuWasm(aff, dic);

  // Warmup: the first suggest builds the SweepIndex (length buckets,
  // packed soundex, indexed lengths). Timed and reported separately —
  // it is a one-off per dictionary lifetime, not per-word latency.
  const warm0 = performance.now();
  dictionary.suggest(WORDLISTS[lang][0], 5);
  const indexMs = performance.now() - warm0;

  const samples = [];
  let rowsTotal = 0;
  for (const word of WORDLISTS[lang]) {
    const t0 = performance.now();
    rowsTotal += dictionary.suggest(word, 5).length;
    samples.push(performance.now() - t0);
  }
  const s = stats(samples);
  const budget = BUDGETS[lang];
  const verdict = budget === undefined ? "info" : s.avg < budget ? "ok" : "OVER";
  if (verdict === "OVER") red = true;
  rows.push({
    lang,
    dictKb: Math.round((aff.length + dic.length) / 1024),
    indexMs,
    ...s,
    budget,
    verdict,
    rowsTotal,
  });
}

const header =
  "lang".padEnd(5) +
  "dict KB".padStart(9) +
  "index ms".padStart(10) +
  "avg ms".padStart(9) +
  "p50".padStart(9) +
  "p95".padStart(9) +
  "worst".padStart(9) +
  "budget".padStart(9) +
  "  verdict";
console.log(header);
for (const row of rows) {
  console.log(
    row.lang.padEnd(5) +
      String(row.dictKb).padStart(9) +
      row.indexMs.toFixed(1).padStart(10) +
      fmt(row.avg) +
      fmt(row.p50) +
      fmt(row.p95) +
      fmt(row.worst) +
      (row.budget === undefined ? "        -" : String(row.budget).padStart(9)) +
      `  ${row.verdict}`,
  );
}
const swept = rows.reduce((sum, row) => sum + row.rowsTotal, 0);
console.log(`swept ${swept} suggestion rows across ${rows.length} languages`);

if (red) {
  console.error("latency budget exceeded — gate fails red");
  process.exit(1);
}
