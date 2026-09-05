#!/usr/bin/env node
// Node smoke test for the `wasm` feature (P4c): loads the REAL wasm-pack
// output (kotoshu-wasm/pkg, --target bundler) into a REAL JS engine and
// drives the REAL engine over REAL fixture dictionaries — no mocks.
//
// Usage: scripts/wasm_build.sh && node scripts/wasm_node_smoke.mjs
// (KOTOSHU_WASM_PKG overrides the pkg dir; the fixtures must be synced
// first — scripts/sync_conformance.sh — exactly like the Ruby smoke.
// The MODEL fixture is checked in at kotoshu/tests/fixtures/models/,
// derived from the real registry en/mini tier by
// scripts/make_model_fixture.py; see its LICENSE-NOTE.md.)
//
// Expectations marked "conformance vector" are frozen by the gem's exported
// vectors (tests/fixtures/vectors.jsonl), not hand-written. Model-rerank
// expectations are frozen by the fixture generator's reference cosines.

import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const pkgDir = process.env.KOTOSHU_WASM_PKG ?? path.join(root, "kotoshu-wasm", "pkg");
const fixturesDir =
  process.env.KOTOSHU_FIXTURES_DIR ?? path.join(root, "tests", "fixtures");
const modelsDir =
  process.env.KOTOSHU_MODEL_FIXTURES_DIR ??
  path.join(root, "kotoshu", "tests", "fixtures", "models");

let failures = 0;
let assertions = 0;

function assert(label, condition) {
  assertions += 1;
  console.log(`${condition ? "PASS" : "FAIL"} ${label}`);
  if (!condition) failures += 1;
}

function assertEqual(label, expected, actual) {
  assert(
    `${label} (expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)})`,
    Object.is(expected, actual),
  );
}

try {
  // pkg/kotoshu_wasm.js is an ES module; when pkg/package.json lacks
  // "type": "module", Node needs the module attribute to parse .js as ESM.
  const pkgJson = JSON.parse(await readFile(path.join(pkgDir, "package.json"), "utf8"));
  const glueUrl = pathToFileURL(path.join(pkgDir, "kotoshu_wasm.js")).href;
  const mod =
    pkgJson.type === "module"
      ? await import(glueUrl)
      : await import(glueUrl, { with: { type: "module" } });
  const { KotoshuWasm } = mod;

  // Initialization depends on the wasm-pack target the pkg was built with
  // (scripts/wasm_build.sh defaults to bundler):
  // - bundler glues self-initialize: they import the .wasm as an ES module
  //   (Node >= 24 supports this, still marked experimental) and call its
  //   start export at module scope;
  // - web/node glues export a default init() accepting the raw bytes.
  if (typeof mod.default === "function") {
    await mod.default(await readFile(path.join(pkgDir, "kotoshu_wasm_bg.wasm")));
  }

  // --- Class surface -----------------------------------------------------
  assert("KotoshuWasm is a class", typeof KotoshuWasm === "function");
  assert(
    "VERSION is a dotted triple String",
    typeof KotoshuWasm.VERSION === "string" &&
      /^\d+\.\d+\.\d+$/.test(KotoshuWasm.VERSION),
  );

  // --- The gem's `en` test dictionary -------------------------------------
  // Every vector on this dictionary is frozen in vectors.jsonl; "helo" IS
  // a word here, so its suggest list is empty (words the dictionary
  // accepts get no suggestions — gem behavior).
  const dicBase = path.join(fixturesDir, "spec/fixtures/dictionaries/hunspell/test");
  // SET UTF-8 in this .aff: reading as UTF-8 hands the engine each file's
  // exact bytes (see the KotoshuWasm constructor docs).
  const aff = await readFile(`${dicBase}.aff`, "utf8");
  const dic = await readFile(`${dicBase}.dic`, "utf8");
  const dictionary = new KotoshuWasm(aff, dic);

  assertEqual("correct('hello') — conformance vector", true, dictionary.correct("hello"));
  assertEqual("correct('ruby') — OOV conformance vector", false, dictionary.correct("ruby"));
  assertEqual("correct('helo') — conformance vector", true, dictionary.correct("helo"));
  assertEqual("correct('') — empty word", false, dictionary.correct(""));

  const rows = dictionary.suggest("hlelo", 5);
  assert("suggest('hlelo') returns an Array", Array.isArray(rows));
  const first = rows[0];
  assertEqual("suggest('hlelo')[0].word — frozen conformance row", "hello", first?.word);
  assertEqual("suggest('hlelo')[0].distance", 1, first?.distance);
  assertEqual("suggest('hlelo')[0].confidence", 1.0, first?.confidence);
  assertEqual("suggest('hlelo')[0].source", "edit_distance", first?.source);
  assert(
    "every row has exactly the four SUGGESTION_KEYS",
    rows.every((row) => {
      const keys = Object.keys(row).sort();
      return (
        keys.length === 4 &&
        keys[0] === "confidence" &&
        keys[1] === "distance" &&
        keys[2] === "source" &&
        keys[3] === "word"
      );
    }),
  );
  assertEqual(
    "suggest('helo') — dictionary word yields nothing",
    0,
    dictionary.suggest("helo", 5).length,
  );
  assertEqual(
    "suggest limit defaults to 5 (gem default)",
    rows.length,
    dictionary.suggest("hlelo").length,
  );

  // --- Error surface ------------------------------------------------------
  let threw = null;
  try {
    // REP announces 2 entries but supplies 1 — a truncated counted block,
    // a genuine LoadError::Aff in the engine.
    new KotoshuWasm("REP 2\nX\n", "0\n");
  } catch (error) {
    threw = error;
  }
  assert("malformed sources reject with an Error", threw instanceof Error);
  assert(
    "the rejection carries the Rust message",
    typeof threw?.message === "string" && threw.message.length > 0,
  );

  // --- Model rerank surface (plan 85) -------------------------------------
  // The fixture is a 40-word truncation of the real registry v1.0.1
  // en/mini tier (int8-per-row, real FastText vectors). Numbers below
  // are the generator's reference cosines, frozen.
  const { loadModel, rerank } = mod;
  assert("loadModel is exported", typeof loadModel === "function");
  assert("rerank is exported", typeof rerank === "function");

  const modelBytes = new Uint8Array(
    await readFile(path.join(modelsDir, "en-mini-truncated.onnx")),
  );
  const vocabBytes = new Uint8Array(
    await readFile(path.join(modelsDir, "en-mini-truncated.vocab.json")),
  );
  const model = loadModel(modelBytes, vocabBytes);
  assert("loadModel returns a KotoshuModel handle", typeof model === "object");
  assert("the handle exposes free()", typeof model.free === "function");

  const near = (expected, actual, tolerance = 1e-4) =>
    assert(
      `rerank ≈ ${expected} (got ${actual})`,
      Number.isFinite(actual) && Math.abs(actual - expected) < tolerance,
    );

  // Single-token contexts are plain cosines: dog is a much nearer
  // neighbor of cat than computer is.
  near(0.707432, rerank(model, "cat", "dog"));
  near(0.187228, rerank(model, "cat", "computer"));
  assert(
    "rerank orders the sensible neighbor first",
    rerank(model, "cat", "dog") > rerank(model, "cat", "computer"),
  );
  // Tokenizer parity through wasm: punctuation stripped, lookups
  // downcased.
  assert(
    "context punctuation is stripped",
    rerank(model, "cat", "dog!") === rerank(model, "cat", "dog"),
  );
  assert(
    "cased words resolve through the lowercase fallback",
    rerank(model, "CAT", "Dog") === rerank(model, "cat", "dog"),
  );

  // Sentence contexts: the more-sensible context scores higher — the
  // same ordering the gem's context boost produces.
  const animal = rerank(model, "puppy", "the dog and the cat");
  const machine = rerank(model, "puppy", "the computer and the keyboard");
  near(0.340374, animal);
  near(0.126196, machine);
  assert("the animal context beats the machine context", animal > machine);

  // Honest zeros for OOV on either side (the gem's `(sim || 0.0)`).
  assertEqual("OOV word scores 0", 0, rerank(model, "florbington", "dog"));
  assertEqual("all-OOV context scores 0", 0, rerank(model, "cat", "zzz qqq"));

  let modelThrew = null;
  try {
    loadModel(new Uint8Array(1024).fill(0x42), vocabBytes);
  } catch (error) {
    modelThrew = error;
  }
  assert("loadModel rejects malformed model bytes", modelThrew instanceof Error);
  assert(
    "the model rejection carries the Rust message",
    typeof modelThrew?.message === "string" && modelThrew.message.length > 0,
  );
} catch (error) {
  failures += 1;
  console.error(`FAIL smoke setup (${error?.stack ?? error})`);
} finally {
  console.log(`${assertions} assertions, ${failures} failures`);
  process.exit(failures === 0 ? 0 : 1);
}
