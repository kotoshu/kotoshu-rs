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
import { createHash } from "node:crypto";
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
      `score ≈ ${expected} (got ${actual})`,
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

  // --- Semantic generation surface ----------------------------------------
  // semanticSuggest asks the MODEL for nearest vocabulary words: an OOV
  // typo embeds through its character n-grams and the intended word
  // comes back as top-1. Expectations frozen from the fixture's real
  // fastText vectors (the same values the Rust unit tests pin).
  const { semanticSuggest } = mod;
  assert("semanticSuggest is exported", typeof semanticSuggest === "function");

  const neighbors = semanticSuggest(model, "catt", 4);
  assert("semanticSuggest returns an Array", Array.isArray(neighbors));
  assert(
    "every row has exactly {word, score}",
    neighbors.every((row) => {
      const keys = Object.keys(row).sort();
      return keys.length === 2 && keys[0] === "score" && keys[1] === "word";
    }),
  );
  assertEqual(
    "semanticSuggest('catt')[0].word — the OOV typo regenerates cat",
    "cat",
    neighbors[0]?.word,
  );
  near(neighbors[0]?.score, 1.0);
  assertEqual("semanticSuggest('catt')[1].word", "dog", neighbors[1]?.word);
  near(neighbors[1]?.score, 0.707432);
  assertEqual(
    "k clamps at the vocabulary minus the query",
    39,
    semanticSuggest(model, "cat", 1000).length,
  );
  assert(
    "the query word itself is excluded",
    semanticSuggest(model, "cat", 1000).every((row) => row.word !== "cat"),
  );
  assertEqual(
    "nothing resolvable is an honest empty list",
    0,
    semanticSuggest(model, "florbington", 4).length,
  );

  // --- Bucket-table sibling surface (plan 103) --------------------------
  // The bucket fixture is a 4-row subset of the real en/buckets
  // artifact (see LICENSE-NOTE.md): just the rows the OOV queries
  // teh/catt/hotcold address. With it attached, "teh" — whose only
  // unmarked n-gram is itself (not in the 40-word fixture vocab) —
  // embeds through its marked 5-gram "<teh>" and surfaces "the" as
  // the top-1 neighbor (the canonical OOV gate).
  const bucketsBytes = new Uint8Array(
    await readFile(path.join(modelsDir, "en-buckets-truncated.onnx")),
  );
  const modelWithBuckets = loadModel(modelBytes, vocabBytes, bucketsBytes);
  assert("loadModel accepts the optional bucketsBytes", typeof modelWithBuckets === "object");

  const teh = semanticSuggest(modelWithBuckets, "teh", 4);
  assertEqual("bucket-backed semanticSuggest('teh')[0].word", "the", teh[0]?.word);
  near(teh[0]?.score, 0.3587);

  const cattBucketed = semanticSuggest(modelWithBuckets, "catt", 4);
  assertEqual("bucket-backed semanticSuggest('catt')[0].word", "cat", cattBucketed[0]?.word);
  near(cattBucketed[0]?.score, 0.8902);

  const hotcold = semanticSuggest(modelWithBuckets, "hotcold", 4);
  assertEqual("bucket-backed semanticSuggest('hotcold')[0].word", "hot", hotcold[0]?.word);
  assertEqual("bucket-backed semanticSuggest('hotcold')[1].word", "cold", hotcold[1]?.word);
  // Top-2 ordering preserved; magnitudes reflect the union composition.
  near(hotcold[0]?.score, 0.8711);
  near(hotcold[1]?.score, 0.7930);

  // The handle still exposes free() with the bucket table attached.
  assert("modelWithBuckets.free() exists", typeof modelWithBuckets.free === "function");

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

  // --- Language pack surface (plan 113) ---------------------------------
  // One call over one artifact replacing the five per-artifact fetches.
  // The pack is framed here exactly as scripts/build_packs.py in the
  // models repo frames it (KPK1: u32 LE length + tag byte + payload +
  // per-section sha256; the Rust core suite pins the same framing
  // against the Python builder through a shared golden sha256), from
  // the same sources the sections above loaded separately — so the
  // parity assertions below prove the pack handles BEHAVE identically
  // to the per-artifact handles, not merely exist.
  const { loadPack } = mod;
  assert("loadPack is exported", typeof loadPack === "function");

  const frameSection = (tag, payload) => {
    const framed = Buffer.alloc(5 + payload.length + 32);
    framed.writeUInt32LE(payload.length, 0);
    framed[4] = tag;
    payload.copy(framed, 5);
    framed.set(createHash("sha256").update(payload).digest(), 5 + payload.length);
    return framed;
  };
  const affBytes = Buffer.from(aff, "utf8");
  const dicBytes = Buffer.from(dic, "utf8");
  const countBuf = Buffer.alloc(4);
  countBuf.writeUInt32LE(5, 0);
  const packBytes = Buffer.concat([
    Buffer.from("KPK1"),
    countBuf,
    frameSection(1, affBytes),
    frameSection(2, dicBytes),
    frameSection(3, Buffer.from(modelBytes)),
    frameSection(4, Buffer.from(vocabBytes)),
    frameSection(5, Buffer.from(bucketsBytes)),
  ]);

  const pack = loadPack(new Uint8Array(packBytes));
  assert("loadPack returns { dictionary, model }", typeof pack === "object" && pack !== null);
  assert(
    "the pack object carries exactly the two handles",
    Object.keys(pack).sort().join() === "dictionary,model",
  );
  assert(
    "the pack dictionary exposes the KotoshuWasm surface",
    typeof pack.dictionary.correct === "function" &&
      typeof pack.dictionary.suggest === "function" &&
      typeof pack.dictionary.free === "function",
  );
  assert(
    "the pack model exposes the KotoshuModel surface",
    typeof pack.model.free === "function",
  );

  // Parity: dictionary identical to the per-artifact constructor over
  // the same sources.
  assertEqual("pack dictionary.correct('helo') matches", dictionary.correct("helo"), pack.dictionary.correct("helo"));
  assertEqual("pack dictionary.correct('ruby') matches", dictionary.correct("ruby"), pack.dictionary.correct("ruby"));
  const packedRows = pack.dictionary.suggest("hlelo", 5);
  assertEqual(
    "pack dictionary.suggest('hlelo') rows are identical",
    JSON.stringify(rows),
    JSON.stringify(packedRows),
  );

  // Parity: model identical to the per-artifact loadModel (buckets
  // attached) over the same bytes — the bucket-backed OOV gate included.
  const packedTeh = semanticSuggest(pack.model, "teh", 4);
  assertEqual("pack model semanticSuggest('teh')[0].word", "the", packedTeh[0]?.word);
  near(0.3587, packedTeh[0]?.score);
  assert(
    "pack model rerank('cat', 'dog') matches the per-artifact handle",
    rerank(modelWithBuckets, "cat", "dog") === rerank(pack.model, "cat", "dog"),
  );
  assert(
    "pack model rerank('puppy', machine context) matches the per-artifact handle",
    rerank(modelWithBuckets, "puppy", "the computer and the keyboard") ===
      rerank(pack.model, "puppy", "the computer and the keyboard"),
  );

  pack.dictionary.free();
  pack.model.free();
  modelWithBuckets.free();

  let packThrew = null;
  try {
    const corrupt = new Uint8Array(packBytes);
    corrupt[corrupt.length - 40] ^= 0xFF; // inside the buckets payload
    loadPack(corrupt);
  } catch (error) {
    packThrew = error;
  }
  assert("loadPack rejects a tampered section (sha256 footer)", packThrew instanceof Error);
  assert(
    "the pack rejection names the section",
    /pack|section|sha256/i.test(String(packThrew?.message)),
  );

  let magicThrew = null;
  try {
    loadPack(new Uint8Array(1024).fill(0x42));
  } catch (error) {
    magicThrew = error;
  }
  assert("loadPack rejects non-pack bytes", magicThrew instanceof Error);

  // --- Language detection (plan 102) ----------------------------------
  // The fixture is the real registry lid.176 artifact pair (the same
  // bytes scripts/build_lid.py writes for the models repo release);
  // numbers below are the gem's frozen detection outputs on a small
  // corpus, asserted to the int8 gate tolerance (1e-3 with headroom).
  const { loadLid, detectLanguage } = mod;
  assert("loadLid is exported", typeof loadLid === "function");
  assert("detectLanguage is exported", typeof detectLanguage === "function");

  const lidDir =
    process.env.KOTOSHU_LID_FIXTURES_DIR ??
    path.join(root, "kotoshu", "tests", "fixtures", "lid");
  const lidOnnxBytes = new Uint8Array(
    await readFile(path.join(lidDir, "lid.176.onnx")),
  );
  const lidVocabBytes = new Uint8Array(
    await readFile(path.join(lidDir, "lid.176.vocab.json")),
  );
  const lid = loadLid(lidOnnxBytes, lidVocabBytes);
  assert("loadLid returns a KotoshuLid handle", typeof lid === "object");
  assert("the LID handle exposes free()", typeof lid.free === "function");

  const lidCases = [
    ["the quick brown fox jumps over the lazy dog", "en"],
    ["今日はとても良い天気ですね", "ja"],
    ["Быстрая бурая лиса прыгает через ленивую собаку", "ru"],
    ["Esta é uma frase de teste para verificar a deteção de idioma", "pt"],
  ];
  for (const [text, code] of lidCases) {
    const detection = detectLanguage(lid, text);
    assert(
      `detectLanguage("${text.slice(0, 30)}…").code == "${code}" (got "${detection.code}")`,
      detection.code === code,
    );
    assert(
      `detectLanguage score is a finite number in [0, 1] for "${code}"`,
      Number.isFinite(detection.score) &&
        detection.score >= 0 &&
        detection.score <= 1,
    );
  }

  let lidThrew = null;
  try {
    loadLid(new Uint8Array(1024).fill(0x42), lidVocabBytes);
  } catch (error) {
    lidThrew = error;
  }
  assert("loadLid rejects malformed model bytes", lidThrew instanceof Error);
} catch (error) {
  failures += 1;
  console.error(`FAIL smoke setup (${error?.stack ?? error})`);
} finally {
  console.log(`${assertions} assertions, ${failures} failures`);
  process.exit(failures === 0 ? 0 : 1);
}
