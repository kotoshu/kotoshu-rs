#!/usr/bin/env ruby
# frozen_string_literal: true

# Smoke test for the `ruby` feature (P4): loads the reference shim extension
# (tests/ruby_ext, built by scripts/ruby_ffi_smoke.sh) into a REAL Ruby VM
# and drives the REAL engine over REAL fixture dictionaries — no mocks.
#
# Usage: scripts/ruby_ffi_smoke.sh (stages the built cdylib on $LOAD_PATH
# first). Expectations marked "conformance vector" are frozen by the gem's
# exported vectors (tests/fixtures/vectors.jsonl), not hand-written.

FIXTURES_DIR = ENV.fetch("KOTOSHU_FIXTURES_DIR") do
  File.expand_path("fixtures", __dir__)
end

unless Dir.exist?(FIXTURES_DIR)
  warn "ruby_ffi_smoke: fixtures not found at #{FIXTURES_DIR} —" \
       " run scripts/sync_conformance.sh first (KOTOSHU_GEM_DIR=... if needed)"
  exit 1
end

require "kotoshu_ruby_ext"

$failures = 0
$assertions = 0

def assert(label, condition)
  $assertions += 1
  if condition
    puts "PASS #{label}"
  else
    $failures += 1
    puts "FAIL #{label}"
  end
end

def assert_equal(label, expected, actual)
  assert("#{label} (expected #{expected.inspect}, got #{actual.inspect})",
         expected == actual)
end

# --- Module surface -------------------------------------------------------

assert("Kotoshu::Native is defined", defined?(Kotoshu::Native) == "constant")
assert("VERSION is a dotted triple String",
       Kotoshu::Native::VERSION.is_a?(String) && Kotoshu::Native::VERSION.match?(/\A\d+\.\d+\.\d+\z/))
assert_equal("available? is true", true, Kotoshu::Native.available?)
assert("Error is a RuntimeError subclass",
       Kotoshu::Native::Error.ancestors.include?(RuntimeError))

# --- The gem's `en` test dictionary ---------------------------------------
# Every vector on this dictionary is frozen in vectors.jsonl; "helo" IS a
# word here, so its suggest list is empty (words the dictionary accepts get
# no suggestions — gem behavior).

test_dic = File.join(FIXTURES_DIR, "spec/fixtures/dictionaries/hunspell/test")
dictionary = Kotoshu::Native::Dictionary.load("#{test_dic}.aff", "#{test_dic}.dic")
assert("Dictionary instance", dictionary.is_a?(Kotoshu::Native::Dictionary))
assert_equal("correct?('hello') — conformance vector", true, dictionary.correct?("hello"))
assert_equal("correct?('ruby') — conformance vector", false, dictionary.correct?("ruby"))
assert_equal("correct?('helo') is a listed word", true, dictionary.correct?("helo"))
assert_equal("suggest('helo', 5) — conformance vector (accepted words suggest nothing)",
             [], dictionary.suggest("helo", 5))
assert_equal("suggest default limit path", [], dictionary.suggest("helo"))

# --- The `base` integrational dictionary ----------------------------------
# The dictionary behind the canonical "hlelo" → "hello" conformance example.

base_dic = File.join(FIXTURES_DIR, "spec/integrational/fixtures/base")
base = Kotoshu::Native::Dictionary.load("#{base_dic}.aff", "#{base_dic}.dic")
assert_equal("base correct?('hello')", true, base.correct?("hello"))
assert_equal("base correct?('hlelo') — conformance vector", false, base.correct?("hlelo"))
assert_equal("base suggest('hlelo', 5) — conformance vector",
             [{ "word" => "hello", "distance" => 1, "confidence" => 1.0,
                "source" => "edit_distance" }],
             base.suggest("hlelo", 5))

suggestions = base.suggest("helo", 5)
assert("base suggest('helo', 5) includes 'hello'",
       suggestions.any? { |suggestion| suggestion["word"] == "hello" })
if suggestions.empty?
  assert("suggestion row shape", false)
else
  row = suggestions.first
  assert("row is a Hash of exactly the gem Suggestion fields",
         row.is_a?(Hash) && row.keys.sort == %w[confidence distance source word])
  assert("distance is an Integer", row["distance"].is_a?(Integer))
  assert("confidence is a Float in [0, 1]",
         row["confidence"].is_a?(Float) && row["confidence"].between?(0.0, 1.0))
  assert("source is a strategy String",
         row["source"].is_a?(String) &&
         %w[edit_distance phonetic keyboard_proximity ngram].include?(row["source"]))
end
assert("default-limit suggest stays within the limit", base.suggest("hlelo").length <= 5)

# --- Error surface --------------------------------------------------------

begin
  Kotoshu::Native::Dictionary.load("/nonexistent-kotoshu.aff", "/nonexistent-kotoshu.dic")
  assert("missing dictionary raises Kotoshu::Native::Error", false)
rescue Kotoshu::Native::Error => e
  assert("missing dictionary raises Kotoshu::Native::Error", true)
  assert("error message names the Rust failure and the path",
         e.message.include?("/nonexistent-kotoshu.aff") && e.message.include?("failed to load"))
end

puts "ruby ffi smoke: #{$assertions} assertions, #{$failures} failures" \
     " (kotoshu-rs #{Kotoshu::Native::VERSION}, ruby #{RUBY_VERSION})"
exit($failures.zero? ? 0 : 1)
