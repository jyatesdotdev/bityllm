// M5 gates (DESIGN §19.3): the KV-cache inference path must agree with the
// training-path forward on the same weights; int8 checkpoints must reload to
// near-identical logits; the shell must never display the stop-sequence.

import { test } from "node:test";
import assert from "node:assert/strict";
import { setBackend, CPUBackend } from "../src/backend/index.ts";
import { noGrad } from "../src/core/tensor.ts";
import { RNG } from "../src/core/rng.ts";
import { GPT } from "../src/nn/gpt.ts";
import { CharTokenizer } from "../src/tokenizer/char.ts";
import { serialize, serializeInt8, deserialize } from "../src/io/checkpoint.ts";
import { InferenceSession } from "../src/infer/session.ts";
import { Shell } from "../src/infer/shell.ts";
import type { ShellIO } from "../src/infer/shell.ts";

setBackend(new CPUBackend(Float32Array));

const VOCAB = [..."abcdefghijklmnopqrstuvwxyz $~:@\n"];
const cfg = { vocabSize: VOCAB.length, blockSize: 32, nLayer: 2, nHead: 2, nEmbd: 16 };

test("infer parity: KV-cache session logits ≈ full forward logits", () => {
  const model = new GPT(cfg, new RNG(11));
  const tok = new CharTokenizer(VOCAB);
  const text = "hello bity terminal";
  const ids = tok.encode(text);
  const T = ids.length;

  // training-path forward: full context, last-position logits
  const full = noGrad(() => model.forward(ids, 1, T));
  const V = cfg.vocabSize;
  const want = full.data.subarray((T - 1) * V, T * V);

  // inference path: token-by-token through the KV cache
  const sess = new InferenceSession(model, tok);
  sess.feed(text);
  const got = sess.generate("", { maxNewTokens: 0 }); // no-op; we compare internals below
  void got;

  // reach the last logits via a 1-token continuation of a fresh session
  const sess2 = new InferenceSession(model, tok);
  sess2.feed(text);
  // @ts-expect-error accessing private field for the parity check
  const logits: Float32Array = sess2.last;
  assert.ok(logits, "session should have logits after feed()");
  for (let j = 0; j < V; j++)
    assert.ok(Math.abs(logits[j] - want[j]) < 1e-3, `logit[${j}]: session ${logits[j]} vs full ${want[j]}`);
});

test("KV-cache rewind: session survives contexts longer than blockSize", () => {
  const model = new GPT(cfg, new RNG(11));
  const tok = new CharTokenizer(VOCAB);
  const sess = new InferenceSession(model, tok);
  sess.feed("abcdefgh ".repeat(12)); // 108 chars > blockSize 32
  assert.ok(sess.length <= cfg.blockSize, `cache overflow: ${sess.length}`);
  const out = sess.generate("", { maxNewTokens: 5, temperature: 1, seed: 1 });
  assert.equal(out.length, 5);
});

test("int8 checkpoint: ~4× smaller, logits close after dequant", () => {
  const model = new GPT(cfg, new RNG(23));
  const tok = new CharTokenizer(VOCAB);

  const f32 = serialize(model, tok);
  const i8 = serializeInt8(model, tok);
  // at toy scale the header + kept-f32 1-D params dominate; real models hit ~27%
  assert.ok(i8.length < f32.length * 0.5, `int8 ${i8.length} not <50% of f32 ${f32.length}`);

  const { model: m2 } = deserialize(i8);
  const ids = tok.encode("hello bity");
  const a = noGrad(() => model.forward(ids, 1, ids.length)).data;
  const b = noGrad(() => m2.forward(ids, 1, ids.length)).data;
  let maxDiff = 0;
  for (let i = 0; i < a.length; i++) maxDiff = Math.max(maxDiff, Math.abs(a[i] - b[i]));
  assert.ok(maxDiff < 0.35, `int8 logit drift too large: ${maxDiff}`);
});

test("speculative peek: snapshot → feed → restore leaves no trace (ghost suggestions)", () => {
  const model = new GPT(cfg, new RNG(17));
  const tok = new CharTokenizer(VOCAB);

  const a = new InferenceSession(model, tok);
  a.feed("hello wor");
  const snap = a.snapshot();
  a.feed("XYZ zebra quokka"); // speculative peek pollutes the cache...
  a.restore(snap);            // ...rollback
  a.feed("ld");               // stale rows must be overwritten identically

  const b = new InferenceSession(model, tok);
  b.feed("hello world");

  // @ts-expect-error private access: compare final logits bitwise
  const la: Float32Array = a.last;
  // @ts-expect-error private access
  const lb: Float32Array = b.last;
  assert.deepEqual([...la], [...lb]);
});

test("shell: stop-sequence is never displayed, prompt context is fed", async () => {
  const model = new GPT(cfg, new RNG(31));
  const tok = new CharTokenizer(VOCAB);
  const sess = new InferenceSession(model, tok);
  const shell = new Shell(sess, { prompt: "bity:~$ ", seed: 7 });

  let screen = "";
  const io: ShellIO = {
    write: (s) => (screen += s),
    clear: () => (screen = ""),
    delay: async () => {},
  };
  await shell.run("hello", io);
  assert.ok(!screen.includes("bity:~$"), `stop-sequence leaked into display: ${JSON.stringify(screen.slice(-40))}`);
});
