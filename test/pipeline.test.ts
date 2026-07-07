// M3 gates: tokenizer round-trip, dataset shapes, checkpoint round-trip
// (bit-identical logits), and a tiny end-to-end train → loss-falls check.

import { test } from "node:test";
import assert from "node:assert/strict";
import { noGrad } from "../src/core/tensor.ts";
import { RNG } from "../src/core/rng.ts";
import { GPT } from "../src/nn/gpt.ts";
import { CharTokenizer } from "../src/tokenizer/char.ts";
import { Dataset } from "../src/data/dataset.ts";
import { train } from "../src/train.ts";
import { generate } from "../src/sample.ts";
import { serialize, deserialize } from "../src/io/checkpoint.ts";

test("tokenizer round-trip", () => {
  const text = "guest@bity:~$ ls -la\ntotal 42\ndrwxr-xr-x 2 guest guest 4096 .\n";
  const tok = CharTokenizer.fromText(text);
  assert.equal(tok.decode(tok.encode(text)), text);
  assert.ok(tok.size > 10 && tok.size < 50);
});

test("dataset batches: y is x shifted by one", () => {
  const tokens = Int32Array.from({ length: 500 }, (_, i) => i % 7);
  const data = new Dataset(tokens, 0.1);
  const rng = new RNG(1);
  const { x, y } = data.getBatch("train", 4, 16, rng);
  assert.equal(x.length, 64);
  assert.equal(y.length, 64);
  // within each row, y[t] must equal x[t+1]
  for (let b = 0; b < 4; b++)
    for (let t = 0; t < 15; t++)
      assert.equal(y[b * 16 + t], x[b * 16 + t + 1], `row ${b} pos ${t}`);
});

test("checkpoint round-trip: bit-identical logits", () => {
  const text = "the quick brown fox jumps over the lazy dog 0123456789";
  const tok = CharTokenizer.fromText(text);
  const cfg = { vocabSize: tok.size, blockSize: 16, nLayer: 2, nHead: 2, nEmbd: 16 };
  const model = new GPT(cfg, new RNG(99));

  const bytes = serialize(model, tok, { step: 123 });
  const { model: m2, tok: t2, step } = deserialize(bytes);

  assert.equal(step, 123);
  assert.deepEqual(t2.vocab, tok.vocab);

  const idx = tok.encode("the quick brown ").subarray(0, 16);
  const a = noGrad(() => model.forward(idx, 1, 16)).data;
  const b = noGrad(() => m2.forward(idx, 1, 16)).data;
  assert.deepEqual([...a], [...b]);
});

test("end-to-end: train a micro model on repetitive text, loss falls, sample decodes", () => {
  const text = "ab".repeat(600); // trivially learnable
  const tok = CharTokenizer.fromText(text);
  const data = new Dataset(tok.encode(text), 0.1);
  const cfg = { vocabSize: tok.size, blockSize: 8, nLayer: 1, nHead: 2, nEmbd: 16 };
  const model = new GPT(cfg, new RNG(5));

  const { finalLoss } = train(model, data, {
    steps: 300,
    batchSize: 4,
    blockSize: 8,
    lr: 5e-3,
    warmup: 10,
    weightDecay: 0,
    seed: 7,
  });
  assert.ok(finalLoss < 0.2, `loss ${finalLoss} should be tiny on 'ababab...'`);

  const out = generate(model, tok, "a", { maxNewTokens: 10, temperature: 0.5, seed: 3 });
  assert.equal(out.length, 10);
  assert.match(out, /^[ab]+$/);
});
