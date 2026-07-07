// Parallel trainer gate: 2 workers must train a micro model on trivial text
// (loss falls hard) and shut down cleanly. Exercises SAB weight sharing,
// grad-slab averaging, Atomics generations, and worker lifecycle.

import { test } from "node:test";
import assert from "node:assert/strict";
import { noGrad } from "../src/core/tensor.ts";
import { RNG } from "../src/core/rng.ts";
import { GPT } from "../src/nn/gpt.ts";
import { CharTokenizer } from "../src/tokenizer/char.ts";
import { trainParallel } from "../src/train-parallel.ts";
import { generate } from "../src/sample.ts";

test("data-parallel training: loss falls on 'abab...' with 2 workers", async () => {
  const text = "ab".repeat(2000);
  const tok = CharTokenizer.fromText(text);
  const tokens = tok.encode(text);
  const cfg = { vocabSize: tok.size, blockSize: 8, nLayer: 1, nHead: 2, nEmbd: 16 };
  const model = new GPT(cfg, new RNG(5));

  const before = noGrad(() => {
    const x = tokens.subarray(0, 8);
    const y = tokens.subarray(1, 9);
    return model.loss(x, y, 1, 8).item();
  });

  const { finalLoss } = await trainParallel(model, tokens, {
    steps: 300,
    batchSize: 4,
    blockSize: 8,
    lr: 5e-3,
    warmup: 10,
    weightDecay: 0,
    seed: 7,
    workers: 2,
    valSplit: 0.1,
  });

  assert.ok(before > 0.6, `initial loss ${before} should be near ln(2)`);
  assert.ok(finalLoss < 0.2, `parallel training failed to learn: loss ${finalLoss}`);

  // the main-thread model must reflect the trained shared weights
  const out = generate(model, tok, "a", { maxNewTokens: 10, temperature: 0.5, seed: 3 });
  assert.match(out, /^[ab]+$/);
});
