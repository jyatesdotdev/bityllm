// M2 gates (DESIGN §19): full-model grad-check in Float64, then the
// overfit-one-batch integration test — the whole training path must be able
// to memorize a single tiny batch before it's allowed near real data.

import { test } from "node:test";
import assert from "node:assert/strict";
import { setBackend, CPUBackend } from "../src/backend/index.ts";
import { noGrad } from "../src/core/tensor.ts";
import { RNG } from "../src/core/rng.ts";
import { GPT } from "../src/nn/gpt.ts";
import { AdamW, clipGradNorm, cosineLR } from "../src/optim/adamw.ts";

test("full-model grad-check (Float64, every parameter)", () => {
  setBackend(new CPUBackend(Float64Array));
  const cfg = { vocabSize: 11, blockSize: 4, nLayer: 1, nHead: 2, nEmbd: 8 };
  const rng = new RNG(7);
  const model = new GPT(cfg, rng);
  const B = 2, T = 4;
  const x = new Int32Array(B * T), y = new Int32Array(B * T);
  for (let i = 0; i < B * T; i++) {
    x[i] = rng.randint(0, cfg.vocabSize);
    y[i] = rng.randint(0, cfg.vocabSize);
  }

  const f = (): number => noGrad(() => model.loss(x, y, B, T).item());

  for (const p of model.parameters()) p.zeroGrad();
  model.loss(x, y, B, T).backward();

  const EPS = 1e-5, RTOL = 1e-3, ATOL = 1e-7;
  let checked = 0;
  for (const [name, p] of model.namedParameters()) {
    assert.ok(p.grad !== null, `${name} received no gradient`);
    for (let i = 0; i < p.size; i++) {
      const orig = p.data[i];
      p.data[i] = orig + EPS;
      const fp = f();
      p.data[i] = orig - EPS;
      const fm = f();
      p.data[i] = orig;
      const num = (fp - fm) / (2 * EPS);
      const ana: number = p.grad![i];
      const tol = ATOL + RTOL * Math.max(Math.abs(num), Math.abs(ana));
      assert.ok(Math.abs(num - ana) <= tol, `${name}[${i}]: analytic=${ana} numeric=${num}`);
      checked++;
    }
  }
  assert.ok(checked > 500, `expected to check hundreds of params, got ${checked}`);
});

test("weight tying: wte gradient gets both embedding and lm-head terms", () => {
  setBackend(new CPUBackend(Float64Array));
  const cfg = { vocabSize: 11, blockSize: 4, nLayer: 1, nHead: 2, nEmbd: 8 };
  const model = new GPT(cfg, new RNG(7));
  const x = new Int32Array([1, 2, 3, 4]);
  const y = new Int32Array([2, 3, 4, 5]);
  model.loss(x, y, 1, 4).backward();
  // rows never used as input tokens still get head gradient (softmax pushes down
  // every vocab logit) — nonzero grad on an unused row proves the head path.
  const C = cfg.nEmbd;
  const unusedRow = 9;
  let norm = 0;
  for (let j = 0; j < C; j++) norm += Math.abs(model.wte.grad![unusedRow * C + j]);
  assert.ok(norm > 0, "tied head gradient missing on unused-token row");
});

test("overfit one batch: loss → ~0 (Float32, full optimizer stack)", () => {
  setBackend(new CPUBackend(Float32Array));
  const cfg = { vocabSize: 16, blockSize: 16, nLayer: 2, nHead: 4, nEmbd: 32 };
  const rng = new RNG(42);
  const model = new GPT(cfg, rng);
  const B = 2, T = 16;
  const x = new Int32Array(B * T), y = new Int32Array(B * T);
  for (let i = 0; i < B * T; i++) {
    x[i] = rng.randint(0, cfg.vocabSize);
    y[i] = rng.randint(0, cfg.vocabSize);
  }

  const { decay, noDecay } = model.paramGroups();
  const opt = new AdamW(decay, noDecay, { lr: 3e-3, weightDecay: 0 });

  const first = model.loss(x, y, B, T).item();
  assert.ok(Math.abs(first - Math.log(cfg.vocabSize)) < 0.5, `init loss ${first} ≉ ln(V)=${Math.log(cfg.vocabSize)}`);

  let last = first;
  const steps = 400;
  for (let s = 0; s < steps; s++) {
    const loss = model.loss(x, y, B, T);
    last = loss.item();
    loss.backward();
    clipGradNorm(opt.params, 1.0);
    opt.step(cosineLR(s, { lr: 3e-3, warmup: 20, total: steps, minRatio: 0.1 }));
    opt.zeroGrad();
  }
  assert.ok(last < 0.1, `overfit failed: loss ${last} after ${steps} steps (start ${first.toFixed(3)})`);
});

test("determinism: same seed ⇒ bit-identical loss trajectory", () => {
  setBackend(new CPUBackend(Float32Array));
  const run = (): number[] => {
    const cfg = { vocabSize: 12, blockSize: 8, nLayer: 1, nHead: 2, nEmbd: 16 };
    const rng = new RNG(1337);
    const model = new GPT(cfg, rng);
    const x = new Int32Array(16), y = new Int32Array(16);
    for (let i = 0; i < 16; i++) {
      x[i] = rng.randint(0, 12);
      y[i] = rng.randint(0, 12);
    }
    const { decay, noDecay } = model.paramGroups();
    const opt = new AdamW(decay, noDecay, { lr: 1e-3 });
    const losses: number[] = [];
    for (let s = 0; s < 5; s++) {
      const loss = model.loss(x, y, 2, 8);
      losses.push(loss.item());
      loss.backward();
      clipGradNorm(opt.params, 1.0);
      opt.step();
      opt.zeroGrad();
    }
    return losses;
  };
  assert.deepEqual(run(), run());
});
