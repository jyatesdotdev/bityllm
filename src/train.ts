// Training loop (DESIGN §13): batch → forward → CE → backward → clip → AdamW
// step → cosine LR. Periodic eval under noGrad + preview/checkpoint hooks.

import { noGrad } from "./core/tensor.ts";
import { RNG } from "./core/rng.ts";
import type { GPT } from "./nn/gpt.ts";
import type { Dataset } from "./data/dataset.ts";
import { AdamW, clipGradNorm, cosineLR } from "./optim/adamw.ts";

export interface TrainConfig {
  steps: number;
  batchSize: number;
  blockSize: number;
  lr: number;
  warmup?: number;
  minLrRatio?: number;
  weightDecay?: number;
  clip?: number;
  evalEvery?: number;
  evalIters?: number;
  seed?: number;
  logEvery?: number;
  onLog?: (info: { step: number; loss: number; lr: number; tokPerSec: number }) => void;
  onEval?: (info: { step: number; trainLoss: number; valLoss: number }) => void;
}

export function train(model: GPT, data: Dataset, cfg: TrainConfig): { finalLoss: number } {
  const B = cfg.batchSize, T = cfg.blockSize;
  const rng = new RNG(cfg.seed ?? 1337);
  const { decay, noDecay } = model.paramGroups();
  const opt = new AdamW(decay, noDecay, { lr: cfg.lr, weightDecay: cfg.weightDecay ?? 0.1 });
  const clip = cfg.clip ?? 1.0;
  const lrOpts = { lr: cfg.lr, warmup: cfg.warmup ?? 100, total: cfg.steps, minRatio: cfg.minLrRatio ?? 0.1 };

  const estimate = (split: "train" | "val", iters: number): number =>
    noGrad(() => {
      let sum = 0;
      for (let i = 0; i < iters; i++) {
        const b = data.getBatch(split, B, T, rng);
        sum += model.loss(b.x, b.y, B, T).item();
      }
      return sum / iters;
    });

  model.train(true);
  let last = NaN;
  let tickTokens = 0;
  let tickStart = performance.now();

  for (let step = 0; step < cfg.steps; step++) {
    const batch = data.getBatch("train", B, T, rng);
    // THE five-step training step — identical for every gradient-based model,
    // only `getBatch` and `loss` are task-specific (see LEARNING.md §2, §13):
    const loss = model.loss(batch.x, batch.y, B, T); // 1. forward → scalar CE loss
    last = loss.item();
    loss.backward();                                 // 2. reverse-mode autograd fills every .grad
    clipGradNorm(opt.params, clip);                  // 3. rescale grads if their global norm is too big
    opt.step(cosineLR(step, lrOpts));                // 4. AdamW nudges each weight down its gradient
    opt.zeroGrad();                                  // 5. clear grads for the next step (they accumulate)
    tickTokens += B * T;

    const logEvery = cfg.logEvery ?? 10;
    if (cfg.onLog && (step + 1) % logEvery === 0) {
      const dt = (performance.now() - tickStart) / 1000;
      cfg.onLog({ step: step + 1, loss: last, lr: cosineLR(step, lrOpts), tokPerSec: tickTokens / dt });
      tickTokens = 0;
      tickStart = performance.now();
    }
    if (cfg.onEval && cfg.evalEvery && (step + 1) % cfg.evalEvery === 0) {
      model.train(false);
      const iters = cfg.evalIters ?? 10;
      cfg.onEval({ step: step + 1, trainLoss: estimate("train", iters), valLoss: estimate("val", iters) });
      model.train(true);
      tickStart = performance.now(); // don't count eval time against tok/s
      tickTokens = 0;
    }
  }
  return { finalLoss: last };
}
