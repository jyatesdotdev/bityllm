// Data-parallel trainer across worker_threads (M6-lite, DESIGN §18/§21-M6).
//
// One step = one generation:
//   main: bump ctrl[0] ─▶ workers: forward/backward on own micro-batch into
//   own grad slab ─▶ main: average slabs → clip → AdamW into the SHARED
//   weight buffer ─▶ next generation. One sync point per step, zero copies.
//
// batchSize is the GLOBAL batch: it is split into workers micro-batches, so a
// step performs (statistically) the same update as single-thread train() —
// just computed on all P-cores at once.

import { Worker } from "node:worker_threads";
import os from "node:os";
import { GPT } from "./nn/gpt.ts";
import { RNG } from "./core/rng.ts";
import { Dataset } from "./data/dataset.ts";
import { noGrad } from "./core/tensor.ts";
import { AdamW, clipGradNorm, cosineLR } from "./optim/adamw.ts";
import type { TrainConfig } from "./train.ts";

export interface ParallelTrainConfig extends TrainConfig {
  workers?: number;  // default: min(8, cores - 2)
  valSplit?: number; // default 0.05 (must match what workers use)
}

async function waitFor(i32: Int32Array, idx: number, expected: number, errs: Promise<never>): Promise<void> {
  for (;;) {
    const v = Atomics.load(i32, idx);
    if (v === expected) return;
    // Wait on the exact value we observed: if it changes between the load and
    // this call, waitAsync returns "not-equal" and we re-check — waiting on a
    // fresh load instead would race a flag flip and deadlock (lost wakeup).
    const r = Atomics.waitAsync(i32, idx, v, 60_000);
    const res = r.async ? await Promise.race([r.value, errs]) : r.value;
    if (res === "timed-out") throw new Error("parallel trainer: worker stalled (60s)");
  }
}

export async function trainParallel(
  model: GPT,
  tokens: Int32Array,
  cfg: ParallelTrainConfig,
): Promise<{ finalLoss: number }> {
  const W = cfg.workers ?? Math.max(1, Math.min(8, os.cpus().length - 2));
  const B = cfg.batchSize, T = cfg.blockSize;
  const microBatch = Math.max(1, Math.floor(B / W));
  const valSplit = cfg.valSplit ?? 0.05;
  const named = model.namedParameters();
  const total = named.reduce((n, [, p]) => n + p.size, 0);

  // --- shared memory ---------------------------------------------------------
  const weightsSab = new SharedArrayBuffer(total * 4);
  const gradsSab = new SharedArrayBuffer(W * total * 4);
  const lossesSab = new SharedArrayBuffer(W * 8);
  const ctrlSab = new SharedArrayBuffer((1 + W) * 4);
  const tokensSab = new SharedArrayBuffer(tokens.length * 4);
  new Int32Array(tokensSab).set(tokens);

  // move the (already-initialized) main model weights into the shared buffer
  let off = 0;
  for (const [, p] of named) {
    const view = new Float32Array(weightsSab, off * 4, p.size);
    view.set(p.data as Float32Array);
    p.data = view;
    off += p.size;
  }
  // main-side grad = views into one averaged buffer (what clip + AdamW read)
  const avg = new Float32Array(total);
  off = 0;
  for (const [, p] of named) {
    p.grad = avg.subarray(off, off + p.size);
    off += p.size;
  }
  const slabs = Array.from({ length: W }, (_, w) => new Float32Array(gradsSab, w * total * 4, total));
  const losses = new Float64Array(lossesSab);
  const ctrl = new Int32Array(ctrlSab);

  // --- workers ---------------------------------------------------------------
  let failWorker: (e: Error) => void;
  const errs = new Promise<never>((_, reject) => (failWorker = reject));
  const workers = Array.from({ length: W }, (_, workerId) => {
    const w = new Worker(new URL("./train-worker.ts", import.meta.url), {
      workerData: {
        cfg: model.cfg, weights: weightsSab, grads: gradsSab, losses: lossesSab,
        ctrl: ctrlSab, tokens: tokensSab, tokenCount: tokens.length,
        valSplit, workerId, microBatch, blockSize: T, seed: cfg.seed ?? 1337,
      },
    });
    w.on("error", (e: Error) => failWorker(new Error(`worker ${workerId}: ${e.message}`)));
    w.on("exit", (code) => {
      if (code !== 0 && Atomics.load(ctrl, 0) !== -1) failWorker(new Error(`worker ${workerId} died (${code})`));
    });
    return w;
  });

  // --- optimizer + eval ------------------------------------------------------
  const { decay, noDecay } = model.paramGroups();
  const opt = new AdamW(decay, noDecay, { lr: cfg.lr, weightDecay: cfg.weightDecay ?? 0.1 });
  const lrOpts = { lr: cfg.lr, warmup: cfg.warmup ?? 100, total: cfg.steps, minRatio: cfg.minLrRatio ?? 0.1 };
  const clip = cfg.clip ?? 1.0;
  const data = new Dataset(tokens, valSplit);

  const estimate = (split: "train" | "val", iters: number, seed: number): number =>
    noGrad(() => {
      const rng = new RNG(seed ^ 0xe7a1);
      let sum = 0;
      for (let i = 0; i < iters; i++) {
        const b = data.getBatch(split, Math.max(2, microBatch), T, rng);
        sum += model.loss(b.x, b.y, Math.max(2, microBatch), T).item();
      }
      return sum / iters;
    });

  let last = NaN;
  let tickTokens = 0;
  let tickStart = performance.now();

  try {
    for (let step = 0; step < cfg.steps; step++) {
      const gen = step + 1;
      Atomics.store(ctrl, 0, gen);
      Atomics.notify(ctrl, 0);
      for (let w = 0; w < W; w++) await waitFor(ctrl, 1 + w, gen, errs);

      // Data-parallel gradient reduction. Each of the W workers computed the
      // gradient of the MEAN loss over its own microBatch. Gradients are linear,
      // so averaging the W per-worker mean-gradients equals the gradient of the
      // mean loss over all W·microBatch samples — i.e. one big batch, computed in
      // parallel. (This is exactly what multi-GPU all-reduce does at scale.)
      for (let i = 0; i < total; i++) {
        let s = 0;
        for (let w = 0; w < W; w++) s += slabs[w][i];
        avg[i] = s / W;
      }
      clipGradNorm(opt.params, clip);
      opt.step(cosineLR(step, lrOpts));

      last = losses.reduce((a, b) => a + b, 0) / W;
      tickTokens += W * microBatch * T;

      const logEvery = cfg.logEvery ?? 10;
      if (cfg.onLog && gen % logEvery === 0) {
        const dt = (performance.now() - tickStart) / 1000;
        cfg.onLog({ step: gen, loss: last, lr: cosineLR(step, lrOpts), tokPerSec: tickTokens / dt });
        tickTokens = 0;
        tickStart = performance.now();
      }
      if (cfg.onEval && cfg.evalEvery && gen % cfg.evalEvery === 0) {
        model.train(false);
        const iters = cfg.evalIters ?? 10;
        cfg.onEval({ step: gen, trainLoss: estimate("train", iters, gen), valLoss: estimate("val", iters, gen) });
        model.train(true);
        tickTokens = 0;
        tickStart = performance.now();
      }
    }
  } finally {
    Atomics.store(ctrl, 0, -1);
    Atomics.notify(ctrl, 0);
    await Promise.allSettled(workers.map((w) => w.terminate()));
  }
  return { finalLoss: last };
}
