// Data-parallel training worker (M6-lite). Runs the existing, grad-checked
// model code unchanged on its own micro-batch each generation:
//
//   weights : Float32Array views on a SharedArrayBuffer (written by main only)
//   grads   : this worker's slab of a shared grad buffer (read by main only)
//   ctrl    : Int32Array — ctrl[0] is the generation counter (-1 = shutdown),
//             ctrl[1+id] is this worker's completed generation.
//
// Plain stores followed by Atomics.store/notify give the main thread a
// happens-before edge, so it always observes completed grad writes.

import { workerData } from "node:worker_threads";
import { GPT } from "./nn/gpt.ts";
import type { GPTConfig } from "./nn/gpt.ts";
import { RNG } from "./core/rng.ts";
import { Dataset } from "./data/dataset.ts";

interface WorkerInit {
  cfg: GPTConfig;
  weights: SharedArrayBuffer;
  grads: SharedArrayBuffer;
  losses: SharedArrayBuffer;
  ctrl: SharedArrayBuffer;
  tokens: SharedArrayBuffer;
  tokenCount: number;
  valSplit: number;
  workerId: number;
  microBatch: number;
  blockSize: number;
  seed: number;
}

const init = workerData as WorkerInit;
const { cfg, workerId, microBatch: B, blockSize: T } = init;

// Same construction as main (structure only — weights come from the SAB).
const model = new GPT(cfg, new RNG(0));
const named = model.namedParameters();

let off = 0;
for (const [, p] of named) {
  p.data = new Float32Array(init.weights, off * 4, p.size);
  off += p.size;
}
const total = off;
let goff = workerId * total;
for (const [, p] of named) {
  p.grad = new Float32Array(init.grads, goff * 4, p.size);
  goff += p.size;
}

const lossArr = new Float64Array(init.losses);
const ctrl = new Int32Array(init.ctrl);
const data = new Dataset(new Int32Array(init.tokens, 0, init.tokenCount), init.valSplit);
const rng = new RNG(init.seed + 7919 * (workerId + 1));

model.train(true);
let gen = 0;
for (;;) {
  // wait for the next generation (or shutdown)
  for (;;) {
    const g = Atomics.load(ctrl, 0);
    if (g === -1) process.exit(0);
    if (g !== gen) {
      gen = g;
      break;
    }
    Atomics.wait(ctrl, 0, g);
  }

  for (const p of model.parameters()) p.zeroGrad();
  const batch = data.getBatch("train", B, T, rng);
  const loss = model.loss(batch.x, batch.y, B, T);
  lossArr[workerId] = loss.item();
  loss.backward(); // accumulates straight into this worker's shared grad slab

  Atomics.store(ctrl, 1 + workerId, gen);
  Atomics.notify(ctrl, 1 + workerId);
}
