// Bisect the step-2011 NaN: replay the exact training trajectory (deterministic
// — same seed, same draws, same command stream), then run the poison step's
// FORWARD ONLY and scan activations in order. The first non-finite buffer
// names the guilty kernel.
//
//   deno run --allow-read bench/gpu-bisect.ts

import { readFileSync } from "node:fs";
import { CharTokenizer } from "../src/tokenizer/char.ts";
import { Dataset } from "../src/data/dataset.ts";
import { RNG } from "../src/core/rng.ts";
import { GPT } from "../src/nn/gpt.ts";
import { cosineLR } from "../src/optim/adamw.ts";
import { GPUTrainer } from "../src/gpu/trainer.ts";

const text = readFileSync("corpus/data/bity.corpus.txt", "utf8");
const tok = CharTokenizer.fromText(text);
const data = new Dataset(tok.encode(text), 0.05);
const cfg = { vocabSize: tok.size, blockSize: 128, nLayer: 6, nHead: 6, nEmbd: 192 };
const model = new GPT(cfg, new RNG(1337));
const rng = new RNG(1337);
const B = 32, T = 128;
const lrOpts = { lr: 1e-3, warmup: 200, total: 12000, minRatio: 0.1 };

const gpu = await GPUTrainer.create(model, { batchSize: B, weightDecay: 0.1, clip: 1.0 });
console.log("replaying 2010 steps (deterministic trajectory)...");
const t0 = performance.now();
for (let step = 1; step <= 2010; step++) {
  const b = data.getBatch("train", B, T, rng);
  await gpu.step(b.x, b.y, cosineLR(step - 1, lrOpts));
  if (step === 1200) for (let i = 0; i < 5; i++) data.getBatch("val", B, T, rng); // eval draws
  if (step % 500 === 0) console.log(`  ${step} (${((performance.now() - t0) / 60000).toFixed(1)} min) loss ${await gpu.readLoss()}`);
}
const check = await gpu.readLoss();
console.log(`state after 2010: loss ${check} (expect ~0.97, finite)`);

const poison = data.getBatch("train", B, T, rng);
console.log("\nrunning step-2011 FORWARD ONLY on the poison batch...");
const loss = await gpu.evalLoss(poison.x, poison.y);
console.log(`fwd-only loss: ${loss}\n`);

console.log("activation scan (forward order):");
for (const a of await gpu.scanActivations()) {
  const flag = a.bad > 0 ? "  ← FIRST BAD" : "";
  console.log(`  ${a.name.padEnd(14)} max ${a.max.toExponential(3).padStart(11)}  bad ${String(a.bad).padStart(7)}  firstIdx ${a.firstBadIdx}${flag}`);
  if (a.bad > 0) break; // first corruption found — everything after is downstream
}
gpu.destroy();
