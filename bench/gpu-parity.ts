// GPU↔CPU parity gate (run under Deno):
//   deno run --allow-read bench/gpu-parity.ts
//
// 1. loss parity: GPU forward CE == CPU forward CE
// 2. gradient parity: every parameter's GPU grad == CPU autograd grad
// 3. training parity: 5 full steps (clip + AdamW) keep weights aligned
//
// Tolerances are f32-accumulation-order realistic, not exact-bitwise.

import { RNG } from "../src/core/rng.ts";
import { GPT } from "../src/nn/gpt.ts";
import { AdamW, clipGradNorm } from "../src/optim/adamw.ts";
import { GPUTrainer } from "../src/gpu/trainer.ts";

const cfg = { vocabSize: 13, blockSize: 8, nLayer: 2, nHead: 2, nEmbd: 16 };
const B = 2, T = 8;

const mk = (): { model: GPT; x: Int32Array; y: Int32Array } => {
  const rng = new RNG(77);
  const model = new GPT(cfg, rng);
  const x = new Int32Array(B * T), y = new Int32Array(B * T);
  for (let i = 0; i < B * T; i++) {
    x[i] = rng.randint(0, cfg.vocabSize);
    y[i] = rng.randint(0, cfg.vocabSize);
  }
  return { model, x, y };
};

const rel = (a: ArrayLike<number>, b: ArrayLike<number>): number => {
  let worst = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    const m = Math.max(Math.abs(a[i]), Math.abs(b[i]), 1e-4);
    worst = Math.max(worst, d / m);
  }
  return worst;
};

let failures = 0;
const check = (name: string, v: number, tol: number): void => {
  const ok = v <= tol;
  if (!ok) failures++;
  console.log(`  ${ok ? "✓" : "✗"} ${name}: ${v.toExponential(2)} (tol ${tol})`);
};

// ---- 1+2: loss + gradient parity ----
{
  const { model, x, y } = mk();
  const cpuLossT = model.loss(x, y, B, T);
  const cpuLoss = cpuLossT.item();
  cpuLossT.backward();

  const { model: gm } = mk(); // identical weights (same seed)
  const gpu = await GPUTrainer.create(gm, { batchSize: B, clip: 1e30, weightDecay: 0 });
  await gpu.evalLoss(x, y);
  const gpuLoss = await gpu.readLoss();

  console.log("loss parity:");
  check("loss", Math.abs(cpuLoss - gpuLoss) / cpuLoss, 2e-4);

  // one grad-producing step with lr=0 (weights unchanged, grads populated)
  await gpu.step(x, y, 0);
  console.log("gradient parity (all parameters):");
  let worstName = "", worst = 0;
  for (const [name, t] of model.namedParameters()) {
    const g = await gpu.readGrad(name, t.size);
    const r = rel(g, t.grad!);
    if (r > worst) {
      worst = r;
      worstName = name;
    }
  }
  check(`worst tensor (${worstName})`, worst, 5e-3);
  gpu.destroy();
}

// ---- 3: five full optimizer steps stay aligned ----
{
  const { model: cm, x, y } = mk();
  const { decay, noDecay } = cm.paramGroups();
  const opt = new AdamW(decay, noDecay, { lr: 1e-3, weightDecay: 0.1 });
  for (let s = 0; s < 5; s++) {
    const loss = cm.loss(x, y, B, T);
    loss.backward();
    clipGradNorm(opt.params, 1.0);
    opt.step();
    opt.zeroGrad();
  }

  const { model: gm2 } = mk();
  const gpu = await GPUTrainer.create(gm2, { batchSize: B, clip: 1.0, weightDecay: 0.1 });
  for (let s = 0; s < 5; s++) await gpu.step(x, y, 1e-3);
  await gpu.readWeights();

  console.log("5-step training parity:");
  let worstName = "", worst = 0;
  for (const [name, t] of cm.namedParameters()) {
    const gpuT = gm2.namedParameters().find(([n]) => n === name)![1];
    const r = rel(gpuT.data, t.data);
    if (r > worst) {
      worst = r;
      worstName = name;
    }
  }
  check(`worst tensor (${worstName})`, worst, 2e-2);
  gpu.destroy();
}

console.log(failures === 0 ? "\nPARITY: ALL GREEN" : `\nPARITY: ${failures} FAILURES`);
if (failures > 0) {
  // Deno exits nonzero so CI-style use catches it
  (globalThis as { Deno?: { exit(code: number): void } }).Deno?.exit(1);
}
