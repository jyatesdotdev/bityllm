// GPU inference gate (Deno): the WebGPU session must agree with the CPU
// session on the same weights, then we measure the speedup on the real mini.
//
//   deno run --allow-read bench/gpu-infer.ts

import { readFileSync } from "node:fs";
import { deserialize } from "../src/io/checkpoint.ts";
import { InferenceSession } from "../src/infer/session.ts";
import { GPUInferenceSession } from "../src/gpu/session.ts";

const { model, tok } = deserialize(readFileSync("models/terminal-mini.bity"));
const P = "guest@bity:~$ ";
const PROMPT = P + "ls -la\n";

// ---- 1. logits parity after identical feed ----
const cpu = new InferenceSession(model, tok);
cpu.feed(PROMPT);
// @ts-expect-error private access for parity
const cpuLogits: Float32Array = cpu.last;

const gpu = await GPUInferenceSession.create(model, tok);
gpu.feed(PROMPT);
const gpuLogits = await gpu.logits();

let worst = 0;
for (let i = 0; i < cpuLogits.length; i++) {
  const d = Math.abs(cpuLogits[i] - gpuLogits[i]);
  const m = Math.max(Math.abs(cpuLogits[i]), Math.abs(gpuLogits[i]), 1e-3);
  worst = Math.max(worst, d / m);
}
console.log(`logits parity (CPU vs WebGPU): worst rel diff ${worst.toExponential(2)} ${worst < 2e-2 ? "✓" : "✗ FAIL"}`);

// ---- 2. identical greedy text over 120 tokens ----
const cpuText = (() => {
  const s = new InferenceSession(model, tok);
  s.feed(PROMPT);
  let out = "";
  for (const ch of s.stream({ maxNewTokens: 120, temperature: 0.001, topK: 1, seed: 7 })) out += ch;
  return out;
})();
const gpuText = await (async () => {
  const s = await GPUInferenceSession.create(model, tok);
  s.feed(PROMPT);
  let out = "";
  for await (const ch of s.stream({ maxNewTokens: 120, temperature: 0.001, topK: 1, seed: 7 })) out += ch;
  return out;
})();
console.log(`greedy text match: ${cpuText === gpuText ? "✓ identical" : `✗ diverged\n  cpu: ${JSON.stringify(cpuText.slice(0, 60))}\n  gpu: ${JSON.stringify(gpuText.slice(0, 60))}`}`);

// ---- 3. throughput ----
const time = async (label: string, fn: () => Promise<number>): Promise<void> => {
  const t0 = performance.now();
  const n = await fn();
  console.log(`${label}: ${(n / ((performance.now() - t0) / 1000)).toFixed(0)} tok/s`);
};
await time("cpu  mini", async () => {
  const s = new InferenceSession(model, tok);
  s.feed(PROMPT);
  let n = 0;
  for (const _ of s.stream({ maxNewTokens: 200, temperature: 0.8, seed: 3 })) n++;
  return n;
});
await time("gpu  mini", async () => {
  const s = await GPUInferenceSession.create(model, tok);
  s.feed(PROMPT);
  let n = 0;
  for await (const _ of s.stream({ maxNewTokens: 200, temperature: 0.8, seed: 3 })) n++;
  return n;
});
gpu.destroy();
