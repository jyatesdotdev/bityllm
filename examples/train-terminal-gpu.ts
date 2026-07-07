// Train on the Apple GPU via WebGPU (Deno):
//
//   deno run --allow-read --allow-write examples/train-terminal-gpu.ts \
//     --steps 12000 --batch 32 --layers 6 --heads 6 --dim 192 --out models/terminal-micro.bity
//
// Same corpus, same checkpoint format, same eval/preview flow as the CPU
// trainer — only the engine differs (DESIGN M6).

import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { GPT, CharTokenizer, Dataset, RNG, cosineLR, generate, serialize } from "../src/index.ts";
import { GPUTrainer } from "../src/gpu/trainer.ts";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const args = (globalThis as unknown as { Deno: { args: string[] } }).Deno?.args ?? process.argv.slice(2);
const opt = (name: string, def: string): string => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 && args[i + 1] ? args[i + 1] : def;
};

const STEPS = parseInt(opt("steps", "12000"), 10);
const BATCH = parseInt(opt("batch", "32"), 10);
const BLOCK = parseInt(opt("block", "128"), 10);
const LR = parseFloat(opt("lr", "1e-3"));
const LAYERS = parseInt(opt("layers", "6"), 10);
const HEADS = parseInt(opt("heads", "6"), 10);
const DIM = parseInt(opt("dim", "192"), 10);
const DATA = opt("data", "corpus/data/bity.corpus.txt");
const OUT = opt("out", "models/terminal-micro.bity");
const SEED = parseInt(opt("seed", "1337"), 10);

const text = readFileSync(resolve(ROOT, DATA), "utf8");
const tok = CharTokenizer.fromText(text);
const data = new Dataset(tok.encode(text), 0.05);
const cfg = { vocabSize: tok.size, blockSize: BLOCK, nLayer: LAYERS, nHead: HEADS, nEmbd: DIM };
const model = new GPT(cfg, new RNG(SEED));
const rng = new RNG(SEED);
const PROMPT = "guest@bity:~$ ";

console.log(`corpus : ${(text.length / 1024 / 1024).toFixed(2)} MB, vocab ${tok.size}`);
console.log(`model  : ${LAYERS}L/${HEADS}H/${DIM}d, block ${BLOCK} → ${model.paramCount().toLocaleString()} params`);
console.log(`train  : ${STEPS} steps, batch ${BATCH}, lr ${LR} — engine: WebGPU\n`);

const gpu = await GPUTrainer.create(model, { batchSize: BATCH, weightDecay: 0.1, clip: 1.0 });
const lrOpts = { lr: LR, warmup: Math.min(200, Math.floor(STEPS / 10)), total: STEPS, minRatio: 0.1 };
const evalEvery = Math.max(100, Math.floor(STEPS / 10));
const t0 = performance.now();
let tick = t0;

for (let step = 1; step <= STEPS; step++) {
  const b = data.getBatch("train", BATCH, BLOCK, rng);
  await gpu.step(b.x, b.y, cosineLR(step - 1, lrOpts));

  // NaN tripwire: per-step loss check; on first non-finite, dump per-tensor
  // diagnostics (which weights/grads broke, and how) and stop.
  const loss = await gpu.readLoss();
  const scanFrom = parseInt(opt("scanFrom", "0"), 10);
  if (scanFrom > 0 && step >= scanFrom) {
    console.log(`step ${step}  loss ${loss}`);
    const sus = (await gpu.scan()).filter((s) => s.wBad || s.gBad || s.wMax > 50 || s.gMax > 50);
    for (const s of sus)
      console.log(`  ${s.name.padEnd(18)} |w|max ${s.wMax.toExponential(2)} badW ${s.wBad}  |g|max ${s.gMax.toExponential(2)} badG ${s.gBad}`);
    if (sus.some((s) => s.wBad || s.gBad)) {
      gpu.destroy();
      throw new Error(`first corruption at step ${step}`);
    }
  }
  if (!Number.isFinite(loss)) {
    console.log(`\n!!! non-finite loss at step ${step} — scanning tensors...`);
    for (const s of await gpu.scan()) {
      if (s.wBad > 0 || s.gBad > 0 || s.wMax > 100 || s.gMax > 100) {
        console.log(`  ${s.name.padEnd(18)} |w|max ${s.wMax.toExponential(2)} badW ${s.wBad}  |g|max ${s.gMax.toExponential(2)} badG ${s.gBad}`);
      }
    }
    gpu.destroy();
    throw new Error(`NaN at step ${step}`);
  }

  if (step % 20 === 0) {
    const dt = (performance.now() - tick) / 1000;
    console.log(`step ${String(step).padStart(5)}  loss ${loss.toFixed(4)}  ${Math.round((20 * BATCH * BLOCK) / dt)} tok/s`);
    tick = performance.now();
  }
  if (step % evalEvery === 0 || step === STEPS) {
    let val = 0;
    const vi = 5;
    for (let i = 0; i < vi; i++) {
      const vb = data.getBatch("val", BATCH, BLOCK, rng);
      val += await gpu.evalLoss(vb.x, vb.y);
    }
    console.log(`\n== eval @ ${step}: val ${(val / vi).toFixed(4)} ==`);
    await gpu.readWeights(); // GPU → CPU model for preview + checkpoint
    const preview = generate(model, tok, PROMPT + "ls\n", {
      maxNewTokens: 100, temperature: 0.8, topK: 40, seed: step, stop: [PROMPT],
    });
    console.log("-- sample -----------------------------------");
    console.log(PROMPT + "ls\n" + preview);
    console.log("---------------------------------------------\n");
    mkdirSync(resolve(ROOT, dirname(OUT)), { recursive: true });
    writeFileSync(resolve(ROOT, OUT), serialize(model, tok, { step }));
    tick = performance.now();
  }
}

console.log(`done in ${((performance.now() - t0) / 1000 / 60).toFixed(1)} min → ${OUT}`);
gpu.destroy();
