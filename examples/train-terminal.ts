// Train the nano GPT on the bity terminal corpus and save a checkpoint.
//
//   npm run train -- --steps 2000 --batch 8 --block 128
//   node examples/train-terminal.ts --steps 200 --out models/terminal.bity

import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { GPT, CharTokenizer, Dataset, RNG, train, trainParallel, generate, serialize } from "../src/index.ts";
import type { TrainConfig } from "../src/index.ts";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
const opt = (name: string, def: string): string => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 && args[i + 1] ? args[i + 1] : def;
};

const STEPS = parseInt(opt("steps", "2000"), 10);
const BATCH = parseInt(opt("batch", "8"), 10);
const BLOCK = parseInt(opt("block", "128"), 10);
const LR = parseFloat(opt("lr", "1e-3"));
const DATA = opt("data", "corpus/data/bity.corpus.txt");
const OUT = opt("out", "models/terminal.bity");
const SEED = parseInt(opt("seed", "1337"), 10);
const WORKERS = parseInt(opt("workers", "0"), 10); // 0 = single-thread
const LAYERS = parseInt(opt("layers", "3"), 10);
const HEADS = parseInt(opt("heads", "4"), 10);
const DIM = parseInt(opt("dim", "64"), 10);

const text = readFileSync(resolve(ROOT, DATA), "utf8");
const tok = CharTokenizer.fromText(text);
const tokens = tok.encode(text);
const data = new Dataset(tokens, 0.05);

// presets (DESIGN §9.4): nano 3/4/64 (default) · milli 6/4/128 · micro 6/6/192
const cfg = { vocabSize: tok.size, blockSize: BLOCK, nLayer: LAYERS, nHead: HEADS, nEmbd: DIM };
const model = new GPT(cfg, new RNG(SEED));

console.log(`corpus : ${(text.length / 1024 / 1024).toFixed(2)} MB, vocab ${tok.size}`);
console.log(`model  : ${cfg.nLayer}L/${cfg.nHead}H/${cfg.nEmbd}d, block ${BLOCK} → ${model.paramCount().toLocaleString()} params`);
console.log(`train  : ${STEPS} steps, batch ${BATCH}, lr ${LR}, seed ${SEED}${WORKERS > 0 ? `, ${WORKERS} workers` : ""}\n`);

const t0 = performance.now();
const PROMPT = "guest@bity:~$ ";

const trainCfg: TrainConfig = {
  steps: STEPS,
  batchSize: BATCH,
  blockSize: BLOCK,
  lr: LR,
  warmup: Math.min(100, Math.floor(STEPS / 10)),
  weightDecay: 0.1,
  clip: 1.0,
  seed: SEED,
  logEvery: 10,
  onLog: ({ step, loss, tokPerSec }) =>
    console.log(`step ${String(step).padStart(5)}  loss ${loss.toFixed(4)}  ${Math.round(tokPerSec)} tok/s`),
  evalEvery: Math.max(50, Math.floor(STEPS / 10)),
  evalIters: 5,
  onEval: ({ step, trainLoss, valLoss }) => {
    console.log(`\n== eval @ ${step}: train ${trainLoss.toFixed(4)}  val ${valLoss.toFixed(4)} ==`);
    const preview = generate(model, tok, PROMPT + "ls\n", {
      maxNewTokens: 120,
      temperature: 0.8,
      topK: 40,
      seed: step,
      stop: [PROMPT],
    });
    console.log("-- sample -----------------------------------");
    console.log(PROMPT + "ls\n" + preview);
    console.log("---------------------------------------------\n");
    // checkpoint at every eval
    mkdirSync(resolve(ROOT, dirname(OUT)), { recursive: true });
    writeFileSync(resolve(ROOT, OUT), serialize(model, tok, { step }));
  },
};

if (WORKERS > 0) await trainParallel(model, tokens, { ...trainCfg, workers: WORKERS, valSplit: 0.05 });
else train(model, data, trainCfg);

mkdirSync(resolve(ROOT, dirname(OUT)), { recursive: true });
writeFileSync(resolve(ROOT, OUT), serialize(model, tok, { step: STEPS }));
console.log(`\ndone in ${((performance.now() - t0) / 1000 / 60).toFixed(1)} min → ${OUT}`);
