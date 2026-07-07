// build.mjs — assemble the training corpus (DESIGN §2.3), v2:
//   ~70% synthetic (per-binary generators + STATEFUL session blocks where
//     mkdir/touch/rm genuinely change what later ls shows)
//   ~30% real captures — now SHUFFLED with synthetic blocks before writing,
//     so the tail val split is representative of the whole mix.
//
//   node corpus/build.mjs [--synth-mb 9] [--seed 1234]

import { readFileSync, writeFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { GENERATORS, BLOCK_GENERATORS, RNG, PROMPT } from "./generators/index.mjs";

const DATA = resolve(dirname(fileURLToPath(import.meta.url)), "data");
const args = process.argv.slice(2);
const opt = (name, def) => {
  const i = args.indexOf(name);
  return i >= 0 && args[i + 1] ? args[i + 1] : def;
};
const SYNTH_TARGET = Math.round(parseFloat(opt("--synth-mb", "9")) * 1024 * 1024);
const rng = new RNG(parseInt(opt("--seed", "1234"), 10));

const fmt = (recs) =>
  recs.map(({ cmd, output, prompt }) =>
    (prompt ?? PROMPT) + cmd + "\n" + (output === "" ? "" : output.endsWith("\n") ? output : output + "\n")).join("");

// ---- synthesize sessions ------------------------------------------------------
const iters = GENERATORS.map((g) => ({ ...g, it: g.gen(rng), count: 0 }));
const blocks = BLOCK_GENERATORS.map((g) => ({ ...g, count: 0 }));
const totalW = iters.reduce((s, g) => s + g.weight, 0) + blocks.reduce((s, g) => s + g.weight, 0);

const sessions = [];
let synthBytes = 0;
while (synthBytes < SYNTH_TARGET) {
  let r = rng.random() * totalW;
  let text = null;
  for (const b of blocks) {
    r -= b.weight;
    if (r <= 0) {
      const recs = b.block(rng);
      b.count += recs.length;
      text = fmt(recs);
      break;
    }
  }
  if (text === null) {
    // plain session: a few records from weighted single-record generators
    const n = 1 + Math.floor(rng.random() * 6);
    const recs = [];
    for (let i = 0; i < n; i++) {
      let r2 = rng.random() * iters.reduce((s, g) => s + g.weight, 0);
      let g = iters[iters.length - 1];
      for (const cand of iters) {
        r2 -= cand.weight;
        if (r2 <= 0) { g = cand; break; }
      }
      recs.push(g.it.next().value);
      g.count++;
    }
    text = fmt(recs);
  }
  sessions.push(text);
  synthBytes += Buffer.byteLength(text);
}

// ---- real captures, chunked at prompt boundaries -------------------------------
const REAL = ["debian.corpus.txt", "debian-vm.corpus.txt", "real-dmesg.corpus.txt"];
const realChunks = [];
let realBytes = 0;
for (const f of REAL) {
  let t;
  try { t = readFileSync(resolve(DATA, f), "utf8"); }
  catch { console.log(`(missing: ${f})`); continue; }
  realBytes += Buffer.byteLength(t);
  const lines = t.split("\n");
  let chunk = [];
  let prompts = 0;
  for (const line of lines) {
    if (line.startsWith(PROMPT) && prompts >= 8) {
      realChunks.push(chunk.join("\n") + "\n");
      chunk = [];
      prompts = 0;
    }
    if (line.startsWith(PROMPT)) prompts++;
    chunk.push(line);
  }
  if (chunk.length) realChunks.push(chunk.join("\n") + "\n");
}

// ---- shuffle everything together ----------------------------------------------
const all = [...sessions, ...realChunks];
for (let i = all.length - 1; i > 0; i--) {
  const j = Math.floor(rng.random() * (i + 1));
  [all[i], all[j]] = [all[j], all[i]];
}
const out = all.join("");
writeFileSync(resolve(DATA, "bity.corpus.txt"), out);

console.log(`synthetic : ${(synthBytes / 1024).toFixed(0)}KB in ${sessions.length} sessions`);
console.log(`real      : ${(realBytes / 1024).toFixed(0)}KB in ${realChunks.length} chunks (shuffled in)`);
console.log(`total     : ${(Buffer.byteLength(out) / 1024).toFixed(0)}KB, vocab ${new Set(out).size}`);
console.log(`records   : ${iters.map((g) => `${g.name}=${g.count}`).join("  ")}  ${blocks.map((g) => `${g.name}=${g.count}`).join("  ")}`);
console.log(`synthetic share: ${((synthBytes / Buffer.byteLength(out)) * 100).toFixed(0)}%`);
