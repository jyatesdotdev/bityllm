// build.mjs — assemble the training corpus (DESIGN §2.3), v3 (hybrid):
//   The programmatic CORE (src/infer/{vfs,coreutils,shell-exec}.ts) now runs all
//   deterministic FS/text/identity commands as real code, so the model only ever
//   dreams the DREAMED set + graceful `command not found`. The corpus is
//   rebalanced to match:
//     - synthetic (net/git/fun/unknown only) tops up the dreamed set + fallback
//     - real captures are FILTERED to drop CORE-command records (ls/cat/pwd/…),
//       which the model no longer needs to learn — pure dilution otherwise.
//   Everything is shuffled before writing so the tail val split is representative.
//
//   node corpus/build.mjs [--synth-mb 9] [--seed 1234]

import { readFileSync, writeFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { GENERATORS, BLOCK_GENERATORS, RNG, PROMPT } from "./generators/index.mjs";

const DATA = resolve(dirname(fileURLToPath(import.meta.url)), "data");

// Commands handled by the programmatic CORE — filtered OUT of the real capture
// (the model never dreams these). Kept in sync with coreutils.ts CORE keys.
const CORE_COMMANDS = new Set([
  "pwd", "echo", "whoami", "id", "hostname", "groups", "arch", "uname", "uptime", "date",
  "true", "false", "clear", "which", "env", "printenv", "ls", "cat", "cd", "mkdir", "rmdir",
  "touch", "rm", "mv", "cp", "chmod", "wc", "head", "tail", "grep", "sort", "uniq", "rev", "tr", "nl", "seq",
]);
// a real-capture prompt line: "<user>@bity:<path>$ " (guest) or "…# " (root)
const REAL_PROMPT = /^[a-z_][a-z0-9_-]*@bity:\S*[$#] /;
const cmdHead = (line) => line.replace(REAL_PROMPT, "").trim().split(/[\s|;&<>]+/)[0];
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

// ---- real captures: parse into records, DROP CORE commands, re-chunk -----------
// Each record = a prompt line + its output lines (until the next prompt). Records
// whose command is now handled by the programmatic CORE are dropped — the model
// never dreams them, so training on them is pure dilution. Files with no prompts
// (e.g. raw dmesg) are kept whole.
const REAL = ["debian.corpus.txt", "debian-vm.corpus.txt", "real-dmesg.corpus.txt"];
const realChunks = [];
let realBytes = 0, realKept = 0, realDropped = 0;
for (const f of REAL) {
  let t;
  try { t = readFileSync(resolve(DATA, f), "utf8"); }
  catch { console.log(`(missing: ${f})`); continue; }
  realBytes += Buffer.byteLength(t);
  const lines = t.split("\n");
  // split into records at prompt boundaries
  const records = [];
  let preamble = [], cur = null;
  for (const line of lines) {
    if (REAL_PROMPT.test(line)) {
      if (cur) records.push(cur);
      cur = { head: cmdHead(line), lines: [line] };
    } else if (cur) cur.lines.push(line);
    else preamble.push(line);
  }
  if (cur) records.push(cur);
  if (!records.length) { realChunks.push(t.endsWith("\n") ? t : t + "\n"); continue; } // e.g. dmesg
  // keep non-CORE records, grouped ~8 per chunk (preamble rides the first chunk)
  let chunk = preamble.length ? [preamble.join("\n")] : [];
  let n = 0;
  for (const r of records) {
    if (r.head && CORE_COMMANDS.has(r.head)) { realDropped++; continue; }
    realKept++;
    chunk.push(r.lines.join("\n"));
    if (++n >= 8) { realChunks.push(chunk.join("\n") + "\n"); chunk = []; n = 0; }
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
console.log(`real      : ${(realBytes / 1024).toFixed(0)}KB read → ${realChunks.length} chunks (kept ${realKept} records, dropped ${realDropped} CORE)`);
console.log(`total     : ${(Buffer.byteLength(out) / 1024).toFixed(0)}KB, vocab ${new Set(out).size}`);
console.log(`records   : ${iters.map((g) => `${g.name}=${g.count}`).join("  ")}${blocks.length ? "  " + blocks.map((g) => `${g.name}=${g.count}`).join("  ") : ""}`);
console.log(`synthetic share: ${((synthBytes / Buffer.byteLength(out)) * 100).toFixed(0)}%`);
