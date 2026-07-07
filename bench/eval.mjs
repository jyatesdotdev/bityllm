// Comprehensive scored evaluation of a bity checkpoint. Multiple seeds per case;
// pass-rate where the answer is checkable, copy-fidelity where it's fuzzy.
import { readFileSync } from "node:fs";
import { deserialize } from "./src/io/checkpoint.ts";
import { InferenceSession } from "./src/infer/session.ts";

const path = process.argv[2] ?? "models/terminal-mini-v7.bity";
const { model, tok, step } = deserialize(readFileSync(path));
console.log(`\n=== ${path} @ step ${step} (${model.paramCount().toLocaleString()} params) ===\n`);

const SEEDS = [1, 2, 3, 4, 5, 6, 7, 8];

// cwd tracker mirroring Shell.applyCd, so location cases use the right prompts
function promptChain(cmds) {
  let cwd = "~";
  const out = [];
  for (const c of cmds) {
    out.push([`guest@bity:${cwd}$ `, c]);
    const a = c.startsWith("cd") ? c.slice(2).trim() : null;
    if (a !== null) {
      if (!a || a === "~") cwd = "~";
      else if (a === "..") cwd = cwd.includes("/") ? cwd.slice(0, cwd.lastIndexOf("/")) || "~" : "~";
      else cwd = (cwd === "~" ? "~" : cwd) + "/" + a.replace(/\/+$/, "");
    }
  }
  return { chain: out, prompt: `guest@bity:${cwd}$ `, cwd };
}

function gen(setups, probe, seed, prompt) {
  const s = new InferenceSession(model, tok);
  for (const [p, c] of setups) s.feed(p + c + "\n");
  s.feed(prompt + probe + "\n");
  let out = "";
  for (const ch of s.stream({ maxNewTokens: 100, temperature: 0.3, topK: 10, stop: ["guest@bity:"], seed }))
    out += ch;
  return out.replace(/guest@bity:.*$/s, "").replace(/\n+$/, "");
}

// checkable case: setupCmds run in home, probe from current cwd, expect(output)->bool
function scored(name, cmds, probe, expect) {
  const { chain, prompt } = promptChain(cmds);
  let pass = 0;
  let sample = "";
  for (const seed of SEEDS) {
    const o = gen(chain, probe, seed, prompt);
    if (seed === SEEDS[0]) sample = o.split("\n").join("\\n").slice(0, 64);
    if (expect(o)) pass++;
  }
  const pct = Math.round((100 * pass) / SEEDS.length);
  const bar = pass === SEEDS.length ? "✅" : pass === 0 ? "❌" : "🟡";
  console.log(`${bar} ${String(pct).padStart(3)}%  ${name.padEnd(34)} e.g. "${sample}"`);
  return pct;
}

const has = (sub) => (o) => o.toLowerCase().includes(sub.toLowerCase());
const eq = (v) => (o) => o.trim() === v;

console.log("— LOCATION STATE (v7 headline) —");
scored("cd projects → pwd", ["cd projects"], "pwd", eq("/home/guest/projects"));
scored("cd projects → ls (dir contents)", ["cd projects"], "ls", (o) => has("README")(o) || has("src")(o));
scored("mkdir zork; cd zork → pwd", ["mkdir zork", "cd zork"], "pwd", eq("/home/guest/zork"));
scored("cd a; cd b → pwd (nested)", ["cd docs", "cd sub"], "pwd", eq("/home/guest/docs/sub"));

console.log("\n— FILE CONTENT (the ceiling) —");
scored("echo hello > x.txt → cat", ["echo hello > x.txt"], "cat x.txt", has("hello"));
scored("echo WORD > f.log → cat (1 word)", ["echo zqwlk > f.log"], "cat f.log", has("zqwlk"));
scored("echo a b c > f.txt → cat (3 words)", ["echo alpha beta gamma > f.txt"], "cat f.txt",
  (o) => ["alpha", "beta", "gamma"].filter((w) => has(w)(o)).length);
scored("  ↑ all 3 words present", ["echo alpha beta gamma > f.txt"], "cat f.txt",
  (o) => ["alpha", "beta", "gamma"].every((w) => has(w)(o)));
scored("append → wc -l counts 2", ["echo a > f.txt", "echo b >> f.txt"], "wc -l f.txt", has("2 f.txt"));

console.log("\n— REGRESSIONS (must still hold) —");
scored("mkdir flowers → ls shows it", ["mkdir flowers"], "ls", has("flowers"));
scored("touch e.txt → cat (empty)", ["touch e.txt"], "cat e.txt", (o) => o.trim() === "");
scored("rm todo.md → cat (ENOENT)", ["rm todo.md"], "cat todo.md", has("No such file"));
scored("mv a.txt z.txt → ls shows z", ["touch a.txt", "mv a.txt z.txt"], "ls", (o) => has("z.txt")(o) && !has("a.txt")(o));
scored("cat dream.csv (uncreated → csv)", [], "cat dream.csv", has("id,name"));
scored("which git", [], "which git", eq("/usr/bin/git"));
scored("echo random (copy circuit)", [], "echo xq7ztk", has("xq7ztk"));
scored("cowsay hello (fun intact)", [], "cowsay hello", (o) => has("< hello >")(o) && has("^__^")(o));

console.log("");
