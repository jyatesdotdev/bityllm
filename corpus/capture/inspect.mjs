// inspect.mjs — quick quality report on a captured corpus.
import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const recs = readFileSync(resolve(HERE, "../data/debian.jsonl"), "utf8")
  .trim().split("\n").map((l) => JSON.parse(l));
const txt = readFileSync(resolve(HERE, "../data/debian.corpus.txt"), "utf8");

const chars = new Set(txt);
const nonAscii = [...chars].filter((c) => { const n = c.charCodeAt(0); return n > 126 || (n < 9); });
console.log("records:", recs.length, "| chars:", txt.length, "| vocab:", chars.size, "| non-ASCII glyphs:", nonAscii.length);
console.log("non-ASCII set:", JSON.stringify(nonAscii.join("")));
console.log("leftover ESC:", (txt.match(/\x1b/g) || []).length, "| NUL:", (txt.match(/\x00/g) || []).length);
console.log("exit!=0 records:", recs.filter((r) => r.exit !== 0).length);

const pick = (cat, n = 1, skip = 3) => recs.filter((r) => r.cat === cat).slice(skip, skip + n);
for (const cat of ["man", "help", "pkg", "err", "fun", "net"]) {
  for (const r of pick(cat)) {
    console.log(`\n===== ${cat} : ${r.cmd}  (exit ${r.exit}) =====`);
    console.log(r.output.split("\n").slice(0, 12).join("\n"));
  }
}
