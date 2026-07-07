// ingest-dmesg.mjs — pull user-provided real dmesg / journal files into the
// corpus, scrubbed of host identity (MAC, UUIDs, machine-id, hostname).
//
//   node corpus/capture/ingest-dmesg.mjs [file ...]   (defaults to root dmesg.*.txt)

import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { resolve, dirname, basename } from "node:path";
import { fileURLToPath } from "node:url";
import { sanitize, normalizeAscii, vmScrub } from "./lib.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, "../..");
const OUT = resolve(HERE, "../data");
const PROMPT = "guest@bity:~$ ";
const hostUser = (process.env.USER || "").trim() || undefined;

const inputs = process.argv.slice(2);
const files = inputs.length ? inputs : ["dmesg.0.txt", "dmesg.1.txt"];

// Replace the host field in syslog-style lines: "Mon DD HH:MM:SS <host> unit[pid]:"
const scrubJournalHost = (t) =>
  t.replace(/^([A-Z][a-z]{2}\s+\d+\s+\d{1,2}:\d{2}:\d{2})\s+(\S+)(\s+\S+?(?:\[\d+\])?:)/gm, "$1 bity$3");

const records = [];
let txt = "";
for (const rel of files) {
  let raw;
  try { raw = readFileSync(resolve(ROOT, rel), "utf8"); }
  catch { console.error("skip (not found):", rel); continue; }

  const isDmesg = /^\[\s*\d+\.\d+\]/.test(raw.trimStart());
  const pre = isDmesg ? raw : scrubJournalHost(raw);
  const clean = normalizeAscii(vmScrub(sanitize(pre), { hostUser }));
  const cmd = isDmesg ? "dmesg" : "journalctl -b -1 | tail -n 60";
  const cat = isDmesg ? "dmesg" : "reboot";

  records.push({ system: "real/" + basename(rel), cat, cmd, exit: 0, output: clean });
  txt += PROMPT + cmd + "\n" + (clean.endsWith("\n") ? clean : clean + "\n");
  console.log(`ingested ${rel}: ${isDmesg ? "boot dmesg" : "shutdown journal"}, ${clean.split("\n").length} lines`);
}

mkdirSync(OUT, { recursive: true });
writeFileSync(resolve(OUT, "real-dmesg.jsonl"), records.map((r) => JSON.stringify(r)).join("\n") + "\n");
writeFileSync(resolve(OUT, "real-dmesg.corpus.txt"), txt);
console.log(`\nwrote real-dmesg.{jsonl,corpus.txt} — ${(Buffer.byteLength(txt) / 1024).toFixed(0)}KB, vocab ${new Set(txt).size}`);

console.log("\n--- scrub verification (identity-bearing lines) ---");
for (const r of records) {
  for (const line of r.output.split("\n")) {
    if (/DMI:|link\/ether|ether |root=|by-uuid|de:ad:be:ef|1c0ffee5| bity /.test(line)) {
      console.log("  " + line.trim().slice(0, 108));
    }
  }
}