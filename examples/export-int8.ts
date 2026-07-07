// Export a training checkpoint as an int8 deployment checkpoint (browser).
//
//   node examples/export-int8.ts [--in models/terminal.bity] [--out models/terminal.int8.bity]

import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { deserialize, serializeInt8 } from "../src/index.ts";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
const opt = (name: string, def: string): string => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 && args[i + 1] ? args[i + 1] : def;
};

const IN = opt("in", "models/terminal.bity");
const OUT = opt("out", "models/terminal.int8.bity");

const bytes = readFileSync(resolve(ROOT, IN));
const { model, tok, step } = deserialize(bytes);
const out = serializeInt8(model, tok, { step });
writeFileSync(resolve(ROOT, OUT), out);

const kb = (n: number): string => (n / 1024).toFixed(1) + " KB";
console.log(`${IN} (${kb(bytes.length)}) → ${OUT} (${kb(out.length)})  [${((out.length / bytes.length) * 100).toFixed(0)}%]`);
console.log(`model: ${model.paramCount().toLocaleString()} params @ step ${step}`);
