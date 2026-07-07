// Generate from a saved checkpoint — a first taste of the virtual terminal.
//
//   npm run generate -- --cmd "ping bity.dev"
//   node examples/generate.ts --ckpt models/terminal.bity --cmd "ls -la" --tokens 300

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { deserialize, generate } from "../src/index.ts";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
const opt = (name: string, def: string): string => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 && args[i + 1] ? args[i + 1] : def;
};

const CKPT = opt("ckpt", "models/terminal.bity");
const CMD = opt("cmd", "ls");
const TOKENS = parseInt(opt("tokens", "400"), 10);
const TEMP = parseFloat(opt("temp", "0.8"));
const SEED = parseInt(opt("seed", `${Date.now() % 100000}`), 10);

const { model, tok, step } = deserialize(readFileSync(resolve(ROOT, CKPT)));
console.log(`checkpoint: ${CKPT} (step ${step}), ${model.paramCount().toLocaleString()} params\n`);

const PROMPT = "guest@bity:~$ ";
const seed = PROMPT + CMD + "\n";
process.stdout.write(seed);
generate(model, tok, seed, {
  maxNewTokens: TOKENS,
  temperature: TEMP,
  topK: 40,
  seed: SEED,
  stop: [PROMPT],
  onToken: (ch) => process.stdout.write(ch),
});
process.stdout.write("\n");
