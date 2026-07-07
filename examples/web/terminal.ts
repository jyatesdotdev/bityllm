// The bity virtual terminal (DESIGN §2, M5): loads the int8 checkpoint, wires
// the Shell + binaries to a DOM screen, and lets you type into the dream.

import { deserialize, InferenceSession, GPUInferenceSession, Shell, BINARIES } from "../../src/infer.ts";
import type { ShellIO, SessionLike } from "../../src/infer.ts";

const PROMPT = "guest@bity:~$ ";
const screen = document.getElementById("screen") as HTMLPreElement;
const crt = document.getElementById("crt") as HTMLDivElement;

const io: ShellIO = {
  write(s: string): void {
    screen.textContent += s;
    crt.scrollTop = crt.scrollHeight;
  },
  clear(): void {
    screen.textContent = "";
  },
  delay: (ms: number) => new Promise((r) => setTimeout(r, ms)),
};

async function main(): Promise<void> {
  io.write("loading model");
  const tick = setInterval(() => io.write("."), 180);
  // page-relative first (GitHub Pages: docs/terminal.int8.bity), repo-root fallback (local dev)
  let res = await fetch("terminal.int8.bity");
  if (!res.ok) res = await fetch("/models/terminal.int8.bity");
  if (!res.ok) {
    clearInterval(tick);
    io.write(`\nfailed to load terminal.int8.bity (${res.status})\nrun: node examples/export-int8.ts\n`);
    return;
  }
  const bytes = new Uint8Array(await res.arrayBuffer());
  const { model, tok, step } = deserialize(bytes);

  // engine selection: race WebGPU vs CPU for 24 tokens, keep the winner —
  // dispatch overhead varies wildly across browsers/GPUs, so measure, don't guess
  const race = async (s: SessionLike): Promise<number> => {
    s.feed("guest@bity:~$ ls\n");
    const t0 = performance.now();
    let n = 0;
    for await (const _ of s.stream({ maxNewTokens: 24, temperature: 0.8, seed: 1 })) n++;
    const rate = n / ((performance.now() - t0) / 1000);
    s.reset();
    return rate;
  };
  let session: SessionLike = new InferenceSession(model, tok);
  let engine = "cpu (pure TS)";
  try {
    const gpuS = await GPUInferenceSession.create(model, tok);
    const gRate = await race(gpuS);
    const cRate = await race(session);
    if (gRate > cRate) {
      session = gpuS;
      engine = `webgpu ${gRate.toFixed(0)} tok/s (cpu: ${cRate.toFixed(0)})`;
    } else {
      gpuS.destroy();
      engine = `cpu ${cRate.toFixed(0)} tok/s (webgpu: ${gRate.toFixed(0)})`;
    }
  } catch {
    /* no WebGPU in this browser — CPU it is */
  }
  clearInterval(tick);
  io.clear();

  io.write(`bity 0.1 — a hallucinated terminal (${(bytes.length / 1024).toFixed(0)} KB model, ${model.paramCount().toLocaleString()} params, step ${step}, engine: ${engine})\n`);
  io.write(`nothing below this line is real. type 'help' for the binaries it dreams best.\n\n`);
  const shell = new Shell(session, { prompt: PROMPT });
  shell.register(...BINARIES);

  let line = "";
  let busy = false;
  io.write(PROMPT);

  document.addEventListener("keydown", async (e: KeyboardEvent) => {
    if (busy || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === "Enter") {
      e.preventDefault();
      io.write("\n");
      const cmd = line;
      line = "";
      busy = true;
      try {
        await shell.run(cmd, io);
      } catch (err) {
        io.write(`\n[bity kernel panic: ${err instanceof Error ? err.message : String(err)}]\n`);
      }
      busy = false;
      io.write(PROMPT);
    } else if (e.key === "Backspace") {
      e.preventDefault();
      if (line.length > 0) {
        line = line.slice(0, -1);
        screen.textContent = screen.textContent!.slice(0, -1);
      }
    } else if (e.key.length === 1) {
      e.preventDefault();
      line += e.key;
      io.write(e.key);
    }
  });
}

void main();
