// The bity virtual terminal (DESIGN §2, M5): loads the int8 checkpoint, wires
// the Shell + binaries to a DOM screen, and lets you type into the dream.
//
// Niceties: bash-style history (↑/↓), Tab-completion of command names, and
// fish-style ghost autosuggestions — dreamed by the model itself via a
// speculative KV-cache peek (snapshot → feed → stream → restore).

import { deserialize, InferenceSession, GPUInferenceSession, Shell, BINARIES } from "../../src/infer.ts";
import type { ShellIO, SessionLike } from "../../src/infer.ts";

const PROMPT = "guest@bity:~$ ";
const textEl = document.getElementById("text") as HTMLSpanElement;
const ghostEl = document.getElementById("ghost") as HTMLSpanElement;
const crt = document.getElementById("crt") as HTMLDivElement;

const io: ShellIO = {
  write(s: string): void {
    textEl.textContent += s;
    crt.scrollTop = crt.scrollHeight;
  },
  clear(): void {
    textEl.textContent = "";
  },
  delay: (ms: number) => new Promise((r) => setTimeout(r, ms)),
};
const erase = (n: number): void => {
  textEl.textContent = textEl.textContent!.slice(0, textEl.textContent!.length - n);
};

// serialize all session use: a running command and a speculative ghost peek
// must never interleave on the shared KV-cache
let lock: Promise<unknown> = Promise.resolve();
function withLock<T>(fn: () => Promise<T>): Promise<T> {
  const r = lock.then(fn);
  lock = r.catch(() => {});
  return r;
}

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
  const commandNames = [...shell.registry.keys()].sort();

  let line = "";
  let busy = false;
  const history: string[] = [];
  let histIdx = 0;
  let savedLine = "";

  // ---- ghost autosuggestions: the model dreams your next keystrokes ----
  let ghostText = "";
  let suggestGen = 0;
  let suggestTimer: ReturnType<typeof setTimeout> | undefined;

  const setGhost = (s: string): void => {
    ghostText = s;
    ghostEl.textContent = s;
  };

  const computeGhost = (): void => {
    const gen = ++suggestGen;
    if (busy || line.length === 0) return;
    if (session.length + line.length + 56 >= model.cfg.blockSize) return; // rewind hazard
    void withLock(async () => {
      if (gen !== suggestGen || busy) return;
      const snap = session.snapshot();
      try {
        session.feed(line);
        let out = "";
        for await (const ch of session.stream({ maxNewTokens: 36, temperature: 0.2, topK: 3, stop: ["\n"], seed: 2 })) {
          out += ch;
          if (gen !== suggestGen) break;
        }
        out = out.split("\n")[0];
        if (gen === suggestGen && !busy && out.length > 0) setGhost(out);
      } finally {
        session.restore(snap);
      }
    });
  };

  const scheduleGhost = (): void => {
    setGhost("");
    clearTimeout(suggestTimer);
    suggestTimer = setTimeout(computeGhost, 220);
  };

  const replaceLine = (nl: string): void => {
    erase(line.length);
    line = nl;
    io.write(line);
    scheduleGhost();
  };

  io.write(PROMPT);

  document.addEventListener("keydown", async (e: KeyboardEvent) => {
    if (busy || e.metaKey || e.ctrlKey || e.altKey) return;

    if (e.key === "Enter") {
      e.preventDefault();
      suggestGen++;
      setGhost("");
      io.write("\n");
      const cmd = line;
      if (cmd.trim()) {
        history.push(cmd);
      }
      histIdx = history.length;
      savedLine = "";
      line = "";
      busy = true;
      try {
        await withLock(() => shell.run(cmd, io));
      } catch (err) {
        io.write(`\n[bity kernel panic: ${err instanceof Error ? err.message : String(err)}]\n`);
      }
      busy = false;
      io.write(PROMPT);
    } else if (e.key === "Backspace") {
      e.preventDefault();
      if (line.length > 0) {
        line = line.slice(0, -1);
        erase(1);
        scheduleGhost();
      }
    } else if (e.key === "Tab") {
      e.preventDefault();
      if (line.length > 0 && !line.includes(" ")) {
        const matches = commandNames.filter((n) => n.startsWith(line));
        if (matches.length === 1) replaceLine(matches[0] + " ");
        else if (matches.length > 1) {
          // longest common prefix; if no progress, show the candidates
          let p = matches[0];
          for (const m of matches) while (!m.startsWith(p)) p = p.slice(0, -1);
          if (p.length > line.length) replaceLine(p);
          else {
            io.write("\n" + matches.join("  ") + "\n" + PROMPT + line);
          }
        }
      }
    } else if (e.key === "ArrowRight" || e.key === "End") {
      if (ghostText) {
        e.preventDefault();
        line += ghostText;
        io.write(ghostText);
        setGhost("");
        scheduleGhost();
      }
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      if (histIdx > 0) {
        if (histIdx === history.length) savedLine = line;
        histIdx--;
        replaceLine(history[histIdx]);
      }
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      if (histIdx < history.length) {
        histIdx++;
        replaceLine(histIdx === history.length ? savedLine : history[histIdx]);
      }
    } else if (e.key === "Escape") {
      suggestGen++;
      setGhost("");
    } else if (e.key.length === 1) {
      e.preventDefault();
      // typing a char that matches the ghost consumes it instead of recomputing
      if (ghostText && ghostText[0] === e.key) {
        ghostText = ghostText.slice(1);
        ghostEl.textContent = ghostText;
        line += e.key;
        io.write(e.key);
        return;
      }
      line += e.key;
      io.write(e.key);
      scheduleGhost();
    }
  });
}

void main();
