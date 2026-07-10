// The virtual shell (DESIGN §2.2, §14.3): a registry of "binaries" over one
// InferenceSession. Model binaries stream conditioned inference with per-binary
// sampling/pacing; scripted binaries are plain functions; hybrids do both.

import type { StreamOpts } from "./session.ts";
import type { SampleOpts } from "./sampler.ts";
import { VFS } from "./vfs.ts";
import { execLine } from "./shell-exec.ts";
import { CORE, DREAMED } from "./coreutils.ts";
import type { ShellState } from "./coreutils.ts";

/** Cap the real output fed back into the model's context: enough for the dream
 *  to stay coherent, not so much it floods the short context window. */
const clampCtx = (s: string): string => (s.length > 600 ? s.slice(0, 600) : s);

/** Any inference engine: CPU InferenceSession (sync generator) or
 *  GPUInferenceSession (async generator) — the Shell consumes both. */
export interface SessionLike {
  feed(text: string): void;
  stream(opts: StreamOpts): Generator<string> | AsyncGenerator<string>;
  reset(): void;
  readonly length: number;
  /** speculative peek: save/restore cache position (ghost suggestions) */
  snapshot(): { t: number; c: number };
  restore(s: { t: number; c: number }): void;
}

export interface ShellIO {
  write(s: string): void;
  clear(): void;
  delay(ms: number): Promise<void>;
}

export interface ShellContext {
  io: ShellIO;
  session: SessionLike;
  prompt: string;
  shell: Shell;
}

export interface Binary {
  name: string;
  synopsis?: string;
  kind: "model" | "scripted";
  /** Rewrite the line fed to the model (display is untouched) — e.g. the
   *  corpus only knows shutdown sequences as `sudo reboot`. */
  rewrite?(line: string): string;
  // model binaries
  sampling?: SampleOpts;
  maxNewTokens?: number;
  pacing?: { charDelayMs?: number; lineDelayMs?: number };
  // scripted / hybrid binaries
  run?(argv: string[], ctx: ShellContext): Promise<void>;
}

export class Shell {
  /** the active inference session — swappable so a model change keeps the shell
   *  (and its knob settings/cwd) intact; see the demo's model selector */
  session: SessionLike;
  readonly registry = new Map<string, Binary>();
  /** front-panel overrides (null/"stock" = per-binary settings) */
  tempOverride: number | null = null;
  pacingMode: "stock" | "turbo" | "slow" = "stock";
  /** display cwd for the prompt ("~", "~/projects"); derived from state.cwd */
  cwd = "~";
  /** the programmatic core's world: a real virtual FS + cwd + env. Deterministic
   *  commands (ls/cat/grep/pipes/…) run against THIS as real code; only generative
   *  commands (git/ping/…/unknown) fall through to the model. */
  state: ShellState;
  private readonly promptPrefix: string | null;
  private readonly staticPrompt: string;
  private seedCounter: number;

  constructor(session: SessionLike, opts: { prompt: string; seed?: number }) {
    this.session = session;
    // prompts shaped "<prefix>:~$ " become location-aware; anything else is static
    const m = opts.prompt.match(/^(.+):~\$ $/);
    this.promptPrefix = m ? m[1] : null;
    this.staticPrompt = opts.prompt;
    this.seedCounter = opts.seed ?? (Date.now() & 0xffff);
    // derive the user from a "user@host" prompt prefix; default guest
    const user = this.promptPrefix?.split("@")[0] || "guest";
    const home = user === "root" ? "/root" : "/home/guest";
    this.state = {
      vfs: new VFS(),
      cwd: home,
      user,
      env: new Map([
        ["HOME", home], ["USER", user], ["SHELL", "/bin/bash"], ["PWD", home],
        ["TERM", "xterm-256color"], ["PATH", "/usr/local/bin:/usr/bin:/bin"], ["LANG", "en_US.UTF-8"],
      ]),
      lastExit: 0,
    };
    this.syncCwd();
    session.feed(this.prompt); // prime the context with the first prompt
  }

  /** Reflect the programmatic cwd into the ~-relative display path. */
  private syncCwd(): void {
    const home = this.state.user === "root" ? "/root" : "/home/guest";
    const p = this.state.cwd;
    this.cwd = p === home ? "~" : p.startsWith(home + "/") ? "~/" + p.slice(home.length + 1) : p;
  }

  /** Reset the working directory to home (used by reboot). */
  resetToHome(): void {
    this.state.cwd = this.state.user === "root" ? "/root" : "/home/guest";
    this.state.env.set("PWD", this.state.cwd);
    this.syncCwd();
  }

  /** All command names the UI can complete/list: programmatic core + dreamed set
   *  + registered binaries (deduped). */
  commandNames(): string[] {
    return [...new Set([...Object.keys(CORE), ...DREAMED, ...this.registry.keys()])].sort();
  }

  get prompt(): string {
    return this.promptPrefix ? `${this.promptPrefix}:${this.cwd}$ ` : this.staticPrompt;
  }

  /** Stop-sequences marking the model starting a new prompt: the exact current
   *  prompt AND the persona prefix ("guest@bity:"). The model frequently emits
   *  a different path than the real cwd (deep/rare paths especially), so the
   *  invariant prefix is what actually stops generation — without it, cat-after-cd
   *  overruns into a cascade of hallucinated commands. */
  get promptStops(): string[] {
    return this.promptPrefix ? [this.prompt, `${this.promptPrefix}:`] : [this.prompt];
  }

  register(...bins: Binary[]): void {
    for (const b of bins) this.registry.set(b.name, b);
  }

  /** Run one command line. The caller displays its own prompt + echo.
   *
   *  Routing (the hybrid split): scripted binaries (help/clear/reboot) do
   *  host-level things and win; then the programmatic CORE runs deterministic
   *  commands as real code over the VFS; only generative/unknown commands fall
   *  through to the model. Real output is still fed to the session so the dream
   *  context stays truthful for whatever gets dreamed next. */
  async run(line: string, io: ShellIO): Promise<void> {
    const trimmed = line.trim();
    const argv = trimmed.split(/\s+/).filter(Boolean);
    const ctx: ShellContext = { io, session: this.session, prompt: this.prompt, shell: this };
    const bin = argv.length ? this.registry.get(argv[0]) : undefined;

    // 1) scripted binaries clear the DOM / reset the session — they take precedence
    if (bin?.kind === "scripted" && bin.run) {
      this.session.feed((bin.rewrite ? bin.rewrite(line) : line) + "\n");
      await bin.run(argv, ctx);
      this.session.feed(this.prompt);
      return;
    }

    // 2) programmatic core: deterministic FS/text/identity commands + pipes,
    //    redirects, globs, $VARs, && || — all real code, always consistent
    if (trimmed) {
      const res = execLine(trimmed, this.state);
      if (!res.dreamed) {
        this.session.feed(trimmed + "\n");
        if (res.out) {
          await this.writePaced(res.out, io);
          this.session.feed(clampCtx(res.out)); // keep the dream context truthful
        }
        this.state.env.set("PWD", this.state.cwd);
        this.syncCwd();
        this.session.feed(this.prompt);
        return;
      }
    }

    // 3) generative/unknown → the model (a binary may rewrite the fed line into
    //    the phrasing the corpus actually knows)
    this.session.feed((bin?.rewrite ? bin.rewrite(line) : line) + "\n");
    if (argv.length === 0) return;
    await this.streamModel(bin, io);
    this.session.feed(this.prompt); // the shell's next prompt is model context too
  }

  /** Render deterministic output. Real commands feel instant (retro slow-mode
   *  still paces them); only dreamed output gets the streaming-typewriter charm. */
  private async writePaced(text: string, io: ShellIO): Promise<void> {
    const charDelay = this.pacingMode === "slow" ? 8 : 0;
    for (const ch of text) {
      io.write(ch);
      if (charDelay > 0) await io.delay(charDelay);
    }
  }

  /** Stream model output with pacing, holding back a possible next-prompt. */
  private async streamModel(bin: Binary | undefined, io: ShellIO): Promise<void> {
    // The model was fed the command line as context (session.feed in run()), so a
    // char-level model literally has the typed chars (e.g. "bity.dev") in its recent
    // context — no argument-passing code is needed; it just continues the transcript.
    const stops = this.promptStops;
    // Holdback window: the stream stops the instant a stop-sequence appears, so any
    // next-prompt the model began emitting lives entirely in the last ≤prompt.length
    // chars. We stream everything older than that immediately, and trim the tail —
    // so the model's own hallucinated next prompt never reaches the screen.
    const guard = this.prompt.length;
    const cutMark = this.promptPrefix ? `${this.promptPrefix}:` : this.prompt;
    const opts = {
      maxNewTokens: bin?.maxNewTokens ?? 512,
      temperature: this.tempOverride ?? bin?.sampling?.temperature ?? 0.7,
      topK: bin?.sampling?.topK ?? 40,
      stop: stops,
      seed: this.seedCounter++,
    };
    let charDelay = bin?.pacing?.charDelayMs ?? 2;
    let lineDelay = bin?.pacing?.lineDelayMs ?? 0;
    if (this.pacingMode === "turbo") {
      charDelay = 0;
      lineDelay = 0;
    } else if (this.pacingMode === "slow") {
      charDelay = 8; // ~1200 baud
    }

    // Hold back up to `guard` chars: the stream ends the moment a stop appears,
    // so any next-prompt lives in the buffer's tail — cut from the persona
    // prefix onward and nothing of the model's own prompt is displayed.
    let buf = "";
    for await (const ch of this.session.stream(opts)) {
      buf += ch;
      while (buf.length > guard) {
        const flush = buf[0];
        buf = buf.slice(1);
        io.write(flush);
        if (charDelay > 0) await io.delay(charDelay);
        if (flush === "\n" && lineDelay > 0) await io.delay(lineDelay);
      }
    }
    const cut = buf.lastIndexOf(cutMark);
    io.write(cut >= 0 ? buf.slice(0, cut) : buf);
  }
}

// ---- extracted interactive-shell logic (UI-agnostic, unit-tested) ----------

export interface Completion {
  kind: "none" | "complete" | "extend" | "list";
  text: string;       // full replacement line ("complete"/"extend") or "" otherwise
  options: string[];  // candidates when kind === "list"
}

/** Tab-completion over registered command names (first word only). */
export function completeCommand(line: string, names: string[]): Completion {
  if (line.length === 0 || line.includes(" ")) return { kind: "none", text: "", options: [] };
  const matches = names.filter((n) => n.startsWith(line)).sort();
  if (matches.length === 0) return { kind: "none", text: "", options: [] };
  if (matches.length === 1) return { kind: "complete", text: matches[0] + " ", options: matches };
  let p = matches[0];
  for (const m of matches) {
    while (!m.startsWith(p)) p = p.slice(0, -1);
  }
  if (p.length > line.length) return { kind: "extend", text: p, options: matches };
  return { kind: "list", text: "", options: matches };
}

/** Bash-style history cursor: up/down navigation with a saved in-progress line. */
export class History {
  private items: string[] = [];
  private idx = 0;
  private saved = "";

  push(line: string): void {
    if (line.trim()) this.items.push(line);
    this.idx = this.items.length;
    this.saved = "";
  }

  /** returns the line to display, or null if navigation hit an edge */
  up(current: string): string | null {
    if (this.idx === 0) return null;
    if (this.idx === this.items.length) this.saved = current;
    this.idx--;
    return this.items[this.idx];
  }

  down(): string | null {
    if (this.idx >= this.items.length) return null;
    this.idx++;
    return this.idx === this.items.length ? this.saved : this.items[this.idx];
  }
}
