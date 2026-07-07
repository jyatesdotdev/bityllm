// The virtual shell (DESIGN §2.2, §14.3): a registry of "binaries" over one
// InferenceSession. Model binaries stream conditioned inference with per-binary
// sampling/pacing; scripted binaries are plain functions; hybrids do both.

import type { StreamOpts } from "./session.ts";
import type { SampleOpts } from "./sampler.ts";

/** Any inference engine: CPU InferenceSession (sync generator) or
 *  GPUInferenceSession (async generator) — the Shell consumes both. */
export interface SessionLike {
  feed(text: string): void;
  stream(opts: StreamOpts): Generator<string> | AsyncGenerator<string>;
  reset(): void;
  readonly length: number;
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
  readonly session: SessionLike;
  readonly prompt: string;
  readonly registry = new Map<string, Binary>();
  private seedCounter: number;

  constructor(session: SessionLike, opts: { prompt: string; seed?: number }) {
    this.session = session;
    this.prompt = opts.prompt;
    this.seedCounter = opts.seed ?? (Date.now() & 0xffff);
    session.feed(this.prompt); // prime the context with the first prompt
  }

  register(...bins: Binary[]): void {
    for (const b of bins) this.registry.set(b.name, b);
  }

  /** Run one command line. The caller displays its own prompt + echo. */
  async run(line: string, io: ShellIO): Promise<void> {
    const argv = line.trim().split(/\s+/).filter(Boolean);
    const ctx: ShellContext = { io, session: this.session, prompt: this.prompt };
    const bin = argv.length ? this.registry.get(argv[0]) : undefined;

    // context: the typed line becomes part of the transcript the model sees
    // (a binary may rewrite it into the phrasing the corpus actually knows)
    this.session.feed((bin?.rewrite ? bin.rewrite(line) : line) + "\n");
    if (argv.length === 0) return;

    if (bin?.kind === "scripted" && bin.run) {
      await bin.run(argv, ctx);
    } else {
      await this.streamModel(bin, io);
    }
    this.session.feed(this.prompt); // the shell's next prompt is model context too
  }

  /** Stream model output with pacing, holding back a possible stop-sequence. */
  private async streamModel(bin: Binary | undefined, io: ShellIO): Promise<void> {
    const stop = this.prompt;
    const opts = {
      maxNewTokens: bin?.maxNewTokens ?? 512,
      temperature: bin?.sampling?.temperature ?? 0.7,
      topK: bin?.sampling?.topK ?? 40,
      stop: [stop],
      seed: this.seedCounter++,
    };
    const charDelay = bin?.pacing?.charDelayMs ?? 2;
    const lineDelay = bin?.pacing?.lineDelayMs ?? 0;

    // Hold back exactly stop.length chars: the stream ends the moment the stop
    // string is fully emitted, so the stop can only ever be the buffer's tail —
    // strip it there and nothing of the model's own prompt is displayed.
    let buf = "";
    for await (const ch of this.session.stream(opts)) {
      buf += ch;
      while (buf.length > stop.length) {
        const flush = buf[0];
        buf = buf.slice(1);
        io.write(flush);
        if (charDelay > 0) await io.delay(charDelay);
        if (flush === "\n" && lineDelay > 0) await io.delay(lineDelay);
      }
    }
    if (buf.endsWith(stop)) buf = buf.slice(0, -stop.length);
    io.write(buf);
  }
}
