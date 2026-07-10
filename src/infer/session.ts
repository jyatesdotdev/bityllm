// InferenceSession (DESIGN §14): the lean, shipped path. Single-token forward
// with a per-layer KV-cache — no Tensor, no tape, no optimizer. Per-token cost
// is O(params + T·nEmbd) instead of O(params·T), which is what makes browser
// inference effortless (DESIGN §9.4).
//
// The cache persists across commands so the running transcript is shared
// context; reset() clears it (e.g. on `reboot`).

import { B } from "../backend/index.ts";
import type { NdArray } from "../backend/index.ts";
import { RNG } from "../core/rng.ts";
import type { GPT } from "../nn/gpt.ts";
import type { Tokenizer } from "../tokenizer/char.ts";
import { sampleLogits } from "./sampler.ts";
import type { SampleOpts } from "./sampler.ts";

export interface StreamOpts extends SampleOpts {
  maxNewTokens: number;
  stop?: string[];
  seed?: number;
}

export class InferenceSession {
  readonly model: GPT;
  readonly tok: Tokenizer;
  private readonly K: Float32Array[]; // per layer [blockSize, C] (heads packed)
  private readonly V: Float32Array[];
  private ctx: number[] = [];
  private curT = 0;
  private last: Float32Array | null = null; // logits after most recent token

  constructor(model: GPT, tok: Tokenizer) {
    this.model = model;
    this.tok = tok;
    const { nLayer, blockSize, nEmbd } = model.cfg;
    this.K = Array.from({ length: nLayer }, () => new Float32Array(blockSize * nEmbd));
    this.V = Array.from({ length: nLayer }, () => new Float32Array(blockSize * nEmbd));
  }

  reset(): void {
    this.ctx = [];
    this.curT = 0;
    this.last = null;
  }

  /** Append text to the context, prefilling the KV-cache. */
  feed(text: string): void {
    for (const id of this.tok.encode(text)) this.push(id);
  }

  /** Stream sampled characters until a stop-sequence or maxNewTokens. */
  *stream(opts: StreamOpts): Generator<string> {
    if (this.last === null) throw new Error("stream: feed() something first");
    const rng = new RNG(opts.seed ?? 0x5eed1);
    const maxStop = Math.max(0, ...(opts.stop ?? []).map((s) => s.length));
    let out = "";
    for (let n = 0; n < opts.maxNewTokens; n++) {
      const id = sampleLogits(this.last!, opts, rng);
      this.push(id);
      const ch = this.tok.decode([id]);
      out += ch;
      if (out.length > maxStop * 4) out = out.slice(-maxStop * 2); // ring buffer
      yield ch;
      if (opts.stop?.some((s) => out.endsWith(s))) return;
    }
  }

  generate(prompt: string, opts: StreamOpts): string {
    this.feed(prompt);
    let s = "";
    for (const ch of this.stream(opts)) s += ch;
    return s;
  }

  /** Context length currently in the cache. */
  get length(): number {
    return this.curT;
  }

  /** Speculative peek support: snapshot → feed/stream → restore. Stale KV rows
   *  beyond the restored position are simply overwritten by later pushes. */
  snapshot(): { t: number; c: number } {
    return { t: this.curT, c: this.ctx.length };
  }

  restore(s: { t: number; c: number }): void {
    this.curT = s.t;
    this.ctx.length = s.c;
    this.last = null;
  }

  // --- internals -------------------------------------------------------------

  private push(id: number): void {
    if (this.curT >= this.model.cfg.blockSize) this.rewind();
    this.last = this.step(id, this.curT);
    this.ctx.push(id);
    this.curT++;
  }

  /** Cache full: rebuild from the most recent half of the context.
   *  The KV-cache is fixed at blockSize rows, so when it fills we keep only the
   *  most recent half of the tokens and replay them from position 0 (a sliding
   *  window — older context is discarded, which is fine for a streaming terminal). */
  private rewind(): void {
    const keep = this.ctx.slice(-Math.floor(this.model.cfg.blockSize / 2));
    this.ctx = [];
    this.curT = 0;
    for (const id of keep) {
      this.last = this.step(id, this.curT);
      this.ctx.push(id);
      this.curT++;
    }
  }

  /** One-token forward at position pos, reading/writing the KV-cache. */
  private step(id: number, pos: number): Float32Array {
    const m = this.model;
    const { nEmbd: C, nHead: nH, vocabSize: V } = m.cfg;
    const hd = C / nH;
    const scale = 1 / Math.sqrt(hd);
    const be = B();

    // x = wte[id] + wpe[pos]
    const x = new Float32Array(C);
    const wte = m.wte.data, wpe = m.wpe.data;
    for (let j = 0; j < C; j++) x[j] = wte[id * C + j] + wpe[pos * C + j];
    const xNd: NdArray = { data: x, shape: [1, C] };

    const T = pos + 1;
    for (let l = 0; l < m.blocks.length; l++) {
      const blk = m.blocks[l];

      // --- attention (pre-norm) ---
      const h1 = be.layerNorm(xNd, blk.ln1.w.nd, blk.ln1.b.nd, 1e-5).y;
      const q = be.add(be.matmul(h1, blk.attn.wq.w.nd), blk.attn.wq.b!.nd).data;
      const k = be.add(be.matmul(h1, blk.attn.wk.w.nd), blk.attn.wk.b!.nd).data;
      const v = be.add(be.matmul(h1, blk.attn.wv.w.nd), blk.attn.wv.b!.nd).data;
      // KV-CACHE write: K[l]/V[l] are flat [blockSize, C] buffers. Row = timestep,
      // heads packed contiguously within a row (head h at columns [h·hd,(h+1)·hd)),
      // so position `pos`, head `h` starts at index pos·C + h·hd. We store THIS
      // token's k/v once; every future token reuses them instead of recomputing the
      // whole prefix — that's why per-token cost is O(params), not O(params·T).
      this.K[l].set(k, pos * C);
      this.V[l].set(v, pos * C);

      // Scaled dot-product attention for this one query token, head by head.
      // scores[t] = (q · K[t]) / √hd for every past position t in [0,T). Causality
      // is FREE: T = pos+1, so we only ever loop over positions ≤ pos.
      const attnOut = new Float32Array(C);
      const scores = new Float32Array(T);
      for (let h = 0; h < nH; h++) {
        const ho = h * hd;
        let max = -Infinity;
        for (let t = 0; t < T; t++) {
          let s = 0;
          const ko = t * C + ho;
          for (let j = 0; j < hd; j++) s += q[ho + j] * this.K[l][ko + j];
          s *= scale;
          scores[t] = s;
          if (s > max) max = s;
        }
        let sum = 0;
        for (let t = 0; t < T; t++) {
          scores[t] = Math.exp(scores[t] - max);
          sum += scores[t];
        }
        const inv = 1 / sum;
        for (let t = 0; t < T; t++) {
          const p = scores[t] * inv;
          const vo = t * C + ho;
          for (let j = 0; j < hd; j++) attnOut[ho + j] += p * this.V[l][vo + j];
        }
      }
      const proj = be.add(be.matmul({ data: attnOut, shape: [1, C] }, blk.attn.wo.w.nd), blk.attn.wo.b!.nd).data;
      for (let j = 0; j < C; j++) x[j] += proj[j];

      // --- MLP (pre-norm) ---
      const h2 = be.layerNorm(xNd, blk.ln2.w.nd, blk.ln2.b.nd, 1e-5).y;
      const f = be.gelu(be.add(be.matmul(h2, blk.mlp.fc.w.nd), blk.mlp.fc.b!.nd));
      const p2 = be.add(be.matmul(f, blk.mlp.proj.w.nd), blk.mlp.proj.b!.nd).data;
      for (let j = 0; j < C; j++) x[j] += p2[j];
    }

    const xf = be.layerNorm(xNd, m.lnf.w.nd, m.lnf.b.nd, 1e-5).y;
    const logits = be.matmul(xf, m.wte.nd, { transposeB: true }); // tied head → [1, V]
    return logits.data as Float32Array;
  }
}
