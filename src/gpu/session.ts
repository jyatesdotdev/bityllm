// GPUInferenceSession — WebGPU single-token inference (browser + Deno).
//
// Same surface as the CPU InferenceSession, same weights, same KV-cache
// semantics — different engine. Weights and KV-cache live on the GPU;
// feed() is fully SYNCHRONOUS (WebGPU submits are sync; queue.writeBuffer
// between submits is ordered by spec), only the logits readback awaits.
// stream() is an async generator; the Shell consumes both engines via
// `for await`, which wraps sync generators transparently.

import * as KI from "./wgsl-infer.ts";
import * as KF from "./wgsl-fused.ts";
import { LN_FWD, GELU_FWD } from "./wgsl.ts";
import type { GPT } from "../nn/gpt.ts";
import type { Tokenizer } from "../tokenizer/char.ts";
import type { StreamOpts } from "../infer/session.ts";

interface Dispatch {
  pipeline: GPUComputePipeline;
  bind: GPUBindGroup;
  x: number;
}

export class GPUInferenceSession {
  readonly model: GPT;
  readonly tok: Tokenizer;
  private device: GPUDevice;
  private pipelines = new Map<string, GPUComputePipeline>();
  private globals!: GPUBuffer;
  private logitsBuf!: GPUBuffer;
  private stage!: GPUBuffer;
  private outIds!: GPUBuffer;
  private sampleUni!: GPUBuffer;
  private sample!: Dispatch;
  private steps: Dispatch[] = [];
  private chunkSeed = 0;

  private ctx: number[] = [];
  private curT = 0;
  private last: Float32Array | null = null;
  private dirty = false; // tokens submitted since the last logits readback

  private constructor(device: GPUDevice, model: GPT, tok: Tokenizer) {
    this.device = device;
    this.model = model;
    this.tok = tok;
  }

  static async create(model: GPT, tok: Tokenizer): Promise<GPUInferenceSession> {
    if (typeof navigator === "undefined" || !navigator.gpu) throw new Error("no WebGPU in this runtime");
    if (model.cfg.blockSize > 1024 || model.cfg.nEmbd > 512)
      throw new Error("GPU session supports blockSize ≤ 1024 and nEmbd ≤ 512");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("no GPU adapter");
    const device = await adapter.requestDevice();
    const s = new GPUInferenceSession(device, model, tok);
    device.pushErrorScope("validation");
    s.build();
    const err = await device.popErrorScope();
    if (err) throw new Error(`GPU session build failed validation: ${err.message}`);
    return s;
  }

  // ---- construction ----------------------------------------------------------

  private pipe(name: string, code: string): GPUComputePipeline {
    let p = this.pipelines.get(name);
    if (!p) {
      p = this.device.createComputePipeline({
        layout: "auto",
        compute: { module: this.device.createShaderModule({ code }), entryPoint: "main" },
      });
      this.pipelines.set(name, p);
    }
    return p;
  }

  private buf(n: number, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC): GPUBuffer {
    return this.device.createBuffer({ size: Math.max(16, n * 4), usage });
  }

  private uni(vals: Array<["u" | "f", number]>): GPUBuffer {
    const size = Math.ceil((vals.length * 4) / 16) * 16;
    const ab = new ArrayBuffer(size);
    const u = new Uint32Array(ab), f = new Float32Array(ab);
    vals.forEach(([k, v], i) => (k === "f" ? (f[i] = v) : (u[i] = v)));
    const b = this.device.createBuffer({ size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(b, 0, ab);
    return b;
  }

  private op(name: string, code: string, uniFields: Array<["u" | "f", number]>, buffers: GPUBuffer[], threads: number, perWorkgroup = 64): void {
    const pipeline = this.pipe(name, code);
    const bind = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [this.uni(uniFields), ...buffers].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    this.steps.push({ pipeline, bind, x: Math.ceil(threads / perWorkgroup) });
  }

  private build(): void {
    const { nEmbd: C, nHead: nH, nLayer: L, vocabSize: V, blockSize: B } = this.model.cfg;
    const hd = C / nH;
    const dev = this.device;

    // weights → GPU
    const W = new Map<string, GPUBuffer>();
    for (const [name, t] of this.model.namedParameters()) {
      const b = this.buf(t.size);
      dev.queue.writeBuffer(b, 0, (t.data as Float32Array).buffer, (t.data as Float32Array).byteOffset, t.size * 4);
      W.set(name, b);
    }
    const w = (n: string): GPUBuffer => W.get(n)!;

    this.globals = dev.createBuffer({ size: KI.GLOBALS_SIZE, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.logitsBuf = this.buf(V);
    this.stage = dev.createBuffer({ size: V * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    this.outIds = this.buf(64);
    this.sampleUni = dev.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    // scratch + caches
    const x = this.buf(C), lnO = this.buf(C);
    const m1 = this.buf(1), r1 = this.buf(1); // LN mean/rstd (rows=1, unused downstream)
    const Kc = Array.from({ length: L }, () => this.buf(B * C));
    const Vc = Array.from({ length: L }, () => this.buf(B * C));

    // per-layer weights concatenated into ONE buffer per layer half — keeps the
    // fused kernels within the baseline 8-storage-buffer browser limit and the
    // whole layer half in a single dispatch (see wgsl-fused.ts for the layout)
    const F = 4 * C;
    const cat = (names: string[]): GPUBuffer => {
      const tens = names.map((n) => this.model.namedParameters().find(([nm]) => nm === n)![1]);
      const total = tens.reduce((s, t) => s + t.size, 0);
      const arr = new Float32Array(total);
      let off = 0;
      for (const t of tens) {
        arr.set(t.data as Float32Array, off);
        off += t.size;
      }
      const b = this.buf(total);
      dev.queue.writeBuffer(b, 0, arr.buffer);
      return b;
    };

    const qRow = this.buf(C), attnO = this.buf(C);
    const fcOut = this.buf(F), gelOut = this.buf(F), projOut = this.buf(C);
    const gemv = (X: GPUBuffer, wgt: GPUBuffer, Y: GPUBuffer, K: number, N: number, bias?: GPUBuffer, atPos = false): void =>
      this.op("gemv", KI.GEMV, [["u", K], ["u", N], ["u", bias ? 1 : 0], ["u", atPos ? 1 : 0]],
        [this.globals, X, wgt, Y, bias ?? m1], N);
    const ln = (X: GPUBuffer, wn: string, bn: string, Y: GPUBuffer): void =>
      this.op("ln", LN_FWD, [["u", C], ["u", 1]], [X, w(wn), w(bn), Y, m1, r1], 128, 128);
    const accum = (A: GPUBuffer, Cc: GPUBuffer): void =>
      this.op("accum", KI.ACCUM, [["u", C]], [A, Cc], C, 256);

    // ---- the per-token program (unfused GEMV chain) ----
    // NOTE: single-workgroup fused layer kernels (wgsl-fused.ts) were measured
    // 2× SLOWER here — one workgroup = one GPU core = 6% occupancy. They stay
    // in-tree as the basis for a future multi-workgroup split. `cat` unused
    // until then.
    void cat;
    void KF;
    this.op("embed", KI.EMBED_ONE, [["u", C]], [this.globals, w("wte"), w("wpe"), x], C);
    for (let l = 0; l < L; l++) {
      const p = (s: string): string => `h${l}.${s}`;
      ln(x, p("ln1.w"), p("ln1.b"), lnO);
      gemv(lnO, w(p("attn.wq.w")), qRow, C, C, w(p("attn.wq.b")));
      gemv(lnO, w(p("attn.wk.w")), Kc[l], C, C, w(p("attn.wk.b")), true); // straight into cache @ posC
      gemv(lnO, w(p("attn.wv.w")), Vc[l], C, C, w(p("attn.wv.b")), true);
      this.op("attn", KI.ATTN_ONE, [["u", C], ["u", nH], ["u", hd], ["f", 1 / Math.sqrt(hd)]],
        [this.globals, qRow, Kc[l], Vc[l], attnO], nH * 128, 128);
      gemv(attnO, w(p("attn.wo.w")), projOut, C, C, w(p("attn.wo.b")));
      accum(projOut, x);
      ln(x, p("ln2.w"), p("ln2.b"), lnO);
      gemv(lnO, w(p("mlp.fc.w")), fcOut, C, F, w(p("mlp.fc.b")));
      this.op("gelu", GELU_FWD, [["u", F]], [fcOut, gelOut], F, 256);
      gemv(gelOut, w(p("mlp.proj.w")), projOut, F, C, w(p("mlp.proj.b")));
      accum(projOut, x);
    }
    ln(x, "lnf.w", "lnf.b", lnO);
    this.op("gemv_t", KI.GEMV_T, [["u", C], ["u", V], ["u", 0], ["u", 0]],
      [lnO, w("wte"), this.logitsBuf], V);

    // GPU-side sampler (chunked generation: one submit, one readback per chunk)
    const sp = this.pipe("sample", KI.SAMPLE);
    this.sample = {
      pipeline: sp,
      bind: dev.createBindGroup({
        layout: sp.getBindGroupLayout(0),
        entries: [this.sampleUni, this.globals, this.logitsBuf, this.outIds]
          .map((buffer, binding) => ({ binding, resource: { buffer } })),
      }),
      x: 1,
    };
  }

  // ---- session surface (mirrors the CPU InferenceSession) --------------------

  reset(): void {
    this.ctx = [];
    this.curT = 0;
    this.last = null;
    this.dirty = false;
  }

  get length(): number {
    return this.curT;
  }

  /** Speculative peek support (see CPU session): stale KV rows are overwritten. */
  snapshot(): { t: number; c: number } {
    return { t: this.curT, c: this.ctx.length };
  }

  restore(s: { t: number; c: number }): void {
    this.curT = s.t;
    this.ctx.length = s.c;
    this.last = null;
    this.dirty = true;
  }

  /** Synchronous: submits GPU work per token; logits are read lazily in stream(). */
  feed(text: string): void {
    for (const id of this.tok.encode(text)) this.push(id);
  }

  /** Chunked generation: the SAMPLE kernel picks each token on-GPU and feeds it
   *  straight into the next forward — N tokens per submit, one 4·N-byte
   *  readback per chunk instead of a fence round-trip per token. */
  async *stream(opts: StreamOpts): AsyncGenerator<string> {
    if (this.curT === 0) throw new Error("stream: feed() something first");
    const C = this.model.cfg.nEmbd;
    const V = this.model.cfg.vocabSize;
    const maxStop = Math.max(0, ...(opts.stop ?? []).map((s) => s.length));
    if (this.chunkSeed === 0) this.chunkSeed = ((opts.seed ?? 0x5eed1) >>> 0) || 1;
    let out = "";
    let produced = 0;
    while (produced < opts.maxNewTokens) {
      const n = Math.min(48, opts.maxNewTokens - produced);
      if (this.curT + n >= this.model.cfg.blockSize) this.rewind();
      // state one position behind: the SAMPLE kernel advances it each iteration
      const g = new Uint32Array([0, this.curT - 1, (this.curT - 1) * C, this.curT, 0, 0, 0, 0]);
      this.device.queue.writeBuffer(this.globals, 0, g.buffer);
      const su = new ArrayBuffer(32);
      const u32 = new Uint32Array(su), f32 = new Float32Array(su);
      u32[0] = V;
      u32[1] = C;
      u32[2] = opts.topK ?? 0;
      f32[3] = opts.temperature ?? 0.8;
      u32[4] = this.chunkSeed++;
      this.device.queue.writeBuffer(this.sampleUni, 0, su);

      const enc = this.device.createCommandEncoder();
      for (let i = 0; i < n; i++) {
        const pass = enc.beginComputePass();
        for (const d of [this.sample, ...this.steps]) {
          pass.setPipeline(d.pipeline);
          pass.setBindGroup(0, d.bind);
          pass.dispatchWorkgroups(d.x);
        }
        pass.end();
      }
      enc.copyBufferToBuffer(this.outIds, 0, this.stage, 0, n * 4);
      this.device.queue.submit([enc.finish()]);
      await this.stage.mapAsync(GPUMapMode.READ);
      const ids = new Uint32Array(this.stage.getMappedRange().slice(0, n * 4));
      this.stage.unmap();
      this.dirty = false;
      this.last = null;

      for (const id of ids) {
        this.ctx.push(id);
        this.curT++;
        const ch = this.tok.decode([id]);
        out += ch;
        if (out.length > maxStop * 4) out = out.slice(-maxStop * 2);
        yield ch;
        produced++;
        // overshoot rollback is implicit: curT stops here; stale KV rows get overwritten
        if (opts.stop?.some((s) => out.endsWith(s))) return;
      }
    }
  }

  async generate(prompt: string, opts: StreamOpts): Promise<string> {
    this.feed(prompt);
    let s = "";
    for await (const ch of this.stream(opts)) s += ch;
    return s;
  }

  /** Current next-token logits (readback). Exposed for parity testing. */
  async logits(): Promise<Float32Array> {
    if (this.dirty || this.last === null) this.last = await this.readLogits();
    return this.last;
  }

  // ---- internals --------------------------------------------------------------

  private push(id: number): void {
    if (this.curT >= this.model.cfg.blockSize) this.rewind();
    const g = new Uint32Array([id, this.curT, this.curT * this.model.cfg.nEmbd, this.curT + 1]);
    this.device.queue.writeBuffer(this.globals, 0, g.buffer); // ordered vs. subsequent submit
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    for (const d of this.steps) {
      pass.setPipeline(d.pipeline);
      pass.setBindGroup(0, d.bind);
      pass.dispatchWorkgroups(d.x);
    }
    pass.end();
    this.device.queue.submit([enc.finish()]);
    this.ctx.push(id);
    this.curT++;
    this.dirty = true;
  }

  private rewind(): void {
    const keep = this.ctx.slice(-Math.floor(this.model.cfg.blockSize / 2));
    this.ctx = [];
    this.curT = 0;
    for (const id of keep) this.push(id);
  }

  private async readLogits(): Promise<Float32Array> {
    const V = this.model.cfg.vocabSize;
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.logitsBuf, 0, this.stage, 0, V * 4);
    this.device.queue.submit([enc.finish()]);
    await this.stage.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(this.stage.getMappedRange().slice(0, V * 4));
    this.stage.unmap();
    this.dirty = false;
    return out;
  }

  destroy(): void {
    this.device.destroy();
  }
}
