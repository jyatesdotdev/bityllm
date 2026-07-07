// GPUTrainer (DESIGN M6): full training step — forward, backward, clip, AdamW —
// as a prebuilt WebGPU dispatch schedule over resident buffers. Weights, grads,
// activations, and optimizer moments never leave the GPU; only the per-row loss
// (for logging) and checkpoints are ever read back.
//
// The schedule mirrors core/ops.ts backward formulas exactly; bench/gpu-parity.ts
// verifies gradient agreement with the CPU autograd path.
//
// Runs anywhere WebGPU exists: Deno (wgpu → Metal) today, browsers unchanged.

import * as K from "./wgsl.ts";
import type { GPT } from "../nn/gpt.ts";

interface Dispatch {
  pipeline: GPUComputePipeline;
  bind: GPUBindGroup;
  x: number;
  y: number;
  z: number;
}

export interface GPUTrainerOpts {
  batchSize: number;
  weightDecay?: number;
  clip?: number;
  beta1?: number;
  beta2?: number;
  eps?: number;
}

export class GPUTrainer {
  private device: GPUDevice;
  private model: GPT;
  private B: number;
  private T: number;
  private opts: Required<GPUTrainerOpts>;

  private pipelines = new Map<string, GPUComputePipeline>();
  private weights = new Map<string, GPUBuffer>();
  private grads = new Map<string, GPUBuffer>();
  private moments = new Map<string, { m: GPUBuffer; v: GPUBuffer }>();
  private decaySet = new Set<string>();

  private ids!: GPUBuffer;
  private tgt!: GPUBuffer;
  private lossBuf!: GPUBuffer;
  private lossStage!: GPUBuffer;
  private normBuf!: GPUBuffer;
  private dummyBias!: GPUBuffer;

  private fwd: Dispatch[] = [];
  private bwd: Dispatch[] = [];
  private clipOps: Dispatch[] = [];
  private optOps: Array<Dispatch & { uni: GPUBuffer; n: number; decay: boolean }> = [];
  private stepCount = 0;
  /** forward-order activation buffers, for debugging (name, buffer, element count) */
  private acts: Array<{ name: string; buf: GPUBuffer; n: number }> = [];

  private constructor(device: GPUDevice, model: GPT, opts: GPUTrainerOpts) {
    this.device = device;
    this.model = model;
    this.B = opts.batchSize;
    this.T = model.cfg.blockSize;
    this.opts = {
      batchSize: opts.batchSize,
      weightDecay: opts.weightDecay ?? 0.1,
      clip: opts.clip ?? 1.0,
      beta1: opts.beta1 ?? 0.9,
      beta2: opts.beta2 ?? 0.95,
      eps: opts.eps ?? 1e-8,
    };
  }

  static async create(model: GPT, opts: GPUTrainerOpts): Promise<GPUTrainer> {
    if (typeof navigator === "undefined" || !navigator.gpu) throw new Error("no WebGPU in this runtime (use Deno or a browser)");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("no GPU adapter");
    const device = await adapter.requestDevice();
    const t = new GPUTrainer(device, model, opts);
    t.build();
    return t;
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

  private buf(size: number, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC): GPUBuffer {
    return this.device.createBuffer({ size: Math.max(16, size), usage });
  }

  /** uniform from mixed u32/f32 fields: "u" | "f" per value */
  private uni(fields: Array<[string, number]>): GPUBuffer {
    const size = Math.ceil((fields.length * 4) / 16) * 16;
    const ab = new ArrayBuffer(size);
    const u = new Uint32Array(ab);
    const f = new Float32Array(ab);
    fields.forEach(([k, v], i) => (k === "f" ? (f[i] = v) : (u[i] = v)));
    const b = this.device.createBuffer({ size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(b, 0, ab);
    return b;
  }

  private bind(pipeline: GPUComputePipeline, buffers: GPUBuffer[]): GPUBindGroup {
    return this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
  }

  private op(list: Dispatch[], name: string, code: string, uniFields: Array<[string, number]>, buffers: GPUBuffer[], x: number, y = 1, z = 1): void {
    const pipeline = this.pipe(name, code);
    list.push({ pipeline, bind: this.bind(pipeline, [this.uni(uniFields), ...buffers]), x, y, z });
  }

  /** GEMM: C[z] = alpha·op(A)op(B) + beta·C (+bias). */
  private gemm(
    list: Dispatch[],
    a: GPUBuffer, b: GPUBuffer, c: GPUBuffer,
    M: number, N: number, Kd: number,
    o: {
      batch?: number; inner?: number;
      aOut?: number; aIn?: number; lda?: number;
      bOut?: number; bIn?: number; ldb?: number;
      cOut?: number; cIn?: number; ldc?: number;
      ta?: boolean; tb?: boolean; alpha?: number; beta?: number; bias?: GPUBuffer;
    } = {},
  ): void {
    const batch = o.batch ?? 1;
    this.op(
      list, "gemm", K.GEMM,
      [
        ["u", M], ["u", N], ["u", Kd], ["u", batch],
        ["u", o.inner ?? 1], ["u", o.aOut ?? 0], ["u", o.aIn ?? 0], ["u", o.lda ?? (o.ta ? M : Kd)],
        ["u", o.bOut ?? 0], ["u", o.bIn ?? 0], ["u", o.ldb ?? (o.tb ? Kd : N)], ["u", o.ldc ?? N],
        ["u", o.cOut ?? 0], ["u", o.cIn ?? 0], ["u", o.ta ? 1 : 0], ["u", o.tb ? 1 : 0],
        ["f", o.alpha ?? 1], ["f", o.beta ?? 0], ["u", o.bias ? 1 : 0], ["u", 0],
      ],
      [a, b, c, o.bias ?? this.dummyBias],
      Math.ceil(N / 16), Math.ceil(M / 16), batch,
    );
  }

  private build(): void {
    const { vocabSize: V, nEmbd: C, nHead: nH, nLayer: L } = this.model.cfg;
    const { B, T } = this;
    const BT = B * T;
    const hd = C / nH;
    const F = 4 * C;

    // --- weight/grad/moment buffers ---
    const groups = this.model.paramGroups();
    const decayTensors = new Set(groups.decay);
    for (const [name, t] of this.model.namedParameters()) {
      const w = this.buf(t.size * 4);
      this.device.queue.writeBuffer(w, 0, (t.data as Float32Array).buffer, (t.data as Float32Array).byteOffset, t.size * 4);
      this.weights.set(name, w);
      this.grads.set(name, this.buf(t.size * 4));
      this.moments.set(name, { m: this.buf(t.size * 4), v: this.buf(t.size * 4) });
      if (decayTensors.has(t)) this.decaySet.add(name);
    }
    const W = (n: string): GPUBuffer => this.weights.get(n)!;
    const G = (n: string): GPUBuffer => this.grads.get(n)!;

    // --- IO + activations ---
    this.ids = this.buf(BT * 4);
    this.tgt = this.buf(BT * 4);
    this.lossBuf = this.buf(BT * 4);
    this.lossStage = this.device.createBuffer({ size: BT * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    this.normBuf = this.buf(16);
    this.dummyBias = this.buf(16);

    const act = (n: number): GPUBuffer => this.buf(n * 4);
    const xIn: GPUBuffer[] = Array.from({ length: L + 1 }, () => act(BT * C));
    const ln1Out = Array.from({ length: L }, () => act(BT * C));
    const ln1M = Array.from({ length: L }, () => act(BT));
    const ln1R = Array.from({ length: L }, () => act(BT));
    const q = Array.from({ length: L }, () => act(BT * C));
    const kB = Array.from({ length: L }, () => act(BT * C));
    const vB = Array.from({ length: L }, () => act(BT * C));
    const probs = Array.from({ length: L }, () => act(B * nH * T * T));
    const attnY = Array.from({ length: L }, () => act(BT * C));
    const xMid = Array.from({ length: L }, () => act(BT * C));
    const ln2Out = Array.from({ length: L }, () => act(BT * C));
    const ln2M = Array.from({ length: L }, () => act(BT));
    const ln2R = Array.from({ length: L }, () => act(BT));
    const fcOut = Array.from({ length: L }, () => act(BT * F));
    const gelOut = Array.from({ length: L }, () => act(BT * F));
    const lnfOut = act(BT * C);
    const lnfM = act(BT);
    const lnfR = act(BT);
    const logits = act(BT * V);
    const lse = act(BT);

    // register activations in forward order for bisect debugging
    for (let l = 0; l < L; l++) {
      this.acts.push(
        { name: `h${l}.xIn`, buf: xIn[l], n: BT * C },
        { name: `h${l}.ln1Out`, buf: ln1Out[l], n: BT * C },
        { name: `h${l}.q`, buf: q[l], n: BT * C },
        { name: `h${l}.k`, buf: kB[l], n: BT * C },
        { name: `h${l}.v`, buf: vB[l], n: BT * C },
        { name: `h${l}.probs`, buf: probs[l], n: B * nH * T * T },
        { name: `h${l}.attnY`, buf: attnY[l], n: BT * C },
        { name: `h${l}.xMid`, buf: xMid[l], n: BT * C },
        { name: `h${l}.ln2Out`, buf: ln2Out[l], n: BT * C },
        { name: `h${l}.fcOut`, buf: fcOut[l], n: BT * F },
        { name: `h${l}.gelOut`, buf: gelOut[l], n: BT * F },
      );
    }
    this.acts.push(
      { name: "xFinal", buf: xIn[L], n: BT * C },
      { name: "lnfOut", buf: lnfOut, n: BT * C },
      { name: "logits", buf: logits, n: BT * V },
      { name: "lse", buf: lse, n: BT },
      { name: "loss", buf: this.lossBuf, n: BT },
    );

    // scratch
    const sC = act(BT * C);      // proj outputs / dLn
    const dX = act(BT * C);      // running dx
    const dQ = act(BT * C), dK = act(BT * C), dV = act(BT * C);
    const dProbs = act(B * nH * T * T);
    const dScores = act(B * nH * T * T);
    const dAttnY = act(BT * C);
    const dFc = act(BT * F);
    const dGel = act(BT * F);
    const dLogits = act(BT * V);

    const headStride = { batch: B * nH, inner: nH, lda: C, ldb: C };

    // ============================ FORWARD ============================
    this.op(this.fwd, "embed", K.EMBED_FWD, [["u", C], ["u", T], ["u", BT * C], ["u", 0]],
      [this.ids, W("wte"), W("wpe"), xIn[0]], Math.ceil((BT * C) / 256));

    for (let l = 0; l < L; l++) {
      const p = (s: string): string => `h${l}.${s}`;
      // ln1 → q,k,v
      this.op(this.fwd, "ln_fwd", K.LN_FWD, [["u", C], ["u", BT]],
        [xIn[l], W(p("ln1.w")), W(p("ln1.b")), ln1Out[l], ln1M[l], ln1R[l]], BT);
      this.gemm(this.fwd, ln1Out[l], W(p("attn.wq.w")), q[l], BT, C, C, { bias: W(p("attn.wq.b")) });
      this.gemm(this.fwd, ln1Out[l], W(p("attn.wk.w")), kB[l], BT, C, C, { bias: W(p("attn.wk.b")) });
      this.gemm(this.fwd, ln1Out[l], W(p("attn.wv.w")), vB[l], BT, C, C, { bias: W(p("attn.wv.b")) });
      // scores = q@kᵀ/√hd → causal softmax → probs
      this.gemm(this.fwd, q[l], kB[l], probs[l], T, T, hd, {
        ...headStride, aOut: T * C, aIn: hd, bOut: T * C, bIn: hd, tb: true,
        cOut: nH * T * T, cIn: T * T, ldc: T, alpha: 1 / Math.sqrt(hd),
      });
      this.op(this.fwd, "sm_fwd", K.SOFTMAX_FWD, [["u", T], ["u", B * nH * T]], [probs[l]], B * nH * T);
      // y = probs@v (per head) → out proj (+residual)
      this.gemm(this.fwd, probs[l], vB[l], attnY[l], T, hd, T, {
        ...headStride, aOut: nH * T * T, aIn: T * T, lda: T, bOut: T * C, bIn: hd,
        cOut: T * C, cIn: hd, ldc: C,
      });
      this.gemm(this.fwd, attnY[l], W(p("attn.wo.w")), sC, BT, C, C, { bias: W(p("attn.wo.b")) });
      this.op(this.fwd, "add", K.ADD, [["u", BT * C]], [xIn[l], sC, xMid[l]], Math.ceil((BT * C) / 256));
      // ln2 → mlp (+residual)
      this.op(this.fwd, "ln_fwd", K.LN_FWD, [["u", C], ["u", BT]],
        [xMid[l], W(p("ln2.w")), W(p("ln2.b")), ln2Out[l], ln2M[l], ln2R[l]], BT);
      this.gemm(this.fwd, ln2Out[l], W(p("mlp.fc.w")), fcOut[l], BT, F, C, { bias: W(p("mlp.fc.b")) });
      this.op(this.fwd, "gelu_fwd", K.GELU_FWD, [["u", BT * F]], [fcOut[l], gelOut[l]], Math.ceil((BT * F) / 256));
      this.gemm(this.fwd, gelOut[l], W(p("mlp.proj.w")), sC, BT, C, F, { bias: W(p("mlp.proj.b")) });
      this.op(this.fwd, "add", K.ADD, [["u", BT * C]], [xMid[l], sC, xIn[l + 1]], Math.ceil((BT * C) / 256));
    }

    this.op(this.fwd, "ln_fwd", K.LN_FWD, [["u", C], ["u", BT]],
      [xIn[L], W("lnf.w"), W("lnf.b"), lnfOut, lnfM, lnfR], BT);
    this.gemm(this.fwd, lnfOut, W("wte"), logits, BT, V, C, { tb: true, ldb: C });
    this.op(this.fwd, "ce_fwd", K.CE_FWD, [["u", V], ["u", BT]], [logits, this.tgt, lse, this.lossBuf], BT);

    // ============================ BACKWARD ============================
    this.op(this.bwd, "ce_bwd", K.CE_BWD, [["u", V], ["u", BT], ["f", 1 / BT], ["u", 0]],
      [logits, this.tgt, lse, dLogits], Math.ceil((BT * V) / 256));
    // logits = lnfOut @ wteᵀ
    this.gemm(this.bwd, dLogits, W("wte"), sC, BT, C, V, {});                                  // dLnfOut
    this.gemm(this.bwd, dLogits, lnfOut, G("wte"), V, C, BT, { ta: true, lda: V, ldb: C, beta: 1 }); // dwte += dLogitsᵀ@lnfOut
    this.op(this.bwd, "ln_dx", K.LN_BWD_DX, [["u", C], ["u", BT], ["u", 0], ["u", 0]],
      [xIn[L], W("lnf.w"), sC, lnfM, lnfR, dX], BT);
    this.op(this.bwd, "ln_dwdb", K.LN_BWD_DWDB, [["u", C], ["u", BT]],
      [xIn[L], sC, lnfM, lnfR, G("lnf.w"), G("lnf.b")], Math.ceil(C / 64));

    for (let l = L - 1; l >= 0; l--) {
      const p = (s: string): string => `h${l}.${s}`;
      // ---- MLP backward (dX = dxOut; also the residual grad into xMid) ----
      this.gemm(this.bwd, dX, W(p("mlp.proj.w")), dGel, BT, F, C, { tb: true, ldb: C });
      // dW = Xᵀ@dY: the *activation* is the transposed operand
      this.gemm(this.bwd, gelOut[l], dX, G(p("mlp.proj.w")), F, C, BT, { ta: true, lda: F, ldb: C, beta: 1 });
      this.op(this.bwd, "colsum", K.COLSUM, [["u", C], ["u", BT]], [dX, G(p("mlp.proj.b"))], Math.ceil(C / 64));
      this.op(this.bwd, "gelu_bwd", K.GELU_BWD, [["u", BT * F]], [fcOut[l], dGel, dFc], Math.ceil((BT * F) / 256));
      this.gemm(this.bwd, dFc, W(p("mlp.fc.w")), sC, BT, C, F, { tb: true, ldb: F });
      this.gemm(this.bwd, ln2Out[l], dFc, G(p("mlp.fc.w")), C, F, BT, { ta: true, lda: C, ldb: F, beta: 1 });
      this.op(this.bwd, "colsum", K.COLSUM, [["u", F], ["u", BT]], [dFc, G(p("mlp.fc.b"))], Math.ceil(F / 64));
      this.op(this.bwd, "ln_dx", K.LN_BWD_DX, [["u", C], ["u", BT], ["u", 1], ["u", 0]],
        [xMid[l], W(p("ln2.w")), sC, ln2M[l], ln2R[l], dX], BT); // dX += ln2-path (residual)
      this.op(this.bwd, "ln_dwdb", K.LN_BWD_DWDB, [["u", C], ["u", BT]],
        [xMid[l], sC, ln2M[l], ln2R[l], G(p("ln2.w")), G(p("ln2.b"))], Math.ceil(C / 64));

      // ---- attention backward (dX = dxMid) ----
      this.gemm(this.bwd, dX, W(p("attn.wo.w")), dAttnY, BT, C, C, { tb: true, ldb: C });
      this.gemm(this.bwd, attnY[l], dX, G(p("attn.wo.w")), C, C, BT, { ta: true, lda: C, ldb: C, beta: 1 });
      this.op(this.bwd, "colsum", K.COLSUM, [["u", C], ["u", BT]], [dX, G(p("attn.wo.b"))], Math.ceil(C / 64));
      // dProbs = dY@vᵀ ; dV = probsᵀ@dY
      this.gemm(this.bwd, dAttnY, vB[l], dProbs, T, T, hd, {
        ...headStride, aOut: T * C, aIn: hd, bOut: T * C, bIn: hd, tb: true,
        cOut: nH * T * T, cIn: T * T, ldc: T,
      });
      this.gemm(this.bwd, probs[l], dAttnY, dV, T, hd, T, {
        ...headStride, aOut: nH * T * T, aIn: T * T, lda: T, ta: true, bOut: T * C, bIn: hd,
        cOut: T * C, cIn: hd, ldc: C,
      });
      this.op(this.bwd, "sm_bwd", K.SOFTMAX_BWD, [["u", T], ["u", B * nH * T]], [probs[l], dProbs, dScores], B * nH * T);
      // dQ = dS@k·s ; dK = dSᵀ@q·s
      const s = 1 / Math.sqrt(hd);
      this.gemm(this.bwd, dScores, kB[l], dQ, T, hd, T, {
        ...headStride, aOut: nH * T * T, aIn: T * T, lda: T, bOut: T * C, bIn: hd,
        cOut: T * C, cIn: hd, ldc: C, alpha: s,
      });
      this.gemm(this.bwd, dScores, q[l], dK, T, hd, T, {
        ...headStride, aOut: nH * T * T, aIn: T * T, lda: T, ta: true, bOut: T * C, bIn: hd,
        cOut: T * C, cIn: hd, ldc: C, alpha: s,
      });
      // qkv projection backward → dLn1 (sC), weight grads
      this.gemm(this.bwd, dQ, W(p("attn.wq.w")), sC, BT, C, C, { tb: true, ldb: C });
      this.gemm(this.bwd, dK, W(p("attn.wk.w")), sC, BT, C, C, { tb: true, ldb: C, beta: 1 });
      this.gemm(this.bwd, dV, W(p("attn.wv.w")), sC, BT, C, C, { tb: true, ldb: C, beta: 1 });
      for (const [db, nm] of [[dQ, "wq"], [dK, "wk"], [dV, "wv"]] as Array<[GPUBuffer, string]>) {
        this.gemm(this.bwd, ln1Out[l], db, G(p(`attn.${nm}.w`)), C, C, BT, { ta: true, lda: C, ldb: C, beta: 1 });
        this.op(this.bwd, "colsum", K.COLSUM, [["u", C], ["u", BT]], [db, G(p(`attn.${nm}.b`))], Math.ceil(C / 64));
      }
      this.op(this.bwd, "ln_dx", K.LN_BWD_DX, [["u", C], ["u", BT], ["u", 1], ["u", 0]],
        [xIn[l], W(p("ln1.w")), sC, ln1M[l], ln1R[l], dX], BT); // dX += ln1-path (residual)
      this.op(this.bwd, "ln_dwdb", K.LN_BWD_DWDB, [["u", C], ["u", BT]],
        [xIn[l], sC, ln1M[l], ln1R[l], G(p("ln1.w")), G(p("ln1.b"))], Math.ceil(C / 64));
    }
    // embeddings
    this.op(this.bwd, "emb_wte", K.EMBED_BWD_WTE, [["u", C], ["u", V], ["u", BT], ["u", 0]],
      [this.ids, dX, G("wte")], Math.ceil((V * C) / 256));
    this.op(this.bwd, "emb_wpe", K.EMBED_BWD_WPE, [["u", C], ["u", T], ["u", B], ["u", 0]],
      [dX, G("wpe")], Math.ceil((T * C) / 256));

    // ---- clip + AdamW ----
    for (const [name, t] of this.model.namedParameters()) {
      this.op(this.clipOps, "sqsum", K.SQSUM, [["u", t.size]], [G(name), this.normBuf], 1);
    }
    for (const [name, t] of this.model.namedParameters()) {
      this.op(this.clipOps, "clip", K.CLIP_SCALE, [["u", t.size], ["f", this.opts.clip]],
        [this.normBuf, G(name)], Math.ceil(t.size / 256));
    }
    for (const [name, t] of this.model.namedParameters()) {
      const pipeline = this.pipe("adamw", K.ADAMW);
      const uni = this.device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const { m, v } = this.moments.get(name)!;
      this.optOps.push({
        pipeline,
        bind: this.bind(pipeline, [uni, G(name), m, v, this.weights.get(name)!]),
        x: Math.ceil(t.size / 256), y: 1, z: 1,
        uni, n: t.size, decay: this.decaySet.has(name),
      });
    }

  }

  // ---- per-step execution ------------------------------------------------------

  /** Enqueue one full training step. Returns after GPU completion. */
  async step(x: Int32Array, y: Int32Array, lr: number): Promise<void> {
    const dev = this.device;
    this.stepCount++;
    dev.queue.writeBuffer(this.ids, 0, Uint32Array.from(x));
    dev.queue.writeBuffer(this.tgt, 0, Uint32Array.from(y));

    // per-step AdamW uniforms (lr, bias correction)
    const bc1 = 1 - Math.pow(this.opts.beta1, this.stepCount);
    const bc2 = 1 - Math.pow(this.opts.beta2, this.stepCount);
    for (const o of this.optOps) {
      const ab = new ArrayBuffer(32);
      const u = new Uint32Array(ab);
      const f = new Float32Array(ab);
      u[0] = o.n;
      f[1] = lr;
      f[2] = this.opts.beta1;
      f[3] = this.opts.beta2;
      f[4] = this.opts.eps;
      f[5] = o.decay ? this.opts.weightDecay : 0;
      f[6] = bc1;
      f[7] = bc2;
      dev.queue.writeBuffer(o.uni, 0, ab);
    }

    const enc = dev.createCommandEncoder();
    for (const g of this.grads.values()) enc.clearBuffer(g);
    enc.clearBuffer(this.normBuf);
    const pass = enc.beginComputePass();
    for (const d of [...this.fwd, ...this.bwd, ...this.clipOps, ...this.optOps]) {
      pass.setPipeline(d.pipeline);
      pass.setBindGroup(0, d.bind);
      pass.dispatchWorkgroups(d.x, d.y, d.z);
    }
    pass.end();
    dev.queue.submit([enc.finish()]);
    await dev.queue.onSubmittedWorkDone();
  }

  /** Forward-only pass on a batch; returns mean loss. */
  async evalLoss(x: Int32Array, y: Int32Array): Promise<number> {
    const dev = this.device;
    dev.queue.writeBuffer(this.ids, 0, Uint32Array.from(x));
    dev.queue.writeBuffer(this.tgt, 0, Uint32Array.from(y));
    const enc = dev.createCommandEncoder();
    const pass = enc.beginComputePass();
    for (const d of this.fwd) {
      pass.setPipeline(d.pipeline);
      pass.setBindGroup(0, d.bind);
      pass.dispatchWorkgroups(d.x, d.y, d.z);
    }
    pass.end();
    dev.queue.submit([enc.finish()]);
    return this.readLoss();
  }

  /** Mean of the per-row loss buffer from the most recent forward. */
  async readLoss(): Promise<number> {
    const dev = this.device;
    const enc = dev.createCommandEncoder();
    enc.copyBufferToBuffer(this.lossBuf, 0, this.lossStage, 0, this.B * this.T * 4);
    dev.queue.submit([enc.finish()]);
    await this.lossStage.mapAsync(GPUMapMode.READ);
    const arr = new Float32Array(this.lossStage.getMappedRange().slice(0));
    this.lossStage.unmap();
    let s = 0;
    for (let i = 0; i < arr.length; i++) s += arr[i];
    return s / arr.length;
  }

  /** Copy GPU weights back into the CPU model (for checkpoints/parity). */
  async readWeights(): Promise<void> {
    for (const [name, t] of this.model.namedParameters()) {
      const data = await this.readBuffer(this.weights.get(name)!, t.size);
      t.data.set(data);
    }
  }

  /** Read a gradient buffer (parity testing). */
  async readGrad(name: string, n: number): Promise<Float32Array> {
    return this.readBuffer(this.grads.get(name)!, n);
  }

  /** Diagnostic: scan forward activations in order; report each buffer's health. */
  async scanActivations(): Promise<Array<{ name: string; max: number; bad: number; firstBadIdx: number }>> {
    const out: Array<{ name: string; max: number; bad: number; firstBadIdx: number }> = [];
    for (const a of this.acts) {
      const d = await this.readBuffer(a.buf, a.n);
      let max = 0, bad = 0, firstBadIdx = -1;
      for (let i = 0; i < d.length; i++) {
        const v = d[i];
        if (!Number.isFinite(v)) {
          if (firstBadIdx < 0) firstBadIdx = i;
          bad++;
        } else if (Math.abs(v) > max) max = Math.abs(v);
      }
      out.push({ name: a.name, max, bad, firstBadIdx });
    }
    return out;
  }

  /** Diagnostic: per-tensor max|w|, max|g| and non-finite counts. */
  async scan(): Promise<Array<{ name: string; wMax: number; wBad: number; gMax: number; gBad: number }>> {
    const out: Array<{ name: string; wMax: number; wBad: number; gMax: number; gBad: number }> = [];
    const stats = (a: Float32Array): { max: number; bad: number } => {
      let max = 0, bad = 0;
      for (let i = 0; i < a.length; i++) {
        const v = a[i];
        if (!Number.isFinite(v)) bad++;
        else if (Math.abs(v) > max) max = Math.abs(v);
      }
      return { max, bad };
    };
    for (const [name, t] of this.model.namedParameters()) {
      const w = stats(await this.readBuffer(this.weights.get(name)!, t.size));
      const g = stats(await this.readBuffer(this.grads.get(name)!, t.size));
      out.push({ name, wMax: w.max, wBad: w.bad, gMax: g.max, gBad: g.bad });
    }
    return out;
  }

  private async readBuffer(src: GPUBuffer, n: number): Promise<Float32Array> {
    const dev = this.device;
    const stage = dev.createBuffer({ size: n * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = dev.createCommandEncoder();
    enc.copyBufferToBuffer(src, 0, stage, 0, n * 4);
    dev.queue.submit([enc.finish()]);
    await stage.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(stage.getMappedRange().slice(0));
    stage.unmap();
    stage.destroy();
    return out;
  }

  destroy(): void {
    this.device.destroy();
  }
}
