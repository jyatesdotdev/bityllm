// The GPT model (DESIGN §9): pre-norm blocks, learned absolute positions,
// weight-tied LM head (logits = x · wteᵀ — the tie is free with autograd's
// += accumulation), GPT-2 init with 1/√(2L) residual-projection scaling.

import { Module } from "./module.ts";
import { Linear, LayerNorm, MLP } from "./layers.ts";
import { CausalSelfAttention } from "./attention.ts";
import { Tensor, randn } from "../core/tensor.ts";
import * as ops from "../core/ops.ts";
import { RNG } from "../core/rng.ts";

export interface GPTConfig {
  vocabSize: number;
  blockSize: number;
  nLayer: number;
  nHead: number;
  nEmbd: number;
  dropout?: number;
}

export class Block extends Module {
  readonly ln1: LayerNorm;
  readonly attn: CausalSelfAttention;
  readonly ln2: LayerNorm;
  readonly mlp: MLP;

  constructor(cfg: GPTConfig, rng: RNG) {
    super();
    const projStd = 0.02 / Math.sqrt(2 * cfg.nLayer);
    this.ln1 = this.sub(new LayerNorm(cfg.nEmbd));
    this.attn = this.sub(new CausalSelfAttention(cfg.nEmbd, cfg.nHead, rng, projStd));
    this.ln2 = this.sub(new LayerNorm(cfg.nEmbd));
    this.mlp = this.sub(new MLP(cfg.nEmbd, rng, projStd));
  }

  forward(x: Tensor): Tensor {
    x = ops.add(x, this.attn.forward(this.ln1.forward(x)));
    x = ops.add(x, this.mlp.forward(this.ln2.forward(x)));
    return x;
  }
}

export class GPT extends Module {
  readonly cfg: GPTConfig;
  readonly wte: Tensor; // [V, C] — also the (tied) LM head
  readonly wpe: Tensor; // [blockSize, C]
  readonly blocks: Block[];
  readonly lnf: LayerNorm;
  private readonly posIds: Int32Array;
  private readonly dropRng: RNG;

  constructor(cfg: GPTConfig, rng: RNG) {
    super();
    this.cfg = cfg;
    this.wte = this.reg(randn([cfg.vocabSize, cfg.nEmbd], rng, 0.02, true));
    this.wpe = this.reg(randn([cfg.blockSize, cfg.nEmbd], rng, 0.02, true));
    this.blocks = [];
    for (let i = 0; i < cfg.nLayer; i++) this.blocks.push(this.sub(new Block(cfg, rng)));
    this.lnf = this.sub(new LayerNorm(cfg.nEmbd));
    this.posIds = new Int32Array(cfg.blockSize);
    for (let i = 0; i < cfg.blockSize; i++) this.posIds[i] = i;
    this.dropRng = new RNG(0xb17b17);
  }

  /** idx: [B*T] token ids → logits [B, T, V]. */
  forward(idx: Int32Array, B: number, T: number): Tensor {
    if (T > this.cfg.blockSize) throw new Error(`T=${T} exceeds blockSize=${this.cfg.blockSize}`);
    const C = this.cfg.nEmbd;
    const tok = ops.embedding(idx, this.wte, [B, T, C]);
    const pos = ops.embedding(this.posIds.subarray(0, T), this.wpe, [T, C]);
    let x = ops.add(tok, pos); // pos broadcasts over batch
    const p = this.cfg.dropout ?? 0;
    if (p > 0 && this.training) x = ops.dropout(x, p, this.dropRng);
    for (const b of this.blocks) x = b.forward(x);
    x = this.lnf.forward(x);
    return ops.matmulT(x, this.wte); // tied head: [B,T,C] @ [V,C]ᵀ
  }

  /** Mean next-token cross-entropy over the batch. */
  loss(idx: Int32Array, targets: Int32Array, B: number, T: number): Tensor {
    const logits = this.forward(idx, B, T);
    return ops.crossEntropyLogits(ops.reshape(logits, [B * T, this.cfg.vocabSize]), targets);
  }

  paramCount(): number {
    return this.parameters().reduce((n, p) => n + p.size, 0);
  }

  /** AdamW param groups: decay 2-D matmul weights; not biases/LN/embeddings. */
  paramGroups(): { decay: Tensor[]; noDecay: Tensor[] } {
    const decay: Tensor[] = [];
    const noDecay: Tensor[] = [this.wte, this.wpe];
    for (const b of this.blocks) {
      for (const lin of [b.attn.wq, b.attn.wk, b.attn.wv, b.attn.wo, b.mlp.fc, b.mlp.proj]) {
        decay.push(lin.w);
        if (lin.b) noDecay.push(lin.b);
      }
      noDecay.push(b.ln1.w, b.ln1.b, b.ln2.w, b.ln2.b);
    }
    noDecay.push(this.lnf.w, this.lnf.b);
    return { decay, noDecay };
  }

  /** Stable name → tensor mapping for checkpoints. */
  namedParameters(): Array<[string, Tensor]> {
    const out: Array<[string, Tensor]> = [
      ["wte", this.wte],
      ["wpe", this.wpe],
    ];
    const lin = (name: string, l: Linear): void => {
      out.push([`${name}.w`, l.w]);
      if (l.b) out.push([`${name}.b`, l.b]);
    };
    this.blocks.forEach((b, i) => {
      out.push([`h${i}.ln1.w`, b.ln1.w], [`h${i}.ln1.b`, b.ln1.b]);
      lin(`h${i}.attn.wq`, b.attn.wq);
      lin(`h${i}.attn.wk`, b.attn.wk);
      lin(`h${i}.attn.wv`, b.attn.wv);
      lin(`h${i}.attn.wo`, b.attn.wo);
      out.push([`h${i}.ln2.w`, b.ln2.w], [`h${i}.ln2.b`, b.ln2.b]);
      lin(`h${i}.mlp.fc`, b.mlp.fc);
      lin(`h${i}.mlp.proj`, b.mlp.proj);
    });
    out.push(["lnf.w", this.lnf.w], ["lnf.b", this.lnf.b]);
    return out;
  }
}
