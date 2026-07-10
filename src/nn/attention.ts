// Multi-head causal self-attention (DESIGN §9).
//
//   q,k,v = Wq·x, Wk·x, Wv·x            [B,T,C]
//   split heads → [B,nH,T,hd]
//   att = softmax(mask(q·kᵀ / √hd))      [B,nH,T,T]
//   y   = att·v → merge heads → Wo·y     [B,T,C]

import { Module } from "./module.ts";
import { Linear } from "./layers.ts";
import type { Tensor } from "../core/tensor.ts";
import * as ops from "../core/ops.ts";
import type { RNG } from "../core/rng.ts";

export class CausalSelfAttention extends Module {
  readonly nHead: number;
  readonly wq: Linear;
  readonly wk: Linear;
  readonly wv: Linear;
  readonly wo: Linear;

  constructor(nEmbd: number, nHead: number, rng: RNG, projStd: number) {
    super();
    if (nEmbd % nHead !== 0) throw new Error(`nEmbd ${nEmbd} not divisible by nHead ${nHead}`);
    this.nHead = nHead;
    this.wq = this.sub(new Linear(nEmbd, nEmbd, rng));
    this.wk = this.sub(new Linear(nEmbd, nEmbd, rng));
    this.wv = this.sub(new Linear(nEmbd, nEmbd, rng));
    this.wo = this.sub(new Linear(nEmbd, nEmbd, rng, { std: projStd }));
  }

  forward(x: Tensor): Tensor {
    const [B, T, C] = x.shape;
    const nH = this.nHead;
    const hd = C / nH;

    const split = (t: Tensor): Tensor => ops.transpose12(ops.reshape(t, [B, T, nH, hd])); // → [B,nH,T,hd]
    const q = split(this.wq.forward(x));
    const k = split(this.wk.forward(x));
    const v = split(this.wv.forward(x));

    // scores[i,j] = how much token i attends to token j = (qᵢ · kⱼ) / √head_dim.
    //   • /√head_dim keeps the dot products from growing with dimension, which
    //     would otherwise push softmax into a near-one-hot regime with tiny gradients.
    //   • causalMask sets j > i to a large negative value (-1e30, effectively -∞ for
    //     softmax but finite to avoid inf−inf NaNs) BEFORE softmax, so token i attends only to
    //     tokens ≤ i. THIS is what makes it an autoregressive LM (no peeking at the
    //     future); drop this one call and the same block becomes a ViT/encoder block.
    const scores = ops.causalMask(ops.scale(ops.matmulT(q, k), 1 / Math.sqrt(hd)), T); // [B,nH,T,T]
    const att = ops.softmaxLastDim(scores); // rows sum to 1: a weighting over past tokens
    const y = ops.reshape(ops.transpose12(ops.matmul(att, v)), [B, T, C]); // weighted sum of V, merge heads
    return this.wo.forward(y);
  }
}
