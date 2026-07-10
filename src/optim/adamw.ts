// AdamW with decoupled weight decay + global-norm clip + warmup/cosine LR
// (DESIGN §10). Decay applies only to the tensors in the `decay` group.
//
// Adam keeps two exponential moving averages (EMAs) PER PARAMETER — each new
// value is blended in with weight (1-beta), so older values decay geometrically:
//   m — EMA of recent gradients          (momentum: which way is downhill lately)
//   v — EMA of recent SQUARED gradients   (how noisy/steep this weight is)
// The update divides the smoothed gradient (m) by √v, so each weight gets its
// OWN effective learning rate: steadily-sloped weights move fast, noisy ones
// move cautiously. AdamW's twist: apply weight decay separately from that
// adaptive step (see `wd * d[i]` below), which regularizes more predictably.
//
// The optimizer only ever touches p.grad and p.data — it knows nothing about
// transformers, so this exact class trains a CNN/RNN/MLP just as well.

import type { Tensor } from "../core/tensor.ts";
import type { FloatArray } from "../backend/index.ts";

export interface AdamWOpts {
  lr: number;
  beta1?: number;
  beta2?: number;
  eps?: number;
  weightDecay?: number;
}

export class AdamW {
  private readonly groups: Array<{ p: Tensor; m: FloatArray; v: FloatArray; decay: boolean }>;
  readonly opts: Required<AdamWOpts>;
  private t = 0;

  constructor(decay: Tensor[], noDecay: Tensor[], opts: AdamWOpts) {
    this.opts = {
      lr: opts.lr,
      beta1: opts.beta1 ?? 0.9,
      beta2: opts.beta2 ?? 0.95,
      eps: opts.eps ?? 1e-8,
      weightDecay: opts.weightDecay ?? 0.1,
    };
    // allocate the m/v state as the SAME typed-array kind as the weight
    // (Float32 in training, Float64 in grad-checks) — that's what the
    // `p.data.constructor as ...` reflection does; it just calls `new Float32Array(n)`.
    const wrap = (p: Tensor, d: boolean) => ({
      p,
      m: new (p.data.constructor as new (n: number) => FloatArray)(p.size),
      v: new (p.data.constructor as new (n: number) => FloatArray)(p.size),
      decay: d,
    });
    this.groups = [...decay.map((p) => wrap(p, true)), ...noDecay.map((p) => wrap(p, false))];
  }

  get params(): Tensor[] {
    return this.groups.map((g) => g.p);
  }

  get stepCount(): number {
    return this.t;
  }

  // `lr` defaults to the configured rate but the training loop passes a per-step
  // value (cosineLR) — so this.opts.lr is really just the fallback/peak.
  step(lr = this.opts.lr): void {
    this.t++;
    const { beta1, beta2, eps, weightDecay } = this.opts;
    // Bias correction: m and v start at 0, so early on they under-estimate the
    // true averages. Dividing by (1 - beta^t) undoes that startup bias; the
    // correction fades to 1 as t grows.
    const bc1 = 1 - Math.pow(beta1, this.t);
    const bc2 = 1 - Math.pow(beta2, this.t);
    for (const { p, m, v, decay } of this.groups) {
      const g = p.grad;
      if (g === null) continue;
      const d = p.data;
      const wd = decay ? weightDecay : 0;
      for (let i = 0; i < d.length; i++) {
        m[i] = beta1 * m[i] + (1 - beta1) * g[i];        // 1st moment: smoothed gradient
        v[i] = beta2 * v[i] + (1 - beta2) * g[i] * g[i]; // 2nd moment: smoothed squared gradient
        const mhat = m[i] / bc1;                          // bias-corrected
        const vhat = v[i] / bc2;
        // THE update: adaptive step (mhat/√vhat) + DECOUPLED weight decay (wd·d).
        // Decay is `wd * d[i]`, applied to the weight directly, NOT folded into
        // the gradient — that decoupling is what the "W" in AdamW means.
        d[i] -= lr * (mhat / (Math.sqrt(vhat) + eps) + wd * d[i]);
      }
    }
  }

  zeroGrad(): void {
    for (const { p } of this.groups) p.zeroGrad();
  }
}

/** Global-norm gradient clipping; returns the pre-clip norm.
 *  Treats ALL gradients as one big vector, measures its length (L2 norm), and if
 *  that exceeds maxNorm, scales every gradient by the same factor — preserving
 *  direction, capping magnitude. This tames the occasional huge-gradient batch
 *  that would otherwise blow up training (a staple far beyond LLMs, e.g. RNNs). */
export function clipGradNorm(params: Tensor[], maxNorm: number): number {
  let sq = 0;
  for (const p of params) {
    const g = p.grad;
    if (g === null) continue;
    for (let i = 0; i < g.length; i++) sq += g[i] * g[i];
  }
  const norm = Math.sqrt(sq);
  if (norm > maxNorm && norm > 0) {
    const s = maxNorm / norm;
    for (const p of params) {
      const g = p.grad;
      if (g === null) continue;
      for (let i = 0; i < g.length; i++) g[i] *= s;
    }
  }
  return norm;
}

/** Linear warmup → cosine decay to minRatio·lr (DESIGN §10). */
export function cosineLR(step: number, opts: { lr: number; warmup: number; total: number; minRatio: number }): number {
  const { lr, warmup, total, minRatio } = opts;
  if (step < warmup) return (lr * (step + 1)) / warmup;
  const t = Math.min(1, (step - warmup) / Math.max(1, total - warmup));
  return lr * (minRatio + (1 - minRatio) * 0.5 * (1 + Math.cos(Math.PI * t)));
}
