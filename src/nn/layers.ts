// Basic layers: Linear, LayerNorm, MLP (DESIGN §8-§9). Each is a thin `Module`
// that composes ops from core/ops.ts — nothing here does its own math or grads.
//
// Two "why"s a learner should carry away:
//   • Init matters. Weights start as small Gaussian noise (std 0.02, GPT-2's
//     default) so early activations/gradients neither vanish nor explode. Zero
//     init would make every neuron identical and unable to differentiate.
//   • The MLP widens ×4 then projects back. That 4× hidden expansion is the
//     transformer's per-position "compute": attention MIXES tokens, the MLP
//     THINKS about each one. The ratio is convention, not magic.

import { Module } from "./module.ts";
import { Tensor, randn, zeros, full } from "../core/tensor.ts";
import * as ops from "../core/ops.ts";
import type { RNG } from "../core/rng.ts";

export class Linear extends Module {
  readonly w: Tensor; // [in, out] — note: PyTorch/MLX store [out, in]; export transposes (invariant #4)
  readonly b: Tensor | null;

  constructor(nIn: number, nOut: number, rng: RNG, opts: { bias?: boolean; std?: number } = {}) {
    super();
    // small-Gaussian init (default std 0.02); some layers pass a smaller `std`
    // (see projStd in gpt.ts) to scale down residual-path projections
    this.w = this.reg(randn([nIn, nOut], rng, opts.std ?? 0.02, true));
    this.b = opts.bias === false ? null : this.reg(zeros([nOut], true)); // bias starts at 0
  }

  forward(x: Tensor): Tensor {
    const y = ops.matmul(x, this.w);
    return this.b ? ops.add(y, this.b) : y;
  }
}

// LayerNorm normalizes each position's feature vector to mean 0 / variance 1,
// then applies a learned per-feature gain (w) and bias (b). It keeps activations
// in a stable range as depth grows — the transformer's analog of BatchNorm.
export class LayerNorm extends Module {
  readonly w: Tensor; // gain, starts at 1 → identity transform before learning
  readonly b: Tensor; // bias, starts at 0

  constructor(n: number) {
    super();
    this.w = this.reg(full([n], 1, true));
    this.b = this.reg(zeros([n], true));
  }

  forward(x: Tensor): Tensor {
    return ops.layerNorm(x, this.w, this.b);
  }
}

export class MLP extends Module {
  readonly fc: Linear;
  readonly proj: Linear;

  constructor(nEmbd: number, rng: RNG, projStd: number) {
    super();
    this.fc = this.sub(new Linear(nEmbd, 4 * nEmbd, rng)); // widen ×4 …
    this.proj = this.sub(new Linear(4 * nEmbd, nEmbd, rng, { std: projStd })); // … then project back down
  }

  forward(x: Tensor): Tensor {
    // up-project → GELU nonlinearity → down-project. Without the nonlinearity,
    // two stacked Linears would collapse into one (a single matmul).
    return this.proj.forward(ops.gelu(this.fc.forward(x)));
  }
}
