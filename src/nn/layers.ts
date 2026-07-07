// Basic layers: Linear, LayerNorm, MLP (DESIGN §8-§9).

import { Module } from "./module.ts";
import { Tensor, randn, zeros, full } from "../core/tensor.ts";
import * as ops from "../core/ops.ts";
import type { RNG } from "../core/rng.ts";

export class Linear extends Module {
  readonly w: Tensor; // [in, out]
  readonly b: Tensor | null;

  constructor(nIn: number, nOut: number, rng: RNG, opts: { bias?: boolean; std?: number } = {}) {
    super();
    this.w = this.reg(randn([nIn, nOut], rng, opts.std ?? 0.02, true));
    this.b = opts.bias === false ? null : this.reg(zeros([nOut], true));
  }

  forward(x: Tensor): Tensor {
    const y = ops.matmul(x, this.w);
    return this.b ? ops.add(y, this.b) : y;
  }
}

export class LayerNorm extends Module {
  readonly w: Tensor;
  readonly b: Tensor;

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
    this.fc = this.sub(new Linear(nEmbd, 4 * nEmbd, rng));
    this.proj = this.sub(new Linear(4 * nEmbd, nEmbd, rng, { std: projStd }));
  }

  forward(x: Tensor): Tensor {
    return this.proj.forward(ops.gelu(this.fc.forward(x)));
  }
}
