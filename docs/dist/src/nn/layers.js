// Basic layers: Linear, LayerNorm, MLP (DESIGN §8-§9).
import { Module } from "./module.js";
import { Tensor, randn, zeros, full } from "../core/tensor.js";
import * as ops from "../core/ops.js";
export class Linear extends Module {
    w; // [in, out]
    b;
    constructor(nIn, nOut, rng, opts = {}) {
        super();
        this.w = this.reg(randn([nIn, nOut], rng, opts.std ?? 0.02, true));
        this.b = opts.bias === false ? null : this.reg(zeros([nOut], true));
    }
    forward(x) {
        const y = ops.matmul(x, this.w);
        return this.b ? ops.add(y, this.b) : y;
    }
}
export class LayerNorm extends Module {
    w;
    b;
    constructor(n) {
        super();
        this.w = this.reg(full([n], 1, true));
        this.b = this.reg(zeros([n], true));
    }
    forward(x) {
        return ops.layerNorm(x, this.w, this.b);
    }
}
export class MLP extends Module {
    fc;
    proj;
    constructor(nEmbd, rng, projStd) {
        super();
        this.fc = this.sub(new Linear(nEmbd, 4 * nEmbd, rng));
        this.proj = this.sub(new Linear(4 * nEmbd, nEmbd, rng, { std: projStd }));
    }
    forward(x) {
        return this.proj.forward(ops.gelu(this.fc.forward(x)));
    }
}
