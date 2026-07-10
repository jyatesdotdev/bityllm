// Tensor + define-by-run autograd tape (DESIGN §6).
//
// Ops (core/ops.ts) build the graph during the forward pass by pushing output
// tensors onto a global tape. backward() seeds the scalar loss gradient and
// walks the tape in reverse insertion order — a valid reverse-topological
// order, since children are always created after their parents. Backward
// closures ACCUMULATE (+=) into parent grads, which is what makes shared
// parameters (tied embedding / lm_head) correct automatically.

import type { FloatArray, NdArray } from "../backend/index.ts";
import { B } from "../backend/index.ts";
import { RNG } from "./rng.ts";

let tape: Tensor[] = [];
let gradEnabled = true;

/** Run fn with the tape disabled: no graph, no activation retention (eval/generation). */
export function noGrad<T>(fn: () => T): T {
  const prev = gradEnabled;
  gradEnabled = false;
  try {
    return fn();
  } finally {
    gradEnabled = prev;
  }
}

export function isGradEnabled(): boolean {
  return gradEnabled;
}

export class Tensor {
  data: FloatArray;
  shape: number[];
  grad: FloatArray | null = null;
  requiresGrad: boolean;
  parents: Tensor[] = [];
  backwardFn: (() => void) | null = null;
  label?: string;

  constructor(nd: NdArray, requiresGrad = false, label?: string) {
    this.data = nd.data;
    this.shape = nd.shape;
    this.requiresGrad = requiresGrad;
    this.label = label;
  }

  get size(): number {
    return this.data.length;
  }

  get nd(): NdArray {
    return { data: this.data, shape: this.shape };
  }

  item(): number {
    if (this.size !== 1) throw new Error(`item() on tensor of size ${this.size}`);
    return this.data[0];
  }

  /** Lazily allocate the gradient buffer (same dtype as data). */
  ensureGrad(): FloatArray {
    if (this.grad === null) {
      this.grad = new (this.data.constructor as new (n: number) => FloatArray)(this.data.length);
    }
    return this.grad;
  }

  zeroGrad(): void {
    if (this.grad !== null) this.grad.fill(0);
  }

  /** Reverse-mode sweep from this scalar; consumes and clears the tape. */
  backward(): void {
    if (this.size !== 1) throw new Error("backward() requires a scalar loss");
    this.ensureGrad()[0] = 1; // seed the chain rule: dL/dL = 1; every other grad derives from this
    for (let i = tape.length - 1; i >= 0; i--) {
      const t = tape[i];
      if (t.backwardFn !== null && t.grad !== null) t.backwardFn();
      t.backwardFn = null; // release closures (and captured activations)
      t.parents = [];
    }
    tape = [];
  }
}

/** Internal: wrap an op output, wiring it onto the tape when grads are on. */
export function opOutput(nd: NdArray, parents: Tensor[], makeBackward: (out: Tensor) => () => void): Tensor {
  // Gradient gating: only record this node on the tape if some parent needs a
  // gradient (i.e. its ancestry reaches a learnable parameter). Ops on pure
  // data/eval inputs build no graph and cost no memory — this is how noGrad()
  // and frozen inputs stay free.
  const req = parents.some((p) => p.requiresGrad);
  const out = new Tensor(nd, req);
  if (gradEnabled && req) {
    out.parents = parents;
    out.backwardFn = makeBackward(out);
    tape.push(out);
  }
  return out;
}

/** += accumulate src into dst. */
export function addInto(dst: FloatArray, src: FloatArray): void {
  for (let i = 0; i < dst.length; i++) dst[i] += src[i];
}

// ---- tensor creation --------------------------------------------------------

export function tensor(values: ArrayLike<number>, shape: number[], requiresGrad = false): Tensor {
  return new Tensor(B().from(values, shape), requiresGrad);
}

export function zeros(shape: number[], requiresGrad = false): Tensor {
  return new Tensor(B().zeros(shape), requiresGrad);
}

export function randn(shape: number[], rng: RNG, std = 1, requiresGrad = false): Tensor {
  const nd = B().zeros(shape);
  for (let i = 0; i < nd.data.length; i++) nd.data[i] = rng.randn() * std;
  return new Tensor(nd, requiresGrad);
}

export function full(shape: number[], v: number, requiresGrad = false): Tensor {
  const nd = B().zeros(shape);
  nd.data.fill(v);
  return new Tensor(nd, requiresGrad);
}
