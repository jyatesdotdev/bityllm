// Differentiable ops (DESIGN §7): forward via Backend kernels, backward as a
// closure capturing only what it needs. All ops here are exercised by the
// finite-difference grad-checks in test/gradcheck.test.ts.

import { B } from "../backend/index.ts";
import type { FloatArray } from "../backend/index.ts";
import { Tensor, opOutput, addInto } from "./tensor.ts";
import type { RNG } from "./rng.ts";

const prod = (s: number[]): number => s.reduce((a, b) => a * b, 1);

/** a + b, with b's shape a suffix of a's shape (bias / row broadcast). */
export function add(a: Tensor, b: Tensor): Tensor {
  const nd = B().add(a.nd, b.nd);
  return opOutput(nd, [a, b], (out) => () => {
    const g = out.grad!;
    if (a.requiresGrad) addInto(a.ensureGrad(), g);
    if (b.requiresGrad) {
      if (b.size === a.size) addInto(b.ensureGrad(), g);
      else addInto(b.ensureGrad(), B().sumEvery(g, b.size)); // reduce broadcast dims (sumTo)
    }
  });
}

/** s · a for scalar constant s. */
export function scale(a: Tensor, s: number): Tensor {
  const nd = B().scale(a.nd, s);
  return opOutput(nd, [a], (out) => () => {
    if (a.requiresGrad) addInto(a.ensureGrad(), B().scale({ data: out.grad!, shape: out.shape }, s).data);
  });
}

/** a @ b. b may be 2-D while a is batched (weight broadcast). */
export function matmul(a: Tensor, b: Tensor): Tensor {
  const nd = B().matmul(a.nd, b.nd);
  return opOutput(nd, [a, b], (out) => () => {
    const g = { data: out.grad!, shape: out.shape };
    if (a.requiresGrad) addInto(a.ensureGrad(), B().matmul(g, b.nd, { transposeB: true }).data);
    if (b.requiresGrad) {
      let gb = B().matmul(a.nd, g, { transposeA: true }); // [...batch, K, N]
      if (b.shape.length === 2 && a.shape.length > 2) gb = B().sumBatch(gb, b.shape);
      addInto(b.ensureGrad(), gb.data);
    }
  });
}

/** a @ bᵀ: a [.., M, K], b [.., N, K] (or [N, K] broadcast) → [.., M, N]. */
export function matmulT(a: Tensor, b: Tensor): Tensor {
  const nd = B().matmul(a.nd, b.nd, { transposeB: true });
  return opOutput(nd, [a, b], (out) => () => {
    const g = { data: out.grad!, shape: out.shape };
    if (a.requiresGrad) addInto(a.ensureGrad(), B().matmul(g, b.nd).data);
    if (b.requiresGrad) {
      let gb = B().matmul(g, a.nd, { transposeA: true }); // gᵀ @ a → [.., N, K]
      if (b.shape.length === 2 && a.shape.length > 2) gb = B().sumBatch(gb, b.shape);
      addInto(b.ensureGrad(), gb.data);
    }
  });
}

export function gelu(a: Tensor): Tensor {
  const nd = B().gelu(a.nd);
  return opOutput(nd, [a], (out) => () => {
    if (a.requiresGrad)
      addInto(a.ensureGrad(), B().geluBackward(a.nd, { data: out.grad!, shape: out.shape }).data);
  });
}

export function softmaxLastDim(a: Tensor): Tensor {
  const nd = B().softmaxLastDim(a.nd);
  return opOutput(nd, [a], (out) => () => {
    if (a.requiresGrad)
      addInto(a.ensureGrad(), B().softmaxBackward(out.nd, { data: out.grad!, shape: out.shape }).data);
  });
}

export function layerNorm(x: Tensor, w: Tensor, b: Tensor, eps = 1e-5): Tensor {
  const { y, mean, rstd } = B().layerNorm(x.nd, w.nd, b.nd, eps);
  return opOutput(y, [x, w, b], (out) => () => {
    const g = { data: out.grad!, shape: out.shape };
    const { dx, dw, db } = B().layerNormBackward(x.nd, w.nd, mean, rstd, g);
    if (x.requiresGrad) addInto(x.ensureGrad(), dx.data);
    if (w.requiresGrad) addInto(w.ensureGrad(), dw);
    if (b.requiresGrad) addInto(b.ensureGrad(), db);
  });
}

/** Row-gather from an embedding table; out is reshaped to outShape. */
export function embedding(ids: Int32Array, table: Tensor, outShape: number[]): Tensor {
  const g = B().gatherRows(table.nd, ids);
  if (g.data.length !== prod(outShape)) throw new Error(`embedding: bad outShape [${outShape}]`);
  return opOutput({ data: g.data, shape: [...outShape] }, [table], (out) => () => {
    if (table.requiresGrad)
      B().scatterAddRows(table.ensureGrad(), table.shape[1], ids, out.grad!);
  });
}

/** View with a new shape (data buffer is shared; tensors are always contiguous). */
export function reshape(a: Tensor, shape: number[]): Tensor {
  if (prod(shape) !== a.size) throw new Error(`reshape: [${a.shape}] → [${shape}]`);
  return opOutput({ data: a.data, shape: [...shape] }, [a], (out) => () => {
    if (a.requiresGrad) addInto(a.ensureGrad(), out.grad!);
  });
}

/** Swap dims 1 and 2 of a 4-D tensor (attention head split/merge). */
export function transpose12(a: Tensor): Tensor {
  const nd = B().transpose12(a.nd);
  return opOutput(nd, [a], (out) => () => {
    if (a.requiresGrad)
      addInto(a.ensureGrad(), B().transpose12({ data: out.grad!, shape: out.shape }).data);
  });
}

/** Causal mask on [.., T, T] attention scores: j > i → -1e30. */
export function causalMask(a: Tensor, T: number): Tensor {
  const nd = B().causalMask(a.nd, T);
  return opOutput(nd, [a], (out) => () => {
    if (a.requiresGrad) {
      const gm = { data: out.grad!.slice() as FloatArray, shape: out.shape };
      B().causalMaskZeroGrad(gm, T);
      addInto(a.ensureGrad(), gm.data);
    }
  });
}

/** Mean cross-entropy over [N, V] logits vs integer targets. Fused & stable. */
export function crossEntropyLogits(logits: Tensor, targets: Int32Array): Tensor {
  if (logits.shape.length !== 2) throw new Error("crossEntropyLogits: expected [N, V]");
  const { loss, lse } = B().ceForward(logits.nd, targets);
  const nd = B().from([loss], [1]);
  return opOutput(nd, [logits], (out) => () => {
    if (logits.requiresGrad)
      addInto(logits.ensureGrad(), B().ceBackward(logits.nd, lse, targets, out.grad![0]).data);
  });
}

/** Inverted dropout: scales kept activations by 1/(1-p). Call only when p > 0. */
export function dropout(a: Tensor, p: number, rng: RNG): Tensor {
  const keep = 1 / (1 - p);
  const mask = new (a.data.constructor as new (n: number) => FloatArray)(a.size);
  for (let i = 0; i < mask.length; i++) mask[i] = rng.random() < p ? 0 : keep;
  const nd = B().mul(a.nd, { data: mask, shape: [...a.shape] });
  return opOutput(nd, [a], (out) => () => {
    if (a.requiresGrad)
      addInto(a.ensureGrad(), B().mul({ data: out.grad!, shape: out.shape }, { data: mask, shape: out.shape }).data);
  });
}
