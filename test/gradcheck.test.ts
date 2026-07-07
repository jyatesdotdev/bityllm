// Grad-checks (DESIGN §19.1): analytic gradients vs central finite differences
// for every differentiable op, computed in Float64 for tight tolerances.

import { test } from "node:test";
import assert from "node:assert/strict";
import { setBackend, CPUBackend, B } from "../src/backend/index.ts";
import type { FloatArray } from "../src/backend/index.ts";
import { Tensor, noGrad, opOutput, randn, tensor } from "../src/core/tensor.ts";
import { RNG } from "../src/core/rng.ts";
import * as ops from "../src/core/ops.ts";

setBackend(new CPUBackend(Float64Array));

const EPS = 1e-5;
const RTOL = 1e-4;
const ATOL = 1e-7;

/** Deterministic linear functional to reduce any op output to a scalar. */
function weightedSum(t: Tensor, w: FloatArray): Tensor {
  let s = 0;
  for (let i = 0; i < t.size; i++) s += t.data[i] * w[i];
  return opOutput(B().from([s], [1]), [t], (out) => () => {
    const g = out.grad![0];
    const gr = t.ensureGrad();
    for (let i = 0; i < t.size; i++) gr[i] += g * w[i];
  });
}

/**
 * Check d(loss)/d(param) for every element of every param.
 * buildLoss must construct a fresh graph from the same param tensors.
 */
function gradcheck(name: string, buildLoss: () => Tensor, params: Tensor[]): void {
  for (const p of params) p.zeroGrad();
  buildLoss().backward();
  const analytic = params.map((p) => (p.grad ? p.grad.slice() : new Float64Array(p.size)));

  params.forEach((p, pi) => {
    for (let i = 0; i < p.size; i++) {
      const orig = p.data[i];
      p.data[i] = orig + EPS;
      const fp = noGrad(() => buildLoss().item());
      p.data[i] = orig - EPS;
      const fm = noGrad(() => buildLoss().item());
      p.data[i] = orig;
      const num = (fp - fm) / (2 * EPS);
      const ana = analytic[pi][i];
      const tol = ATOL + RTOL * Math.max(Math.abs(num), Math.abs(ana));
      assert.ok(
        Math.abs(num - ana) <= tol,
        `${name}: param#${pi}[${i}] analytic=${ana} numeric=${num} (Δ=${Math.abs(num - ana)})`,
      );
    }
  });
}

const rng = new RNG(1234);
const W = (n: number): Float64Array => {
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) w[i] = rng.randn();
  return w;
};

test("add (same shape + suffix broadcast)", () => {
  const a = randn([2, 3, 4], rng, 1, true);
  const b = randn([2, 3, 4], rng, 1, true);
  const w = W(24);
  gradcheck("add-same", () => weightedSum(ops.add(a, b), w), [a, b]);

  const bias = randn([4], rng, 1, true);
  gradcheck("add-bias", () => weightedSum(ops.add(a, bias), w), [a, bias]);

  const row = randn([3, 4], rng, 1, true);
  gradcheck("add-row", () => weightedSum(ops.add(a, row), w), [a, row]);
});

test("scale", () => {
  const a = randn([3, 5], rng, 1, true);
  const w = W(15);
  gradcheck("scale", () => weightedSum(ops.scale(a, -1.7), w), [a]);
});

test("matmul (batched, 2-D weight broadcast, batched×batched)", () => {
  const x = randn([2, 3, 4], rng, 1, true);
  const wgt = randn([4, 5], rng, 1, true);
  const w1 = W(2 * 3 * 5);
  gradcheck("matmul-broadcast", () => weightedSum(ops.matmul(x, wgt), w1), [x, wgt]);

  const a = randn([2, 3, 4], rng, 1, true);
  const b = randn([2, 4, 5], rng, 1, true);
  gradcheck("matmul-batched", () => weightedSum(ops.matmul(a, b), w1), [a, b]);
});

test("matmulT (attention scores + tied logits shapes)", () => {
  const q = randn([2, 2, 3, 4], rng, 1, true); // [B,H,T,hd]
  const k = randn([2, 2, 3, 4], rng, 1, true);
  const w1 = W(2 * 2 * 3 * 3);
  gradcheck("matmulT-batched", () => weightedSum(ops.matmulT(q, k), w1), [q, k]);

  const x = randn([2, 3, 4], rng, 1, true); // [B,T,C]
  const wte = randn([6, 4], rng, 1, true);  // [V,C] → logits [B,T,V]
  const w2 = W(2 * 3 * 6);
  gradcheck("matmulT-tied", () => weightedSum(ops.matmulT(x, wte), w2), [x, wte]);
});

test("gelu", () => {
  const a = randn([4, 7], rng, 1, true);
  const w = W(28);
  gradcheck("gelu", () => weightedSum(ops.gelu(a), w), [a]);
});

test("softmaxLastDim", () => {
  const a = randn([3, 2, 5], rng, 1, true);
  const w = W(30);
  gradcheck("softmax", () => weightedSum(ops.softmaxLastDim(a), w), [a]);
});

test("layerNorm", () => {
  const x = randn([2, 3, 6], rng, 1, true);
  const g = randn([6], rng, 0.3, true);
  g.data.set(g.data.map((v: number) => v + 1)); // weight ≈ 1
  const b = randn([6], rng, 0.3, true);
  const w = W(36);
  gradcheck("layernorm", () => weightedSum(ops.layerNorm(x, g, b), w), [x, g, b]);
});

test("embedding (repeated ids accumulate)", () => {
  const table = randn([7, 4], rng, 1, true);
  const ids = new Int32Array([1, 3, 1, 0, 6, 1]); // id 1 three times → grad accumulation
  const w = W(6 * 4);
  gradcheck("embedding", () => weightedSum(ops.embedding(ids, table, [2, 3, 4]), w), [table]);
});

test("reshape + transpose12", () => {
  const a = randn([2, 3, 4, 5], rng, 1, true);
  const w = W(120);
  gradcheck("transpose12", () => weightedSum(ops.transpose12(a), w), [a]);
  gradcheck("reshape", () => weightedSum(ops.reshape(a, [6, 20]), w), [a]);
});

test("causalMask (masked positions get zero grad)", () => {
  const T = 4;
  const a = randn([2, 2, T, T], rng, 1, true);
  const s = ops.softmaxLastDim(ops.causalMask(a, T));
  const w = W(a.size);
  gradcheck("causalMask", () => weightedSum(ops.softmaxLastDim(ops.causalMask(a, T)), w), [a]);
  // sanity: masked scores are dead — every row sums to 1 over j ≤ i only
  noGrad(() => {
    const sm = ops.softmaxLastDim(ops.causalMask(a, T));
    assert.ok(Math.abs(sm.data[0 * T + 1]) < 1e-12, "position (0,1) must be masked");
  });
  void s;
});

test("crossEntropyLogits", () => {
  const z = randn([6, 9], rng, 1.5, true);
  const y = new Int32Array([0, 4, 8, 2, 2, 7]);
  gradcheck("crossentropy", () => ops.crossEntropyLogits(z, y), [z]);
});

test("chained ops (mini-MLP end-to-end)", () => {
  const x = randn([2, 3, 4], rng, 1, true);
  const w1 = randn([4, 8], rng, 0.5, true);
  const b1 = randn([8], rng, 0.5, true);
  const w2 = randn([8, 5], rng, 0.5, true);
  const y = new Int32Array([1, 0, 4, 2, 3, 1]);
  gradcheck(
    "mlp-chain",
    () => {
      const h = ops.gelu(ops.add(ops.matmul(x, w1), b1));
      const logits = ops.reshape(ops.matmul(h, w2), [6, 5]);
      return ops.crossEntropyLogits(logits, y);
    },
    [x, w1, b1, w2],
  );
});
