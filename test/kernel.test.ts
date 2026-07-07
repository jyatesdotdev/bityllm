// Kernel correctness: the blocked/unrolled matmul must match a naive reference
// for every transpose combo, including odd K/N (unroll tails) and batching.

import { test } from "node:test";
import assert from "node:assert/strict";
import { CPUBackend } from "../src/backend/cpu.ts";
import type { NdArray } from "../src/backend/backend.ts";
import { RNG } from "../src/core/rng.ts";

const be = new CPUBackend(Float64Array);
const rng = new RNG(2024);

function rand(shape: number[]): NdArray {
  const nd = be.zeros(shape);
  for (let i = 0; i < nd.data.length; i++) nd.data[i] = rng.randn();
  return nd;
}

/** dead-simple reference: index arithmetic only, no blocking */
function reference(a: NdArray, b: NdArray, tA: boolean, tB: boolean): Float64Array {
  const as = a.shape, bs = b.shape;
  const p = as[as.length - 2], q = as[as.length - 1];
  const r = bs[bs.length - 2], s = bs[bs.length - 1];
  const M = tA ? q : p, K = tA ? p : q, N = tB ? r : s;
  const batch = as.slice(0, -2).reduce((x, y) => x * y, 1);
  const bBroadcast = bs.length === 2 && as.length > 2;
  const out = new Float64Array(batch * M * N);
  for (let ib = 0; ib < batch; ib++)
    for (let m = 0; m < M; m++)
      for (let n = 0; n < N; n++) {
        let acc = 0;
        for (let k = 0; k < K; k++) {
          const av = a.data[ib * p * q + (tA ? k * M + m : m * K + k)];
          const bv = b.data[(bBroadcast ? 0 : ib * r * s) + (tB ? n * K + k : k * N + n)];
          acc += av * bv;
        }
        out[ib * M * N + m * N + n] = acc;
      }
  return out;
}

const close = (x: Float64Array, y: ArrayLike<number>): void => {
  for (let i = 0; i < x.length; i++)
    assert.ok(Math.abs(x[i] - y[i]) < 1e-9, `idx ${i}: ${x[i]} vs ${y[i]}`);
};

test("matmul matches naive reference (all transpose combos, odd shapes, batch)", () => {
  const cases: Array<[number[], number[], boolean, boolean]> = [
    [[5, 7], [7, 3], false, false],
    [[5, 7], [3, 7], false, true],
    [[7, 5], [7, 3], true, false],
    [[7, 5], [3, 7], true, true],
    [[2, 3, 9], [9, 5], false, false],          // 2-D broadcast
    [[2, 3, 9], [5, 9], false, true],
    [[2, 2, 4, 6], [2, 2, 6, 3], false, false], // batched 4-D
    [[2, 2, 4, 6], [2, 2, 5, 6], false, true],  // attention-scores shape
    [[3, 9, 4], [3, 9, 5], true, false],        // grad path
    [[4, 64, 65], [65, 33], false, false],      // odd K/N tails at scale
    [[4, 64, 65], [33, 65], false, true],
  ];
  for (const [sa, sb, tA, tB] of cases) {
    const a = rand(sa), b = rand(sb);
    const got = be.matmul(a, b, { transposeA: tA, transposeB: tB });
    close(reference(a, b, tA, tB), got.data);
  }
});
