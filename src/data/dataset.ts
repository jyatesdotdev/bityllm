// Dataset (DESIGN §12): one flat token array; batches are B random windows of
// blockSize+1 — x is the first T, y is shifted one. Tail split for validation.

import type { RNG } from "../core/rng.ts";

export interface Batch {
  x: Int32Array; // [B*T]
  y: Int32Array; // [B*T]
}

export class Dataset {
  private readonly train: Int32Array;
  private readonly val: Int32Array;

  constructor(tokens: Int32Array, valSplit = 0.1) {
    const nVal = Math.floor(tokens.length * valSplit);
    this.train = tokens.subarray(0, tokens.length - nVal);
    this.val = tokens.subarray(tokens.length - nVal);
    if (this.train.length < 2) throw new Error("dataset too small");
  }

  get trainTokens(): number {
    return this.train.length;
  }
  get valTokens(): number {
    return this.val.length;
  }

  getBatch(split: "train" | "val", B: number, T: number, rng: RNG): Batch {
    const src = split === "val" && this.val.length > T ? this.val : this.train;
    const x = new Int32Array(B * T);
    const y = new Int32Array(B * T);
    for (let b = 0; b < B; b++) {
      const start = rng.randint(0, src.length - T - 1);
      for (let t = 0; t < T; t++) {
        x[b * T + t] = src[start + t];
        y[b * T + t] = src[start + t + 1];
      }
    }
    return { x, y };
  }
}
