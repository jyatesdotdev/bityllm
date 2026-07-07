// Shared sampling: temperature + top-k over raw logits → categorical draw.

import type { FloatArray } from "../backend/index.ts";
import type { RNG } from "../core/rng.ts";

export interface SampleOpts {
  temperature?: number; // default 0.8
  topK?: number;        // 0 = disabled
}

export function sampleLogits(logits: ArrayLike<number>, opts: SampleOpts, rng: RNG): number {
  const V = logits.length;
  const temperature = Math.max(opts.temperature ?? 0.8, 1e-6);
  const topK = opts.topK ?? 0;

  const scaled = new Float64Array(V);
  let max = -Infinity;
  for (let j = 0; j < V; j++) {
    scaled[j] = (logits as FloatArray)[j] / temperature;
    if (scaled[j] > max) max = scaled[j];
  }
  if (topK > 0 && topK < V) {
    const kth = [...scaled].sort((a, b) => b - a)[topK - 1];
    for (let j = 0; j < V; j++) if (scaled[j] < kth) scaled[j] = -Infinity;
  }
  let sum = 0;
  for (let j = 0; j < V; j++) {
    scaled[j] = Math.exp(scaled[j] - max);
    sum += scaled[j];
  }
  let r = rng.random() * sum;
  for (let j = 0; j < V; j++) {
    r -= scaled[j];
    if (r <= 0) return j;
  }
  return V - 1;
}
