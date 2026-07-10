// Autoregressive sampling (DESIGN §13): crop context to blockSize, forward
// under noGrad, temperature + top-k on the last position, seeded categorical
// draw. No KV-cache in v1 — recompute is fine at tiny T.

import { noGrad } from "./core/tensor.ts";
import { RNG } from "./core/rng.ts";
import type { GPT } from "./nn/gpt.ts";
import type { Tokenizer } from "./tokenizer/char.ts";

export interface GenOpts {
  maxNewTokens: number;
  temperature?: number;
  topK?: number;
  seed?: number;
  stop?: string[];
  onToken?: (ch: string) => void; // streaming hook
}

export function generate(model: GPT, tok: Tokenizer, prompt: string, opts: GenOpts): string {
  const wasTraining = model.training;
  model.train(false);
  const rng = new RNG(opts.seed ?? 0x5eed1);
  const temperature = opts.temperature ?? 0.8;
  const topK = opts.topK ?? 0;
  const V = model.cfg.vocabSize;
  const block = model.cfg.blockSize;

  const ctx: number[] = [...tok.encode(prompt)];
  if (ctx.length === 0) ctx.push(0);
  let out = "";

  try {
    for (let n = 0; n < opts.maxNewTokens; n++) {
      const start = Math.max(0, ctx.length - block);
      const window = Int32Array.from(ctx.slice(start));
      const T = window.length;

      const logits = noGrad(() => model.forward(window, 1, T));
      const last = logits.data.subarray((T - 1) * V, T * V);

      // temperature + top-k → softmax → categorical (the inference mirror of CE):
      const probs = new Float64Array(V);
      let max = -Infinity;
      for (let j = 0; j < V; j++) {
        probs[j] = last[j] / Math.max(temperature, 1e-6); // T<1 sharpens, T>1 flattens
        if (probs[j] > max) max = probs[j];
      }
      if (topK > 0 && topK < V) {
        // keep only the k largest logits; set the rest to -Infinity so exp() below
        // makes them exactly 0 → they can never be drawn, no renormalization needed
        const kth = [...probs].sort((a, b) => b - a)[topK - 1];
        for (let j = 0; j < V; j++) if (probs[j] < kth) probs[j] = -Infinity;
      }
      let sum = 0;
      for (let j = 0; j < V; j++) {
        probs[j] = Math.exp(probs[j] - max); // unnormalized softmax weight
        sum += probs[j];
      }
      // Seeded categorical draw by inverse-CDF: pick r uniformly in [0, sum), walk
      // the weights subtracting each until r ≤ 0. Token j is chosen with probability
      // probs[j]/sum — i.e. exactly softmax — without ever normalizing.
      let r = rng.random() * sum;
      let id = V - 1;
      for (let j = 0; j < V; j++) {
        r -= probs[j];
        if (r <= 0) {
          id = j;
          break;
        }
      }

      ctx.push(id);
      const ch = tok.decode([id]);
      out += ch;
      opts.onToken?.(ch);

      if (opts.stop?.some((s) => out.endsWith(s))) break;
    }
  } finally {
    model.train(wasTraining);
  }
  return out;
}
