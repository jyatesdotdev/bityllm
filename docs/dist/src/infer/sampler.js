// Shared sampling: temperature + top-k over raw logits → categorical draw.
//
// Turns the model's output scores (logits) into ONE chosen token:
//   1. temperature — divide logits by T. T<1 sharpens (more confident/repetitive),
//      T>1 flattens (more random/creative), T→0 approaches argmax.
//   2. top-k — optionally keep only the k highest-scoring tokens (set the rest to
//      -∞) so unlikely tokens can never be drawn.
//   3. softmax → a probability distribution, then draw one token from it using the
//      SEEDED rng (so a given seed reproduces a given generation).
// This is the inference-time mirror of cross-entropy: both center on a softmax,
// one to SCORE the distribution, this one to SAMPLE from it.
//
// Note: top-k here fully sorts all V logits (simple, fine for a small char vocab;
// a partial-select would be the idiomatic choice at large vocab sizes).
export function sampleLogits(logits, opts, rng) {
    const V = logits.length;
    const temperature = Math.max(opts.temperature ?? 0.8, 1e-6);
    const topK = opts.topK ?? 0;
    const scaled = new Float64Array(V);
    let max = -Infinity;
    for (let j = 0; j < V; j++) {
        scaled[j] = logits[j] / temperature;
        if (scaled[j] > max)
            max = scaled[j];
    }
    if (topK > 0 && topK < V) {
        const kth = [...scaled].sort((a, b) => b - a)[topK - 1];
        for (let j = 0; j < V; j++)
            if (scaled[j] < kth)
                scaled[j] = -Infinity;
    }
    let sum = 0;
    for (let j = 0; j < V; j++) {
        scaled[j] = Math.exp(scaled[j] - max);
        sum += scaled[j];
    }
    let r = rng.random() * sum;
    for (let j = 0; j < V; j++) {
        r -= scaled[j];
        if (r <= 0)
            return j;
    }
    return V - 1;
}
