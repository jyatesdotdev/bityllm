// bityllm public API (DESIGN §23).

export { B, setBackend, CPUBackend } from "./backend/index.ts";
export type { Backend, NdArray, FloatArray } from "./backend/index.ts";
export { Tensor, noGrad, tensor, zeros, randn, full } from "./core/tensor.ts";
export { RNG } from "./core/rng.ts";
export * as ops from "./core/ops.ts";
export { Module } from "./nn/module.ts";
export { Linear, LayerNorm, MLP } from "./nn/layers.ts";
export { CausalSelfAttention } from "./nn/attention.ts";
export { GPT, Block } from "./nn/gpt.ts";
export type { GPTConfig } from "./nn/gpt.ts";
export { AdamW, clipGradNorm, cosineLR } from "./optim/adamw.ts";
export { CharTokenizer } from "./tokenizer/char.ts";
export type { Tokenizer } from "./tokenizer/char.ts";
export { Dataset } from "./data/dataset.ts";
export { train } from "./train.ts";
export type { TrainConfig } from "./train.ts";
export { trainParallel } from "./train-parallel.ts";
export type { ParallelTrainConfig } from "./train-parallel.ts";
export { generate } from "./sample.ts";
export type { GenOpts } from "./sample.ts";
export { serialize, serializeInt8, deserialize } from "./io/checkpoint.ts";
export { InferenceSession } from "./infer/session.ts";
export { Shell } from "./infer/shell.ts";
export type { Binary, ShellIO } from "./infer/shell.ts";
export { BINARIES } from "./infer/binaries.ts";
