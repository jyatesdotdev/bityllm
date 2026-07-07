// bityllm/infer — the lean, browser-safe entry (DESIGN §14).
// No node: imports anywhere in this graph; no trainer, optimizer, or dataset.

export { InferenceSession } from "./infer/session.ts";
export type { StreamOpts } from "./infer/session.ts";
export { GPUInferenceSession } from "./gpu/session.ts";
export { Shell } from "./infer/shell.ts";
export type { Binary, ShellIO, ShellContext, SessionLike } from "./infer/shell.ts";
export { completeCommand, History } from "./infer/shell.ts";
export type { Completion } from "./infer/shell.ts";
export { BINARIES } from "./infer/binaries.ts";
export { sampleLogits } from "./infer/sampler.ts";
export { deserialize } from "./io/checkpoint.ts";
export { CharTokenizer } from "./tokenizer/char.ts";
export { GPT } from "./nn/gpt.ts";
export type { GPTConfig } from "./nn/gpt.ts";
