// Active-backend registry. Default is Float32 CPU; grad-check tests swap in a
// Float64 CPU backend for tight finite-difference tolerances (DESIGN §19.1).

import type { Backend } from "./backend.ts";
import { CPUBackend } from "./cpu.ts";

export type { Backend, FloatArray, FloatArrayCtor, NdArray, MatmulOpts } from "./backend.ts";
export { CPUBackend } from "./cpu.ts";

let current: Backend = new CPUBackend(Float32Array);

export function B(): Backend {
  return current;
}

export function setBackend(b: Backend): void {
  current = b;
}
