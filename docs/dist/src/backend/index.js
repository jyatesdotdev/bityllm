// Active-backend registry. Default is Float32 CPU; grad-check tests swap in a
// Float64 CPU backend for tight finite-difference tolerances (DESIGN §19.1).
import { CPUBackend } from "./cpu.js";
export { CPUBackend } from "./cpu.js";
let current = new CPUBackend(Float32Array);
export function B() {
    return current;
}
export function setBackend(b) {
    current = b;
}
