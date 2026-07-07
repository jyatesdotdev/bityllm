// Backend interface — every numeric kernel lives behind this seam (DESIGN §5).
// Tensors are flat, row-major, contiguous FloatArrays plus a shape.
//
// The backend knows nothing about autograd: it provides forward kernels and
// (for the fused ops where composing forwards would be slow or unstable)
// explicit backward kernels. The autograd layer in core/ops.ts composes these.
//
// The array constructor is parameterized so grad-checks can run the whole
// stack in Float64 while the real model stays Float32 (DESIGN §19.1).
export {};
