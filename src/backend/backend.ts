// Backend interface — every numeric kernel lives behind this seam (DESIGN §5).
// Tensors are flat, row-major, contiguous FloatArrays plus a shape.
//
// The backend knows nothing about autograd: it provides forward kernels and
// (for the fused ops where composing forwards would be slow or unstable)
// explicit backward kernels. The autograd layer in core/ops.ts composes these.
//
// The array constructor is parameterized so grad-checks can run the whole
// stack in Float64 while the real model stays Float32 (DESIGN §19.1).

export type FloatArray = Float32Array | Float64Array;
export type FloatArrayCtor = Float32ArrayConstructor | Float64ArrayConstructor;

export interface NdArray {
  data: FloatArray;
  shape: number[];
}

export interface MatmulOpts {
  transposeA?: boolean;
  transposeB?: boolean;
}

export interface Backend {
  readonly name: string;
  readonly Arr: FloatArrayCtor;

  // allocation
  alloc(n: number): FloatArray;
  zeros(shape: number[]): NdArray;
  from(values: ArrayLike<number>, shape: number[]): NdArray;
  copy(a: NdArray): NdArray;

  // the 90%: batched matmul. Leading dims are batch; last two are the matrix.
  // b may be 2-D while a is batched (b broadcasts across a's batch dims).
  matmul(a: NdArray, b: NdArray, opt?: MatmulOpts): NdArray;

  // elementwise; b's shape must equal a suffix of a's shape (cyclic broadcast)
  add(a: NdArray, b: NdArray): NdArray;
  mul(a: NdArray, b: NdArray): NdArray;
  scale(a: NdArray, s: number): NdArray;

  // reductions used by broadcast backward passes
  sumEvery(a: FloatArray, size: number): FloatArray;     // out[j] = Σ_i a[j + i*size]
  sumBatch(a: NdArray, keepShape: number[]): NdArray;    // reduce leading dims down to keepShape

  // unary
  gelu(a: NdArray): NdArray;
  geluBackward(x: NdArray, g: NdArray): NdArray;

  // fused normalized ops (stable formulations + cached stats)
  softmaxLastDim(a: NdArray): NdArray;
  softmaxBackward(s: NdArray, g: NdArray): NdArray;      // gx = s ⊙ (g − rowsum(g⊙s))
  layerNorm(x: NdArray, w: NdArray, b: NdArray, eps: number): { y: NdArray; mean: FloatArray; rstd: FloatArray };
  layerNormBackward(x: NdArray, w: NdArray, mean: FloatArray, rstd: FloatArray, g: NdArray):
    { dx: NdArray; dw: FloatArray; db: FloatArray };

  // indexing (embeddings)
  gatherRows(table: NdArray, ids: Int32Array): NdArray;
  scatterAddRows(gradTable: FloatArray, cols: number, ids: Int32Array, g: FloatArray): void;

  // shape ops
  transpose12(a: NdArray): NdArray;                      // 4-D: [d0,d1,d2,d3] → [d0,d2,d1,d3]

  // attention causal mask on [.., T, T] scores
  causalMask(a: NdArray, T: number): NdArray;            // positions j>i → -1e30
  causalMaskZeroGrad(g: NdArray, T: number): void;       // zero grad at masked positions

  // fused cross-entropy over [N, V] logits (stable, via logsumexp)
  ceForward(logits: NdArray, targets: Int32Array): { loss: number; lse: FloatArray };
  ceBackward(logits: NdArray, lse: FloatArray, targets: Int32Array, gscale: number): NdArray;
}
