// WGSL kernels for the WebGPU training backend (DESIGN M6).
//
// --- WGSL crash-course (read once; applies to every kernel below) ---
// A compute shader runs the SAME `main` function across thousands of parallel
// threads on the GPU. Key vocabulary:
//   @workgroup_size(N)         — N threads form one "workgroup" that shares fast
//                                on-chip memory and can synchronize.
//   @builtin(global_invocation_id) — this thread's unique (x,y,z) index; kernels
//                                use it to pick which output element THEY compute.
//   var<storage>               — a big buffer in slow global GPU memory (the tensors).
//   var<workgroup>             — small, fast memory shared within one workgroup
//                                (used below to cache matmul tiles).
//   workgroupBarrier()         — wait until all threads in the workgroup reach here.
// The art of a GPU kernel is minimizing slow global-memory reads by cooperatively
// staging data into fast shared memory — that's what the tiled GEMM below does.
//
// One über-GEMM covers every matmul in fwd+bwd (batching via inner/outer
// strides, transpose flags, alpha/beta accumulate, fused bias); row-wise
// kernels handle the normalized ops; elementwise kernels do the rest.
// All f32, all deterministic (fixed reduction orders).
// ---- über-GEMM ---------------------------------------------------------------
// C[z] = alpha * op(A[z]) @ op(B[z]) + beta * C[z] (+ bias)
// z ∈ [0, batch); offsets: off = (z / inner) * outerStride + (z % inner) * innerStride
// ta: A stored [K, lda≥M] read as A(m,k) = a[k*lda + m]; else [M, lda≥K].
// tb: B stored [N, ldb≥K] read as B(k,n) = b[n*ldb + k]; else [K, ldb≥N].
export const GEMM = /* wgsl */ `
struct P {
  M: u32, N: u32, K: u32, batch: u32,
  inner: u32, aOut: u32, aIn: u32, lda: u32,
  bOut: u32, bIn: u32, ldb: u32, ldc: u32,
  cOut: u32, cIn: u32, ta: u32, tb: u32,
  alpha: f32, beta: f32, useBias: u32, pad: u32,
}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;
@group(0) @binding(4) var<storage, read> bias: array<f32>;

var<workgroup> As: array<f32, 256>;
var<workgroup> Bs: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let z = wid.z;
  let ao = (z / p.inner) * p.aOut + (z % p.inner) * p.aIn;
  let bo = (z / p.inner) * p.bOut + (z % p.inner) * p.bIn;
  let co = (z / p.inner) * p.cOut + (z % p.inner) * p.cIn;
  let row = gid.y;
  let col = gid.x;
  var acc = 0.0;
  let tiles = (p.K + 15u) / 16u;
  for (var t = 0u; t < tiles; t = t + 1u) {
    let ak = t * 16u + lid.x;
    let bk = t * 16u + lid.y;
    var av = 0.0;
    if (row < p.M && ak < p.K) {
      if (p.ta == 1u) { av = A[ao + ak * p.lda + row]; } else { av = A[ao + row * p.lda + ak]; }
    }
    var bv = 0.0;
    if (bk < p.K && col < p.N) {
      if (p.tb == 1u) { bv = B[bo + col * p.ldb + bk]; } else { bv = B[bo + bk * p.ldb + col]; }
    }
    As[lid.y * 16u + lid.x] = av;
    Bs[lid.y * 16u + lid.x] = bv;
    workgroupBarrier();
    for (var k = 0u; k < 16u; k = k + 1u) {
      acc = acc + As[lid.y * 16u + k] * Bs[k * 16u + lid.x];
    }
    workgroupBarrier();
  }
  if (row < p.M && col < p.N) {
    let idx = co + row * p.ldc + col;
    var v = p.alpha * acc;
    if (p.useBias == 1u) { v = v + bias[col]; }
    if (p.beta != 0.0) { v = v + p.beta * C[idx]; }
    C[idx] = v;
  }
}`;
// ---- row-wise: causal softmax fwd/bwd -----------------------------------------
// rows of length T; row r belongs to query position t = r % T; cols > t masked.
export const SOFTMAX_FWD = /* wgsl */ `
struct P { T: u32, rows: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> S: array<f32>;
var<workgroup> red: array<f32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let r = wid.x;
  if (r >= p.rows) { return; }
  let t = r % p.T;
  let base = r * p.T;
  var m = -3.0e38;
  for (var j = lid.x; j <= t; j = j + 128u) { m = max(m, S[base + j]); }
  red[lid.x] = m;
  workgroupBarrier();
  for (var s = 64u; s > 0u; s = s >> 1u) {
    if (lid.x < s) { red[lid.x] = max(red[lid.x], red[lid.x + s]); }
    workgroupBarrier();
  }
  let mx = red[0];
  workgroupBarrier();
  var sum = 0.0;
  for (var j = lid.x; j <= t; j = j + 128u) {
    let e = exp(S[base + j] - mx);
    S[base + j] = e;
    sum = sum + e;
  }
  red[lid.x] = sum;
  workgroupBarrier();
  for (var s = 64u; s > 0u; s = s >> 1u) {
    if (lid.x < s) { red[lid.x] = red[lid.x] + red[lid.x + s]; }
    workgroupBarrier();
  }
  let inv = 1.0 / red[0];
  for (var j = lid.x; j < p.T; j = j + 128u) {
    if (j <= t) { S[base + j] = S[base + j] * inv; } else { S[base + j] = 0.0; }
  }
}`;
// dS = P ⊙ (dP − Σ_j P·dP), masked positions → 0. dP in DP, probs in PR, out DS.
export const SOFTMAX_BWD = /* wgsl */ `
struct P { T: u32, rows: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> PR: array<f32>;
@group(0) @binding(2) var<storage, read> DP: array<f32>;
@group(0) @binding(3) var<storage, read_write> DS: array<f32>;
var<workgroup> red: array<f32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let r = wid.x;
  if (r >= p.rows) { return; }
  let t = r % p.T;
  let base = r * p.T;
  var dot = 0.0;
  for (var j = lid.x; j <= t; j = j + 128u) { dot = dot + PR[base + j] * DP[base + j]; }
  red[lid.x] = dot;
  workgroupBarrier();
  for (var s = 64u; s > 0u; s = s >> 1u) {
    if (lid.x < s) { red[lid.x] = red[lid.x] + red[lid.x + s]; }
    workgroupBarrier();
  }
  let d = red[0];
  for (var j = lid.x; j < p.T; j = j + 128u) {
    if (j <= t) { DS[base + j] = PR[base + j] * (DP[base + j] - d); } else { DS[base + j] = 0.0; }
  }
}`;
// ---- row-wise: layernorm ------------------------------------------------------
export const LN_FWD = /* wgsl */ `
struct P { C: u32, rows: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read> Bp: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> MEAN: array<f32>;
@group(0) @binding(6) var<storage, read_write> RSTD: array<f32>;
var<workgroup> red: array<f32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let r = wid.x;
  if (r >= p.rows) { return; }
  let base = r * p.C;
  var s = 0.0;
  for (var j = lid.x; j < p.C; j = j + 128u) { s = s + X[base + j]; }
  red[lid.x] = s;
  workgroupBarrier();
  for (var st = 64u; st > 0u; st = st >> 1u) {
    if (lid.x < st) { red[lid.x] = red[lid.x] + red[lid.x + st]; }
    workgroupBarrier();
  }
  let mu = red[0] / f32(p.C);
  workgroupBarrier();
  var v = 0.0;
  for (var j = lid.x; j < p.C; j = j + 128u) {
    let d = X[base + j] - mu;
    v = v + d * d;
  }
  red[lid.x] = v;
  workgroupBarrier();
  for (var st = 64u; st > 0u; st = st >> 1u) {
    if (lid.x < st) { red[lid.x] = red[lid.x] + red[lid.x + st]; }
    workgroupBarrier();
  }
  let rstd = inverseSqrt(red[0] / f32(p.C) + 1e-5);
  if (lid.x == 0u) { MEAN[r] = mu; RSTD[r] = rstd; }
  for (var j = lid.x; j < p.C; j = j + 128u) {
    Y[base + j] = (X[base + j] - mu) * rstd * W[j] + Bp[j];
  }
}`;
// dx = rstd*(g⊙w − m1 − xhat⊙m2); acc=1 accumulates into DX.
export const LN_BWD_DX = /* wgsl */ `
struct P { C: u32, rows: u32, acc: u32, pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read> G: array<f32>;
@group(0) @binding(4) var<storage, read> MEAN: array<f32>;
@group(0) @binding(5) var<storage, read> RSTD: array<f32>;
@group(0) @binding(6) var<storage, read_write> DX: array<f32>;
var<workgroup> red: array<f32, 128>;
var<workgroup> red2: array<f32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let r = wid.x;
  if (r >= p.rows) { return; }
  let base = r * p.C;
  let mu = MEAN[r];
  let rstd = RSTD[r];
  var s1 = 0.0;
  var s2 = 0.0;
  for (var j = lid.x; j < p.C; j = j + 128u) {
    let gw = G[base + j] * W[j];
    s1 = s1 + gw;
    s2 = s2 + gw * (X[base + j] - mu) * rstd;
  }
  red[lid.x] = s1;
  red2[lid.x] = s2;
  workgroupBarrier();
  for (var st = 64u; st > 0u; st = st >> 1u) {
    if (lid.x < st) {
      red[lid.x] = red[lid.x] + red[lid.x + st];
      red2[lid.x] = red2[lid.x] + red2[lid.x + st];
    }
    workgroupBarrier();
  }
  let m1 = red[0] / f32(p.C);
  let m2 = red2[0] / f32(p.C);
  for (var j = lid.x; j < p.C; j = j + 128u) {
    let xhat = (X[base + j] - mu) * rstd;
    let dx = rstd * (G[base + j] * W[j] - m1 - xhat * m2);
    if (p.acc == 1u) { DX[base + j] = DX[base + j] + dx; } else { DX[base + j] = dx; }
  }
}`;
// per-column: dw[j] = Σ_r g⊙xhat, db[j] = Σ_r g (column-parallel, row loop)
export const LN_BWD_DWDB = /* wgsl */ `
struct P { C: u32, rows: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> G: array<f32>;
@group(0) @binding(3) var<storage, read> MEAN: array<f32>;
@group(0) @binding(4) var<storage, read> RSTD: array<f32>;
@group(0) @binding(5) var<storage, read_write> DW: array<f32>;
@group(0) @binding(6) var<storage, read_write> DB: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j = gid.x;
  if (j >= p.C) { return; }
  var dw = 0.0;
  var db = 0.0;
  for (var r = 0u; r < p.rows; r = r + 1u) {
    let g = G[r * p.C + j];
    dw = dw + g * (X[r * p.C + j] - MEAN[r]) * RSTD[r];
    db = db + g;
  }
  DW[j] = DW[j] + dw;
  DB[j] = DB[j] + db;
}`;
// ---- column sum (bias gradients): db[j] += Σ_r G[r*ld + j] --------------------
export const COLSUM = /* wgsl */ `
struct P { cols: u32, rows: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read_write> OUT: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j = gid.x;
  if (j >= p.cols) { return; }
  var s = 0.0;
  for (var r = 0u; r < p.rows; r = r + 1u) { s = s + G[r * p.cols + j]; }
  OUT[j] = OUT[j] + s;
}`;
// ---- elementwise --------------------------------------------------------------
export const ADD = /* wgsl */ `
struct P { n: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x < p.n) { C[gid.x] = A[gid.x] + B[gid.x]; }
}`;
export const GELU_FWD = /* wgsl */ `
struct P { n: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.n) { return; }
  let x = X[gid.x];
  // clamp: Metal fast-math tanh = (e^2u-1)/(e^2u+1) overflows f32 for |u|>44.36
  // (→ inf/inf → NaN at x ≈ ±10.06). tanh saturates by |u|=15 anyway.
  let u = clamp(0.7978845608028654 * (x + 0.044715 * x * x * x), -15.0, 15.0);
  Y[gid.x] = 0.5 * x * (1.0 + tanh(u));
}`;
export const GELU_BWD = /* wgsl */ `
struct P { n: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> G: array<f32>;
@group(0) @binding(3) var<storage, read_write> DX: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.n) { return; }
  let x = X[gid.x];
  // same fast-math tanh overflow guard as GELU_FWD
  let u = clamp(0.7978845608028654 * (x + 0.044715 * x * x * x), -15.0, 15.0);
  let t = tanh(u);
  let du = 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * x * x);
  DX[gid.x] = G[gid.x] * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du);
}`;
// ---- embedding ----------------------------------------------------------------
export const EMBED_FWD = /* wgsl */ `
struct P { C: u32, T: u32, n: u32, pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> IDS: array<u32>;
@group(0) @binding(2) var<storage, read> WTE: array<f32>;
@group(0) @binding(3) var<storage, read> WPE: array<f32>;
@group(0) @binding(4) var<storage, read_write> X: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.n) { return; }
  let i = gid.x / p.C;
  let j = gid.x % p.C;
  X[gid.x] = WTE[IDS[i] * p.C + j] + WPE[(i % p.T) * p.C + j];
}`;
// dwte[v,j] += Σ_{i: ids[i]==v} dx[i,j] — one thread per (v,j), race-free.
export const EMBED_BWD_WTE = /* wgsl */ `
struct P { C: u32, V: u32, BT: u32, pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> IDS: array<u32>;
@group(0) @binding(2) var<storage, read> DX: array<f32>;
@group(0) @binding(3) var<storage, read_write> DWTE: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.V * p.C) { return; }
  let v = gid.x / p.C;
  let j = gid.x % p.C;
  var s = 0.0;
  for (var i = 0u; i < p.BT; i = i + 1u) {
    if (IDS[i] == v) { s = s + DX[i * p.C + j]; }
  }
  DWTE[gid.x] = DWTE[gid.x] + s;
}`;
export const EMBED_BWD_WPE = /* wgsl */ `
struct P { C: u32, T: u32, B: u32, pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> DX: array<f32>;
@group(0) @binding(2) var<storage, read_write> DWPE: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.T * p.C) { return; }
  let t = gid.x / p.C;
  let j = gid.x % p.C;
  var s = 0.0;
  for (var b = 0u; b < p.B; b = b + 1u) { s = s + DX[(b * p.T + t) * p.C + j]; }
  DWPE[gid.x] = DWPE[gid.x] + s;
}`;
// ---- cross entropy ------------------------------------------------------------
export const CE_FWD = /* wgsl */ `
struct P { V: u32, rows: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Z: array<f32>;
@group(0) @binding(2) var<storage, read> TGT: array<u32>;
@group(0) @binding(3) var<storage, read_write> LSE: array<f32>;
@group(0) @binding(4) var<storage, read_write> LOSS: array<f32>;
var<workgroup> red: array<f32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let r = wid.x;
  if (r >= p.rows) { return; }
  let base = r * p.V;
  var m = -3.0e38;
  for (var j = lid.x; j < p.V; j = j + 128u) { m = max(m, Z[base + j]); }
  red[lid.x] = m;
  workgroupBarrier();
  for (var s = 64u; s > 0u; s = s >> 1u) {
    if (lid.x < s) { red[lid.x] = max(red[lid.x], red[lid.x + s]); }
    workgroupBarrier();
  }
  let mx = red[0];
  workgroupBarrier();
  var sum = 0.0;
  for (var j = lid.x; j < p.V; j = j + 128u) { sum = sum + exp(Z[base + j] - mx); }
  red[lid.x] = sum;
  workgroupBarrier();
  for (var s = 64u; s > 0u; s = s >> 1u) {
    if (lid.x < s) { red[lid.x] = red[lid.x] + red[lid.x + s]; }
    workgroupBarrier();
  }
  if (lid.x == 0u) {
    let lse = mx + log(red[0]);
    LSE[r] = lse;
    LOSS[r] = lse - Z[base + TGT[r]];
  }
}`;
export const CE_BWD = /* wgsl */ `
struct P { V: u32, rows: u32, invN: f32, pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Z: array<f32>;
@group(0) @binding(2) var<storage, read> TGT: array<u32>;
@group(0) @binding(3) var<storage, read> LSE: array<f32>;
@group(0) @binding(4) var<storage, read_write> DZ: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.rows * p.V) { return; }
  let r = gid.x / p.V;
  let j = gid.x % p.V;
  var g = exp(Z[gid.x] - LSE[r]) * p.invN;
  if (j == TGT[r]) { g = g - p.invN; }
  DZ[gid.x] = g;
}`;
// ---- optimizer + clipping -------------------------------------------------------
export const ADAMW = /* wgsl */ `
struct P { n: u32, lr: f32, beta1: f32, beta2: f32, eps: f32, wd: f32, bc1: f32, bc2: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read_write> M: array<f32>;
@group(0) @binding(3) var<storage, read_write> V: array<f32>;
@group(0) @binding(4) var<storage, read_write> W: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.n) { return; }
  let g = G[gid.x];
  let m = p.beta1 * M[gid.x] + (1.0 - p.beta1) * g;
  let v = p.beta2 * V[gid.x] + (1.0 - p.beta2) * g * g;
  M[gid.x] = m;
  V[gid.x] = v;
  let mhat = m / p.bc1;
  let vhat = v / p.bc2;
  W[gid.x] = W[gid.x] - p.lr * (mhat / (sqrt(vhat) + p.eps) + p.wd * W[gid.x]);
}`;
// one workgroup per buffer: NORM[0] += Σ g² (dispatches are ordered within a pass)
export const SQSUM = /* wgsl */ `
struct P { n: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read_write> NORM: array<f32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  var s = 0.0;
  for (var i = lid.x; i < p.n; i = i + 256u) { let g = G[i]; s = s + g * g; }
  red[lid.x] = s;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st = st >> 1u) {
    if (lid.x < st) { red[lid.x] = red[lid.x] + red[lid.x + st]; }
    workgroupBarrier();
  }
  if (lid.x == 0u) { NORM[0] = NORM[0] + red[0]; }
}`;
export const CLIP_SCALE = /* wgsl */ `
struct P { n: u32, maxNorm: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> NORM: array<f32>;
@group(0) @binding(2) var<storage, read_write> G: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.n) { return; }
  let n2 = NORM[0];
  // non-finite or absurd norm → zero ALL grads (skip this update entirely);
  // inf * 0 would otherwise mint NaNs and cascade into the weights.
  // bitcast exponent test: fast-math-proof inf/NaN detection.
  let bits = bitcast<u32>(n2);
  if ((bits & 0x7f800000u) == 0x7f800000u || n2 > 1.0e30) {
    G[gid.x] = 0.0;
    return;
  }
  let len = sqrt(n2);
  if (len > p.maxNorm) { G[gid.x] = G[gid.x] * (p.maxNorm / len); }
}`;
