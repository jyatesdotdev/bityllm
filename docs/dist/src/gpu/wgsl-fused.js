// Fused per-layer inference kernels: one workgroup per kernel, whole layer
// halves in single dispatches. Collapses ~64 dispatches/token to ~16 — at
// GEMV sizes, dispatch overhead dominates compute, so fusion IS the speedup.
//
// Weight layout (one concatenated buffer per layer half, offsets derived from
// C in-shader):
//   attn: [lnW C][lnB C][Wq C²][bq C][Wk C²][bk C][Wv C²][bv C][Wo C²][bo C]
//   mlp:  [lnW C][lnB C][W1 C·F][b1 F][W2 F·C][b2 C]   (F = 4C)
// Limits: C ≤ 512, 4C ≤ 2048, blockSize ≤ 1024 (workgroup arrays).
export const ATTN_HALF = /* wgsl */ `
struct P { C: u32, nH: u32, hd: u32, pad: u32, scale: f32, p1: u32, p2: u32, p3: u32 }
struct G { id: u32, pos: u32, posC: u32, T: u32, outIdx: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> g: G;
@group(0) @binding(2) var<storage, read_write> X: array<f32>;
@group(0) @binding(3) var<storage, read> WL: array<f32>;
@group(0) @binding(4) var<storage, read_write> KC: array<f32>;
@group(0) @binding(5) var<storage, read_write> VC: array<f32>;

var<workgroup> XH: array<f32, 512>;
var<workgroup> Q: array<f32, 512>;
var<workgroup> AO: array<f32, 512>;
var<workgroup> probs: array<f32, 1024>;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let C = p.C;

  // ---- LayerNorm(X) -> XH (shared) ----
  var s = 0.0;
  for (var j = lid.x; j < C; j = j + 256u) { s = s + X[j]; }
  red[lid.x] = s;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st = st >> 1u) {
    if (lid.x < st) { red[lid.x] = red[lid.x] + red[lid.x + st]; }
    workgroupBarrier();
  }
  let mu = red[0] / f32(C);
  workgroupBarrier();
  var vv = 0.0;
  for (var j = lid.x; j < C; j = j + 256u) {
    let d = X[j] - mu;
    vv = vv + d * d;
  }
  red[lid.x] = vv;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st = st >> 1u) {
    if (lid.x < st) { red[lid.x] = red[lid.x] + red[lid.x + st]; }
    workgroupBarrier();
  }
  let rstd = inverseSqrt(red[0] / f32(C) + 1e-5);
  workgroupBarrier();
  for (var j = lid.x; j < C; j = j + 256u) {
    XH[j] = (X[j] - mu) * rstd * WL[j] + WL[C + j];
  }
  workgroupBarrier();

  // ---- q,k,v (k/v straight into the caches at posC) ----
  let WQ = 2u * C;
  let BQ = WQ + C * C;
  let WK = BQ + C;
  let BK = WK + C * C;
  let WV = BK + C;
  let BV = WV + C * C;
  let WO = BV + C;
  let BO = WO + C * C;
  for (var j = lid.x; j < C; j = j + 256u) {
    var q = 0.0;
    var k = 0.0;
    var v = 0.0;
    for (var i = 0u; i < C; i = i + 1u) {
      let xh = XH[i];
      q = q + xh * WL[WQ + i * C + j];
      k = k + xh * WL[WK + i * C + j];
      v = v + xh * WL[WV + i * C + j];
    }
    Q[j] = q + WL[BQ + j];
    KC[g.posC + j] = k + WL[BK + j];
    VC[g.posC + j] = v + WL[BV + j];
  }
  storageBarrier();
  workgroupBarrier();

  // ---- attention (serial over heads; probs shared, reused) ----
  let T = g.T;
  for (var h = 0u; h < p.nH; h = h + 1u) {
    let ho = h * p.hd;
    var m = -3.0e38;
    for (var t = lid.x; t < T; t = t + 256u) {
      var sc = 0.0;
      let ko = t * C + ho;
      for (var i = 0u; i < p.hd; i = i + 1u) { sc = sc + Q[ho + i] * KC[ko + i]; }
      sc = sc * p.scale;
      probs[t] = sc;
      m = max(m, sc);
    }
    red[lid.x] = m;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st = st >> 1u) {
      if (lid.x < st) { red[lid.x] = max(red[lid.x], red[lid.x + st]); }
      workgroupBarrier();
    }
    let mx = red[0];
    workgroupBarrier();
    var sum = 0.0;
    for (var t = lid.x; t < T; t = t + 256u) {
      let e = exp(probs[t] - mx);
      probs[t] = e;
      sum = sum + e;
    }
    red[lid.x] = sum;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st = st >> 1u) {
      if (lid.x < st) { red[lid.x] = red[lid.x] + red[lid.x + st]; }
      workgroupBarrier();
    }
    let inv = 1.0 / red[0];
    workgroupBarrier();
    for (var j = lid.x; j < p.hd; j = j + 256u) {
      var y = 0.0;
      for (var t = 0u; t < T; t = t + 1u) { y = y + probs[t] * VC[t * C + ho + j]; }
      AO[ho + j] = y * inv;
    }
    workgroupBarrier();
  }

  // ---- output projection + residual ----
  for (var j = lid.x; j < C; j = j + 256u) {
    var o = 0.0;
    for (var i = 0u; i < C; i = i + 1u) { o = o + AO[i] * WL[WO + i * C + j]; }
    X[j] = X[j] + o + WL[BO + j];
  }
}`;
export const MLP_HALF = /* wgsl */ `
struct P { C: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> X: array<f32>;
@group(0) @binding(2) var<storage, read> WL: array<f32>;

var<workgroup> XH: array<f32, 512>;
var<workgroup> H: array<f32, 2048>;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let C = p.C;
  let F = 4u * C;

  // ---- LayerNorm(X) -> XH ----
  var s = 0.0;
  for (var j = lid.x; j < C; j = j + 256u) { s = s + X[j]; }
  red[lid.x] = s;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st = st >> 1u) {
    if (lid.x < st) { red[lid.x] = red[lid.x] + red[lid.x + st]; }
    workgroupBarrier();
  }
  let mu = red[0] / f32(C);
  workgroupBarrier();
  var vv = 0.0;
  for (var j = lid.x; j < C; j = j + 256u) {
    let d = X[j] - mu;
    vv = vv + d * d;
  }
  red[lid.x] = vv;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st = st >> 1u) {
    if (lid.x < st) { red[lid.x] = red[lid.x] + red[lid.x + st]; }
    workgroupBarrier();
  }
  let rstd = inverseSqrt(red[0] / f32(C) + 1e-5);
  workgroupBarrier();
  for (var j = lid.x; j < C; j = j + 256u) {
    XH[j] = (X[j] - mu) * rstd * WL[j] + WL[C + j];
  }
  workgroupBarrier();

  // ---- fc + GELU -> H ----
  let W1 = 2u * C;
  let B1 = W1 + C * F;
  let W2 = B1 + F;
  let B2 = W2 + F * C;
  for (var j = lid.x; j < F; j = j + 256u) {
    var a = 0.0;
    for (var i = 0u; i < C; i = i + 1u) { a = a + XH[i] * WL[W1 + i * F + j]; }
    a = a + WL[B1 + j];
    // guarded tanh GELU (Metal fast-math overflow — see gpu-fast-math-tanh-bug)
    let u = clamp(0.7978845608028654 * (a + 0.044715 * a * a * a), -15.0, 15.0);
    H[j] = 0.5 * a * (1.0 + tanh(u));
  }
  workgroupBarrier();

  // ---- proj + residual ----
  for (var j = lid.x; j < C; j = j + 256u) {
    var o = 0.0;
    for (var i = 0u; i < F; i = i + 1u) { o = o + H[i] * WL[W2 + i * C + j]; }
    X[j] = X[j] + o + WL[B2 + j];
  }
}`;
