// WGSL kernels for single-token inference (browser + Deno).
//
// Inference is GEMV-shaped (M=1): dedicated column-parallel kernels instead of
// the training GEMM's 16×16 tiles (which would idle 15/16 threads). A single
// 16-byte "globals" uniform {id, pos, posC, T} is rewritten once per token; all
// other bindings and uniforms are prebuilt.
/** per-token dynamic state, shared by all kernels (GPU-mutated during chunked
 *  generation by the SAMPLE kernel; CPU-written during prefill) */
export const GLOBALS_SIZE = 32; // u32 id, pos, posC (=pos*C), T (=pos+1), outIdx (+pad)
// GPU-side sampling: temperature + top-k + hash-PRNG, so a whole chunk of
// tokens generates in one submit with a single 4·N-byte readback at the end.
// Runs BEFORE each token's forward: advances position state, picks the token
// id from the previous forward's logits, and records it.
export const SAMPLE = /* wgsl */ `
struct SP { V: u32, C: u32, topK: u32, temp: f32, seed: u32, pad1: u32, pad2: u32, pad3: u32 }
struct G { id: u32, pos: u32, posC: u32, T: u32, outIdx: u32 }
@group(0) @binding(0) var<uniform> p: SP;
@group(0) @binding(1) var<storage, read_write> g: G;
@group(0) @binding(2) var<storage, read> LOGITS: array<f32>;
@group(0) @binding(3) var<storage, read_write> OUT: array<u32>;

fn pcg(v: u32) -> u32 {
  var s = v * 747796405u + 2891336453u;
  s = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (s >> 22u) ^ s;
}

var<workgroup> sv: array<f32, 512>;   // scaled logits (mutated during extraction)
var<workgroup> red: array<f32, 128>;
var<workgroup> redI: array<u32, 128>;
var<workgroup> chosen: array<u32, 64>;
var<workgroup> vals: array<f32, 64>;

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  if (lid.x == 0u) {
    // advance position state (CPU pre-sets pos one behind for the first iteration)
    g.pos = g.pos + 1u;
    g.posC = g.posC + p.C;
    g.T = g.T + 1u;
  }
  workgroupBarrier();

  let invT = 1.0 / max(p.temp, 1e-6);
  for (var j = lid.x; j < p.V; j = j + 128u) { sv[j] = LOGITS[j] * invT; }
  workgroupBarrier();

  let k = select(min(p.topK, p.V), p.V, p.topK == 0u);
  let kk = min(k, 64u);
  // parallel top-k: kk rounds of a 128-thread argmax reduction
  for (var i = 0u; i < kk; i = i + 1u) {
    var bv = -3.0e38;
    var bi = 0u;
    for (var j = lid.x; j < p.V; j = j + 128u) {
      if (sv[j] > bv) { bv = sv[j]; bi = j; }
    }
    red[lid.x] = bv;
    redI[lid.x] = bi;
    workgroupBarrier();
    for (var st = 64u; st > 0u; st = st >> 1u) {
      if (lid.x < st && red[lid.x + st] > red[lid.x]) {
        red[lid.x] = red[lid.x + st];
        redI[lid.x] = redI[lid.x + st];
      }
      workgroupBarrier();
    }
    if (lid.x == 0u) {
      chosen[i] = redI[0];
      vals[i] = red[0];
      sv[redI[0]] = -3.0e38; // remove from contention
    }
    workgroupBarrier();
  }

  // softmax over the k survivors + categorical draw (thread 0; k ≤ 64 — cheap)
  if (lid.x == 0u) {
    let mx = vals[0];
    var sum = 0.0;
    for (var i = 0u; i < kk; i = i + 1u) {
      vals[i] = exp(vals[i] - mx);
      sum = sum + vals[i];
    }
    let r = f32(pcg(p.seed ^ (g.T * 2654435761u))) / 4294967296.0 * sum;
    var acc = 0.0;
    var id = chosen[kk - 1u];
    for (var i = 0u; i < kk; i = i + 1u) {
      acc = acc + vals[i];
      if (acc >= r) { id = chosen[i]; break; }
    }
    g.id = id;
    OUT[g.outIdx] = id;
    g.outIdx = g.outIdx + 1u;
  }
}`;
// y[N] = x[K] · W[K,N] (+bias). writeAtPos=1 → write into Y at globals.posC
// (used to emit k/v rows straight into the KV-cache, no copy passes).
export const GEMV = /* wgsl */ `
struct P { K: u32, N: u32, useBias: u32, writeAtPos: u32 }
struct G { id: u32, pos: u32, posC: u32, T: u32, outIdx: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> g: G;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read> W: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read> BIAS: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j = gid.x;
  if (j >= p.N) { return; }
  var s = 0.0;
  for (var k = 0u; k < p.K; k = k + 1u) { s = s + X[k] * W[k * p.N + j]; }
  if (p.useBias == 1u) { s = s + BIAS[j]; }
  if (p.writeAtPos == 1u) { Y[g.posC + j] = s; } else { Y[j] = s; }
}`;
// y[N] = x[K] · Wᵀ where W is stored [N, K] (tied LM head: logits = x·wteᵀ).
// NB: no globals binding — layout:"auto" prunes statically-unused bindings.
export const GEMV_T = /* wgsl */ `
struct P { K: u32, N: u32, useBias: u32, writeAtPos: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j = gid.x;
  if (j >= p.N) { return; }
  var s = 0.0;
  let base = j * p.K;
  for (var k = 0u; k < p.K; k = k + 1u) { s = s + X[k] * W[base + k]; }
  Y[j] = s;
}`;
// C += A (residual accumulate — avoids binding one buffer read+read_write)
export const ACCUM = /* wgsl */ `
struct P { n: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x < p.n) { C[gid.x] = C[gid.x] + A[gid.x]; }
}`;
// x[C] = wte[id] + wpe[pos]
export const EMBED_ONE = /* wgsl */ `
struct P { C: u32 }
struct G { id: u32, pos: u32, posC: u32, T: u32, outIdx: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> g: G;
@group(0) @binding(2) var<storage, read> WTE: array<f32>;
@group(0) @binding(3) var<storage, read> WPE: array<f32>;
@group(0) @binding(4) var<storage, read_write> X: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j = gid.x;
  if (j >= p.C) { return; }
  X[j] = WTE[g.id * p.C + j] + WPE[g.pos * p.C + j];
}`;
// One attention row over the KV-cache: workgroup per head.
//   scores[t] = q_h · K[t]_h / √hd  (t < globals.T)
//   probs = softmax(scores);  out_h[j] = Σ_t probs[t] · V[t]_h[j]
export const ATTN_ONE = /* wgsl */ `
struct P { C: u32, nH: u32, hd: u32, scale: f32 }
struct G { id: u32, pos: u32, posC: u32, T: u32, outIdx: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> g: G;
@group(0) @binding(2) var<storage, read> Q: array<f32>;
@group(0) @binding(3) var<storage, read> KC: array<f32>;
@group(0) @binding(4) var<storage, read> VC: array<f32>;
@group(0) @binding(5) var<storage, read_write> OUT: array<f32>;

var<workgroup> probs: array<f32, 1024>;
var<workgroup> red: array<f32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wid.x;
  let ho = h * p.hd;
  let T = g.T;

  var m = -3.0e38;
  for (var t = lid.x; t < T; t = t + 128u) {
    var s = 0.0;
    let ko = t * p.C + ho;
    for (var j = 0u; j < p.hd; j = j + 1u) { s = s + Q[ho + j] * KC[ko + j]; }
    s = s * p.scale;
    probs[t] = s;
    m = max(m, s);
  }
  red[lid.x] = m;
  workgroupBarrier();
  for (var st = 64u; st > 0u; st = st >> 1u) {
    if (lid.x < st) { red[lid.x] = max(red[lid.x], red[lid.x + st]); }
    workgroupBarrier();
  }
  let mx = red[0];
  workgroupBarrier();
  var sum = 0.0;
  for (var t = lid.x; t < T; t = t + 128u) {
    let e = exp(probs[t] - mx);
    probs[t] = e;
    sum = sum + e;
  }
  red[lid.x] = sum;
  workgroupBarrier();
  for (var st = 64u; st > 0u; st = st >> 1u) {
    if (lid.x < st) { red[lid.x] = red[lid.x] + red[lid.x + st]; }
    workgroupBarrier();
  }
  let inv = 1.0 / red[0];
  workgroupBarrier();
  for (var j = lid.x; j < p.hd; j = j + 128u) {
    var y = 0.0;
    for (var t = 0u; t < T; t = t + 1u) { y = y + probs[t] * inv * VC[t * p.C + ho + j]; }
    OUT[ho + j] = y;
  }
}`;
