// CPUBackend — pure-TypeScript kernels on flat FloatArrays (DESIGN §5, §18).
// Loop orders are chosen so inner loops stream contiguous memory (i-k-j for
// plain matmul, row-dot for transposed-B). No allocation inside hot loops.

import type { Backend, FloatArray, FloatArrayCtor, MatmulOpts, NdArray } from "./backend.ts";

const prod = (s: number[]): number => s.reduce((a, b) => a * b, 1);
const NEG = -1e30; // effectively -inf for softmax; avoids inf-inf NaN traps

export class CPUBackend implements Backend {
  readonly name: string;
  readonly Arr: FloatArrayCtor;

  constructor(Arr: FloatArrayCtor = Float32Array) {
    this.Arr = Arr;
    this.name = `cpu-${Arr === Float32Array ? "f32" : "f64"}`;
  }

  alloc(n: number): FloatArray {
    return new this.Arr(n);
  }
  zeros(shape: number[]): NdArray {
    return { data: this.alloc(prod(shape)), shape: [...shape] };
  }
  from(values: ArrayLike<number>, shape: number[]): NdArray {
    if (values.length !== prod(shape)) throw new Error(`from: ${values.length} values for shape [${shape}]`);
    const data = this.alloc(values.length);
    data.set(values as ArrayLike<number>);
    return { data, shape: [...shape] };
  }
  copy(a: NdArray): NdArray {
    const data = this.alloc(a.data.length);
    data.set(a.data);
    return { data, shape: [...a.shape] };
  }

  matmul(a: NdArray, b: NdArray, opt: MatmulOpts = {}): NdArray {
    const tA = !!opt.transposeA, tB = !!opt.transposeB;
    const as = a.shape, bs = b.shape;
    if (as.length < 2 || bs.length < 2) throw new Error("matmul: operands must be ≥2-D");
    const p = as[as.length - 2], q = as[as.length - 1];
    const r = bs[bs.length - 2], s = bs[bs.length - 1];
    const M = tA ? q : p, K = tA ? p : q;
    const Kb = tB ? s : r, N = tB ? r : s;
    if (K !== Kb) throw new Error(`matmul: inner dims ${K}≠${Kb} (a=[${as}] b=[${bs}] tA=${tA} tB=${tB})`);

    const aBatch = prod(as.slice(0, -2));
    const bBatch = prod(bs.slice(0, -2));
    const broadcastB = bs.length === 2 && as.length > 2;
    if (!broadcastB && (aBatch !== bBatch || as.length !== bs.length))
      throw new Error(`matmul: batch mismatch a=[${as}] b=[${bs}]`);

    const ad = a.data, bd = b.data;
    const out = this.alloc(aBatch * M * N);
    const aStride = p * q, bStride = broadcastB ? 0 : r * s, cStride = M * N;

    for (let ib = 0; ib < aBatch; ib++) {
      const ao = ib * aStride, bo = ib * bStride, co = ib * cStride;
      if (!tA && !tB) {
        // C[m,:] += A[m,k..k+3] * B[k..k+3,:] — 4-wide k-blocking: one pass over
        // the C row does the work of four, cutting C-row memory traffic 4×.
        for (let m = 0; m < M; m++) {
          const aRow = ao + m * K, cRow = co + m * N;
          let k = 0;
          for (; k + 3 < K; k += 4) {
            const av0 = ad[aRow + k], av1 = ad[aRow + k + 1], av2 = ad[aRow + k + 2], av3 = ad[aRow + k + 3];
            const b0 = bo + k * N, b1 = b0 + N, b2 = b1 + N, b3 = b2 + N;
            for (let n = 0; n < N; n++)
              out[cRow + n] += av0 * bd[b0 + n] + av1 * bd[b1 + n] + av2 * bd[b2 + n] + av3 * bd[b3 + n];
          }
          for (; k < K; k++) {
            const av = ad[aRow + k];
            if (av === 0) continue;
            const bRow = bo + k * N;
            for (let n = 0; n < N; n++) out[cRow + n] += av * bd[bRow + n];
          }
        }
      } else if (!tA && tB) {
        // C[m,n] = dot(A row m, B row n) — two output columns per pass reuse the
        // A row from registers/L1; independent accumulators pipeline the FMAs.
        for (let m = 0; m < M; m++) {
          const aRow = ao + m * K, cRow = co + m * N;
          let n = 0;
          for (; n + 1 < N; n += 2) {
            const b0 = bo + n * K, b1 = b0 + K;
            let s0 = 0, s1 = 0;
            for (let k = 0; k < K; k++) {
              const av = ad[aRow + k];
              s0 += av * bd[b0 + k];
              s1 += av * bd[b1 + k];
            }
            out[cRow + n] = s0;
            out[cRow + n + 1] = s1;
          }
          for (; n < N; n++) {
            const bRow = bo + n * K;
            let acc = 0;
            for (let k = 0; k < K; k++) acc += ad[aRow + k] * bd[bRow + k];
            out[cRow + n] = acc;
          }
        }
      } else if (tA && !tB) {
        // A(m,k) = a[k*M + m] — 4-wide k-blocking with k-outer layout
        let k = 0;
        for (; k + 3 < K; k += 4) {
          const a0 = ao + k * M, a1 = a0 + M, a2 = a1 + M, a3 = a2 + M;
          const b0 = bo + k * N, b1 = b0 + N, b2 = b1 + N, b3 = b2 + N;
          for (let m = 0; m < M; m++) {
            const av0 = ad[a0 + m], av1 = ad[a1 + m], av2 = ad[a2 + m], av3 = ad[a3 + m];
            const cRow = co + m * N;
            for (let n = 0; n < N; n++)
              out[cRow + n] += av0 * bd[b0 + n] + av1 * bd[b1 + n] + av2 * bd[b2 + n] + av3 * bd[b3 + n];
          }
        }
        for (; k < K; k++) {
          const aRow = ao + k * M, bRow = bo + k * N;
          for (let m = 0; m < M; m++) {
            const av = ad[aRow + m];
            if (av === 0) continue;
            const cRow = co + m * N;
            for (let n = 0; n < N; n++) out[cRow + n] += av * bd[bRow + n];
          }
        }
      } else {
        // both transposed (rare): C[m,n] = Σ_k a[k*M+m] * b[n*K+k]
        for (let m = 0; m < M; m++) {
          const cRow = co + m * N;
          for (let n = 0; n < N; n++) {
            const bRow = bo + n * K;
            let acc = 0;
            for (let k = 0; k < K; k++) acc += ad[ao + k * M + m] * bd[bRow + k];
            out[cRow + n] = acc;
          }
        }
      }
    }
    return { data: out, shape: [...as.slice(0, -2), M, N] };
  }

  add(a: NdArray, b: NdArray): NdArray {
    return this.zipSuffix(a, b, (x, y) => x + y);
  }
  mul(a: NdArray, b: NdArray): NdArray {
    return this.zipSuffix(a, b, (x, y) => x * y);
  }
  private zipSuffix(a: NdArray, b: NdArray, f: (x: number, y: number) => number): NdArray {
    const an = a.data.length, bn = b.data.length;
    if (an % bn !== 0) throw new Error(`broadcast: |b|=${bn} does not divide |a|=${an}`);
    const ad = a.data, bd = b.data, out = this.alloc(an);
    if (an === bn) {
      for (let i = 0; i < an; i++) out[i] = f(ad[i], bd[i]);
    } else {
      for (let o = 0; o < an; o += bn)
        for (let j = 0; j < bn; j++) out[o + j] = f(ad[o + j], bd[j]);
    }
    return { data: out, shape: [...a.shape] };
  }

  scale(a: NdArray, s: number): NdArray {
    const ad = a.data, out = this.alloc(ad.length);
    for (let i = 0; i < ad.length; i++) out[i] = ad[i] * s;
    return { data: out, shape: [...a.shape] };
  }

  sumEvery(a: FloatArray, size: number): FloatArray {
    const out = this.alloc(size);
    for (let o = 0; o < a.length; o += size)
      for (let j = 0; j < size; j++) out[j] += a[o + j];
    return out;
  }

  sumBatch(a: NdArray, keepShape: number[]): NdArray {
    return { data: this.sumEvery(a.data, prod(keepShape)), shape: [...keepShape] };
  }

  // tanh-approximation GELU (GPT-2 flavor)
  gelu(a: NdArray): NdArray {
    const ad = a.data, out = this.alloc(ad.length);
    const C = Math.sqrt(2 / Math.PI);
    for (let i = 0; i < ad.length; i++) {
      const x = ad[i];
      out[i] = 0.5 * x * (1 + Math.tanh(C * (x + 0.044715 * x * x * x)));
    }
    return { data: out, shape: [...a.shape] };
  }
  geluBackward(x: NdArray, g: NdArray): NdArray {
    const xd = x.data, gd = g.data, out = this.alloc(xd.length);
    const C = Math.sqrt(2 / Math.PI);
    for (let i = 0; i < xd.length; i++) {
      const v = xd[i];
      const u = C * (v + 0.044715 * v * v * v);
      const t = Math.tanh(u);
      const sech2 = 1 - t * t;
      const du = C * (1 + 3 * 0.044715 * v * v);
      out[i] = gd[i] * (0.5 * (1 + t) + 0.5 * v * sech2 * du);
    }
    return { data: out, shape: [...x.shape] };
  }

  softmaxLastDim(a: NdArray): NdArray {
    const D = a.shape[a.shape.length - 1];
    const ad = a.data, out = this.alloc(ad.length);
    for (let o = 0; o < ad.length; o += D) {
      let max = -Infinity;
      for (let j = 0; j < D; j++) if (ad[o + j] > max) max = ad[o + j];
      let sum = 0;
      for (let j = 0; j < D; j++) {
        const e = Math.exp(ad[o + j] - max);
        out[o + j] = e;
        sum += e;
      }
      const inv = 1 / sum;
      for (let j = 0; j < D; j++) out[o + j] *= inv;
    }
    return { data: out, shape: [...a.shape] };
  }
  softmaxBackward(s: NdArray, g: NdArray): NdArray {
    const D = s.shape[s.shape.length - 1];
    const sd = s.data, gd = g.data, out = this.alloc(sd.length);
    for (let o = 0; o < sd.length; o += D) {
      let dot = 0;
      for (let j = 0; j < D; j++) dot += gd[o + j] * sd[o + j];
      for (let j = 0; j < D; j++) out[o + j] = sd[o + j] * (gd[o + j] - dot);
    }
    return { data: out, shape: [...s.shape] };
  }

  layerNorm(x: NdArray, w: NdArray, b: NdArray, eps: number): { y: NdArray; mean: FloatArray; rstd: FloatArray } {
    const C = x.shape[x.shape.length - 1];
    const R = x.data.length / C;
    const xd = x.data, wd = w.data, bd = b.data;
    const y = this.alloc(xd.length), mean = this.alloc(R), rstd = this.alloc(R);
    for (let r = 0; r < R; r++) {
      const o = r * C;
      let mu = 0;
      for (let j = 0; j < C; j++) mu += xd[o + j];
      mu /= C;
      let vsum = 0;
      for (let j = 0; j < C; j++) {
        const d = xd[o + j] - mu;
        vsum += d * d;
      }
      const rs = 1 / Math.sqrt(vsum / C + eps);
      mean[r] = mu;
      rstd[r] = rs;
      for (let j = 0; j < C; j++) y[o + j] = (xd[o + j] - mu) * rs * wd[j] + bd[j];
    }
    return { y: { data: y, shape: [...x.shape] }, mean, rstd };
  }
  layerNormBackward(x: NdArray, w: NdArray, mean: FloatArray, rstd: FloatArray, g: NdArray):
    { dx: NdArray; dw: FloatArray; db: FloatArray } {
    const C = x.shape[x.shape.length - 1];
    const R = x.data.length / C;
    const xd = x.data, wd = w.data, gd = g.data;
    const dx = this.alloc(xd.length), dw = this.alloc(C), db = this.alloc(C);
    for (let r = 0; r < R; r++) {
      const o = r * C, mu = mean[r], rs = rstd[r];
      // m1 = mean_j(g⊙w), m2 = mean_j(g⊙w⊙xhat)
      let m1 = 0, m2 = 0;
      for (let j = 0; j < C; j++) {
        const xhat = (xd[o + j] - mu) * rs;
        const gw = gd[o + j] * wd[j];
        m1 += gw;
        m2 += gw * xhat;
        dw[j] += gd[o + j] * xhat;
        db[j] += gd[o + j];
      }
      m1 /= C;
      m2 /= C;
      for (let j = 0; j < C; j++) {
        const xhat = (xd[o + j] - mu) * rs;
        dx[o + j] = rs * (gd[o + j] * wd[j] - m1 - xhat * m2);
      }
    }
    return { dx: { data: dx, shape: [...x.shape] }, dw, db };
  }

  gatherRows(table: NdArray, ids: Int32Array): NdArray {
    const C = table.shape[1];
    const td = table.data, out = this.alloc(ids.length * C);
    for (let i = 0; i < ids.length; i++) {
      const src = ids[i] * C, dst = i * C;
      for (let j = 0; j < C; j++) out[dst + j] = td[src + j];
    }
    return { data: out, shape: [ids.length, C] };
  }
  scatterAddRows(gradTable: FloatArray, cols: number, ids: Int32Array, g: FloatArray): void {
    for (let i = 0; i < ids.length; i++) {
      const dst = ids[i] * cols, src = i * cols;
      for (let j = 0; j < cols; j++) gradTable[dst + j] += g[src + j];
    }
  }

  transpose12(a: NdArray): NdArray {
    const [d0, d1, d2, d3] = a.shape;
    if (a.shape.length !== 4) throw new Error(`transpose12: expected 4-D, got [${a.shape}]`);
    const ad = a.data, out = this.alloc(ad.length);
    for (let i0 = 0; i0 < d0; i0++)
      for (let i1 = 0; i1 < d1; i1++)
        for (let i2 = 0; i2 < d2; i2++) {
          const src = ((i0 * d1 + i1) * d2 + i2) * d3;
          const dst = ((i0 * d2 + i2) * d1 + i1) * d3;
          for (let j = 0; j < d3; j++) out[dst + j] = ad[src + j];
        }
    return { data: out, shape: [d0, d2, d1, d3] };
  }

  causalMask(a: NdArray, T: number): NdArray {
    const out = this.copy(a);
    const od = out.data;
    for (let o = 0; o < od.length; o += T * T)
      for (let i = 0; i < T; i++)
        for (let j = i + 1; j < T; j++) od[o + i * T + j] = NEG;
    return out;
  }
  causalMaskZeroGrad(g: NdArray, T: number): void {
    const gd = g.data;
    for (let o = 0; o < gd.length; o += T * T)
      for (let i = 0; i < T; i++)
        for (let j = i + 1; j < T; j++) gd[o + i * T + j] = 0;
  }

  // Cross-entropy for next-token prediction, the numerically stable way. For one
  // row of logits z with target class t:  loss = −log softmax(z)[t] = logsumexp(z) − z[t].
  // logsumexp(z) = max + log Σ exp(z − max); subtracting the row max before exp keeps
  // the exponentials in [0,1] so nothing overflows. We stash lse per row so the
  // backward pass can reuse it (the two are fused — see ops.crossEntropyLogits).
  ceForward(logits: NdArray, targets: Int32Array): { loss: number; lse: FloatArray } {
    const [N, V] = logits.shape;
    const zd = logits.data, lse = this.alloc(N);
    let loss = 0;
    for (let i = 0; i < N; i++) {
      const o = i * V;
      let max = -Infinity;
      for (let j = 0; j < V; j++) if (zd[o + j] > max) max = zd[o + j];
      let sum = 0;
      for (let j = 0; j < V; j++) sum += Math.exp(zd[o + j] - max);
      const l = max + Math.log(sum);
      lse[i] = l;
      loss += l - zd[o + targets[i]];
    }
    return { loss: loss / N, lse };
  }
  // The famously clean CE gradient:  dL/dz = (softmax(z) − onehot(target)) / N.
  // exp(z − lse) IS softmax(z) (lse already folds in the row max), and s = gscale/N
  // carries the mean-over-N factor. So every logit gets +softmax·s, and the target
  // logit additionally gets −s: "push the right token up, all others down."
  ceBackward(logits: NdArray, lse: FloatArray, targets: Int32Array, gscale: number): NdArray {
    const [N, V] = logits.shape;
    const zd = logits.data, out = this.alloc(zd.length);
    const s = gscale / N;
    for (let i = 0; i < N; i++) {
      const o = i * V, l = lse[i];
      for (let j = 0; j < V; j++) out[o + j] = Math.exp(zd[o + j] - l) * s; // softmax(z) · s
      out[o + targets[i]] -= s;                                             // − onehot(target) · s
    }
    return { data: out, shape: [N, V] };
  }
}
