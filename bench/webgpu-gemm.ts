// WebGPU GEMM proof-of-concept (M6 scouting): can the M-series GPU beat our
// blocked pure-TS CPU kernel at training-relevant matrix sizes — and by how
// much? Written against the standard WebGPU API: runs under Deno (wgpu→Metal)
// and, unchanged, in any WebGPU browser.
//
//   deno run --allow-read bench/webgpu-gemm.ts
//
// Measures: correctness vs CPU, GFLOP/s per size, and per-dispatch overhead
// (the number that decides whether small-model training can win on GPU).

import { CPUBackend } from "../src/backend/cpu.ts";
import { RNG } from "../src/core/rng.ts";

const WGSL = /* wgsl */ `
struct Dims { M: u32, N: u32, K: u32, pad: u32 };
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;

const TILE = 16u;
var<workgroup> As: array<f32, 256>;
var<workgroup> Bs: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = gid.y;
  let col = gid.x;
  var acc = 0.0;
  let tiles = (dims.K + TILE - 1u) / TILE;
  for (var t = 0u; t < tiles; t = t + 1u) {
    let ak = t * TILE + lid.x;
    let bk = t * TILE + lid.y;
    As[lid.y * TILE + lid.x] = select(0.0, A[row * dims.K + ak], row < dims.M && ak < dims.K);
    Bs[lid.y * TILE + lid.x] = select(0.0, B[bk * dims.N + col], bk < dims.K && col < dims.N);
    workgroupBarrier();
    for (var k = 0u; k < TILE; k = k + 1u) {
      acc = acc + As[lid.y * TILE + k] * Bs[k * TILE + lid.x];
    }
    workgroupBarrier();
  }
  if (row < dims.M && col < dims.N) {
    C[row * dims.N + col] = acc;
  }
}`;

const cpu = new CPUBackend(Float32Array);
const rng = new RNG(99);

function randArr(n: number): Float32Array {
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = rng.randn() * 0.5;
  return a;
}

async function main(): Promise<void> {
  if (typeof navigator === "undefined" || !navigator.gpu) {
    console.error("no WebGPU in this runtime — run with Deno (or a browser)");
    return;
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("no GPU adapter");
  const device = await adapter.requestDevice();
  console.log(`adapter: ${(adapter.info && `${adapter.info.vendor} ${adapter.info.architecture}`) || "unknown"}\n`);

  const module = device.createShaderModule({ code: WGSL });
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });

  const gemm = async (M: number, K: number, N: number, iters: number, check = false): Promise<number> => {
    const a = randArr(M * K), b = randArr(K * N);
    const mk = (data: Float32Array, usage: number): GPUBuffer => {
      const buf = device.createBuffer({ size: data.byteLength, usage: usage | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(buf, 0, data.buffer as ArrayBuffer);
      return buf;
    };
    const aBuf = mk(a, GPUBufferUsage.STORAGE);
    const bBuf = mk(b, GPUBufferUsage.STORAGE);
    const cBuf = device.createBuffer({ size: M * N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const dims = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(dims, 0, new Uint32Array([M, N, K, 0]).buffer as ArrayBuffer);

    const bind = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: dims } },
        { binding: 1, resource: { buffer: aBuf } },
        { binding: 2, resource: { buffer: bBuf } },
        { binding: 3, resource: { buffer: cBuf } },
      ],
    });

    const dispatch = (n: number): void => {
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bind);
      for (let i = 0; i < n; i++) pass.dispatchWorkgroups(Math.ceil(N / 16), Math.ceil(M / 16));
      pass.end();
      device.queue.submit([enc.finish()]);
    };

    dispatch(3); // warmup
    await device.queue.onSubmittedWorkDone();

    const t0 = performance.now();
    dispatch(iters);
    await device.queue.onSubmittedWorkDone();
    const dt = (performance.now() - t0) / 1000;

    if (check) {
      const read = device.createBuffer({ size: M * N * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(cBuf, 0, read, 0, M * N * 4);
      device.queue.submit([enc.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      const got = new Float32Array(read.getMappedRange().slice(0));
      read.unmap();
      const want = cpu.matmul({ data: a, shape: [M, K] }, { data: b, shape: [K, N] }).data;
      let maxDiff = 0;
      for (let i = 0; i < want.length; i++) maxDiff = Math.max(maxDiff, Math.abs(want[i] - got[i]));
      if (maxDiff > 1e-2) throw new Error(`GPU wrong: maxDiff ${maxDiff}`);
      console.log(`  correctness vs CPU: maxDiff ${maxDiff.toExponential(1)} ✓`);
    }
    for (const buf of [aBuf, bBuf, cBuf, dims]) buf.destroy();
    return (2 * M * K * N * iters) / dt / 1e9;
  };

  const cpuGemm = (M: number, K: number, N: number, iters: number): number => {
    const a = { data: randArr(M * K), shape: [M, K] };
    const b = { data: randArr(K * N), shape: [K, N] };
    cpu.matmul(a, b); // warmup
    const t0 = performance.now();
    for (let i = 0; i < iters; i++) cpu.matmul(a, b);
    const dt = (performance.now() - t0) / 1000;
    return (2 * M * K * N * iters) / dt / 1e9;
  };

  console.log("size (MxKxN)              CPU 1-thread     GPU (WebGPU)     speedup");
  const cases: Array<[number, number, number, number, number]> = [
    // [M, K, N, gpuIters, cpuIters] — training-shaped first (milli/micro), then peak
    [2048, 128, 512, 200, 20],
    [2048, 512, 128, 200, 20],
    [2048, 192, 768, 200, 10],
    [1024, 1024, 1024, 100, 3],
    [2048, 2048, 2048, 50, 1],
  ];
  let first = true;
  for (const [M, K, N, gi, ci] of cases) {
    const g = await gemm(M, K, N, gi, first);
    first = false;
    const c = cpuGemm(M, K, N, ci);
    console.log(
      `${`${M}x${K}x${N}`.padEnd(25)} ${c.toFixed(1).padStart(7)} GF/s   ${g.toFixed(1).padStart(9)} GF/s   ${(g / c).toFixed(1).padStart(6)}x`,
    );
  }

  // per-dispatch overhead: tiny matmul, many dispatches
  const t = await gemm(64, 64, 64, 2000);
  const flops = 2 * 64 * 64 * 64;
  console.log(`\nper-dispatch floor (64³): ${((flops / (t * 1e9)) * 1e6).toFixed(1)} µs/dispatch — ` +
    `a milli/micro train step needs ~250 dispatches`);
}

await main();
