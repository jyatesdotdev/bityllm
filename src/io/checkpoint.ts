// Checkpoint format (DESIGN §15) — dependency-free, safetensors-shaped:
//   [ uint32 headerLen ][ headerLen bytes UTF-8 JSON ][ raw LE f32 blobs ]
// Pure bytes in/out — callers own file I/O (isomorphic).

import type { GPTConfig } from "../nn/gpt.ts";
import { GPT } from "../nn/gpt.ts";
import { CharTokenizer } from "../tokenizer/char.ts";
import { RNG } from "../core/rng.ts";

interface TensorMeta {
  shape: number[];
  dtype: "f32" | "i8";
  offset: number;        // bytes from start of blob section
  length: number;        // element count
  scalesOffset?: number; // i8 only: byte offset of per-row f32 scales
  rows?: number;         // i8 only: number of quantization rows
}

interface Header {
  format: "bity1";
  config: GPTConfig;
  tokenizer: { type: "char"; vocab: string[] };
  tensors: Record<string, TensorMeta>;
  step?: number;
}

export function serialize(model: GPT, tok: CharTokenizer, opts: { step?: number } = {}): Uint8Array {
  const named = model.namedParameters();
  const tensors: Record<string, TensorMeta> = {};
  let offset = 0;
  for (const [name, t] of named) {
    tensors[name] = { shape: [...t.shape], dtype: "f32", offset, length: t.size };
    offset += t.size * 4;
  }
  const header: Header = {
    format: "bity1",
    config: model.cfg,
    tokenizer: { type: "char", vocab: tok.vocab },
    tensors,
    step: opts.step,
  };
  const headerBytes = new TextEncoder().encode(JSON.stringify(header));
  const out = new Uint8Array(4 + headerBytes.length + offset);
  new DataView(out.buffer).setUint32(0, headerBytes.length, true);
  out.set(headerBytes, 4);

  const blobStart = 4 + headerBytes.length;
  for (const [name, t] of named) {
    const f32 = t.data instanceof Float32Array ? t.data : Float32Array.from(t.data);
    out.set(new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength), blobStart + tensors[name].offset);
  }
  return out;
}

/**
 * int8 deployment checkpoint (DESIGN §15): 2-D tensors quantized per-row
 * (scale = max|row|/127), 1-D tensors (biases, LayerNorm) kept f32.
 * ~4× smaller download; dequantized to f32 once at load.
 */
export function serializeInt8(model: GPT, tok: CharTokenizer, opts: { step?: number } = {}): Uint8Array {
  const named = model.namedParameters();
  const tensors: Record<string, TensorMeta> = {};
  let offset = 0;
  const plans: Array<{ name: string; quant: boolean }> = [];
  for (const [name, t] of named) {
    const quant = t.shape.length === 2;
    plans.push({ name, quant });
    if (quant) {
      const rows = t.shape[0];
      tensors[name] = { shape: [...t.shape], dtype: "i8", offset, length: t.size, scalesOffset: offset + t.size, rows };
      offset += t.size + rows * 4;
    } else {
      tensors[name] = { shape: [...t.shape], dtype: "f32", offset, length: t.size };
      offset += t.size * 4;
    }
  }
  const header: Header = {
    format: "bity1",
    config: model.cfg,
    tokenizer: { type: "char", vocab: tok.vocab },
    tensors,
    step: opts.step,
  };
  const headerBytes = new TextEncoder().encode(JSON.stringify(header));
  const out = new Uint8Array(4 + headerBytes.length + offset);
  new DataView(out.buffer).setUint32(0, headerBytes.length, true);
  out.set(headerBytes, 4);
  const blobStart = 4 + headerBytes.length;

  for (const [name, t] of named) {
    const meta = tensors[name];
    if (meta.dtype === "i8") {
      const rows = meta.rows!;
      const cols = t.size / rows;
      const q = new Int8Array(t.size);
      const scales = new Float32Array(rows);
      for (let r = 0; r < rows; r++) {
        let max = 0;
        for (let c = 0; c < cols; c++) max = Math.max(max, Math.abs(t.data[r * cols + c]));
        const s = max > 0 ? max / 127 : 1;
        scales[r] = s;
        for (let c = 0; c < cols; c++) q[r * cols + c] = Math.round(t.data[r * cols + c] / s);
      }
      out.set(new Uint8Array(q.buffer), blobStart + meta.offset);
      out.set(new Uint8Array(scales.buffer), blobStart + meta.scalesOffset!);
    } else {
      const f32 = t.data instanceof Float32Array ? t.data : Float32Array.from(t.data);
      out.set(new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength), blobStart + meta.offset);
    }
  }
  return out;
}

export function deserialize(bytes: Uint8Array): { model: GPT; tok: CharTokenizer; step: number } {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const headerLen = view.getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(4, 4 + headerLen))) as Header;
  if (header.format !== "bity1") throw new Error(`unknown checkpoint format: ${header.format}`);

  const tok = new CharTokenizer(header.tokenizer.vocab);
  const model = new GPT(header.config, new RNG(0)); // weights overwritten below
  const blobStart = 4 + headerLen;

  for (const [name, t] of model.namedParameters()) {
    const meta = header.tensors[name];
    if (!meta) throw new Error(`checkpoint missing tensor: ${name}`);
    if (meta.length !== t.size) throw new Error(`shape mismatch for ${name}`);
    if (meta.dtype === "i8") {
      const rows = meta.rows!;
      const cols = meta.length / rows;
      // reinterpret unsigned bytes as signed int8
      const raw = bytes.subarray(blobStart + meta.offset, blobStart + meta.offset + meta.length);
      const q = new Int8Array(meta.length);
      for (let i = 0; i < raw.length; i++) q[i] = raw[i] > 127 ? raw[i] - 256 : raw[i];
      const sbuf = new ArrayBuffer(rows * 4);
      new Uint8Array(sbuf).set(bytes.subarray(blobStart + meta.scalesOffset!, blobStart + meta.scalesOffset! + rows * 4));
      const scales = new Float32Array(sbuf);
      for (let r = 0; r < rows; r++)
        for (let c = 0; c < cols; c++) t.data[r * cols + c] = q[r * cols + c] * scales[r];
    } else {
      // copy to a fresh buffer — blob offsets are not guaranteed 4-byte aligned
      const src = bytes.subarray(blobStart + meta.offset, blobStart + meta.offset + meta.length * 4);
      const buf = new ArrayBuffer(meta.length * 4);
      new Uint8Array(buf).set(src);
      t.data.set(new Float32Array(buf));
    }
  }
  return { model, tok, step: header.step ?? 0 };
}
