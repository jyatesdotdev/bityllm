#!/usr/bin/env python3
"""
MLX training backend for bityllm (optional fast path — DESIGN "checkpoint is the
contract"). Trains the SAME GPT architecture as the pure-TS trainer, on Apple
Silicon via MLX/Metal (real fused kernels vs our naive WGSL), and exports the
exact `bity1` checkpoint the browser loads unchanged.

Fidelity requirements (or the exported weights won't evaluate correctly in the
TS inference engine):
  - tanh-approx GELU (our CPU + WebGPU inference hardcode it)
  - tied LM head (logits = x @ wte.T), learned absolute positions, pre-norm
  - LayerNorm eps 1e-5
  - MLX nn.Linear stores weight [out,in]; our format is [in,out] -> transpose on export

  python3 train/mlx_train.py --steps 16000 --batch 32 --block 128 --dim 384 \
    --layers 6 --heads 6 --out models/terminal-mlx.bity
"""
import argparse, json, math, struct, time
import numpy as np
from functools import partial
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

GELU_C = 0.7978845608028654  # sqrt(2/pi)


def gelu_tanh(x):
    return 0.5 * x * (1 + mx.tanh(GELU_C * (x + 0.044715 * x * x * x)))


class Block(nn.Module):
    def __init__(self, C, H):
        super().__init__()
        self.ln1 = nn.LayerNorm(C)
        self.wq = nn.Linear(C, C); self.wk = nn.Linear(C, C)
        self.wv = nn.Linear(C, C); self.wo = nn.Linear(C, C)
        self.ln2 = nn.LayerNorm(C)
        self.fc = nn.Linear(C, 4 * C); self.proj = nn.Linear(4 * C, C)
        self.H = H

    def __call__(self, x, mask):
        B, T, C = x.shape
        hd = C // self.H
        h = self.ln1(x)
        q = self.wq(h).reshape(B, T, self.H, hd).transpose(0, 2, 1, 3)
        k = self.wk(h).reshape(B, T, self.H, hd).transpose(0, 2, 1, 3)
        v = self.wv(h).reshape(B, T, self.H, hd).transpose(0, 2, 1, 3)
        att = (q @ k.transpose(0, 1, 3, 2)) * (hd ** -0.5) + mask
        att = mx.softmax(att, axis=-1)
        y = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        x = x + self.wo(y)
        x = x + self.proj(gelu_tanh(self.fc(self.ln2(x))))
        return x


class GPT(nn.Module):
    def __init__(self, V, block, L, H, C):
        super().__init__()
        self.wte = nn.Embedding(V, C)
        self.wpe = nn.Embedding(block, C)
        self.blocks = [Block(C, H) for _ in range(L)]
        self.lnf = nn.LayerNorm(C)

    def __call__(self, idx):
        B, T = idx.shape
        x = self.wte(idx) + self.wpe(mx.arange(T))
        mask = mx.triu(mx.full((T, T), -1e9), k=1)
        for blk in self.blocks:
            x = blk(x, mask)
        return self.lnf(x) @ self.wte.weight.T  # tied head


def gpt2_init(model, L):
    """normal(0, 0.02); residual projections scaled 1/sqrt(2L); biases 0; LN default."""
    def nrm(shape, std):
        return mx.random.normal(shape) * std
    model.wte.weight = nrm(model.wte.weight.shape, 0.02)
    model.wpe.weight = nrm(model.wpe.weight.shape, 0.02)
    for blk in model.blocks:
        for lin in (blk.wq, blk.wk, blk.wv, blk.fc):
            lin.weight = nrm(lin.weight.shape, 0.02); lin.bias = mx.zeros(lin.bias.shape)
        for lin in (blk.wo, blk.proj):  # GPT-2 residual scaling
            lin.weight = nrm(lin.weight.shape, 0.02 / math.sqrt(2 * L)); lin.bias = mx.zeros(lin.bias.shape)


def read_vocab(path):
    with open(path, "rb") as f:
        n = struct.unpack("<I", f.read(4))[0]
        return json.loads(f.read(n).decode("utf-8"))["tokenizer"]["vocab"]


def export_bity1(model, vocab, cfg, step, path):
    C, F = cfg["nEmbd"], 4 * cfg["nEmbd"]
    tensors = []  # (name, float32 C-order flat, shape in OUR [in,out] convention)

    def add(name, arr, shape):
        tensors.append((name, np.ascontiguousarray(arr, dtype=np.float32).reshape(-1), shape))

    add("wte", np.asarray(model.wte.weight), [cfg["vocabSize"], C])
    add("wpe", np.asarray(model.wpe.weight), [cfg["blockSize"], C])
    for i, blk in enumerate(model.blocks):
        add(f"h{i}.ln1.w", np.asarray(blk.ln1.weight), [C])
        add(f"h{i}.ln1.b", np.asarray(blk.ln1.bias), [C])
        for nm, lin in (("wq", blk.wq), ("wk", blk.wk), ("wv", blk.wv), ("wo", blk.wo)):
            add(f"h{i}.attn.{nm}.w", np.asarray(lin.weight).T, [C, C])  # [out,in]->[in,out]
            add(f"h{i}.attn.{nm}.b", np.asarray(lin.bias), [C])
        add(f"h{i}.ln2.w", np.asarray(blk.ln2.weight), [C])
        add(f"h{i}.ln2.b", np.asarray(blk.ln2.bias), [C])
        add(f"h{i}.mlp.fc.w", np.asarray(blk.fc.weight).T, [C, F])
        add(f"h{i}.mlp.fc.b", np.asarray(blk.fc.bias), [F])
        add(f"h{i}.mlp.proj.w", np.asarray(blk.proj.weight).T, [F, C])
        add(f"h{i}.mlp.proj.b", np.asarray(blk.proj.bias), [C])
    add("lnf.w", np.asarray(model.lnf.weight), [C])
    add("lnf.b", np.asarray(model.lnf.bias), [C])

    meta, off = {}, 0
    for name, a, shape in tensors:
        meta[name] = {"shape": shape, "dtype": "f32", "offset": off, "length": int(a.size)}
        off += a.size * 4
    header = {"format": "bity1", "config": cfg,
              "tokenizer": {"type": "char", "vocab": vocab}, "tensors": meta, "step": step}
    hb = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(hb)))
        f.write(hb)
        for _, a, _ in tensors:
            f.write(a.astype("<f4").tobytes())
    return off + 4 + len(hb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=16000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--data", default="corpus/data/bity.corpus.txt")
    ap.add_argument("--vocab-from", default="models/terminal.int8.bity")
    ap.add_argument("--out", default="models/terminal-mlx.bity")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--parity", default=None, help="init+export only, dump logits for a fixed prompt to this json (cross-impl parity gate)")
    a = ap.parse_args()

    mx.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)

    # tokenizer: reuse the canonical vocab so the checkpoint matches ours exactly
    vocab = read_vocab(a.vocab_from)
    stoi = {c: i for i, c in enumerate(vocab)}
    text = open(a.data, encoding="utf-8").read()
    ids = np.fromiter((stoi[c] for c in text if c in stoi), dtype=np.int32)
    V = len(vocab)
    cfg = {"vocabSize": V, "blockSize": a.block, "nLayer": a.layers, "nHead": a.heads, "nEmbd": a.dim}

    model = GPT(V, a.block, a.layers, a.heads, a.dim)
    gpt2_init(model, a.layers)
    mx.eval(model.parameters())

    if a.parity:  # cross-implementation parity gate: export + dump reference logits
        prompt = "guest@bity:~$ ls\ntotal 24\ndrwxr-xr-x 2 guest guest 4096"
        pids = [stoi[c] for c in prompt if c in stoi][:a.block]
        export_bity1(model, vocab, cfg, 0, a.out)
        last = np.asarray(model(mx.array([pids]))[0, len(pids) - 1])
        json.dump({"ids": pids, "logits": [float(x) for x in last]}, open(a.parity, "w"))
        print(f"parity: exported {a.out}; dumped {len(pids)}-token logits -> {a.parity}")
        return
    nparams = sum(p.size for _, p in _flat(model.parameters()))
    print(f"corpus {len(text)/1e6:.2f}M chars, vocab {V} | model {a.layers}L/{a.heads}H/{a.dim}d "
          f"-> {nparams:,} params | engine: MLX/Metal")

    warmup = min(200, a.steps // 10)
    sched = optim.join_schedules(
        [optim.linear_schedule(1e-7, a.lr, warmup), optim.cosine_decay(a.lr, max(1, a.steps - warmup), a.lr * 0.1)],
        [warmup])
    # AdamW with DECOUPLED weight decay applied to 2-D matmul weights only —
    # matches the TS trainer's param groups (embeddings, LayerNorm, biases are
    # excluded). MLX's built-in weight_decay hits every param, so we set it to 0
    # and apply the decay ourselves to just the block Linear weights.
    opt = optim.AdamW(learning_rate=sched, betas=[0.9, 0.95], weight_decay=0.0)
    WD = 0.1
    decay_linears = [lin for blk in model.blocks
                     for lin in (blk.wq, blk.wk, blk.wv, blk.wo, blk.fc, blk.proj)]

    def loss_fn(model, x, y):
        logits = model(x)
        return nn.losses.cross_entropy(logits.reshape(-1, V), y.reshape(-1), reduction="mean")

    lg = nn.value_and_grad(model, loss_fn)
    state = [model.state, opt.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(x, y):
        loss, grads = lg(model, x, y)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        opt.update(model, grads)
        return loss

    T = a.block
    hi = len(ids) - T - 1

    def batch():
        s = rng.integers(0, hi, size=a.batch)
        x = np.stack([ids[i:i + T] for i in s])
        y = np.stack([ids[i + 1:i + 1 + T] for i in s])
        return mx.array(x), mx.array(y)

    t0 = time.time()
    tick, tick_t = 0, time.time()
    for it in range(1, a.steps + 1):
        x, y = batch()
        loss = step(x, y)
        lr = opt.learning_rate  # decoupled weight decay, 2-D matmul weights only
        for lin in decay_linears:
            lin.weight = lin.weight * (1 - lr * WD)
        mx.eval(state, loss)
        tick += a.batch * T
        if it % 20 == 0:
            dt = time.time() - tick_t
            print(f"step {it:5d}  loss {loss.item():.4f}  {tick/dt:,.0f} tok/s")
            tick, tick_t = 0, time.time()
        if it % max(1, a.steps // 10) == 0 or it == a.steps:
            export_bity1(model, vocab, cfg, it, a.out)

    print(f"\ndone in {(time.time()-t0)/60:.1f} min -> {a.out}")


def _flat(tree, prefix=""):
    if isinstance(tree, dict):
        for k, v in tree.items():
            yield from _flat(v, f"{prefix}.{k}")
    elif isinstance(tree, list):
        for i, v in enumerate(tree):
            yield from _flat(v, f"{prefix}.{i}")
    else:
        yield prefix, tree


if __name__ == "__main__":
    main()
