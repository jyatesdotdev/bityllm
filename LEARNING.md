# LEARNING.md — a guided tour of a from-scratch ML stack

This repo is a working, ~4,600-line, **zero-dependency** implementation of everything
it takes to train a neural network from nothing: the automatic differentiation
engine, a transformer, an optimizer, a data pipeline, a training loop, serialization,
and inference. It happens to be configured as a tiny char-level GPT that dreams a
terminal — but the machinery underneath is **general-purpose gradient-based ML**.

This guide is the *how do I read and grow the code* companion. It complements the
others; it does not repeat them:

| Doc | Answers |
|---|---|
| **README.md** | What it is, the live demo, quick start |
| **[DESIGN.md](DESIGN.md)** | *Why* it's built this way (the decisions) |
| **[RUNBOOK.md](RUNBOOK.md)** | The exact *commands* to build/train/eval/deploy |
| **[JOURNEY.md](JOURNEY.md)** | The *story* of building it (a lab notebook) |
| **LEARNING.md** (this) | *How to read the code, and how to extend it* |

> **The big idea to hold onto:** almost none of this code is "about language." It is
> the universal gradient-descent stack — autograd, a module tree, an optimizer, a
> loss, a training loop — with a thin transformer + tokenizer on top. Section 13
> makes that precise: what transfers to CNNs, RNNs, diffusion, and RL, and what
> doesn't.

---

## Table of contents

1. [Prerequisites](#1-prerequisites)
2. [The 60-second mental model](#2-the-60-second-mental-model)
3. [The guided reading path](#3-the-guided-reading-path)
4. [Deep dive: the autograd tape](#4-deep-dive-the-autograd-tape)
5. [Deep dive: one op, end to end](#5-deep-dive-one-op-end-to-end)
6. [Deep dive: logits ↔ token (loss and sampling are mirrors)](#6-deep-dive-logits--token)
7. [The three gates you must respect](#7-the-three-gates)
8. [The checkpoint is the contract](#8-the-checkpoint-is-the-contract)
9. [Two forward passes (training vs inference)](#9-two-forward-passes)
10. [How to extend safely](#10-how-to-extend-safely)
11. [Extension projects, smallest to largest](#11-extension-projects)
12. [Glossary & gotcha index](#12-glossary--gotcha-index)
13. [Does this generalize beyond LLMs?](#13-does-this-generalize-beyond-llms)

---

## 1. Prerequisites

You do **not** need prior ML experience. You do need:

- **Comfortable TypeScript/JS** — classes, generics, and especially **closures** and
  **typed arrays** (`Float32Array`/`Int32Array`). The entire autograd engine is
  closures capturing activations; if closures feel shaky, shore that up first.
- **High-school calculus** — what a derivative is, the **chain rule**, partial
  derivatives. You won't derive backprop; you must be willing to read a `backward()`
  closure as "the chain rule for this one op."
- **Basic linear algebra** — vectors, matrices, **matmul**, transpose. Every `Linear`
  layer and all of attention are matmuls; their gradients are matmuls with a transpose.
- **The shape of the loop** (one paragraph): parameters → forward pass → a single
  scalar **loss** → gradients → gradient-descent update, repeated. This repo builds
  every arrow of that loop from scratch.

No GPU and no Python are needed for the core reading path. `npm install` pulls **types
only** (zero runtime deps); `npm test` runs everything. MLX (Python) and WebGPU (Deno)
are optional side quests.

---

## 2. The 60-second mental model

```
        ┌─────────────────────────── one training step ───────────────────────────┐
 params │ forward: run ops → they record a tape → produce a scalar loss             │
   │    │ backward: seed dL/dL = 1, walk the tape in reverse, accumulate grads      │
   │    │ clip:    scale grads if their global norm is too big                      │
   ▼    │ step:    AdamW nudges each param down its gradient                         │
 update │ zero:    clear grads for the next step                                    │
        └───────────────────────────────────────────────────────────────────────────┘
                                  repeat ~16,000×
                                       │
                                       ▼
                export a checkpoint ── the browser re-runs the SAME math,
                                       without any autograd, one token at a time
```

Two ideas do most of the work:

1. **Define-by-run autograd.** There is no separate "graph you declare." The graph
   *is* the sequence of op calls you make in the forward pass, recorded as a side
   effect onto a global **tape**. `backward()` replays that tape in reverse. (§4)
2. **The loss is just one scalar node.** Nothing *upstream* of the loss — the model,
   the optimizer, the loop — cares what it measures. Swap cross-entropy for an MSE op
   and you're training a regressor; the only change is writing that one new loss op.
   (§6, §13)

---

## 3. The guided reading path

Read the code in this order. It builds strictly upward: determinism → kernels →
autograd → layers → model → optimizer → data → the loop → serialization → inference.
Each row has a **checkpoint question** — if you can answer it, move on.

| # | File | What you learn | Before moving on, you should be able to answer… |
|---|------|----------------|--------------------------------------------------|
| 1 | `src/core/rng.ts` | A seeded PRNG (mulberry32) + Box–Muller normals; **no `Math.random` anywhere** | Why must every init/dropout/sample draw from a seeded RNG? (repro & parity) |
| 2 | `src/backend/backend.ts` | The **backend seam**: primitive kernels (`matmul`, `gelu`, `layerNorm`, …) and their `*Backward` halves as plain loops | Where does the actual arithmetic live — in autograd, or here? |
| 3 | `src/core/tensor.ts` | **Define-by-run autograd**: the tape, `opOutput`, `backward()`, `+=` accumulation | Why is reverse *insertion* order a valid reverse-topological order? |
| 4 | `src/core/ops.ts` | How one differentiable op is built: forward kernel + backward **closure** | What does `matmul`'s backward do, and why the transposes? |
| 5 | `test/gradcheck.test.ts` | How you *know* backward is right: finite-difference checks in f64 | What would a failing grad-check tell you? |
| 6 | `src/nn/module.ts` | The tiny `Module` base: `reg()`, `sub()`, recursive `parameters()` | How are all trainable tensors collected for the optimizer? |
| 7 | `src/nn/layers.ts` | `Linear` / `LayerNorm` / `MLP` as compositions of ops | Why does `Linear` store its weight as `[in, out]`? |
| 8 | `src/nn/attention.ts` | Causal multi-head self-attention, built only from ops | What does the **causal mask** enforce, and why `/√head_dim`? |
| 9 | `src/nn/gpt.ts` | The whole model: embeddings → pre-norm blocks → tied head | Why is the LM head "free" (`logits = x @ wteᵀ`)? |
| 10 | `src/optim/adamw.ts` | AdamW + weight decay + grad clip + cosine LR | Which line *is* the Adam update rule? |
| 11 | `src/tokenizer/char.ts` | Char-level tokenization; the vocab ships in the checkpoint | What is "a token" here, versus in a BPE model? |
| 12 | `src/data/dataset.ts` | Batching for next-token prediction; `y` is `x` shifted by one | Where does supervision come from with no labels? |
| 13 | `src/train.ts` | The training loop in ~50 lines | Name the 5 steps of the inner loop, in order. |
| 14 | `examples/train-terminal.ts` | The real CLI: corpus → tokenizer → dataset → `GPT` → `train()` → checkpoint | What turns size flags into a model? |
| 15 | `src/io/checkpoint.ts` | The `bity1` format + per-row int8 quantization | What travels *inside* the checkpoint besides weights? |
| 16 | `src/infer/sampler.ts` | logits → a token: temperature, top-k, categorical draw | What does temperature actually change? |
| 17 | `src/infer/session.ts` | Autograd-free inference with a **KV-cache** | Why is per-token cost O(params), not O(params·T)? |

Everything else (`src/gpu/*`, `src/infer/shell*.ts`, `corpus/`) is *infrastructure and
the application* — worth reading once you understand the core above.

---

## 4. Deep dive: the autograd tape

The single most important file is `src/core/tensor.ts`. Its whole job:

- A `Tensor` carries `data` (the values), `grad` (the accumulated gradient, lazily
  allocated), and — if it's an op output — a list of `parents` and a `backwardFn`
  closure.
- Every op in `ops.ts` ends by calling `opOutput(...)`, which (when gradients are on)
  **pushes the output tensor onto a global `tape`** and stores its backward closure.
- `Tensor.backward()` seeds `dL/dL = 1` on the scalar loss, then walks the tape from
  the last node to the first, calling each `backwardFn`.

**Why reverse insertion order just works.** A node is always created *after* the nodes
it depends on (you can't `matmul` two tensors before you have them). So walking the
tape backward visits every node only after all its consumers — a valid reverse
topological order, with no graph sort needed.

**Trace it by hand.** Say `x` and `W` are parameters and we compute `y = x + b`, then
`z = y @ W`, then a scalar `L = sum(z)`:

```
forward (tape grows):   [ y=x+b ,  z=y@W ,  L=sum(z) ]
backward (reverse):     seed dL/dL = 1
    L=sum(z).backwardFn  → dz  += 1 (broadcast)
    z=y@W.backwardFn     → dy  += dz @ Wᵀ ;  dW += yᵀ @ dz
    y=x+b.backwardFn     → dx  += dy       ;  db += sumTo(dy, shape(b))
```

Two things a newcomer must internalize here (they are the usual sticking points):

- **Backward `+=`, not `=`.** Grads *accumulate* (`addInto`). This is why a parameter
  used in two places (or a **tied** weight used as both the embedding and the output
  head, `gpt.ts`) gets the *sum* of both gradient paths — automatically, with no
  special-case code. People hunt for the magic; the `+=` **is** the magic.
- **Broadcast forward ⇒ sum backward.** `b` (a bias) was *copied* across positions in
  the forward `add`. The adjoint of a copy is a **sum**: `db = sumTo(dy, shape(b))`.
  Getting this "sumTo" right is the one discipline that keeps shapes correct.

The tape is **single-use**: `backward()` nulls each closure (freeing the captured
activations for GC) and clears the tape, so you can't call it twice on one graph.
`noGrad()` disables the tape entirely — that's how eval and generation avoid paying
for a graph they'll never backprop.

---

## 5. Deep dive: one op, end to end

Every op in `ops.ts` is the same shape — **forward via a backend kernel, backward as a
closure that captures only what it needs**. `matmul` is the template you'll copy when
you add your own op (§10):

```ts
export function matmul(a: Tensor, b: Tensor): Tensor {
  const nd = B().matmul(a.nd, b.nd);            // forward: ask the backend
  return opOutput(nd, [a, b], (out) => () => {  // backward closure, captures a, b, out
    const g = { data: out.grad!, shape: out.shape };            // upstream grad dL/dC
    if (a.requiresGrad) addInto(a.ensureGrad(), B().matmul(g, b.nd, { transposeB: true }).data); // dL/dA = g @ Bᵀ
    if (b.requiresGrad) { /* dL/dB = Aᵀ @ g, with a batch-sum if B is broadcast */ }
  });
}
```

The backward rule for `C = A @ B` is pure matrix calculus:

```
dL/dA = g @ Bᵀ        dL/dB = Aᵀ @ g          (g = dL/dC, the upstream gradient)
```

Intuition: each is "the *other* input, transposed, times the incoming gradient," which
is the only way to make the shapes line up (`dL/dA` must have `A`'s shape). The
`transposeA/transposeB` flags let the CPU/GPU kernel do it without allocating a
transposed copy. `test/gradcheck.test.ts` verifies this exact rule numerically — read
that test right after this op and the loop closes: *you can see that the math is right.*

---

## 6. Deep dive: logits ↔ token

Training and generation are **mirror images** that share a softmax.

- **Training (logits → loss), `ops.crossEntropyLogits`.** The model outputs `logits`
  (one score per vocab entry). Cross-entropy is
  `loss = mean( logsumexp(z) − z[target] )` — computed via `logsumexp` so it never
  forms a softmax that could overflow. Its gradient is unusually clean:
  `dL/dz = (softmax(z) − onehot(target)) / N`. "Push the probability of the right token
  up, everything else down."
- **Inference (logits → token), `src/infer/sampler.ts`.** Divide logits by a
  **temperature** (higher = flatter = more random), optionally keep only the **top-k**,
  softmax into probabilities, and draw one token categorically from a seeded RNG.

Same softmax, opposite directions: training *measures* how wrong the distribution is;
sampling *draws* from it. A classifier does the training half and then just `argmax`es
once — no feedback loop. An autoregressive LM feeds its drawn token back in and repeats
(§9).

---

## 7. The three gates

The project's core discipline (DESIGN §19, AGENTS invariant #3) is an **order of
trust**. When something looks wrong, never skip upstream:

1. **grad-check** (`test/gradcheck.test.ts`) — finite differences in f64 confirm every
   backward closure matches the analytic gradient. *If this fails, nothing downstream
   is meaningful.*
2. **overfit one batch** (`test/*` overfit gate) — the full optimizer stack can drive
   loss on a single batch to ~0. Proves the loop (forward→loss→backward→step) is wired
   correctly end to end.
3. **parity** (`bench/gpu-parity.ts`, MLX `--parity`) — a second backend/trainer
   produces the same logits (≤ ~1e-6). Only *then* do you trust GPU or a new trainer.

"Loss went down" is **not** a gate — it can fall while a subtle bug quietly caps the
model's ceiling. Internalizing this order is the biggest mindset shift for a newcomer.

---

## 8. The checkpoint is the contract

`src/io/checkpoint.ts` defines `bity1`:

```
[u32 headerLen][JSON header: config + tokenizer vocab][raw f32 or per-row int8 blobs]
```

The key idea (**invariant #1**): the *config and the vocabulary travel inside the
checkpoint*. You can train in TypeScript, Python/MLX, or the WebGPU trainer — the
browser loads the same `bity1` unchanged and reconstructs the exact tokenizer. Two
learner traps live here:

- **Per-row int8 quantization.** Each weight row is scaled to fit `[-127, 127]` by
  `scale = max(|row|) / 127`; store `int8 + scale`, dequantize on load. ~4× smaller
  file, tiny accuracy cost — a general model-compression technique, not LLM-specific.
- **The `[in, out]` weight-layout rule.** This repo stores `Linear` weights as
  `[in, out]`; PyTorch/MLX store `[out, in]`. Exporting from another framework
  **without transposing** produces garbage logits that still *look* plausible. This is
  the classic "checkpoint looks fine, outputs are nonsense" bug.

---

## 9. Two forward passes

There are **two** implementations of the model's forward pass, and they must stay
identical:

- **Training** — `src/nn/gpt.ts` `forward()`. Batched, builds the autograd tape,
  processes all `T` positions at once.
- **Inference** — `src/infer/session.ts`. Hand-written, **no autograd**, one token at a
  time, with a per-layer **KV-cache**: the keys/values for past positions are stored,
  so a new token only computes its own Q and attends to the cached K/V. That's why
  per-token cost is **O(params)**, not **O(params·T)** — you don't recompute the whole
  prefix each step. When the context fills `blockSize`, `rewind()` slides the window.

The `session.ts` code re-implements `gpt.ts` by hand — which is exactly why the
**parity test** exists: it's the guarantee the two never drift. If you change the model,
you change *both*, then re-run parity.

---

## 10. How to extend safely

The golden rule: **add the gate before the feature.** To add a new differentiable op:

1. Add the forward kernel + its `*Backward` half to `src/backend/backend.ts` +
   `src/backend/cpu.ts` (plain array loops).
2. Add the op wrapper to `src/core/ops.ts` (forward call + backward closure with
   `addInto`).
3. **Add a finite-difference case to `test/gradcheck.test.ts` and make it pass** — this
   is your correctness proof, before you use the op anywhere.
4. Use it (e.g. in a layer), then re-run the overfit-one-batch gate.
5. If it appears at inference, mirror it in `src/infer/session.ts` and re-check parity.

That sequence (grad-check → overfit → parity) is the same one from §7 — it's not
ceremony, it's how you localize bugs to one layer of the stack.

---

## 11. Extension projects

Ordered smallest → largest. Each names the files you touch and the gate that tells you
it worked.

1. **Train a name generator** *(flags only)* — put first names in a text file, point
   `examples/train-terminal.ts` at it with `--data`, shrink the model
   (`--layers 2 --heads 2 --dim 64 --block 16`), train a few thousand steps, generate.
   Proves you can drive the whole pipeline; reuses the char tokenizer unchanged.
2. **Add a new op** *(the canonical autograd exercise)* — implement `tanh` (or
   `silu`, `mul`) as forward-kernel + backward-closure, then add a grad-check case.
   The passing check is your grade.
3. **Swap GELU for your activation** in `src/nn/layers.ts` `MLP.forward`, retrain a
   tiny model, confirm it still overfits one batch. Catch: inference in `session.ts`
   hardcodes tanh-GELU — update it too. A hands-on lesson in the parity contract.
4. **Train a tiny MLP classifier** *(no GPT at all)* — a ~40-line script using
   `Tensor`/`ops`/`Linear`/`AdamW` (all exported from `src/index.ts`) to classify a toy
   2-D dataset with cross-entropy. Proves the autograd engine is **general**, not
   GPT-specific — the point of §13.
5. **Swap the tokenizer** — implement a word-level `Tokenizer` (same `size`/`encode`/
   `decode` interface) and carry its vocab in the checkpoint header. Teaches the
   tokenizer → dataset → checkpoint contract.
6. **Add nucleus (top-p) sampling** to `src/infer/sampler.ts` beside temperature/top-k,
   wire it through `InferenceSession.stream`, compare generations at a fixed seed.
7. **Sinusoidal positions** — replace learned `wpe` with fixed sinusoidal encodings in
   `gpt.ts`, keep them out of the AdamW param groups, retrain a micro model. Forces you
   to understand embeddings + `paramGroups` + the checkpoint's named tensors.
8. **Add a corpus generator family** — a new `corpus/generators/*.mjs` (follow
   `net.mjs`), wire it into `index.mjs`, `npm run corpus`, retrain, score with
   `bench/eval.mjs`. The project's real lever: *the model learns what the data forces.*
9. **Prove a second trainer stays honest** — run the MLX parity gate
   (`train/mlx_train.py --parity`), reproduce the `[in,out]` transpose-on-export rule,
   diff logits against the TS forward. Teaches *why* the checkpoint is the contract.
10. **Instrument the tape** — after one forward pass, print each tape node's op label,
    shapes, and parent count (via `Tensor.label`/`parents`). Turns the abstract "graph
    on a tape" into something you can literally see.

---

## 12. Glossary & gotcha index

- **Tape** — the global list of op-output tensors, in creation order; `backward()`
  replays it in reverse. (`tensor.ts`)
- **Backward closure** — the function an op stores to turn its output-gradient into its
  inputs'-gradients (a vector-Jacobian product). It `+=`s into parents. (`ops.ts`)
- **`+=` accumulation** — why shared/tied parameters are correct for free; the whole
  mechanism behind weight tying. (§4)
- **sumTo / broadcast-reduction** — backward of a forward broadcast is a sum over the
  broadcast axes. The #1 shape-bug source. (`ops.ts add/matmul`)
- **Tied head** — `logits = x @ wteᵀ`; the input embedding table doubles as the output
  projection. Meaningful *because* both live in the same vocab space. (`gpt.ts`)
- **Pre-norm block** — `x += attn(LN(x)); x += mlp(LN(x))`. LayerNorm *before* the
  sublayer, residual *around* it. Stabilizes deep training. (`gpt.ts`)
- **KV-cache** — stored keys/values for past tokens so each new token is O(params).
  (`session.ts`)
- **Per-row int8** — the ~4× quantization for deploy; `scale = max(|row|)/127`.
  (`checkpoint.ts`)
- **Decoupled weight decay** — AdamW shrinks weights *outside* the adaptive step
  (`w -= lr·wd·w`), applied only to 2-D matmul weights, not biases/LayerNorm/embeddings.
  (`adamw.ts`, `gpt.ts paramGroups`)
- **`[in,out]` layout** — this repo's Linear weight shape; transpose when importing
  PyTorch/MLX weights or logits will be garbage. (§8)
- **Gate order** — grad-check → overfit-one-batch → parity. "Loss went down" is not a
  gate. (§7)

---

## 13. Does this generalize beyond LLMs?

**Yes — overwhelmingly.** This is the most important thing to take away. Strip the word
"language" and what remains is the standard **gradient-based deep-learning stack**, and
that stack is shared by essentially every neural network. The transformer-and-tokenizer
part is a thin, swappable cap on top.

### 13a. The universal half (transfers to *all* gradient-based ML)

Every item below is model-agnostic — swap the layers and loss, and this exact code
trains a different family:

| Building block | Where it lives | Why it's universal |
|---|---|---|
| **Reverse-mode autograd** | `core/tensor.ts` tape + `core/ops.ts` | *The* engine of all gradient-based ML. Any model that's a composition of differentiable ops — MLP, CNN, RNN, ViT, VAE, diffusion U-Net, an RL policy net — trains by exactly this: build a tape, seed `dL/dL=1`, sweep back. Only the *ops on the tape* change. |
| **Module / parameter tree** | `nn/module.ts` | PyTorch `nn.Module`, Keras `Layer`, Flax `Module` are all this: a tree of layers whose trainable leaves are collected for the optimizer. Orthogonal to model family. |
| **Optimizer (AdamW + clip + cosine)** | `optim/adamw.ts` | The optimizer sees only `(param, grad)` pairs — it knows nothing about transformers. Adam/SGD/RMSProp, decoupled weight decay, grad clipping, warmup+cosine are standard across CNNs, RNNs, ViTs, diffusion, RL. |
| **Scalar loss → backward** | `ops.crossEntropyLogits`, `gpt.loss` | The contract everywhere: forward → reduce to **one** scalar → backward. The loss is the *swappable, task-defining* piece (see below). |
| **The training loop** | `train.ts` | `batch → forward → loss → backward → clip → step → zero`, plus periodic `noGrad` eval. Identical for every gradient model; only `getBatch` and `loss` are model-specific. |
| **train/eval toggle** | `noGrad()`, `Module.train()` | Every framework has `train()`/`eval()` and `no_grad()`; at inference you drop the tape, and stochastic/normalization layers switch behavior. |
| **Weight init** | `randn(std)`, GPT-2 `0.02` + `1/√(2L)` scaling | Controlling activation/gradient variance is universal — Xavier (tanh), He/Kaiming (ReLU CNNs), orthogonal (RNNs). The residual scaling here is the transformer flavor of the same concern. |
| **Regularization** | AdamW weight decay, `ops.dropout` | Weight decay and dropout are model-agnostic; the general principle (penalize/perturb to close the train/val gap) even has non-NN analogs (tree depth, boosting shrinkage, `k` in k-NN). |
| **Mini-batch SGD** | `data/dataset.ts` | Sampling random minibatches to cheaply estimate the full-data gradient is the backbone of *all* large-scale training. Only the *sampler* differs (image crops, replay-buffer transitions, labeled rows). |
| **Checkpointing + quantization** | `io/checkpoint.ts` | Save/restore params+config+preprocessing is universal MLOps (safetensors/ONNX are the same idea); int8 quantization compresses CNNs/ViTs/LLMs alike. |
| **Backend seam + grad-check** | `backend/*`, `test/gradcheck.test.ts` | Separating kernels from autograd lets one autograd graph run on swappable backends — here the CPU backend at f32 (training) or f64 (tight grad-check tolerances). (The WebGPU path is a *separate* f32 inference/training route, not this pluggable seam; bf16 lives only in the MLX trainer.) That device/dtype abstraction is what every framework has, and finite-difference grad-checking is the universal correctness gate for *any* hand-written op. |

**If you understand this repo, you understand how a diffusion model or an RL policy net
is trained.** They reuse this whole column — same autograd, same AdamW, same loop.

### 13b. The genuinely LLM/transformer-specific half (and its analogs elsewhere)

A short list — and each has a clear analog in other domains:

| LLM-specific piece | Where | Analog elsewhere |
|---|---|---|
| **Causal self-attention** (`Q·Kᵀ/√d`, causal mask, softmax, `·V`) | `nn/attention.ts` | Attention is **not** language-specific. *Bidirectional* self-attention powers Vision Transformers and BERT; cross-attention conditions diffusion image models. Only the **causal mask** (attend to ≤ t) is autoregressive-LM-specific — delete it and the same block is a ViT/encoder block. |
| **Next-token target** (`y = x` shifted by 1) | `data/dataset.ts`, `gpt.loss` | The self-supervision idea (predict part of the input from the rest) reappears as masked-patch prediction (MAE/BERT), autoregressive pixels/audio (PixelCNN/WaveNet), contrastive learning. Supervised CNNs use an external label instead. |
| **Token + positional embeddings** | `gpt.ts` `wte`/`wpe` | Embedding lookups generalize to any categorical input (users/items in recommenders, tabular categoricals). **Positional** embeddings exist *only because attention is order-blind* — CNNs/RNNs get order for free from convolution/recurrence and need none. |
| **Char tokenizer / closed vocab** | `tokenizer/char.ts` | Vision doesn't tokenize — it normalizes pixels (ViT cuts images into patches, the visual analog of tokenizing). The general concept is "deterministic preprocessing into model-ready tensors," which every pipeline has. |
| **Autoregressive temperature/top-k sampling** | `infer/sampler.ts` | A classifier `argmax`es once — no feedback loop. Diffusion also generates iteratively, but over a fixed denoising schedule on continuous latents, not categorical draws fed back as input. |
| **Pre-norm block + tied head** | `gpt.ts` | Residual connections and normalization are **universal** (ResNet skips predate transformers; BatchNorm/GroupNorm are the CNN norm analogs). Only the exact ordering and the embedding↔output *tie* are LM conventions. |

### 13c. Could this codebase train other model families?

Concretely, what you'd swap — and where reality bites:

| Family | Verdict | What you'd change / what's missing |
|---|---|---|
| **Logistic regression** | ✅ **Today** | It *is* the small end of this stack: one `Linear` + `crossEntropyLogits` + `AdamW`. Extension project #4 is basically this. |
| **MLP classifier** | ✅ **Today** | Stack `Linear` + `gelu` + `Linear`, use cross-entropy. All exported from `src/index.ts`. |
| **Vision Transformer (ViT)** | 🟡 **Minor changes** | Bidirectional attention is already expressible (drop `causalMask`); you'd add a *patchify* step and an MSE/CE head. No new autograd needed. |
| **RNN / LSTM** | 🟡 **A few ops** | The tape handles recurrence fine (`+=` makes weight-reuse-across-time correct for free). You'd add a couple of elementwise ops (sigmoid, `tanh`, `mul`) and a small loop. |
| **Diffusion model** | 🟡 **New loss + noising** | **Same autograd, same AdamW, same loop.** You change only (a) the loss (a denoising **MSE**, not cross-entropy) and (b) a stochastic forward process that manufactures noisy targets. |
| **RL policy-gradient** | 🟡 **External env** | The network+optimizer half maps directly on. The policy-gradient update `−Σ log π(a|s)·advantage` is an *ordinary* scalar loss differentiated by this same autograd. The environment/rollout/advantage machinery lives *outside* the repo. |
| **CNN (conv nets)** | 🔴 **New primitive** | There is **no convolution or pooling op** anywhere. A CNN needs a genuinely new differentiable primitive (+ its grad-check), not a config change. |
| **Gradient-boosted trees** | 🔴 **Different mechanism** | Despite the name, GBTs don't backprop. They use the gradient of the loss w.r.t. the *predictions* to define pseudo-residuals that the next tree fits — none of this repo's autograd/optimizer applies. |
| **k-NN** | 🔴 **No training at all** | No parameters, no gradients, no loss — it just stores data. Entirely outside this machinery. |

### 13d. Guardrails — five things that sound right but aren't

Teaching material is worse than useless if it's wrong. Watch these:

1. **"Attention is what makes language models work / attention is language-specific."**
   No. Attention is a general sequence/set-**mixing** primitive (ViT, BERT, diffusion
   cross-attention). What's LM-specific here is the **causal mask**, not attention.
2. **"Diffusion / RL are a totally different kind of learning."** No — they use the
   *same* autograd, the *same* AdamW, the *same* forward→scalar-loss→backward loop. Only
   the **loss** (and, for diffusion, the data-noising process) differs.
3. **"Gradient-boosted trees are gradient descent too."** No. "Gradient" refers to the
   gradient of the loss w.r.t. *outputs* to define residuals; there's no backprop into
   continuous parameters. Shared word, different mechanism.
4. **"Positional embeddings are a fundamental neural-net component."** No — they exist
   *only* to fix attention's order-blindness. CNNs/RNNs need none.
5. **"It's all matmuls, so it could train images fine."** Conceptually yes, in practice
   no: there's no conv op, and the dense TypeScript tape is tuned for a ~10M-param char
   LM — image-scale training would be memory/time-prohibitive. That's an *engineering*
   limit, not a conceptual one — but state both halves honestly.

### 13e. Worked example: what would reinforcement learning actually take?

RL feels like a different world, but it reuses **this repo's entire
network + optimizer + autograd half unchanged**. Here's the honest accounting.

**What stays exactly as-is:**
- **The network.** A *policy* is just a model that outputs logits over **actions**
  instead of over tokens — structurally identical to what `GPT.forward` /
  `Linear` / `MLP` already do. For a classic control task it's a small MLP you can
  build today from `src/index.ts` exports.
- **Action selection.** `sampleLogits` (`src/infer/sampler.ts`) already samples from
  a categorical distribution — that *is* sampling an action from a policy.
  Temperature/top-k become your **exploration** knobs.
- **Autograd, AdamW, `clipGradNorm`, the loop.** Untouched. (Grad clipping is
  *especially* standard in RL — returns are high-variance.)

**What changes — two things:**

1. **Where data comes from.** There's no fixed corpus. You replace
   `Dataset.getBatch` with an **environment + rollout loop** (external code, e.g.
   CartPole): `reset → observe state → policy samples action → env returns (reward,
   next state) → repeat`. A "batch" becomes a buffer of `(state, action, reward)`
   transitions you just collected. The environment is a *simulator*, not ML — it
   lives outside this repo.

2. **The loss.** Not cross-entropy against a fixed label. The canonical
   policy-gradient (REINFORCE) objective is:

   ```
   L = − Σ_t  log π(aₜ | sₜ) · Aₜ
   ```

   where `π(aₜ|sₜ)` is the policy's probability of the action it actually took (a
   softmax over the policy logits — the *same* softmax as everywhere), and `Aₜ` is
   the **advantage** (how much better that action turned out than expected — e.g. the
   discounted future reward, minus a baseline).

**The beautiful part — you already have most of the loss.** Cross-entropy *is*
`−log softmax(z)[target]`. So `−log π(aₜ|sₜ)` is exactly `crossEntropyLogits(policy_logits, actions)`.
That makes **REINFORCE literally advantage-weighted cross-entropy** — the loss you're
already reading in `src/core/ops.ts` and `src/backend/cpu.ts`, just with a per-sample
weight `Aₜ`. "Reward-weighted next-action prediction."

**What you'd add (small, concrete):**
- **Per-sample weighting** in cross-entropy (it's currently an unweighted mean) — a
  few lines in `ceForward`/`ceBackward`.
- **Advantage estimation** — compute discounted returns `Gₜ = Σ γᵏ rₜ₊ₖ` in plain
  JS; optionally subtract a **value baseline** `V(sₜ)` from a second network head
  trained with **MSE** against `Gₜ` (that needs the **MSE op** from §13c — one new
  differentiable op). Actor-critic / PPO add a value head, GAE, and a clipped
  surrogate — but every one of those is "assemble a *different scalar*, then
  `backward()` the same way."

**What's genuinely outside the repo:** the environment/rollout machinery (external),
one MSE op, and a small weighted-CE tweak. **Nothing in the autograd or optimizer
changes.** The "special kind of gradient" people imagine RL needs is a myth — the
cleverness is entirely in *constructing the reward-weighted scalar and estimating
advantages*; the differentiation is the same reverse-mode sweep from §4.

### 13f. The one-sentence answer

> The transformer is the interesting *cap*; the autograd engine, the module tree, the
> optimizer, the loss-and-loop, the checkpointing — **the ~80% of this repo that isn't
> "language" — is the universal machinery of gradient-based ML, and it transfers, mostly
> unchanged, to nearly every neural network you'll ever train.**

---

*Found something confusing while reading? That's a documentation bug — the code is meant
to be read. See AGENTS.md for the repo map and invariants.*
