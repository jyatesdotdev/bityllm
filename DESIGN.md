# bityllm — Design Document

> A tiny GPT you **train from scratch** in **pure TypeScript** with **zero runtime dependencies**, that ships to the **browser** for **cheap, streaming inference** — powering a **hybrid terminal**: deterministic commands run as real code over an in-memory filesystem, and generative "binaries" (`ping`, `reboot`, `fortune`, …) are hallucinated by the model.

Status: **Draft v5** · Train: Node ≥ 26 (canonical) · GPU via Deno/WebGPU · fast path via Python/MLX · Run: modern browsers (and Node) · License: MIT

> **v5 note (the hybrid pivot — supersedes the "everything is dreamed" framing below).** The breadth ceiling — the model gibberished outside ~50 commands, and the referential-consistency corpus burden (`ll`/`cat`-after-`rm`) was structural — forced a split. **Deterministic + stateful + must-be-consistent commands are now REAL CODE** (a virtual filesystem `src/infer/vfs.ts`, ~35 coreutils `src/infer/coreutils.ts`, and a mini-shell with pipes/redirects/globs/`&&` `src/infer/shell-exec.ts`), routed ahead of the model in `Shell.run`. The **model is reserved for generative commands** (`ping`/`git`/`ps`/`man`/fun) + a graceful `command not found`. The corpus was rebalanced capture-heavy accordingly (68% real / 32% synthetic; CORE commands filtered out), and the deployed default is now **MINI v9** trained on that corpus. Where §2 below says "every binary is hallucinated," read it as "every *generative* binary." The autograd/training/checkpoint/inference core (§4–§15) is unchanged.
>
> **v4 note.** The v1–v3 design below shipped. This revision folds in the findings that came *after* v1 — a second training backend (MLX, §15.1), the corpus-v8 coverage audit and the data-vs-capacity lesson (§2.4, §9.4), the measured zero-overfitting result (§12), and the in-browser model-selector (§2.5). They are integrated where they belong rather than bolted on as an appendix; each is a decision or a measured result, not a how-to (that's `RUNBOOK.md`).

---

## 1. Goals & Non-Goals

### Goals
- **Educational, from-scratch GPT.** Implement a decoder-only transformer *and* the reverse-mode autograd that trains it. Nothing imported does the "AI" for us.
- **Zero runtime dependencies.** No tfjs/onnx/BLAS packages. Only Node/Web built-ins.
- **Cheap browser inference is a primary goal.** The trained model loads and generates **token-by-token, batch-1, in the browser, at interactive latency, in a tiny JS bundle** with **no training code shipped**.
- **Drives a hybrid virtual terminal.** The end product is a shell that routes deterministic commands to real code (a virtual filesystem + coreutils + mini-shell) and *generative* ones (`ping`, `reboot`, `fortune`, …) to thin wrappers that run **conditioned inference** to stream realistic, fun output. See §2 and the v5 note above.
- **Isomorphic core.** All math/model code runs unchanged in Node (training) and the browser (inference). Platform specifics live behind `Env` adapters.
- **Correctness first.** Gradients validated by finite-difference checks; the full train path validated by an "overfit one batch" test before any real training.
- **A pluggable compute backend.** All numerics sit behind one `Backend` interface so WASM SIMD / worker threads / WebGPU can drop in later without touching model code.

### Non-Goals (v1)
- Not competitive with PyTorch/GGML training speed — pure TS is the *starting* backend, by choice.
- No BPE in v1 (char-level is ideal for terminals: ~100-char ASCII vocab). `Tokenizer` interface leaves room.
- No GPU in v1 (WebGPU is a designed-for future backend).
- No distributed training, no mixed precision, no autograd in the shipped inference bundle.
- Not loading pretrained GPT-2 weights in v1 (checkpoint format is shaped to allow it later).
- The terminal touches **no real host/network** — no sockets, no subprocesses, no real disk. (Since the v5 hybrid pivot there *is* a real in-memory **virtual** filesystem for FS commands, but it is a sandboxed data structure, not the machine's disk; generative binaries still hallucinate plausible output.)

### Locked decisions
| Question | Decision | Consequence |
|---|---|---|
| What to build | **Train from scratch** | Full autograd + optimizer + training loop |
| Where it runs | **Isomorphic; browser inference is first-class** | Pure-compute core; separate **lean inference path**; `Env` adapters |
| Math backend | **Pure TypeScript first** | Naive `Float32Array` kernels behind a `Backend` seam |
| Application | **Virtual terminal: a registry of "binaries" over one conditioned model** | Per-binary corpus + prompt/sampling/stop/pacing; streaming; KV-cache; small quantized checkpoint |

---

## 2. Driving Application: A Virtual Terminal of Hallucinated "Binaries"

The product is a web page that *feels like* a shell — but its commands are not real programs. Each **binary** (`ping`, `reboot`, `fortune`, `sl`, …) is a thin wrapper that runs the tiny model with a command-specific prompt and streams back realistic, **fun** output. It's a dream of a computer: a char-level model captures the *texture* of each tool (ping's per-second replies, reboot's shutdown sequence, git's status) far better than its correctness.

### 2.1 One model, many binaries (recommended)
A single char-level GPT is **prompt-conditioned** on the command line. To "run" a binary, the shell seeds the context with that command's line and streams the model's continuation:

```text
guest@bity:~$ ping -c 3 bity.dev
PING bity.dev (93.184.216.34): 56 data bytes
64 bytes from 93.184.216.34: icmp_seq=0 ttl=54 time=11.4 ms
64 bytes from 93.184.216.34: icmp_seq=1 ttl=54 time=10.9 ms
64 bytes from 93.184.216.34: icmp_seq=2 ttl=54 time=11.1 ms
--- bity.dev ping statistics ---
3 packets transmitted, 3 received, 0.0% packet loss
guest@bity:~$ ▮
```

Because the model is char-level, it naturally **copies parameters out of the prompt** — `ping bity.dev` tends to produce `PING bity.dev (...)` because those characters are right there in context. That single property is why parameterized commands work with no special machinery.

> **Design choice — one model vs per-command models.** We use **one shared, prompt-conditioned model** exposed through a registry of binaries: tiny download, simple training, and commands share structure (prompts, paths, error styles). The alternative — a separate micro-model per binary — is more modular and could sharpen each command, but multiplies training and download. **Recommended default: one model**; the registry is designed so per-command models could slot in later behind the same `Binary` interface. Say the word if you'd rather go per-command.

### 2.2 The `Binary` abstraction
The virtual shell is a **registry of binaries**. Running a line = parse `argv` → look up `argv[0]` → run its handler.

```ts
export interface Binary {
  name: string;                                   // "ping"
  synopsis?: string;                              // shown by `help`
  kind: "model" | "scripted";                     // hallucinated vs deterministic
  buildPrompt?(argv: string[]): string;           // model: argv → prompt line(s)
  sampling?: { temperature?: number; topK?: number; topP?: number };
  stop?: string[];                                // e.g. [PROMPT]
  maxNewTokens?: number;
  pacing?: { charsPerSec?: number; perLineDelayMs?: number };   // the "fun" is partly timing
  run?(argv: string[], ctx: ShellContext): Promise<void>;       // scripted / hybrid: full control
}
// ShellContext gives scripted binaries the renderer + a handle to the model:
export interface ShellContext {
  io: { write(s: string): void; delay(ms: number): Promise<void>; clear(): void };
  session: InferenceSession;                       // hybrids can call the model for sub-steps
  prompt: string;
}
```

- **Model binaries** (`ping`, `reboot`, `traceroute`, `neofetch`, `fortune`, `cowsay`, `sudo`, `curl`, `sl`) build a prompt from argv, then stream conditioned inference until a stop-sequence.
- **Scripted binaries** (`clear`, `help`, `echo`, maybe `cd`/`ls`) are plain deterministic functions — no model — for things that should be exact or interactive.
- **Hybrids** are scripted binaries that *also* call the model: e.g. `reboot` prints a model-generated shutdown sequence, `ctx.io.delay(1500)`, `ctx.io.clear()`, `ctx.session.reset()`, then a model-generated boot/login banner.
- **Fallback** (`argv[0]` unknown): free-form transcript continuation, or a hallucinated `bity: command not found: <cmd>` (itself a binary).
- **Pacing** gives each binary charm: `ping` ≈ one line/second; `sl` animates; `reboot` pauses then "reconnects". The renderer honors `pacing` regardless of how fast inference actually is.

This is exactly the target mental model: *the terminal has binaries; a binary runs inference (with its parameters) and produces realistic, fun output.*

### 2.3 Corpus, organized by binary
Char-level + ASCII keeps vocab ~100. The corpus is the **union of many example transcripts per supported binary**, so the model learns each command's distribution and to echo its arguments:
- **Synthetic generators (primary).** One small generator per binary emits many varied, realistic invocations + outputs: `ping` with different hosts/counts/latencies/loss; `reboot`/`shutdown` sequences; `git status/log`; `ls`/`cat` over a small invented filesystem; `curl` headers; `uname`/`whoami`/`neofetch`; fun ones (`fortune`, `cowsay`, `sl`). **Vary args** so the model learns to use them.
- **Recorded (seasoning).** Real `script(1)`/asciinema/`tldr`/man snippets add texture and irregularity so output isn't robotically templated.
- **Normalization.** Strip most ANSI for v1 (smaller vocab; the DOM adds styling). Keep the prompt string **exact and consistent** — it's the model's primary anchor and the universal stop-sequence.
- Even ~2–8 MB across the binary set is plenty for a nano/micro char model. **The binary set and its generators are the biggest levers on quality and fun** — the core of M4.

### 2.4 Coverage audit — and the data-vs-capacity lesson (corpus v8)

The corpus grew by **auditing coverage adversarially, not by adding whatever felt missing.** For v8 we ran an *exhaustive parallel coverage audit* — seven auditors, one per command category, feeding an adversarial consolidating critic — and let the gaps it surfaced drive the expansion (to ~35 MB). The result broke three behavioral ceilings that had stood for several corpus generations, all at the deployed **10.7M** size:

| Ceiling | Before | After v8 |
|---|---|---|
| multi-word content copy — `echo a b c > f; cat f` → `a b c` | 0% | **100%** |
| nested `cd` — `cd a; cd b; pwd` → deep path | 0% | **100%** |
| `touch` → empty file — `touch x; cat x` → empty | 0% | **100%** |

> **The lesson (the headline finding).** The multi-word ceiling was **data-coverage, not model capacity.** It *survived a 2.4× scale-up to 25M params unchanged* (three independent models — WebGPU-10.7M, MLX-10.7M, MLX-25M — failed it identically), then **fell to a data fix at 10.7M.** The copy circuit had learned to read back the *first* token because that's all the data ever forced; longer spans were permitted by the architecture but never demanded by the corpus. **The model learns what the data forces, not what it permits.** Scale is necessary but is rarely the cheap lever; coverage usually is.

The same audit caught two real generator **bugs** (the `rm -rf` failsafe was wrongly bound to *all* `rm -rf …` instead of the literal `/`; `cd` emitted a random wrong `errno`) — a reminder that **referential consistency is paramount: a *wrong* corpus addition (a self-contradiction) is worse than a missing one**, because the model will faithfully learn the contradiction. It also broadened breadth: `env`/`printenv`/`echo $VAR`/exit-codes/`alias`/`type`; `curl` response **bodies** (not just headers); `ip a` / `ip route` / `dig`; toolchain version banners (`node -v`, `git --version`, `python3 --version`); `git commit`; and a permission-denied persona family.

### 2.5 Model selector — the size sweep, live in the browser

The terminal **header** shows the running model's name/version, param count, and which inference engine won the load-time race (browser inference races a WebGPU session against the pure-TS CPU one and keeps the winner). Beside it, a retro **CRT-themed "channel" control** flips between a **four-size sweep** — **MICRO 2.7M**, **MINI 10.7M (default)**, **MAX 25M**, **ULTRA 57M** (wide → WebGPU wins). The default MINI is now the **hybrid-corpus v9**; MICRO/MAX/ULTRA remain corpus-v8 (retraining them on the hybrid corpus is an open follow-up).

> **Design choice — why ship three sizes instead of one.** The whole project is an argument that a *tiny* model can hold a computer's texture; letting a visitor turn a dial and watch coherence trade against latency in real time makes that argument visceral in a way no table can. Holding the corpus constant across the sweep isolates the one variable (size), and the control doubles as a live proof of **"the checkpoint is the contract"** (§15.1): three differently-sized `bity1` files, one loader, one corpus, zero code changes between them.

---

## 3. Design Principles

1. **Train offline, infer cheaply.** Training is a one-time developer cost in Node; inference is the hot, shipped, repeated path. They share kernels and the checkpoint format but are **separate code paths** (§14.1) so the browser bundle is tiny.
2. **One seam for compute.** If it touches a `Float32Array` element-by-element, it lives in a `Backend`, not in model code.
3. **Flat arrays, not nested.** Every tensor is one `Float32Array` + `shape`. No `number[][]`.
4. **Define-by-run autograd** for training; **cache-aware raw-kernel forward** for inference.
5. **Synchronous v1.** Pure-TS & WASM are sync; keep the core sync. WebGPU (async) is isolated to a future backend that awaits once per step (§17).
6. **Determinism by construction.** One seedable PRNG feeds init, dropout, batching, and sampling. Same seed ⇒ same result.
7. **Honest sizing.** Presets are chosen so each milestone is feasible on the backend that exists then, and so the deployed model is cheap to download and run (§9.4).
8. **The checkpoint is the contract.** The `bity1` file — config + tokenizer vocab + weights — is the *only* interface between training and inference. Train in **any** language or framework: if it emits a valid `bity1`, the browser loads it unchanged. This deliberately decouples the trainer from the runtime, and it's what lets a second, ~15× faster trainer (MLX, §15.1) coexist with the canonical from-scratch one without touching a line of the shipped inference path — and what lets the browser size-sweep (§2.5) be three files behind one loader.

---

## 4. Architecture Overview

```
            TRAINING (Node)                         INFERENCE (browser + Node)
┌──────────────────────────────────┐      ┌────────────────────────────────────────┐
│ examples/train-*.ts               │      │ web/  virtual-terminal demo             │
│ train.ts  (loop, AdamW, sched)    │      │ Shell + Binary registry (ping, reboot…) │  ← lean entry `bityllm/infer`
│ nn Modules + GPT (autograd)       │      │ InferenceSession (KV-cache, streaming)  │
│ core: Tensor + tape + ops + rng   │      │ forwardInfer (raw NdArray, no tape)     │
└───────────────┬──────────────────┘      └───────────────────┬────────────────────┘
                └──────────────┬───────────────────────────────┘
                     ┌─────────▼──────────┐   shared kernels + checkpoint format
                     │ Backend (CPU f32)  │   [later: wasm, webgpu]
                     └─────────┬──────────┘
                     ┌─────────▼──────────┐
                     │ Env (node|browser) │   files, checkpoints, threads, clock
                     └────────────────────┘
```

Dependency rule: **arrows point downward only.** Core calls `Backend`; `nn` calls core; the inference path calls `Backend` directly (skipping the tape). The core imports neither `fs` nor `window`. **Two package entrypoints:** `bityllm` (full, training) and `bityllm/infer` (inference + shell/binaries only — no autograd/optim/train/dataset, so it tree-shakes to a tiny bundle).

---

## 5. The Backend Seam

Owns **all** numeric kernels; knows nothing about autograd. Tensors are `{ data: Float32Array, shape: number[] }`, row-major, contiguous. Backward passes are expressed by the autograd layer *composing forward kernels*, so a backend only implements forward math.

```ts
export interface NdArray { data: Float32Array; shape: number[]; }

export interface Backend {
  readonly name: string;
  zeros(shape: number[]): NdArray;
  from(data: Float32Array, shape: number[]): NdArray;

  // the 90%: batched matmul (leading dims batch; last two are the matrix)
  matmul(a: NdArray, b: NdArray, opt?: { transposeA?: boolean; transposeB?: boolean }): NdArray;

  add(a: NdArray, b: NdArray): NdArray;      // broadcasting on trailing dims
  mul(a: NdArray, b: NdArray): NdArray;
  scale(a: NdArray, s: number): NdArray;
  addBias(a: NdArray, bias: NdArray): NdArray;

  gelu(a: NdArray): NdArray;
  geluBackward(a: NdArray, gOut: NdArray): NdArray;

  softmaxLastDim(a: NdArray): NdArray;                        // stable (row-max subtract)
  layerNorm(a: NdArray, w: NdArray, b: NdArray, eps: number): { y: NdArray; mean: NdArray; rstd: NdArray };

  gatherRows(table: NdArray, ids: Int32Array): NdArray;       // embeddings
  scatterAddRows(gradTable: NdArray, ids: Int32Array, g: NdArray): void;

  transposeLast2(a: NdArray): NdArray;
  reshape(a: NdArray, shape: number[]): NdArray;              // view when contiguous
  fill(a: NdArray, v: number): void;
  copy(a: NdArray): NdArray;
}
```

- **`matmul` is the whole ballgame** (~90% of FLOPs). Naive is a triple loop; §18 shows how the CPU backend gets several× faster while staying pure TS.
- **Fused ops** (`layerNorm`, `softmaxLastDim`, `gelu`) live here for stability + speed; a future backend can override them.
- **Sync in v1.** Same interface serves training and the inference `forwardInfer` (which additionally reads/writes the KV-cache).

Default backend is `CPUBackend`, overridable via `setBackend(...)`.

---

## 6. Tensor & Autograd (training path)

```ts
export class Tensor {
  data: Float32Array; shape: number[];
  grad: Float32Array | null; requiresGrad: boolean;
  private _parents: Tensor[]; private _backward: (() => void) | null; label?: string;
}
```

- **Define-by-run tape.** Ops append nodes while `gradEnabled`. `loss.backward()` seeds the scalar grad = 1, walks the tape in **reverse insertion order**, and each node's `_backward()` **accumulates** (`+=`) into its parents' grads — which is what makes **tied weights** (shared embedding/LM head) correct automatically.
- **`noGrad(fn)`** disables the tape (eval). Inference uses the separate `forwardInfer` path (§14) which never builds a tape at all.
- **Broadcasting.** Right-aligned in forward; in backward, gradients of broadcast inputs are reduced back with `sumTo(shape)` — the single discipline that prevents a whole family of shape bugs (grad-check enforces it).
- **Memory.** Training holds the activation graph until `backward()`: ≈ `O(nLayer·B·T·nEmbd)` + attention `O(nLayer·B·nHead·T²)`. Tiny for nano; bound larger presets with small `B`/`T` and gradient accumulation. Hot-loop allocations use a scratch arena to avoid GC pauses.

---

## 7. Differentiable Ops

Each op = forward (via `Backend`) + a `_backward` closure capturing only what it needs. All grad-checked.

| Op | Forward | Backward sketch |
|---|---|---|
| `add/mul (bcast)` | elementwise | `sumTo` each input's grad |
| `scale(a,s)` | s·a | `s·g` |
| `matmul(a,b)` | batched a·b | `g·bᵀ`, `aᵀ·g` (transpose flags, no copies) |
| `addBias(x,b)` | row bcast | `gx += g`, `gb += Σrows g` |
| `gelu` | tanh-approx GELU | `g ⊙ gelu'(x)` |
| `softmaxLastDim` | stable softmax | `s ⊙ (g − Σ(g⊙s))` |
| `layerNorm` | normalize + affine | vectorized LN grad from cached mean/rstd |
| `embedding` | gather rows | `scatterAdd` into `gW` at ids |
| `crossEntropyLogits` | logsumexp − z[y], mean | `(softmax(z) − onehot(y)) / N` (fused, stable) |
| `transpose/reshape/split/concat` | view/copy | rearrange g |
| `causalMaskAdd` | +(−∞) above diagonal | pass g through unmasked |
| `dropout(p)` | train: mask·1/(1−p) | `g ⊙ mask/(1−p)` |

Stability rules baked in: softmax subtracts row max; cross-entropy via logsumexp with fused `softmax − onehot` gradient; LayerNorm caches mean/rstd for an exact backward.

---

## 8. NN Modules

`Module` base with `forward`, recursive `parameters()`, and `train(mode)`. Modules: `Linear`, `Embedding`, `LayerNorm`, `Dropout`, `CausalSelfAttention` (multi-head), `MLP` (Linear→GELU→Linear), `Block` (pre-norm residual), `GPT`.

---

## 9. The GPT Model

GPT-2-style, **pre-norm**, learned absolute positional embeddings.

```
tokens ─▶ wte[token] ┐
pos    ─▶ wpe[pos]   ┴▶(+)▶ dropout ▶ [ Block × nLayer ] ▶ ln_f ▶ lm_head(=wteᵀ) ▶ logits
Block:  x = x + attn(ln1(x));  x = x + mlp(ln2(x))
Attn:   q,k,v = proj(x) → [B,nHead,T,hd];  softmax(qkᵀ/√hd + causalMask)·v → merge → out-proj
MLP:    Linear(nEmbd→4·nEmbd) → GELU → Linear(4·nEmbd→nEmbd)
```

### 9.1 Config
```ts
export interface GPTConfig {
  vocabSize: number; blockSize: number;   // context length T
  nLayer: number; nHead: number; nEmbd: number;
  dropout?: number;                        // 0 for tiny data; 0.1–0.2 to regularize
  bias?: boolean;                          // biases in Linear/LN (default true)
}
```

### 9.2 Tying & init
- **Weight tying**: `lm_head.weight` shares storage with `wte.weight` (`[vocab, nEmbd]`); `logits = x·wteᵀ`. Autograd `+=` makes the shared gradient correct.
- **Init**: weights `~N(0, 0.02)`; LN `weight=1, bias=0`; biases `0`.
- **Residual scaling (GPT-2 trick)**: scale attn/MLP **output-projection** weights by `1/√(2·nLayer)`.

### 9.3 Loss
Next-token cross-entropy, mean over all `B·T` positions (`y` = `x` shifted left one).

### 9.4 Presets & cost (chosen for cheap browser inference)
| Preset | nEmbd | nLayer | nHead | block | ~Params | int8 download | Train (pure-TS) |
|---|---|---|---|---|---|---|---|
| **nano**  | 64  | 3 | 4 | 128 | ~0.2M | ~0.2 MB | feasible (~1–5 s/step) |
| **micro** | 192 | 6 | 6 | 128 | ~2.7M | ~2.7 MB | slow on CPU; wants WASM/threads (M6) |
| **mini**  | 384 | 6 | 6 | 256 | ~11M  | ~11 MB | WebGPU/WASM only |

**Inference cost (batch-1, per new token)** — the number that matters for the browser terminal:

| | FLOPs/token (KV-cache) | FLOPs/token (no cache, T≈128) | naive JS ≈0.3 GFLOP/s | tuned JS ≈2 GFLOP/s |
|---|---|---|---|---|
| **nano**  | ~0.45 M | ~26 M | cache: **~670 tok/s** · no-cache: ~12 | cache: ~4400 · no-cache: ~77 |
| **micro** | ~6 M | ~690 M | cache: **~50 tok/s** · no-cache: ~0.4 | cache: ~330 · no-cache: ~3 |

Takeaways: **the KV-cache is what makes it cheap** — it turns per-token cost from `O(params·T)` into `O(params + T·nEmbd)`. **nano-with-cache is effortless** in naive pure TS; **micro needs the cache** to feel interactive (and pacing throttles output anyway). *Postscript:* the faster backends this section anticipated **did land** (GPU via Deno/WebGPU, then MLX, §15.1), so the deployed default ultimately moved *up* from nano to the **10.7M "v8"** model — the extra capacity buys crisp copy-from-context and referential consistency that nano can't hold — with an in-browser **size-sweep selector** spanning **2.7M → 10.7M → 25M** (§2.5).

> **Capacity note — necessary but rarely the cheap lever.** One model covers *all* the binaries' output styles; char-level, stylized, repetitive terminal output compresses well, so a small model holds a lot of texture. But the sharpest capacity lesson of the project was a *negative* one: the **multi-word content-copy ceiling survived a 2.4× scale-up (10.7M → 25M) unchanged, then fell to a corpus fix at 10.7M** (§2.4). Capacity is a floor, not the dial — once a behavior is *representable*, whether the model exhibits it is usually a question of **whether the data forced it**, not of parameter count. We size by measured per-behavior pass-rates (the `bench/eval.mjs` harness, 8 seeds/case), and reach for more parameters only after coverage is exhausted.

---

## 10. Optimizer, Schedule, Clipping (training only)

- **AdamW** (decoupled weight decay), `β=(0.9,0.95)`, `ε=1e-8`.
- **Param groups**: decay on 2-D weights only; none on biases/LayerNorm/embeddings.
- **Global-norm gradient clipping** (default 1.0).
- **LR schedule**: linear warmup → cosine decay to a floor.
- **SGD** as a minimal reference. `zeroGrad()` between steps.

---

## 11. Tokenizer

```ts
export interface Tokenizer { size: number; encode(s: string): Int32Array; decode(ids: ArrayLike<number>): string; }
```
**v1: `CharTokenizer`** — vocab = sorted unique chars of the corpus → `stoi`/`itos`. Zero-dep, trivially correct, ideal for terminal text. The vocab is stored in the checkpoint so inference reconstructs it exactly. Interface leaves room for byte-level BPE later.

---

## 12. Data Pipeline (training)

```ts
export class Dataset {
  constructor(tokenIds: Int32Array, opts: { blockSize; batchSize; valSplit?; seed? });
  getBatch(split: "train" | "val"): { x: Int32Array /*[B,T]*/; y: Int32Array /*[B,T]*/ };
}
```
Encode the corpus once into one `Int32Array`; hold out ~5% for validation (shuffled at *session* granularity so the split is representative, not a single tail-slice of one command); a batch = `B` random windows of `blockSize+1` (`x` = first `T`, `y` = shifted one), sampled via the seeded PRNG. **First smoke-test corpus: makemore-style names** (learns in seconds). **Real target: the per-binary terminal corpus** (§2.3) — union of synthetic generators over the supported command set, seasoned with recorded transcripts.

> **Overfitting — measured, not assumed.** Every eval reports the train/val gap. On the v8 run the final gap was **−0.0055** (val *below* train) — i.e. **zero overfitting** despite a small model on 35 MB. (The v9 hybrid corpus, being more capture-heavy and diverse, runs a small *positive* gap of **+0.024** — still healthy — because real captures are less repetitive than the synthetic-random v8 corpus.) This isn't luck: the corpus is dominated by **synthetic-random** examples (RNG-generated filenames and content, §2.3), so a specific string is almost never seen twice. Memorization is structurally unavailable, which *forces* the model onto the rules — copy-from-context, path bookkeeping, byte-accurate metadata — rather than a lookup table. The held-out split is a guardrail; the corpus design is why the guardrail rarely has to fire.

---

## 13. Training Loop (Node)

Per step: sample batch → optional grad-accum micro-steps → forward → CE loss → `backward()` → clip → `optimizer.step()` → schedule LR → `zeroGrad()`. Periodically: eval train/val loss under `noGrad`, sample a **terminal preview** (run a few binaries) via `onEval`, checkpoint (weights + optimizer state + step, for exact resume). Timing via `Env.now()`.

---

## 14. Inference (the shipped path)

The browser never runs the training machinery. Inference is a distinct, lean path sharing only the `Backend` kernels and the checkpoint format.

### 14.1 `InferenceSession` — raw-kernel forward + KV-cache
```ts
export class InferenceSession {
  static async load(env: Env, path: string): Promise<InferenceSession>;  // f32 or int8 checkpoint → NdArray weights + tokenizer
  reset(): void;                                    // clear KV-cache + context
  feed(text: string): void;                         // append + prefill KV-cache (prompt, prior transcript, command line)
  *stream(opts: GenOpts): Generator<string>;        // yield chars until a stop-sequence or maxNewTokens
  generate(prompt: string, opts: GenOpts): string;  // convenience wrapper
}
export interface GenOpts { maxNewTokens: number; temperature?: number; topK?: number; topP?: number; stop?: string[]; seed?: number; }
```
- **`forwardInfer`** calls `Backend` directly on plain `NdArray`s — **no `Tensor`, no tape, no `_backward`** — so `bityllm/infer` excludes all autograd/optim/train/dataset code and tree-shakes to a tiny bundle.
- **KV-cache**: per layer, cache K and V (`[nHead, T, hd]`). `feed()` prefills; each generated token appends one K/V column and attends over the cache → per-token cost independent of how much has already been generated (up to `blockSize`; oldest dropped when full). Cache memory ≈ `2·nLayer·T·nEmbd` floats (micro@T256 ≈ 2.4 MB).

### 14.2 Streaming & stop-sequences
- **Sampling** per token: crop context → `forwardInfer` → last-position logits → `/temperature` → optional **top-k** / **top-p** → softmax → categorical sample (seeded PRNG) → decode char → **yield**.
- **Stop-sequences**: maintain a small ring buffer of recent output; after each token, stop if the tail matches any `stop` string (e.g. the prompt `"guest@bity:~$ "`) or `maxNewTokens` is hit. This is what bounds one command's output.

### 14.3 Virtual shell: binaries over the session
A `Shell` owns one `InferenceSession` + a `Map<string, Binary>` registry. `shell.run(line, io)`:
1. Parse `line` → `argv`; look up `argv[0]`.
2. **Scripted / hybrid** binary → `await binary.run(argv, ctx)` (may itself call `ctx.session` and `ctx.io`).
3. **Model** binary → `session.feed(binary.buildPrompt(argv))`, then stream `session.stream({ ...binary.sampling, stop: binary.stop ?? [PROMPT], maxNewTokens })` into the renderer, **honoring `binary.pacing`** (per-line/char delays independent of inference speed).
4. **Unknown** → fallback binary (`command not found` or free-form).
5. Emit a fresh prompt; await next line.

The session's KV-cache **persists across commands**, so the running transcript is shared context (an earlier `cd`/`export` can nudge later output) up to `blockSize`; `reset()` clears it (e.g. `reboot`, `clear` optionally). Adding a command = registering a `Binary`: scripted ones need no retraining; model ones need corpus coverage (§2.3) to look right.

### 14.4 No KV-cache? Still works
Without a cache the app re-feeds the cropped context each token (`O(params·T)`) — fine for nano, sluggish for micro (§9.4). The cache is strongly recommended and part of v1's inference path.

---

## 15. Checkpoints & Quantization

Dependency-free, **safetensors-shaped** so interop stays possible:
```
[ uint32 headerLen ][ headerLen bytes UTF-8 JSON ][ raw little-endian tensor blobs ]
```
- **Header JSON**: `{ format, config, tokenizer:{type,vocab}, tensors:{ name:{ shape, dtype, offset, length, scale?, zeroPoint? } }, optimizerState?, step? }`.
- **Training checkpoints**: f32 weights + optimizer moments + step ⇒ exact resume.
- **Deployment checkpoints (browser)**: **int8-quantized weights** (per-row scale, optional zero-point) ⇒ ~4× smaller (nano ≈ 0.2 MB). **Dequantize once to f32 at load** — inference compute unchanged, only download shrinks. (int4 / dequant-in-matmul are future options.)
- Round-trip (f32 and int8) is a required test (§19).

### 15.1 The contract in practice: a second trainer (MLX)

The `bity1` format is not just a serialization detail — it's the **seam between "how the weights were made" and "how they're run"** (principle §3.8). Because the browser only ever sees `bity1`, the trainer is swappable. We have two:

- **The from-scratch WebGPU/WGSL trainer stays canonical and educational** — hand-written autograd and Metal-via-WebGPU kernels are the point of the project.
- **MLX (`train/mlx_train.py`, Python + Apple MLX/Metal) is an optional *speed path*.** It trains the **same architecture** with Apple's fused Metal kernels at **~46,500 tok/s on an M4 Pro vs ~3,000 for the from-scratch WebGPU trainer (~15×)** — a full 16k-step run drops from **~6 h to ~24 min**. It exists *because of* the contract, not in spite of it: it emits our exact `bity1`, so **inference is byte-for-byte unchanged.** The naive WGSL kernels, not the silicon, were the ceiling the speed path clears.

What the contract *demands* of any second trainer — and why the design holds:

- **Parity is the proof, not a hope.** MLX's forward is checked against the **independent** TypeScript forward on the same exported weights: **max Δlogit 1.2e-6, argmax identical.** Two implementations, written from scratch in two languages, agreeing to ~1e-6 is what certifies that the format/transpose/GELU-approx mapping is exact — and it's the guardrail against a fast trainer that silently drifts from the runtime.
- **Match the *decisions*, not just the shapes.** The subtle risk is optimizer semantics, not tensor layout. MLX AdamW decays every parameter by default; the TS trainer uses **decoupled 2-D-only weight decay** (embeddings, LayerNorm, and biases excluded). The MLX path replicates that grouping by hand (`weight_decay=0` globally + explicit decay on the block `Linear` weights). It matters: an early decay-*all* version cost points on the fuzziest behaviors — the 2-D-only grouping recovers them. **The checkpoint carries the weights; the trainer must carry the training decisions.**
- **Same guardrails.** A held-out 5% validation split with train/val gap reporting (§12) rides along, so the speed path can't buy throughput by quietly overfitting.

### 15.2 bf16 — a measured no-op on Apple Silicon (kept, off by default)

`--bf16` runs the **matmuls** in bf16 while **master weights, LayerNorm, softmax, loss, and the optimizer stay fp32**, and the **export stays fp32** — so inference and parity are untouched by design. Measured A/B on an M4 Pro: **+2–4% only** (noise), *not* the 1.5–2× bf16 gives on NVIDIA. The reasoning is the interesting part, and it's a silicon fact, not a bug:

- **Apple GPUs have no tensor cores** — fp32 and bf16 matmul run at similar rates, so there's little compute to reclaim.
- **The fp32 master defeats the bandwidth argument** — keeping fp32 master weights means casting fp32→bf16 *every forward*, and that traffic roughly cancels the bandwidth a bf16 matmul would have saved.

So bf16 is kept **opt-in and off by default**: it's safe (loss identical to three decimals) but pointless *here*, and would pay off on a CUDA backend. More broadly, **the fp32 MLX trainer is already near the M4 Pro's practical throughput ceiling** — **~46–52% MFU with the GPU pinned at its max clock (1578 MHz, ~22 W)**. The remaining MFU gap is *not* bf16-recoverable; it's the cost of a memory-bound, small-batch workload on a chip without matmul-specialized units. That's why the speed path stops at fused fp32 kernels rather than chasing mixed precision.

---

## 16. Isomorphic Env Adapters

```ts
export interface Env {
  readFile(path: string): Promise<Uint8Array>;
  writeFile(path: string, data: Uint8Array): Promise<void>;
  now(): number; hardwareThreads(): number;
}
```
- **`env.node.ts`**: `node:fs/promises`, `perf_hooks`, `os.cpus()`, `worker_threads` (future parallel matmul).
- **`env.browser.ts`**: `fetch` (load model), `Blob`/`IndexedDB` (persist), `performance.now()`, `navigator.hardwareConcurrency`, Web Workers. The terminal demo `fetch`es the int8 checkpoint from a static path.
- The core imports neither; the app wires in the right `Env`.

---

## 17. Sync/Async & the WebGPU Future

v1 core is **synchronous** — training loop, tape, kernels, and `forwardInfer` all return directly (pure-TS & WASM are sync). A future **WebGPU** backend is async; rather than infect the core, it isolates async at the **step boundary** (enqueue a whole forward as GPU commands, `await` once). The `Backend` interface is written so this is additive. Documented so we don't paint ourselves into a corner.

---

## 18. Determinism & the Pure-TS Performance Playbook

**Determinism**: library ships its own PRNG (mulberry32/xorshift128+): `random`, `randn` (Box–Muller), `randint`. Init, dropout, batching, **and sampling** draw from it ⇒ reproducible. Sampling seed is exposed in `GenOpts` so a terminal session can be replayed.

**Pure-TS speed (still zero-dep)**: flat `Float32Array`; matmul loop order `i-k-j`; pre-transpose/flag `B` for contiguous dot products; register/tile blocking; no hot-loop allocation; monomorphic JIT-friendly kernels; `forwardInfer` skips the tape. A careful GEMM reaches a few GFLOP/s single-thread. Beyond that = M6 backends (`worker_threads`/Web Workers over `SharedArrayBuffer`, then WASM SIMD, then WebGPU) — all behind the same seam.

---

## 19. Testing Strategy

Zero-dep via Node's built-in `node:test` + `node:assert`, run with native `.ts` execution.

1. **Gradient checking (the gate).** Analytic vs central finite differences per op. The `CPUBackend` is parameterized by array type so grad-checks run in **Float64** with tight tolerances (`rtol ≈ 1e-4`); the model stays Float32.
2. **Overfit-one-batch (integration gate).** Full GPT memorizes a single tiny batch, loss → ~0. Exercises forward + autograd + optimizer + tying end-to-end. **No real training until this passes.**
3. **Inference parity.** `forwardInfer` (with and without KV-cache) matches training `Module.forward` logits to tolerance on the same weights — ensures the two code paths agree.
4. **Determinism.** Same seed ⇒ identical loss curve and identical sampled text.
5. **Serialization round-trip.** f32 and int8 checkpoints reload to identical/tolerance-equal logits.
6. **Shape/broadcast**, **tokenizer round-trip**, and **binary smoke tests** (each model binary produces non-degenerate, stop-terminated output on a trained checkpoint).

---

## 20. Project Structure

```
bityllm/
  package.json    # "type":"module"; exports "." (full), "./infer" + "./infer/binaries" (lean); devDep: typescript
  tsconfig.json   # strict; ESNext; allowImportingTsExtensions + rewriteRelativeImportExtensions
  DESIGN.md
  src/
    backend/  backend.ts  cpu.ts  index.ts
    core/     tensor.ts  ops.ts  rng.ts  shape.ts
    nn/       module.ts  linear.ts  embedding.ts  layernorm.ts  dropout.ts  attention.ts  mlp.ts  block.ts  gpt.ts
    optim/    adamw.ts  sgd.ts  clip.ts  schedule.ts
    tokenizer/ tokenizer.ts  char.ts
    data/     dataset.ts
    infer/    session.ts  forward-infer.ts  kv-cache.ts  sampler.ts          # the lean, shipped engine
              shell.ts  binary.ts  binaries/ ( ping.ts reboot.ts fortune.ts sl.ts clear.ts help.ts … )
    io/       checkpoint.ts  quantize.ts  env.ts  env.node.ts  env.browser.ts
    train.ts  index.ts  infer.ts                                             # infer.ts = `bityllm/infer` entry
  examples/
    train-names.ts  train-terminal.ts  generate.ts
    web/  index.html  terminal.ts     # loads dist/infer + int8 checkpoint; DOM terminal; typewriter + pacing
  test/  gradcheck  overfit  infer-parity  determinism  serialize  tokenizer  shapes  binaries  (.test.ts)
  corpus/  generators/ ( one per binary → transcripts )  build.ts    models/ ( exported int8 checkpoints )
```
**Tooling (near-zero dep):** Node 26 runs `.ts` directly and provides the test runner. `tsc --noEmit` type-checks; `tsc` emits `dist/` ESM for the browser (native ES modules — **no bundler needed**); the `./infer` export keeps the browser bundle free of training code. Relative imports use `.ts` + `rewriteRelativeImportExtensions`.

---

## 21. Milestones (each a verifiable vertical slice)

| # | Deliverable | Done when |
|---|---|---|
| **M0** | Scaffold: package.json (dual exports), tsconfig, test runner | `tsc --noEmit` clean; trivial test passes |
| **M1** | `Backend`+`CPUBackend` + `Tensor`/autograd + ops | **all grad-checks pass** (Float64) |
| **M2** | `nn` + `GPT` + AdamW | **overfit-one-batch** loss → ~0 |
| **M3** | Tokenizer + Dataset + training loop + checkpoints | trains **nano** on **names**; val loss falls; samples |
| **M4** | **Binary set** + per-binary synthetic corpus generators + train nano | each target binary (`ping`, `reboot`, …) streams plausible, fun output in Node |
| **M5** | `InferenceSession` (KV-cache, streaming, stop-seq) + **`Shell`/`Binary` registry** + int8 export + **browser terminal demo** | run `ping bity.dev`, `reboot`, `fortune` in-page → streamed, paced output; tiny bundle, sub-MB model — **headline deliverable** |
| **M6** (future) | Faster backends behind the seam | worker+WASM SIMD GEMM → micro practical; WebGPU; BPE/RoPE; GPT-2 import; int4 |

Build order note: M0→M2 is unchanged (training must be correct first); M4–M5 turn a correct trainer into the shipped virtual terminal.

---

## 22. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Wrong gradients | Silent garbage training | Grad-check gate (M1) + overfit gate (M2) |
| Inference path diverges from train path | Browser output ≠ trained behavior | **Infer-parity test** (§19.3) |
| Command conditioning is fuzzy (args ignored / styles bleed) | Wrong-looking binary output | Strict, consistent prompt format; per-binary corpus coverage; optional control tokens (§24); binary smoke tests |
| Binary set exceeds nano capacity | Blurry per-command output | Measure per-binary quality (M4); bump to micro or split models behind `Binary` |
| Corpus too repetitive/thin | Boring or degenerate output | Mix synthetic + recorded; vary args; dedupe; track val loss & sample diversity |
| Download too large | Slow first load | int8 quantized deployment checkpoint (§15); default nano |
| Latency grows with scrollback | Terminal feels laggy over time | **KV-cache** (§14.1) → per-token cost flat |
| Pure-TS too slow for micro | Can't train bigger models on CPU | Ship nano; presets sized to backend; M6 WASM/threads/WebGPU |
| Numeric instability | NaNs/bad loss | Stable softmax/CE/LN; fused CE |
| GC pauses | Jittery steps | Flat arrays, scratch pool, monomorphic kernels |
| Async WebGPU vs sync core | Future rewrite risk | Keep v1 sync; isolate async at step boundary (§17) |

---

## 23. Public API (concrete targets)

**Train (Node):**
```ts
import { GPT, CharTokenizer, Dataset, train, exportInt8, nodeEnv } from "bityllm";
const text  = new TextDecoder().decode(await nodeEnv.readFile("corpus/terminal.txt"));
const tok   = CharTokenizer.fromText(text);
const model = new GPT({ vocabSize: tok.size, blockSize: 128, nLayer: 3, nHead: 4, nEmbd: 64 }); // nano
const data  = new Dataset(tok.encode(text), { blockSize: 128, batchSize: 16, valSplit: 0.1, seed: 1337 });
await train(model, data, { steps: 5000, lr: 3e-4, warmup: 100, minLrRatio: 0.1, weightDecay: 0.1, clip: 1.0,
  evalEvery: 200, evalIters: 20, seed: 1337 });
await exportInt8(model, tok, nodeEnv, "models/terminal.bity");   // sub-MB, browser-ready
```

**A model binary is just a manifest:**
```ts
export const ping: Binary = {
  name: "ping", kind: "model",
  synopsis: "ping [-c count] <host>",
  buildPrompt: (argv) => `guest@bity:~$ ${["ping", ...argv.slice(1)].join(" ")}\n`,
  sampling: { temperature: 0.7, topK: 40 },
  stop: ["guest@bity:~$ "], maxNewTokens: 512,
  pacing: { perLineDelayMs: 1000 },      // one reply per second — the "fun"
};
```

**Run (browser virtual terminal):**
```ts
import { InferenceSession, Shell } from "bityllm/infer";
import { ping, reboot, fortune, clear, help } from "bityllm/infer/binaries";

const sess  = await InferenceSession.load(browserEnv, "/models/terminal.bity");
const shell = new Shell(sess, { prompt: "guest@bity:~$ " });
shell.register(ping, reboot, fortune, clear, help);      // model + scripted binaries

// on Enter:
await shell.run(line, { write: (s) => term.write(s),     // typewriter render, honoring binary.pacing
                        delay: (ms) => sleep(ms), clear: () => term.clear() });
```

---

## 24. Open Questions (non-blocking for M0–M3)

1. **Binary set & personas** — which commands to ship (`ping`, `reboot`, `traceroute`, `neofetch`, `fortune`, `cowsay`, `sl`, `curl`, `git`, `uname`, `sudo`, …), each one's args, sampling, stop, and **pacing**; the prompt string / host persona; how much ANSI to keep. The main M4 decision and the biggest lever on fun.
2. **One model vs per-command models** — default one shared prompt-conditioned model; per-binary models remain possible behind the `Binary` interface (§2.1).
3. **Control tokens** — char-level prompt-conditioning may suffice; if command conditioning is fuzzy, add per-command control tokens or a strict `\x00cmd\x00`-style header to sharpen it.
4. **RoPE vs learned positions** — v1 learned absolute (simplest); RoPE generalizes past `blockSize`, a clean later swap.
5. **BPE** — only if char-level proves limiting; interface already allows it.
6. **int4 / dequant-in-matmul** — if nano int8 download or memory ever matters more.
7. **Which WASM path (M6)** — hand-written WAT vs tiny AssemblyScript/Zig/C → single `.wasm`, dependency-free either way.
8. **GPT-2 import** — checkpoint is safetensors-shaped to keep this open (needs BPE + GELU/LN parity).

---

*End of draft v4. Intended order: M0 → M1 (grad-checks) → M2 (overfit) → M3 (train nano on names) → M4 (binary set + per-binary corpus) → M5 (browser virtual terminal — the headline). Everything below the `Backend` seam, and the whole `infer`/shell path's speed, is drop-in optimization, not redesign.*
