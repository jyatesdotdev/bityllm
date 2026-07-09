# AGENTS.md — orientation for AI coding agents (read this first)

**What this is:** bityllm is a tiny char-level GPT (~10.7M params, pure TypeScript,
**zero runtime dependencies**) trained from scratch that powers a browser
"hallucinated terminal" — every command's output is *dreamed* by the model,
there is no real shell. Live: https://jyatesdotdev.github.io/bityllm/

Companion docs: **[DESIGN.md](DESIGN.md)** (the *why*), **[RUNBOOK.md](RUNBOOK.md)**
(the *how* — most current source of truth; cross-check it before quoting numbers),
**[JOURNEY.md](JOURNEY.md)** (the narrative). When in doubt, RUNBOOK wins.

---

## Repo map

| Path | What lives there |
|---|---|
| `src/core/` | `tensor.ts` (define-by-run autograd tape), `ops.ts` (differentiable ops), `rng.ts` (seeded RNG) |
| `src/nn/` | `gpt.ts` (the model), `attention.ts`, `layers.ts` (LayerNorm/Linear/GELU), `module.ts` |
| `src/backend/` | backend seam: `backend.ts` (interface), `cpu.ts` (blocked-kernel CPU), `index.ts` |
| `src/gpu/` | WebGPU trainer + inference: `wgsl.ts`, `wgsl-fused.ts`, `wgsl-infer.ts`, `trainer.ts`, `session.ts` |
| `src/optim/` | `adamw.ts` (decoupled weight decay, 2-D weights only) |
| `src/tokenizer/` | `char.ts` (char-level vocab, travels inside the checkpoint) |
| `src/io/` | `checkpoint.ts` — the `bity1` format: `serialize`/`deserialize`, f32 + per-row int8 |
| `src/data/` | `dataset.ts` (training data pipeline, session-granularity shuffle) |
| `src/infer/` | shipped inference: `session.ts`, `sampler.ts`, `shell.ts`, `binaries.ts` (per-command manifests, the `reboot`/`sudo` theater) |
| `corpus/generators/` | synthetic generators: `fs.mjs` (nested-path fs + metadata), `sys.mjs`, `net.mjs`, `git.mjs`, `fun.mjs`, `copy.mjs`, `lib.mjs`, `index.mjs` |
| `corpus/capture/` | real-data capture: Debian container (`run.mjs`), Lima/QEMU VM (`vm/`), dmesg ingest |
| `corpus/build.mjs` | assembles `corpus/data/bity.corpus.txt` (~68% synthetic + ~32% real) |
| `corpus/COVERAGE_SPEC.md` | the coverage audit spec (drove corpus v8) |
| `corpus/data/` | committed captures (`*.jsonl`, `*.corpus.txt`); built `bity.corpus.txt` is gitignored |
| `train/mlx_train.py` | **optional** Apple MLX/Metal fast-path trainer (Python; lives in `.venv`) |
| `examples/` | CLI entry points: `train-terminal.ts` (CPU), `train-terminal-gpu.ts` (WebGPU/Deno), `generate.ts`, `export-int8.ts` |
| `examples/web/` | the browser terminal: `terminal.ts` (DOM glue + WebGPU/CPU engine race + **MODEL selector** that hot-swaps size variants in place), `index.html` (green-phosphor CRT + knob panel), `serve.mjs`, built `dist/` (gitignored) |
| `bench/` | `eval.mjs` (scored behavior eval), `gpu-parity.ts`, `gpu-infer.ts`, `gpu-bisect.ts`, `webgpu-gemm.ts` |
| `test/` | `node:test` suite: gradcheck, model, infer, kernel, parallel, pipeline, interactive |
| `docs/` | GitHub Pages deploy target: `index.html`, `dist/`, `terminal.int8.bity` (committed) |
| `models/` | model zoo; f32 `*.bity` gitignored, **only `terminal.int8.bity` (deployed v8) is committed** |

---

## Core invariants (do NOT break these)

1. **The checkpoint is the contract.** `bity1` = `[u32 headerLen][JSON header][raw f32/i8 blobs]`;
   config + tokenizer vocab travel *inside* the header. Train in any language/framework;
   the browser loads the same `bity1` unchanged. Any change to `src/io/checkpoint.ts`
   or the header schema must keep old checkpoints loadable and stay parity-verified.

2. **Corpus referential consistency is paramount.** A *wrong/contradictory* corpus
   addition (e.g. `cat` returning content a prior `rm` deleted, or an inconsistent IP
   for a host) is **worse than a missing one** — it teaches the model a falsehood.
   Use random filenames/content (memorization impossible) and append-order listings.
   Behavioral ceilings here are almost always **data coverage, not capacity**
   ("the model learns what the data forces, not what it permits" — multi-word copy
   survived a 2.4x scale-up to 25M then fell to a *data* fix at 10.7M).

3. **Order of trust: grad-check → overfit-one-batch → parity** (DESIGN §19).
   If a grad-check fails, nothing downstream is meaningful; then confirm the model
   can overfit a single batch; only then trust GPU/inference parity. Never skip
   upstream when something downstream looks off.

4. **New-trainer fidelity rules** (or exported weights won't evaluate right in the TS engine):
   - **tanh-approx GELU** — `0.5*x*(1+tanh(√(2/π)*(x+0.044715*x³)))`; CPU + WebGPU inference hardcode it.
   - **tied lm_head** — logits `= x @ wte.T`; do not add a separate output matrix.
   - **LayerNorm eps = 1e-5.**
   - **Weight transpose on export** — MLX/PyTorch `nn.Linear` stores `[out,in]`;
     our format is `[in,out]` → transpose every Linear weight on export
     (see `train/mlx_train.py` `add(..., .T, ...)`). Reuse the canonical vocab
     via `--vocab-from`. Prove it with the parity gate below (target ≤ ~1e-6 Δlogit,
     identical argmax) before trusting a new trainer.

---

## Build / test / train / eval / export / deploy (exact commands)

```bash
npm install                        # types only — zero runtime deps
npm run typecheck                  # tsc --noEmit
npm test                           # node:test: gradchecks(f64), overfit gate, infer parity, int8 round-trip
npm run corpus                     # rebuild corpus/data/bity.corpus.txt (offline, deterministic)

# Trust the GPU stack (Deno is the only desktop WebGPU runtime):
deno run --allow-read bench/gpu-parity.ts        # GPU vs CPU loss/grads/5-step
deno run --allow-read bench/webgpu-gemm.ts       # GEMM correctness + GFLOP/s

# Train — CPU (pure TS, data-parallel workers):
npm run train -- --steps 12000 --batch 16 --block 128 --lr 1e-3 \
  --layers 6 --heads 4 --dim 128 --workers 8 --out models/terminal-milli.bity
# Train — GPU (Deno/WebGPU→Metal), recommended micro+:
deno run --allow-read --allow-write examples/train-terminal-gpu.ts \
  --steps 12000 --batch 32 --block 128 --lr 1e-3 \
  --layers 6 --heads 6 --dim 192 --out models/terminal-micro.bity
# Train — MLX fast path (~15x faster, Apple Silicon; emits the same bity1):
uv venv .venv && uv pip install --python .venv/bin/python mlx numpy      # one-time
.venv/bin/python train/mlx_train.py --steps 16000 --batch 32 --block 128 \
  --lr 6e-4 --layers 6 --heads 6 --dim 384 --out models/terminal-mlx.bity
# MLX parity gate (proves format/transpose/GELU mapping):
.venv/bin/python train/mlx_train.py --parity /tmp/p.json --out /tmp/p.bity   # then Node forward

# Evaluate behavior (8 seeds/case, grep-scored pass-rates):
node bench/eval.mjs models/terminal-v8.bity

# Generate from a checkpoint:
npm run generate -- --cmd "ping -c 3 bity.dev" --temp 0.7 --seed 42

# Export int8 (~26% of f32) + serve the browser terminal:
node examples/export-int8.ts --in models/terminal-mlx.bity --out models/terminal.int8.bity
npm run web                        # tsc -p tsconfig.web.json + serve → localhost:8143/examples/web/
# Deploy to GitHub Pages (docs/):
npm run pages
```

- **Model swap needs no rebuild:** the page fetches `terminal.int8.bity` (the default,
  MINI) — replace that file to swap the default brain. The demo also has an in-browser
  **MODEL knob** (`terminal.ts` `MODELS`/`switchModel`) that lazy-fetches size variants
  (`terminal-micro-v8` 2.7M, `terminal-25m-v8` 25M, `terminal-ultra-v8` 57M `.int8.bity`)
  and swaps `Shell.session` in place — those files must sit beside `terminal.int8.bity`
  (in `docs/` when deployed) for the knob to switch. The 57M ULTRA is *wide* on purpose
  (8L/12H/768d) so WebGPU beats CPU in-browser (see the GPU gotcha), but it overfits this
  corpus — MINI/MAX generalize better. **Code** changes need `npx tsc -p tsconfig.web.json`.

---

## Key gotchas (bite people repeatedly)

- **The `rtk` shell hook masks/summarizes output** on this machine and can report
  "No errors found" over a *failing* command. When output looks inconsistent with
  exit codes, re-run via **`rtk proxy <cmd>`** for raw output.
- **Verify the artifact, not the build log.** `tsconfig.web.json` extends the parent;
  a stale/misconfigured `exclude` once let tsc "succeed" emitting nothing while old
  `dist/` kept serving. After a web build, `grep <new-symbol> examples/web/dist/...`
  to confirm the change actually landed. (Browsers also cache ES modules — hard-refresh Cmd-Shift-R.)
- **`pgrep -f <cmd>` matches the zsh wrapper shell too**, not just the trainer.
  When pausing a run, `kill -STOP` the *child* (deno/node) PID and confirm with
  `ps -o stat=` that the trainer shows state `T` — stopping the wrapper does nothing.
- **Metal fast-math `tanh(u)` NaNs for |u| > 44.36.** Any new WGSL `tanh`/`exp`
  composition needs a range guard (`clamp(u, -15, 15)`); the clip kernel must zero
  grads on non-finite norms or one inf cascades to a fully-NaN model in ~2 steps.
- **MLX lives in the gitignored `.venv`** (`train/mlx_train.py`); it is the *speed
  path*, not the canonical trainer. The from-scratch WebGPU/WGSL trainer remains the
  canonical/educational one. `--bf16` is a measured **no-op on Apple Silicon** (kept opt-in, off by default).
- **In the browser, CPU (pure-TS) beats WebGPU — that's expected, not a broken GPU
  path.** Interactive generation is batch-1, tiny-model, *serial* token-by-token →
  latency-bound (GEMV + per-dispatch/readback overhead the CPU doesn't pay). The
  load-time engine race (`terminal.ts` `loadModel`) legitimately picks CPU; inference
  parity is verified, so tokens are identical. It warms up both engines first (so the
  GPU isn't charged for one-time shader compilation) and shows both rates. WebGPU only
  pulls ahead on much bigger models, *batched* inference, or the parallel prefill phase.
