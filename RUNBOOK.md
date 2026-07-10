# bityllm Runbook

Everything needed to reproduce the project from a cold start — corpus capture →
corpus build → training (CPU/GPU) → export → browser terminal — plus every
gotcha we hit along the way. Companion to `DESIGN.md` (the *why*; this is the *how*).

---

## 0. Prerequisites

| Tool | Used for | Install |
|---|---|---|
| **Node ≥ 26** | everything (runs `.ts` natively, `node:test`) | nodejs.org / nvm |
| **npm** | dev deps (`typescript`, `@types/node`, `@webgpu/types` — types only, zero runtime deps) | with Node |
| **Deno ≥ 2.x** | GPU training (only runtime with WebGPU on desktop) | `brew install deno` |
| Docker (or colima) | corpus capture: Debian container | `brew install colima docker && colima start` |
| Lima + QEMU | corpus capture: real Debian VM (boot logs, real reboot) | `brew install lima qemu` |

**Offline note:** once `node_modules/`, the Deno binary, the Docker image
(`debian:stable-slim` + apt packages), and the Lima VM image are on disk,
*everything below runs offline*. The corpus data in `corpus/data/` and the
checkpoints in `models/` are already-built artifacts — with them, you can skip
straight to §4 or §6.

```bash
npm install        # one-time; installs type packages only
```

---

## 1. Corpus capture (optional — outputs are committed in corpus/data/)

### 1a. Debian container bulk capture (~4 MB real terminal output)
```bash
npm run capture                    # ≈ 4 MB → corpus/data/debian.{jsonl,corpus.txt}
node corpus/capture/run.mjs --mb 8 --fresh    # bigger / rebuild container
npm run capture:clean              # remove the container
```
- Harvests man pages, `--help`, package metadata, /etc, /usr/share/doc, ls/cat,
  deliberate errors, cowsay/fortune from a throwaway container (hostname `bity`,
  user `guest`). Sanitizes ANSI, MACs, identity. See `corpus/capture/README.md`.
- **Gotcha:** slim images strip man pages via `/etc/dpkg/dpkg.cfg.d/docker`;
  the harness removes that file and reinstalls base packages.
- **Gotcha:** `/usr/games` isn't on PATH — harness adds it (cowsay/fortune).

### 1b. Real Debian VM (boot dmesg, journalctl, a real captured reboot)
```bash
limactl start --vm-type=qemu --name bity-vm --tty=false template://debian-13
node corpus/capture/vm/run.mjs     # → corpus/data/debian-vm.{jsonl,corpus.txt}
limactl delete -f bity-vm          # cleanup
```
- Captures dmesg/journalctl/systemd state, then **reboots the VM** and scrapes
  the serial console (`~/.lima/bity-vm/serial.log`) for the full shutdown →
  kernel boot → login sequence.
- **Gotcha:** read the serial log as **UTF-8**, not latin1 — systemd truncates
  console lines with a multi-byte ellipsis (`…`); latin1 shreds it.
- **Gotcha:** scrub the host user/hostname (Lima uses your macOS username) and
  drop `lima-*` lines — handled by `vmScrub` in `corpus/capture/lib.mjs`.

### 1c. Your own dmesg files (optional seasoning)
```bash
# drop dmesg.*.txt in the repo root, then:
node corpus/capture/ingest-dmesg.mjs   # → corpus/data/real-dmesg.{jsonl,corpus.txt}
```
Scrubs MACs/UUIDs/hostnames. Don't feed logs containing secrets — the model
memorizes and the checkpoint ships to a browser.

## 2. Corpus build (required; runs offline from corpus/data/)

```bash
npm run corpus     # = node corpus/build.mjs  → corpus/data/bity.corpus.txt
# hybrid corpus is ~28.5 MB; `node corpus/build.mjs --synth-mb 9` sets the synthetic target
```
- **Hybrid corpus (current):** ~68% **real** captures / ~32% synthetic, **shuffled at
  session granularity** so the val split is representative. Since the programmatic CORE
  owns deterministic commands, the corpus trains **only the dreamed set**:
  - synthetic generators (`corpus/generators/`): `net`, `git`, `fun`, `sysinfo`
    (df/free/ps/top/lscpu/…), and `unknown` (graceful `command not found`). The old
    `fs`/`sys`/`copy`/`fs-session` generators are **retired** (unimported).
  - `build.mjs` **filters CORE-command records out of the real capture** — the model
    never dreams `ls`/`cat`/`pwd`, so training on them is pure dilution.
- **Gotcha (2026-07-09):** a graceful-`command not found` generator will POISON any
  real-but-thinly-captured command unless its positive signal roughly matches the
  not-found volume. `ps`/`df` (~8–10 capture records each) started returning
  `command not found` until `sysinfoGen` restored their positive signal and the
  `unknown` weight was halved. Whenever you drop a generator, check the capture
  actually covers those commands, or the fallback wins.
- Key lesson (corpus v3, still true for what the model dreams): teach copy-from-context
  with **random names** (memorization impossible) and **append-order listings** —
  mirror how `ping <host>` echoes its argument at a fixed template position.

---

## 3. Verify the stack (always safe to run; ~5 s)

```bash
npm run typecheck                  # tsc --noEmit
npm test                           # 26 tests: grad-checks (f64), overfit gate,
                                   # infer parity, int8 round-trip, shell holdback…
deno run --allow-read bench/gpu-parity.ts    # GPU vs CPU: loss/grads/5-step parity
deno run --allow-read bench/webgpu-gemm.ts   # GPU GEMM correctness + GFLOP/s
```
Order of trust (DESIGN §19): grad-checks → overfit-one-batch → parity. If any
of these fail after a change, nothing downstream is meaningful.

---

## 4. Training

### Presets (M4 Pro reference numbers)

| Preset | Flags | Params | Engine | tok/s | 12k steps |
|---|---|---|---|---|---|
| nano  | `--layers 3 --heads 4 --dim 64`  | 165k | CPU ×8 workers | ~22k | ~20 min |
| milli | `--layers 6 --heads 4 --dim 128` | 1.2M | CPU ×8 workers | ~3.3k | ~2.1 h |
| micro | `--layers 6 --heads 6 --dim 192` | 2.7M | **GPU** | ~9.8k | **~85 min** |

### CPU (Node, pure TS, data-parallel workers)
```bash
npm run train -- --steps 12000 --batch 16 --block 128 --lr 1e-3 \
  --layers 6 --heads 4 --dim 128 --workers 8 --out models/terminal-milli.bity
```
`--workers 0` = single-thread reference. Batch is the *global* batch, split
across workers; grads averaged each step (SharedArrayBuffer + Atomics).

### GPU (Deno, WebGPU → Metal) — recommended at micro+
```bash
deno run --allow-read --allow-write examples/train-terminal-gpu.ts \
  --steps 12000 --batch 32 --block 128 --lr 1e-3 \
  --layers 6 --heads 6 --dim 192 --out models/terminal-micro.bity
```
- Checkpoints (`bity1` format, loadable everywhere) are written at every eval
  (every `steps/10`), so you can kill/generate mid-run.
- Per-step NaN **tripwire** built in; add `--scanFrom N` to dump per-tensor
  diagnostics every step ≥ N (this is how the tanh bug was caught).
- **Gotcha (fixed, keep in mind for new kernels):** Metal fast-math
  `tanh(u)` NaNs for |u| > 44.36 → any WGSL `tanh`/`exp` composition needs
  range guards (`clamp(u, -15, 15)`); the clip kernel must zero grads on
  non-finite norms or one inf cascades to a fully-NaN model in 2 steps.

### MLX (Apple Silicon fast path) — ~15× faster than our WebGPU trainer
The from-scratch WebGPU trainer is the canonical/educational one; MLX is an
optional backend that trains the SAME architecture with Apple's fused Metal
kernels. "The checkpoint is the contract" — it emits our exact `bity1` format,
so browser inference is unchanged.
```bash
uv venv .venv && uv pip install --python .venv/bin/python mlx numpy   # one-time
.venv/bin/python train/mlx_train.py --steps 16000 --batch 32 --block 128 \
  --lr 6e-4 --layers 6 --heads 6 --dim 384 --out models/terminal-mlx.bity
```
- **~46,500 tok/s on an M4 Pro** (vs ~3,000 for WebGPU) → the 6 h mini run
  becomes **~24 min**. The naive WGSL kernels, not the silicon, were the ceiling.
- **Fidelity (or exported weights won't evaluate right in the TS engine):**
  tanh-approx GELU, tied LM head, LayerNorm eps 1e-5, and MLX `nn.Linear`
  stores `[out,in]` → transpose to our `[in,out]` on export. Reuses the
  canonical vocab from an existing checkpoint (`--vocab-from`).
- **Parity gate:** `python train/mlx_train.py --parity /tmp/p.json --out /tmp/p.bity`
  then a Node forward on the same weights — verified **max Δlogit 1.2e-6**,
  argmax agrees. That's what proves the format/transpose/GELU mapping.
- Weight decay: MLX AdamW decays every param, but the TS trainer (and this
  script) decay only the 2-D matmul weights. We set MLX `weight_decay=0` and
  apply decoupled decay by hand to the block Linear weights (`decay_linears`).
  An early version using MLX's decay-all cost a few points on the fuzziest cases
  (cat-of-uncreated `.csv` 75%→0%, `mv`→ls) — the 2-D-only grouping recovers them.

### Scale finding (10.7M → 25M, `8L/8H/512d`, MLX, 48 min)
Ceiling experiment holding corpus + recipe constant, scaling only params
(`bench/eval.mjs`, 8 seeds/case). **Scale broke some ceilings but not others:**
- **Won:** `wc -l` line-count 0%→**100%**, `mv`→ls →**100%**, all fuzzy cases
  solid → 25M is the best-behaved model yet (val loss 0.359).
- **Held at 0% even at 25M:** **multi-word content copy** (`echo a b c > f; cat f`
  returns only "a"). Three independent models (WebGPU-10.7M, MLX-10.7M, MLX-25M)
  fail it identically → **NOT pure capacity** as first assumed; the copy circuit
  reads back the *first* token reliably but was never forced to copy a longer
  span. Next lever is **corpus** (heavier/longer multi-word write→read drills),
  not just parameters.
- nested `cd`, touch→empty stay 0% — pure **corpus gaps** (never taught), not
  capacity; scale can't add what the data omits.

**Sequel — corpus v8 broke all three (the fix).** An exhaustive coverage audit
(see the model zoo §7 + `corpus/COVERAGE_SPEC.md`) added dense multi-word
write→read round-trips, a nested-`cd` path stack, and touch→empty drills. At
**10.7M** params, multi-word content copy, nested `cd`, and touch→empty all went
**0% → 100%** — confirming coverage, not capacity. The "held at 0%" above is what
v8 subsequently overturned. (Those filesystem behaviors are now **real code** since
the v9 hybrid pivot — the deployed default is **MINI v9**; see §7 and the DESIGN v5 note.)

### bf16 mixed precision (`--bf16`) — measured no-op on Apple Silicon
`--bf16` runs the matmuls in bf16; master weights, LayerNorm, softmax, loss, and
the optimizer stay fp32, and the export stays fp32 (inference + parity unaffected).
Measured A/B on M4 Pro: **+2–4% only** (25M 22.3k→22.7k tok/s; 10.7M 45.0k→46.9k) —
noise, not the 1.5–2× bf16 gives on NVIDIA. Why: Apple GPUs have no tensor cores
(fp32 and bf16 matmul run at similar rates), and keeping an fp32 master means
casting fp32→bf16 every forward — that traffic cancels the bandwidth saved. Kept as
opt-in; fp32 is the default. Loss identical to 3 decimals, so it's safe, just
pointless here (would pay off on a CUDA backend). The ~46–52% MFU gap on this M4 Pro
is **not** bf16-recoverable — fp32 is already near the practical throughput ceiling.

### Pause / resume (battery etc.)
```bash
pgrep -fl 'train-terminal'    # ⚠ lists BOTH the zsh wrapper AND the real process
kill -STOP <deno/node PID>    # stop the CHILD (deno/node), not the wrapper!
kill -CONT <both PIDs>        # resume
```
**Gotcha:** `pgrep -f <cmd> | head -1` usually returns the *wrapper shell*
(its command line contains the same string). Stopping it does nothing —
verify with `ps -o stat=` that the actual trainer shows state `T`.

### Debugging a bad run (the recipe that found the tanh bug)
1. Tripwire fires → note the step. Batches are seeded → **deterministic replay**.
2. Re-run with `--scanFrom <step-10>`: which tensor corrupts first, weights or grads?
3. `deno run --allow-read bench/gpu-bisect.ts` (edit step count): replays the
   trajectory, runs the poison step **forward-only**, scans activations in
   dataflow order — the first non-finite buffer names the kernel.

---

## 5. Evaluate / generate from a checkpoint

```bash
npm run generate -- --cmd "ping -c 3 bity.dev" --temp 0.7 --seed 42
node examples/generate.ts --ckpt models/terminal-micro-v2.bity --cmd "cowsay moo"
```

## 6. Export + browser terminal

```bash
node examples/export-int8.ts --in models/terminal-micro-v2.bity \
  --out models/terminal.int8.bity          # ~26% of f32 size (per-row int8)
npm run web                                 # build bundle + serve
# → http://localhost:8143/examples/web/
```
- The page fetches `/models/terminal.int8.bity`; swapping that file swaps the
  brain — no rebuild needed for model changes.
- Rebuild IS needed for code changes: `npx tsc -p tsconfig.web.json`.
- **Gotcha:** `tsconfig.web.json` *extends* `tsconfig.json`; a parent `exclude`
  is inherited and can silently exclude the web entry point → tsc "succeeds"
  emitting nothing while stale dist files keep serving. The web config now sets
  `"exclude": []`. **Verify the artifact, not the build log**
  (`grep <new-symbol> examples/web/dist/...`).
- **Gotcha:** browsers cache ES modules — hard refresh (Cmd-Shift-R) after rebuilds.
- **Gotcha (this machine):** the `rtk` shell hook summarizes/filters tool output
  and can report "No errors found" over a failing command. When output looks
  inconsistent with exit codes, re-run via `rtk proxy <cmd>` for raw output.

Shell/binary behavior (pacing, temperatures, the `reboot` theater, the
`rewrite` hook that feeds `sudo reboot` to the model) lives in
`src/infer/binaries.ts` — registering a new command is a ~5-line manifest.

**Inference engines:** the page races WebGPU (`src/gpu/session.ts` — resident
weights/KV-cache, GPU-side top-k sampling, 48-token chunks per readback)
against the pure-JS CPU session for 24 tokens at load and keeps the winner
(banner shows both rates). Parity gate: `deno run --allow-read bench/gpu-infer.ts`
(logits ≤1e-4, greedy text identical). Perf notes: per-dispatch overhead
dominates GEMV-sized work — under Deno/wgpu (~110µs/dispatch) CPU wins for
mini (118 vs 79 tok/s); browsers differ, hence the race. Single-workgroup
fused layer kernels (`wgsl-fused.ts`) measured 2× slower (one workgroup = one
GPU core); a multi-workgroup split is the documented next step if browser
numbers disappoint.

---

## 7. Model zoo (what's on disk)

| File | Params | Trained on | Notes |
|---|---|---|---|
| `models/terminal.bity` | 165k | corpus v1 (12k steps, CPU) | nano; dreams in paths |
| `models/terminal-milli.bity` | 1.2M | corpus v1 (12k, CPU ×8) | first arg-copying ping |
| `models/terminal-micro.bity` | 2.7M | corpus v1→v2 (12k, GPU) | val 1.75 record on v1 split |
| `models/terminal-micro-v2.bity` | 2.7M | corpus v2 (12k, GPU) | rm-consistency, cowsay fixed |
| `models/terminal-micro-v3.bity` | 2.7M | corpus v3 (12k, GPU) | stateful structure, but names dreamed not copied |
| `models/terminal-micro-v4.bity` | 2.7M | corpus v4 (16k, GPU) | copy circuit formed — syllable-dialect only |
| `models/terminal-micro-v5.bity` | 2.7M | corpus v5 (16k, GPU) | universal copy: mkdir flowers → ls shows flowers |
| `models/terminal-mini.bity` | 10.7M | corpus v5 @ 22.8MB (16k, GPU, 6h) | crisp copier (echo xk4vw9 ✓), perfect cowsay, IP-consistent ping |
| `models/terminal-mini-v6.bity` | 10.7M | corpus v6 (16k, GPU, 6h) | filesystem consistency: mv/rm/pwd, echo>file read-back, cat-after-rm ENOENT |
| `models/terminal-mini-v7.bity` | 10.7M | corpus v7 (16k, GPU, 6h) | persistent location: `cd` walks the tree, prompt carries the path, per-dir `ls`; `wc -l`, `which`. Ceiling: multi-word `cat` first-word only |
| `models/terminal-25m.bity` | 25.3M | corpus v7 (16k, MLX, 48m) | scale test: broke `wc -l`/`mv`, but multi-word content STILL 0% → proved it's coverage, not capacity |
| `models/terminal-v8.bity` | 10.7M | corpus v8 @ 35MB (16k, MLX, 24m) | **exhaustive coverage: multi-word content 100%, nested `cd` 100%, touch→empty 100%; env/curl-body/ip-a/version-banners; val gap −0.0055 (no overfitting)** |
| `models/terminal-mini-v9.bity` | 10.7M | **hybrid corpus** @ 28.5MB (16k, MLX, 24m) | **the hybrid pivot: FS/text/identity are real code, so this trains only the dreamed set. Dreams `ps`/`df`/`free`/`top` correctly; `kubectl`/garbage → clean `command not found`; `git`/`ping` good. train 0.521 / val 0.545 (gap +0.024)** |
| `models/terminal-micro-v9.int8.bity` | 2.7M | hybrid corpus (16k, MLX) | MICRO size-sweep slot, retrained on the hybrid corpus |
| `models/terminal-25m-v9.int8.bity` | 25.3M | hybrid corpus (16k, MLX) | MAX size-sweep slot, retrained on the hybrid corpus |
| `models/terminal.int8.bity` | — | — | **currently deployed: MINI v9 (hybrid).** The demo's three-size sweep (MICRO/MINI/MAX) is all hybrid-v9. (The wide 57M ULTRA slot was retired.) |

**Programmatic core (v9+):** deterministic commands no longer come from any model —
`src/infer/vfs.ts` (in-memory FS), `coreutils.ts` (~35 binaries), `shell-exec.ts`
(pipes/redirects/globs/`&&`), routed ahead of the model in `Shell.run`. See `LEARNING.md`
and the DESIGN v5 note.

Checkpoint format (`bity1`): `[u32 headerLen][JSON header][raw f32/i8 blobs]` —
config + tokenizer vocab travel inside; `deserialize()` handles f32 and int8.
**The checkpoint is the contract**: train anywhere (CPU/GPU/another framework
with a 30-line exporter), the browser loads it unchanged.

---

## 8. Fast reference — full cold-start reproduction

```bash
npm install
npm test                                          # trust the stack
npm run corpus                                    # (re)build training text
deno run --allow-read bench/gpu-parity.ts         # trust the GPU
deno run --allow-read --allow-write examples/train-terminal-gpu.ts \
  --steps 12000 --batch 32 --layers 6 --heads 6 --dim 192 \
  --out models/terminal-micro.bity                # ~85 min on M4 Pro
node examples/export-int8.ts --in models/terminal-micro.bity --out models/terminal.int8.bity
npm run web                                       # open http://localhost:8143/examples/web/
```
