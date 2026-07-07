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
npm run corpus     # = node corpus/build.mjs  → corpus/data/bity.corpus.txt (~13.6 MB)
```
- ~68% synthetic (generators in `corpus/generators/`: fs, sys, net, git, fun,
  **fs-session** — stateful mkdir/touch/rm → ls blocks) + ~32% real captures,
  **shuffled at session granularity** so the val split is representative.
- Key lesson (corpus v3): to teach copy-from-context (mkdir X → ls shows X),
  use **random names** (memorization impossible) and **append-order listings**
  (no sorted-insertion program) — mirror how `ping <host>` echoes its argument
  at a fixed template position.

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
| `models/terminal-mini-v7.bity` | 10.7M | corpus v7 (16k, GPU, 6h) | **persistent location: `cd` walks the tree, prompt carries the path (`~/projects$`), per-dir `ls`; `wc -l`, `which`.** Ceiling: multi-word `cat` content copy still first-word (0% at 11M → capacity, scale trigger). Audit: `node bench/eval.mjs` |
| `models/terminal.int8.bity` | — | — | **currently deployed: mini-v7** |

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
