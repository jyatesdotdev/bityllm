<p align="center"><img src="examples/web/bity.svg" width="170" alt="bity — a terminal dreaming of a flower"/></p>

# bityllm

A tiny GPT trained **from scratch** in **pure TypeScript** with **zero runtime
dependencies**, powering a **hybrid terminal**: deterministic commands (`ls`,
`cat`, pipes, redirects, `cd`) run as **real code** over an in-memory filesystem,
while generative ones — `ping`, `cowsay`, `git`, `reboot` — are **hallucinated**,
character-by-character, by a ~10.7M-parameter model trained from scratch on a single Mac.

**▶ Live demo:** https://jyatesdotdev.github.io/bityllm/ — type `ls`, `cat notes.txt`,
`ping bity.dev`, `cowsay moo`, or `mkdir x && cd x && pwd` (the filesystem is real and
remembers). A retro CRT **channel dial** in the header swaps between model sizes so you
can watch size trade coherence for speed, live.

```
guest@bity:~$ echo one two three > f && cat f    # real code: an in-memory virtual FS
one two three
guest@bity:~$ ping bity.dev                      # dreamed by a 10.7M model
PING bity.dev (172.18.201.175): 56 data bytes
64 bytes from 172.18.201.175: icmp_seq=0 ttl=55
```

## Highlights

- **Hybrid execution.** The breakthrough for breadth: deterministic + stateful
  commands (a real in-memory filesystem for `ls`/`cd`/`cat`/`mkdir`/…, plus a
  mini-shell with pipes, redirects, globs, and `&&`) are **real code** — always
  correct and consistent. The model is reserved for what it *should* dream
  (`ping`/`git`/`ps`/`man`/fun) plus a graceful `command not found` for the
  unknown. Deterministic → code; generative → model.
- **The checkpoint is the contract.** Train in any language or framework —
  pure-TS CPU, from-scratch WebGPU, or Apple MLX — and ship the *same* `bity1`
  file; the browser loads it unchanged.
- **The ceiling was data, not capacity.** Corpus **v8** broke three
  long-standing behavioral ceilings at 10.7M params — multi-word content copy
  (`echo a b c > f; cat f`), nested `cd`, and `touch x → cat x` (empty) — all
  **0% → 100%**. The multi-word ceiling *survived* a 2.4× scale-up to 25M
  unchanged, then fell to a corpus fix at 10.7M. **The model learns what the
  data forces, not what it permits.** (Those filesystem behaviors are now **real
  code** — see *Hybrid execution* above — but the lesson holds for everything the
  model still dreams. Measured train/val gap **−0.0055** — zero overfitting; the
  synthetic random-name corpus structurally killed memorization.)
- **Three training backends, one architecture.** Data-parallel CPU
  (`worker_threads` + Atomics), a **from-scratch WebGPU trainer** (13 WGSL
  kernels, the canonical/educational path, ~6 h for the 10.7M model on an M4 Pro),
  and an optional **MLX fast path** — ~46,500 tok/s vs ~3,000, ~15× faster, so
  that run drops to **~24 min**. MLX vs the independent TS forward agree to **max
  Δlogit 1.2e-6** (argmax identical).
- **Races WebGPU vs pure-JS CPU at load** and keeps the winner (both rates shown
  in the banner). KV-cached inference in both flavors.
- **Built from nothing:** the autograd engine (grad-checked against finite
  differences), the transformer, AdamW, the char tokenizer, and the checkpoint
  format — no ML libraries in the runtime.

Training data is **capture-heavy** (~68% real): an exhaustive multi-user harvest
of a real Debian container (534 distinct commands, guest + root), a QEMU VM that
was actually rebooted for its boot logs, and synthetic generators for the dreamed
families (`net`/`git`/`fun`/`sysinfo`) plus a graceful-`command not found` drill —
the corpus now trains *only* what the model still hallucinates, since the rest is
real code.

| Doc | What's in it |
|---|---|
| [**LEARNING.md**](LEARNING.md) | **start here to learn** — a guided reading path through the code, deep dives on autograd/attention/sampling, extension projects, and how these concepts generalize beyond LLMs (CNNs, RNNs, diffusion, RL) |
| [DESIGN.md](DESIGN.md) | the architecture and every decision, written before the code |
| [RUNBOOK.md](RUNBOOK.md) | reproduce everything: corpus capture → training (CPU/GPU/MLX) → deploy, with all gotchas |
| [JOURNEY.md](JOURNEY.md) | the narrative — model generations, real bugs (incl. a floating-point cliff inside Apple's fast-math `tanh`), the emergent copy circuit, and the corpus-v8 ceiling break |

## Model zoo

Every model shares the exact same `bity1` contract; the demo's channel dial
swaps across the size sweep. Full table (all generations) in [RUNBOOK §7](RUNBOOK.md).

| Model | Params | Highlight |
|---|---|---|
| MICRO | 2.7M | copy circuit emerges — fastest, fuzziest (**hybrid v9**) |
| **MINI** (deployed) | 10.7M | **hybrid corpus v9** — dreams `ping`/`git`/`ps`/`df`; graceful `command not found`; FS/text are real code |
| MAX | 25.3M | the scale test that proved the ceiling was **coverage, not capacity** (**hybrid v9**) |

(All three sizes now run the same hybrid-corpus v9.)

## Quickstart

```bash
npm install            # types only — zero runtime dependencies
npm test               # 49 tests: grad-checks, overfit gate, parity, round-trips, shell core
npm run corpus         # rebuild the training corpus from committed captures
npm run web            # serve the terminal at http://localhost:8143/examples/web/

# train your own (see RUNBOOK for CPU / WebGPU / MLX options and presets):
deno run --allow-read --allow-write examples/train-terminal-gpu.ts \
  --steps 16000 --batch 32 --layers 6 --heads 6 --dim 384 --out models/mine.bity

# ...or the Apple-Silicon fast path (~15× faster, same bity1 output):
.venv/bin/python train/mlx_train.py --steps 16000 --batch 32 --block 128 \
  --lr 6e-4 --layers 6 --heads 6 --dim 384 --out models/mine.bity
```

*License: [MIT](LICENSE).*
