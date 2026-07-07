<p align="center"><img src="examples/web/bity.svg" width="170" alt="bity — a terminal dreaming of a flower"/></p>

# bityllm

A tiny GPT trained **from scratch** in **pure TypeScript** with **zero runtime
dependencies**, powering a terminal that dreams: every command in the browser
demo — `ping`, `cowsay`, `reboot`, even `mkdir` — is hallucinated,
character-by-character, by an 11M-parameter model trained from scratch on a single Mac.

**▶ Live demo:** https://jyatesdotdev.github.io/bityllm/ — type `help`, `ping bity.dev`,
`cowsay moo`, or `mkdir something` and then `ls` (it remembers — the model
learned to copy your names from context).

```
guest@bity:~$ mkdir flowers
guest@bity:~$ ls
notes.txt  projects  todo.md  flowers        ← pure hallucination, with memory
```

Everything here is built from nothing: the autograd engine (grad-checked
against finite differences), the transformer, AdamW, the char tokenizer, the
checkpoint format, a data-parallel CPU trainer (`worker_threads` + Atomics), a
**WebGPU trainer** (13 WGSL kernels, trains the 11M model in ~6 h on an M4 Pro),
and a KV-cached inference engine in both pure-JS and WebGPU flavors (the demo
races them and keeps the winner). Training data comes from real Debian
containers, a QEMU VM that was actually rebooted for its boot logs, and
synthetic generators — including the "copy curriculum" that forced an
induction circuit to form at 2.7M parameters.

| Doc | What's in it |
|---|---|
| [DESIGN.md](DESIGN.md) | the architecture and every decision, written before the code |
| [RUNBOOK.md](RUNBOOK.md) | reproduce everything: corpus capture → training → deploy, with all gotchas |
| [JOURNEY.md](JOURNEY.md) | the narrative — six model generations, three real bugs (incl. a floating-point cliff inside Apple's fast-math `tanh`), one emergent copy circuit |

## Quickstart

```bash
npm install            # types only — zero runtime dependencies
npm test               # 26 tests: grad-checks, overfit gate, parity, round-trips
npm run corpus         # rebuild the training corpus from committed captures
npm run web            # serve the terminal at http://localhost:8143/examples/web/
# train your own (see RUNBOOK for CPU/GPU options and presets):
deno run --allow-read --allow-write examples/train-terminal-gpu.ts \
  --steps 16000 --batch 32 --layers 6 --heads 6 --dim 384 --out models/mine.bity
```

*License: [MIT](LICENSE).*
