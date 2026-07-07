# The bityllm Journey

*A lab-notebook narrative of building a language model from absolutely nothing —
no frameworks, no datasets, no GPU libraries — until a terminal dreamed in a
browser tab. One long day, six models, three real bugs, one poisoned batch.*

---

## I. The premise

The rules were almost a dare: a tiny LLM in **pure TypeScript**, **zero
dependencies**, trained **from scratch**. Not a wrapper around an API, not
weights borrowed from someone else's run — the autograd, the transformer, the
optimizer, the tokenizer, all of it ours, small enough to read in an afternoon.

Then the premise sharpened into a *purpose*. This wouldn't be a chatbot. The
model would power a **virtual terminal** — a web page with a blinking cursor
where every command is a "binary" that runs inference. Type `ping bity.dev`
and a 2.7-million-parameter dream replies with ICMP echoes from an address
that has never existed. Nothing executes. It is, as the design doc put it, *a
dream of a computer* — and a char-level model is weirdly perfect for dreaming
terminals, because terminals are all texture: prompts, columns, permission
strings, the liturgy of systemd shutting down.

The design doc made three bets early: train from scratch (so we'd need real
autograd), run isomorphically (so inference had to be lean enough for a
browser), and pure TypeScript first (so every optimization would be a choice,
behind a seam, not a dependency). All three bets paid — none of them cheaply.

## II. The data problem nobody has solved for us

Here was the first surprise: **there is no dataset of terminal sessions.**
There are datasets of *commands* (nl2bash, tldr) and descriptions of commands
(man pages), but almost nothing pairs a command with what it *prints*. The one
clean dataset we found had exactly 100 rows.

So we built the data. A capture harness spun up a throwaway Debian container
(hostname `bity`, user `guest`) and *harvested the machine itself* — every man
page, every `--help`, every package's file list, `/etc`, deliberate errors,
`cowsay`. Four megabytes of real, format-perfect terminal text, sanitized of
every MAC address and identity. The gotchas were period-authentic: slim images
strip their own man pages; `fortune` hides in `/usr/games`; the vocabulary
bloated to 432 characters until we discovered Armenian man pages and tamed it
to a clean 101.

Then the user asked the right question: *what about `reboot`?* You cannot
capture a reboot from a container — a container never boots. You need a
machine that can die and come back. So a real Debian VM (Lima + QEMU) was
born, rebooted on command, and its serial console scraped for the full
liturgy: `[ OK ] Stopped target Multi-User System` → EFI stub → kernel boot →
`bity login:`. The user contributed their own dmesg files from real hardware —
a Framework desktop's boot log, a genuine shutdown journal — and when asked
whether the persona should be ARM or x86, chose the only honest answer for a
hallucination: **"eclectic dream machine."** Both kernels live in the corpus.
The terminal dreams of being several computers at once.

## III. Gates, not vibes

The model core went up in one straight line because every layer had a gate in
front of the next: finite-difference **grad-checks in float64** for every op;
a **full-model grad-check** across every parameter of a real GPT; the
**overfit-one-batch** test (memorize or you don't ship); bit-exact determinism
from a seeded PRNG. All green on first run — not because the code was
miraculous, but because the tricky formulas (LayerNorm backward, fused
cross-entropy, tied-embedding gradients) had to answer to arithmetic instead
of opinion.

Then we trained nano — 165k parameters — on the captured corpus, and got our
first humbling: it **dreamed in file paths**. Every command answered with
Debian path-salad, because `dpkg -L` listings dominated the corpus by volume.
The lesson that shaped everything after: *for a tiny model, the corpus is the
product*. Real captures give truth; only **synthetic generators** give
repetition-with-variation, and repetition-with-variation is how a char model
learns that `ping bity.dev` should answer *about bity.dev*.

So we wrote generators — format-faithful to the byte, randomized in every
slot. Retrained. At step 2,400 the babble snapped into `ls -la` rows. At
7,200 came the moment the whole design hinged on: `ls` produced a clean,
alphabetized, persona-consistent listing **and then emitted the next prompt
on its own** — the model had learned where commands end. And in the audition,
there it was: `PING bity.dev (140.34.216...)` — the argument copied from the
user's command into the dream.

## IV. The need for speed (or: how many ways can a Mac compute?)

Pure-TS training started at ~1,800 tok/s. Three escalations later:

1. **Data-parallel workers** — 8 P-cores, weights in a SharedArrayBuffer,
   Atomics generations. 6.8×. Stress-testing found a classic **lost-wakeup
   race** (waiting on a re-loaded value instead of the observed one) that
   would otherwise have been "training randomly hangs, sometimes."
2. **Register-blocked kernels** — 4-wide k-unrolling. Another 1.8×. Twelve×
   over day one, still zero dependencies.
3. **The GPU question.** The user asked about "Apple ML cores." The honest
   answer: the Neural Engine is inference-only and nobody trains on it — but
   the 16-core **Metal GPU speaks WebGPU**, and WebGPU is just TypeScript's
   dialect of the future. A 200-line WGSL GEMM hit **230 GFLOP/s — 44× our
   CPU kernel** — and that settled it. One day later: thirteen WGSL kernels,
   an llm.c-style explicit schedule, weights/grads/moments resident on-GPU,
   and a parity gate proving every gradient against the CPU autograd to 5e-5.
   Micro (2.7M params): **~85 minutes** instead of 5.7 hours.

## V. The poisoned batch (the best bug)

Two thousand steps into the first GPU run, a model with loss 0.62 turned
**100% NaN in eleven steps.** No divergence ramp. Deterministic to the exact
step — 2011 — across three runs.

The hunt was a staircase of eliminations. The scan said *everything* was
corrupt, weights included — that turned out to be a cascade amplifier of our
own: the clip kernel multiplied infinite gradients by zero, and `inf × 0`
mints NaN. Fixed, but the *source* remained. The poison batch, replayed
through our seeded RNG, was ordinary traceroute text. A boundedness argument
said the forward pass mathematically *could not* overflow from the verified
state. Impossible bug, deterministic reproduction — the best kind.

So we bisected reality: replayed 2,010 steps, ran the poison step
forward-only, and scanned every activation buffer in dataflow order.
**`h4.gelOut`: one bad element. Its input: finite, max 10.08.**

And there it was. Metal's fast-math `tanh(u)` computes `(e^{2u}−1)/(e^{2u}+1)`,
which overflows float32 when `u > 44.36`. GELU's inner term crosses that at
activation `x ≈ 10.06`. For two thousand and ten steps, no activation in the
network had ever exceeded 10.06. On step 2,011, exactly one did — and one
`tanh`, in one element, in one layer, detonated a 2.7-million-parameter model.
JavaScript's `Math.tanh` saturates correctly, which is why the CPU never saw
it and parity couldn't catch it. The fix is a clamp that costs nothing:
`tanh(15)` is already 1.0 to fifteen decimal places. The retrained run sailed
through step 2,011 without a flicker.

Somewhere in the middle of this, a comedy: asked to pause training for
battery, we SIGSTOP'd a process and training… continued. The user's "I didn't
do anything" cracked it — we'd frozen the *shell wrapper* while its child
burned GPU underneath. `pgrep | head -1` is not your friend. The runbook
remembers.

## VI. The terminal comes alive

The browser demo is the design doc made flesh: a green-phosphor CRT, a **KV-
cached inference session** (per-token cost independent of scrollback), int8
checkpoints at 26% size, per-binary manifests — `ping` paced at a heartbeat
per line, `traceroute` hesitating at each hop, `reboot` doing full hybrid
theater: a model-dreamed shutdown, a pause, darkness, and a login banner.

Each model generation earned its place on stage. Nano dreamed paths. Milli
(1.2M) obeyed `-c 3` and kept one IP consistent across a whole ping. Micro
(2.7M) finally mastered the hardest texture in the corpus — **cowsay's bubble
arithmetic**, borders sized to the message — and produced a `git status` on a
branch named, with no prompting, `fix/atomics-race`. The dream has a sense of
humor about its own repository.

Even the failures grew interesting. `cowsay hello` once printed no cow at all —
not a capacity failure but **lexical crosstalk**: "hello" is soaked in `echo`
contexts, and the word dragged the conditioning sideways. The fix was more
cows saying ordinary words. Corpus design at this scale is closer to animal
training than data engineering.

## VII. Teaching the dream to remember

The last frontier (so far): the user asked for `mkdir` to *matter* — create a
directory, and later `ls` should show it. Pure hallucination, no scripted
state. The first attempt taught the model that listings *sometimes contain
pool names* — memorization, not memory. Deletions worked; creations didn't.
The diagnosis was humbling and precise: inserting an arbitrary name into an
*alphabetically sorted* listing is a genuinely hard program for 2.7M
parameters, while ping's argument-echo is trivial because the host lands at a
fixed template position.

So the corpus was reshaped to make memory *ping-shaped*: created names are
random syllables (`kelbo.txt`, `mirzu` — memorization impossible, copying the
only strategy), listings append creations at the end in creation order (no
sorting program required), and the base filesystem never varies. That model
is training as this document is written. If it works, the dream gains object
permanence. If it doesn't, the next lever is scale — an 11M-parameter mini is
now just an evening's GPU run.

## VIII. What it added up to

- **~3,500 lines of TypeScript**, zero runtime dependencies, every layer ours:
  tensor → tape → GPT → AdamW → tokenizer → checkpoint → KV-cache → shell → CRT.
- **Six models** trained end-to-end; throughput journey **150 tok/s → 22,000
  (CPU) → 9,800 at 16× the model size (GPU)**.
- **Three real bugs**, each now a permanent lesson in the code: an Atomics
  lost-wakeup, a NaN cascade in gradient clipping, and a floating-point cliff
  inside Apple's tanh.
- One **terminal that dreams**, in a tab, offline, in 2.7 MB.

The recurring theme, if there is one: at this scale nothing is a black box.
Every mystery — path-dreams, missing cows, the poisoned batch — had a concrete
cause you could bisect your way to, because we owned every layer and every
layer had a gate. The model is tiny. That's the point. It's small enough to
understand, and it still surprises.

*— written at step ~6,000 of the v3 run, while the terminal learns that
`mkdir` means something.*

---

## Postscript: it learned

v3 failed beautifully — right slot, right *count*, wrong names: the model
appended freshly-dreamed syllables instead of yours. The diagnosis went
deeper than the feature: **nothing in the model could copy an arbitrary
string at all.** Even ping, our showpiece, had been *retrieving* pool hosts,
not copying. The mechanism required — an induction circuit — had simply never
been given a reason to form.

So we gave it one. v4 added a **copy curriculum**: thousands of `echo
<random-string>` drills and error messages that parrot their arguments —
tasks where copying is the only way to be right. It half-worked, and the
half that failed was self-inflicted: drilled only on twenty syllables, the
model learned a copier that spoke *syllable*. `echo fenlodov` → `fenlodov`,
flawless; `echo zanzibar` → dialect gibberish. v5 replaced the drills with
uniform random characters — no pattern to lean on, copy or die — and we
watched the circuit assemble across checkpoints like a time-lapse: two
correct characters at step 4,800, four at 8,000 (`quokka` → `quikka`, one
edit from true), and at 16,000:

```
guest@bity:~$ mkdir flowers
guest@bity:~$ ls
notes.txt  projects  todo.md  flowers
```

Object permanence, hallucinated. It's a *young* circuit — it stutters
(`zanzi zanzibar`), drops a letter under pressure (`quoka`), and random
alphanumerics still strain it — but the mechanism exists, it generalized
from echo drills to filesystem listings unprompted, and it ships in the
browser demo. The sharpening, if we want it, is one scale step away (the
11M mini — possibly on the very Strix Halo whose boot log this model
dreams in fragments).

The last lesson of the journey may be the best one: **a model this small
doesn't learn what your data permits — it learns exactly what your data
forces.** Twenty syllables taught a twenty-syllable copier. The alphabet
taught the alphabet.

---

## IX. Mass, speed, and a URL

The 11M **mini** trained overnight on the M4 Pro's GPU — six hours, one
battery-pause intermission (during which we discovered we'd SIGSTOP'd an
innocent wrapper shell while the actual trainer burned on; process trees
keep their own counsel) — and woke up *sober*. The copy circuit that
stuttered at 2.7M was crisp at 11M: `echo xk4vw9` verbatim, `mkdir` and
`rm` both remembered, one IP per ping, byte-perfect cows. Capacity didn't
create the mechanism — the curriculum did that — but capacity made it
reliable.

Mini ran at 116 tok/s in browser JavaScript — coincidentally the speed of a
1200-baud modem, which felt almost too thematically convenient to fix. We
fixed it anyway, or tried: a WebGPU inference engine, weights and KV-cache
resident, sampling *on the GPU* so tokens generate in 48-token chunks with
one readback each. It produced byte-identical greedy text to the CPU engine
(the parity discipline held to the end) and taught two final lessons in
GPU humility: per-dispatch overhead dominates GEMV-sized work, and the
"obvious" fix — fusing whole layers into mega-kernels — ran 2× *slower*,
because one workgroup occupies one of sixteen GPU cores. Correct, elegant,
measured, rejected. The shipping answer is characteristically empirical:
the page **races both engines for 24 tokens at load and keeps whichever
wins on your machine.**

Then the dream got a door. `git init` (the repo's first commit, after six
models and four documents), a pre-publication scrub that caught the last
two hardware identifiers, an MIT license, and an icon that draws the whole
story in one image: a green CRT, cursor blinking, dreaming of a flower.

**https://jyatesdotdev.github.io/bityllm/**

Anyone, anywhere, can now type `mkdir flowers` into a browser tab and watch
eleven million parameters — trained from a blank TypeScript file, on one
desk, in three days — remember it.
