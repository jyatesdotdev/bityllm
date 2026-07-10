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
- **Six models** trained end-to-end; throughput journey **~1,800 tok/s → 22,000
  (CPU, ~12×) → 9,800 at 16× the model size (GPU)**. (Later: MLX made it 46,500 —
  see Ch. X.)
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

---

## X. Fifteen times faster, and the checkpoint that made it free

The demo shipped. The runbook was written. And then the project's own founding
principle came back to collect.

*"The checkpoint is the contract"* had been a design slogan — train anywhere,
ship the same `bity1`, the browser loads it unchanged. It was time to cash it.
Apple's **MLX** speaks Metal in fused kernels our hand-written WGSL never could,
and the temptation was obvious: keep the from-scratch trainer as the canonical,
educational one — every gate, every op, ours to read — but add an *optional*
fast path for the runs where we just wanted the weights.

The number was almost rude. The WebGPU trainer managed **~3,000 tok/s** on the
M4 Pro; MLX did **~46,500** — fifteen times faster, same architecture, same
10.7M parameters. The six-hour overnight mini run became a **twenty-four-minute**
coffee break. And the lesson was quietly deflating for the WGSL we'd been so
proud of: *the naive kernels, not the silicon, had been the ceiling all along.*
The GPU was never the bottleneck; our GEMM was.

But speed you can't trust is worse than slow you can. So the same discipline
that gated everything else gated this: MLX runs its forward, exports a `bity1`,
and the **independent TypeScript engine** — code that shares not a single line
with MLX — runs its own forward on the identical weights. They agreed to **max
Δlogit 1.2e-6**, argmax identical on every token. That number is the whole
contract made auditable: the tanh-approx GELU, the tied LM head, the LayerNorm
epsilon, and MLX's `[out,in]` Linear layout transposed to our `[in,out]` — all
of it either lines up to six decimals or it doesn't, and it did. (The script
also grew a held-out 5% validation split and train/val gap reporting, so a fast
run still tells you the truth about itself.)

One subtlety earned its own scar. MLX's AdamW decays *every* parameter; our
trainer, faithfully, decays only the 2-D matmul weights and leaves embeddings,
LayerNorms, and biases alone. The first MLX run used the framework default and
paid for it — the fuzziest behaviors quietly regressed (`cat` of an uncreated
`.csv` went 75%→0%, `mv`→`ls` wobbled). Setting MLX's decay to zero and applying
decoupled, 2-D-only decay by hand recovered them exactly. A reminder that a
"backend swap" is only lossless if you swap *all* of it, param groups included.

### The bf16 that wasn't

There was an obvious next lever, and it was a trap. `--bf16`: run the matmuls in
half precision, keep master weights, LayerNorm, softmax, loss, and optimizer in
fp32, keep the *export* fp32 so inference and parity never notice. On an NVIDIA
card this is a free 1.5–2×. We built it, measured it carefully, and got
**+2–4%** — 10.7M went 45.0k→46.9k tok/s, 25M went 22.3k→22.7k. Noise wearing a
costume.

The why is pure hardware honesty: **Apple GPUs have no tensor cores.** fp32 and
bf16 matmuls run at nearly the same rate, so there's no compute to reclaim — and
keeping an fp32 master means casting fp32→bf16 on every forward, whose memory
traffic *cancels* the bandwidth the smaller matmul was supposed to save. Loss
stayed identical to three decimals, so the flag is safe; it's just pointless
here. We kept it opt-in and off by default, a door left open for a CUDA backend
that may never come. The fp32 trainer, it turned out, was already sitting near
the practical ceiling — **~46–52% MFU**, the GPU pinned at its max **1578 MHz**
clock drawing a steady **~22 W**. There was no faster to find. Correct, elegant,
measured, shelved — the same verdict the fused mega-kernels earned a chapter
ago. Retiring good ideas for being empirically pointless was becoming a house
virtue.

## XI. The wall that scale could not break

Now that a full run cost twenty-four minutes instead of an evening, we could
finally afford to be greedy — and greed found a wall.

Every model since mini had one stubborn failure. Ask it to copy a *single* token
from context and it was flawless: `echo hello > f; cat f` gave back `hello`. Ask
it to copy *more than one* — `echo a b c > f; cat f` — and it returned, with
total confidence, `a`. The copy circuit read back the first token and stopped.
It had never been the showpiece bug, but it sat at exactly **0%** across every
eval, and it would not move.

The obvious diagnosis was capacity: 10.7M parameters, maybe the induction
circuit just didn't have the width to hold a multi-token span. And now we had
the throughput to test it properly. We scaled to **25.3M** — `8L/8H/512d`, a
clean 2.4× — held the corpus and the recipe *constant*, and let MLX churn it out
in **48 minutes**.

Scale won some things handsomely. `wc -l` line-counting went **0%→100%**;
`mv`→`ls` went **100%**; every fuzzy case firmed up; the 25M model posted a
**0.359** val loss and was, by the eval harness, the best-behaved model we'd ever
trained. It was a genuinely better brain.

And the multi-word wall stood at **0%.** Exactly, insultingly, zero. Three
independent models now — the WebGPU 10.7M, the MLX 10.7M, and the MLX 25M —
failed the identical case in the identical way, returning the identical first
token. When 2.4× the parameters, a *different training framework*, and a better
validation loss all fail a task the same way, the hypothesis is dead. It was
never capacity. The circuit read the first token reliably because the data had
*only ever asked it to.* No example in the corpus had forced a longer copy, so
no longer copy existed. The wall wasn't in the model. It was in the data, and it
had been all along.

## XII. Seven auditors and a critic

If the gap was coverage, the fix was to find every gap at once — and by now the
corpus was too large for one pair of eyes to audit honestly. So we ran it in
parallel: **seven auditors, one per command category** — filesystem, network,
git, system, fun, toolchains, errors — each tasked with adversarially hunting
for what the model *couldn't* do, feeding a single **consolidating critic** whose
only job was to be meaner than any of them and throw out the false alarms.

It earned its keep immediately by catching two real bugs in our own generators —
not model failures, *corpus* failures, the worst kind because the model
faithfully learns them. The `rm -rf` failsafe had been bound to *every* `rm -rf`
instead of the literal `/` it was meant to guard, so the corpus was teaching the
model to refuse deletions it should have performed. And `cd` into a missing
directory emitted a *random* errno instead of the right one — a small lie, but
the model memorizes lies as eagerly as truths. Both had been quietly poisoning
behavior for generations.

Then the breadth. The audit turned up whole categories the model had simply
never seen: `env` / `printenv` / `echo $VAR` and exit codes and `alias` and
`type`; `curl` that returned a response *body*, not just headers; `ip a` /
`ip route` / `dig`; the little version liturgy every developer knows by heart
(`node -v`, `git --version`, `python3 --version`); `git commit`; and a whole
permission-denied persona family. `fs.mjs` grew a real **nested-path stack** and
**byte-accurate file metadata** — the kind of consistency where `ls -l`, `wc`,
and `stat` all report the *same* size because they read the same underlying
truth — plus `wc`/`head`/`tail`/`chmod` and honest pipes and redirects. The
corpus reached **35 MB**, and the governing rule was stated plainly:
**referential consistency is paramount — a *wrong* addition is worse than a
missing one.** A gap the model routes around; a contradiction it dutifully
learns.

And then the reckoning. Same 10.7M parameters, same twenty-four-minute MLX run,
new corpus — and the walls came down. Multi-word content copy (`echo a b c > f;
cat f` → `a b c`): **0%→100%.** Nested `cd` (`cd a; cd b; pwd` walking to the
deep path): **0%→100%.** `touch x; cat x` returning a genuinely *empty* file:
**0%→100%.** The three ceilings that had survived a scale to 25M all fell to a
*smaller* model on better data. There is no cleaner statement of the whole
project's central law, so we let the model state it: **it learns what the data
forces, not what it permits.** Twenty-five million parameters permitted the
multi-word copy. Only the corpus forced it.

The instrument that adjudicated all of this — `bench/eval.mjs`, **eight seeds
per case**, grepping for behaviors and reporting cold pass-rates — is the reason
none of these numbers are vibes. It's what turned "seems better" into 0% and
100%. And it delivered one last verdict worth framing: a final **train/val gap
of −0.0055** — validation loss *below* training loss — meaning **zero
overfitting**, at 35 MB and 10.7M parameters, because a corpus of RNG-generated
names and contents gives the model nothing to memorize. It can only learn the
*grammar* of the machine, never a specific answer. The synthetic corpus that
began as a workaround for having no data turned out to be the thing keeping the
model honest.

That model — **v8** — is the one now dreaming behind the URL.

## XIII. Turn the channel

One thing was still missing, and it was the most fun. The whole scale saga —
micro at 2.7M, v8 at 10.7M, the 25M — was invisible to anyone but us, buried in
eval tables. But the models were *right there*, all three trained on the same v8
corpus, differing only in mass. So we gave the terminal a dial.

The header now names the brain — version and size, so you know which mind you're
talking to — and a retro, CRT-appropriate **channel selector** switches between
the three live: **micro 2.7M**, **v8 10.7M** (the default), and the **25M**.
Same corpus, same commands, three different amounts of parameter. Type `cowsay`
on the small one and watch the coherence fray at the edges; flip to the big one
and watch it snap into focus but pace slower. It is the entire
capacity-versus-coherence-versus-speed trade of this whole document, turned into
a knob a stranger can turn in a browser tab.

The lab notebook, finally, made playable.

## XIV. The half that shouldn't dream

The dial was live, the ceilings were broken, and then someone typed `find`.

Gibberish. `docker ps` — gibberish. `tar`, `sed`, `awk`, `man ssh` — a smear of
plausible-looking characters that meant nothing. The model was crisp inside a band
of maybe fifty commands and dissolved the instant you stepped outside it. We had
spent the whole project learning to read one law — *the model learns what the data
forces, not what it permits* — and here it was again, wearing a new face. The corpus
forced about fifty commands. So the model knew about fifty commands. Everything past
the edge of the training set was a confabulation.

There were two ways out. Pour in more data until the band was wide enough to cover
a Debian install — or stop asking the model to do things it had no business doing.
The second idea took a while to say out loud, because it sounded like giving up. But
`reboot` had been *scripted* since almost the beginning — real code driving the
shutdown lifecycle while the model only dreamed the systemd lines. If `reboot` could
be real, why was `ls` a hallucination? `ls` should never be *creative*. It should be
*correct*. We had been asking a language model to remember which files existed, and
paying for it in a corpus burden we'd been fighting for generations — the `ll`
alias that wouldn't stick, the `cat` after `rm` that still showed the deleted file,
the referential-consistency bugs that were never really the model's fault. They were
the fault of asking a dream to be a database.

So we split the shell down the middle. **Deterministic, stateful, must-be-consistent
→ real code. Generative, variety-is-the-point → the model.** A real in-memory
filesystem (`vfs.ts`), thirty-odd real coreutils over it (`coreutils.ts`), and a
real little shell to tie them together (`shell-exec.ts`) — tokenizing quotes, wiring
pipes, honoring `>` and `>>` and `&&`, expanding globs. `ls` and `cat` and
`mkdir x && cd x && pwd` became *code*, always right, always consistent, feeding
their real output back into the model's context so the dreams that followed stayed
honest too. The consistency burden we'd carried for months didn't get fixed. It
ceased to exist. You cannot teach a falsehood to a filesystem that is simply true.

Which meant the corpus could finally forget most of itself. We went back to the
Docker container and took an *exhaustive* capture this time — every command, under
both `guest` and `root`, 534 distinct programs where the synthetic set had known
about fifty — then threw away every record for a command the code now owned. What
remained trained only what the model still dreams: `ping`, `git`, `ps`, `man`, the
fun ones, and a new drill teaching the most useful trick of all — how to say
`command not found` with dignity instead of hallucinating.

That last drill nearly sank us, and it did it by obeying the law one more time. The
first v9 came back and `kubectl` failed gracefully, exactly as designed — but so did
`ps aux`, and so did `df -h`. Real commands, answered with *command not found*. The
capture held maybe eight examples each of `ps` and `df`; the not-found drill held
twelve thousand. The model had drawn the only conclusion the data allowed: an
unfamiliar system command is one that doesn't exist. We fed it a focused diet of real
`df`/`free`/`ps`/`top` output to balance the ledger, retrained, and watched the tables
come back. Twenty-four minutes on the Mac's GPU, a number that used to be six hours.

**v9** is the mind behind the URL now. Type `ls` and it is real. Type `ping bity.dev`
and a model dreams you an address that never existed. Type `kubectl get pods` and it
tells you, correctly, that it has never heard of such a thing. The terminal that once
dreamed *everything* now dreams only the parts worth dreaming — and stands on real
ground for the rest.
