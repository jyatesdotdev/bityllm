# LOG

Running log of noteworthy work; newest first.

## 2026-07-09 01:05 — Find the corpus-diversity scaling ceiling (~57M overfits; 25M is the sweet spot)
- **What:** The overnight 57M model (8L/12H/768d, 40k steps) hit the project's **first positive train/val gap** — final: train 0.337, **val 0.411, gap +0.073**. Every prior model (10.7M, 25M) had a *negative* gap. Critically the 57M's val loss (0.411) is **worse than the 25M's (0.378)** — it fits train better but *generalizes worse*.
- **Why:** At 2.25× the 25M's params, the model fits the training sample beyond what generalizes. The corpus is 53 MB but drawn from ~7 generator families — high *volume*, bounded *diversity* — so a 57M brain outgrows the data's information content.
- **Impact:** ~25M is the sweet spot for this corpus; **more parameters make a *worse*-generalizing model** here — the next scaling lever is corpus **diversity** (new generator families / richer content), not params or volume. Bonus: the 57M *did* hit its engineering goal — CPU drops to 36 tok/s while the dispatch-bound GPU stays ~48, so **WebGPU wins** for it in-browser (wired in as the opt-in ULTRA slot, 54.7 MB).

## 2026-07-08 21:42 — Train wide 57M model overnight to make WebGPU win in-browser
- **What:** Launched an overnight MLX run — 8L/12H/**768-dim ≈ 57M params**, enlarged 53.5MB v8 corpus, 40k steps (~4.5h on the Mac GPU). Smoke-tested clean first (params confirmed, trains, no NaN).
- **Why:** In-browser token generation is **dispatch-bound** — ~100 tiny kernel launches/token at ~200µs each dominate (the actual matmul math is microseconds), so WebGPU loses to pure-TS CPU for small models. Key lever: the **GPU time is flat vs model width** (fixed dispatch count) while the **CPU time scales with compute** — so *widening* (not deepening — depth adds dispatches) makes the CPU slower while the GPU stays ~48 tok/s, tipping the engine race to WebGPU.
- **Impact:** When done → export int8 (~57MB) as an opt-in **ULTRA** slot in the demo's MODEL knob; WebGPU should beat CPU (~48 vs ~22 tok/s), so the GPU finally gets used. Caveat: "GPU wins" ≠ "GPU fast" — still dispatch-bound; **kernel fusion** (fewer, bigger dispatches) is the separate fix for actual GPU speed.
