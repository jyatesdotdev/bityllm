// The bity virtual terminal (DESIGN §2, M5): loads the int8 checkpoint, wires
// the Shell + binaries to a DOM screen, and lets you type into the dream.
//
// Interactive niceties: bash-style history (↑/↓), Tab-completion, and
// fish-style ghost autosuggestions dreamed by the model via a speculative
// KV-cache peek (snapshot → feed → stream → restore). Completion/history
// logic lives in src/infer/shell.ts and is unit-tested; this file is DOM glue.
import { deserialize, InferenceSession, GPUInferenceSession, Shell, BINARIES, completeCommand, History } from "../../src/infer.js";
const PROMPT = "guest@bity:~$ ";
// the MODEL knob sweeps these — MINI is the hybrid-corpus v9 (real code owns
// FS/text, model dreams the rest); the other sizes are still corpus v8
const MODELS = [
    { label: "MICRO", note: "2.7M", ver: "v8", file: "terminal-micro-v8.int8.bity" },
    { label: "MINI", note: "10.7M", ver: "v9", file: "terminal.int8.bity" },
    { label: "MAX", note: "25M", ver: "v8", file: "terminal-25m-v8.int8.bity" },
    { label: "ULTRA", note: "57M", ver: "v8", file: "terminal-ultra-v8.int8.bity" }, // wide → WebGPU wins here
];
const DEFAULT_MODEL = 1; // MINI (the deployed default)
const textEl = document.getElementById("text");
const ghostEl = document.getElementById("ghost");
const crt = document.getElementById("crt");
const io = {
    write(s) {
        textEl.textContent += s;
        crt.scrollTop = crt.scrollHeight;
    },
    clear() {
        textEl.textContent = "";
    },
    delay: (ms) => new Promise((r) => setTimeout(r, ms)),
};
const erase = (n) => {
    textEl.textContent = textEl.textContent.slice(0, textEl.textContent.length - n);
};
// serialize all session use: a running command and a speculative ghost peek
// must never interleave on the shared KV-cache
let lock = Promise.resolve();
function withLock(fn) {
    const r = lock.then(fn);
    lock = r.catch(() => { });
    return r;
}
async function loadModel(file) {
    let res = await fetch(file);
    if (!res.ok)
        res = await fetch("/models/" + file); // repo-root fallback (local dev)
    if (!res.ok)
        return null;
    const buf = new Uint8Array(await res.arrayBuffer());
    const { model, tok, step } = deserialize(buf);
    const race = async (s) => {
        s.feed("guest@bity:~$ ls\n");
        const t0 = performance.now();
        let n = 0;
        for await (const _ of s.stream({ maxNewTokens: 24, temperature: 0.8, seed: 1 }))
            n++;
        const rate = n / ((performance.now() - t0) / 1000);
        s.reset();
        return rate;
    };
    // untimed warmup: pay WebGPU's one-time shader/pipeline compilation (and prime
    // CPU caches) BEFORE the timed race, so the GPU isn't penalized for cold-start.
    const warm = async (s) => {
        s.feed("guest@bity:~$ ls\n");
        for await (const _ of s.stream({ maxNewTokens: 8, temperature: 0.8, seed: 1 })) { /* discard */ }
        s.reset();
    };
    let session = new InferenceSession(model, tok);
    let engine = "cpu (pure TS)";
    try {
        const gpuS = await GPUInferenceSession.create(model, tok);
        await warm(gpuS);
        await warm(session);
        const gRate = await race(gpuS);
        const cRate = await race(session);
        if (gRate > cRate) {
            session = gpuS;
            engine = `webgpu ${gRate.toFixed(0)} tok/s (cpu ${cRate.toFixed(0)})`;
        }
        else {
            gpuS.destroy();
            engine = `cpu ${cRate.toFixed(0)} tok/s (webgpu ${gRate.toFixed(0)})`;
        }
    }
    catch { /* no WebGPU in this browser — CPU it is */ }
    return { session, engine, model, step, kb: Math.round(buf.length / 1024) };
}
const banner = (m, l) => `bity 0.1 · model: ${m.label} (${m.note} params, ${m.ver}, step ${l.step}) · ${l.kb} KB int8 · engine: ${l.engine}\n` +
    `nothing below this line is real. type 'help' for the binaries it dreams best. turn the MODEL knob to swap brains.\n\n`;
async function main() {
    const saved = parseInt(localStorage.getItem("bity.model") ?? "", 10);
    let modelIdx = saved >= 0 && saved < MODELS.length ? saved : DEFAULT_MODEL;
    io.write("loading model");
    const tick = setInterval(() => io.write("."), 180);
    let loaded = await loadModel(MODELS[modelIdx].file);
    if (!loaded && modelIdx !== DEFAULT_MODEL) {
        modelIdx = DEFAULT_MODEL;
        loaded = await loadModel(MODELS[modelIdx].file);
    }
    clearInterval(tick);
    if (!loaded) {
        io.write(`\nfailed to load ${MODELS[modelIdx].file}\nrun: node examples/export-int8.ts\n`);
        return;
    }
    io.clear();
    let session = loaded.session;
    let model = loaded.model;
    io.write(banner(MODELS[modelIdx], loaded));
    const shell = new Shell(session, { prompt: PROMPT });
    shell.register(...BINARIES);
    const commandNames = shell.commandNames(); // core + dreamed + registered
    const history = new History();
    let line = "";
    let busy = false;
    let powered = true;
    let suggestOn = false; // SUGGEST knob — off by default (it's a young circuit)
    // ---- ghost autosuggestions: the model dreams your next keystrokes ----
    let ghostText = "";
    let suggestGen = 0;
    let suggestTimer;
    let peekInFlight = false;
    const setGhost = (s) => {
        ghostText = s;
        ghostEl.textContent = s;
    };
    const computeGhost = () => {
        if (!suggestOn || !powered || busy || line.length === 0)
            return;
        if (peekInFlight)
            return; // the in-flight peek re-checks freshness when done
        if (session.length + line.length + 40 >= model.cfg.blockSize)
            return; // rewind hazard
        const gen = ++suggestGen;
        const forLine = line;
        peekInFlight = true;
        void withLock(async () => {
            try {
                if (gen !== suggestGen || busy || !powered)
                    return;
                const snap = session.snapshot();
                try {
                    session.feed(forLine);
                    let out = "";
                    let i = 0;
                    for await (const ch of session.stream({ maxNewTokens: 24, temperature: 0.2, topK: 3, stop: ["\n"], seed: 2 })) {
                        out += ch;
                        // macrotask yield every few tokens: CPU-engine steps are synchronous
                        // and would otherwise starve keyboard events (typing jank)
                        if (++i % 4 === 0)
                            await new Promise((r) => setTimeout(r));
                        if (gen !== suggestGen)
                            break;
                    }
                    out = out.split("\n")[0];
                    if (gen === suggestGen && !busy && powered && out.length > 0)
                        setGhost(out);
                }
                finally {
                    session.restore(snap);
                }
            }
            finally {
                peekInFlight = false;
                // typed more while we were dreaming? dream again for the fresh prefix
                if (gen !== suggestGen && suggestOn && powered && !busy && line.length > 0) {
                    clearTimeout(suggestTimer);
                    suggestTimer = setTimeout(computeGhost, 60);
                }
            }
        });
    };
    const scheduleGhost = () => {
        setGhost("");
        suggestGen++;
        clearTimeout(suggestTimer);
        if (suggestOn)
            suggestTimer = setTimeout(computeGhost, 220);
    };
    const replaceLine = (nl) => {
        erase(line.length);
        line = nl;
        io.write(line);
        scheduleGhost();
    };
    io.write(shell.prompt);
    // ---- model selector: swap the brain in place, keeping the shell + knob settings ----
    let switching = false;
    async function switchModel(idx) {
        if (idx === modelIdx || switching)
            return;
        switching = true;
        await withLock(async () => {
            suggestGen++;
            setGhost("");
            line = "";
            io.clear();
            io.write(`tuning to ${MODELS[idx].label}`);
            const t = setInterval(() => io.write("."), 180);
            const next = await loadModel(MODELS[idx].file);
            clearInterval(t);
            if (!next) {
                io.clear();
                io.write(`could not load ${MODELS[idx].label} (${MODELS[idx].file})\n${shell.prompt}`);
                return;
            }
            shell.session.destroy?.(); // free the old GPU session
            shell.session = next.session;
            shell.cwd = "~";
            session = next.session;
            model = next.model;
            modelIdx = idx;
            localStorage.setItem("bity.model", String(idx));
            shell.session.feed(shell.prompt);
            io.clear();
            io.write(banner(MODELS[idx], next));
            io.write(shell.prompt);
        });
        switching = false;
    }
    const knob = (id, positions, start, persist) => {
        const el = document.getElementById(id);
        const dial = el.querySelector(".dial");
        const val = el.querySelector(".kval");
        let idx = start;
        if (persist) {
            const saved = parseInt(localStorage.getItem(persist) ?? "", 10);
            if (saved >= 0 && saved < positions.length)
                idx = saved;
        }
        const render = () => {
            const pos = positions[idx];
            dial.style.setProperty("--rot", `${pos.rot}deg`);
            dial.style.transform = `rotate(${pos.rot}deg)`;
            val.textContent = pos.label;
            el.classList.toggle("on", !!pos.led);
            pos.apply();
        };
        el.addEventListener("click", () => {
            idx = (idx + 1) % positions.length;
            if (persist)
                localStorage.setItem(persist, String(idx));
            render();
        });
        render();
    };
    // MODEL: cycles the size lineup (micro → mini → max), swapping the brain in place
    knob("k-model", MODELS.map((m, i) => ({
        label: m.label, rot: -42 + i * (84 / (MODELS.length - 1)), led: true,
        apply: () => { void switchModel(i); },
    })), modelIdx, "bity.model");
    let everBooted = false;
    knob("k-power", [
        {
            label: "ON", rot: 30, led: true,
            apply: () => {
                crt.classList.remove("off");
                powered = true;
                if (everBooted) {
                    // cold boot QUEUED behind any running command — never reset a
                    // session mid-stream
                    void withLock(async () => {
                        suggestGen++;
                        setGhost("");
                        line = "";
                        io.clear();
                        session.reset();
                        shell.cwd = "~";
                        session.feed(shell.prompt);
                        io.write("bity login: guest\n\n" + shell.prompt);
                    });
                }
                everBooted = true;
            },
        },
        {
            label: "OFF", rot: -30,
            apply: () => {
                powered = false;
                suggestGen++;
                setGhost("");
                crt.classList.add("off");
            },
        },
    ], 0);
    knob("k-suggest", [
        { label: "OFF", rot: -30, apply: () => { suggestOn = false; suggestGen++; setGhost(""); } },
        { label: "ON", rot: 30, led: true, apply: () => { suggestOn = true; scheduleGhost(); } },
    ], 0, "bity.suggest");
    knob("k-temp", [
        { label: "CHILL", rot: -40, apply: () => { shell.tempOverride = 0.45; } },
        { label: "STOCK", rot: 0, apply: () => { shell.tempOverride = null; } },
        { label: "FEVER", rot: 40, apply: () => { shell.tempOverride = 1.15; } },
    ], 1, "bity.temp");
    knob("k-baud", [
        { label: "1200", rot: -40, apply: () => { shell.pacingMode = "slow"; } },
        { label: "9600", rot: 0, apply: () => { shell.pacingMode = "stock"; } },
        { label: "TURBO", rot: 40, apply: () => { shell.pacingMode = "turbo"; } },
    ], 1, "bity.baud");
    knob("k-fx", [
        { label: "ON", rot: 30, led: true, apply: () => document.body.classList.remove("nofx") },
        { label: "OFF", rot: -30, apply: () => document.body.classList.add("nofx") },
    ], 0, "bity.fx");
    document.addEventListener("keydown", async (e) => {
        if (!powered || busy || e.metaKey || e.ctrlKey || e.altKey)
            return;
        if (e.key === "Enter") {
            e.preventDefault();
            suggestGen++;
            setGhost("");
            io.write("\n");
            const cmd = line;
            history.push(cmd);
            line = "";
            busy = true;
            try {
                await withLock(() => shell.run(cmd, io));
            }
            catch (err) {
                io.write(`\n[bity kernel panic: ${err instanceof Error ? err.message : String(err)}]\n`);
            }
            busy = false;
            io.write(shell.prompt);
        }
        else if (e.key === "Backspace") {
            e.preventDefault();
            if (line.length > 0) {
                line = line.slice(0, -1);
                erase(1);
                scheduleGhost();
            }
        }
        else if (e.key === "Tab") {
            e.preventDefault();
            const c = completeCommand(line, commandNames);
            if (c.kind === "complete" || c.kind === "extend")
                replaceLine(c.text);
            else if (c.kind === "list")
                io.write("\n" + c.options.join("  ") + "\n" + shell.prompt + line);
        }
        else if (e.key === "ArrowRight" || e.key === "End") {
            if (ghostText) {
                e.preventDefault();
                line += ghostText;
                io.write(ghostText);
                setGhost("");
                scheduleGhost();
            }
        }
        else if (e.key === "ArrowUp") {
            e.preventDefault();
            const h = history.up(line);
            if (h !== null)
                replaceLine(h);
        }
        else if (e.key === "ArrowDown") {
            e.preventDefault();
            const h = history.down();
            if (h !== null)
                replaceLine(h);
        }
        else if (e.key === "Escape") {
            suggestGen++;
            setGhost("");
        }
        else if (e.key.length === 1) {
            e.preventDefault();
            // typing a char that matches the ghost consumes it instead of recomputing
            if (ghostText && ghostText[0] === e.key) {
                ghostText = ghostText.slice(1);
                ghostEl.textContent = ghostText;
                line += e.key;
                io.write(e.key);
                return;
            }
            line += e.key;
            io.write(e.key);
            scheduleGhost();
        }
    });
}
void main();
