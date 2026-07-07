// The bity virtual terminal (DESIGN §2, M5): loads the int8 checkpoint, wires
// the Shell + binaries to a DOM screen, and lets you type into the dream.
//
// Interactive niceties: bash-style history (↑/↓), Tab-completion, and
// fish-style ghost autosuggestions dreamed by the model via a speculative
// KV-cache peek (snapshot → feed → stream → restore). Completion/history
// logic lives in src/infer/shell.ts and is unit-tested; this file is DOM glue.
import { deserialize, InferenceSession, GPUInferenceSession, Shell, BINARIES, completeCommand, History } from "../../src/infer.js";
const PROMPT = "guest@bity:~$ ";
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
async function main() {
    io.write("loading model");
    const tick = setInterval(() => io.write("."), 180);
    // page-relative first (GitHub Pages: docs/terminal.int8.bity), repo-root fallback (local dev)
    let res = await fetch("terminal.int8.bity");
    if (!res.ok)
        res = await fetch("/models/terminal.int8.bity");
    if (!res.ok) {
        clearInterval(tick);
        io.write(`\nfailed to load terminal.int8.bity (${res.status})\nrun: node examples/export-int8.ts\n`);
        return;
    }
    const bytes = new Uint8Array(await res.arrayBuffer());
    const { model, tok, step } = deserialize(bytes);
    // engine selection: race WebGPU vs CPU for 24 tokens, keep the winner
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
    let session = new InferenceSession(model, tok);
    let engine = "cpu (pure TS)";
    try {
        const gpuS = await GPUInferenceSession.create(model, tok);
        const gRate = await race(gpuS);
        const cRate = await race(session);
        if (gRate > cRate) {
            session = gpuS;
            engine = `webgpu ${gRate.toFixed(0)} tok/s (cpu: ${cRate.toFixed(0)})`;
        }
        else {
            gpuS.destroy();
            engine = `cpu ${cRate.toFixed(0)} tok/s (webgpu: ${gRate.toFixed(0)})`;
        }
    }
    catch {
        /* no WebGPU in this browser — CPU it is */
    }
    clearInterval(tick);
    io.clear();
    io.write(`bity 0.1 — a hallucinated terminal (${(bytes.length / 1024).toFixed(0)} KB model, ${model.paramCount().toLocaleString()} params, step ${step}, engine: ${engine})\n`);
    io.write(`nothing below this line is real. type 'help' for the binaries it dreams best.\n\n`);
    const shell = new Shell(session, { prompt: PROMPT });
    shell.register(...BINARIES);
    const commandNames = [...shell.registry.keys()].sort();
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
    io.write(PROMPT);
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
                        session.feed(PROMPT);
                        io.write("bity login: guest\n\n" + PROMPT);
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
            io.write(PROMPT);
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
                io.write("\n" + c.options.join("  ") + "\n" + PROMPT + line);
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
