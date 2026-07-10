// The virtual shell (DESIGN §2.2, §14.3): a registry of "binaries" over one
// InferenceSession. Model binaries stream conditioned inference with per-binary
// sampling/pacing; scripted binaries are plain functions; hybrids do both.
import { VFS } from "./vfs.js";
import { execLine } from "./shell-exec.js";
import { CORE, DREAMED } from "./coreutils.js";
/** Cap the real output fed back into the model's context: enough for the dream
 *  to stay coherent, not so much it floods the short context window. */
const clampCtx = (s) => (s.length > 600 ? s.slice(0, 600) : s);
export class Shell {
    /** the active inference session — swappable so a model change keeps the shell
     *  (and its knob settings/cwd) intact; see the demo's model selector */
    session;
    registry = new Map();
    /** front-panel overrides (null/"stock" = per-binary settings) */
    tempOverride = null;
    pacingMode = "stock";
    /** display cwd for the prompt ("~", "~/projects"); derived from state.cwd */
    cwd = "~";
    /** the programmatic core's world: a real virtual FS + cwd + env. Deterministic
     *  commands (ls/cat/grep/pipes/…) run against THIS as real code; only generative
     *  commands (git/ping/…/unknown) fall through to the model. */
    state;
    promptPrefix;
    staticPrompt;
    seedCounter;
    constructor(session, opts) {
        this.session = session;
        // prompts shaped "<prefix>:~$ " become location-aware; anything else is static
        const m = opts.prompt.match(/^(.+):~\$ $/);
        this.promptPrefix = m ? m[1] : null;
        this.staticPrompt = opts.prompt;
        this.seedCounter = opts.seed ?? (Date.now() & 0xffff);
        // derive the user from a "user@host" prompt prefix; default guest
        const user = this.promptPrefix?.split("@")[0] || "guest";
        const home = user === "root" ? "/root" : "/home/guest";
        this.state = {
            vfs: new VFS(),
            cwd: home,
            user,
            env: new Map([
                ["HOME", home], ["USER", user], ["SHELL", "/bin/bash"], ["PWD", home],
                ["TERM", "xterm-256color"], ["PATH", "/usr/local/bin:/usr/bin:/bin"], ["LANG", "en_US.UTF-8"],
            ]),
            lastExit: 0,
        };
        this.syncCwd();
        session.feed(this.prompt); // prime the context with the first prompt
    }
    /** Reflect the programmatic cwd into the ~-relative display path. */
    syncCwd() {
        const home = this.state.user === "root" ? "/root" : "/home/guest";
        const p = this.state.cwd;
        this.cwd = p === home ? "~" : p.startsWith(home + "/") ? "~/" + p.slice(home.length + 1) : p;
    }
    /** Reset the working directory to home (used by reboot). */
    resetToHome() {
        this.state.cwd = this.state.user === "root" ? "/root" : "/home/guest";
        this.state.env.set("PWD", this.state.cwd);
        this.syncCwd();
    }
    /** All command names the UI can complete/list: programmatic core + dreamed set
     *  + registered binaries (deduped). */
    commandNames() {
        return [...new Set([...Object.keys(CORE), ...DREAMED, ...this.registry.keys()])].sort();
    }
    get prompt() {
        return this.promptPrefix ? `${this.promptPrefix}:${this.cwd}$ ` : this.staticPrompt;
    }
    /** Stop-sequences marking the model starting a new prompt: the exact current
     *  prompt AND the persona prefix ("guest@bity:"). The model frequently emits
     *  a different path than the real cwd (deep/rare paths especially), so the
     *  invariant prefix is what actually stops generation — without it, cat-after-cd
     *  overruns into a cascade of hallucinated commands. */
    get promptStops() {
        return this.promptPrefix ? [this.prompt, `${this.promptPrefix}:`] : [this.prompt];
    }
    register(...bins) {
        for (const b of bins)
            this.registry.set(b.name, b);
    }
    /** Run one command line. The caller displays its own prompt + echo.
     *
     *  Routing (the hybrid split): scripted binaries (help/clear/reboot) do
     *  host-level things and win; then the programmatic CORE runs deterministic
     *  commands as real code over the VFS; only generative/unknown commands fall
     *  through to the model. Real output is still fed to the session so the dream
     *  context stays truthful for whatever gets dreamed next. */
    async run(line, io) {
        const trimmed = line.trim();
        const argv = trimmed.split(/\s+/).filter(Boolean);
        const ctx = { io, session: this.session, prompt: this.prompt, shell: this };
        const bin = argv.length ? this.registry.get(argv[0]) : undefined;
        // 1) scripted binaries clear the DOM / reset the session — they take precedence
        if (bin?.kind === "scripted" && bin.run) {
            this.session.feed((bin.rewrite ? bin.rewrite(line) : line) + "\n");
            await bin.run(argv, ctx);
            this.session.feed(this.prompt);
            return;
        }
        // 2) programmatic core: deterministic FS/text/identity commands + pipes,
        //    redirects, globs, $VARs, && || — all real code, always consistent
        if (trimmed) {
            const res = execLine(trimmed, this.state);
            if (!res.dreamed) {
                this.session.feed(trimmed + "\n");
                if (res.out) {
                    await this.writePaced(res.out, io);
                    this.session.feed(clampCtx(res.out)); // keep the dream context truthful
                }
                this.state.env.set("PWD", this.state.cwd);
                this.syncCwd();
                this.session.feed(this.prompt);
                return;
            }
        }
        // 3) generative/unknown → the model (a binary may rewrite the fed line into
        //    the phrasing the corpus actually knows)
        this.session.feed((bin?.rewrite ? bin.rewrite(line) : line) + "\n");
        if (argv.length === 0)
            return;
        await this.streamModel(bin, io);
        this.session.feed(this.prompt); // the shell's next prompt is model context too
    }
    /** Render deterministic output. Real commands feel instant (retro slow-mode
     *  still paces them); only dreamed output gets the streaming-typewriter charm. */
    async writePaced(text, io) {
        const charDelay = this.pacingMode === "slow" ? 8 : 0;
        for (const ch of text) {
            io.write(ch);
            if (charDelay > 0)
                await io.delay(charDelay);
        }
    }
    /** Stream model output with pacing, holding back a possible next-prompt. */
    async streamModel(bin, io) {
        const stops = this.promptStops;
        const guard = this.prompt.length; // longest stop → safe holdback window
        const cutMark = this.promptPrefix ? `${this.promptPrefix}:` : this.prompt;
        const opts = {
            maxNewTokens: bin?.maxNewTokens ?? 512,
            temperature: this.tempOverride ?? bin?.sampling?.temperature ?? 0.7,
            topK: bin?.sampling?.topK ?? 40,
            stop: stops,
            seed: this.seedCounter++,
        };
        let charDelay = bin?.pacing?.charDelayMs ?? 2;
        let lineDelay = bin?.pacing?.lineDelayMs ?? 0;
        if (this.pacingMode === "turbo") {
            charDelay = 0;
            lineDelay = 0;
        }
        else if (this.pacingMode === "slow") {
            charDelay = 8; // ~1200 baud
        }
        // Hold back up to `guard` chars: the stream ends the moment a stop appears,
        // so any next-prompt lives in the buffer's tail — cut from the persona
        // prefix onward and nothing of the model's own prompt is displayed.
        let buf = "";
        for await (const ch of this.session.stream(opts)) {
            buf += ch;
            while (buf.length > guard) {
                const flush = buf[0];
                buf = buf.slice(1);
                io.write(flush);
                if (charDelay > 0)
                    await io.delay(charDelay);
                if (flush === "\n" && lineDelay > 0)
                    await io.delay(lineDelay);
            }
        }
        const cut = buf.lastIndexOf(cutMark);
        io.write(cut >= 0 ? buf.slice(0, cut) : buf);
    }
}
/** Tab-completion over registered command names (first word only). */
export function completeCommand(line, names) {
    if (line.length === 0 || line.includes(" "))
        return { kind: "none", text: "", options: [] };
    const matches = names.filter((n) => n.startsWith(line)).sort();
    if (matches.length === 0)
        return { kind: "none", text: "", options: [] };
    if (matches.length === 1)
        return { kind: "complete", text: matches[0] + " ", options: matches };
    let p = matches[0];
    for (const m of matches) {
        while (!m.startsWith(p))
            p = p.slice(0, -1);
    }
    if (p.length > line.length)
        return { kind: "extend", text: p, options: matches };
    return { kind: "list", text: "", options: matches };
}
/** Bash-style history cursor: up/down navigation with a saved in-progress line. */
export class History {
    items = [];
    idx = 0;
    saved = "";
    push(line) {
        if (line.trim())
            this.items.push(line);
        this.idx = this.items.length;
        this.saved = "";
    }
    /** returns the line to display, or null if navigation hit an edge */
    up(current) {
        if (this.idx === 0)
            return null;
        if (this.idx === this.items.length)
            this.saved = current;
        this.idx--;
        return this.items[this.idx];
    }
    down() {
        if (this.idx >= this.items.length)
            return null;
        this.idx++;
        return this.idx === this.items.length ? this.saved : this.items[this.idx];
    }
}
