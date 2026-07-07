// The virtual shell (DESIGN §2.2, §14.3): a registry of "binaries" over one
// InferenceSession. Model binaries stream conditioned inference with per-binary
// sampling/pacing; scripted binaries are plain functions; hybrids do both.
export class Shell {
    session;
    registry = new Map();
    /** front-panel overrides (null/"stock" = per-binary settings) */
    tempOverride = null;
    pacingMode = "stock";
    /** current directory — the prompt carries it (v7); cd is a shell builtin */
    cwd = "~";
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
        session.feed(this.prompt); // prime the context with the first prompt
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
    /** cd builtin: pure string logic over a dreamed filesystem — no validation */
    applyCd(arg) {
        if (!arg || arg === "~" || arg === "$HOME")
            this.cwd = "~";
        else if (arg === "..")
            this.cwd = this.cwd.includes("/") ? this.cwd.slice(0, this.cwd.lastIndexOf("/")) || "~" : "~";
        else if (arg === ".") { /* no-op */ }
        else if (arg.startsWith("/"))
            this.cwd = arg === "/home/guest" ? "~" : arg.startsWith("/home/guest/") ? "~/" + arg.slice(12) : arg;
        else
            this.cwd = (this.cwd === "~" ? "~" : this.cwd) + "/" + arg.replace(/\/+$/, "");
    }
    register(...bins) {
        for (const b of bins)
            this.registry.set(b.name, b);
    }
    /** Run one command line. The caller displays its own prompt + echo. */
    async run(line, io) {
        const argv = line.trim().split(/\s+/).filter(Boolean);
        const ctx = { io, session: this.session, prompt: this.prompt, shell: this };
        const bin = argv.length ? this.registry.get(argv[0]) : undefined;
        // context: the typed line becomes part of the transcript the model sees
        // (a binary may rewrite it into the phrasing the corpus actually knows)
        this.session.feed((bin?.rewrite ? bin.rewrite(line) : line) + "\n");
        if (argv.length === 0)
            return;
        if (argv[0] === "cd") {
            // builtin, like a real shell: silent success, the prompt moves
            this.applyCd(argv[1]);
            this.session.feed(this.prompt);
            return;
        }
        if (bin?.kind === "scripted" && bin.run) {
            await bin.run(argv, ctx);
        }
        else {
            await this.streamModel(bin, io);
        }
        this.session.feed(this.prompt); // the shell's next prompt is model context too
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
