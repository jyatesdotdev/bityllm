// The virtual shell (DESIGN §2.2, §14.3): a registry of "binaries" over one
// InferenceSession. Model binaries stream conditioned inference with per-binary
// sampling/pacing; scripted binaries are plain functions; hybrids do both.
export class Shell {
    session;
    prompt;
    registry = new Map();
    seedCounter;
    constructor(session, opts) {
        this.session = session;
        this.prompt = opts.prompt;
        this.seedCounter = opts.seed ?? (Date.now() & 0xffff);
        session.feed(this.prompt); // prime the context with the first prompt
    }
    register(...bins) {
        for (const b of bins)
            this.registry.set(b.name, b);
    }
    /** Run one command line. The caller displays its own prompt + echo. */
    async run(line, io) {
        const argv = line.trim().split(/\s+/).filter(Boolean);
        const ctx = { io, session: this.session, prompt: this.prompt };
        const bin = argv.length ? this.registry.get(argv[0]) : undefined;
        // context: the typed line becomes part of the transcript the model sees
        // (a binary may rewrite it into the phrasing the corpus actually knows)
        this.session.feed((bin?.rewrite ? bin.rewrite(line) : line) + "\n");
        if (argv.length === 0)
            return;
        if (bin?.kind === "scripted" && bin.run) {
            await bin.run(argv, ctx);
        }
        else {
            await this.streamModel(bin, io);
        }
        this.session.feed(this.prompt); // the shell's next prompt is model context too
    }
    /** Stream model output with pacing, holding back a possible stop-sequence. */
    async streamModel(bin, io) {
        const stop = this.prompt;
        const opts = {
            maxNewTokens: bin?.maxNewTokens ?? 512,
            temperature: bin?.sampling?.temperature ?? 0.7,
            topK: bin?.sampling?.topK ?? 40,
            stop: [stop],
            seed: this.seedCounter++,
        };
        const charDelay = bin?.pacing?.charDelayMs ?? 2;
        const lineDelay = bin?.pacing?.lineDelayMs ?? 0;
        // Hold back exactly stop.length chars: the stream ends the moment the stop
        // string is fully emitted, so the stop can only ever be the buffer's tail —
        // strip it there and nothing of the model's own prompt is displayed.
        let buf = "";
        for await (const ch of this.session.stream(opts)) {
            buf += ch;
            while (buf.length > stop.length) {
                const flush = buf[0];
                buf = buf.slice(1);
                io.write(flush);
                if (charDelay > 0)
                    await io.delay(charDelay);
                if (flush === "\n" && lineDelay > 0)
                    await io.delay(lineDelay);
            }
        }
        if (buf.endsWith(stop))
            buf = buf.slice(0, -stop.length);
        io.write(buf);
    }
}
