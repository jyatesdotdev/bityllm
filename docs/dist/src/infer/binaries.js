// Starter binaries (DESIGN §2.2): model binaries are manifests — sampling,
// pacing, budget; the model does the rest. Scripted ones are plain functions.
// Anything NOT registered falls through to free-form model continuation, so
// every command "works"; the registry just tunes the experience.
const model = (name, extra = {}) => ({
    name,
    kind: "model",
    sampling: { temperature: 0.7, topK: 40 },
    ...extra,
});
export const BINARIES = [
    // steady, table-like output → low temperature
    model("ls", { sampling: { temperature: 0.55, topK: 30 } }),
    model("cat", { sampling: { temperature: 0.6, topK: 40 } }),
    model("pwd", { sampling: { temperature: 0.4, topK: 10 }, maxNewTokens: 40 }),
    model("whoami", { sampling: { temperature: 0.4, topK: 10 }, maxNewTokens: 20 }),
    model("uname", { sampling: { temperature: 0.5, topK: 20 }, maxNewTokens: 140 }),
    model("date", { sampling: { temperature: 0.6, topK: 20 }, maxNewTokens: 60 }),
    model("uptime", { sampling: { temperature: 0.6, topK: 20 }, maxNewTokens: 100 }),
    model("df", { sampling: { temperature: 0.55, topK: 30 } }),
    model("free", { sampling: { temperature: 0.55, topK: 30 } }),
    model("id", { sampling: { temperature: 0.5, topK: 20 }, maxNewTokens: 90 }),
    model("history", { sampling: { temperature: 0.8, topK: 40 } }),
    model("ps", { sampling: { temperature: 0.65, topK: 40 } }),
    model("git", { sampling: { temperature: 0.65, topK: 40 } }),
    model("curl", { sampling: { temperature: 0.7, topK: 40 } }),
    // the fun ones — pacing is the charm
    model("ping", { synopsis: "ping [-c count] <host>", pacing: { lineDelayMs: 900 }, maxNewTokens: 420 }),
    model("traceroute", { pacing: { lineDelayMs: 550 } }),
    model("fortune", { sampling: { temperature: 0.85, topK: 50 }, maxNewTokens: 220 }),
    model("cowsay", { synopsis: "cowsay <message>", sampling: { temperature: 0.6, topK: 30 } }),
    model("neofetch", { pacing: { lineDelayMs: 40 } }),
    model("sudo", { sampling: { temperature: 0.6, topK: 30 }, maxNewTokens: 160 }),
    // scripted
    {
        name: "help",
        kind: "scripted",
        synopsis: "help — list known commands",
        run: async (_argv, ctx) => {
            const names = ["ls", "cat", "pwd", "whoami", "uname", "date", "uptime", "df", "free", "ps",
                "git", "curl", "ping", "traceroute", "fortune", "cowsay", "neofetch", "history", "reboot", "clear", "help"];
            ctx.io.write("bity — a hallucinated terminal. binaries on this system:\n");
            ctx.io.write(names.join("  ") + "\n");
            ctx.io.write("(anything else is dreamed up on the spot)\n");
        },
    },
    {
        name: "clear",
        kind: "scripted",
        run: async (_argv, ctx) => ctx.io.clear(),
    },
    {
        // hybrid: model dreams the shutdown, script does the lifecycle.
        // The corpus only knows shutdown sequences as `sudo reboot`, so the fed
        // context is rewritten; if the dream still comes up dry, canned systemd
        // lines (from the real VM capture) keep the theater going.
        name: "reboot",
        kind: "scripted",
        synopsis: "reboot — restart the dream",
        rewrite: () => "sudo reboot",
        run: async (_argv, ctx) => {
            const { io, session } = ctx;
            let streamed = 0;
            for await (const ch of session.stream({ maxNewTokens: 380, temperature: 0.6, topK: 30, stop: [...ctx.shell.promptStops, "login: "] })) {
                io.write(ch);
                streamed++;
                if (ch === "\n")
                    await io.delay(140);
            }
            if (streamed < 40) {
                // dream was empty — scripted shutdown, byte-faithful to the VM capture
                const lines = [
                    "[  OK  ] Stopped target Multi-User System.",
                    "[  OK  ] Stopped target Login Prompts.",
                    "         Stopping ssh.service - OpenBSD Secure Shell server...",
                    "[  OK  ] Stopped target Basic System.",
                    "[  OK  ] Reached target System Reboot.",
                    "         Rebooting...",
                ];
                for (const l of lines) {
                    io.write(l + "\n");
                    await io.delay(160);
                }
            }
            await io.delay(1200);
            io.clear();
            session.reset();
            ctx.shell.cwd = "~"; // reboot forgets where you were
            io.write("Debian GNU/Linux 13 bity ttyAMA0\n\nbity login: guest\n");
            await io.delay(350);
            // no session.feed(prompt) here — Shell.run adds the post-command prompt
        },
    },
];
