// Tests for the interactive-shell features behind the demo UI: tab completion,
// history navigation, and the front-panel Shell overrides (TEMP/BAUD knobs).
// The DOM layer stays thin glue; everything with logic is covered here.

import { test } from "node:test";
import assert from "node:assert/strict";
import { completeCommand, History, Shell } from "../src/infer/shell.ts";
import type { SessionLike, ShellIO } from "../src/infer/shell.ts";
import type { StreamOpts } from "../src/infer/session.ts";

const NAMES = ["cat", "clear", "cowsay", "curl", "date", "df", "fortune", "free", "help", "history", "ls", "neofetch", "ping", "ps", "pwd", "reboot", "traceroute", "uname", "uptime", "whoami"];

test("completeCommand: unique, extend, list, none", () => {
  // unique → full completion with trailing space
  assert.deepEqual(completeCommand("nefetch".slice(0, 2), NAMES).text, "neofetch ");
  assert.equal(completeCommand("pi", NAMES).kind, "complete");
  assert.equal(completeCommand("pi", NAMES).text, "ping ");

  // shared prefix → extend to longest common prefix (c → "c", no progress → list)
  const c1 = completeCommand("c", NAMES);
  assert.equal(c1.kind, "list"); // cat/clear/cowsay/curl share only "c"
  assert.deepEqual(c1.options, ["cat", "clear", "cowsay", "curl"]);
  const c2 = completeCommand("co", NAMES);
  assert.equal(c2.kind, "complete");
  assert.equal(c2.text, "cowsay ");
  // "up" extends nothing beyond... uptime unique
  assert.equal(completeCommand("up", NAMES).text, "uptime ");

  // no match / has args / empty → none
  assert.equal(completeCommand("zz", NAMES).kind, "none");
  assert.equal(completeCommand("ping bity", NAMES).kind, "none");
  assert.equal(completeCommand("", NAMES).kind, "none");
});

test("completeCommand: extend to common prefix when it makes progress", () => {
  const names = ["gitlog", "gitstatus"];
  const c = completeCommand("g", names);
  assert.equal(c.kind, "extend");
  assert.equal(c.text, "git");
});

test("History: up/down navigation with saved in-progress line", () => {
  const h = new History();
  h.push("ls");
  h.push("ping bity.dev");
  h.push("   "); // blank: not recorded
  assert.equal(h.up("draft"), "ping bity.dev"); // saves "draft"
  assert.equal(h.up("x"), "ls");
  assert.equal(h.up("x"), null);                // top edge
  assert.equal(h.down(), "ping bity.dev");
  assert.equal(h.down(), "draft");              // restores the saved line
  assert.equal(h.down(), null);                 // bottom edge
});

// ---- Shell knob overrides, observed through a fake session ----

function fakeSession(reply: string): { session: SessionLike; seen: StreamOpts[] } {
  const seen: StreamOpts[] = [];
  const session: SessionLike = {
    feed: () => {},
    reset: () => {},
    length: 0,
    snapshot: () => ({ t: 0, c: 0 }),
    restore: () => {},
    *stream(opts: StreamOpts) {
      seen.push(opts);
      for (const ch of reply) yield ch;
    },
  };
  return { session, seen };
}

function countingIO(): { io: ShellIO; delays: number[]; screen: () => string } {
  let out = "";
  const delays: number[] = [];
  return {
    io: {
      write: (s) => (out += s),
      clear: () => (out = ""),
      delay: async (ms) => {
        delays.push(ms);
      },
    },
    delays,
    screen: () => out,
  };
}

test("Shell.tempOverride: knob overrides per-binary temperature; STOCK restores it", async () => {
  const { session, seen } = fakeSession("ok\n");
  const shell = new Shell(session, { prompt: "$ ", seed: 1 });
  // a dreamed (non-core) command actually consults the model
  shell.register({ name: "ping", kind: "model", sampling: { temperature: 0.55, topK: 30 } });

  await shell.run("ping x", countingIO().io);
  assert.equal(seen[0].temperature, 0.55); // stock: per-binary

  shell.tempOverride = 1.15; // FEVER
  await shell.run("ping x", countingIO().io);
  assert.equal(seen[1].temperature, 1.15);

  shell.tempOverride = null; // back to STOCK
  await shell.run("ping x", countingIO().io);
  assert.equal(seen[2].temperature, 0.55);
});

test("Shell.pacingMode: turbo strips all delays, slow forces ~1200 baud", async () => {
  const mk = (): Shell => {
    const { session } = fakeSession("line one\nline two\n");
    const shell = new Shell(session, { prompt: "$ ", seed: 1 });
    shell.register({ name: "ping", kind: "model", pacing: { lineDelayMs: 900 } });
    return shell;
  };

  // stock: per-char delay (2ms) + per-line delay (900ms)
  const a = countingIO();
  await mk().run("ping x", a.io);
  assert.ok(a.delays.includes(900), "stock keeps the binary's line pacing");
  assert.ok(a.delays.includes(2), "stock keeps char pacing");

  // turbo: no delays at all
  const b = countingIO();
  const st = mk();
  st.pacingMode = "turbo";
  await st.run("ping x", b.io);
  assert.equal(b.delays.length, 0);

  // slow: ~1200 baud char pacing
  const c = countingIO();
  const ss = mk();
  ss.pacingMode = "slow";
  await ss.run("ping x", c.io);
  assert.ok(c.delays.includes(8), "slow forces 8ms char delay");
});

test("Shell with fake session: dreamed output reaches the screen, prompt stripped", async () => {
  const { session } = fakeSession("total 4\nnotes.txt\n$ ");
  const shell = new Shell(session, { prompt: "$ ", seed: 1 });
  const { io, screen } = countingIO();
  await shell.run("ping x", io); // dreamed command → streams from the model
  assert.ok(screen().includes("notes.txt"));
  assert.ok(!screen().includes("$ "), "the model's own prompt must never display");
});

test("Shell cd builtin: VFS-validated navigation, location-aware prompt", async () => {
  const { session } = fakeSession("x\n");
  const shell = new Shell(session, { prompt: "guest@bity:~$ ", seed: 1 });
  const io = () => countingIO().io;

  assert.equal(shell.prompt, "guest@bity:~$ ");
  await shell.run("cd projects", io());            // real seeded dir
  assert.equal(shell.prompt, "guest@bity:~/projects$ ");
  await shell.run("cd ..", io());
  assert.equal(shell.prompt, "guest@bity:~$ ");
  await shell.run("cd .config", io());             // real (dotfile) dir
  assert.equal(shell.prompt, "guest@bity:~/.config$ ");
  await shell.run("cd", io());                      // bare cd → home
  assert.equal(shell.prompt, "guest@bity:~$ ");
  await shell.run("cd /home/guest/projects", io()); // absolute path
  assert.equal(shell.prompt, "guest@bity:~/projects$ ");

  // cd into a missing dir fails like real bash: prompt unchanged, error shown
  const { io: eio, screen } = countingIO();
  await shell.run("cd nope", eio);
  assert.equal(shell.prompt, "guest@bity:~/projects$ ", "a failed cd does not move");
  assert.match(screen(), /No such file or directory/);
});

test("Shell static prompt stays static (cd still navigates silently)", async () => {
  const { session, seen } = fakeSession("x\n");
  const shell = new Shell(session, { prompt: "$ ", seed: 1 });
  await shell.run("cd somewhere", (countingIO()).io);
  assert.equal(shell.prompt, "$ "); // non-templated prompt unchanged
  assert.equal(seen.length, 0);     // cd is a builtin: the model is never consulted
});

// A fake session that honors opts.stop exactly like the real InferenceSession,
// so we can drive the shell's next-prompt detection deterministically.
function stoppingSession(reply: string): SessionLike {
  return {
    feed: () => {},
    reset: () => {},
    length: 0,
    snapshot: () => ({ t: 0, c: 0 }),
    restore: () => {},
    *stream(opts: StreamOpts) {
      let out = "";
      const maxStop = Math.max(0, ...(opts.stop ?? []).map((s) => s.length));
      for (const ch of reply) {
        out += ch;
        if (out.length > maxStop * 4) out = out.slice(-maxStop * 2);
        yield ch;
        if (opts.stop?.some((s) => s.length && out.endsWith(s))) return;
      }
    },
  };
}

test("dreamed command after cd stops at the next prompt even when the model dreams a different path", async () => {
  // in ~/projects the real prompt is guest@bity:~/projects$, but the model emits
  // the far-more-common HOME prompt after the output — the exact-prompt stop
  // would miss it and overrun into the extra hallucinated commands.
  const reply = "commit a1b2c3  fix\nguest@bity:~$ ls\nnotes.txt\nguest@bity:~$ ";
  const shell = new Shell(stoppingSession(reply), { prompt: "guest@bity:~$ ", seed: 1 });
  shell.register({ name: "git", kind: "model" });

  await shell.run("cd projects", countingIO().io); // real seeded dir
  assert.equal(shell.prompt, "guest@bity:~/projects$ ");

  const { io, screen } = countingIO();
  await shell.run("git log", io); // dreamed → streams from the model
  assert.ok(screen().includes("commit a1b2c3  fix"), "shows the actual command output");
  assert.ok(!screen().includes("notes.txt"), "must NOT run the model's extra hallucinated commands");
  assert.ok(!screen().includes("guest@bity:"), "must not leak any prompt text");
});

test("promptStops: exact prompt + persona prefix (dynamic); static prompt has just itself", () => {
  const dyn = new Shell(stoppingSession(""), { prompt: "guest@bity:~$ ", seed: 1 });
  dyn.cwd = "~/deep/path";
  assert.deepEqual(dyn.promptStops, ["guest@bity:~/deep/path$ ", "guest@bity:"]);
  const stat = new Shell(stoppingSession(""), { prompt: "$ ", seed: 1 });
  assert.deepEqual(stat.promptStops, ["$ "]);
});
