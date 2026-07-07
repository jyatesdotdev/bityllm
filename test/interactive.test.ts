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
  shell.register({ name: "ls", kind: "model", sampling: { temperature: 0.55, topK: 30 } });

  await shell.run("ls", countingIO().io);
  assert.equal(seen[0].temperature, 0.55); // stock: per-binary

  shell.tempOverride = 1.15; // FEVER
  await shell.run("ls", countingIO().io);
  assert.equal(seen[1].temperature, 1.15);

  shell.tempOverride = null; // back to STOCK
  await shell.run("ls", countingIO().io);
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

test("Shell with fake session: output reaches the screen, prompt stripped", async () => {
  const { session } = fakeSession("total 4\nnotes.txt\n$ ");
  const shell = new Shell(session, { prompt: "$ ", seed: 1 });
  const { io, screen } = countingIO();
  await shell.run("ls", io);
  assert.ok(screen().includes("notes.txt"));
  assert.ok(!screen().includes("$ "), "the model's own prompt must never display");
});
