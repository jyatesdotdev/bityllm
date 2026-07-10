// Tests for the programmatic core — the "real code" half of the hybrid shell.
// A virtual filesystem (vfs.ts), deterministic binaries (coreutils.ts), and a
// mini-shell executor (shell-exec.ts) run FS/text/identity commands as real,
// always-consistent code; only generative/unknown commands are dreamed. These
// cover the core in isolation plus its routing through Shell.run.

import { test } from "node:test";
import assert from "node:assert/strict";
import { VFS } from "../src/infer/vfs.ts";
import { CORE } from "../src/infer/coreutils.ts";
import type { ShellState } from "../src/infer/coreutils.ts";
import { execLine } from "../src/infer/shell-exec.ts";
import { Shell } from "../src/infer/shell.ts";
import type { SessionLike, ShellIO } from "../src/infer/shell.ts";
import type { StreamOpts } from "../src/infer/session.ts";

function newState(): ShellState {
  return {
    vfs: new VFS(), cwd: "/home/guest", user: "guest",
    env: new Map([["HOME", "/home/guest"], ["USER", "guest"], ["PWD", "/home/guest"]]),
    lastExit: 0,
  };
}
const run = (st: ShellState, line: string) => execLine(line, st);
const out = (st: ShellState, line: string) => execLine(line, st).out.trim();

test("VFS: seeded home resolves, ~ and .. normalize, missing paths are null", () => {
  const vfs = new VFS();
  assert.equal(vfs.isDir("/home/guest", "/"), true);
  assert.equal(vfs.isDir("~", "/home/guest"), true);
  assert.equal(vfs.abs("../guest/./projects", "/home/guest"), "/home/guest/projects");
  assert.equal(vfs.lookup("notes.txt", "/home/guest")?.kind, "file");
  assert.equal(vfs.lookup("nope", "/home/guest"), null);
  assert.deepEqual(vfs.list("~", "/home/guest"), ["notes.txt", "projects", "todo.md"]);
});

test("VFS: mkdir -p / write / move / copy / remove keep the tree consistent", () => {
  const vfs = new VFS();
  assert.equal(vfs.mkdir("a/b/c", "/tmp", true).ok, true);
  assert.equal(vfs.isDir("/tmp/a/b/c", "/"), true);
  vfs.writeFile("/tmp/a/x.txt", "/", "hi");
  assert.equal(vfs.move("/tmp/a/x.txt", "/tmp/a/b", "/").ok, true);
  assert.equal((vfs.lookup("/tmp/a/b/x.txt", "/") as { content: string }).content, "hi");
  assert.equal(vfs.copy("/tmp/a/b/x.txt", "/tmp/a/y.txt", "/").ok, true);
  assert.equal(vfs.remove("/tmp/a", "/", true).ok, true);
  assert.equal(vfs.exists("/tmp/a", "/"), false);
});

test("coreutils: identity + text tools produce real output", () => {
  const st = newState();
  assert.equal(CORE.whoami([], st, "").out.trim(), "guest");
  assert.equal(CORE.pwd([], st, "").out.trim(), "/home/guest");
  assert.equal(CORE.echo(["a", "b", "c"], st, "").out.trim(), "a b c");
  assert.equal(CORE.wc(["-w"], st, "one two three").out.trim().split(/\s+/)[0], "3");
  assert.equal(CORE.grep(["model"], st, "a\nfeed the model\nb").out.trim(), "feed the model");
});

test("executor: pipes thread stdin → stdout", () => {
  const st = newState();
  assert.equal(out(st, "cat notes.txt | grep model | wc -l"), "1");
  assert.equal(out(st, "echo hello | tr a-z A-Z"), "HELLO");
});

test("executor: redirects write/append to the VFS", () => {
  const st = newState();
  run(st, "echo one > f.txt");
  run(st, "echo two >> f.txt");
  assert.equal(out(st, "cat f.txt"), "one\ntwo");
  assert.equal((st.vfs.lookup("f.txt", st.cwd) as { content: string }).content, "one\ntwo\n");
});

test("executor: && / || honor exit codes; assignment + $VAR expand", () => {
  const st = newState();
  assert.equal(out(st, "true && echo yes"), "yes");
  assert.equal(out(st, "false && echo no"), "");
  assert.equal(out(st, "false || echo recovered"), "recovered");
  assert.equal(out(st, "X=42; echo $X"), "42");
});

test("executor: globs expand against the cwd", () => {
  const st = newState();
  run(st, "echo hi > a.txt");
  run(st, "echo hi > b.txt");
  // matches the two new files plus the seeded notes.txt, sorted
  assert.equal(out(st, "ls *.txt"), "a.txt\nb.txt\nnotes.txt");
});

test("executor: a non-core command dreams the whole line with no side effects", () => {
  const st = newState();
  const res = execLine("mkdir keep; ping bity.dev", st);
  assert.equal(res.dreamed, "mkdir keep; ping bity.dev");
  assert.equal(res.out, "");
  assert.equal(st.vfs.exists("keep", st.cwd), false, "no partial mutation on a dreamed line");
});

// ---- routing through the real Shell ----------------------------------------

function fakeSession(reply: string): SessionLike {
  return {
    feed: () => {}, reset: () => {}, length: 0,
    snapshot: () => ({ t: 0, c: 0 }), restore: () => {},
    *stream(_opts: StreamOpts) { for (const ch of reply) yield ch; },
  };
}
function screenIO(): { io: ShellIO; screen: () => string } {
  let s = "";
  return { io: { write: (t) => (s += t), clear: () => (s = ""), delay: async () => {} }, screen: () => s };
}

test("Shell: core commands run as real code; dreamed commands stream from the model", async () => {
  const shell = new Shell(fakeSession("DREAMED-OUTPUT"), { prompt: "guest@bity:~$ ", seed: 1 });
  const a = screenIO();
  await shell.run("whoami", a.io);
  assert.equal(a.screen().trim(), "guest", "whoami is real, not dreamed");

  const b = screenIO();
  await shell.run("cat notes.txt", b.io);
  assert.match(b.screen(), /feed the model/, "cat reads the real seeded file");
  assert.ok(!b.screen().includes("DREAMED-OUTPUT"), "the model is not consulted for core commands");

  const c = screenIO();
  await shell.run("git status", c.io); // git isn't core → dreamed
  assert.equal(c.screen(), "DREAMED-OUTPUT", "unknown/generative commands are dreamed");
});

test("Shell: cd mutates real cwd, and a real command in the new dir is consistent", async () => {
  const shell = new Shell(fakeSession(""), { prompt: "guest@bity:~$ ", seed: 1 });
  await shell.run("cd projects", screenIO().io);
  assert.equal(shell.prompt, "guest@bity:~/projects$ ");
  const s = screenIO();
  await shell.run("ls", s.io);
  assert.equal(s.screen().trim(), "README.md", "ls reflects the real cwd");
});

test("Shell.commandNames: includes core, dreamed, and registered names", () => {
  const shell = new Shell(fakeSession(""), { prompt: "$ ", seed: 1 });
  const names = shell.commandNames();
  for (const n of ["ls", "cat", "grep", "wc", "cd", "ping", "git"]) {
    assert.ok(names.includes(n), `expected ${n} in command names`);
  }
});
