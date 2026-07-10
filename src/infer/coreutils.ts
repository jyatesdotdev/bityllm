// coreutils.ts — deterministic binaries that run against the VFS (not the model).
// Each is a pure function (args, state, stdin) -> { out, exit }. The mini-shell
// (shell-exec.ts) parses the line, expands vars/globs, and pipes stdin between them.

import { VFS, baseName } from "./vfs.ts";
import type { VNode, VFile } from "./vfs.ts";

export interface ShellState { vfs: VFS; cwd: string; user: string; env: Map<string, string>; lastExit: number; }
export interface Output { out: string; exit: number; }
export type CoreBin = (args: string[], st: ShellState, stdin: string) => Output;

const ok = (out = ""): Output => ({ out, exit: 0 });
const fail = (msg: string, code = 1): Output => ({ out: msg + "\n", exit: code });
const home = (st: ShellState) => (st.user === "root" ? "/root" : "/home/guest");
const short = (p: string, st: ShellState) => (p === home(st) ? "~" : p.startsWith(home(st) + "/") ? "~" + p.slice(home(st).length) : p);
const nonFlags = (args: string[]) => args.filter((a) => !a.startsWith("-"));
const hasFlag = (args: string[], ...fs: string[]) => args.some((a) => a.startsWith("-") && !a.startsWith("--") && fs.some((f) => a.includes(f)) || fs.includes(a));

function modeStr(node: VNode): string {
  const t = node.kind === "dir" ? "d" : node.link ? "l" : "-";
  const m = node.mode, b = (v: number, c: string) => (m & v ? c : "-");
  return t + b(0o400, "r") + b(0o200, "w") + b(0o100, "x") + b(0o040, "r") + b(0o020, "w") + b(0o010, "x") + b(0o004, "r") + b(0o002, "w") + b(0o001, "x");
}
const DATE = "Jul  9 17:30";
const sizeOf = (n: VNode) => (n.kind === "dir" ? 4096 : (n as VFile).content.length);

function longRow(name: string, node: VNode): string {
  const links = node.kind === "dir" ? node.children.size + 2 : 1;
  return `${modeStr(node)} ${links} ${node.owner.padEnd(5)} ${node.group.padEnd(5)} ${String(sizeOf(node)).padStart(5)} ${DATE} ${name}`;
}

export const CORE: Record<string, CoreBin> = {
  pwd: (_a, st) => ok(st.cwd + "\n"),
  echo: (a, st) => ok((a.includes("-n") ? a.filter((x) => x !== "-n").join(" ") : a.join(" ")) + (a.includes("-n") ? "" : "\n")),
  whoami: (_a, st) => ok(st.user + "\n"),
  id: (_a, st) => ok(st.user === "root"
    ? "uid=0(root) gid=0(root) groups=0(root)\n"
    : "uid=1000(guest) gid=1000(guest) groups=1000(guest),100(users)\n"),
  hostname: (a) => ok(a.includes("-I") ? "10.0.2.15 \n" : "bity\n"),
  groups: (_a, st) => ok(st.user === "root" ? "root\n" : "guest users\n"),
  arch: () => ok("aarch64\n"),
  uname: (a) => {
    if (a.includes("-a")) return ok("Linux bity 6.12.74+deb13+1-cloud-arm64 #1 SMP Debian 6.12.74-2 aarch64 GNU/Linux\n");
    if (a.includes("-r")) return ok("6.12.74+deb13+1-cloud-arm64\n");
    if (a.includes("-m")) return ok("aarch64\n");
    return ok("Linux\n");
  },
  uptime: () => ok(" 17:30:04 up 3 days,  2:14,  1 user,  load average: 0.08, 0.03, 0.01\n"),
  date: (a) => ok((a[0]?.startsWith("+%s") ? "1752082204" : "Wed Jul  9 17:30:04 UTC 2026") + "\n"),
  true: () => ok(),
  false: () => ({ out: "", exit: 1 }),
  clear: () => ok("\x1b[H\x1b[2J"),
  which: (a, st) => { const c = nonFlags(a)[0]; return c && (c in CORE || DREAMED.has(c)) ? ok(`/usr/bin/${c}\n`) : { out: "", exit: 1 }; },
  env: (_a, st) => ok([...st.env].map(([k, v]) => `${k}=${v}`).join("\n") + "\n"),
  printenv: (a, st) => { const v = st.env.get(a[0]); return v !== undefined ? ok(v + "\n") : { out: "", exit: 1 }; },

  ls: (a, st) => {
    const all = hasFlag(a, "a"), long = hasFlag(a, "l"), targets = nonFlags(a);
    const paths = targets.length ? targets : ["."];
    const out: string[] = [];
    for (const p of paths) {
      const node = st.vfs.lookup(p, st.cwd);
      if (!node) { out.push(`ls: cannot access '${p}': No such file or directory`); continue; }
      if (node.kind === "file") { out.push(long ? longRow(baseName(p), node) : p); continue; }
      const names = st.vfs.list(p, st.cwd, all) ?? [];
      const entries = all ? [".", "..", ...names] : names;
      if (paths.length > 1) out.push(`${p}:`);
      if (long) {
        out.push(`total ${Math.max(0, entries.length * 4)}`);
        for (const e of entries) {
          const n = e === "." ? node : e === ".." ? node : node.children.get(e)!;
          out.push(longRow(e, n));
        }
      } else out.push(entries.join("  "));
      if (paths.length > 1) out.push("");
    }
    return ok(out.join("\n").replace(/\n+$/, "") + "\n");
  },

  cat: (a, st, stdin) => {
    const files = nonFlags(a);
    if (!files.length) return ok(stdin);
    let out = "", exit = 0;
    for (const f of files) {
      const node = st.vfs.lookup(f, st.cwd);
      if (!node) { out += `cat: ${f}: No such file or directory\n`; exit = 1; }
      else if (node.kind === "dir") { out += `cat: ${f}: Is a directory\n`; exit = 1; }
      else out += node.content + (node.content.endsWith("\n") || node.content === "" ? "" : "\n");
    }
    return { out, exit };
  },

  cd: (a, st) => {
    const t = nonFlags(a)[0] ?? "~";
    const abs = st.vfs.abs(t === "-" ? (st.env.get("OLDPWD") ?? st.cwd) : t, st.cwd);
    const node = st.vfs.lookup(abs, "/");
    if (!node) return fail(`bash: cd: ${t}: No such file or directory`);
    if (node.kind !== "dir") return fail(`bash: cd: ${t}: Not a directory`);
    st.env.set("OLDPWD", st.cwd);
    st.cwd = abs;
    return ok();
  },

  mkdir: (a, st) => {
    const p = hasFlag(a, "p");
    for (const d of nonFlags(a)) { const r = st.vfs.mkdir(d, st.cwd, p, st.user); if (!r.ok && !p) return fail(`mkdir: ${r.err}`); }
    return ok();
  },
  rmdir: (a, st) => { for (const d of nonFlags(a)) { const n = st.vfs.lookup(d, st.cwd); if (n?.kind === "dir" && n.children.size) return fail(`rmdir: failed to remove '${d}': Directory not empty`); st.vfs.remove(d, st.cwd, true); } return ok(); },
  touch: (a, st) => { for (const f of nonFlags(a)) st.vfs.touch(f, st.cwd, st.user); return ok(); },
  rm: (a, st) => {
    const rec = hasFlag(a, "r", "R"), force = hasFlag(a, "f");
    for (const f of nonFlags(a)) { if (f === "/") return fail("rm: it is dangerous to operate recursively on '/'\nrm: use --no-preserve-root to override this failsafe"); const r = st.vfs.remove(f, st.cwd, rec); if (!r.ok && !force) return fail(`rm: ${r.err}`); }
    return ok();
  },
  mv: (a, st) => { const t = nonFlags(a); const dst = t.pop()!; for (const s of t) { const r = st.vfs.move(s, dst, st.cwd); if (!r.ok) return fail(`mv: ${r.err}`); } return ok(); },
  cp: (a, st) => { const rec = hasFlag(a, "r", "R"); const t = nonFlags(a); const dst = t.pop()!; for (const s of t) { const r = st.vfs.copy(s, dst, st.cwd, rec); if (!r.ok) return fail(`cp: ${r.err}`); } return ok(); },
  chmod: (a, st) => { for (const f of nonFlags(a).slice(1)) if (!st.vfs.exists(f, st.cwd)) return fail(`chmod: cannot access '${f}': No such file or directory`); return ok(); },

  wc: (a, st, stdin) => {
    const files = nonFlags(a), lFlag = hasFlag(a, "l"), wFlag = hasFlag(a, "w"), cFlag = hasFlag(a, "c");
    const count = (s: string, label: string) => {
      const lines = s === "" ? 0 : s.replace(/\n$/, "").split("\n").length;
      const words = s.trim() === "" ? 0 : s.trim().split(/\s+/).length;
      const bytes = s.length;
      if (lFlag && !wFlag && !cFlag) return `${lines}${label}`;
      if (wFlag && !lFlag && !cFlag) return `${words}${label}`;
      if (cFlag && !lFlag && !wFlag) return `${bytes}${label}`;
      return `${String(lines).padStart(2)} ${String(words).padStart(2)} ${String(bytes).padStart(2)}${label}`;
    };
    if (!files.length) return ok(count(stdin, "") + "\n");
    let out = "";
    for (const f of files) { const n = st.vfs.lookup(f, st.cwd); if (!n || n.kind === "dir") { out += `wc: ${f}: ${n ? "Is a directory" : "No such file or directory"}\n`; continue; } out += count(n.content, " " + f) + "\n"; }
    return ok(out);
  },
  head: (a, st, stdin) => headTail(a, st, stdin, true),
  tail: (a, st, stdin) => headTail(a, st, stdin, false),
  grep: (a, st, stdin) => {
    const pos = nonFlags(a); const pat = pos[0]; const files = pos.slice(1);
    const inv = hasFlag(a, "v"), ci = hasFlag(a, "i");
    const rx = new RegExp(pat.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), ci ? "i" : "");
    const scan = (text: string) => text.replace(/\n$/, "").split("\n").filter((l) => rx.test(l) !== inv);
    if (!files.length) { const m = scan(stdin); return { out: m.join("\n") + (m.length ? "\n" : ""), exit: m.length ? 0 : 1 }; }
    let out = "", hit = false;
    for (const f of files) { const n = st.vfs.lookup(f, st.cwd); if (n?.kind === "file") { const m = scan(n.content); if (m.length) hit = true; out += m.map((l) => (files.length > 1 ? `${f}:${l}` : l)).join("\n") + (m.length ? "\n" : ""); } }
    return { out, exit: hit ? 0 : 1 };
  },
  sort: (a, st, stdin) => { const lines = stdin.replace(/\n$/, "").split("\n"); lines.sort(); if (hasFlag(a, "r")) lines.reverse(); return ok(lines.join("\n") + "\n"); },
  uniq: (a, st, stdin) => { const lines = stdin.replace(/\n$/, "").split("\n"); const out: string[] = []; let prev: string | null = null, cnt = 0; for (const l of lines) { if (l === prev) cnt++; else { if (prev !== null) out.push(hasFlag(a, "c") ? `${String(cnt).padStart(7)} ${prev}` : prev); prev = l; cnt = 1; } } if (prev !== null) out.push(hasFlag(a, "c") ? `${String(cnt).padStart(7)} ${prev}` : prev); return ok(out.join("\n") + "\n"); },
  rev: (_a, _st, stdin) => ok(stdin.replace(/\n$/, "").split("\n").map((l) => [...l].reverse().join("")).join("\n") + "\n"),
  tr: (a, st, stdin) => { const [from, to] = nonFlags(a); if (from === "a-z" && to === "A-Z") return ok(stdin.toUpperCase()); if (from === "A-Z" && to === "a-z") return ok(stdin.toLowerCase()); return ok(stdin); },
  nl: (_a, _st, stdin) => ok(stdin.replace(/\n$/, "").split("\n").map((l, i) => `${String(i + 1).padStart(6)}\t${l}`).join("\n") + "\n"),
  seq: (a) => { const n = nonFlags(a).map(Number); const [s, e] = n.length === 1 ? [1, n[0]] : [n[0], n[1]]; const out = []; for (let i = s; i <= e; i++) out.push(i); return ok(out.join("\n") + "\n"); },
};

function headTail(a: string[], st: ShellState, stdin: string, isHead: boolean): Output {
  const ni = a.findIndex((x) => x === "-n"); let n = 10;
  if (ni >= 0 && a[ni + 1]) n = parseInt(a[ni + 1], 10);
  else { const m = a.find((x) => /^-\d+$/.test(x)); if (m) n = parseInt(m.slice(1), 10); }
  const files = nonFlags(a).filter((x) => !/^\d+$/.test(x));
  const slice = (text: string) => { const lines = text.replace(/\n$/, "").split("\n"); return (isHead ? lines.slice(0, n) : lines.slice(-n)).join("\n"); };
  if (!files.length) return ok(slice(stdin) + "\n");
  let out = "";
  for (const f of files) { const node = st.vfs.lookup(f, st.cwd); if (!node || node.kind === "dir") { out += `${isHead ? "head" : "tail"}: cannot open '${f}' for reading: No such file or directory\n`; continue; } out += slice(node.content) + "\n"; }
  return ok(out);
}

// commands the MODEL still dreams (so `which` knows them, and the shell routes to the model)
export const DREAMED = new Set([
  "ping", "traceroute", "curl", "wget", "dig", "nslookup", "host", "ssh", "scp", "nc", "ip", "ifconfig", "ss", "netstat",
  "git", "ps", "top", "htop", "apt", "apt-get", "dpkg", "systemctl", "journalctl", "docker", "make", "gcc", "vim", "nano",
  "man", "fortune", "cowsay", "neofetch", "figlet", "sudo", "reboot", "df", "free", "lscpu", "lsblk", "mount", "python3", "node", "npm",
]);
