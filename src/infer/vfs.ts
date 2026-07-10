// vfs.ts — an in-memory POSIX-ish filesystem for the programmatic shell.
//
// The hybrid architecture: deterministic, stateful commands (ls/cd/cat/mkdir/mv…)
// run against THIS instead of being hallucinated by the model — so they are
// always correct and consistent (no more teaching the model to "remember" files).
// Pure TS, zero deps.

export interface VFile { kind: "file"; content: string; mode: number; owner: string; group: string; mtime: number; link?: string; }
export interface VDir { kind: "dir"; children: Map<string, VNode>; mode: number; owner: string; group: string; mtime: number; }
export type VNode = VFile | VDir;

const MT = 1_752_000_000; // a fixed "now" so listings are stable (Date.now() is banned in some paths anyway)
export const mkFile = (content = "", owner = "guest", mode = 0o644): VFile =>
  ({ kind: "file", content, mode, owner, group: owner, mtime: MT });
export const mkDir = (owner = "guest", mode = 0o755): VDir =>
  ({ kind: "dir", children: new Map(), mode, owner, group: owner, mtime: MT });

export interface OpResult { ok: boolean; err?: string; }

export class VFS {
  root: VDir = mkDir("root");

  constructor() { seed(this); }

  /** Normalize a path against cwd into absolute segments (resolves ~, ., ..). */
  private segs(path: string, cwd: string): string[] {
    let p = path;
    if (p === "~" || p.startsWith("~/")) p = "/home/guest" + p.slice(1);
    if (!p.startsWith("/")) p = (cwd === "/" ? "" : cwd) + "/" + p;
    const out: string[] = [];
    for (const s of p.split("/")) {
      if (s === "" || s === ".") continue;
      if (s === "..") out.pop();
      else out.push(s);
    }
    return out;
  }

  /** Absolute, normalized path string. */
  abs(path: string, cwd: string): string {
    const s = this.segs(path, cwd);
    return "/" + s.join("/");
  }

  /** Resolve a path to a node, or null if any component is missing. */
  lookup(path: string, cwd: string): VNode | null {
    let node: VNode = this.root;
    for (const seg of this.segs(path, cwd)) {
      if (node.kind !== "dir") return null;
      const next = node.children.get(seg);
      if (!next) return null;
      node = next;
    }
    return node;
  }

  /** Parent directory node + the final path component (for create/remove). */
  private parent(path: string, cwd: string): { dir: VDir | null; name: string } {
    const s = this.segs(path, cwd);
    const name = s.pop() ?? "";
    let node: VNode = this.root;
    for (const seg of s) {
      if (node.kind !== "dir") return { dir: null, name };
      const next = node.children.get(seg);
      if (!next) return { dir: null, name };
      node = next;
    }
    return { dir: node.kind === "dir" ? node : null, name };
  }

  isDir(path: string, cwd: string): boolean { return this.lookup(path, cwd)?.kind === "dir"; }
  exists(path: string, cwd: string): boolean { return this.lookup(path, cwd) !== null; }

  /** Sorted child names of a directory (optionally including dotfiles). */
  list(path: string, cwd: string, all = false): string[] | null {
    const node = this.lookup(path, cwd);
    if (!node || node.kind !== "dir") return null;
    return [...node.children.keys()].filter((n) => all || !n.startsWith(".")).sort();
  }

  mkdir(path: string, cwd: string, parents = false, owner = "guest"): OpResult {
    if (parents) {
      let acc = "/";
      for (const seg of this.segs(path, cwd)) {
        acc = acc === "/" ? "/" + seg : acc + "/" + seg;
        const at = this.lookup(acc, cwd);
        if (!at) { const p = this.parent(acc, cwd); p.dir?.children.set(p.name, mkDir(owner)); }
        else if (at.kind !== "dir") return { ok: false, err: `cannot create directory '${path}': Not a directory` };
      }
      return { ok: true };
    }
    const { dir, name } = this.parent(path, cwd);
    if (!dir) return { ok: false, err: `cannot create directory '${path}': No such file or directory` };
    if (dir.children.has(name)) return { ok: false, err: `cannot create directory '${path}': File exists` };
    dir.children.set(name, mkDir(owner));
    return { ok: true };
  }

  writeFile(path: string, cwd: string, content: string, owner = "guest", append = false): OpResult {
    const { dir, name } = this.parent(path, cwd);
    if (!dir) return { ok: false, err: `cannot create '${path}': No such file or directory` };
    const existing = dir.children.get(name);
    if (existing?.kind === "dir") return { ok: false, err: `${path}: Is a directory` };
    if (append && existing?.kind === "file") existing.content += content;
    else dir.children.set(name, mkFile(content, owner));
    return { ok: true };
  }

  touch(path: string, cwd: string, owner = "guest"): OpResult {
    if (this.exists(path, cwd)) return { ok: true };
    return this.writeFile(path, cwd, "", owner);
  }

  remove(path: string, cwd: string, recursive = false): OpResult {
    const node = this.lookup(path, cwd);
    if (!node) return { ok: false, err: `cannot remove '${path}': No such file or directory` };
    if (node.kind === "dir" && !recursive) return { ok: false, err: `cannot remove '${path}': Is a directory` };
    const { dir, name } = this.parent(path, cwd);
    dir?.children.delete(name);
    return { ok: true };
  }

  move(src: string, dst: string, cwd: string): OpResult {
    const node = this.lookup(src, cwd);
    if (!node) return { ok: false, err: `cannot stat '${src}': No such file or directory` };
    // moving into an existing directory keeps the basename
    let target = dst;
    if (this.isDir(dst, cwd)) target = this.abs(dst, cwd) + "/" + baseName(src);
    const to = this.parent(target, cwd);
    if (!to.dir) return { ok: false, err: `cannot move '${src}' to '${dst}': No such file or directory` };
    const from = this.parent(src, cwd);
    from.dir?.children.delete(from.name);
    to.dir.children.set(to.name, node);
    return { ok: true };
  }

  copy(src: string, dst: string, cwd: string, recursive = false): OpResult {
    const node = this.lookup(src, cwd);
    if (!node) return { ok: false, err: `cannot stat '${src}': No such file or directory` };
    if (node.kind === "dir" && !recursive) return { ok: false, err: `-r not specified; omitting directory '${src}'` };
    let target = dst;
    if (this.isDir(dst, cwd)) target = this.abs(dst, cwd) + "/" + baseName(src);
    const to = this.parent(target, cwd);
    if (!to.dir) return { ok: false, err: `cannot create '${dst}': No such file or directory` };
    to.dir.children.set(to.name, clone(node));
    return { ok: true };
  }
}

export const baseName = (p: string): string => p.replace(/\/+$/, "").split("/").pop() || "/";
export const dirName = (p: string): string => { const s = p.replace(/\/+$/, "").split("/"); s.pop(); return s.join("/") || "/"; };

function clone(node: VNode): VNode {
  if (node.kind === "file") return { ...node };
  const d = mkDir(node.owner, node.mode);
  for (const [k, v] of node.children) d.children.set(k, clone(v));
  return d;
}

// ---- a plausible, lived-in Debian home ----
function seed(vfs: VFS): void {
  const add = (path: string, content: string, owner = "guest"): void => { vfs.writeFile(path, "/", content, owner); };
  vfs.mkdir("/home", "/", true, "root");
  vfs.mkdir("/home/guest", "/", true, "guest");
  vfs.mkdir("/home/guest/projects", "/", true, "guest");
  vfs.mkdir("/home/guest/.config", "/", true, "guest");
  vfs.mkdir("/root", "/", true, "root");
  vfs.mkdir("/etc", "/", true, "root");
  vfs.mkdir("/tmp", "/", true, "root");
  vfs.mkdir("/var/log", "/", true, "root");
  vfs.mkdir("/usr/bin", "/", true, "root");
  vfs.mkdir("/proc", "/", true, "root");

  add("/home/guest/notes.txt", "remember to feed the model\nbackup corpus to /var/backups\ncheck disk space on friday");
  add("/home/guest/todo.md", "# todo\n- [x] capture corpus\n- [ ] train the model\n- [ ] ship the terminal");
  add("/home/guest/.bashrc", "# ~/.bashrc\nalias ll='ls -alF'\nalias la='ls -A'\nalias l='ls -CF'\nexport EDITOR=vim");
  add("/home/guest/.profile", "# ~/.profile\n[ -n \"$BASH_VERSION\" ] && . ~/.bashrc");
  add("/home/guest/projects/README.md", "# bityllm\n\nA tiny LLM in pure TypeScript.");
  add("/etc/hostname", "bity\n", "root");
  add("/etc/os-release", 'PRETTY_NAME="Debian GNU/Linux 13 (trixie)"\nNAME="Debian GNU/Linux"\nVERSION_ID="13"\nVERSION_CODENAME=trixie\nID=debian\n', "root");
  add("/etc/debian_version", "13.1\n", "root");
  add("/etc/hosts", "127.0.0.1\tlocalhost\n127.0.1.1\tbity\n", "root");
  add("/etc/passwd", "root:x:0:0:root:/root:/bin/bash\nguest:x:1000:1000:guest:/home/guest:/bin/bash\n", "root");
  add("/etc/group", "root:x:0:\ndaemon:x:1:\nbin:x:2:\nsys:x:3:\nadm:x:4:\ntty:x:5:\ndisk:x:6:\nsudo:x:27:guest\nusers:x:100:\nguest:x:1000:\n", "root");
  add("/etc/fstab", "# UNCONFIGURED FSTAB FOR BASE SYSTEM\n", "root");
  add("/etc/resolv.conf", "nameserver 192.168.1.1\nsearch localdomain\n", "root");
  add("/etc/issue", "Debian GNU/Linux 13 \\n \\l\n", "root");

  // /proc — static system info, kept consistent with the VFS's Debian 13 /
  // aarch64 / kernel 6.12.74 identity (so `cat /proc/*` agrees with `uname`).
  add("/proc/version", "Linux version 6.12.74+deb13+1-cloud-arm64 (debian-kernel@lists.debian.org) (gcc-14 (Debian 14.2.0-3) 14.2.0, GNU ld (GNU Binutils for Debian) 2.43.1) #1 SMP Debian 6.12.74-2 (2025-05-15)\n", "root");
  const cpu = (n: number): string =>
    `processor\t: ${n}\nBogoMIPS\t: 48.00\nFeatures\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm lrcpc dcpop asimddp\nCPU implementer\t: 0x61\nCPU architecture: 8\nCPU variant\t: 0x0\nCPU part\t: 0x000\nCPU revision\t: 0\n`;
  add("/proc/cpuinfo", cpu(0) + "\n" + cpu(1) + "\n", "root");
  add("/proc/meminfo", [
    "MemTotal:        2006468 kB", "MemFree:          146992 kB", "MemAvailable:    1122824 kB",
    "Buffers:           68948 kB", "Cached:           844992 kB", "SwapCached:            0 kB",
    "Active:           520140 kB", "Inactive:        1024036 kB", "SwapTotal:       1000444 kB",
    "SwapFree:        1000444 kB", "Dirty:               220 kB", "Writeback:             0 kB",
  ].join("\n") + "\n", "root");
  add("/proc/loadavg", "0.00 0.02 0.00 1/312 1847\n", "root");
  add("/proc/uptime", "82835.37 163551.40\n", "root");
  add("/proc/cmdline", "BOOT_IMAGE=/boot/vmlinuz-6.12.74+deb13+1-cloud-arm64 root=UUID=e7f2a1c4-8b3d-4e6f-9a0b-1c2d3e4f5a6b ro quiet\n", "root");
}
