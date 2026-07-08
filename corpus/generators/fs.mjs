// Filesystem binaries over a small invented world: ls, cat, cd, pwd, mkdir,
// touch, rm, mv, cp, chmod, wc, head/tail, du, redirects, pipes. Formats match
// real Debian coreutils captures.

import { pick, randint, chance, shortDate, copyArg, contentFor, promptFor, rec } from "./lib.mjs";

const HOME_BASE = ["projects", "notes.txt", "todo.md"];
const HOME_EXTRA = ["backup.tar.gz", "data.csv", "scripts", "main.py", "index.html", "draft.txt", "photos", "bin", "logs", "config.yaml"];
const HIDDEN = [".bashrc", ".profile", ".config", ".cache", ".gitconfig", ".ssh"];

const DIRS = new Set(["projects", "scripts", "photos", "bin", "logs", ".config", ".cache", ".ssh", "src", "docs"]);

// permission triads (chmod → ls -l consistency)
const MODE_STR = { 644: "-rw-r--r--", 600: "-rw-------", 755: "-rwxr-xr-x", 700: "-rwx------", 444: "-r--r--r--", 755:  "-rwxr-xr-x" };
const permStr = (m, dir) => dir ? "drwxr-xr-x" : (MODE_STR[m] ?? "-rw-r--r--");

// byte-accurate file metadata: one stored size that ls -l / wc / stat / du all read
function fileMeta(content, mode = 644) {
  const bytes = content === "" ? 0 : content.length + 1; // single trailing newline from echo
  const lines = content === "" ? 0 : content.split("\n").length;
  const words = content.trim() === "" ? 0 : content.trim().split(/\s+/).length;
  return { dir: false, content, bytes, lines, words, mode };
}
const dirMeta = () => ({ dir: true, content: "", bytes: 4096, mode: 755 });

const NOTES = "remember to feed the model\nbackup corpus to /var/backups\ncheck disk space on friday";
const TODO = "# todo\n- [x] capture corpus\n- [ ] train the model\n- [ ] ship the terminal";
const README = "# bityllm\n\nA tiny LLM in pure TypeScript.";

function homeEntries(rng) {
  const extra = [...HOME_EXTRA].filter(() => chance(rng, 0.3)).slice(0, 4);
  return [...HOME_BASE, ...extra].sort();
}

function lsLong(rng, entries, withHidden) {
  const rows = [];
  const all = withHidden ? [".", "..", ...HIDDEN.filter(() => chance(rng, 0.8)), ...entries].sort() : entries;
  rows.push(`total ${randint(rng, 20, 96)}`);
  for (const e of all) {
    const dir = e === "." || e === ".." || DIRS.has(e);
    const perms = dir ? "drwxr-xr-x" : "-rw-r--r--";
    const links = dir ? randint(rng, 2, 6) : 1;
    const owner = e === ".." ? "root  root " : "guest guest";
    const size = dir ? 4096 : pick(rng, [220, 807, 1024, 1943, 3526, 4523, 5842, 12288, 28190]);
    rows.push(`${perms} ${links} ${owner} ${String(size).padStart(5, " ")} ${shortDate(rng)} ${e}`);
  }
  return rows.join("\n");
}

const FILES = {
  "notes.txt": () => NOTES,
  "todo.md": () => TODO,
  ".bashrc": () =>
    "# ~/.bashrc: executed by bash(1) for non-login shells.\nalias ll='ls -alF'\nalias la='ls -A'\nalias l='ls -CF'\nexport EDITOR=vim",
  "/etc/hostname": () => "bity",
  "/etc/os-release": () =>
    'PRETTY_NAME="Debian GNU/Linux 13 (trixie)"\nNAME="Debian GNU/Linux"\nVERSION_ID="13"\nVERSION="13 (trixie)"\nVERSION_CODENAME=trixie\nID=debian\nHOME_URL="https://www.debian.org/"',
  "/etc/debian_version": () => "13.1",
  "projects/README.md": () => README,
  "/proc/loadavg": (rng) =>
    `${(rng.random() * 2).toFixed(2)} ${(rng.random() * 1.5).toFixed(2)} ${(rng.random()).toFixed(2)} ${randint(rng, 1, 4)}/${randint(rng, 120, 400)} ${randint(rng, 1000, 60000)}`,
};

const MISSING = ["config.json", "secrets.txt", "output.log", "/etc/shadow2", "test.txt", "results.csv", "foo", "notes.bak"];

const SYL = ["ka", "lo", "mir", "ten", "zu", "bel", "rin", "dov", "pax", "qui", "fen", "sha", "gro", "vim", "ost"];
const EXT = ["", "", ".txt", ".log", ".md", ".csv", ".sh"];

function randName(rng) {
  return copyArg(rng) + pick(rng, EXT);
}
const randDirName = (rng) => copyArg(rng); // no extension → reads as a directory

export function fsSessionBlock(rng) {
  // v8: NESTED location (a real path stack, arbitrary depth), byte-accurate file
  // metadata (ls -l / wc / stat / du all read one stored size), and DENSE
  // multi-word write→read round-trips (the anti-ceiling move for content copy).
  // Every directory is registered in `locs` keyed by path ("~", "~/a", "~/a/b").
  const locs = new Map();
  const dir = (p) => { if (!locs.has(p)) locs.set(p, { base: [], created: [], meta: new Map() }); return locs.get(p); };
  dir("~").base = ["notes.txt", "projects", "todo.md"];
  dir("~").meta.set("notes.txt", fileMeta(NOTES));
  dir("~").meta.set("todo.md", fileMeta(TODO));
  dir("~").meta.set("projects", dirMeta());
  dir("~/projects").base = ["README.md", "src"];
  dir("~/projects").meta.set("README.md", fileMeta(README));
  dir("~/projects").meta.set("src", dirMeta());

  let cwd = "~";
  const here = () => dir(cwd);
  const R = (cmd, output) => ({ cmd, output, prompt: promptFor(cwd) });
  const cmds = [], recs = [];
  const push = (r) => { cmds.push(r.cmd); recs.push(r); };

  const childPath = (name) => (cwd === "~" ? "~/" + name : cwd + "/" + name);
  const parentPath = () => (cwd === "~" ? "~" : cwd.slice(0, cwd.lastIndexOf("/")) || "~");
  const pathAbs = (p = cwd) => (p === "~" ? "/home/guest" : "/home/guest/" + p.slice(2));
  const listing = (p = cwd) => [...dir(p).base, ...dir(p).created];
  const isDir = (name, p = cwd) => dir(p).meta.get(name)?.dir === true;
  const metaOf = (name, p = cwd) => dir(p).meta.get(name);

  const ls = (p = cwd) => R(p === cwd ? "ls" : `ls ${relName(p)}`, listing(p).join("  "));
  const relName = (p) => (p === parentPath() ? ".." : p.slice(p.lastIndexOf("/") + 1));
  const lsLa = () => {
    const rows = [`total ${randint(rng, 12, 96)}`];
    for (const e of [".", "..", ...listing()]) {
      const m = e === "." || e === ".." ? { dir: true, bytes: 4096, mode: 755 } : metaOf(e);
      const d = m?.dir === true;
      const owner = e === ".." ? "root  root " : "guest guest";
      const size = d ? 4096 : (m?.bytes ?? 0);
      rows.push(`${permStr(m?.mode ?? 644, d)} ${d ? randint(rng, 2, 5) : 1} ${owner} ${String(size).padStart(5, " ")} ${shortDate(rng)} ${e}`);
    }
    return R("ls -la", rows.join("\n"));
  };
  const catName = (name, p = cwd) => {
    const m = metaOf(name, p);
    const label = p === cwd ? name : "../" + name;
    if (!m) return R(`cat ${label}`, `cat: ${label}: No such file or directory`);
    if (m.dir) return R(`cat ${label}`, `cat: ${label}: Is a directory`);
    return R(`cat ${label}`, m.content);
  };
  const wc = (flag, name) => {
    const m = metaOf(name);
    if (!m || m.dir) return R(`wc ${flag} ${name}`, `wc: ${name}: ${m ? "Is a directory" : "No such file or directory"}`);
    if (flag === "-l") return R(`wc -l ${name}`, `${m.lines} ${name}`);
    if (flag === "-w") return R(`wc -w ${name}`, `${m.words} ${name}`);
    if (flag === "-c") return R(`wc -c ${name}`, `${m.bytes} ${name}`);
    return R(`wc ${name}`, `${String(m.lines).padStart(2)} ${String(m.words).padStart(2)} ${String(m.bytes).padStart(2)} ${name}`);
  };

  // writable helpers ---------------------------------------------------------
  const writeFile = (f, content, mode = 644) => { here().created.push(f); here().meta.set(f, fileMeta(content, mode)); };
  const makeDir = (name) => { here().created.push(name); here().meta.set(name, dirMeta()); dir(childPath(name)); };
  const createdFiles = () => here().created.filter((n) => !isDir(n));
  const contentFiles = () => here().created.filter((n) => metaOf(n)?.content);
  const localDirs = () => [...here().created.filter((n) => isDir(n)), ...here().base.filter((n) => isDir(n))];

  if (chance(rng, 0.35)) push(ls());
  const mutations = randint(rng, 3, 6);
  for (let m = 0; m < mutations; m++) {
    const roll = rng.random();
    let removed = null;

    if (roll < 0.13) {
      // mkdir (sometimes -p a/b/c: registers every segment for nested cd)
      if (chance(rng, 0.3)) {
        const segs = Array.from({ length: randint(rng, 2, 3) }, () => randDirName(rng));
        let acc = cwd;
        segs.forEach((s, i) => { const d = dir(acc); if (i === 0) { d.created.push(s); d.meta.set(s, dirMeta()); } acc = acc === "~" ? "~/" + s : acc + "/" + s; dir(acc); });
        // register nested chain fully so later cd a/b/c works
        let p = cwd;
        for (const s of segs) { dir(p).meta.set(s, dirMeta()); if (!dir(p).base.includes(s) && !dir(p).created.includes(s)) dir(p).created.push(s); p = p === "~" ? "~/" + s : p + "/" + s; }
        push(R(`mkdir -p ${segs.join("/")}`, ""));
        if (chance(rng, 0.4)) { const deep = cwd === "~" ? "~/" + segs.join("/") : cwd + "/" + segs.join("/"); cwd = deep; push(R("pwd", pathAbs())); cwd = parentUp(cwd, segs.length); }
      } else {
        const d = randDirName(rng);
        makeDir(d);
        push(R(`mkdir ${d}`, ""));
        if (chance(rng, 0.2)) push(ls(childPath(d))); // fresh dir empty
      }
    } else if (roll < 0.30) {
      // cd — NESTED: descend into a real dir, or `cd ..` to pop exactly one level
      const dirs = localDirs();
      if (cwd !== "~" && chance(rng, 0.4)) {
        cwd = parentPath();
        push(R("cd ..", ""));
      } else if (dirs.length) {
        const d = pick(rng, dirs);
        cwd = childPath(d);
        push(R(`cd ${d}`, ""));
      } else { push(R("pwd", pathAbs())); continue; }
      const pv = rng.random();
      if (pv < 0.45) push(R("pwd", pathAbs()));
      else if (pv < 0.8) push(ls());
      continue;
    } else if (roll < 0.42) {
      // touch → EMPTY file (known gap c). Read-back proves it's known-but-empty.
      const f = randName(rng);
      writeFile(f, "");
      push(R(`touch ${f}`, ""));
      const pv = rng.random();
      if (pv < 0.5) push(catName(f));            // → "" (never ENOENT)
      else if (pv < 0.68) push(wc("-l", f));      // → "0 f"
      else if (pv < 0.8) push(R(`file ${f}`, `${f}: empty`));
    } else if (roll < 0.63) {
      // echo <words> > f  — DENSE multi-word round-trip (known gap a).
      // Bias to 2-4 words and almost always read the WHOLE thing straight back.
      const nWords = chance(rng, 0.8) ? randint(rng, 2, 4) : 1;
      const words = Array.from({ length: nWords }, () => copyArg(rng)).join(" ");
      const f = randName(rng);
      writeFile(f, words);
      push(R(`echo ${words} > ${f}`, ""));
      const pv = rng.random();
      if (pv < 0.6) push(catName(f));                        // → w1 w2 w3 (ALL words)
      else if (pv < 0.78) push(wc("-w", f));                 // → "N f"
      else if (pv < 0.9) push(R(`cat ${f} | wc -w`, String(metaOf(f).words))); // piped: bare number
      else push(wc("-c", f));
    } else if (roll < 0.72 && contentFiles().length) {
      // append → multi-line file; head/tail/wc slice the SAME accumulated content
      const f = pick(rng, contentFiles());
      const words = Array.from({ length: randint(rng, 1, 3) }, () => copyArg(rng)).join(" ");
      const mm = metaOf(f);
      here().meta.set(f, fileMeta(mm.content + "\n" + words, mm.mode));
      push(R(`echo ${words} >> ${f}`, ""));
      const nl = metaOf(f).content.split("\n");
      const pv = rng.random();
      if (pv < 0.35) push(catName(f));
      else if (pv < 0.55) push(wc("-l", f));
      else if (pv < 0.75) push(R(`head -n 1 ${f}`, nl[0]));
      else push(R(`tail -n 1 ${f}`, nl[nl.length - 1]));
    } else if (roll < 0.79 && createdFiles().length) {
      // chmod → ls -l reflects the new perm triad for the rest of the session
      const f = pick(rng, createdFiles());
      const target = chance(rng, 0.5) ? "+x" : pick(rng, ["600", "755", "700", "644"]);
      const mode = target === "+x" ? 755 : parseInt(target, 10);
      metaOf(f).mode = mode;
      push(R(`chmod ${target} ${f}`, ""));
      if (chance(rng, 0.7)) push(R(`ls -l ${f}`, `${permStr(mode, false)} 1 guest guest ${String(metaOf(f).bytes).padStart(5)} ${shortDate(rng)} ${f}`));
    } else if (roll < 0.85 && createdFiles().length) {
      // mv / cp — listings AND contents follow the new name; cp preserves bytes
      const f = pick(rng, createdFiles());
      const g = randName(rng);
      if (chance(rng, 0.6)) {
        here().created[here().created.indexOf(f)] = g;
        here().meta.set(g, metaOf(f)); here().meta.delete(f);
        push(R(`mv ${f} ${g}`, ""));
      } else {
        here().created.push(g); here().meta.set(g, { ...metaOf(f) });
        push(R(`cp ${f} ${g}`, ""));
        if (chance(rng, 0.3)) push(catName(g));
      }
    } else if (roll < 0.91) {
      // redirect a command's stdout into a file, then read it back
      if (chance(rng, 0.6)) {
        const f = randName(rng).replace(/\.\w+$/, "") + ".txt";
        const body = listing().join("\n");
        writeFile(f, body);
        push(R(`ls > ${f}`, ""));
        if (chance(rng, 0.7)) push(catName(f)); // listing, one per line
      } else if (contentFiles().length) {
        // pipe: cat f | grep <word> → only matching lines
        const f = pick(rng, contentFiles());
        const lines = metaOf(f).content.split("\n");
        const w = pick(rng, lines.join(" ").split(/\s+/).filter(Boolean));
        const hit = lines.filter((l) => l.includes(w));
        push(R(`cat ${f} | grep ${w}`, hit.join("\n")));
      } else push(R(`ls | wc -l`, String(listing().length)));
    } else if (here().created.length && chance(rng, 0.8)) {
      // rm (files) or rm -r (dirs) → gone from later ls/cat
      const f = pick(rng, here().created);
      here().created.splice(here().created.indexOf(f), 1);
      push(R(`rm ${isDir(f) ? "-r " : ""}${f}`, ""));
      here().meta.delete(f);
      removed = f;
    } else if (cwd === "~" && here().base.filter((n) => !isDir(n)).length > 1) {
      const f = pick(rng, here().base.filter((n) => !isDir(n)));
      here().base.splice(here().base.indexOf(f), 1);
      here().meta.delete(f);
      push(R(`rm ${f}`, ""));
      removed = f;
    } else {
      push(R(`rm ghost.txt`, `rm: cannot remove 'ghost.txt': No such file or directory`));
    }

    // payoff after the mutation
    const pv = rng.random();
    if (removed !== null && pv < 0.3) { push(catName(removed)); push(ls()); }   // gone means gone
    else if (pv < 0.5) push(ls());
    else if (pv < 0.66) push(lsLa());
    else if (pv < 0.76) push(R("pwd", pathAbs()));
    else if (contentFiles().length && chance(rng, 0.75)) push(catName(pick(rng, contentFiles())));
    else push(ls());
  }
  if (cmds.length >= 3 && chance(rng, 0.12)) {
    const start = randint(rng, 100, 900);
    const body = cmds.map((c, i) => `  ${start + i}  ${c}`).join("\n") + `\n  ${start + cmds.length}  history`;
    push(R("history", body));
  }
  return recs;
}

// walk cwd up n segments (helper for the mkdir -p pwd probe)
function parentUp(p, n) {
  for (let i = 0; i < n && p !== "~"; i++) p = p.slice(0, p.lastIndexOf("/")) || "~";
  return p;
}

export function* fsGen(rng) {
  for (;;) {
    const r = rng.random();
    if (r < 0.32) {
      const entries = homeEntries(rng);
      const v = rng.random();
      if (v < 0.3) yield rec("ls", entries.join("  "));
      else if (v < 0.52) yield rec("ls -la", lsLong(rng, entries, true));
      else if (v < 0.66) yield rec("ls -l", lsLong(rng, entries, false));
      else if (v < 0.76) yield rec("ls -a", [".", "..", ...HIDDEN.filter(() => chance(rng, 0.8)), ...entries].sort().join("  "));
      else if (v < 0.86) yield rec("ls projects", pick(rng, ["README.md  src", "README.md  bityllm  dotfiles", "README.md  src  test  package.json"]));
      else yield rec(`ls ${pick(rng, MISSING)}`, `ls: cannot access '${pick(rng, MISSING)}': No such file or directory`);
    } else if (r < 0.55) {
      const v = rng.random();
      if (v < 0.45) {
        const name = pick(rng, Object.keys(FILES));
        yield rec(`cat ${name}`, FILES[name](rng));
      } else if (v < 0.78) {
        const name = chance(rng, 0.5)
          ? pick(rng, ["data.csv", "main.py", "index.html", "config.yaml", "draft.txt", "backup.log", "notes2.md"])
          : copyArg(rng) + pick(rng, [".txt", ".csv", ".py", ".md", ".log", ".sh", ".yaml", ".json", ".html"]);
        yield rec(`cat ${name}`, contentFor(rng, name));
      } else if (v < 0.88) {
        const d = pick(rng, ["projects", "scripts", "photos", "bin", "logs"]);
        yield rec(`cat ${d}`, `cat: ${d}: Is a directory`);
      } else {
        const m = pick(rng, MISSING);
        yield rec(`cat ${m}`, `cat: ${m}: No such file or directory`);
      }
    } else if (r < 0.66) {
      // pwd / cd — errno now NAMES the typed arg (fixed) and matches the arg kind
      const v = rng.random();
      if (v < 0.4) yield rec("pwd", "/home/guest");
      else if (v < 0.62) yield rec(`cd ${pick(rng, ["projects", "/tmp", "..", "~", "-"])}`, "");
      else if (v < 0.8) {
        const f = pick(rng, ["notes.txt", "todo.md", "main.py"]); // a real FILE → Not a directory
        yield rec(`cd ${f}`, `bash: cd: ${f}: Not a directory`);
      } else {
        const g = pick(rng, ["nowhere", "missing", "foo", "config"]); // absent → No such file
        yield rec(`cd ${g}`, `bash: cd: ${g}: No such file or directory`);
      }
    } else if (r < 0.82) {
      const f = pick(rng, ["test.txt", "newdir", "tmp.log", "draft2.md", "data"]);
      const v = rng.random();
      if (v < 0.18) yield rec(`touch ${f}`, "");
      else if (v < 0.28) {
        const g = pick(rng, ["old.txt", "missing.csv", "ghost"]);
        const op = chance(rng, 0.5) ? "mv" : "cp";
        yield rec(`${op} ${g} new.txt`, `${op}: cannot stat '${g}': No such file or directory`);
      } else if (v < 0.46) yield rec(`mkdir ${f}`, chance(rng, 0.8) ? "" : `mkdir: cannot create directory '${f}': File exists`);
      else if (v < 0.64) yield rec(`rm ${f}`, chance(rng, 0.7) ? "" : `rm: cannot remove '${f}': No such file or directory`);
      else if (v < 0.74) {
        // rm -rf <name> is a SILENT ordinary delete; the failsafe is ONLY for literal /
        yield rec(`rm -rf ${pick(rng, ["build", "dist", "node_modules", "tmp", ".cache"])}`, "");
      } else if (v < 0.82) yield rec(`rm -rf /`, "rm: it is dangerous to operate recursively on '/'\nrm: use --no-preserve-root to override this failsafe");
      else if (v < 0.9) yield rec(`chmod ${pick(rng, ["+x deploy.sh", "600 notes.txt", "755 run.sh"])}`, "");
      else yield rec(`rmdir ${pick(rng, ["empty", "olddir"])}`, chance(rng, 0.6) ? "" : `rmdir: failed to remove 'projects': Directory not empty`);
    } else {
      const v = rng.random();
      if (v < 0.4) yield rec(`du -sh ${pick(rng, ["projects", ".", "/etc", "/var/log"])}`, `${pick(rng, ["4.0K", "24K", "136K", "1.2M", "18M", "142M"])}\t${pick(rng, ["projects", ".", "/etc", "/var/log"])}`);
      else if (v < 0.6) yield rec("wc -l notes.txt", "3 notes.txt");
      else if (v < 0.75) yield rec("head -n 2 notes.txt", NOTES.split("\n").slice(0, 2).join("\n"));
      else if (v < 0.9) yield rec("tail -n 1 notes.txt", NOTES.split("\n").slice(-1)[0]);
      else yield rec("cat notes.txt | wc -l", "3");
    }
  }
}
