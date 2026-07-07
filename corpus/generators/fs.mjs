// Filesystem binaries over a small invented world: ls, cat, cd, pwd, mkdir,
// touch, rm, du, tree. Formats match real Debian coreutils captures.

import { pick, randint, chance, shortDate, copyArg, contentFor, rec } from "./lib.mjs";

const HOME_BASE = ["projects", "notes.txt", "todo.md"];
const HOME_EXTRA = ["backup.tar.gz", "data.csv", "scripts", "main.py", "index.html", "draft.txt", "photos", "bin", "logs", "config.yaml"];
const HIDDEN = [".bashrc", ".profile", ".config", ".cache", ".gitconfig", ".ssh"];

const DIRS = new Set(["projects", "scripts", "photos", "bin", "logs", ".config", ".cache", ".ssh", "src", "docs"]);

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
  "notes.txt": (rng) =>
    pick(rng, [
      "remember to feed the model\nbackup corpus to /var/backups\ncheck disk space on friday",
      "training run started, check loss curve\nrotate ssh keys\nbuy more coffee",
      "the terminal dreams of being real\nfix the flaky test in ci\nread the systemd docs",
    ]),
  "todo.md": (rng) =>
    pick(rng, [
      "# todo\n- [x] capture corpus\n- [ ] train the model\n- [ ] ship the terminal",
      "# todo\n- [ ] write generators\n- [x] fix the race condition\n- [ ] deploy to bity.dev",
    ]),
  ".bashrc": () =>
    "# ~/.bashrc: executed by bash(1) for non-login shells.\nalias ll='ls -alF'\nalias la='ls -A'\nalias l='ls -CF'\nexport EDITOR=vim",
  "/etc/hostname": () => "bity",
  "/etc/os-release": () =>
    'PRETTY_NAME="Debian GNU/Linux 13 (trixie)"\nNAME="Debian GNU/Linux"\nVERSION_ID="13"\nVERSION="13 (trixie)"\nVERSION_CODENAME=trixie\nID=debian\nHOME_URL="https://www.debian.org/"',
  "/etc/debian_version": () => "13.1",
  "projects/README.md": () => "# bityllm\n\nA tiny LLM in pure TypeScript.",
  "/proc/loadavg": (rng) =>
    `${(rng.random() * 2).toFixed(2)} ${(rng.random() * 1.5).toFixed(2)} ${(rng.random()).toFixed(2)} ${randint(rng, 1, 4)}/${randint(rng, 120, 400)} ${randint(rng, 1000, 60000)}`,
};

const MISSING = ["config.json", "secrets.txt", "output.log", "/etc/shadow2", "test.txt", "results.csv", "foo", "notes.bak"];

// Stateful sessions (corpus v3): mkdir/touch/rm genuinely mutate the world and
// EVERY subsequent listing reflects it. Design for learnability (ping-style):
//   - created names are RANDOM syllable strings → memorization impossible,
//     copy-from-context is the only winning strategy
//   - created names APPEND at the END of listings in creation order → the copy
//     program is "repeat base, then repeat created names", no sorted insertion
//   - the base filesystem is FIXED → the only variance is what was created
const SYL = ["ka", "lo", "mir", "ten", "zu", "bel", "rin", "dov", "pax", "qui", "fen", "sha", "gro", "vim", "ost"];
const EXT = ["", "", ".txt", ".log", ".md", ".csv", ".sh"];

function randName(rng) {
  // v5: universal-alphabet names (see lib.copyArg) so listing-copy generalizes
  return copyArg(rng) + pick(rng, EXT);
}

export function fsSessionBlock(rng) {
  const BASE = ["notes.txt", "projects", "todo.md"]; // fixed, always sorted first
  const created = []; // creation order, appended after base in listings
  const meta = new Map(); // name → { dir: bool, content: string }

  const listing = () => [...BASE, ...created];
  const ls = () => rec("ls", listing().join("  "));
  const lsLa = () => {
    const rows = [`total ${randint(rng, 20, 96)}`];
    for (const e of [".", "..", ...listing()]) {
      const isDir = e === "." || e === ".." || e === "projects" || meta.get(e)?.dir === true;
      const owner = e === ".." ? "root  root " : "guest guest";
      const size = isDir ? 4096 : pick(rng, [220, 807, 1024, 1943, 3526, 5842]);
      rows.push(`${isDir ? "drwxr-xr-x" : "-rw-r--r--"} ${isDir ? randint(rng, 2, 5) : 1} ${owner} ${String(size).padStart(5, " ")} ${shortDate(rng)} ${e}`);
    }
    return rec("ls -la", rows.join("\n"));
  };
  // cat a created name: written → its words, touched → empty, dir → error
  const catCreated = (name) => {
    const m = meta.get(name);
    if (m.dir) return rec(`cat ${name}`, `cat: ${name}: Is a directory`);
    return rec(`cat ${name}`, m.content);
  };

  const recs = [];
  if (chance(rng, 0.4)) recs.push(ls()); // "before" listing
  const mutations = randint(rng, 1, 4);
  for (let m = 0; m < mutations; m++) {
    const roll = rng.random();
    let lastRemoved = null;
    if (roll < 0.2) {
      const d = randName(rng).replace(/\.\w+$/, ""); // dirs: no extension
      created.push(d);
      meta.set(d, { dir: true, content: "" });
      recs.push(rec(`mkdir ${d}`, ""));
      if (chance(rng, 0.25)) recs.push(rec(`ls ${d}`, "")); // fresh dir is empty
      if (chance(rng, 0.25)) {
        recs.push(rec(`cd ${d}`, ""));
        recs.push(rec("pwd", `/home/guest/${d}`));
        recs.push(rec("cd ..", ""));
      }
    } else if (roll < 0.4) {
      const f = randName(rng);
      created.push(f);
      meta.set(f, { dir: false, content: "" }); // touch → empty file
      recs.push(rec(`touch ${f}`, ""));
    } else if (roll < 0.6) {
      // write-then-read-back: the referential-consistency drill
      const f = randName(rng);
      const words = Array.from({ length: randint(rng, 1, 3) }, () => copyArg(rng)).join(" ");
      created.push(f);
      meta.set(f, { dir: false, content: words });
      recs.push(rec(`echo ${words} > ${f}`, ""));
      if (chance(rng, 0.2)) recs.push(rec(`grep ${words.split(" ")[0]} ${f}`, words));
    } else if (roll < 0.72 && created.some((n) => !meta.get(n)?.dir)) {
      // mv: rename — listings and contents follow the new name
      const files = created.filter((n) => !meta.get(n)?.dir);
      const f = pick(rng, files);
      const g = randName(rng);
      created[created.indexOf(f)] = g;
      meta.set(g, meta.get(f));
      meta.delete(f);
      recs.push(rec(`mv ${f} ${g}`, ""));
    } else if (roll < 0.82 && created.some((n) => !meta.get(n)?.dir)) {
      // cp: duplicate — both names list, both cat identically
      const files = created.filter((n) => !meta.get(n)?.dir);
      const f = pick(rng, files);
      const g = randName(rng);
      created.push(g);
      meta.set(g, { ...meta.get(f) });
      recs.push(rec(`cp ${f} ${g}`, ""));
      if (chance(rng, 0.3)) recs.push(catCreated(g));
    } else if (created.length > 0 && chance(rng, 0.7)) {
      const f = pick(rng, created);
      created.splice(created.indexOf(f), 1);
      recs.push(rec(`rm ${meta.get(f)?.dir ? "-r " : ""}${f}`, ""));
      meta.delete(f);
      lastRemoved = f;
    } else if (chance(rng, 0.5) && BASE.length > 1) {
      const f = pick(rng, BASE.filter((n) => n !== "projects"));
      BASE.splice(BASE.indexOf(f), 1);
      recs.push(rec(`rm ${f}`, ""));
      lastRemoved = f;
    } else {
      recs.push(rec(`rm ghost.txt`, `rm: cannot remove 'ghost.txt': No such file or directory`));
    }
    // payoff after EVERY mutation: usually a listing, sometimes a read-back
    const pv = rng.random();
    if (lastRemoved !== null && pv < 0.25) {
      recs.push(rec(`cat ${lastRemoved}`, `cat: ${lastRemoved}: No such file or directory`)); // gone means gone
      recs.push(ls());
    } else if (pv < 0.55 || created.length === 0) recs.push(ls());
    else if (pv < 0.75) recs.push(lsLa());
    else {
      const written = created.filter((n) => meta.get(n)?.content);
      recs.push(catCreated(written.length > 0 && chance(rng, 0.75) ? pick(rng, written) : pick(rng, created)));
    }
  }
  // closing read-back reinforces cat-after-create at longer range
  if (created.length > 0 && chance(rng, 0.5)) recs.push(catCreated(pick(rng, created)));
  return recs;
}

export function* fsGen(rng) {
  for (;;) {
    const r = rng.random();
    if (r < 0.34) {
      // ls variants
      const entries = homeEntries(rng);
      const v = rng.random();
      if (v < 0.3) yield rec("ls", entries.join("  "));
      else if (v < 0.55) yield rec("ls -la", lsLong(rng, entries, true));
      else if (v < 0.7) yield rec("ls -l", lsLong(rng, entries, false));
      else if (v < 0.8) yield rec("ls -a", [".", "..", ...HIDDEN.filter(() => chance(rng, 0.8)), ...entries].sort().join("  "));
      else if (v < 0.9) yield rec("ls projects", pick(rng, ["README.md  src", "README.md  bityllm  dotfiles", "README.md  src  test  package.json"]));
      else yield rec(`ls ${pick(rng, MISSING)}`, `ls: cannot access '${pick(rng, MISSING)}': No such file or directory`);
    } else if (r < 0.58) {
      // cat: known files, extension-typed dreams, dirs, and errors
      const v = rng.random();
      if (v < 0.45) {
        const name = pick(rng, Object.keys(FILES));
        yield rec(`cat ${name}`, FILES[name](rng));
      } else if (v < 0.78) {
        // any extension-bearing name gets extension-shaped contents — covers
        // the listed-but-never-catted pool AND generalizes to dreamed names
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
    } else if (r < 0.7) {
      const v = rng.random();
      if (v < 0.4) yield rec("pwd", "/home/guest");
      else if (v < 0.6) yield rec(`cd ${pick(rng, ["projects", "/tmp", "..", "~"])}`, "");
      else yield rec(`cd ${pick(rng, ["/root", "nowhere", "secret"])}`, `bash: cd: ${pick(rng, ["/root", "nowhere", "secret"])}: ${pick(rng, ["Permission denied", "No such file or directory"])}`);
    } else if (r < 0.84) {
      const f = pick(rng, ["test.txt", "newdir", "tmp.log", "draft2.md", "data"]);
      const v = rng.random();
      if (v < 0.2) yield rec(`touch ${f}`, "");
      else if (v < 0.3) {
        const g = pick(rng, ["old.txt", "missing.csv", "ghost"]);
        yield rec(chance(rng, 0.5) ? `mv ${g} new.txt` : `cp ${g} new.txt`,
          `${chance(rng, 0.5) ? "mv" : "cp"}: cannot stat '${g}': No such file or directory`);
      }
      else if (v < 0.5) yield rec(`mkdir ${f}`, chance(rng, 0.8) ? "" : `mkdir: cannot create directory '${f}': File exists`);
      else if (v < 0.75) yield rec(`rm ${f}`, chance(rng, 0.7) ? "" : `rm: cannot remove '${f}': No such file or directory`);
      else yield rec(`rm -rf /`, "rm: it is dangerous to operate recursively on '/'\nrm: use --no-preserve-root to override this failsafe");
    } else {
      const v = rng.random();
      if (v < 0.5) yield rec(`du -sh ${pick(rng, ["projects", ".", "/etc", "/var/log"])}`, `${pick(rng, ["4.0K", "24K", "136K", "1.2M", "18M", "142M"])}\t${pick(rng, ["projects", ".", "/etc", "/var/log"])}`);
      else if (v < 0.75) yield rec("wc -l notes.txt", `${randint(rng, 2, 40)} notes.txt`);
      else yield rec("head -n 2 notes.txt", FILES["notes.txt"](rng).split("\n").slice(0, 2).join("\n"));
    }
  }
}
