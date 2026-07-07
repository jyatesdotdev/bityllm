// Filesystem binaries over a small invented world: ls, cat, cd, pwd, mkdir,
// touch, rm, du, tree. Formats match real Debian coreutils captures.

import { pick, randint, chance, shortDate, copyArg, rec } from "./lib.mjs";

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
  const isDirCreated = new Map();

  const listing = () => [...BASE, ...created];
  const ls = () => rec("ls", listing().join("  "));
  const lsLa = () => {
    const rows = [`total ${randint(rng, 20, 96)}`];
    for (const e of [".", "..", ...listing()]) {
      const isDir = e === "." || e === ".." || e === "projects" || isDirCreated.get(e) === true;
      const owner = e === ".." ? "root  root " : "guest guest";
      const size = isDir ? 4096 : pick(rng, [220, 807, 1024, 1943, 3526, 5842]);
      rows.push(`${isDir ? "drwxr-xr-x" : "-rw-r--r--"} ${isDir ? randint(rng, 2, 5) : 1} ${owner} ${String(size).padStart(5, " ")} ${shortDate(rng)} ${e}`);
    }
    return rec("ls -la", rows.join("\n"));
  };

  const recs = [];
  if (chance(rng, 0.4)) recs.push(ls()); // "before" listing
  const mutations = randint(rng, 1, 4);
  for (let m = 0; m < mutations; m++) {
    const roll = rng.random();
    if (roll < 0.42) {
      const d = randName(rng).replace(/\.\w+$/, ""); // dirs: no extension
      created.push(d);
      isDirCreated.set(d, true);
      recs.push(rec(`mkdir ${d}`, ""));
    } else if (roll < 0.84) {
      const f = randName(rng);
      created.push(f);
      isDirCreated.set(f, false);
      recs.push(rec(`touch ${f}`, ""));
    } else if (created.length > 0 && chance(rng, 0.7)) {
      const f = pick(rng, created);
      created.splice(created.indexOf(f), 1);
      recs.push(rec(`rm ${isDirCreated.get(f) ? "-r " : ""}${f}`, ""));
    } else {
      const f = pick(rng, ["notes.txt", "todo.md"]);
      if (chance(rng, 0.5)) {
        const i = BASE.indexOf(f);
        if (i >= 0) BASE.splice(i, 1);
        recs.push(rec(`rm ${f}`, ""));
      } else {
        recs.push(rec(`rm ghost.txt`, `rm: cannot remove 'ghost.txt': No such file or directory`));
      }
    }
    recs.push(chance(rng, 0.7) ? ls() : lsLa()); // payoff after EVERY mutation
  }
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
      // cat: known files, invented-but-plausible, and errors
      const v = rng.random();
      if (v < 0.72) {
        const name = pick(rng, Object.keys(FILES));
        yield rec(`cat ${name}`, FILES[name](rng));
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
      if (v < 0.3) yield rec(`touch ${f}`, "");
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
