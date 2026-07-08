// git binaries: status, log, diff --stat, branch, pull/push flavor.

import { pick, randint, chance, rec } from "./lib.mjs";

const FILES = ["src/index.ts", "src/train.ts", "README.md", "package.json", "corpus/build.mjs", "src/nn/gpt.ts", "test/model.test.ts", "DESIGN.md"];
const MSGS = [
  "fix the race condition in the parallel trainer",
  "add ping generator",
  "initial commit",
  "train nano on the terminal corpus",
  "wire up checkpoint round-trip",
  "make grad-checks pass in f64",
  "bump block size to 128",
  "capture harness: strip ansi properly",
  "add cowsay to the fun pack",
];
const AUTHORS = ["guest <guest@bity.dev>"];
const BRANCHES = ["main", "dev", "feat/terminal", "fix/atomics-race"];

const hash = (rng) => Array.from({ length: 7 }, () => "0123456789abcdef"[randint(rng, 0, 16)]).join("");

function status(rng) {
  const branch = pick(rng, BRANCHES);
  if (chance(rng, 0.45))
    return `On branch ${branch}\nYour branch is up to date with 'origin/${branch}'.\n\nnothing to commit, working tree clean`;
  const lines = [`On branch ${branch}`];
  if (chance(rng, 0.5)) lines.push(`Your branch is ahead of 'origin/${branch}' by ${randint(rng, 1, 4)} commit${chance(rng, 0.5) ? "s" : ""}.`);
  lines.push("", "Changes not staged for commit:", '  (use "git add <file>..." to update what will be committed)');
  for (let i = 0; i < randint(rng, 1, 4); i++) lines.push(`\tmodified:   ${pick(rng, FILES)}`);
  if (chance(rng, 0.4)) {
    lines.push("", "Untracked files:", '  (use "git add <file>..." to include in what will be committed)');
    lines.push(`\t${pick(rng, ["notes.txt", "models/", "scratch.ts", "corpus/data/"])}`);
  }
  lines.push("", 'no changes added to commit (use "git add" and/or "git commit -a")');
  return lines.join("\n");
}

function log(rng) {
  const n = randint(rng, 2, 5);
  const out = [];
  for (let i = 0; i < n; i++) {
    out.push(`commit ${hash(rng)}${hash(rng)}${hash(rng)}${hash(rng)}${hash(rng)}${hash(rng)[0]}`.slice(0, 47));
    out.push(`Author: ${pick(rng, AUTHORS)}`);
    out.push(`Date:   ${pick(rng, ["Mon", "Tue", "Fri", "Sat"])} Jul ${randint(rng, 1, 28)} ${String(randint(rng, 8, 23)).padStart(2, "0")}:${String(randint(rng, 0, 60)).padStart(2, "0")}:04 2026 +0000`);
    out.push("", `    ${pick(rng, MSGS)}`, "");
  }
  return out.join("\n").trimEnd();
}

// toolchain version banners — stable strings a user's first move always hits
const VERSIONS = [
  ["git --version", "git version 2.47.2"],
  ["node -v", "v20.19.4"],
  ["node --version", "v20.19.4"],
  ["npm --version", "10.8.2"],
  ["npm -v", "10.8.2"],
  ["python3 --version", "Python 3.13.5"],
  ["python -V", "Python 3.13.5"],
  ["pip3 --version", "pip 25.0.1 from /usr/lib/python3/dist-packages/pip (python 3.13)"],
  ["make --version", "GNU Make 4.4.1\nBuilt for aarch64-unknown-linux-gnu\nCopyright (C) 1988-2023 Free Software Foundation, Inc."],
  ["gcc --version", "gcc (Debian 14.2.0-8) 14.2.0\nCopyright (C) 2024 Free Software Foundation, Inc.\nThis is free software; see the source for copying conditions.  There is NO\nwarranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE."],
  ["tsc --version", "Version 5.7.3"],
  ["deno --version", "deno 2.1.4 (stable, release, aarch64-apple-darwin)\nv8 13.0.245.12-rusty\ntypescript 5.6.2"],
];

export function* gitGen(rng) {
  for (;;) {
    const r = rng.random();
    if (r < 0.34) yield rec("git status", status(rng));
    else if (r < 0.5) yield rec(chance(rng, 0.5) ? "git log" : "git log --oneline -5",
      chance(rng, 0.5) ? log(rng) : Array.from({ length: 5 }, () => `${hash(rng)} ${pick(rng, MSGS)}`).join("\n"));
    else if (r < 0.62) yield rec("git branch", BRANCHES.map((b, i) => (i === 0 ? `* ${b}` : `  ${b}`)).join("\n"));
    else if (r < 0.72) yield rec("git diff --stat",
      Array.from({ length: randint(rng, 1, 3) }, () => ` ${pick(rng, FILES).padEnd(22)} | ${randint(rng, 1, 40)} ${"+".repeat(randint(rng, 1, 8))}${"-".repeat(randint(rng, 0, 4))}`).join("\n") +
      `\n ${randint(rng, 1, 3)} file${chance(rng, 0.5) ? "s" : ""} changed, ${randint(rng, 2, 80)} insertions(+), ${randint(rng, 0, 30)} deletions(-)`);
    else if (r < 0.8) {
      // version banners (git + the toolchain)
      const [cmd, out] = pick(rng, VERSIONS);
      yield rec(cmd, out);
    } else if (r < 0.87) {
      // git commit -m — plausible hash + change summary
      const files = randint(rng, 1, 3), ins = randint(rng, 2, 60), del = randint(rng, 0, 20);
      yield rec(`git commit -m "${pick(rng, MSGS)}"`, `[${pick(rng, BRANCHES)} ${hash(rng)}] ${pick(rng, MSGS)}\n ${files} file${files > 1 ? "s" : ""} changed, ${ins} insertion${ins > 1 ? "s" : ""}(+)${del ? `, ${del} deletion${del > 1 ? "s" : ""}(-)` : ""}`);
    } else if (r < 0.94) yield rec("git pull", chance(rng, 0.6) ? "Already up to date." : `Updating ${hash(rng)}..${hash(rng)}\nFast-forward\n ${pick(rng, FILES)} | ${randint(rng, 2, 30)} ++++----\n 1 file changed`);
    else yield rec("git push", chance(rng, 0.5) ? "Everything up-to-date" : `To github.com:guest/bityllm.git\n   ${hash(rng)}..${hash(rng)}  main -> main`);
  }
}
