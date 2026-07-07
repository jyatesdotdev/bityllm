// Generator framework: seeded RNG + shared helpers + the bity persona.
// Every output format here is copied from real Debian captures (corpus/data/
// debian.jsonl) — generators only randomize the variable slots.

export class RNG {
  constructor(seed) { this.s = seed >>> 0; }
  random() {
    this.s = (this.s + 0x6d2b79f5) >>> 0;
    let t = this.s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
}

export const PROMPT = "guest@bity:~$ ";
/** prompt for a cwd like "~" or "~/pod" — the path lives in the prompt */
export const promptFor = (cwd) => `guest@bity:${cwd}$ `;
export const pick = (rng, arr) => arr[Math.floor(rng.random() * arr.length)];
export const randint = (rng, lo, hi) => lo + Math.floor(rng.random() * (hi - lo)); // [lo, hi)
export const chance = (rng, p) => rng.random() < p;

export const HOSTS = [
  "bity.dev", "example.com", "github.com", "debian.org", "kernel.org",
  "google.com", "npmjs.org", "archive.org", "wikipedia.org", "localhost",
  "eff.org", "sr.ht", "tilde.town", "mirror.bity.dev", "api.bity.dev",
];

export const ip = (rng) =>
  chance(rng, 0.15) ? "127.0.0.1" : `${pick(rng, [93, 140, 151, 172, 185, 198, 203, 45, 66, 88])}.${randint(rng, 1, 255)}.${randint(rng, 0, 255)}.${randint(rng, 1, 255)}`;

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const DOW = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

export function clock(rng) {
  return `${String(randint(rng, 0, 24)).padStart(2, "0")}:${String(randint(rng, 0, 60)).padStart(2, "0")}:${String(randint(rng, 0, 60)).padStart(2, "0")}`;
}

/** "Jul  3 19:40" — ls/who style */
export function shortDate(rng) {
  const d = randint(rng, 1, 29);
  return `${pick(rng, MONTHS)} ${String(d).padStart(2, " ")} ${String(randint(rng, 0, 24)).padStart(2, "0")}:${String(randint(rng, 0, 60)).padStart(2, "0")}`;
}

/** "Sun Jul  5 04:40:05 UTC 2026" — date(1) style */
export function longDate(rng) {
  return `${pick(rng, DOW)} ${pick(rng, MONTHS)} ${String(randint(rng, 1, 29)).padStart(2, " ")} ${clock(rng)} UTC ${randint(rng, 2024, 2028)}`;
}

/** The eclectic dream machine: two kernels it may report (both from real captures). */
export const KERNELS = [
  { uname: "6.12.74+deb13+1-cloud-arm64 #1 SMP Debian 6.12.74-2 (2026-03-08) aarch64", arch: "aarch64", os: "Debian GNU/Linux 13 (trixie)" },
  { uname: "6.17.0-40-generic #40-Ubuntu SMP PREEMPT_DYNAMIC Fri Jun 19 16:42:13 UTC 2026 x86_64", arch: "x86_64", os: "Debian GNU/Linux 13 (trixie)" },
];

/** A record is one command + its (possibly empty) output. */
export const rec = (cmd, output) => ({ cmd, output });

/** Random pronounceable string — for copy-curriculum tasks where reproducing
 *  the argument from context is the ONLY way to predict the output. */
const CSYL = ["ka", "lo", "mir", "ten", "zu", "bel", "rin", "dov", "pax", "qui", "fen", "sha", "gro", "vim", "ost", "tra", "ble", "nod", "yur", "cas"];
export function randWord(rng) {
  let s = "";
  const n = randint(rng, 2, 4);
  for (let i = 0; i < n; i++) s += pick(rng, CSYL);
  return s;
}

// v5: UNIFORM random strings — syllable-only drills taught a copier that only
// works over syllable transitions (echo fenlodov ✓, echo zanzibar ✗). Universal
// copying must be trained on universal characters.
const LOWER = "abcdefghijklmnopqrstuvwxyz";
const ALNUM = LOWER + "0123456789";
export function randChars(rng) {
  const alpha = chance(rng, 0.75) ? LOWER : ALNUM;
  const n = randint(rng, 3, 11);
  let s = "";
  for (let i = 0; i < n; i++) s += alpha[randint(rng, 0, alpha.length)];
  return s;
}

/** copy-drill argument: mostly uniform-random, some pronounceable, some real-ish */
export function copyArg(rng) {
  const r = rng.random();
  if (r < 0.6) return randChars(rng);
  if (r < 0.85) return randWord(rng);
  return pick(rng, ["backup", "report", "photos", "main", "config", "notes2", "hello", "data", "src", "temp"]);
}

/** Plausible file contents by extension — teaches `cat <any-name>` to dream
 *  content matching the extension instead of defaulting to ENOENT. */
export function contentFor(rng, name) {
  const ext = (name.match(/\.(\w+)$/) || [])[1] ?? "";
  const w = () => randWord(rng);
  switch (ext) {
    case "csv":
      return `id,name,value\n1,${w()},${randint(rng, 1, 99)}\n2,${w()},${randint(rng, 1, 99)}\n3,${w()},${randint(rng, 1, 99)}`;
    case "py":
      return pick(rng, [
        `#!/usr/bin/env python3\nprint("${w()}")`,
        `def main():\n    return ${randint(rng, 0, 42)}\n\nif __name__ == "__main__":\n    main()`,
      ]);
    case "sh":
      return `#!/bin/sh\necho "${w()}"\nexit 0`;
    case "log":
      return `[${randint(rng, 10, 23)}:${String(randint(rng, 10, 59))}:0${randint(rng, 1, 9)}] started\n[${randint(rng, 10, 23)}:${String(randint(rng, 10, 59))}:1${randint(rng, 1, 9)}] ok: ${w()}\n[${randint(rng, 10, 23)}:${String(randint(rng, 10, 59))}:2${randint(rng, 1, 9)}] done`;
    case "md":
      return `# ${w()}\n\n- ${w()}\n- ${w()}`;
    case "yaml":
    case "yml":
      return `name: ${w()}\nport: ${pick(rng, [8080, 3000, 9090, 8143])}\nenabled: ${pick(rng, ["true", "false"])}`;
    case "html":
      return `<html>\n<body>\n<h1>${w()}</h1>\n</body>\n</html>`;
    case "json":
      return `{"name": "${w()}", "count": ${randint(rng, 1, 99)}}`;
    case "txt":
    default:
      return pick(rng, [
        `${w()} ${w()} ${w()}`,
        `remember: ${w()}\nthen ${w()}`,
        `${w()}\n${w()} ${w()}`,
      ]);
  }
}
