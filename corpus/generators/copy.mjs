// Copy curriculum (corpus v4): dense, short-range argument-echo tasks whose
// outputs are predictable ONLY by copying from context — designed to force an
// induction/copy circuit to form. Arguments are random strings (memorization
// impossible). Once the mechanism exists at echo-range, it should generalize
// to ping headers and stateful ls listings (positional, not lexical).

import { pick, randint, chance, copyArg as randWord, contentFor, rec } from "./lib.mjs";

const EXT = ["", ".txt", ".log", ".md", ".csv", ".sh", ".py"];
const name = (rng) => randWord(rng) + pick(rng, EXT);

export function* copyGen(rng) {
  for (;;) {
    const r = rng.random();
    if (r < 0.34) {
      // echo — the purest copy task (distance ~6 chars)
      const v = rng.random();
      if (v < 0.4) {
        const w = randWord(rng);
        yield rec(`echo ${w}`, w);
      } else if (v < 0.75) {
        const words = Array.from({ length: randint(rng, 2, 4) }, () => randWord(rng)).join(" ");
        yield rec(`echo ${words}`, words);
      } else {
        // humans quote things: quotes strip, contents echo verbatim
        const words = Array.from({ length: randint(rng, 1, 3) }, () => randWord(rng)).join(" ");
        const q = chance(rng, 0.5) ? `"` : `'`;
        yield rec(`echo ${q}${words}${q}`, words);
      }
    } else if (r < 0.55) {
      // file errors that echo their argument
      const f = name(rng);
      const v = rng.random();
      if (v < 0.2) yield rec(`cat ${f}`, `cat: ${f}: No such file or directory`);
      else if (v < 0.4) yield rec(`cat ${f}`, contentFor(rng, f)); // dreamed contents (still echoes name context)
      else if (v < 0.65) yield rec(`rm ${f}`, `rm: cannot remove '${f}': No such file or directory`);
      else if (v < 0.85) yield rec(`ls ${f}`, `ls: cannot access '${f}': No such file or directory`);
      else {
        const d = randWord(rng);
        yield rec(`cd ${d}`, `bash: cd: ${d}: No such file or directory`);
      }
    } else if (r < 0.72) {
      // command-not-found echoes the command itself
      const c = randWord(rng);
      yield rec(c, `bash: ${c}: command not found`);
    } else if (r < 0.86) {
      // network errors / headers that echo the host
      const h = `${randWord(rng)}.${pick(rng, ["dev", "com", "org", "net", "io"])}`;
      if (chance(rng, 0.5)) yield rec(`ping ${h}`, `ping: ${h}: Name or service not known`);
      else yield rec(`ping -c 1 ${h}`,
        `PING ${h} (10.0.2.15) 56(84) bytes of data.\n64 bytes from 10.0.2.15: icmp_seq=1 ttl=64 time=${(rng.random() * 30 + 1).toFixed(1)} ms\n\n--- ${h} ping statistics ---\n1 packets transmitted, 1 received, 0% packet loss, time 0ms`);
    } else {
      // mkdir/touch feedback forms that echo the argument
      const d = randWord(rng);
      const v = rng.random();
      if (v < 0.5) yield rec(`mkdir ${d}`, chance(rng, 0.5) ? "" : `mkdir: cannot create directory '${d}': File exists`);
      else yield rec(`mkdir -v ${d}`, `mkdir: created directory '${d}'`);
    }
  }
}