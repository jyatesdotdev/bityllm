// Generator registry: relative weights control the corpus mix.

import { fsGen, fsSessionBlock } from "./fs.mjs";
import { sysGen } from "./sys.mjs";
import { netGen } from "./net.mjs";
import { gitGen } from "./git.mjs";
import { funGen } from "./fun.mjs";
import { copyGen } from "./copy.mjs";

export { RNG, PROMPT } from "./lib.mjs";

export const GENERATORS = [
  { name: "fs", weight: 0.14, gen: fsGen },
  { name: "sys", weight: 0.15, gen: sysGen },
  { name: "net", weight: 0.19, gen: netGen },  // ping is the headline act
  { name: "git", weight: 0.09, gen: gitGen },
  { name: "fun", weight: 0.13, gen: funGen },
  { name: "copy", weight: 0.10, gen: copyGen }, // the induction-circuit drill
];

// block generators emit whole coherent sessions (stateful mkdir/touch → ls)
export const BLOCK_GENERATORS = [
  { name: "fs-session", weight: 0.2, block: fsSessionBlock },
];
