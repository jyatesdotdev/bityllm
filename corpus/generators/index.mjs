// Generator registry: relative weights control the corpus mix.
//
// HYBRID SPLIT (2026-07-09): deterministic FS/text/identity commands now run in
// the programmatic CORE (src/infer/{vfs,coreutils,shell-exec}.ts), NOT the model
// — so their generators (fs, fs-session, sys, copy) were REMOVED: the model no
// longer needs to learn ls/cat/pwd/whoami/mkdir/echo output. Synthetic now only
// tops up the DREAMED set with controlled variety, and teaches the graceful
// `command not found` fallback. Real breadth comes from the capture (build.mjs),
// which is itself filtered to drop CORE commands.

import { netGen } from "./net.mjs";
import { gitGen } from "./git.mjs";
import { funGen } from "./fun.mjs";
import { unknownGen } from "./unknown.mjs";

export { RNG, PROMPT } from "./lib.mjs";

export const GENERATORS = [
  { name: "net", weight: 0.32, gen: netGen },         // ping/traceroute/curl — the headline act
  { name: "git", weight: 0.21, gen: gitGen },         // git status/log/diff variety
  { name: "fun", weight: 0.21, gen: funGen },         // fortune/cowsay/neofetch charm
  { name: "unknown", weight: 0.26, gen: unknownGen }, // graceful command-not-found fallback (~8% of corpus)
];

// no block generators: the stateful mkdir/touch → ls sessions are now the
// programmatic CORE's job (a real VFS), so there is nothing left to dream in a block.
export const BLOCK_GENERATORS = [];
