// Graceful-unknown: real-but-not-installed tools, fat-finger typos, and random
// gibberish all resolve to a clean `bash: <cmd>: command not found`.
//
// In the hybrid shell, everything not in the programmatic CORE is dreamed by the
// model. This generator teaches the model the correct FALLBACK — answer
// "command not found" for what it genuinely doesn't know — instead of
// hallucinating plausible-looking output for a command that doesn't exist. This
// is the single highest-leverage breadth fix (it kills most out-of-band gibberish).

import { pick, randint, chance, rec } from "./lib.mjs";

// developer tools that are NOT on a base Debian slim box → command not found.
// (Deliberately excludes anything in CORE or the DREAMED set, and anything the
// container capture actually installed.)
const NOT_INSTALLED = [
  "kubectl", "terraform", "cargo", "rustc", "rustup", "go", "gofmt", "javac", "java",
  "mvn", "gradle", "php", "ruby", "gem", "bundle", "perl", "emacs", "tmux", "jq", "yq",
  "aws", "gcloud", "az", "helm", "ansible", "ansible-playbook", "vagrant", "brew",
  "yarn", "pnpm", "deno", "bun", "zig", "ghc", "cabal", "sbt", "lua", "julia", "code",
  "subl", "nvim", "rg", "fd", "bat", "eza", "fzf", "tldr", "httpie", "aria2c", "ncdu",
  "glances", "btop", "delta", "kustomize", "podman", "kind", "minikube", "packer",
  "consul", "nomad", "vault", "stern", "k9s", "kubens", "kubectx", "dotnet", "cmake",
  "ninja", "clang", "clang-format", "valgrind", "gdb", "lldb", "strace", "hyperfine",
  "tokei", "scc", "shellcheck", "prettier", "eslint", "tsc", "webpack", "vite",
];

// common fat-finger typos of real commands → command not found
const TYPOS = [
  "sl", "gti", "grpe", "lls", "mkdri", "claer", "whcih", "cta", "les", "pign", "tial",
  "haed", "suod", "gerp", "chmdo", "tpuch", "mvoe", "ecoh", "mkidr", "pytohn", "gut",
  "dokcer", "systemclt", "aptget", "sudp", "exti", "clera", "amn", "hepl", "veim",
];

// random lowercase gibberish → command not found
function garbage(rng) {
  const cs = "abcdefghijklmnopqrstuvwxyz";
  let s = "";
  for (let i = 0, n = randint(rng, 3, 10); i < n; i++) s += cs[randint(rng, 0, cs.length)];
  return s;
}

const ARGS = [
  "", "", " --help", " -v", " --version", " status", " init", " build", " run",
  " install foo", " -la", " --list", " deploy", " up", " start", " test .", " get pods",
];

export function* unknownGen(rng) {
  while (true) {
    const roll = rng.random();
    const cmd = roll < 0.5 ? pick(rng, NOT_INSTALLED) : roll < 0.8 ? pick(rng, TYPOS) : garbage(rng);
    const line = cmd + (chance(rng, 0.45) ? pick(rng, ARGS) : "");
    const head = line.trim().split(/\s+/)[0]; // bash reports the pipeline head
    yield rec(line, `bash: ${head}: command not found`);
  }
}
