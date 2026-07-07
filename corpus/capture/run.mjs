// run.mjs — capture harness orchestrator.
//
// Boots a throwaway Debian container, installs a toolset, then harvests as much
// real command output as possible (sanitized) until we reach a target size.
// Writes corpus/data/debian.jsonl (structured) and debian.corpus.txt (training).
//
//   node corpus/capture/run.mjs [--mb 4] [--batch 250] [--timeout 8]
//                               [--image debian:stable-slim] [--fresh] [--clean]

import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { PERSONA, TOOLSET, STATIC, HARVEST } from "./commands.mjs";
import {
  RUNNER, containerRunning, removeContainer, dockerCapture, dockerInherit,
  sh, writeContainerFile, runBatch, sanitize, isBinaryish, normalizeAscii, nonAsciiRatio,
} from "./lib.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const OUT_DIR = resolve(HERE, "../data");
const CONTAINER = "bity-capture";

// ---- args -------------------------------------------------------------------
const args = process.argv.slice(2);
const flag = (name) => args.includes(name);
const opt = (name, def) => {
  const i = args.indexOf(name);
  return i >= 0 && args[i + 1] ? args[i + 1] : def;
};
const TARGET = Math.round(parseFloat(opt("--mb", "4")) * 1024 * 1024);
const BATCH = parseInt(opt("--batch", "250"), 10);
const TIMEOUT = parseInt(opt("--timeout", "8"), 10);
const IMAGE = opt("--image", "debian:stable-slim");

const log = (...m) => console.log("[capture]", ...m);
const kb = (n) => (n / 1024).toFixed(0) + "KB";

// ---- clean-only mode --------------------------------------------------------
if (flag("--clean")) {
  log(`removing container ${CONTAINER} ...`);
  removeContainer(CONTAINER);
  log("done.");
  process.exit(0);
}

// ---- container setup --------------------------------------------------------
function setup() {
  if (flag("--fresh")) removeContainer(CONTAINER);

  if (!containerRunning(CONTAINER)) {
    log(`starting ${IMAGE} as "${CONTAINER}" (host=${PERSONA.host}) ...`);
    removeContainer(CONTAINER);
    const r = dockerCapture([
      "run", "-d", "--name", CONTAINER, "--hostname", PERSONA.host, IMAGE, "sleep", "infinity",
    ]);
    if (r.code !== 0) {
      console.error("failed to start container:\n" + r.stderr);
      process.exit(1);
    }

    log(`installing toolset (${TOOLSET.length} packages) — this takes a few minutes ...`);
    // slim images strip /usr/share/man + /usr/share/doc via a dpkg exclude; drop it so
    // freshly-installed packages bring their man pages (our biggest text source).
    sh(CONTAINER, "rm -f /etc/dpkg/dpkg.cfg.d/docker");
    dockerInherit(["exec", CONTAINER, "bash", "-c", "apt-get update"]);
    // best-effort, per package: a tool missing on this Debian release (e.g. neofetch
    // on newer releases) is skipped instead of aborting the whole install.
    dockerInherit([
      "exec", "-e", "DEBIAN_FRONTEND=noninteractive", CONTAINER, "bash", "-c",
      `for p in ${TOOLSET.join(" ")}; do ` +
        `apt-get install -y --no-install-recommends "$p" >/dev/null 2>&1 ` +
        `&& echo "  + $p" || echo "  - $p (unavailable)"; done`,
    ]);
    // restore man pages for base packages that shipped stripped in the slim image
    log("restoring man pages for base packages ...");
    dockerInherit([
      "exec", "-e", "DEBIAN_FRONTEND=noninteractive", CONTAINER, "bash", "-c",
      "apt-get install -y --reinstall coreutils util-linux findutils grep sed gawk " +
        "tar gzip hostname bsdutils bash dash >/dev/null 2>&1 || true; echo '  man pages restored'",
    ]);

    log("creating guest user + workspace ...");
    sh(CONTAINER, "id -u guest >/dev/null 2>&1 || useradd -m -s /bin/bash guest");
    sh(CONTAINER, "mkdir -p /tmp/bity && chmod 777 /tmp/bity");
    // a small invented home so ls/cat of ~ look lived-in
    sh(CONTAINER, "install -d -o guest -g guest /home/guest/projects /home/guest/.config");
    sh(CONTAINER, "echo 'remember to feed the model' > /home/guest/notes.txt && chown guest:guest /home/guest/notes.txt");
    sh(CONTAINER, "echo '# bityllm' > /home/guest/projects/README.md && chown -R guest:guest /home/guest/projects");
  } else {
    log(`reusing running container "${CONTAINER}" (use --fresh to rebuild).`);
  }

  writeContainerFile(CONTAINER, "/tmp/bity/runner.sh", RUNNER);
}

// ---- build the command plan -------------------------------------------------
function buildPlan() {
  const seen = new Set();
  const push = (list, item) => {
    if (!item.cmd || item.cmd.includes("\n") || seen.has(item.cmd)) return;
    seen.add(item.cmd);
    list.push(item);
  };

  const curated = [];
  for (const item of STATIC) push(curated, item);

  // expand each harvest category by querying the container
  const buckets = [];
  for (const spec of HARVEST) {
    const lines = sh(CONTAINER, spec.query, "guest")
      .split("\n").map((s) => s.trim()).filter(Boolean);
    const bucket = [];
    for (const line of lines) {
      if (bucket.length >= spec.limit) break;
      for (const cmd of spec.make(line)) push(bucket, { cat: spec.cat, cmd });
    }
    log(`  harvested ${spec.cat}: ${bucket.length} commands (from ${lines.length} items)`);
    buckets.push(bucket);
  }

  // interleave harvest buckets (round-robin) so a truncated run stays diverse
  const interleaved = [];
  for (let i = 0; buckets.some((b) => i < b.length); i++) {
    for (const b of buckets) if (i < b.length) interleaved.push(b[i]);
  }

  return [...curated, ...interleaved];
}

// ---- capture loop -----------------------------------------------------------
function capture(plan) {
  const records = [];
  let txt = "";
  let bytes = 0;
  const byCat = {};
  let dropped = 0;
  let skippedTxt = 0;

  for (let i = 0; i < plan.length && bytes < TARGET; i += BATCH) {
    const chunk = plan.slice(i, i + BATCH);
    const results = runBatch(CONTAINER, chunk.map((c) => c.cmd), TIMEOUT);
    const catByCmd = new Map(chunk.map((c) => [c.cmd, c.cat]));

    let added = 0;
    for (const r of results) {
      if (isBinaryish(r.output)) { dropped++; continue; }
      const output = sanitize(r.output);
      const cat = catByCmd.get(r.cmd) ?? "misc";
      records.push({ system: IMAGE, cat, cmd: r.cmd, exit: r.exit, output }); // JSONL stays faithful
      if (nonAsciiRatio(output) > 0.06) { skippedTxt++; continue; } // drop exotic charset tables from training text
      const clean = normalizeAscii(output);
      const block = PERSONA.prompt + r.cmd + "\n" + (clean.endsWith("\n") || clean === "" ? clean : clean + "\n");
      txt += block;
      bytes += Buffer.byteLength(block);
      byCat[cat] = (byCat[cat] ?? 0) + 1;
      added++;
      if (bytes >= TARGET) break;
    }
    log(`batch ${i / BATCH + 1}: +${added} records, ${kb(bytes)} / ${kb(TARGET)}`);
  }

  return { records, txt, bytes, byCat, dropped, skippedTxt };
}

// ---- main -------------------------------------------------------------------
setup();
log("building command plan ...");
const plan = buildPlan();
log(`plan: ${plan.length} unique commands. capturing up to ${kb(TARGET)} ...`);

const { records, txt, bytes, byCat, dropped, skippedTxt } = capture(plan);
const txtRecords = Object.values(byCat).reduce((a, b) => a + b, 0);

mkdirSync(OUT_DIR, { recursive: true });
const jsonlPath = resolve(OUT_DIR, "debian.jsonl");
const txtPath = resolve(OUT_DIR, "debian.corpus.txt");
writeFileSync(jsonlPath, records.map((r) => JSON.stringify(r)).join("\n") + "\n");
writeFileSync(txtPath, txt);

console.log("\n===== capture summary =====");
console.log(`jsonl records: ${records.length}  (dropped ${dropped} binary/garbage)`);
console.log(`train records: ${txtRecords}  (excluded ${skippedTxt} exotic-charset from .txt)`);
console.log(`corpus size  : ${(bytes / 1024 / 1024).toFixed(2)} MB`);
console.log(`by category  : ${Object.entries(byCat).map(([k, v]) => `${k}=${v}`).join("  ")}`);
console.log(`vocab (chars): ${new Set(txt).size}`);
console.log(`jsonl        : ${jsonlPath}`);
console.log(`corpus       : ${txtPath}`);
console.log(`container    : ${CONTAINER} kept running (npm run capture:clean to remove)`);
console.log("\n----- sample (first ~28 lines) -----");
console.log(txt.split("\n").slice(0, 28).join("\n"));
