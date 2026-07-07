// lib.mjs — docker orchestration, output sanitization, and batch parsing.
// Pure Node built-ins, no dependencies.

import { spawnSync, execFileSync } from "node:child_process";

const MAXBUF = 512 * 1024 * 1024; // batches can be many MB
const RS = "\x1e"; // ASCII Record Separator — our marker delimiter (never in text)
const decoder = new TextDecoder("utf-8", { fatal: false });
const decode = (buf) => decoder.decode(buf ?? Buffer.alloc(0));

// ---- docker helpers ---------------------------------------------------------

// Run a docker subcommand, inheriting stdio (for setup/install progress).
export function dockerInherit(args) {
  execFileSync("docker", args, { stdio: "inherit", maxBuffer: MAXBUF });
}

// Run a docker subcommand, capturing stdout (never throws).
export function dockerCapture(args, input) {
  const r = spawnSync("docker", args, { input, maxBuffer: MAXBUF });
  return { code: r.status ?? -1, stdout: decode(r.stdout), stderr: decode(r.stderr) };
}

export function containerRunning(name) {
  const r = dockerCapture(["inspect", "-f", "{{.State.Running}}", name]);
  return r.code === 0 && r.stdout.trim() === "true";
}

export function removeContainer(name) {
  dockerCapture(["rm", "-f", name]);
}

// Run a bash script inside the container (optionally as a user); returns stdout.
export function sh(container, script, user) {
  const args = ["exec"];
  if (user) args.push("-u", user);
  args.push(container, "bash", "-c", script);
  return dockerCapture(args).stdout;
}

// Write a file into the container by piping content over stdin.
export function writeContainerFile(container, path, content) {
  const r = dockerCapture(["exec", "-i", container, "bash", "-c", `cat > ${path}`], content);
  if (r.code !== 0) throw new Error(`failed to write ${path}: ${r.stderr}`);
}

// The in-container runner: reads one command per line, wraps each in a timeout
// with stdin closed and stderr merged, and brackets its output with RS markers
// carrying the command index and exit code.
export const RUNNER = [
  "#!/usr/bin/env bash",
  "export TERM=dumb PAGER=cat MANPAGER=cat GIT_PAGER=cat SYSTEMD_PAGER=cat",
  "export MANWIDTH=80 COLUMNS=80 LANG=C.UTF-8 LC_ALL=C.UTF-8 DEBIAN_FRONTEND=noninteractive",
  'export PATH="$PATH:/usr/games:/usr/local/games:/sbin:/usr/sbin"',
  "cd /home/guest 2>/dev/null || cd /",
  "RS=$'\\036'",
  "i=0",
  "while IFS= read -r cmd; do",
  '  printf \'\\n%sBITYCMD%s%s%s\\n\' "$RS" "$RS" "$i" "$RS"',
  '  timeout "${BITY_TIMEOUT:-8}" bash -c "$cmd" </dev/null 2>&1 | head -c "${BITY_MAXBYTES:-200000}"',
  "  rc=${PIPESTATUS[0]}",
  '  printf \'\\n%sBITYEND%s%s%s%s%s\\n\' "$RS" "$RS" "$i" "$RS" "$rc" "$RS"',
  "  i=$((i+1))",
  "done < /tmp/bity/cmds.txt",
  "",
].join("\n");

// Run one batch of commands; returns [{ cmd, output, exit }].
export function runBatch(container, cmds, timeoutSec) {
  writeContainerFile(container, "/tmp/bity/cmds.txt", cmds.join("\n") + "\n");
  const args = [
    "exec", "-e", `BITY_TIMEOUT=${timeoutSec}`, "-u", "guest", "-w", "/home/guest",
    container, "bash", "/tmp/bity/runner.sh",
  ];
  const raw = dockerCapture(args).stdout;
  return parseBatch(raw, cmds);
}

// ---- parsing ----------------------------------------------------------------

export function parseBatch(raw, cmds) {
  const re = new RegExp(
    RS + "BITYCMD" + RS + "(\\d+)" + RS + "\\n([\\s\\S]*?)" + RS + "BITYEND" + RS + "\\1" + RS + "(-?\\d+)" + RS,
    "g",
  );
  const out = [];
  let m;
  while ((m = re.exec(raw)) !== null) {
    const idx = Number(m[1]);
    const output = m[2].replace(/\n$/, ""); // drop the newline injected before the END marker
    out.push({ cmd: cmds[idx], output, exit: Number(m[3]) });
  }
  return out;
}

// ---- sanitization -----------------------------------------------------------

// True if the raw (pre-sanitize) output looks like binary/garbage we shouldn't keep.
export function isBinaryish(raw) {
  if (raw.length === 0) return false;
  if (raw.includes("\x00")) return true;
  let bad = 0;
  for (let i = 0; i < raw.length; i++) {
    const c = raw.charCodeAt(i);
    if (c === 9 || c === 10 || c === 27 || c === 8) continue; // tab, lf, esc, backspace = formatting
    if (c < 32 || c === 127 || c === 0xfffd) bad++;
  }
  return bad / raw.length > 0.1;
}

// Render a line the way a terminal actually would: apply carriage-return
// overwrites, backspaces, and erase-line codes instead of just deleting them.
// This turns systemd's progress rewrites ("Starting X..." -> "\r[ OK ] Started X")
// and man's overstrike into their final visible text.
function renderConsole(text) {
  const lines = [];
  for (const raw of text.split("\n")) {
    const buf = [];
    let col = 0;
    for (let i = 0; i < raw.length; i++) {
      const ch = raw[i];
      if (ch === "\r") col = 0;
      else if (ch === "\x08") { if (col > 0) col--; }
      else if (ch === "\x1b" && raw[i + 1] === "[") {
        let j = i + 2;
        while (j < raw.length && /[0-9;?]/.test(raw[j])) j++;
        if (raw[j] === "K" || raw[j] === "k") buf.length = col; // erase to end of line
        i = j;
      } else { buf[col] = ch; col++; }
    }
    lines.push(buf.join(""));
  }
  return lines.join("\n");
}

// Strip ANSI, render console overwrites, remove control junk, and fake any MACs.
export function sanitize(text) {
  let t = text;
  t = t.replace(/\x1b\][\s\S]*?(?:\x07|\x1b\\)/g, ""); // OSC
  t = t.replace(/\x1b\[[0-9;?]*[ -/]*[@-~]/g, (m) => (/[Kk]$/.test(m) ? m : "")); // drop CSI except erase-line
  t = t.replace(/\x1b[()][\s\S]/g, ""); // charset selection
  t = t.replace(/\x1b[@-Z\\-_]/g, ""); // other 2-char escapes
  t = renderConsole(t); // apply \r, \b, and erase-line as a terminal would
  t = t.replace(/\x1b./g, ""); // stray escapes
  t = t.replace(/[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/g, ""); // leftover control bytes (keep \t \n)
  t = t.replace(/\b(?:[0-9a-f]{2}:){5}[0-9a-f]{2}\b/gi, "de:ad:be:ef:00:42"); // MAC -> fake
  t = t.replace(/\n{4,}/g, "\n\n\n"); // collapse big gaps
  return t;
}

// Fraction of characters outside plain ASCII (used to drop exotic charset tables).
export function nonAsciiRatio(text) {
  if (!text.length) return 0;
  let n = 0;
  for (const ch of text) { const c = ch.codePointAt(0); if (c > 126 || c < 9) n++; }
  return n / text.length;
}

// Box-drawing + a few symbols worth keeping so tables/tree output still look right.
const KEEP = new Set([..."│─├└┌┐┘┴┬┼┤╭╮╯╰═║╔╗╚╝╬╠╣╦╩▲▶▼◀●○◦"]);
// Common typographic characters mapped down to ASCII.
const MAP = {
  "‘": "'", "’": "'", "“": '"', "”": '"', "„": '"',
  "–": "-", "—": "-", "‑": "-", "−": "-",
  "…": "...", "×": "x", "÷": "/", "·": ".", "•": "*",
  "©": "(c)", "®": "(r)", "™": "(tm)", "µ": "u", "°": " deg",
  "€": "EUR", "£": "GBP", "¥": "JPY", "¢": "c", " ": " ",
};

// Reduce text to a compact terminal vocabulary: printable ASCII + newline/tab,
// mapped typographic chars, and a small box-drawing whitelist. Everything else drops.
export function normalizeAscii(text) {
  let out = "";
  for (const ch of text) {
    const c = ch.codePointAt(0);
    if (c === 9 || c === 10 || (c >= 32 && c <= 126)) out += ch;
    else if (ch in MAP) out += MAP[ch];
    else if (KEEP.has(ch)) out += ch;
  }
  return out;
}

const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

// Extra scrubbing for VM captures: fake UUIDs / machine-ids / boot-ids, private
// IPs, and — critically — replace the host-derived VM username and hostname with
// the bity persona so no real identity leaks from dmesg/journalctl/logs.
export function vmScrub(text, { hostUser, hostname } = {}) {
  let t = text;
  t = t.replace(/\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b/gi, "1c0ffee5-dead-4bee-8faf-0000deadbeef");
  t = t.replace(/\b[0-9a-f]{32}\b/gi, "0badc0de0badc0de0badc0de0badc0de"); // machine-id / boot-id
  t = t.replace(/\b(?:10|127|169\.254|192\.168|172\.(?:1[6-9]|2\d|3[01]))(?:\.\d{1,3}){1,3}\b/g, "10.0.2.15");
  // only replace distinctive hostnames as whole words (never blanket-replace a
  // short substring like "ai", which would corrupt words such as "available")
  if (hostname && hostname.length >= 4) t = t.replace(new RegExp("\\b" + escapeRe(hostname) + "\\b", "g"), "bity");
  if (hostUser) {
    t = t.split(`/home/${hostUser}`).join("/home/guest");
    t = t.replace(new RegExp(`\\b${escapeRe(hostUser)}\\b`, "g"), "guest");
  }
  t = t.replace(/\/home\/guest\.(?:guest|linux)\b/g, "/home/guest"); // scrub artifact of Lima's user.linux home
  // drop lines that leak the capture tooling itself (Lima guest agent, etc.)
  t = t.split("\n").filter((l) => !/lima/i.test(l) || /America\/Lima/.test(l)).join("\n");
  return t;
}
