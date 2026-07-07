// vm/run.mjs — capture real boot/log/reboot content from a Lima Debian VM.
//
// Runs the VM_COMMANDS inside the VM (dmesg, journalctl, systemd, boot identity),
// then reboots it and extracts the full shutdown -> boot -> login lifecycle from
// Lima's serial-console log. Everything is scrubbed of host identity (username,
// hostname, UUIDs, IPs) and normalized to the bity persona.
//
//   node corpus/capture/vm/run.mjs     (VM "bity-vm" must be running)

import { readFileSync, writeFileSync, statSync, mkdirSync } from "node:fs";
import { homedir } from "node:os";
import { resolve, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import { RUNNER, sanitize, isBinaryish, normalizeAscii, vmScrub, parseBatch } from "../lib.mjs";
import { VM_COMMANDS, VM_TIMEOUT } from "./commands.mjs";

const VM = "bity-vm";
const PROMPT = "guest@bity:~$ ";
const SYSTEM = "lima/debian-13 (qemu,aarch64)";
const MAXBUF = 256 * 1024 * 1024;
const dec = new TextDecoder("utf-8", { fatal: false });
const decode = (b) => dec.decode(b ?? Buffer.alloc(0));
const HERE = dirname(fileURLToPath(import.meta.url));
const OUT = resolve(HERE, "../../data");
const log = (...m) => console.log("[vm]", ...m);

const hostUser = decode(spawnSync("whoami").stdout).trim();
const hostname = decode(spawnSync("limactl", ["shell", VM, "hostname"]).stdout).trim() || "lima-bity-vm";
const scrub = (t) => vmScrub(sanitize(t), { hostUser, hostname });

function shell(cmdString, { input, sudo, timeout } = {}) {
  const pre = sudo ? ["sudo", "bash", "-c", cmdString] : ["bash", "-c", cmdString];
  return decode(spawnSync("limactl", ["shell", VM, ...pre], { input, timeout, maxBuffer: MAXBUF }).stdout);
}
function pushRunner(cmds) {
  shell("mkdir -p /tmp/bity");
  shell("cat > /tmp/bity/runner.sh", { input: RUNNER });
  shell("cat > /tmp/bity/cmds.txt", { input: cmds.join("\n") + "\n" });
}
function runRunner(cmds) {
  pushRunner(cmds);
  const raw = decode(spawnSync("limactl",
    ["shell", VM, "sudo", "env", `BITY_TIMEOUT=${VM_TIMEOUT}`, "bash", "/tmp/bity/runner.sh"],
    { maxBuffer: MAXBUF }).stdout);
  return parseBatch(raw, cmds);
}
const alive = () => spawnSync("limactl", ["shell", VM, "true"], { timeout: 20000 }).status === 0;
const sleep = (s) => spawnSync("sleep", [String(s)]);

// ---- records ----
const records = [];
let txt = "";
function add(cat, cmd, exit, output) {
  const clean = scrub(output);
  records.push({ system: SYSTEM, cat, cmd, exit, output: clean });
  const t = normalizeAscii(clean);
  txt += PROMPT + cmd + "\n" + (t.endsWith("\n") || t === "" ? t : t + "\n");
}

// ---- 1) command captures ----
log(`user=${hostUser} host=${hostname} -> persona guest@bity`);
log(`running ${VM_COMMANDS.length} capture commands ...`);
const catByCmd = new Map(VM_COMMANDS.map((c) => [c.cmd, c.cat]));
for (const r of runRunner(VM_COMMANDS.map((c) => c.cmd))) {
  if (isBinaryish(r.output)) continue;
  add(catByCmd.get(r.cmd) ?? "vm", r.cmd, r.exit, r.output);
}

// ---- 2) reboot lifecycle from the serial console ----
const serialPath = join(homedir(), ".lima", VM, "serial.log");
let before = 0;
try { before = statSync(serialPath).size; } catch {}
// persistent journal so `journalctl -b -1` survives the reboot
shell("mkdir -p /var/log/journal && systemd-tmpfiles --create --prefix /var/log/journal >/dev/null 2>&1; systemctl restart systemd-journald; sync", { sudo: true });
log("rebooting VM to capture shutdown -> boot -> login ...");
shell("systemctl reboot", { sudo: true, timeout: 30000 });
sleep(8);
let back = false;
for (let i = 0; i < 50 && !back; i++) { if (alive()) back = true; else sleep(3); }
log(back ? "VM back up; flushing serial console ..." : "VM slow to return; capturing serial so far");
sleep(6);

let tail = "";
// UTF-8, not latin1: systemd truncates status lines with a multi-byte ellipsis
// (e2 80 a6); latin1 would shred it before normalizeAscii can map it to "...".
try { tail = readFileSync(serialPath).slice(before).toString("utf8"); } catch {}
const rebootOut = scrub(tail).replace(/\n{3,}/g, "\n\n").trim();
if (rebootOut) {
  records.push({ system: SYSTEM, cat: "reboot", cmd: "sudo reboot", exit: 0, output: rebootOut });
  const t = normalizeAscii(rebootOut).split("\n").slice(0, 400).join("\n");
  txt += PROMPT + "sudo reboot\n" + t + "\n";
}

// ---- 3) post-reboot: structured shutdown tail + fresh boot identity ----
if (back) {
  log("capturing post-reboot journal + boot identity ...");
  const post = [
    "journalctl -b -1 --no-pager | tail -n 120", // the shutdown we just performed
    "dmesg | head -n 30", "uptime", "uptime -s", "who -b", "last reboot --no-pager | head -n 8",
  ];
  for (const r of runRunner(post)) {
    if (isBinaryish(r.output)) continue;
    add("boot", r.cmd, r.exit, r.output);
  }
}

// ---- write + report ----
mkdirSync(OUT, { recursive: true });
writeFileSync(resolve(OUT, "debian-vm.jsonl"), records.map((r) => JSON.stringify(r)).join("\n") + "\n");
writeFileSync(resolve(OUT, "debian-vm.corpus.txt"), txt);

const byCat = {};
for (const r of records) byCat[r.cat] = (byCat[r.cat] ?? 0) + 1;
console.log("\n===== VM capture summary =====");
console.log(`records: ${records.length} | size: ${(Buffer.byteLength(txt) / 1024).toFixed(0)}KB | vocab: ${new Set(txt).size}`);
console.log(`by category: ${Object.entries(byCat).map(([k, v]) => `${k}=${v}`).join("  ")}`);
console.log(`reboot lifecycle: ${rebootOut.length} chars captured`);
console.log("\n----- reboot sample (first 34 lines) -----");
console.log(normalizeAscii(rebootOut).split("\n").slice(0, 34).join("\n"));
