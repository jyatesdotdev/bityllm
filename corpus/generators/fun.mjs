// The fun pack: fortune, cowsay, sudo, command-not-found, history, neofetch,
// reboot. cowsay template is byte-exact from the real Debian capture.

import { pick, randint, chance, clock, KERNELS, rec } from "./lib.mjs";

const FORTUNES = [
  "A journey of a thousand miles begins with a single step.",
  "The best way to predict the future is to invent it.",
  "Real programmers count from 0.",
  "There is no place like 127.0.0.1.",
  "A watched pot never boils, but an unwatched build always fails.",
  "It is easier to optimize correct code than to correct optimized code.",
  "The terminal dreams, therefore it is.",
  "Any sufficiently advanced bug is indistinguishable from a feature.",
  "He who has not tested has not shipped.",
  "Simplicity is the ultimate sophistication.",
  "First, solve the problem. Then, write the code.",
  "A tidy home directory is a sign of a misspent life.",
  "The cheapest, fastest, and most reliable components are those that aren't there.",
  "Deleted code is debugged code.",
  "One person's constant is another person's variable.",
  "The best error message is the one that never shows up.",
  "In theory, theory and practice are the same. In practice, they are not.",
  "Weeks of coding can save you hours of planning.",
  "There are two hard things in computer science: cache invalidation, naming things, and off-by-one errors.",
  "Never trust a computer you can't throw out a window.",
  "A ship in harbor is safe, but that is not what ships are built for.",
  "Fortune favors the bold.",
  "To iterate is human, to recurse divine.",
  "The night is long that never finds the day.",
];

function cowsay(msg) {
  const top = " " + "_".repeat(msg.length + 2);
  const bot = " " + "-".repeat(msg.length + 2);
  return [top, `< ${msg} >`, bot,
    "        \\   ^__^",
    "         \\  (oo)\\_______",
    "            (__)\\       )\\/\\",
    "                ||----w |",
    "                ||     ||"].join("\n");
}

const TYPOS = ["sl", "gti", "pign", "lls", "cta", "grpe", "pyhton", "nod", "vmi", "claer", "exot", "hlep", "mkae", "suod"];

function history(rng) {
  const cmds = ["ls -la", "cd projects", "git status", "ping -c 3 bity.dev", "cat notes.txt", "df -h", "vim todo.md", "git log", "uptime", "free -h", "curl -I https://bity.dev", "fortune", "clear"];
  const n = randint(rng, 5, 10);
  const start = randint(rng, 100, 900);
  return Array.from({ length: n }, (_, i) => `  ${start + i}  ${pick(rng, cmds)}`).join("\n");
}

function neofetch(rng) {
  const k = pick(rng, KERNELS);
  const mem = pick(rng, [16, 32]);
  return [
    "guest@bity",
    "----------",
    `OS: ${k.os} ${k.arch}`,
    "Host: QEMU Virtual Machine virt-9.2",
    `Kernel: ${k.uname.split(" ")[0]}`,
    `Uptime: ${randint(rng, 1, 20)} hours, ${randint(rng, 1, 59)} mins`,
    `Packages: ${randint(rng, 300, 700)} (dpkg)`,
    "Shell: bash 5.2.37",
    `CPU: (${pick(rng, [4, 8, 12])})`,
    `Memory: ${randint(rng, 900, 4000)}MiB / ${mem * 1024}MiB`,
  ].join("\n");
}

function reboot(rng) {
  const units = [
    "Session 1 of User guest", "User Manager for UID 1000", "OpenBSD Secure Shell server",
    "System Logging Service", "Network Time Synchronization", "Regular background program processing daemon",
  ];
  const lines = [];
  for (const u of units) if (chance(rng, 0.8)) lines.push(`[  OK  ] Stopped ${u}.`);
  lines.push("[  OK  ] Stopped target Multi-User System.", "[  OK  ] Stopped target Basic System.",
    "[  OK  ] Reached target System Reboot.", "         Rebooting...");
  const k = pick(rng, KERNELS);
  lines.push("", `[    0.000000] Linux version ${k.uname.split(" ")[0]} (debian-kernel@lists.debian.org)`,
    `[    ${(rng.random() * 3 + 0.5).toFixed(6)}] systemd[1]: Detected virtualization qemu.`,
    `[  OK  ] Reached target Multi-User System.`,
    "", `Debian GNU/Linux 13 bity ttyAMA0`, "", `bity login: guest`, `Last login: ${pick(rng, ["Fri", "Sat", "Sun"])} Jul ${randint(rng, 1, 28)} ${clock(rng)} 2026`);
  return lines.join("\n");
}

export function* funGen(rng) {
  for (;;) {
    const r = rng.random();
    if (r < 0.26) {
      yield rec(chance(rng, 0.8) ? "fortune" : "fortune -s", pick(rng, FORTUNES));
    } else if (r < 0.5) {
      const v = rng.random();
      if (v < 0.55) {
        // includes echo-adjacent words (hello, test, hi) so cowsay conditioning
        // beats the lexical pull of echo contexts
        const msg = pick(rng, ["moo", "hello", "hello world", "hi", "test", "ok", "moo from bity",
          "i am a terminal", "have you fed the model today?", "42", "good morning", "beep"]);
        yield rec(`cowsay ${msg.includes(" ") ? `'${msg}'` : msg}`, cowsay(msg));
      } else {
        const f = pick(rng, FORTUNES);
        yield rec("fortune | cowsay", cowsay(f.length > 40 ? f.slice(0, 38) + "..." : f));
      }
    } else if (r < 0.62) {
      const v = rng.random();
      if (v < 0.4) yield rec(`sudo ${pick(rng, ["apt update", "reboot", "cat /etc/shadow", "systemctl restart ssh"])}`,
        "[sudo] password for guest: \nguest is not in the sudoers file.  This incident will be reported.");
      else if (v < 0.7) yield rec(`${pick(rng, ["cat /etc/shadow", "ls /root", "touch /etc/x"])}`,
        pick(rng, ["cat: /etc/shadow: Permission denied", "ls: cannot open directory '/root': Permission denied", "touch: cannot touch '/etc/x': Permission denied"]));
      else if (v < 0.85) {
        // unprivileged-guest persona: can't install, can't touch hardware sensors
        const [cmd, out] = pick(rng, [
          ["apt install cowsay", "E: Could not open lock file /var/lib/dpkg/lock-frontend - open (13: Permission denied)\nE: Unable to acquire the dpkg frontend lock (/var/lib/dpkg/lock-frontend), are you root?"],
          ["pip install requests", "error: externally-managed-environment\n\nThis environment is externally managed. To install Python packages\nsystem-wide, try apt install python3-xyz. See PEP 668 for details."],
          ["sensors", "No sensors found!\nMake sure you loaded all the kernel drivers you need.\nTry sensors-detect to find out which these are."],
        ]);
        yield rec(cmd, out);
      } else {
        // executable-bit family: ./script needs chmod +x first
        const s = pick(rng, ["./run.sh", "./deploy.sh", "./build.sh", "./start.sh"]);
        yield rec(s, `bash: ${s}: Permission denied`);
      }
    } else if (r < 0.82) {
      const v = rng.random();
      if (v < 0.75) { const t = pick(rng, TYPOS); yield rec(t, `bash: ${t}: command not found`); }
      else yield rec(`echo ${pick(rng, ["hello )", "'unterminated", "hi | "])}`,
        pick(rng, ["bash: syntax error near unexpected token `)'", "bash: unexpected EOF while looking for matching `''", "bash: syntax error near unexpected token `|'"]));
    } else if (r < 0.9) {
      yield rec("history", history(rng));
    } else if (r < 0.96) {
      yield rec(chance(rng, 0.7) ? "neofetch" : "neofetch --stdout", neofetch(rng));
    } else {
      yield rec("sudo reboot", reboot(rng));
    }
  }
}
