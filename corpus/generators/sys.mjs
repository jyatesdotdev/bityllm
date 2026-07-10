// System-identity binaries: uname, whoami, id, hostname, date, uptime, free,
// df, nproc, arch, echo, ps. Formats match real Debian procps/coreutils.

import { pick, randint, chance, longDate, clock, KERNELS, rec } from "./lib.mjs";

const PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games";
const VARS = { USER: "guest", LOGNAME: "guest", SHELL: "/bin/bash", HOSTNAME: "bity", HOME: "/home/guest", PWD: "/home/guest", UID: "1000", LANG: "en_US.UTF-8", TERM: "xterm-256color", EDITOR: "vim", PATH };
const ENV = ["SHELL=/bin/bash", "PWD=/home/guest", "LOGNAME=guest", "HOME=/home/guest", "LANG=en_US.UTF-8", "TERM=xterm-256color", "USER=guest", "SHLVL=1", `PATH=${PATH}`, "EDITOR=vim", "HOSTNAME=bity", "_=/usr/bin/env"].join("\n");
const EXIT = [["true; echo $?", "0"], ["false; echo $?", "1"], ["[ 1 = 1 ]; echo $?", "0"], ["[ 1 = 2 ]; echo $?", "1"], ["ls > /dev/null; echo $?", "0"]];
const ALIASES = "alias l='ls -CF'\nalias la='ls -A'\nalias ll='ls -alF'";

function uptimeStr(rng) {
  const days = randint(rng, 0, 40);
  const users = randint(rng, 1, 3); // who/w always show you → never 0 users
  const load = () => (rng.random() * 2.5).toFixed(2);
  const up = days > 0 ? `${days} day${days > 1 ? "s" : ""}, ${randint(rng, 0, 24)}:${String(randint(rng, 0, 60)).padStart(2, "0")}` : `${randint(rng, 1, 59)} min`;
  return ` ${clock(rng)} up ${up},  ${users} user${users === 1 ? "" : "s"},  load average: ${load()}, ${load()}, ${load()}`;
}

function freeH(rng) {
  const tot = pick(rng, [4, 8, 16, 32]);
  const used = (rng.random() * tot * 0.4 + 0.4).toFixed(1);
  const cache = (rng.random() * tot * 0.3 + 0.5).toFixed(1);
  const free = (tot - used - cache).toFixed(1);
  const avail = (tot - used).toFixed(1);
  return [
    "               total        used        free      shared  buff/cache   available",
    `Mem:            ${String(tot + "Gi").padStart(4)}       ${used}Gi       ${free}Gi        ${randint(rng, 10, 99)}Mi       ${cache}Gi        ${avail}Gi`,
    "Swap:          977Mi          0B       977Mi",
  ].join("\n");
}

function dfH(rng) {
  const size = pick(rng, [40, 59, 120, 240]);
  const pct = randint(rng, 9, 81);
  const used = ((size * pct) / 100).toFixed(1);
  const avail = (size - used).toFixed(0);
  return [
    "Filesystem      Size  Used Avail Use% Mounted on",
    `/dev/vda1        ${size}G  ${used}G   ${avail}G  ${pct}% /`,
    "tmpfs           7.8G     0  7.8G   0% /dev/shm",
    `tmpfs           1.6G  ${pick(rng, ["8.9M", "1.2M", "38M"])}  1.6G   1% /run`,
    `/dev/vda15      124M   12M  112M  ${randint(rng, 8, 12)}% /boot/efi`,
  ].join("\n");
}

function psAux(rng) {
  const procs = [
    ["root", 1, "/sbin/init"],
    ["root", 2, "[kthreadd]"],
    ["root", randint(rng, 100, 400), "/lib/systemd/systemd-journald"],
    ["root", randint(rng, 400, 700), "/usr/sbin/sshd -D"],
    ["guest", randint(rng, 800, 1600), "-bash"],
    ["guest", randint(rng, 1700, 9000), "ps aux"],
  ];
  const rows = ["USER         PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND"];
  for (const [user, pid, cmd] of procs) {
    const cpu = (rng.random() * 1.5).toFixed(1);
    const mem = (rng.random() * 2).toFixed(1);
    rows.push(
      `${user.padEnd(8)} ${String(pid).padStart(7)}  ${cpu}  ${mem} ${String(randint(rng, 2000, 22000)).padStart(6)} ${String(randint(rng, 1000, 12000)).padStart(5)} ${cmd.startsWith("[") || user === "root" ? "?" : "pts/0"}        ${pick(rng, ["Ss", "S", "R+", "S<s"])}   ${String(randint(rng, 0, 23)).padStart(2, "0")}:${String(randint(rng, 0, 60)).padStart(2, "0")}   0:0${randint(rng, 0, 9)} ${cmd}`,
    );
  }
  return rows.join("\n");
}

export function* sysGen(rng) {
  for (;;) {
    const k = pick(rng, KERNELS);
    const r = rng.random();
    if (r < 0.14) {
      const v = rng.random();
      if (v < 0.4) yield rec("uname -a", `Linux bity ${k.uname} GNU/Linux`);
      else if (v < 0.6) yield rec("uname -r", k.uname.split(" ")[0]);
      else if (v < 0.8) yield rec("uname -m", k.arch);
      else yield rec("uname", "Linux");
    } else if (r < 0.26) {
      const v = rng.random();
      if (v < 0.35) yield rec("whoami", "guest");
      else if (v < 0.6) yield rec("id", "uid=1000(guest) gid=1000(guest) groups=1000(guest),100(users)");
      else if (v < 0.8) yield rec("hostname", "bity");
      else yield rec("groups", "guest users");
    } else if (r < 0.4) {
      const v = rng.random();
      if (v < 0.5) yield rec("date", longDate(rng));
      else if (v < 0.7) yield rec("date -u", longDate(rng));
      else yield rec("date +%s", String(randint(rng, 1750000000, 1800000000)));
    } else if (r < 0.52) {
      yield rec("uptime", uptimeStr(rng));
    } else if (r < 0.66) {
      yield rec(chance(rng, 0.8) ? "free -h" : "free -m", freeH(rng));
    } else if (r < 0.8) {
      yield rec(chance(rng, 0.8) ? "df -h" : "df -hT", dfH(rng));
    } else if (r < 0.88) {
      const v = rng.random();
      if (v < 0.4) yield rec("nproc", String(pick(rng, [2, 4, 8, 12])));
      else if (v < 0.7) yield rec("arch", k.arch);
      else yield rec("cat /etc/debian_version", "13.1");
    } else if (r < 0.95) {
      const v = rng.random();
      if (v < 0.12) {
        yield rec(pick(rng, ["exit", "logout"]), "logout");
      } else if (v < 0.34) {
        const msg = pick(rng, ["hello", "hello world", "test", "$HOME", "$((6 * 7))", "done", "it works", "$USER", "$SHELL"]);
        const out = msg === "$HOME" ? "/home/guest" : msg === "$((6 * 7))" ? "42" : msg === "$USER" ? "guest" : msg === "$SHELL" ? "/bin/bash" : msg;
        yield rec(`echo ${msg}`, out);
      } else if (v < 0.52) {
        // echo $VAR — every var consistent with env/printenv
        const name = pick(rng, Object.keys(VARS));
        yield rec(`echo $${name}`, VARS[name]);
      } else if (v < 0.68) {
        if (chance(rng, 0.45)) yield rec("env", ENV);
        else { const name = pick(rng, ["HOME", "USER", "SHELL", "PATH", "PWD", "LANG", "TERM", "EDITOR", "HOSTNAME"]); yield rec(`printenv ${name}`, VARS[name]); }
      } else if (v < 0.82) {
        const [cmd, out] = pick(rng, EXIT); // exit codes
        yield rec(cmd, out);
      } else if (v < 0.92) {
        if (chance(rng, 0.5)) yield rec("alias", ALIASES);
        else {
          const [cmd, out] = pick(rng, [["type cd", "cd is a shell builtin"], ["type ll", "ll is aliased to `ls -alF'"], ["type ls", "ls is /usr/bin/ls"], ["type python3", "python3 is /usr/bin/python3"], ["type git", "git is /usr/bin/git"], ["type nosuch", "bash: type: nosuch: not found"]]);
          yield rec(cmd, out);
        }
      } else {
        const c = pick(rng, ["ls", "git", "python3", "curl", "node", "cat", "grep", "cowsay"]);
        yield rec(`which ${c}`, `/usr/bin/${c}`);
      }
    } else {
      yield rec("ps aux", psAux(rng));
    }
  }
}

// ---- dreamed system/hardware info (post-hybrid) --------------------------------
// The programmatic CORE owns identity commands (whoami/uname/date/uptime/arch/env),
// but df/free/ps/top/lscpu/lsblk/mount/vmstat are still DREAMED. The capture is thin
// on them (~8-10 records each), so without this focused generator the graceful
// `command not found` drill overwhelms them and the model answers "not found" for
// real commands. This restores strong positive signal for exactly those commands.
function psShort(rng) {
  return ["    PID TTY          TIME CMD",
    `   ${randint(rng, 800, 1600)} pts/0    00:00:00 bash`,
    `   ${randint(rng, 1700, 9000)} pts/0    00:00:00 ps`].join("\n");
}
function topOut(rng) {
  const rows = [
    `top - ${uptimeStr(rng).trim()}`,
    `Tasks: ${randint(rng, 90, 140)} total,   1 running, ${randint(rng, 89, 139)} sleeping,   0 stopped,   0 zombie`,
    `%Cpu(s):  ${(rng.random() * 4).toFixed(1)} us,  ${(rng.random() * 2).toFixed(1)} sy,  0.0 ni, ${(93 + rng.random() * 6).toFixed(1)} id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st`,
    `MiB Mem :   1959.0 total,    ${randint(rng, 120, 400)}.0 free,    ${randint(rng, 400, 800)}.0 used,    ${randint(rng, 600, 1000)}.0 buff/cache`,
    "MiB Swap:    977.0 total,    977.0 free,      0.0 used.   " + randint(rng, 900, 1400) + ".0 avail Mem",
    "",
    "    PID USER      PR  NI    VIRT    RES    SHR S  %CPU  %MEM     TIME+ COMMAND",
  ];
  for (const [u, c] of [["root", "systemd"], ["root", "sshd"], ["guest", "bash"], ["guest", "top"]]) {
    rows.push(`${String(randint(rng, 1, 9000)).padStart(7)} ${u.padEnd(8)}  20   0 ${String(randint(rng, 10000, 200000)).padStart(7)} ${String(randint(rng, 5000, 20000)).padStart(6)} ${String(randint(rng, 3000, 10000)).padStart(6)} S   ${(rng.random() * 2).toFixed(1)}   ${(rng.random() * 2).toFixed(1)}   0:0${randint(rng, 0, 9)}.${randint(rng, 0, 9)} ${c}`);
  }
  return rows.join("\n");
}
function lscpuOut(rng) {
  const n = pick(rng, [2, 4, 8]);
  return [
    "Architecture:            aarch64", "  CPU op-mode(s):        64-bit", "  Byte Order:            Little Endian",
    `CPU(s):                  ${n}`, `  On-line CPU(s) list:   0-${n - 1}`,
    "Vendor ID:               ARM", "Caches (sum of all):", "  L1d:                   128 KiB", "  L2:                    4 MiB",
  ].join("\n");
}
function lsblkOut(rng) {
  const g = pick(rng, [40, 60, 120, 240]);
  return [
    "NAME    MAJ:MIN RM  SIZE RO TYPE MOUNTPOINTS",
    `vda     254:0    0  ${g}G  0 disk `,
    `|-vda1  254:1    0 ${g - 1}G  0 part /`,
    "|-vda14 254:14   0    4M  0 part ",
    "`-vda15 254:15   0  124M  0 part /boot/efi",
  ].join("\n");
}
function mountOut(rng) {
  return [
    "/dev/vda1 on / type ext4 (rw,relatime)",
    "proc on /proc type proc (rw,nosuid,nodev,noexec,relatime)",
    "sysfs on /sys type sysfs (rw,nosuid,nodev,noexec,relatime)",
    `tmpfs on /run type tmpfs (rw,nosuid,nodev,size=${randint(rng, 300, 800)}M)`,
    "tmpfs on /dev/shm type tmpfs (rw,nosuid,nodev)",
  ].join("\n");
}
function vmstatOut(rng) {
  return [
    "procs -----------memory---------- ---swap-- -----io---- -system-- ------cpu-----",
    " r  b   swpd   free   buff  cache   si   so    bi    bo   in   cs us sy id wa st",
    ` ${randint(rng, 0, 3)}  0      0 ${String(randint(rng, 120000, 400000)).padStart(6)} ${String(randint(rng, 50000, 90000)).padStart(6)} ${String(randint(rng, 700000, 900000)).padStart(6)}    0    0    ${randint(rng, 1, 40)}    ${randint(rng, 1, 60)}  ${randint(rng, 30, 90)}  ${randint(rng, 40, 99)}  ${randint(rng, 1, 5)} ${randint(rng, 2, 8)} ${randint(rng, 88, 97)}  ${randint(rng, 0, 2)}  0`,
  ].join("\n");
}

export function* sysinfoGen(rng) {
  for (;;) {
    const r = rng.random();
    if (r < 0.22) yield rec(chance(rng, 0.8) ? "df -h" : "df -hT", dfH(rng));
    else if (r < 0.44) yield rec(chance(rng, 0.8) ? "free -h" : "free -m", freeH(rng));
    else if (r < 0.60) yield rec(pick(rng, ["ps aux", "ps aux", "ps -ef"]), psAux(rng));
    else if (r < 0.68) yield rec("ps", psShort(rng));
    else if (r < 0.74) yield rec("nproc", String(pick(rng, [2, 4, 8])));
    else if (r < 0.82) yield rec("top -bn1", topOut(rng));
    else if (r < 0.89) yield rec("lscpu", lscpuOut(rng));
    else if (r < 0.94) yield rec("lsblk", lsblkOut(rng));
    else if (r < 0.98) yield rec("mount", mountOut(rng));
    else yield rec("vmstat", vmstatOut(rng));
  }
}
