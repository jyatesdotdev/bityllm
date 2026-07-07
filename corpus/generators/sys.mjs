// System-identity binaries: uname, whoami, id, hostname, date, uptime, free,
// df, nproc, arch, echo, ps. Formats match real Debian procps/coreutils.

import { pick, randint, chance, longDate, clock, KERNELS, rec } from "./lib.mjs";

function uptimeStr(rng) {
  const days = randint(rng, 0, 40);
  const users = randint(rng, 0, 3);
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
      if (chance(rng, 0.15)) {
        yield rec(pick(rng, ["exit", "logout"]), "logout");
      } else {
        const msg = pick(rng, ["hello", "hello world", "test", "$HOME", "$((6 * 7))", "done", "it works"]);
        const out = msg === "$HOME" ? "/home/guest" : msg === "$((6 * 7))" ? "42" : msg;
        yield rec(`echo ${msg}`, out);
      }
    } else {
      yield rec("ps aux", psAux(rng));
    }
  }
}
