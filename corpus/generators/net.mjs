// Network binaries: ping (the headline act), traceroute, curl, ip, ss.
// iputils/iproute2 formats from real Debian captures.

import { pick, randint, chance, ip, HOSTS, rec } from "./lib.mjs";

function pingRun(rng, host, count) {
  const addr = host === "localhost" ? "127.0.0.1" : ip(rng);
  const base = host === "localhost" ? 0.05 : rng.random() * 40 + 4;
  const lines = [`PING ${host} (${addr}) 56(84) bytes of data.`];
  const times = [];
  let received = 0;
  for (let i = 1; i <= count; i++) {
    if (host !== "localhost" && chance(rng, 0.04)) continue; // dropped packet
    const t = Math.max(0.03, base + (rng.random() - 0.5) * base * 0.2);
    times.push(t);
    received++;
    lines.push(`64 bytes from ${addr}: icmp_seq=${i} ttl=${pick(rng, [52, 54, 57, 63, 64, 118])} time=${t.toFixed(t < 1 ? 3 : 1)} ms`);
  }
  const loss = Math.round(((count - received) / count) * 100);
  lines.push("", `--- ${host} ping statistics ---`);
  lines.push(`${count} packets transmitted, ${received} received, ${loss}% packet loss, time ${count * 1000 + randint(rng, 1, 80)}ms`);
  if (received > 0) {
    const min = Math.min(...times), max = Math.max(...times);
    const avg = times.reduce((a, b) => a + b, 0) / times.length;
    const mdev = (max - min) / 4;
    const f = (x) => x.toFixed(3);
    lines.push(`rtt min/avg/max/mdev = ${f(min)}/${f(avg)}/${f(max)}/${f(mdev)} ms`);
  }
  return lines.join("\n");
}

function traceRun(rng, host) {
  const addr = ip(rng);
  const lines = [`traceroute to ${host} (${addr}), 30 hops max, 60 byte packets`];
  const hops = randint(rng, 4, 11);
  for (let h = 1; h <= hops; h++) {
    if (chance(rng, 0.18)) {
      lines.push(` ${h}  * * *`);
      continue;
    }
    const hopIp = h === 1 ? "10.0.2.2" : ip(rng);
    const name = h === 1 ? "gateway" : h === hops ? host : `${pick(rng, ["ae", "et", "core", "edge", "po"])}-${randint(rng, 0, 9)}.${pick(rng, ["fra", "ams", "lhr", "cdg", "nyc"])}${randint(rng, 1, 4)}.net`;
    const t = () => `${(rng.random() * 8 * h + 0.3).toFixed(3)} ms`;
    lines.push(` ${h}  ${name} (${hopIp})  ${t()}  ${t()}  ${t()}`);
  }
  return lines.join("\n");
}

function curlRun(rng, host) {
  const codes = [["HTTP/2 200", 0.75], ["HTTP/2 301", 0.15], ["HTTP/2 404", 0.1]];
  const r = rng.random();
  let code = "HTTP/2 200";
  let acc = 0;
  for (const [c, p] of codes) { acc += p; if (r < acc) { code = c; break; } }
  const lines = [code];
  lines.push(`server: ${pick(rng, ["nginx", "nginx/1.26.0", "cloudflare", "Apache", "GitHub.com"])}`);
  lines.push(`content-type: text/html; charset=utf-8`);
  if (code.endsWith("301")) lines.push(`location: https://www.${host}/`);
  lines.push(`content-length: ${randint(rng, 612, 90000)}`);
  return lines.join("\n");
}

export function* netGen(rng) {
  for (;;) {
    const host = pick(rng, HOSTS);
    const r = rng.random();
    if (r < 0.55) {
      const v = rng.random();
      if (v < 0.55) yield rec(`ping -c ${randint(rng, 2, 6)} ${host}`, pingRun(rng, host, randint(rng, 2, 6)));
      else if (v < 0.8) yield rec(`ping ${host}`, pingRun(rng, host, randint(rng, 3, 7)) + "\n^C");
      else yield rec(`ping ${pick(rng, ["bity.local", "nowhere.invalid", "srv9"])}`, `ping: ${pick(rng, ["bity.local", "nowhere.invalid", "srv9"])}: Name or service not known`);
    } else if (r < 0.7) {
      yield rec(`traceroute ${host}`, traceRun(rng, host));
    } else if (r < 0.85) {
      yield rec(`curl -I https://${host}`, curlRun(rng, host));
    } else {
      const v = rng.random();
      if (v < 0.5)
        yield rec("ip -br addr", `lo               UNKNOWN        127.0.0.1/8 ::1/128\neth0             UP             10.0.2.15/24 metric 100 fe80::5054:ff:fe12:3456/64`);
      else if (v < 0.75)
        yield rec("hostname -I", "10.0.2.15");
      else
        yield rec("ss -tuln", `Netid State  Recv-Q Send-Q Local Address:Port  Peer Address:Port\ntcp   LISTEN 0      128          0.0.0.0:22         0.0.0.0:*\ntcp   LISTEN 0      511          0.0.0.0:${pick(rng, [80, 443, 8080, 3000])}       0.0.0.0:*`);
    }
  }
}
