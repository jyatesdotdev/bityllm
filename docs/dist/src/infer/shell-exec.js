// shell-exec.ts — a tiny real shell over the VFS + coreutils.
//
// Parses a command line (quotes, $VAR/$?, globs, pipes, redirects, ; && ||),
// runs CORE binaries deterministically threading stdin between pipe stages, and
// applies > / >> to the VFS. If any command in a pipeline isn't a CORE binary,
// the whole line is handed back for the MODEL to dream (result.dreamed).
import { CORE } from "./coreutils.js";
// split on top-level separators (outside quotes)
function splitTop(s, seps) {
    const out = [];
    let buf = "", q = null;
    for (let i = 0; i < s.length; i++) {
        const c = s[i];
        if (q) {
            buf += c;
            if (c === q)
                q = null;
            continue;
        }
        if (c === "'" || c === '"') {
            q = c;
            buf += c;
            continue;
        }
        const two = s.slice(i, i + 2);
        const sep = seps.find((x) => (x.length === 2 ? two === x : c === x && !seps.includes(two)));
        if (sep) {
            out.push({ part: buf, sep });
            buf = "";
            i += sep.length - 1;
            continue;
        }
        buf += c;
    }
    out.push({ part: buf, sep: "" });
    return out;
}
// tokenize one command stage, honoring quotes (quotes are stripped)
function tokenize(s) {
    const toks = [];
    let buf = "", q = null, has = false;
    for (let i = 0; i < s.length; i++) {
        const c = s[i];
        if (q) {
            if (c === q)
                q = null;
            else
                buf += c;
            has = true;
            continue;
        }
        if (c === "'" || c === '"') {
            q = c;
            has = true;
            continue;
        }
        if (c === " " || c === "\t") {
            if (has) {
                toks.push(buf);
                buf = "";
                has = false;
            }
            continue;
        }
        buf += c;
        has = true;
    }
    if (has)
        toks.push(buf);
    return toks;
}
// $VAR / ${VAR} / $? / $(...) expansion + ~ + globs, per token
function expand(tok, st, quotedDouble) {
    let t = tok;
    t = t.replace(/\$\?/g, String(st.lastExit));
    t = t.replace(/\$\{(\w+)\}|\$(\w+)/g, (_m, a, b) => st.env.get(a ?? b) ?? "");
    if (t === "~" || t.startsWith("~/"))
        t = (st.user === "root" ? "/root" : "/home/guest") + t.slice(1);
    // glob *.ext against the cwd (only when unquoted and contains *)
    if (!quotedDouble && t.includes("*")) {
        const names = st.vfs.list(".", st.cwd, false) ?? [];
        const rx = new RegExp("^" + t.replace(/[.+^${}()|[\]\\]/g, "\\$&").replace(/\*/g, ".*") + "$");
        const hits = names.filter((n) => rx.test(n)).sort();
        if (hits.length)
            return hits;
    }
    return [t];
}
// run one pipeline (stages joined by |); returns null if it must be dreamed
function runPipeline(pipeline, st) {
    const stages = splitTop(pipeline, ["|"]).map((x) => x.part.trim()).filter(Boolean);
    // a leading VAR=value assignment (whole "stage" is one assignment)
    if (stages.length === 1) {
        const m = stages[0].match(/^(\w+)=(.*)$/);
        if (m && !/\s/.test(m[1])) {
            st.env.set(m[1], stripQuotes(m[2]));
            return { out: "", exit: 0 };
        }
    }
    // parse each stage into { cmd, args, redir }
    const parsed = stages.map((stage) => {
        const raw = tokenize(stage);
        const words = [];
        let redir = null;
        for (let i = 0; i < raw.length; i++) {
            if (raw[i] === ">" || raw[i] === ">>") {
                redir = { op: raw[i], file: raw[i + 1] };
                i++;
                continue;
            }
            const m = raw[i].match(/^(>>?)(.+)$/);
            if (m) {
                redir = { op: m[1], file: m[2] };
                continue;
            }
            words.push(raw[i]);
        }
        const expanded = words.flatMap((w) => expand(w, st, false));
        return { cmd: expanded[0], args: expanded.slice(1), redir };
    });
    // if any command isn't a CORE binary, the model dreams the whole line
    if (parsed.some((p) => !(p.cmd in CORE)))
        return null;
    let stdin = "";
    let last = { out: "", exit: 0 };
    for (const p of parsed) {
        last = CORE[p.cmd](p.args, st, stdin);
        if (p.redir) {
            st.vfs.writeFile(p.redir.file, st.cwd, last.out, st.user, p.redir.op === ">>");
            last = { out: "", exit: last.exit };
        }
        stdin = last.out;
    }
    st.lastExit = last.exit;
    return last;
}
const stripQuotes = (s) => s.replace(/^["']|["']$/g, "");
const isAssign = (seg) => /^\w+=/.test(seg) && !/^\w+=\S*\s/.test(seg.replace(/^\w+=/, "x="));
export function execLine(line, st) {
    const trimmed = line.trim();
    if (!trimmed)
        return { out: "", exit: st.lastExit };
    const segments = splitTop(trimmed, ["&&", "||", ";"]);
    // pre-scan (no side effects): if any command isn't a CORE binary, the model
    // dreams the whole line — so we never half-mutate the VFS on a dreamed line.
    for (const { part } of segments) {
        const seg = part.trim();
        if (!seg || isAssign(seg))
            continue;
        for (const stage of splitTop(seg, ["|"])) {
            const first = tokenize(stage.part.trim())[0];
            const cmd = first ? expand(first, st, false)[0] : "";
            if (cmd && !(cmd in CORE))
                return { out: "", exit: st.lastExit, dreamed: trimmed };
        }
    }
    // all-core: execute in order, honoring && (run next iff prev ok) and || (iff prev failed)
    let out = "";
    let exit = 0;
    for (let i = 0; i < segments.length; i++) {
        const seg = segments[i].part.trim();
        if (!seg)
            continue;
        const prevSep = i > 0 ? segments[i - 1].sep : "";
        if ((prevSep === "&&" && exit !== 0) || (prevSep === "||" && exit === 0))
            continue;
        const res = runPipeline(seg, st);
        if (res === null)
            return { out, exit, dreamed: trimmed };
        out += res.out;
        exit = res.exit;
    }
    return { out, exit };
}
