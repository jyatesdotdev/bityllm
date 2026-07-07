# Capture harness

Generates a training corpus of **real terminal output** by running commands inside a
throwaway Debian container and capturing what they print. This is the front end of the
M4 corpus work: real captures give exact, format-perfect output, which we later
templatize into the per-binary synthetic generators.

## Usage

```bash
npm run capture                 # ~4 MB into corpus/data/
node corpus/capture/run.mjs --mb 8 --batch 300
npm run capture:clean           # remove the container
```

Flags: `--mb <n>` target size · `--batch <n>` commands per docker exec ·
`--timeout <s>` per-command timeout · `--image <ref>` · `--fresh` rebuild the container.

## Output

- `corpus/data/debian.jsonl` — one record per command: `{ system, cat, cmd, exit, output }`.
- `corpus/data/debian.corpus.txt` — the same, formatted as a `guest@bity:~$ ` transcript,
  ready to char-tokenize and train on.

## How it reaches a few MB

A hand-written list is only tens of KB. The volume comes from **harvesting the container**:
`man` pages, `--help`/`--version` for every safe binary, `apt-cache show` + `dpkg -L` per
package, `/etc` configs, `/usr/share/doc`, and lots of `ls`/`cat`. Categories are
interleaved so a truncated run still has variety. See `commands.mjs`.

## Safety model (why containers)

A tiny char model memorizes, and this corpus ships to a browser demo, so we never want
real personal data or secrets in it. The harness:

- runs everything in a **disposable container** (hostname `bity`, non-root `guest`) — no
  access to your real machine, home dir, keys, or network identity;
- **excludes** interactive/destructive/secret-leaking commands (`env` dumps, editors,
  pagers, shells, `passwd`, etc. — see `HELP_BLACKLIST` and the curated lists);
- wraps every command in `timeout` with **stdin closed** and stderr merged;
- **sanitizes** output: strips ANSI/backspace formatting and control bytes, fakes MAC
  addresses, and drops binary/garbage output.

Deliberate permission-denied and "command not found" errors *are* kept — they're great
texture for a realistic terminal.
