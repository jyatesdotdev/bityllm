I've read all generators and the corpus context. Here is the deduplicated, prioritized implementation spec.

---

# bityllm corpus coverage spec — content-inspection, mutation, navigation, system, net, dev

Verified against `corpus/generators/{fs,sys,net,git,fun,copy,lib,index}.mjs`. Anything already well-covered (ls/cat/ping/traceroute/`curl -I`/git status-log-branch-diff-stat/fortune/cowsay/sudo/neofetch, `echo $HOME`, `wc -l notes.txt`, bare `hostname`, `printenv PATH`) is dropped.

**Foundational refactor (blocks most HIGH content items):** in `fsSessionBlock` the per-entry `meta` currently stores only `{dir, content}`, and `ls -la`/`lsLong` re-roll `size` with `pick(rng,[...])` on every call. Before adding wc/stat/du/file, store a fixed byte size on each created entry and have every sizer read it:
- `bytes = content === "" ? 0 : content.length + 1` (single trailing newline from `echo`)
- `lines = content === "" ? 0 : content.split("\n").length`
- `words = content.trim() === "" ? 0 : content.trim().split(/\s+/).length`
- Pin `ls -l/-la` size for created files to `bytes` so listings never contradict `wc -c`/`stat`/`du -b`.

---

## HIGH

### fs.mjs
- **Multi-word content round-trip (known gap a):** densify the existing `echo <w1> <w2> <w3> > f` path so it is FOLLOWED in the same block by `cat f` -> `w1 w2 w3` (all words, single-spaced). Add same-block `wc -w f` -> `3 f` and `wc -c f` -> `<bytes> f`. (every word preserved; `cat`/`wc -w`/`wc -c` all read the one stored content.)
- **touch -> empty file (known gap c):** after `touch f`, emit `cat f` -> `` (empty), and optionally `wc -l f` -> `0 f`, `file f` -> `f: empty` in the SAME block. Never `cat: f: No such file or directory` once touched. (known-but-empty beats ENOENT.)
- **Nested cd / deep pwd (known gap b):** replace the one-level `cwd` machine with a real path stack. Register every intermediate in `locs`. `cd projects; cd src; pwd` -> `/home/guest/projects/src`; prompt `guest@bity:~/projects/src$`. `cd ..` strips exactly one segment; `cd projects/src` descends two at once. (prompt path === pwd at every depth; `cd ..` from depth-2 lands depth-1, never resets to ~.)
- **`cat ../f` and `ls ..`:** from `~/projects`, `cat ../notes.txt` -> home file content; `ls ..` -> home listing. (parent file MUST NOT ENOENT; equals what `cat f`/`ls` give after `cd ..`.)
- **head/tail of a session multi-line file:** after echo+`>>` appends, `head -n 2 list.txt` / `tail -n 1 list.txt` slice the SAME accumulated content; bare `head`/`tail` (<=10 lines) == `cat`. (line order = append order; `head`/`tail`/`cat` identical for short files.)
- **wc bare + single-flag:** `wc f` -> ` <lines> <words> <bytes> f`; `wc -w f` -> `<n> f`; `wc -c f` -> `<n> f`. (all three agree with content and with `wc -l`.)
- **chmod -> ls -l perms (state):** `chmod +x deploy.sh` / `chmod 600 notes.txt` silent; every later `ls -l`/`ls -la` shows the new perm triad. Store mode in meta. Map: 644→`-rw-r--r--`, 600→`-rw-------`, 755/+x→`-rwxr-xr-x`, 700→`-rwx------`, 444→`-r--r--r--`. (perm field reflects last chmod for rest of session.)
- **mkdir -p (state):** `mkdir -p a/b/c` silent, registers a, a/b, a/b/c so later `cd a/b/c`/`ls a/b` succeed; top-level `ls` shows only `a`. `mkdir -p existing` silent (vs plain `mkdir existing` -> `File exists`). (each segment enterable; `cat a/b/c` -> `Is a directory`.)
- **`rm -rf <name>` silent (FIX):** currently `rm -rf` is 100% bound to the `rm -rf /` failsafe. Gate the danger message strictly to literal `/`; `rm -rf build` (name != `/`) silently deletes the subtree — gone from later `ls`/`cat`. `rm -rf nonexistent` -> silent (‑f suppresses). 
- **`cd <file>` -> Not a directory (FIX):** fsGen (L247-248) picks error-path and errno independently of the arg, producing wrong `cd secret: Permission denied`. Fix: error path must name the typed arg; absent name -> `No such file or directory`; known non-dir file -> `bash: cd: notes.txt: Not a directory`; reserve `Permission denied` for `/root`.
- **Pipes into counters/grep:** `cat f | wc -l` -> bare number (NO filename); `ls | grep .txt` -> matching entries one-per-line; `cat f | grep word` -> only matching LINES; `ls | wc -l` -> entry count. (piped `wc` omits filename column; count equals standalone `wc -l f`.)
- **Redirect any command then read back:** `ls > files.txt; cat files.txt` -> listing one-per-line (incl. files.txt on next listing); `date > now.txt; cat now.txt`; `>>` appends. (file content == command stdout; new file appears in later `ls`.)
- **`;` / `&&` / `||` chaining with state:** `mkdir demo; cd demo; pwd` -> `/home/guest/demo` and next prompt reflects demo. `cd projects && ls` runs right side only on success; `cat missing || echo nope` -> error line then `nope`. (state from left half persists into right half; `&&` skips right on failure.)
- **Var assignment + command substitution:** `name=bity; echo "hello $name"` -> `hello bity` (bare assign prints nothing); `echo "user: $(whoami) at $(pwd)"` -> `user: guest at /home/guest`. (`$(cmd)` === that cmd's own output; unset var -> empty string.)
- **Globbing `*`:** `ls *.txt` -> sorted matches from live cwd; `echo *` -> all non-hidden entries space-joined; no match -> `ls: cannot access '*.xyz': No such file or directory`. (reflects created/removed files.)

### sys.mjs
- **`echo $?` (state):** two-command blocks. `false; echo $?` -> `1`; `true`/`ls`/`cat f`(ok) -> `0`; `cat nope`/`ls missing` -> `2`; unknown command -> `127`. (reflects ONLY the previous command's exit.)
- **`env` full dump + `printenv <VAR>`:** stable block with `USER=guest HOME=/home/guest HOSTNAME=bity SHELL=/bin/bash EDITOR=vim PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games …`; `printenv HOME`->`/home/guest`, `printenv USER`->`guest`, missing var -> empty + `$?=1`. (every line matches its `echo $VAR`; PATH byte-identical across env/printenv/`echo $PATH`; EDITOR matches .bashrc.)
- **`echo $VAR` beyond HOME:** `echo $USER`->`guest`, `$SHELL`->`/bin/bash`, `$HOSTNAME`->`bity`, `$UID`->`1000`, `$PWD`-> current prompt path, `$PATH`-> env PATH; unset -> blank line.
- **`export VAR=val` (state):** silent; later `echo $VAR`/`printenv VAR` return it for the session; reassignment overrides.
- **`alias` (bare):** list exactly the .bashrc aliases alpha-sorted: `alias l='ls -CF'` / `alias la='ls -A'` / `alias ll='ls -alF'`. (must match .bashrc corpus.)
- **FIX uptime users:** `uptimeStr` allows `0 users`, contradicting who/w always showing you. Clamp `users` to `>=1`.

### net.mjs
- **`curl <url>` body (no -I):** plain `curl https://example.com` -> the Example Domain HTML; `curl -s https://api.bity.dev/status` -> `{"status":"ok","version":"1.4.2","uptime":90271,"host":"bity"}`. (body agrees with `curl -I` content-type; JSON `host` == `hostname`=bity.)
- **`ip a` (verbose sibling of `ip -br addr`):** full lo + eth0 block with `10.0.2.15/24`, MAC `52:54:00:12:34:56`, link-local `fe80::5054:ff:fe12:3456`. (same identity as `ip -br addr`/`hostname -I`; avoid the real-capture `de:ad:be:ef`/172.17 drift.)
- **`ip route` / `ip r`:** `default via 10.0.2.2 dev eth0 proto dhcp src 10.0.2.15 metric 100` + LAN route. (gateway 10.0.2.2 == traceroute hop 1; src 10.0.2.15 == ip a.)
- **`dig <host>` + `dig +short`:** full answer section / bare IP. (A record == ping/traceroute/host IP for that host in-session; SERVER 127.0.0.53#53.)

### git.mjs
- **Version banners:** `git --version`->`git version 2.47.2`; `node -v`->`v20.19.4`; `npm --version`->`9.2.0`; `python3 --version`->`Python 3.13.5`; `pip3 --version`->`pip 25.0.1 … (python 3.13)`; `gcc --version` 4-line Debian 14.2.0 banner; `make --version`. (stable within session; consistent toolchain versions.)
- **`git add X` -> status change (state):** empty output; a following `git status` shows X under "Changes to be committed:" instead of not-staged. (git analogue of mkdir->ls.)
- **`git commit -m "msg"`:** `[main 3f9a2c1] msg\n 1 file changed, N insertions(+), M deletions(-)`; after it `git status` -> clean/ahead-by-1, `git log` top == that hash+msg.
- **`git status -s`:** ` M src/index.ts` / `?? notes.txt` porcelain; must agree with long-form for same session state.

### fun.mjs
- **`figlet <word>`:** render the ASCII banner of the argument (figlet is dpkg-confirmed installed — must NOT be command-not-found). (banner corresponds to input word.)
- **`./run.sh` -> Permission denied (state):** `bash: ./run.sh: Permission denied`; after `chmod +x run.sh` it runs instead. (echoes the ./arg; complements existing permission-denied family.)

---

## MEDIUM

### fs.mjs
- **rmdir:** empty dir -> silent + gone; non-empty (projects) -> `rmdir: failed to remove 'projects': Directory not empty` (still listed); on a file -> `Not a directory`.
- **cp -r / mv+cp into dir:** `cp -r A B` -> both exist, `ls B`==`ls A`; plain `cp` on dir -> `cp: -r not specified; omitting directory 'A'`. `mv f dir/` -> f leaves cwd, appears in `ls dir`, `cat dir/f`==content; `cp f dir/` -> f in both. (copies preserve content/size.)
- **ln -s:** `ls -l link` -> `lrwxrwxrwx … link -> target`; link in plain ls; `cat link`==`cat target`; dangling link lists but `cat` -> No such file.
- **tree / tree -L 1 / tree <dir>:** ASCII tree of current world + `N directories, M files`. (reflects created/removed; count matches. If not modeling install, use `bash: tree: command not found` consistently.)
- **ls -lh / -lt / -lS / -R:** human sizes (4.0K/220/27) with `total`; `-lt` newest-first (just-touched leads); `-lS` largest-first; `-R` `./path:` section headers. (rows identical to ls -la; only sort/format differs; -R sections match `ls <dir>`.)
- **cat -n / nl / less / more:** `cat -n`/`nl` -> `     N\t<line>` (nl skips blanks, cat -n numbers all); `less f`/`more f` == `cat f` (whole short file, same ENOENT/empty behavior).
- **file / stat / du-of-file:** `file note.txt`->`ASCII text`, touched->`empty`, shebang script->`POSIX shell script, ASCII text executable`; `stat f` Size == wc -c, empty->`Size: 0 … regular empty file`, owner guest/guest 0644; `du -h f`->`4.0K\tf` (0 for empty), `du -b f`->`<bytes>\tf`==wc -c.
- **Multi-arg / existing-file touch:** `touch a b c` creates all; `touch keep.txt` on a non-empty file does NOT truncate (`cat` still shows content).
- **cat multiple files:** `cat a b` -> a's content then b's, in arg order; a missing arg errors inline but others still print.
- **cd variants:** bare `cd` / `cd ~` -> home from any depth; `cd ~/projects` from anywhere; `cd -` prints and toggles OLDPWD; absolute `cd /tmp` -> prompt `guest@bity:/tmp$` (no ~), `cd /home/guest/projects` collapses to `~/projects`.
- **realpath / dirname / basename:** `realpath ../f` -> parent-abs/f; `dirname`/`basename` pure string splits, `basename main.py .py`->`main`.
- **pushd / popd / dirs:** stack ops; leftmost == cwd; over-pop -> `bash: popd: directory stack empty`.
- **Text pipes:** `cat f | sort | uniq -c`; `echo hello world | tr a-z A-Z`->`HELLO WORLD`, `rev`, `cut -d' ' -f1`. (deterministic transform of piped input.)
- **Quote expansion + stderr redirect:** `echo '$HOME'`->`$HOME` vs `echo "$HOME"`->`/home/guest`; `cat missing 2>/dev/null; echo done`->`done` (but `$?`=1); `ls nope 2>&1 | grep cannot` -> the error via the pipe.
- **Heredoc:** `cat > f << EOF … EOF; cat f` -> body verbatim.

### sys.mjs
- **type:** `type cd`->shell builtin; `type ll`->aliased to `ls -alF'` (matches alias); `type python3`->/usr/bin/python3 (matches which); `type nosuch`->`bash: type: nosuch: not found`.
- **lscpu / lsblk / top -bn1 / w / who / users / vmstat / free (bare KiB):** all cross-consistent — CPU(s)==nproc, arch==uname -m; lsblk vda size==df; top/w header==uptime (users>=1, same load triple); who/w row guest pts/0 from 10.0.2.2; free KiB scales to `free -h`.
- **cal / cal 2026:** month grid whose month+year match `date`.
- **kill:** `kill 99999`->`bash: kill: (99999) - No such process`; `kill 1`->`Operation not permitted`; `kill %1`->`no such job`.
- **lsb_release -d/-r/-c/-i:** match os-release (Debian 13 trixie).
- **id -u/-un/-g/-gn/-G:** decompose full `id` (1000/guest/1000/guest/`1000 100`).
- **date +FORMAT:** `+%Y-%m-%d`/`+%F`/`+%T`/`+%A` internally consistent with bare `date`.
- **hostnamectl / timedatectl:** hostname bity, OS/kernel/arch match uname/os-release, Chassis vm (justifies sensors nothing).
- **uptime -p / -s:** same elapsed as bare uptime.
- **sensors:** `No sensors found! …` + `$?=1` (VM persona).

### net.mjs
- **curl echo-IP + wget (stateful):** `curl -s ifconfig.me`->stable public IP (distinct from 10.0.2.15); `wget <url>` progress block that CREATES the saved file (later `ls`/`cat` show it). (Resolving IP == dig/ping.)
- **nslookup / host / DNS-failure:** `nslookup`/`host` addresses == dig A record; NXDOMAIN forms consistent with `ping … Name or service not known`.
- **ifconfig:** eth0/lo block, inet 10.0.2.15, EUI-64-consistent MAC/link-local.
- **ss -tulpn / -s, netstat -tlnp / -rn:** ports {22,80,443,8080,3000} match `ss -tuln`; gateway 10.0.2.2 matches ip route.
- **arp -a / ip neigh:** `_gateway (10.0.2.2) at <mac> [ether] on eth0` (gateway MAC differs from host).
- **ssh / scp / nc / whois:** ssh host-key prompt / Connection refused / Could not resolve (IP matches dig); scp progress (source must exist); `nc -zv host 443 … succeeded!`; whois echoes queried domain.
- **chown:** as guest -> `Operation not permitted`, owner column unchanged; `chown guest:guest f` silent.

### git.mjs (dev tooling)
- **git clone (stateful):** progress block; after it `ls` shows repo dir, `cd <repo>` succeeds. (dir name == URL basename.)
- **git remote -v:** `origin git@github.com:guest/bityllm.git (fetch/push)` (matches existing push URL).
- **Full git diff:** `diff --git … @@ hunks …` +/- lines; files drawn from the same modified set `git status` reports.
- **git checkout -b / switch -c (state):** `Switched to a new branch 'X'`; later `git branch` stars X (currently always stars main — must not contradict).
- **git init / git-outside-repo fatal:** `Initialized empty Git repository in <cwd>/.git/`; `fatal: not a git repository …` gated ONLY to non-repo context (fresh mkdir'd dir / after cd /tmp), never after add/commit/clone.
- **python3 -c / node -e:** `python3 -c 'print(2**10)'`->`1024`, `print("hello")`->`hello` (copy circuit).
- **python3 <missing>.py:** `python3: can't open file '/home/guest/build.py': [Errno 2] No such file or directory`; suppress if session created build.py.
- **pip install -> PEP 668:** `error: externally-managed-environment …` (default for bare pip install; unprivileged persona).
- **apt update / install (no sudo):** `E: Could not open lock file … (13: Permission denied)`. `apt show curl` -> package stanza (versions consistent with dpkg/neofetch).
- **Clean `ls --help` / `grep --help`:** well-formed Usage block (flags match what corpus honors).

### fun.mjs
- **yes / cmatrix-toilet-not-found / bash syntax error:** `yes bity` -> repeated `bity`…`^C`; `cmatrix`/`toilet` -> `command not found` (NOT installed; keep sl/cmatrix/toilet as not-found to match existing TYPOS); `echo hello )` -> ``bash: syntax error near unexpected token `)'``.

---

## LOW

### fs.mjs
- **tac:** reverse line order of created multi-line file.
- **hexdump -C / xxd / od -c:** byte view of a very short fixed string (e.g. `bity`); offset trailer == wc -c. Keep char-for-char, non-generalizable.
- **strings:** for text file == its printable lines (== cat here).

### sys.mjs
- **`echo $$` / `$0` / pidof / pgrep / `ps -ef`:** PID matches guest -bash/pts0 row in ps aux; bare `set` -> skip (high cost, low value).

### fun.mjs
- **lolcat pipe passthrough:** `fortune | lolcat` == plain `fortune` (ANSI stripped); reuse FORTUNES list.
- **man ls pager:** NAME/SYNOPSIS opening ending at the pager status line `Manual page ls(1) line 1 (press h for help or q to quit)`.

---

## CONSISTENCY RISKS

- **`rm -rf <name>` vs `rm -rf /`:** existing corpus binds ALL `rm -rf` to the failsafe warning. New silent-delete MUST match `^rm -rf /$` for the danger message only; any other arg is silent. Wrong gating reintroduces the failsafe on ordinary deletes.
- **touch->empty vs cat-unknown->ENOENT (the classic bug):** the empty-file `cat` must be emitted in the SAME stateful block right after `touch`; never let fsGen/copy.mjs's ENOENT `cat` fire on a name the session just touched.
- **`cat ../f` / `ls ..` / `cd a/b/c`:** parent and nested paths that genuinely exist must NOT return ENOENT/Permission denied. Requires the `locs` map to register every intermediate directory; the fsGen independent-errno bug (`cd secret -> Permission denied`) must be fixed in the same pass or it contradicts the new "error names the typed arg" rule.
- **git status fatal-not-a-repo:** contradicts the always-succeeds `git status`. Gate strictly to clearly-non-repo sessions (fresh dir / `cd /tmp`); never after `git add`/`commit`/`clone`/`init`.
- **git checkout -b then git branch:** existing `git branch` unconditionally stars `main`. After a `checkout -b X` in a session, `git branch` must star X, not main.
- **wc -c / stat Size / du -b / file empty must all agree on ONE byte count.** Today `ls -la` re-rolls `size` per call; leaving it random will contradict the new exact-byte sizers. Store a single `bytes` in meta and read it everywhere (see Foundational refactor).
- **cp preserves size/content:** `cp a b` currently gives b an independently-rolled ls size; once wc -c/stat exist, b must report the SAME bytes as a.
- **uptime users >= 1:** current `uptimeStr` can print `0 users`; who/w/top all show >=1. Clamp before adding who/w/top so headers don't contradict.
- **cmatrix/toilet/sl stay command-not-found:** figlet/cowsay/fortune are dpkg-confirmed installed (render output); the others are NOT — do not add ASCII/animation output for them or it contradicts existing `sl -> command not found` (181x via TYPOS).
- **`cd <file> -> Not a directory` only for KNOWN files:** must fire on a name the session created as a file, and must never contradict `cd <newdir>` succeeding after mkdir. Reserve `Permission denied` for genuinely protected paths (`/root`), not merely-absent names.
- **sensors / pip PEP668 / apt lock / chown denied / sudo:** all lean on the single unprivileged-guest persona — keep them mutually reinforcing (guest can't install, can't chown to root, isn't in sudoers, has no hardware sensors on a QEMU VM).