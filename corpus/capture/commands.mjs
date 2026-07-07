// commands.mjs — what to install and what to run inside the Debian container.
//
// STATIC   : a curated, high-value command set (system state, filesystem, text
//            tools, deliberate errors, and "fun" binaries).
// HARVEST  : dynamic expansions that enumerate the container itself (man pages,
//            --help for every binary, package metadata, /etc, /usr/share/doc, …)
//            to reach a few MB of real, format-perfect terminal text.
//
// Everything here is designed to be non-interactive and non-destructive; the
// runner additionally wraps each command in `timeout` with stdin closed.

export const PERSONA = { user: "guest", host: "bity", prompt: "guest@bity:~$ " };

// Packages installed once at container setup. --no-install-recommends keeps it
// lean; man pages + a broad CLI toolset maximize harvestable text and give the
// terminal persona lots of "binaries" to imitate later.
export const TOOLSET = [
  "man-db", "manpages", "manpages-dev",
  "coreutils", "util-linux", "bsdextrautils", "bsdmainutils", "procps", "psmisc",
  "iproute2", "net-tools", "iputils-ping", "dnsutils", "traceroute", "whois",
  "curl", "wget", "git", "vim-tiny", "nano", "less", "tree", "file", "lsof", "pciutils",
  "htop", "locales", "tzdata", "lsb-release", "hostname",
  "sysstat", "rsync", "tar", "gzip", "xz-utils", "zip", "unzip", "jq", "bc",
  "cowsay", "fortune-mod", "fortunes", "figlet", "neofetch", "fastfetch", "screenfetch",
  "ca-certificates",
];

const group = (cat, ...cmds) => cmds.map((cmd) => ({ cat, cmd }));

export const STATIC = [
  // --- system identity / state (format-rich, the "fun" ones) ---
  ...group("sys",
    "uname -a", "uname -r", "uname -m", "hostname", "hostname -f", "uptime",
    "date", "date -u", "date +%s", "cal", "cal 2026", "ncal", "arch", "nproc",
    "whoami", "id", "groups", "who", "w", "users",
    "lscpu", "free -h", "free -m", "df -h", "df -hT", "vmstat", "lsblk",
    "cat /proc/version", "cat /proc/cpuinfo", "cat /proc/meminfo", "cat /proc/loadavg", "cat /proc/uptime",
    "cat /etc/os-release", "cat /etc/debian_version", "lsb_release -a", "locale",
    "ps aux", "ps -ef", "ps aux --sort=-%mem", "top -bn1", "pstree",
    "mount", "cat /etc/fstab", "getent passwd", "getent group",
    "echo $SHELL", "echo $HOME", "echo $PATH", "echo $LANG", "printenv PATH",
  ),

  // --- networking ---
  ...group("net",
    "ip addr", "ip -br addr", "ip route", "ip link", "ip -s link", "ifconfig",
    "ip neigh", "cat /etc/hosts", "cat /etc/resolv.conf", "cat /etc/nsswitch.conf",
    "ss -tuln", "netstat -i", "hostname -I", "getent hosts localhost",
    "ping -c 3 localhost", "ping -c 4 bity.dev", "traceroute -m 5 localhost",
  ),

  // --- filesystem exploration ---
  ...group("fs",
    "pwd", "ls", "ls -a", "ls -l", "ls -la", "ls -lh", "ls -lS", "ls -lt", "ls --color=never -la",
    "ls -la /", "ls -la /etc", "ls -la /home", "ls -la /home/guest", "ls -la /tmp",
    "ls -la /var", "ls -la /var/log", "ls -la /usr", "ls /usr/bin | head -n 80",
    "tree -L 1 /", "tree -L 2 /etc | head -n 100", "tree -L 2 /home",
    "stat /etc/os-release", "stat /home/guest", "file /bin/ls", "file /etc/hosts",
    "du -sh /etc", "du -h /etc | tail -n 15", "find /etc -maxdepth 1 -type f | head -n 40",
  ),

  // --- reading common config/text files ---
  ...group("cat",
    "cat /etc/passwd", "cat /etc/group", "head -n 20 /etc/services", "tail -n 12 /etc/services",
    "cat /etc/protocols | head -n 30", "wc -l /etc/passwd", "wc /etc/services",
    "cut -d: -f1 /etc/passwd", "grep bash /etc/passwd", "sed -n '1,5p' /etc/passwd",
    "cat /etc/apt/sources.list", "cat /etc/login.defs | head -n 25", "cat /etc/shells",
  ),

  // --- text / coreutils toys ---
  ...group("tool",
    "echo hello world", "echo $((6 * 7))", "seq 1 20", "seq 1 2 30", "seq -w 1 10",
    "factor 360", "factor 1234567", "expr 12 \\* 12", "printf '%s\\n' alpha beta gamma",
    "yes bity | head -n 5", "sha256sum /etc/hostname", "md5sum /etc/os-release",
    "base64 /etc/hostname", "od -c /etc/hostname | head", "wc -c /etc/os-release",
    "awk -F: '{print $1}' /etc/passwd | head", "sort -r /etc/shells", "uniq -c /etc/shells",
    "tr a-z A-Z <<< 'hello bity'", "rev <<< 'hello bity'", "head -c 24 /dev/urandom | base64",
  ),

  // --- deliberate errors (great texture for a realistic terminal) ---
  ...group("err",
    "cat /etc/shadow", "ls /root", "ls /nonexistent-dir", "cat /does/not/exist",
    "cd /nowhere", "definitely-not-a-command", "mkdir /etc/nope", "rmdir /",
    "false", "ls --this-flag-does-not-exist", "grep", "ping", "git status",
    "apt install cowsay", "kill 999999", "head /no/such/file", "touch /etc/x",
  ),

  // --- "fun" binaries (their real output seeds the fun generators later) ---
  ...group("fun",
    "cowsay 'moo from bity'", "cowsay -f tux 'hello world'", "cowsay -d 'i am tired'",
    "fortune", "fortune", "fortune", "fortune -s", "fortune | cowsay", "fortune -s | cowsay -f tux",
    "cowsay -l | head -n 20", "cowthink 'hmm'",
    "figlet bity", "figlet -f slant term", "figlet 'hello world'", "figlet -k BITY",
    "neofetch --stdout", "screenfetch -N", "fastfetch --pipe", "fastfetch --logo none",
  ),
];

// Binaries we must NOT pass `--help` to (they ignore the flag and go
// interactive, prompt, mangle the terminal, or are destructive). The runner's
// timeout + closed stdin is a backstop, but excluding them keeps output clean.
export const HELP_BLACKLIST = new Set([
  "vi", "vim", "vim.tiny", "vim.basic", "vim.nox", "view", "nano", "emacs", "ed",
  "sensible-editor", "editor", "select-editor", "run-mailcap",
  "less", "more", "most", "pager", "pico", "joe", "mcedit",
  "top", "htop", "atop", "watch", "man", "apropos", "whatis", "mandb",
  "reset", "clear", "tput", "tset", "setterm", "openvt", "chvt",
  "su", "sudo", "runuser", "login", "sulogin", "newgrp",
  "passwd", "gpasswd", "chsh", "chfn", "vipw", "vigr", "visudo",
  "sh", "bash", "dash", "rbash", "zsh", "fish", "csh", "tcsh", "ksh", "mksh", "busybox",
  "screen", "tmux", "byobu", "telnet", "nc", "nc.openbsd", "ncat", "socat", "ftp", "tftp",
  "python", "python3", "perl", "ruby", "irb", "node", "nodejs", "php", "lua", "gdb", "lldb",
  "openssl", "mysql", "mariadb", "psql", "sqlite3", "redis-cli", "ipython",
  "startx", "xinit", "X", "Xorg", "systemctl", "journalctl", "systemd", "init", "telinit",
  "shutdown", "reboot", "halt", "poweroff",
  "dd", "mkfs", "mkswap", "fdisk", "cfdisk", "sfdisk", "parted", "wipefs", "shred",
  "crontab", "dpkg-reconfigure", "debconf", "whiptail", "dialog",
  "rlwrap", "script", "scriptreplay", "expect", "minicom", "picocom", "cu", "yes",
  "chroot", "unshare", "nsenter", "pivot_root", "switch_root",
]);

const NAME_OK = /^[A-Za-z0-9][A-Za-z0-9._+-]*$/;
const BINARY_EXT = /\.(gz|xz|bz2|zst|db|bin|dat|pyc|so|o|a|png|jpe?g|gif|ico|gpg|kbx|key|pem|der|crt|deb|img)$/i;

export const isTextPath = (p) =>
  !BINARY_EXT.test(p) && !p.includes("/private/") && !p.includes("/ssl/private");

// Each harvest spec runs `query` in the container, then turns each output line
// into one or more commands. `limit` caps how many lines we consume so no single
// category dominates when we interleave.
export const HARVEST = [
  {
    cat: "man",
    limit: 700,
    query:
      "ls /usr/share/man/man1 /usr/share/man/man5 /usr/share/man/man7 /usr/share/man/man8 2>/dev/null " +
      "| sed -E 's/\\.[0-9].*$//' | sort -u",
    make: (name) => (NAME_OK.test(name) ? [`man ${name} 2>/dev/null | head -n 200`] : []),
  },
  {
    cat: "help",
    limit: 450,
    query: "ls /usr/bin /bin /usr/sbin /sbin 2>/dev/null | sort -u",
    make: (bin) =>
      NAME_OK.test(bin) && !HELP_BLACKLIST.has(bin)
        ? [`${bin} --help`, `${bin} --version`]
        : [],
  },
  {
    cat: "pkg",
    limit: 400,
    query: "dpkg-query -W -f='${Package}\\n' 2>/dev/null | sort -u",
    make: (p) =>
      NAME_OK.test(p) ? [`apt-cache show ${p} 2>/dev/null`, `dpkg -L ${p} 2>/dev/null | head -n 40`] : [],
  },
  {
    cat: "etc",
    limit: 300,
    query: "find /etc -maxdepth 3 -type f -size -24k 2>/dev/null | sort | head -n 400",
    make: (f) => (isTextPath(f) ? [`cat ${f}`] : []),
  },
  {
    cat: "doc",
    limit: 250,
    query:
      "find /usr/share/doc -maxdepth 2 \\( -name copyright -o -name 'README*' -o -name 'changelog*' \\) " +
      "2>/dev/null | sort | head -n 400",
    make: (f) =>
      f.endsWith(".gz")
        ? [`zcat ${f} 2>/dev/null | head -n 60`]
        : [`cat ${f} 2>/dev/null | head -n 60`],
  },
  {
    cat: "ls",
    limit: 200,
    query:
      "find / -maxdepth 3 -type d 2>/dev/null | grep -vE '^/(proc|sys|dev|run)' | sort -u | head -n 400",
    make: (d) => [`ls -la ${d}`],
  },
];
