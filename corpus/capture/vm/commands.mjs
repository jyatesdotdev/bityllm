// vm/commands.mjs — what to capture inside the real Debian VM.
//
// Only things a container can't produce: the kernel ring buffer (boot dmesg), the
// systemd journal, service/unit state, and boot identity. The full reboot
// lifecycle (shutdown -> kernel boot -> startup -> login) is captured separately
// from Lima's serial-console log, not as a command here.

export const VM_TIMEOUT = 15;
const g = (cat, ...cmds) => cmds.map((cmd) => ({ cat, cmd }));

export const VM_COMMANDS = [
  // --- kernel ring buffer: the boot dmesg ---
  ...g("dmesg",
    "dmesg", "dmesg -T", "dmesg --level=err,warn",
    "dmesg | tail -n 40", "dmesg | grep -iE 'memory|Memory' | head -n 20",
    "dmesg | grep -iE 'cpu|smp|clock' | head -n 20",
    "dmesg | grep -iE 'virtio|scsi|blk|net|eth' | head -n 25",
  ),

  // --- systemd journal ---
  ...g("journal",
    "journalctl -b --no-pager | head -n 250",
    "journalctl -b -k --no-pager | head -n 200",
    "journalctl -b -p err --no-pager | head -n 60",
    "journalctl -b -1 --no-pager | tail -n 70", // previous boot: contains the shutdown tail
    "journalctl -u ssh --no-pager | tail -n 25",
    "journalctl -u systemd-logind --no-pager | tail -n 20",
    "journalctl --disk-usage",
  ),

  // --- systemd / service state ---
  ...g("systemd",
    "systemctl status --no-pager | head -n 25",
    "systemctl list-units --type=service --state=running --no-pager | head -n 40",
    "systemctl list-units --failed --no-pager",
    "systemctl is-system-running",
    "systemd-analyze",
    "systemd-analyze blame --no-pager | head -n 25",
    "systemd-analyze critical-chain --no-pager | head -n 25",
    "hostnamectl", "timedatectl", "localectl",
  ),

  // --- boot / uptime identity (the reconnect banner half) ---
  ...g("boot",
    "uptime", "uptime -p", "uptime -s", "who -b", "runlevel",
    "last reboot --no-pager | head -n 10", "last --no-pager | head -n 12",
    "cat /etc/issue", "cat /etc/motd", "cat /etc/os-release", "cat /proc/cmdline",
  ),
];
