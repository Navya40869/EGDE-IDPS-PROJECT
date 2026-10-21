"""
engine/firewall.py — Phase 10: Firewall Module

Three modes, per the blueprint:
    - Simulation (default): writes to active_blocklist.txt — safe on any
      network, this is what the demo uses.
    - Windows: integrates with Windows Defender Firewall via netsh —
      LAB NETWORK / LAB VM ONLY. Actually blocks traffic on this machine.
    - Linux: not implemented (project runs on Windows) — stub included for
      completeness/documentation only.

Takes the action string from RiskEngine.assess() ("LOG_ONLY", "THROTTLE",
"DROP_PACKET", "BLOCK_IP") and executes (or simulates) the corresponding
response. Only DROP_PACKET/BLOCK_IP result in a blocklist entry — LOG_ONLY
and THROTTLE are informational and don't block anything.

Usage:
    python engine/firewall.py                      # standalone simulation test
    python engine/firewall.py --cleanup --dry-run  # list this project's Windows Firewall rules
    python engine/firewall.py --cleanup            # delete them (needs Administrator)
"""

import argparse
import os
import re
import subprocess
import time

LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
BLOCKLIST_PATH = os.path.join(LOGS_DIR, "active_blocklist.txt")

# Every Windows Firewall rule this project creates is named RULE_PREFIX + <ip>.
# Cleanup only ever touches names that fully match RULE_NAME_RE.
RULE_PREFIX = "EdgeIDPS_Block_"
RULE_NAME_RE = re.compile(r"^EdgeIDPS_Block_[0-9A-Fa-f.:]+$")

os.makedirs(LOGS_DIR, exist_ok=True)


class Firewall:
    def __init__(self, mode: str = "simulation"):
        """
        mode: "simulation" (default, safe anywhere) or "windows"
              (LAB NETWORK ONLY — actually modifies Windows Defender Firewall).
        """
        if mode not in ("simulation", "windows", "linux"):
            raise ValueError(f"Unknown firewall mode: {mode}")
        self.mode = mode
        self._blocked_ips = set()   # IPs with a full BLOCK_IP (real firewall rule)
        self._dropped_ips = set()   # IPs already logged for DROP_PACKET (dedup only, not a real block)

        if mode == "windows":
            print("=" * 60)
            print("WARNING: Firewall running in WINDOWS ENFORCEMENT mode.")
            print("This will actually modify your Windows Firewall rules.")
            print("Use ONLY on an isolated lab network/VM you control.")
            print("=" * 60)

        # Load prior-run state from the blocklist so restarts don't re-log
        # (or, for BLOCK_IP, re-block) the same IP. Keyed by trigger type
        # (column 3) so a DROP_PACKET-only entry doesn't get treated as a
        # real BLOCK_IP on restart — that would let an IP that escalates to
        # BLOCK_IP later get silently skipped because it looked blocked
        # without ever having a real firewall rule.
        if os.path.exists(BLOCKLIST_PATH):
            with open(BLOCKLIST_PATH) as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 3:
                        ip, trigger = parts[1], parts[2]
                        if trigger == "BLOCK_IP":
                            self._blocked_ips.add(ip)
                        elif trigger == "DROP_PACKET":
                            self._dropped_ips.add(ip)

        print(f"[Firewall] Initialized in '{self.mode}' mode. "
              f"{len(self._blocked_ips)} IP(s) already blocked, "
              f"{len(self._dropped_ips)} IP(s) already logged for DROP_PACKET.")

    def act(self, action: str, ip: str, predicted_class: str, risk_score: float) -> dict:
        """
        Executes the given action. Returns a dict describing what happened,
        suitable for telemetry logging.
        """
        if action == "LOG_ONLY":
            return self._log_only(ip, predicted_class, risk_score)
        elif action == "THROTTLE":
            return self._throttle(ip, predicted_class, risk_score)
        elif action == "DROP_PACKET":
            return self._drop_packet(ip, predicted_class, risk_score)
        elif action == "BLOCK_IP":
            return self._block_ip(ip, predicted_class, risk_score)
        else:
            raise ValueError(f"Unknown action: {action}")

    def _log_only(self, ip, predicted_class, risk_score):
        return {"action_taken": "LOG_ONLY", "ip": ip, "blocked": False}

    def _throttle(self, ip, predicted_class, risk_score):
        # Simulation: no real traffic shaping implemented — this is a
        # placeholder for a future rate-limiting integration. Logged as
        # informational only, not written to the blocklist.
        print(f"[Firewall] THROTTLE (simulated, no real rate-limiting applied): {ip}")
        return {"action_taken": "THROTTLE", "ip": ip, "blocked": False}

    def _drop_packet(self, ip, predicted_class, risk_score):
        if ip in self._dropped_ips:
            return {"action_taken": "DROP_PACKET", "ip": ip, "blocked": False, "already_logged": True}
        self._write_blocklist_entry(ip, predicted_class, risk_score, trigger="DROP_PACKET")
        self._dropped_ips.add(ip)
        return {"action_taken": "DROP_PACKET", "ip": ip, "blocked": False,
                "note": "Logged for review; not a full block (see BLOCK_IP for that)."}

    def _block_ip(self, ip, predicted_class, risk_score):
        if ip in self._blocked_ips:
            return {"action_taken": "BLOCK_IP", "ip": ip, "blocked": True, "already_blocked": True}

        self._write_blocklist_entry(ip, predicted_class, risk_score, trigger="BLOCK_IP")
        self._blocked_ips.add(ip)

        if self.mode == "windows":
            self._windows_block(ip)
        elif self.mode == "linux":
            self._linux_block(ip)
        # simulation mode: blocklist entry alone is the "action"

        print(f"[Firewall] BLOCKED: {ip} (predicted_class={predicted_class}, "
              f"risk_score={risk_score})")
        return {"action_taken": "BLOCK_IP", "ip": ip, "blocked": True}

    def _write_blocklist_entry(self, ip, predicted_class, risk_score, trigger):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"{timestamp},{ip},{trigger},{predicted_class},{risk_score}\n"
        with open(BLOCKLIST_PATH, "a") as f:
            f.write(line)

    def _windows_block(self, ip):
        """LAB NETWORK ONLY — actually adds a Windows Firewall rule."""
        rule_name = f"{RULE_PREFIX}{ip}"
        try:
            subprocess.run([
                "netsh", "advfirewall", "firewall", "add", "rule",
                f"name={rule_name}", "dir=in", "action=block",
                f"remoteip={ip}",
            ], check=True, capture_output=True, text=True)
            print(f"[Firewall] Windows Firewall rule added: {rule_name}")
        except subprocess.CalledProcessError as e:
            print(f"[Firewall] ERROR adding Windows Firewall rule: {e.stderr}")
            print("[Firewall] Common cause: not running as Administrator.")

    def _linux_block(self, ip):
        """LAB NETWORK ONLY — not implemented (project targets Windows)."""
        print(f"[Firewall] Linux mode not implemented — would run: "
              f"iptables -A INPUT -s {ip} -j DROP")

    def is_blocked(self, ip: str) -> bool:
        return ip in self._blocked_ips


def list_project_rules() -> list:
    """Display names of Windows Firewall rules created by _windows_block().

    Uses Get-NetFirewallRule rather than parsing `netsh ... show rule`, whose
    output labels are localized. A netsh rule's name= is its DisplayName.
    """
    result = subprocess.run([
        "powershell", "-NoProfile", "-NonInteractive", "-Command",
        f"Get-NetFirewallRule -DisplayName '{RULE_PREFIX}*' -ErrorAction SilentlyContinue "
        f"| ForEach-Object {{ $_.DisplayName }}",
    ], check=True, capture_output=True, text=True)
    names = {line.strip() for line in result.stdout.splitlines()}
    return sorted(n for n in names if RULE_NAME_RE.match(n))


def cleanup_windows_rules(dry_run: bool = False) -> int:
    """Deletes only EdgeIDPS_Block_<ip> rules. Does not touch active_blocklist.txt."""
    names = list_project_rules()
    if not names:
        print(f"[Firewall] No '{RULE_PREFIX}*' rules found.")
        return 0
    print(f"[Firewall] Found {len(names)} project rule name(s):")
    for name in names:
        print(f"    {name}")
    if dry_run:
        print("[Firewall] --dry-run: nothing deleted.")
        return 0

    failures = 0
    for name in names:
        # netsh deletes every rule with this exact name, so duplicates go too.
        result = subprocess.run([
            "netsh", "advfirewall", "firewall", "delete", "rule", f"name={name}",
        ], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"[Firewall] Deleted: {name}")
        else:
            failures += 1
            print(f"[Firewall] ERROR deleting {name}: {(result.stdout + result.stderr).strip()}")
    if failures:
        print("[Firewall] Common cause: not running as Administrator.")
    return 1 if failures else 0


def run_simulation_test():
    print("Standalone Firewall test (Simulation mode) — no real network changes made.\n")
    fw = Firewall(mode="simulation")

    print("\n--- Test: LOG_ONLY ---")
    print(fw.act("LOG_ONLY", "10.0.0.1", "Benign", 0.0))

    print("\n--- Test: THROTTLE ---")
    print(fw.act("THROTTLE", "10.0.0.2", "Spoofing", 4.5))

    print("\n--- Test: DROP_PACKET ---")
    print(fw.act("DROP_PACKET", "10.0.0.3", "Reconnaissance", 7.0))

    print("\n--- Test: BLOCK_IP ---")
    print(fw.act("BLOCK_IP", "10.0.0.4", "DoS/DDoS", 9.5))

    print("\n--- Test: duplicate BLOCK_IP (should not re-log) ---")
    print(fw.act("BLOCK_IP", "10.0.0.4", "DoS/DDoS", 9.8))

    print(f"\nBlocklist file contents ({BLOCKLIST_PATH}):")
    with open(BLOCKLIST_PATH) as f:
        print(f.read())


def main():
    parser = argparse.ArgumentParser(description="Edge-IDPS firewall module")
    parser.add_argument("--cleanup", action="store_true",
                        help=f"Delete Windows Firewall rules named {RULE_PREFIX}<ip>")
    parser.add_argument("--dry-run", action="store_true",
                        help="With --cleanup: list matching rules without deleting")
    args = parser.parse_args()

    if args.dry_run and not args.cleanup:
        parser.error("--dry-run only applies with --cleanup")
    if args.cleanup:
        raise SystemExit(cleanup_windows_rules(dry_run=args.dry_run))
    run_simulation_test()


if __name__ == "__main__":
    main()