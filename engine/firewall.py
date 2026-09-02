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

Usage (standalone test):
    python engine/firewall.py
"""

import os
import subprocess
import time

LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
BLOCKLIST_PATH = os.path.join(LOGS_DIR, "active_blocklist.txt")

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
        self._blocked_ips = set()  # avoid duplicate blocking/log spam

        if mode == "windows":
            print("=" * 60)
            print("WARNING: Firewall running in WINDOWS ENFORCEMENT mode.")
            print("This will actually modify your Windows Firewall rules.")
            print("Use ONLY on an isolated lab network/VM you control.")
            print("=" * 60)

        # Load any IPs already blocked from a previous run, so we don't
        # re-block (and re-log) the same IP every time this process restarts.
        if os.path.exists(BLOCKLIST_PATH):
            with open(BLOCKLIST_PATH) as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 2:
                        self._blocked_ips.add(parts[1])

        print(f"[Firewall] Initialized in '{self.mode}' mode. "
              f"{len(self._blocked_ips)} IP(s) already on blocklist.")

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
        if ip in self._blocked_ips:
            return {"action_taken": "DROP_PACKET", "ip": ip, "blocked": True, "already_blocked": True}
        self._write_blocklist_entry(ip, predicted_class, risk_score, trigger="DROP_PACKET")
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
        rule_name = f"EdgeIDPS_Block_{ip}"
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


def main():
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


if __name__ == "__main__":
    main()