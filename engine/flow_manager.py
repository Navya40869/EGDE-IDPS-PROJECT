"""
engine/flow_manager.py — Phase 6: Flow Management

Groups packets by the 5-tuple (source IP, destination IP, source port,
destination port, protocol) into Flow objects. Closes a flow on:
    - TCP FIN
    - TCP RST
    - Idle timeout (~5s, no packets seen)
    - Max flow duration (~15s, regardless of activity)
UDP/ICMP flows: timeout-based closure only (no FIN/RST concept).

Does NOT compute the frozen feature set here — that's feature_extractor.py's
job (Phase 7). This module's only responsibility is grouping packets into
flows and deciding when a flow is "done", handing the RAW packet list for a
closed flow to whatever consumes it next.

Usage (standalone test):
    python engine/flow_manager.py --interface "Wi-Fi" --duration 30
"""

import argparse
import queue
import threading
import time
from dataclasses import dataclass, field

from scapy.all import IP, TCP, UDP, ICMP, ARP

IDLE_TIMEOUT_SECONDS = 5
MAX_FLOW_DURATION_SECONDS = 15
CLEANUP_INTERVAL_SECONDS = 1  # how often to scan for timed-out flows


@dataclass
class Flow:
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str  # "TCP", "UDP", "ICMP"

    start_time: float = field(default_factory=time.time)
    last_packet_time: float = field(default_factory=time.time)
    packets: list = field(default_factory=list)
    closed_reason: str = None  # "FIN", "RST", "IDLE_TIMEOUT", "MAX_DURATION"

    @property
    def flow_id(self):
        return (self.src_ip, self.dst_ip, self.src_port, self.dst_port, self.protocol)

    def add_packet(self, packet):
        self.packets.append(packet)
        self.last_packet_time = time.time()


def extract_5tuple(packet):
    if packet.haslayer(ARP):
        arp = packet[ARP]
        # ARP has no ports — use the claimed sender/target IPs as the key,
        # matching how ICMP already zeros out ports below.
        return (arp.psrc, arp.pdst, 0, 0, "ARP")

    if not packet.haslayer(IP):
        return None
    # ... rest of the function stays exactly the same

    ip_layer = packet[IP]
    src_ip = ip_layer.src
    dst_ip = ip_layer.dst

    if packet.haslayer(TCP):
        proto = "TCP"
        src_port = packet[TCP].sport
        dst_port = packet[TCP].dport
    elif packet.haslayer(UDP):
        proto = "UDP"
        src_port = packet[UDP].sport
        dst_port = packet[UDP].dport
    elif packet.haslayer(ICMP):
        proto = "ICMP"
        src_port = 0  # ICMP has no ports
        dst_port = 0
    else:
        return None  # unsupported protocol for this project's scope

    return (src_ip, dst_ip, src_port, dst_port, proto)


class FlowManager:
    def __init__(self, packet_queue: queue.Queue, closed_flow_queue: queue.Queue = None,
                 idle_timeout: int = IDLE_TIMEOUT_SECONDS,
                 max_duration: int = MAX_FLOW_DURATION_SECONDS):
        self.packet_queue = packet_queue
        self.closed_flow_queue = closed_flow_queue if closed_flow_queue is not None else queue.Queue()
        self.idle_timeout = idle_timeout
        self.max_duration = max_duration

        self.active_flows = {}  # flow_id -> Flow
        self._lock = threading.Lock()  # protects active_flows across threads

        self._stop_event = threading.Event()
        self._consumer_thread = None
        self._cleanup_thread = None

        self._flows_closed_count = 0

    def start(self):
        self._stop_event.clear()
        self._consumer_thread = threading.Thread(target=self._consume_packets, daemon=True)
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._consumer_thread.start()
        self._cleanup_thread.start()
        print(f"[FlowManager] Started. idle_timeout={self.idle_timeout}s, "
              f"max_duration={self.max_duration}s")

    def _consume_packets(self):
        """Pulls packets from the sniffer's queue and routes them into flows."""
        while not self._stop_event.is_set():
            try:
                packet = self.packet_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            tup = extract_5tuple(packet)
            if tup is None:
                continue
            src_ip, dst_ip, src_port, dst_port, proto = tup
            flow_id = (src_ip, dst_ip, src_port, dst_port, proto)

            with self._lock:
                flow = self.active_flows.get(flow_id)
                if flow is None:
                    flow = Flow(src_ip, dst_ip, src_port, dst_port, proto)
                    self.active_flows[flow_id] = flow
                flow.add_packet(packet)

                # TCP FIN/RST closure — check flags on this packet
                if proto == "TCP":
                    flags = packet[TCP].flags
                    if flags & 0x01:  # FIN
                        self._close_flow(flow_id, "FIN")
                    elif flags & 0x04:  # RST
                        self._close_flow(flow_id, "RST")

    def _cleanup_loop(self):
        """Periodically scans active flows for idle timeout / max duration closure."""
        while not self._stop_event.is_set():
            time.sleep(CLEANUP_INTERVAL_SECONDS)
            now = time.time()
            with self._lock:
                # list() to avoid mutating dict while iterating
                for flow_id, flow in list(self.active_flows.items()):
                    if now - flow.last_packet_time >= self.idle_timeout:
                        self._close_flow(flow_id, "IDLE_TIMEOUT")
                    elif now - flow.start_time >= self.max_duration:
                        self._close_flow(flow_id, "MAX_DURATION")

    def _close_flow(self, flow_id, reason):
        """Must be called with self._lock already held."""
        flow = self.active_flows.pop(flow_id, None)
        if flow is None:
            return  # already closed by another path
        flow.closed_reason = reason
        self._flows_closed_count += 1
        try:
            self.closed_flow_queue.put_nowait(flow)
        except queue.Full:
            print(f"[FlowManager] WARNING: closed_flow_queue full, dropping flow {flow_id}")

    def stop(self):
        print("[FlowManager] Stopping ...")
        self._stop_event.set()
        # Close any remaining active flows so nothing is silently lost
        with self._lock:
            for flow_id in list(self.active_flows.keys()):
                self._close_flow(flow_id, "SHUTDOWN")
        if self._consumer_thread:
            self._consumer_thread.join(timeout=3)
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=3)
        print(f"[FlowManager] Stopped. Total flows closed: {self._flows_closed_count}")

    def stats(self):
        with self._lock:
            active_count = len(self.active_flows)
        return {
            "active_flows": active_count,
            "closed_flows_total": self._flows_closed_count,
            "closed_flow_queue_depth": self.closed_flow_queue.qsize(),
        }


def main():
    from packet_sniffer import PacketSniffer

    parser = argparse.ArgumentParser(description="Edge-IDPS flow manager (standalone test)")
    parser.add_argument("--interface", default=None)
    parser.add_argument("--duration", type=int, default=30)
    args = parser.parse_args()

    sniffer = PacketSniffer(interface=args.interface)
    flow_manager = FlowManager(packet_queue=sniffer.queue)

    sniffer.start()
    flow_manager.start()

    print(f"Running for {args.duration}s ... (Ctrl+C to stop early)")
    try:
        start = time.time()
        while time.time() - start < args.duration:
            time.sleep(1)
            sniff_stats = sniffer.stats()
            flow_stats = flow_manager.stats()
            print(f"  captured={sniff_stats['captured']} | "
                  f"active_flows={flow_stats['active_flows']} | "
                  f"closed_flows={flow_stats['closed_flows_total']}")
    except KeyboardInterrupt:
        print("\nStopped early by user.")

    flow_manager.stop()
    sniffer.stop()

    # Drain and summarize closed flows
    print("\nClosed flow summary (first 10):")
    count = 0
    while not flow_manager.closed_flow_queue.empty() and count < 10:
        flow = flow_manager.closed_flow_queue.get()
        print(f"  {flow.flow_id} | {len(flow.packets)} packets | "
              f"closed_reason={flow.closed_reason} | "
              f"duration={flow.last_packet_time - flow.start_time:.2f}s")
        count += 1


if __name__ == "__main__":
    main()