"""
engine/packet_sniffer.py — Phase 5: Packet Capture

Captures raw packets in a dedicated thread and pushes them onto a thread-safe
queue. Does NOT perform inference here — that's the flow manager's and
feature extractor's job downstream. Capture and processing must stay in
separate threads, or the flood generators from scapy_attacker.py will
overwhelm a single blocking sniff-then-infer loop and packets will be dropped.

Requires admin/root privileges (same as scapy_attacker.py) and Npcap on Windows.

Usage (standalone test):
    python engine/packet_sniffer.py --interface "Wi-Fi" --duration 20

In the full pipeline, PacketSniffer is imported and driven by
stream_pipeline.py (Phase 13/14), which also owns the shared shutdown Event.
"""

import argparse
import queue
import threading
import time

from scapy.all import sniff, get_if_list


class PacketSniffer:
    def __init__(self, interface: str = None, packet_queue: queue.Queue = None,
                 queue_maxsize: int = 10000, bpf_filter: str = "ip or arp"):
        """
        interface: network interface name to sniff on (None = Scapy default).
        packet_queue: thread-safe queue to push captured packets onto. If not
                      provided, one is created with queue_maxsize.
        bpf_filter: BPF filter string — "ip" captures all IPv4 traffic, the
                    baseline for this project (flow manager keys on 5-tuple).
        """
        self.interface = interface
        self.queue = packet_queue if packet_queue is not None else queue.Queue(maxsize=queue_maxsize)
        self.bpf_filter = bpf_filter

        self._stop_event = threading.Event()
        self._thread = None
        self._packets_captured = 0
        self._packets_dropped = 0

    def _packet_callback(self, packet):
        """Called by Scapy for every captured packet — push to queue, never process here."""
        try:
            self.queue.put_nowait(packet)
            self._packets_captured += 1
        except queue.Full:
            # Queue overflow policy: drop the newest packet and log a warning.
            # Never block capture indefinitely waiting for space (Developer Guide
            # error-handling table).
            self._packets_dropped += 1
            if self._packets_dropped % 100 == 1:  # don't spam the console
                print(f"[WARNING] Packet queue full — dropped {self._packets_dropped} "
                      f"packets so far. Consider increasing queue_maxsize or speeding "
                      f"up the consumer (flow manager).")

    def _stop_filter(self, packet):
        """Scapy calls this after every packet; returning True stops the sniff loop."""
        return self._stop_event.is_set()

    def start(self):
        """Start capturing in a background thread. Non-blocking — returns immediately."""
        if self._thread is not None and self._thread.is_alive():
            print("[PacketSniffer] Already running.")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"[PacketSniffer] Started on interface={self.interface or 'default'}, "
              f"filter='{self.bpf_filter}'")

    def _run(self):
        try:
            sniff(
                iface=self.interface,
                filter=self.bpf_filter,
                prn=self._packet_callback,
                stop_filter=self._stop_filter,
                store=False,  # never store the full packet list in memory
            )
        except Exception as e:
            print(f"[PacketSniffer] ERROR: sniff() failed: {e}")
            print("[PacketSniffer] Common causes: not running as Administrator, "
                  "Npcap not installed, or invalid interface name.")

    def stop(self):
        """Signal the capture loop to stop. Waits briefly for the thread to exit cleanly."""
        print("[PacketSniffer] Stopping ...")
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        print(f"[PacketSniffer] Stopped. Captured: {self._packets_captured}, "
              f"Dropped: {self._packets_dropped}")

    def stats(self):
        return {
            "captured": self._packets_captured,
            "dropped": self._packets_dropped,
            "queue_depth": self.queue.qsize(),
        }


def main():
    parser = argparse.ArgumentParser(description="Edge-IDPS packet sniffer (standalone test)")
    parser.add_argument("--interface", default=None,
                         help="Network interface name. Omit to use Scapy's default. "
                              "Run with --list to see available interfaces.")
    parser.add_argument("--duration", type=int, default=20,
                         help="How long to capture, in seconds")
    parser.add_argument("--list", action="store_true", help="List available interfaces and exit")
    args = parser.parse_args()

    if args.list:
        print("Available interfaces:")
        for iface in get_if_list():
            print(f"  {iface}")
        return

    sniffer = PacketSniffer(interface=args.interface)
    sniffer.start()

    print(f"Capturing for {args.duration}s ... (Ctrl+C to stop early)")
    try:
        start = time.time()
        while time.time() - start < args.duration:
            time.sleep(1)
            stats = sniffer.stats()
            print(f"  captured={stats['captured']} dropped={stats['dropped']} "
                  f"queue_depth={stats['queue_depth']}")
    except KeyboardInterrupt:
        print("\nStopped early by user.")

    sniffer.stop()
    print(f"\nFinal stats: {sniffer.stats()}")


if __name__ == "__main__":
    main()