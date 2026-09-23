"""
engine/inspect_features.py — Phase 7 diagnostic.
For real, well-formed captured flows (packet_count >= 3, so we skip the
already-confirmed single-packet degenerate case), print each of the 37 raw
features next to the scaler's mean/scale and the resulting z-score, sorted
by |z| descending, with the frozen-20 features flagged. This tells us
exactly which features are driving the CNN's decision for legitimate DNS/
HTTPS traffic.

Usage:
    python engine/inspect_features.py --interface "Wi-Fi" --duration 30 --max-flows 8
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from packet_sniffer import PacketSniffer
from flow_manager import FlowManager
from feature_extractor import FeatureExtractor


def inspect_flow(extractor, flow):
    raw = extractor._compute_raw_features(flow)
    full_vector = np.array([[raw[f] for f in extractor.full_feature_order]], dtype=np.float32)
    mean = extractor.scaler.mean_
    scale = extractor.scaler.scale_
    z = (full_vector[0] - mean) / scale

    duration = flow.last_packet_time - flow.start_time
    print(f"\n=== Flow {flow.flow_id} | proto={flow.protocol} "
          f"packets={len(flow.packets)} closed={flow.closed_reason} "
          f"duration={duration:.6f} ===")
    print(f"{'feature':<20}{'raw':>14}{'mean':>14}{'scale':>14}{'z-score':>12}  ")

    rows = []
    for i, fname in enumerate(extractor.full_feature_order):
        is_frozen = fname in extractor.frozen_feature_order
        rows.append((abs(z[i]), fname, full_vector[0][i], mean[i], scale[i], z[i], is_frozen))
    rows.sort(reverse=True)
    for _, fname, raw_v, m, s, zz, frozen in rows:
        marker = "  <-- FROZEN" if frozen else ""
        print(f"{fname:<20}{raw_v:>14.4f}{m:>14.4f}{s:>14.4f}{zz:>12.2f}{marker}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface", default=None)
    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument("--max-flows", type=int, default=8)
    args = parser.parse_args()

    extractor = FeatureExtractor()
    sniffer = PacketSniffer(interface=args.interface)
    flow_manager = FlowManager(packet_queue=sniffer.queue)
    sniffer.start()
    flow_manager.start()

    print(f"Capturing for {args.duration}s (skipping packet_count < 3 flows) ...")
    start = time.time()
    inspected = 0
    try:
        while time.time() - start < args.duration and inspected < args.max_flows:
            time.sleep(0.5)
            while not flow_manager.closed_flow_queue.empty() and inspected < args.max_flows:
                flow = flow_manager.closed_flow_queue.get()
                if len(flow.packets) < 3:
                    continue
                inspect_flow(extractor, flow)
                inspected += 1
    except KeyboardInterrupt:
        pass

    flow_manager.stop()
    sniffer.stop()


if __name__ == "__main__":
    main()