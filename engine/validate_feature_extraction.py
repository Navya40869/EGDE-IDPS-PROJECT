"""
engine/validate_feature_extraction.py — Phase 7 mandatory validation
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from scapy.all import IP, TCP

from feature_extractor import FeatureExtractor, PORT_MAP
from flow_manager import FlowManager
from packet_sniffer import PacketSniffer

PROCESSED_DIR = os.path.join("data", "processed")
FROZEN_FEATURES_PATH = os.path.join(PROCESSED_DIR, "frozen_features.json")

TOLERANCE = 1e-6


def independent_recompute(flow) -> dict:
    packets = flow.packets

    records = []
    for p in packets:
        rec = {
            "size": len(p),
            "time": float(p.time),
            "ttl": p[IP].ttl if p.haslayer(IP) else None,
            "ihl": (p[IP].ihl * 4) if (p.haslayer(IP) and hasattr(p[IP], "ihl")) else None,
            "is_fin": bool(p.haslayer(TCP) and (p[TCP].flags & 0x01)),
            "is_syn": bool(p.haslayer(TCP) and (p[TCP].flags & 0x02)),
            "is_ack": bool(p.haslayer(TCP) and (p[TCP].flags & 0x10)),
            "is_psh": bool(p.haslayer(TCP) and (p[TCP].flags & 0x08)),
        }
        records.append(rec)
    df = pd.DataFrame(records)

    duration = max(flow.last_packet_time - flow.start_time, 1e-6)
    n = len(df)

    results = {
        "Tot sum": df["size"].sum(),
        "Max": df["size"].max(),
        "AVG": df["size"].mean(),
        "Tot size": df["size"].sum(),
        "Rate": n / duration,
        "Time_To_Live": df["ttl"].mean(skipna=True) if df["ttl"].notna().any() else 0.0,
        "Header_Length": df["ihl"].mean(skipna=True) if df["ihl"].notna().any() else 0.0,
        "ack_flag_number": df["is_ack"].sum(),
        "IAT": df["time"].diff().dropna().mean() if n > 1 else 0.0,
        "Min": df["size"].min(),
        "Std": df["size"].std(ddof=0) if n > 1 else 0.0,
        "ack_count": df["is_ack"].sum(),
        "HTTPS": 1.0 if (flow.src_port == PORT_MAP["HTTPS"] or flow.dst_port == PORT_MAP["HTTPS"]) else 0.0,
        "Protocol Type": {"TCP": 6, "UDP": 17, "ICMP": 1}.get(flow.protocol, 0),
        "psh_flag_number": df["is_psh"].sum(),
        "TCP": 1.0 if flow.protocol == "TCP" else 0.0,
        "ICMP": 1.0 if flow.protocol == "ICMP" else 0.0,
        "UDP": 1.0 if flow.protocol == "UDP" else 0.0,
        "syn_count": df["is_syn"].sum(),
        "syn_flag_number": df["is_syn"].sum(),
    }
    return {k: float(v) for k, v in results.items()}


def main():
    parser = argparse.ArgumentParser(description="Validate feature_extractor.py against independent recomputation")
    parser.add_argument("--interface", default=None)
    parser.add_argument("--duration", type=int, default=20)
    args = parser.parse_args()

    sniffer = PacketSniffer(interface=args.interface)
    flow_manager = FlowManager(packet_queue=sniffer.queue)

    sniffer.start()
    flow_manager.start()

    print(f"Capturing for {args.duration}s to find a multi-packet flow to validate ...")
    time.sleep(args.duration)

    flow_manager.stop()
    sniffer.stop()

    target_flow = None
    while not flow_manager.closed_flow_queue.empty():
        flow = flow_manager.closed_flow_queue.get()
        if len(flow.packets) > 1:
            target_flow = flow
            break

    if target_flow is None:
        print("\nNo multi-packet flow captured in this window — try again with more "
              "traffic (e.g. run a benign or DoS attack in a second terminal during "
              "this capture window).")
        return

    print(f"\nValidating against flow: {target_flow.flow_id}, "
          f"{len(target_flow.packets)} packets, closed_reason={target_flow.closed_reason}\n")

    extractor = FeatureExtractor()
    online_raw = extractor._compute_raw_features(target_flow)
    offline_raw = independent_recompute(target_flow)

    print(f"{'Feature':<20} {'Online':>15} {'Offline':>15} {'Diff':>12} {'Result':>8}")
    print("-" * 75)
    all_passed = True
    for feature in extractor.frozen_feature_order:
        online_val = online_raw.get(feature, float("nan"))
        offline_val = offline_raw.get(feature, float("nan"))
        diff = abs(online_val - offline_val)
        passed = diff < TOLERANCE
        if not passed:
            all_passed = False
        status = "PASS" if passed else "FAIL"
        print(f"{feature:<20} {online_val:>15.4f} {offline_val:>15.4f} {diff:>12.6f} {status:>8}")

    print("-" * 75)
    if all_passed:
        print("\nALL 20 FROZEN FEATURES PASSED — feature_extractor.py is validated "
              "against independent recomputation. Safe to proceed to Phase 8.")
    else:
        print("\nSOME FEATURES FAILED — do not proceed to Phase 8 until these are "
              "fixed. Check the failing feature's definition in both "
              "feature_extractor.py and this script's independent_recompute().")


if __name__ == "__main__":
    main()