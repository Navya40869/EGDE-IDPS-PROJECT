"""
engine/stream_pipeline.py — Full pipeline integration (brought forward from
Phase 14, since the Dashboard needs a live telemetry source to poll).

Wires together everything built so far:
    PacketSniffer -> FlowManager -> InferenceEngine -> RiskEngine -> Firewall

For every classified flow, writes one append-only row to
logs/live_stream_output.csv: timestamp, flow_id, source_ip, destination_ip,
predicted_class, confidence, risk_score, action, model_version.

This is the only piece of the whole project that WRITES telemetry — the
dashboard only ever reads it (see dashboard/app.py).

Usage:
    python engine/stream_pipeline.py --interface "Wi-Fi" --duration 60 --firewall-mode simulation
"""

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))

from packet_sniffer import PacketSniffer
from flow_manager import FlowManager
from risk_engine import RiskEngine
from firewall import Firewall

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
TELEMETRY_PATH = os.path.join(LOGS_DIR, "live_stream_output.csv")

TELEMETRY_HEADER = [
    "timestamp", "flow_id", "source_ip", "destination_ip",
    "predicted_class", "confidence", "risk_score", "action", "model_version",
]


def get_model_version(inference_engine) -> str:
    meta = inference_engine.metadata
    return f"5class_{meta.get('training_date', 'unknown')}"


def write_telemetry_row(row: dict):
    """Append-only — never truncates or rewrites existing rows."""
    file_exists = os.path.exists(TELEMETRY_PATH)
    with open(TELEMETRY_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TELEMETRY_HEADER)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    from inference import InferenceEngine  # deferred import, avoids circular path issues

    parser = argparse.ArgumentParser(description="Edge-IDPS full pipeline")
    parser.add_argument("--interface", default=None)
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--firewall-mode", default="simulation", choices=["simulation", "windows"])
    args = parser.parse_args()

    print("Initializing pipeline components ...")
    inference_engine = InferenceEngine()
    risk_engine = RiskEngine()
    firewall = Firewall(mode=args.firewall_mode)
    model_version = get_model_version(inference_engine)

    sniffer = PacketSniffer(interface=args.interface)
    flow_manager = FlowManager(packet_queue=sniffer.queue)

    sniffer.start()
    flow_manager.start()

    print(f"\nPipeline running for {args.duration}s ... (Ctrl+C to stop early)")
    print(f"Telemetry: {TELEMETRY_PATH}")
    print(f"{'Time':<10} {'Source IP':<16} {'Predicted':<12} {'Conf':<6} "
          f"{'Risk':<6} {'Action':<12}")
    print("-" * 70)

    start = time.time()
    try:
        while time.time() - start < args.duration:
            time.sleep(0.5)
            while not flow_manager.closed_flow_queue.empty():
                flow = flow_manager.closed_flow_queue.get()
                try:
                    prediction = inference_engine.predict(flow)
                except Exception as e:
                    print(f"  [ERROR extracting/predicting flow {flow.flow_id}]: {e}")
                    continue

                source_ip = flow.src_ip
                risk_result = risk_engine.assess(
                    source_ip=source_ip,
                    predicted_class=prediction["predicted_class"],
                    confidence=prediction["confidence"],
                )
                firewall.act(
                    action=risk_result["action"],
                    ip=source_ip,
                    predicted_class=prediction["predicted_class"],
                    risk_score=risk_result["risk_score"],
                )

                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                write_telemetry_row({
                    "timestamp": timestamp,
                    "flow_id": str(flow.flow_id),
                    "source_ip": flow.src_ip,
                    "destination_ip": flow.dst_ip,
                    "predicted_class": prediction["predicted_class"],
                    "confidence": round(prediction["confidence"], 4),
                    "risk_score": risk_result["risk_score"],
                    "action": risk_result["action"],
                    "model_version": model_version,
                })

                print(f"{time.strftime('%H:%M:%S'):<10} {source_ip:<16} "
                      f"{prediction['predicted_class']:<12} "
                      f"{prediction['confidence']:.2f}   "
                      f"{risk_result['risk_score']:<6} {risk_result['action']:<12}")
    except KeyboardInterrupt:
        print("\nStopped early by user.")

    flow_manager.stop()
    sniffer.stop()
    print("\nPipeline stopped. Telemetry saved to:", TELEMETRY_PATH)


if __name__ == "__main__":
    main()