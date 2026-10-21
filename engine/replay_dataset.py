"""
engine/replay_dataset.py — Dataset-replay demo mode.

Simulates attacks WITHOUT any live traffic or ESP32 by replaying held-out
TEST rows from the 5-class dataset through the same back half of the live
pipeline:

    test row (already scaled, 37 cols) -> frozen 20 cols -> CNN
        -> RiskEngine.assess() -> Firewall.act() (simulation by default)
        -> logs/live_stream_output.csv

Test-set provenance: preprocess.py does the stratified 80/20 split
(random_state=42) BEFORE scaling and saves X_test.npy/y_test.npy;
filter_classes.py and merge_dos_ddos.py carry that same split forward to
X_test_5class.npy/y_test_5class.npy, which the 5-class model was evaluated
on. Only those *_test_5class files are ever loaded here — never train rows.

FeatureExtractor is skipped because the rows are already StandardScaled on
all 37 columns; we only slice out the frozen 20 in frozen_features.json
order (same thing FeatureExtractor.extract() does after scaling).

Every telemetry row is marked closed_reason="DATASET_REPLAY" and
flow_id="replay-<test-row-index>" so replayed traffic is never mistaken for
live captures.

Usage:
    python engine/replay_dataset.py --classes "DoS/DDoS" "Mirai" --n 20 --rate 5
    python engine/replay_dataset.py --firewall-mode windows   # LAB ONLY, needs Administrator

Remove the Windows Firewall rules this project created:
    python engine/firewall.py --cleanup [--dry-run]
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))

from cnn_model import EdgeIDPSCNN
from risk_engine import RiskEngine
from firewall import Firewall
from stream_pipeline import TELEMETRY_PATH, write_telemetry_row

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "edge_idps_model_5class.pth")
MODEL_METADATA_PATH = os.path.join(PROCESSED_DIR, "model_metadata_5class.json")
PREPROCESSING_METADATA_PATH = os.path.join(PROCESSED_DIR, "preprocessing_metadata.json")
FROZEN_FEATURES_PATH = os.path.join(PROCESSED_DIR, "frozen_features.json")
X_TEST_PATH = os.path.join(PROCESSED_DIR, "X_test_5class.npy")
Y_TEST_PATH = os.path.join(PROCESSED_DIR, "y_test_5class.npy")
CLASS_ORDER_5_PATH = os.path.join(PROCESSED_DIR, "class_order_5class.json")
RISK_RULES_PATH = os.path.join(PROJECT_ROOT, "engine", "risk_rules.json")

DEFAULT_ATTACKER_IP = "10.10.10.50"
DEFAULT_VICTIM_IP = "10.10.10.1"
CLOSED_REASON = "DATASET_REPLAY"


class ReplayModel:
    """Same model load + softmax/argmax as InferenceEngine.predict(), minus FeatureExtractor."""

    def __init__(self):
        with open(MODEL_METADATA_PATH) as f:
            self.metadata = json.load(f)
        self.class_order = self.metadata["class_order"]
        self.num_classes = self.metadata["num_classes"]
        self.feature_count = self.metadata["feature_count"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EdgeIDPSCNN(
            num_features=self.feature_count, num_classes=self.num_classes
        ).to(self.device)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.eval()

    def predict(self, vector: np.ndarray) -> dict:
        if vector.shape != (1, self.feature_count):
            raise ValueError(f"Feature vector shape mismatch: got {vector.shape}, "
                             f"expected (1, {self.feature_count})")
        tensor = torch.tensor(vector, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            probabilities = F.softmax(self.model(tensor), dim=1).cpu().numpy()[0]
        predicted_idx = int(probabilities.argmax())
        return {
            "predicted_class": self.class_order[predicted_idx],
            "confidence": float(probabilities[predicted_idx]),
        }


def load_frozen_indices() -> list:
    with open(FROZEN_FEATURES_PATH) as f:
        frozen = json.load(f)["feature_order"]
    with open(PREPROCESSING_METADATA_PATH) as f:
        all_features = json.load(f)["feature_names"]
    return [all_features.index(name) for name in frozen]


def select_rows(y_test: np.ndarray, class_order: list, classes: list, n: int, seed: int):
    """Returns shuffled [(test_row_index, true_class), ...], n per requested class."""
    rng = np.random.default_rng(seed)
    selected = []
    for cls in classes:
        cls_idx = class_order.index(cls)
        candidates = np.flatnonzero(y_test == cls_idx)
        if len(candidates) < n:
            raise ValueError(f"Only {len(candidates)} test rows for {cls}, asked for {n}")
        picks = rng.choice(candidates, size=n, replace=False)
        selected.extend((int(i), cls) for i in picks)
    rng.shuffle(selected)
    return selected


def main():
    parser = argparse.ArgumentParser(description="Edge-IDPS dataset-replay demo (no live traffic)")
    parser.add_argument("--classes", nargs="+", default=["DoS/DDoS", "Mirai"],
                        help='Classes to replay, e.g. --classes "DoS/DDoS" "Mirai"')
    parser.add_argument("--n", type=int, default=20, help="Rows per class")
    parser.add_argument("--rate", type=float, default=5.0, help="Flows per second")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attacker-ip", default=DEFAULT_ATTACKER_IP)
    parser.add_argument("--victim-ip", default=DEFAULT_VICTIM_IP)
    parser.add_argument("--firewall-mode", default="simulation", choices=["simulation", "windows"],
                        help="windows = real netsh block rules (LAB ONLY, needs Administrator)")
    args = parser.parse_args()

    with open(CLASS_ORDER_5_PATH) as f:
        class_order = json.load(f)["class_order"]
    unknown = [c for c in args.classes if c not in class_order]
    if unknown:
        parser.error(f"Unknown class(es) {unknown}; choose from {class_order}")
    if args.rate <= 0:
        parser.error("--rate must be > 0")

    with open(RISK_RULES_PATH) as f:
        allowlist = json.load(f).get("ip_allowlist", [])
    if args.attacker_ip in allowlist:
        parser.error(f"--attacker-ip {args.attacker_ip} is in ip_allowlist; "
                     f"the risk engine would score it 0. Pick another IP.")

    print("Initializing replay components ...")
    model = ReplayModel()
    if model.class_order != class_order:
        raise ValueError(f"Model class order {model.class_order} != "
                         f"class_order_5class.json {class_order}")
    frozen_indices = load_frozen_indices()
    risk_engine = RiskEngine()
    firewall = Firewall(mode=args.firewall_mode)
    model_version = f"5class_{model.metadata.get('training_date', 'unknown')}"

    # Test split only. mmap so we don't pull ~480MB into RAM for a few rows.
    X_test = np.load(X_TEST_PATH, mmap_mode="r")
    y_test = np.load(Y_TEST_PATH)
    rows = select_rows(y_test, class_order, args.classes, args.n, args.seed)

    print(f"\nReplaying {len(rows)} held-out test rows "
          f"({args.n} x {args.classes}) at {args.rate} flows/s")
    print(f"Attacker {args.attacker_ip} -> victim {args.victim_ip} | Firewall: {args.firewall_mode}")
    print(f"Telemetry: {TELEMETRY_PATH}")
    print(f"{'Time':<10} {'Row':<14} {'True':<10} {'Predicted':<12} {'Conf':<6} "
          f"{'Risk':<7} {'Action':<12}")
    print("-" * 78)

    stats = defaultdict(lambda: {"rows": 0, "correct": 0, "conf_sum": 0.0,
                                 "last_risk": None, "last_action": None})
    last_risk = None
    interval = 1.0 / args.rate
    next_tick = time.monotonic()

    try:
        for row_idx, true_class in rows:
            vector = np.asarray(X_test[row_idx], dtype=np.float32)[frozen_indices].reshape(1, -1)
            prediction = model.predict(vector)

            risk_result = risk_engine.assess(
                source_ip=args.attacker_ip,
                predicted_class=prediction["predicted_class"],
                confidence=prediction["confidence"],
            )
            firewall.act(
                action=risk_result["action"],
                ip=args.attacker_ip,
                predicted_class=prediction["predicted_class"],
                risk_score=risk_result["risk_score"],
            )

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            write_telemetry_row({
                "timestamp": timestamp,
                "flow_id": f"replay-{row_idx}",
                "source_ip": args.attacker_ip,
                "destination_ip": args.victim_ip,
                # Per-flow packet/port metadata doesn't exist for dataset rows
                "protocol": "",
                "src_port": "",
                "dst_port": "",
                "packet_count": "",
                "closed_reason": CLOSED_REASON,
                "flow_duration": "",
                "predicted_class": prediction["predicted_class"],
                "confidence": round(prediction["confidence"], 4),
                "risk_score": risk_result["risk_score"],
                "action": risk_result["action"],
                "model_version": model_version,
            })

            s = stats[true_class]
            s["rows"] += 1
            s["correct"] += int(prediction["predicted_class"] == true_class)
            s["conf_sum"] += prediction["confidence"]
            s["last_risk"] = risk_result["risk_score"]
            s["last_action"] = risk_result["action"]
            last_risk = risk_result

            print(f"{time.strftime('%H:%M:%S'):<10} {'replay-' + str(row_idx):<14} "
                  f"{true_class:<10} {prediction['predicted_class']:<12} "
                  f"{prediction['confidence']:.2f}   {risk_result['risk_score']:<7} "
                  f"{risk_result['action']:<12}")

            next_tick += interval
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        print("\nStopped early by user.")

    print("\n=== Replay summary ===")
    print(f"{'True class':<12} {'Rows':>5} {'Correct':>8} {'Acc':>7} {'Mean conf':>10} "
          f"{'Last risk':>10}  {'Last action':<12}")
    total_rows = total_correct = 0
    for cls in args.classes:
        s = stats[cls]
        if not s["rows"]:
            continue
        total_rows += s["rows"]
        total_correct += s["correct"]
        print(f"{cls:<12} {s['rows']:>5} {s['correct']:>8} {s['correct'] / s['rows']:>7.1%} "
              f"{s['conf_sum'] / s['rows']:>10.4f} {s['last_risk']:>10}  {s['last_action']:<12}")
    if total_rows:
        print(f"{'TOTAL':<12} {total_rows:>5} {total_correct:>8} "
              f"{total_correct / total_rows:>7.1%}")
        print(f"\nFinal state for {args.attacker_ip}: risk_score={last_risk['risk_score']}, "
              f"action={last_risk['action']}, "
              f"frequency_in_window={last_risk['frequency_in_window']}, "
              f"on blocklist={firewall.is_blocked(args.attacker_ip)}")


if __name__ == "__main__":
    main()
