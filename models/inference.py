"""
models/inference.py — Phase 8: Inference Pipeline

Completed Flow -> Feature Vector -> StandardScaler -> CNN -> Prediction -> Confidence

Loads: edge_idps_model_5class.pth, scaler.pkl (via feature_extractor.py),
label info from model_metadata_5class.json (class_order, feature_count),
frozen_features.json.

Checks incoming vector length against feature_count; raises rather than
silently continuing on a mismatch (per blueprint Phase 8 requirement).

NOTE on model lineage: this is the 5-class model (Benign, DoS/DDoS merged,
Mirai, Reconnaissance, Spoofing) — NOT the original 8-class taxonomy in
utils/config.py. Class names for this model come from
model_metadata_5class.json, not from utils.config.CLASS_ORDER, since that
constant reflects the original (superseded) 8-class design.

Usage (standalone live test):
    python models/inference.py --interface "Wi-Fi" --duration 30
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

# Make the project root importable regardless of which directory this is run from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "engine"))

from cnn_model import EdgeIDPSCNN
from feature_extractor import FeatureExtractor

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(MODELS_DIR)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

MODEL_PATH = os.path.join(MODELS_DIR, "edge_idps_model_5class.pth")
MODEL_METADATA_PATH = os.path.join(PROCESSED_DIR, "model_metadata_5class.json")


class InferenceEngine:
    def __init__(self, model_path: str = MODEL_PATH,
                 model_metadata_path: str = MODEL_METADATA_PATH):
        with open(model_metadata_path) as f:
            self.metadata = json.load(f)

        self.class_order = self.metadata["class_order"]
        self.num_classes = self.metadata["num_classes"]
        self.feature_count = self.metadata["feature_count"]

        self.feature_extractor = FeatureExtractor()

        # Defensive cross-check: the extractor's frozen feature count must
        # match what this model was actually trained on.
        if self.feature_extractor.frozen_feature_count != self.feature_count:
            raise ValueError(
                f"Feature count mismatch: model expects {self.feature_count}, "
                f"but feature_extractor.py is producing "
                f"{self.feature_extractor.frozen_feature_count}. "
                f"frozen_features.json may be out of sync with this model."
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EdgeIDPSCNN(
            num_features=self.feature_count, num_classes=self.num_classes
        ).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()  # disables Dropout, switches BatchNorm to running stats

        print(f"[InferenceEngine] Loaded model: {self.num_classes} classes "
              f"{self.class_order}, {self.feature_count} features")
        print(f"[InferenceEngine] Test accuracy at training time: "
              f"{self.metadata.get('test_accuracy', 'unknown')}, "
              f"macro F1: {self.metadata.get('macro_f1', 'unknown')}")

    def predict(self, flow) -> dict:
        """
        Runs the full inference step for one closed flow.
        Returns: {"predicted_class": str, "confidence": float,
                  "class_probabilities": {class_name: prob, ...}, "flow_id": tuple}
        """
        vector = self.feature_extractor.extract(flow)  # (1, feature_count), scaled

        if vector.shape[1] != self.feature_count:
            raise ValueError(
                f"Feature vector shape mismatch: got {vector.shape[1]}, "
                f"expected {self.feature_count}. Refusing to run inference "
                f"on a wrong-shaped vector — check frozen_features.json "
                f"and this model's schema are in sync."
            )

        tensor = torch.tensor(vector, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]

        predicted_idx = int(probabilities.argmax())
        predicted_class = self.class_order[predicted_idx]
        confidence = float(probabilities[predicted_idx])

        return {
            "flow_id": flow.flow_id,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "class_probabilities": {
                self.class_order[i]: float(probabilities[i])
                for i in range(self.num_classes)
            },
            "packet_count": len(flow.packets),
            "closed_reason": flow.closed_reason,
        }


def main():
    # Deferred imports — only needed for the standalone live-capture test
    from packet_sniffer import PacketSniffer
    from flow_manager import FlowManager

    parser = argparse.ArgumentParser(description="Edge-IDPS inference (standalone live test)")
    parser.add_argument("--interface", default=None)
    parser.add_argument("--duration", type=int, default=30)
    args = parser.parse_args()

    engine = InferenceEngine()

    sniffer = PacketSniffer(interface=args.interface)
    flow_manager = FlowManager(packet_queue=sniffer.queue)

    sniffer.start()
    flow_manager.start()

    print(f"\nCapturing and classifying for {args.duration}s ... (Ctrl+C to stop early)")
    print(f"{'Flow ID':<55} {'Prediction':<15} {'Confidence':<12} {'Packets':<8}")
    print("-" * 95)

    start = time.time()
    try:
        while time.time() - start < args.duration:
            time.sleep(0.5)
            while not flow_manager.closed_flow_queue.empty():
                flow = flow_manager.closed_flow_queue.get()
                try:
                    result = engine.predict(flow)
                    flow_str = f"{result['flow_id']}"
                    print(f"{flow_str:<55} {result['predicted_class']:<15} "
                          f"{result['confidence']:.4f}      {result['packet_count']:<8}")
                except Exception as e:
                    print(f"  [ERROR classifying flow {flow.flow_id}]: {e}")
    except KeyboardInterrupt:
        print("\nStopped early by user.")

    flow_manager.stop()
    sniffer.stop()
    print("\nDone.")


if __name__ == "__main__":
    main()