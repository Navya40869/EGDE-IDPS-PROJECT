"""
engine/explain_prediction.py — Explainable AI: WHY was this flow classified this way?

Runs Integrated Gradients (Captum) on the 5-class CNN for ONE fresh, live
prediction and reports how much each of the 20 frozen features pushed the
model toward (or away from) the class it predicted.

SCOPE: this explains a flow captured and classified in the same run. It can
NOT explain historical rows in logs/live_stream_output.csv, because that file
only stores the prediction output, not the 20-feature vector it was made from.

Does not modify or re-implement any pipeline step: feature extraction is
FeatureExtractor.extract() and the forward pass is InferenceEngine.model,
exactly as InferenceEngine.predict() uses them.

Usage (standalone live test — needs Administrator + Npcap, like inference.py):
    python engine/explain_prediction.py --interface "Wi-Fi" --duration 20
"""

import argparse
import contextlib
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "engine"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "models"))

LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
DEFAULT_CHART_PATH = os.path.join(LOGS_DIR, "last_explanation.png")

# Same name and value as the INSUFFICIENT_EVIDENCE gate in
# engine/stream_pipeline.py. That constant is local to stream_pipeline.main()
# so it can't be imported; tests/test_explain_prediction.py parses
# stream_pipeline.py and fails if the two ever drift apart.
MIN_PACKETS_FOR_CLASSIFICATION = 2

IG_STEPS = 200  # path-integral steps; ~45ms on CPU, keeps completeness error < 1% of the logit

# Chart colors — green/red as specified, stepped apart in lightness so the
# pair stays distinguishable under red-green colorblindness (validated:
# deutan dE 16.1). Sign is also encoded by bar side, hatching and labels.
COLOR_TOWARD = "#3fae5a"
COLOR_AWAY = "#a8323a"

# Attributions smaller than this (in logit units) are labelled negligible
# rather than given a direction, so near-zero noise isn't over-read.
NEGLIGIBLE_ATTRIBUTION = 1e-3


@dataclass
class Explanation:
    flow_id: tuple
    predicted_class: str
    confidence: float
    # (feature_name, attribution, scaled_feature_value), largest |attribution| first
    attributions: list
    class_probabilities: dict = field(default_factory=dict)
    # Integrated Gradients completeness error: sum(attributions) should equal
    # logit(input) - logit(baseline) for the target class. Near 0 = trustworthy.
    convergence_delta: float = 0.0
    packet_count: int = 0


@contextlib.contextmanager
def _project_root_cwd():
    # FeatureExtractor's default paths are relative ("data/processed/..."),
    # so InferenceEngine must be constructed with the project root as CWD.
    previous = os.getcwd()
    os.chdir(PROJECT_ROOT)
    try:
        yield
    finally:
        os.chdir(previous)


def load_inference_engine():
    """Constructs InferenceEngine regardless of the caller's working directory."""
    from inference import InferenceEngine
    with _project_root_cwd():
        return InferenceEngine()


def explain_flow(engine, flow, n_steps: int = IG_STEPS) -> Explanation:
    """
    Classifies one closed flow and explains the prediction with Integrated
    Gradients, targeting the predicted class.
    """
    from captum.attr import IntegratedGradients

    if len(flow.packets) < MIN_PACKETS_FOR_CLASSIFICATION:
        # The live pipeline never classifies these (INSUFFICIENT_EVIDENCE), so
        # explaining one would describe a prediction that never happens.
        raise ValueError(
            f"Flow has {len(flow.packets)} packet(s); the pipeline requires "
            f">= {MIN_PACKETS_FOR_CLASSIFICATION} to classify (INSUFFICIENT_EVIDENCE gate)."
        )

    feature_names = engine.feature_extractor.frozen_feature_order
    vector = engine.feature_extractor.extract(flow)  # (1, 20), already StandardScaler-scaled
    if vector.shape[1] != engine.feature_count:
        raise ValueError(
            f"Feature vector shape mismatch: got {vector.shape[1]}, expected {engine.feature_count}."
        )

    model = engine.model
    model.eval()  # same mode as InferenceEngine.predict(): no Dropout, BatchNorm running stats
    inputs = torch.tensor(vector, dtype=torch.float32).to(engine.device)

    with torch.no_grad():
        probabilities = F.softmax(model(inputs), dim=1).cpu().numpy()[0]
    predicted_idx = int(probabilities.argmax())

    # Baseline = all zeros. The model's inputs are StandardScaler outputs, and
    # StandardScaler centres every feature on its training-set mean, so 0 in
    # scaled space means "exactly average for that feature". The all-zero
    # vector is therefore the "average training flow", and each attribution
    # answers: how much did THIS feature's deviation from average move the
    # predicted class's score? (A raw-space zero — 0 bytes, TTL 0 — would be
    # an impossible flow and a misleading reference point.)
    baseline = torch.zeros_like(inputs)

    # Attributions are computed on the logit (pre-softmax) of the predicted
    # class, the standard IG target: softmax couples the classes and saturates
    # near 1.0, which would squash attributions for confident predictions.
    ig = IntegratedGradients(model)
    attributions, delta = ig.attribute(
        inputs, baselines=baseline, target=predicted_idx,
        n_steps=n_steps, return_convergence_delta=True,
    )
    attributions = attributions.detach().cpu().numpy()[0]

    rows = [
        (name, float(attr), float(value))
        for name, attr, value in zip(feature_names, attributions, vector[0])
    ]
    rows.sort(key=lambda r: abs(r[1]), reverse=True)

    return Explanation(
        flow_id=flow.flow_id,
        predicted_class=engine.class_order[predicted_idx],
        confidence=float(probabilities[predicted_idx]),
        attributions=rows,
        class_probabilities={
            engine.class_order[i]: float(probabilities[i]) for i in range(len(probabilities))
        },
        convergence_delta=float(delta.abs().max()),
        packet_count=len(flow.packets),
    )



def direction_label(attribution: float, predicted_class: str) -> str:
    if abs(attribution) < NEGLIGIBLE_ATTRIBUTION:
        return "negligible"
    if attribution > 0:
        return f"pushed TOWARD {predicted_class}"
    return f"pushed AWAY from {predicted_class}"


def print_explanation(explanation: Explanation):
    e = explanation
    print(f"\nFlow:        {e.flow_id}  ({e.packet_count} packets)")
    print(f"Prediction:  {e.predicted_class}  (confidence {e.confidence:.4f})")
    others = ", ".join(f"{c}={p:.3f}" for c, p in e.class_probabilities.items())
    print(f"All classes: {others}")
    print(f"\n{'#':<3} {'Feature':<18} {'Attribution':>12} {'Scaled value':>13}  Direction")
    print("-" * 80)
    for i, (name, attr, value) in enumerate(e.attributions, start=1):
        print(f"{i:<3} {name:<18} {attr:>+12.4f} {value:>+13.3f}  "
              f"{direction_label(attr, e.predicted_class)}")
    print("-" * 80)
    print(f"Sum of attributions: {sum(r[1] for r in e.attributions):+.4f} "
          f"(= {e.predicted_class} logit vs. the average-flow baseline; "
          f"IG convergence delta {e.convergence_delta:.2e})")
    print("Scaled value = standard deviations from the training-set mean (0 = average).")


def build_chart(explanation: Explanation):
    """Returns a matplotlib Figure (Agg backend; never opens a window)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e = explanation
    rows = e.attributions[::-1]  # barh draws bottom-up; put largest |attribution| on top
    names = [r[0] for r in rows]
    values = [r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=110)
    bars = ax.barh(
        names, values, height=0.7,
        color=[COLOR_TOWARD if v >= 0 else COLOR_AWAY for v in values],
        hatch=[None if v >= 0 else "//" for v in values],
        edgecolor="white", linewidth=0.8,
    )
    ax.axvline(0, color="#555555", linewidth=1)

    span = max((abs(v) for v in values), default=1.0) or 1.0
    for bar, v in zip(bars, values):
        offset = 0.015 * span
        ax.text(v + (offset if v >= 0 else -offset), bar.get_y() + bar.get_height() / 2,
                f"{v:+.3f}" if abs(v) >= NEGLIGIBLE_ATTRIBUTION else "≈0",
                va="center", ha="left" if v >= 0 else "right",
                fontsize=8, color="#333333")
    ax.set_xlim(-span * 1.25, span * 1.25)

    ax.set_title(f"Why the CNN predicted {e.predicted_class} (confidence {e.confidence:.1%})",
                 fontsize=12, loc="left", color="#111111")
    ax.set_xlabel(f"Integrated Gradients attribution  "
                  f"(→ toward {e.predicted_class}  |  ← away from it)",
                  fontsize=9, color="#333333")
    ax.tick_params(axis="y", labelsize=9, colors="#333333", length=0)
    ax.tick_params(axis="x", labelsize=8, colors="#555555")
    ax.grid(axis="x", color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#bbbbbb")

    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(facecolor=COLOR_TOWARD, label=f"toward {e.predicted_class}"),
                 Patch(facecolor=COLOR_AWAY, hatch="//", edgecolor="white",
                       label=f"away from {e.predicted_class}")],
        loc="lower right", fontsize=8, frameon=False,
    )
    fig.text(0.01, 0.01, f"flow {e.flow_id}  ·  baseline: average training flow "
             f"(all-zero scaled vector)", fontsize=7, color="#777777")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def save_chart(explanation: Explanation, path: str = DEFAULT_CHART_PATH) -> str:
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig = build_chart(explanation)
    fig.savefig(path)
    plt.close(fig)
    return path


def capture_and_explain(engine, interface=None, duration: float = 20,
                        on_status=print) -> Explanation | None:
    """
    Captures live traffic until the first flow with enough packets to be
    classified closes, then explains it. Returns None if nothing eligible was
    seen within `duration` seconds. Read-only on the network: sniffs, never sends.

    Flows normally close on FIN/RST or a 5s idle timeout; at the deadline,
    FlowManager.stop() flushes still-active flows (closed_reason="SHUTDOWN"),
    so a short duration still yields a flow if any traffic was seen.
    """
    from packet_sniffer import PacketSniffer
    from flow_manager import FlowManager

    sniffer = PacketSniffer(interface=interface)
    flow_manager = FlowManager(packet_queue=sniffer.queue)
    sniffer.start()
    flow_manager.start()

    def first_eligible():
        skipped = 0
        while not flow_manager.closed_flow_queue.empty():
            flow = flow_manager.closed_flow_queue.get()
            if len(flow.packets) >= MIN_PACKETS_FOR_CLASSIFICATION:
                return flow, skipped
            skipped += 1
        return None, skipped

    chosen = None
    skipped_total = 0
    stopped = False
    try:
        deadline = time.time() + duration
        while time.time() < deadline and chosen is None:
            time.sleep(0.5)
            chosen, skipped = first_eligible()
            skipped_total += skipped
        if chosen is None:
            flow_manager.stop()  # flush active flows into the closed queue
            stopped = True
            chosen, skipped = first_eligible()
            skipped_total += skipped
    finally:
        # Always release the capture threads, even if interrupted mid-capture
        if not stopped:
            flow_manager.stop()
        sniffer.stop()

    if skipped_total:
        on_status(f"Skipped {skipped_total} flow(s) below the "
                  f"{MIN_PACKETS_FOR_CLASSIFICATION}-packet INSUFFICIENT_EVIDENCE gate.")
    if chosen is None:
        stats = sniffer.stats()
        on_status(f"No classifiable flow within {duration}s (packets captured: "
                  f"{stats['captured']}). If 0, check you're running as Administrator "
                  f"and the interface name is right.")
        return None

    on_status(f"Explaining flow {chosen.flow_id} ({len(chosen.packets)} packets, "
              f"closed_reason={chosen.closed_reason}) ...")
    return explain_flow(engine, chosen)


def main():
    parser = argparse.ArgumentParser(
        description="Edge-IDPS: capture one live flow and explain its CNN prediction")
    parser.add_argument("--interface", default=None)
    parser.add_argument("--duration", type=int, default=20,
                        help="Max seconds to wait for a classifiable flow")
    parser.add_argument("--output", default=DEFAULT_CHART_PATH, help="Chart PNG path")
    args = parser.parse_args()

    engine = load_inference_engine()
    print(f"\nCapturing up to {args.duration}s for the first flow with "
          f">= {MIN_PACKETS_FOR_CLASSIFICATION} packets ... (Ctrl+C to stop early)")
    try:
        explanation = capture_and_explain(engine, args.interface, args.duration)
    except KeyboardInterrupt:
        print("\nStopped early by user.")
        return
    if explanation is None:
        sys.exit(1)

    print_explanation(explanation)
    print(f"\nChart saved to: {save_chart(explanation, args.output)}")


if __name__ == "__main__":
    main()
