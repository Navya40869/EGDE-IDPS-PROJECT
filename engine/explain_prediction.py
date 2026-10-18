"""
engine/explain_prediction.py — Explainable AI: WHY was this flow classified this way?

Runs Integrated Gradients (Captum) on the 5-class CNN for ONE fresh, live
prediction (or one held-out test row, see explain_test_row) and reports how
much each of the 20 frozen features pushed the model toward (or away from)
the class it predicted.

SCOPE: this explains a flow captured and classified in the same run. It can
NOT explain historical rows in logs/live_stream_output.csv, because that file
only stores the prediction output, not the 20-feature vector it was made from.

Does not modify or re-implement any pipeline step: feature extraction is
FeatureExtractor.extract() and the forward pass is InferenceEngine.model,
exactly as InferenceEngine.predict() uses them.

Usage (standalone live test — needs Administrator + Npcap, like inference.py):
    python engine/explain_prediction.py --interface "Wi-Fi" --duration 20 [--target-ip 192.168.1.50]
Held-out test row (no capture, no Administrator):
    python engine/explain_prediction.py --test-row Mirai
"""

import argparse
import contextlib
import functools
import json
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
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
X_TEST_PATH = os.path.join(PROCESSED_DIR, "X_test_5class.npy")
Y_TEST_PATH = os.path.join(PROCESSED_DIR, "y_test_5class.npy")
CLASS_ORDER_5_PATH = os.path.join(PROCESSED_DIR, "class_order_5class.json")

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
    # "live" = a flow captured from the network; "test_row" = a held-out
    # CICIoT2023 test row (explain_test_row), whose true label is known.
    source: str = "live"
    true_class: str | None = None


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
    if len(flow.packets) < MIN_PACKETS_FOR_CLASSIFICATION:
        # The live pipeline never classifies these (INSUFFICIENT_EVIDENCE), so
        # explaining one would describe a prediction that never happens.
        raise ValueError(
            f"Flow has {len(flow.packets)} packet(s); the pipeline requires "
            f">= {MIN_PACKETS_FOR_CLASSIFICATION} to classify (INSUFFICIENT_EVIDENCE gate)."
        )

    vector = engine.feature_extractor.extract(flow)  # (1, 20), already StandardScaler-scaled
    return _explain_vector(engine, vector, flow.flow_id, len(flow.packets), n_steps)


def _explain_vector(engine, vector, flow_id, packet_count, n_steps=IG_STEPS, **extra) -> Explanation:
    """Classifies one scaled (1, 20) vector and explains it with Integrated Gradients."""
    from captum.attr import IntegratedGradients

    feature_names = engine.feature_extractor.frozen_feature_order
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
        flow_id=flow_id,
        predicted_class=engine.class_order[predicted_idx],
        confidence=float(probabilities[predicted_idx]),
        attributions=rows,
        class_probabilities={
            engine.class_order[i]: float(probabilities[i]) for i in range(len(probabilities))
        },
        convergence_delta=float(delta.abs().max()),
        packet_count=packet_count,
        **extra,
    )


@functools.lru_cache(maxsize=1)
def _load_test_split():
    # Held-out TEST split only (same files engine/replay_dataset.py replays),
    # never train rows. mmap: a few rows are read, not the ~480MB array.
    with open(CLASS_ORDER_5_PATH) as f:
        class_order = json.load(f)["class_order"]
    return np.load(X_TEST_PATH, mmap_mode="r"), np.load(Y_TEST_PATH), class_order


def explain_test_row(engine, class_name: str, row_index: int | None = None,
                     rng=None, n_steps: int = IG_STEPS) -> Explanation:
    """
    Explains one held-out CICIoT2023 test row of `class_name` through the same
    model and Integrated Gradients path as a live flow. For classes the lab
    can't generate live (e.g. Mirai). A random row of that class unless
    row_index is given.

    Test rows are already StandardScaled on all 37 columns, so FeatureExtractor
    is skipped and only its frozen-20 slice is applied — the same thing
    FeatureExtractor.extract() does after scaling.
    """
    X_test, y_test, class_order = _load_test_split()
    if class_order != engine.class_order:
        raise ValueError(f"Model class order {engine.class_order} != "
                         f"class_order_5class.json {class_order}")
    if class_name not in class_order:
        raise ValueError(f"Unknown class {class_name!r}; choose from {class_order}")

    class_idx = class_order.index(class_name)
    if row_index is None:
        rng = rng if rng is not None else np.random.default_rng()
        row_index = int(rng.choice(np.flatnonzero(y_test == class_idx)))
    elif int(y_test[row_index]) != class_idx:
        raise ValueError(f"Test row {row_index} is {class_order[int(y_test[row_index])]}, "
                         f"not {class_name}")

    vector = np.asarray(X_test[row_index], dtype=np.float32)[
        engine.feature_extractor.frozen_indices].reshape(1, -1)
    return _explain_vector(engine, vector, ("test-row", row_index), packet_count=0,
                           n_steps=n_steps, source="test_row", true_class=class_name)


def direction_label(attribution: float, predicted_class: str) -> str:
    if abs(attribution) < NEGLIGIBLE_ATTRIBUTION:
        return "negligible"
    if attribution > 0:
        return f"pushed TOWARD {predicted_class}"
    return f"pushed AWAY from {predicted_class}"


def print_explanation(explanation: Explanation):
    e = explanation
    if e.source == "test_row":
        print(f"\nTest row:    {e.flow_id[1]}  (held-out CICIoT2023, true class {e.true_class})")
    else:
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


def select_flow_to_explain(flows, target_ip=None):
    """
    Picks the flow to explain from everything captured in the window: the one
    with the MOST packets among flows that clear the INSUFFICIENT_EVIDENCE
    gate (and, if target_ip is set, have it as source or destination).

    Largest rather than first-to-close: a flood flow never sends FIN, so it
    only closes on the 5s idle timeout, and incidental background flows
    (HTTPS keepalives, SSDP, DNS) used to close first and get explained in
    its place. Returns (flow or None, number of candidate flows).
    """
    candidates = [f for f in flows if len(f.packets) >= MIN_PACKETS_FOR_CLASSIFICATION
                  and (target_ip is None or target_ip in (f.src_ip, f.dst_ip))]
    if not candidates:
        return None, 0
    return max(candidates, key=lambda f: len(f.packets)), len(candidates)


def capture_and_explain(engine, interface=None, duration: float = 20,
                        on_status=print, target_ip: str | None = None) -> Explanation | None:
    """
    Captures live traffic for the full `duration`, then explains the flow with
    the most packets (see select_flow_to_explain). target_ip, if given, limits
    the choice to flows to or from that address, e.g. the lab victim. Returns
    None if no eligible flow was seen. Read-only on the network: sniffs, never
    sends.

    Flows still active at the deadline are flushed by FlowManager.stop()
    (closed_reason="SHUTDOWN") and are candidates too, so an ongoing flood is
    included even though it hasn't closed yet.
    """
    from packet_sniffer import PacketSniffer
    from flow_manager import FlowManager

    target_ip = (target_ip or "").strip() or None
    sniffer = PacketSniffer(interface=interface)
    flow_manager = FlowManager(packet_queue=sniffer.queue)
    sniffer.start()
    flow_manager.start()

    flows = []

    def drain():
        while not flow_manager.closed_flow_queue.empty():
            flows.append(flow_manager.closed_flow_queue.get())

    try:
        deadline = time.time() + duration
        while time.time() < deadline:
            time.sleep(min(0.5, max(deadline - time.time(), 0)))
            drain()
    finally:
        # Always release the capture threads, even if interrupted mid-capture.
        # stop() flushes still-active flows into the closed queue.
        flow_manager.stop()
        sniffer.stop()
    drain()

    chosen, n_candidates = select_flow_to_explain(flows, target_ip)
    scope = f" to/from {target_ip}" if target_ip else ""
    if chosen is None:
        stats = sniffer.stats()
        on_status(f"No classifiable flow{scope} within {duration}s ({len(flows)} flow(s) seen, "
                  f"packets captured: {stats['captured']}). If 0, check you're running as "
                  f"Administrator and the interface name is right."
                  + (" If flows were seen, check the target IP." if target_ip and flows else ""))
        return None

    on_status(f"Captured {len(flows)} flow(s); {n_candidates} had >= "
              f"{MIN_PACKETS_FOR_CLASSIFICATION} packets{scope}.")
    on_status(f"Explaining the largest: {chosen.flow_id} ({len(chosen.packets)} packets, "
              f"closed_reason={chosen.closed_reason}) ...")
    return explain_flow(engine, chosen)


def main():
    parser = argparse.ArgumentParser(
        description="Edge-IDPS: explain one CNN prediction (live flow or held-out test row)")
    parser.add_argument("--interface", default=None)
    parser.add_argument("--duration", type=int, default=20,
                        help="Seconds to capture before explaining the largest flow")
    parser.add_argument("--target-ip", default=None,
                        help="Only consider flows to/from this IP (e.g. the lab victim)")
    parser.add_argument("--test-row", metavar="CLASS", default=None,
                        help='Explain a random held-out test row of CLASS (e.g. "Mirai") '
                             "instead of capturing live traffic")
    parser.add_argument("--output", default=DEFAULT_CHART_PATH, help="Chart PNG path")
    args = parser.parse_args()

    engine = load_inference_engine()
    if args.test_row:
        explanation = explain_test_row(engine, args.test_row)
    else:
        scope = f" to/from {args.target_ip}" if args.target_ip else ""
        print(f"\nCapturing {args.duration}s, then explaining the largest flow{scope} with "
              f">= {MIN_PACKETS_FOR_CLASSIFICATION} packets ... (Ctrl+C to stop early)")
        try:
            explanation = capture_and_explain(engine, args.interface, args.duration,
                                              target_ip=args.target_ip)
        except KeyboardInterrupt:
            print("\nStopped early by user.")
            return
    if explanation is None:
        sys.exit(1)

    print_explanation(explanation)
    print(f"\nChart saved to: {save_chart(explanation, args.output)}")


if __name__ == "__main__":
    main()
