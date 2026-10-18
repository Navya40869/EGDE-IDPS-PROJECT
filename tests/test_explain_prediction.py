"""
tests/test_explain_prediction.py — explainability module, no live capture.

Builds closed flows from synthetic Scapy packets (nothing is sent or sniffed)
and runs them through the real InferenceEngine + Integrated Gradients on CPU.

Run from the project root:
    python -m unittest tests.test_explain_prediction -v
"""

import json
import os
import re
import sys
import tempfile
import unittest

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "engine"))

import explain_prediction as xp  # noqa: E402  (sets up engine/ and models/ import paths)
from flow_manager import Flow  # noqa: E402
from scapy.all import Ether, IP, TCP, ICMP, Raw  # noqa: E402

CLASS_ORDER_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "class_order_5class.json")
T0 = 1_700_000_000.0


def make_flow(protocol, packets, dst_port, src_port=40000):
    # Round-trip through bytes so packets look like sniffed ones: a freshly
    # constructed Scapy packet has unset auto-fields (IP.ihl is None) until built.
    packets = [Ether(bytes(p)) for p in packets]
    for i, p in enumerate(packets):
        p.time = T0 + i * 0.01
    return Flow(src_ip="10.0.0.5", dst_ip="10.0.0.9", src_port=src_port, dst_port=dst_port,
                protocol=protocol, start_time=T0, last_packet_time=packets[-1].time,
                packets=packets, closed_reason="FIN")


def tcp_https_flow():
    flags = ["S", "SA", "A", "PA", "PA", "A", "FA"]
    pkts = [Ether() / IP(src="10.0.0.5", dst="10.0.0.9", ttl=64) /
            TCP(sport=40000, dport=443, flags=f) / Raw(b"x" * (200 if "P" in f else 0))
            for f in flags]
    return make_flow("TCP", pkts, dst_port=443)


def syn_burst_flow():
    pkts = [Ether() / IP(src="10.0.0.5", dst="10.0.0.9", ttl=64) /
            TCP(sport=40001, dport=80, flags="S") for _ in range(40)]
    return make_flow("TCP", pkts, dst_port=80, src_port=40001)


def icmp_flow():
    pkts = [Ether() / IP(src="10.0.0.5", dst="10.0.0.9", ttl=128) / ICMP() / Raw(b"p" * 32)
            for _ in range(5)]
    return make_flow("ICMP", pkts, dst_port=0, src_port=0)


class TestExplainPrediction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = xp.load_inference_engine()
        with open(CLASS_ORDER_PATH) as f:
            cls.valid_classes = json.load(f)["class_order"]
        cls.flows = {"tcp_https": tcp_https_flow(), "syn_burst": syn_burst_flow(),
                     "icmp": icmp_flow()}
        cls.explanations = {k: xp.explain_flow(cls.engine, f) for k, f in cls.flows.items()}

    def test_exactly_20_attributions(self):
        for name, e in self.explanations.items():
            with self.subTest(flow=name):
                self.assertEqual(len(e.attributions), 20)

    def test_feature_names_match_frozen_order(self):
        frozen = self.engine.feature_extractor.frozen_feature_order
        self.assertEqual(len(frozen), 20)
        for name, e in self.explanations.items():
            with self.subTest(flow=name):
                names = [r[0] for r in e.attributions]
                self.assertEqual(len(set(names)), 20, "duplicate feature names")
                self.assertEqual(sorted(names), sorted(frozen))

    def test_predicted_class_is_valid(self):
        self.assertEqual(len(self.valid_classes), 5)
        for name, e in self.explanations.items():
            with self.subTest(flow=name):
                self.assertIn(e.predicted_class, self.valid_classes)
                self.assertGreater(e.confidence, 0.0)
                self.assertLessEqual(e.confidence, 1.0)

    def test_sorted_by_absolute_attribution(self):
        for name, e in self.explanations.items():
            with self.subTest(flow=name):
                mags = [abs(r[1]) for r in e.attributions]
                self.assertEqual(mags, sorted(mags, reverse=True))

    def test_matches_production_predict_path(self):
        # The explanation must describe the SAME prediction the pipeline makes
        for name, flow in self.flows.items():
            with self.subTest(flow=name):
                prod = self.engine.predict(flow)
                e = self.explanations[name]
                self.assertEqual(e.predicted_class, prod["predicted_class"])
                self.assertAlmostEqual(e.confidence, prod["confidence"], places=5)

    def test_scaled_values_are_the_extractor_output(self):
        for name, flow in self.flows.items():
            with self.subTest(flow=name):
                vector = self.engine.feature_extractor.extract(flow)[0]
                by_name = {r[0]: r[2] for r in self.explanations[name].attributions}
                expected = dict(zip(self.engine.feature_extractor.frozen_feature_order, vector))
                for feat, val in expected.items():
                    self.assertAlmostEqual(by_name[feat], float(val), places=5)

    def test_ig_completeness(self):
        # IG axiom: attributions sum to logit(input) - logit(zero baseline).
        # Checked relative to that sum, since logits can be large (~20).
        for name, e in self.explanations.items():
            with self.subTest(flow=name):
                total = abs(sum(r[1] for r in e.attributions))
                self.assertLess(e.convergence_delta / total, 0.02)

    def test_gate_rejects_single_packet_flow(self):
        single = make_flow("TCP", [Ether() / IP() / TCP(dport=80)], dst_port=80)
        with self.assertRaises(ValueError):
            xp.explain_flow(self.engine, single)

    def test_gate_constant_matches_stream_pipeline(self):
        with open(os.path.join(PROJECT_ROOT, "engine", "stream_pipeline.py")) as f:
            match = re.search(r"MIN_PACKETS_FOR_CLASSIFICATION\s*=\s*(\d+)", f.read())
        self.assertIsNotNone(match, "gate constant not found in stream_pipeline.py")
        self.assertEqual(int(match.group(1)), xp.MIN_PACKETS_FOR_CLASSIFICATION)

    def test_chart_saves_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = xp.save_chart(self.explanations["tcp_https"], os.path.join(tmp, "x.png"))
            with open(path, "rb") as f:
                self.assertEqual(f.read(8), b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
