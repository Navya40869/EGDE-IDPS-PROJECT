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


def sized_flow(n, src_ip="10.0.0.5", dst_ip="10.0.0.9"):
    return Flow(src_ip=src_ip, dst_ip=dst_ip, src_port=40000, dst_port=80, protocol="TCP",
                packets=[object()] * n)


class TestSelectFlowToExplain(unittest.TestCase):
    def test_largest_flow_wins_over_earlier_smaller_ones(self):
        background, flood = sized_flow(2), sized_flow(150)
        chosen, n = xp.select_flow_to_explain([background, sized_flow(8), flood])
        self.assertIs(chosen, flood)
        self.assertEqual(n, 3)

    def test_flows_below_the_gate_are_never_chosen(self):
        self.assertEqual(xp.select_flow_to_explain([sized_flow(1), sized_flow(0)]), (None, 0))

    def test_target_ip_matches_source_or_destination(self):
        to_victim = sized_flow(5, dst_ip="192.168.1.50")
        from_victim = sized_flow(4, src_ip="192.168.1.50")
        bigger_elsewhere = sized_flow(500)
        flows = [to_victim, from_victim, bigger_elsewhere]
        self.assertIs(xp.select_flow_to_explain(flows)[0], bigger_elsewhere)
        chosen, n = xp.select_flow_to_explain(flows, target_ip="192.168.1.50")
        self.assertIs(chosen, to_victim)
        self.assertEqual(n, 2)

    def test_target_ip_with_no_match_gives_nothing(self):
        self.assertEqual(xp.select_flow_to_explain([sized_flow(50)], "1.2.3.4"), (None, 0))


VICTIM_IP = "192.168.1.50"


class FakeSniffer:
    """
    Stands in for PacketSniffer (no Npcap/Administrator needed): plays a
    scripted packet list onto the queue the real FlowManager consumes.
    Background flows go first, as they would on a real NIC, then the attack.
    """
    script = []

    def __init__(self, interface=None):
        import queue
        self.queue = queue.Queue()
        self.captured = 0

    def start(self):
        import threading
        import time

        def play():
            for p in self.script:
                p.time = time.time()  # sniffed packets carry their capture time
                self.queue.put(p)
                self.captured += 1
        threading.Thread(target=play, daemon=True).start()

    def stop(self):
        pass

    def stats(self):
        return {"captured": self.captured}


def wire(p):
    return Ether(bytes(p))


def background_packets(big_download=0):
    """HTTPS keepalive, SSDP and DNS (each closes or idles out like real traffic),
    plus an optional large HTTPS download bigger than the attack flow."""
    from scapy.all import UDP
    pkts = [wire(Ether() / IP(src="192.168.1.10", dst="142.250.1.1") /
                 TCP(sport=50001, dport=443, flags=f)) for f in ("A", "A", "FA")]
    pkts += [wire(Ether() / IP(src="192.168.1.10", dst="239.255.255.250") /
                  UDP(sport=50003, dport=1900) / Raw(b"x" * 150)) for _ in range(3)]
    pkts += [wire(Ether() / IP(src="192.168.1.10", dst="8.8.8.8") /
                  UDP(sport=55555, dport=53) / Raw(b"x" * 30)) for _ in range(2)]
    pkts += [wire(Ether() / IP(src="142.250.1.2", dst="192.168.1.10") /
                  TCP(sport=443, dport=50010, flags="PA") / Raw(b"x" * 1400))
             for _ in range(big_download)]
    return pkts


def dos_packets(src_ip, n=150):
    # scapy_attacker.dos_flood's shape: one spoofed source, one src_port, n SYNs, no FIN
    return [wire(Ether() / IP(src=src_ip, dst=VICTIM_IP) / TCP(sport=41000, dport=80, flags="S"))
            for _ in range(n)]


class TestCaptureAndExplainSimulated(unittest.TestCase):
    """End to end through the real FlowManager + capture_and_explain, with only
    the packet source faked. Before the largest-flow fix, the background flows
    (which close first) were what got explained."""

    @classmethod
    def setUpClass(cls):
        cls.engine = xp.load_inference_engine()

    def run_capture(self, script, target_ip=None):
        from unittest import mock
        FakeSniffer.script = script
        status = []
        with mock.patch("packet_sniffer.PacketSniffer", FakeSniffer):
            e = xp.capture_and_explain(self.engine, duration=3, on_status=status.append,
                                       target_ip=target_ip)
        return e, status

    def test_dos_flood_is_explained_as_dos_repeatedly(self):
        for i, src in enumerate(["10.1.2.3", "10.44.5.6", "10.200.7.8"]):
            with self.subTest(run=i):
                e, status = self.run_capture(background_packets() + dos_packets(src))
                self.assertIsNotNone(e, status)
                self.assertEqual(e.flow_id[:2], (src, VICTIM_IP), status)
                self.assertEqual(e.packet_count, 150)
                self.assertEqual(e.predicted_class, "DoS/DDoS")

    def test_target_ip_picks_the_attack_over_a_bigger_background_flow(self):
        script = background_packets(big_download=300) + dos_packets("10.9.9.9")
        unfiltered, _ = self.run_capture(script)
        self.assertNotIn(VICTIM_IP, unfiltered.flow_id[:2])  # the download is larger
        e, status = self.run_capture(script, target_ip=VICTIM_IP)
        self.assertEqual(e.flow_id[1], VICTIM_IP, status)
        self.assertEqual(e.predicted_class, "DoS/DDoS")

    def test_unmatched_target_ip_returns_none_with_a_reason(self):
        e, status = self.run_capture(background_packets(), target_ip="203.0.113.7")
        self.assertIsNone(e)
        self.assertIn("203.0.113.7", status[-1])


class TestExplainTestRow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = xp.load_inference_engine()

    def test_chosen_class_is_explained_reliably(self):
        rng = np.random.default_rng(7)
        for cls_name in ("DoS/DDoS", "Mirai"):
            for i in range(5):
                with self.subTest(cls=cls_name, run=i):
                    e = xp.explain_test_row(self.engine, cls_name, rng=rng, n_steps=50)
                    self.assertEqual(e.source, "test_row")
                    self.assertEqual(e.true_class, cls_name)
                    self.assertEqual(e.flow_id[0], "test-row")
                    self.assertEqual(e.predicted_class, cls_name)
                    self.assertEqual(len(e.attributions), 20)

    def test_matches_dataset_replay_prediction(self):
        # Same row, same model: must agree with engine/replay_dataset.py
        from replay_dataset import ReplayModel, load_frozen_indices
        X, _, _ = xp._load_test_split()
        replay = ReplayModel()
        e = xp.explain_test_row(self.engine, "Mirai", rng=np.random.default_rng(1))
        vector = np.asarray(X[e.flow_id[1]], dtype=np.float32)[load_frozen_indices()]
        pred = replay.predict(vector.reshape(1, -1))
        self.assertEqual(e.predicted_class, pred["predicted_class"])
        self.assertAlmostEqual(e.confidence, pred["confidence"], places=5)
        total = abs(sum(r[1] for r in e.attributions))
        self.assertLess(e.convergence_delta / total, 0.02)

    def test_row_of_wrong_class_and_unknown_class_are_rejected(self):
        _, y, class_order = xp._load_test_split()
        mirai_row = int(np.flatnonzero(y == class_order.index("Mirai"))[0])
        with self.assertRaises(ValueError):
            xp.explain_test_row(self.engine, "Benign", row_index=mirai_row)
        with self.assertRaises(ValueError):
            xp.explain_test_row(self.engine, "NotAClass")


if __name__ == "__main__":
    unittest.main()
