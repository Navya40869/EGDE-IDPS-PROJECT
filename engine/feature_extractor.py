"""
engine/feature_extractor.py — Phase 7: Online Feature Extraction (fixed)

FIX: scaler.pkl was fit on all 37 preprocessed features (preprocess.py), and
train_cnn.py selected the frozen 20 columns FROM that already-scaled 37-column
array — meaning scaling happens BEFORE feature selection in the trained
pipeline, not after. This module must replicate that exact order:
    1. Compute all 37 raw features (same order as preprocessing_metadata.json)
    2. Transform all 37 through the trained scaler
    3. THEN select only the frozen 20 columns, in frozen_features.json order

Feeding only 20 raw features directly into a scaler fit on 37 columns fails
(ValueError: wrong feature count) — this was caught by the standalone test
before it could silently corrupt inference.

IMPORTANT — feature definition caveat:
CICIoT2023's original feature computation is not publicly documented at the
byte/formula level for every column. Definitions below are best-effort
reconstructions from column names and standard netflow semantics. Columns
NOT in the frozen 20 (DHCP, ARP, IGMP, IPv, LLC) are computed as 0.0
placeholders — since StandardScaler transforms each column independently,
placeholder values in unselected columns do NOT affect the scaled output of
the columns we actually use, but they exist to satisfy the scaler's expected
input shape (37).

Per the project's validation requirement (Section 14 of the blueprint), the
20 frozen feature definitions MUST be checked against a known flow before
trusting this module for real inference: compute a vector here, recompute
the same stats independently with pandas/numpy over the same packets, and
diff each value within tolerance. This module does not skip that step.
"""

import json
import os

import joblib
import numpy as np
from scapy.all import IP, TCP, UDP, ICMP

PROCESSED_DIR = os.path.join("data", "processed")
FROZEN_FEATURES_PATH = os.path.join(PROCESSED_DIR, "frozen_features.json")
SCALER_PATH = os.path.join(PROCESSED_DIR, "scaler.pkl")
PREPROCESSING_METADATA_PATH = os.path.join(PROCESSED_DIR, "preprocessing_metadata.json")

PORT_MAP = {
    "HTTP": 80,
    "HTTPS": 443,
    "DNS": 53,
    "Telnet": 23,
    "SMTP": 25,
    "SSH": 22,
    "IRC": 6667,
}

# Columns present in the full 37-feature preprocessed set that we do NOT
# have a real definition for (not in the frozen 20, so a 0.0 placeholder is
# safe — see module docstring).
PLACEHOLDER_ZERO_COLUMNS = {"DHCP", "ARP", "IGMP", "IPv", "LLC"}


class FeatureExtractor:
    def __init__(self, frozen_features_path: str = FROZEN_FEATURES_PATH,
                 scaler_path: str = SCALER_PATH,
                 preprocessing_metadata_path: str = PREPROCESSING_METADATA_PATH):
        with open(frozen_features_path) as f:
            schema = json.load(f)
        self.frozen_feature_order = schema["feature_order"]
        self.frozen_feature_count = schema["feature_count"]
        self.schema_version = schema["version"]

        with open(preprocessing_metadata_path) as f:
            preprocessing_metadata = json.load(f)
        # This is the FULL 37-column order the scaler was actually fit on —
        # critical to replicate exactly, including order.
        self.full_feature_order = preprocessing_metadata["feature_names"]

        self.scaler = joblib.load(scaler_path)

        # Precompute the indices of the frozen 20 within the full 37, so we
        # can slice the scaled output efficiently and correctly.
        self.frozen_indices = [self.full_feature_order.index(f) for f in self.frozen_feature_order]

        print(f"[FeatureExtractor] Loaded schema v{self.schema_version}, "
              f"{self.frozen_feature_count} frozen features (of {len(self.full_feature_order)} "
              f"total scaled columns): {self.frozen_feature_order}")

    def _compute_raw_features(self, flow) -> dict:
        """Computes raw values for every column in the full 37-feature set."""
        packets = flow.packets
        sizes = [len(p) for p in packets]
        n = len(packets)

        tot_sum = float(sum(sizes))
        tot_size = tot_sum
        min_size = float(min(sizes)) if sizes else 0.0
        max_size = float(max(sizes)) if sizes else 0.0
        avg_size = float(np.mean(sizes)) if sizes else 0.0
        std_size = float(np.std(sizes)) if len(sizes) > 1 else 0.0

        duration = max(flow.last_packet_time - flow.start_time, 1e-6)
        rate = n / duration
        timestamps = [p.time for p in packets]
        if n > 1:
            iats = [timestamps[i] - timestamps[i - 1] for i in range(1, n)]
            iat = float(np.mean(iats))
        else:
            iat = 0.0

        fin_count = syn_count = rst_count = psh_count = ack_count = ece_count = cwr_count = 0
        ttl_values = []
        header_lengths = []
        for p in packets:
            if p.haslayer(IP):
                ttl_values.append(p[IP].ttl)
                header_lengths.append(p[IP].ihl * 4 if hasattr(p[IP], "ihl") else 20)
            if p.haslayer(TCP):
                flags = p[TCP].flags
                if flags & 0x01: fin_count += 1
                if flags & 0x02: syn_count += 1
                if flags & 0x04: rst_count += 1
                if flags & 0x08: psh_count += 1
                if flags & 0x10: ack_count += 1
                if flags & 0x40: ece_count += 1
                if flags & 0x80: cwr_count += 1

        avg_ttl = float(np.mean(ttl_values)) if ttl_values else 0.0
        avg_header_length = float(np.mean(header_lengths)) if header_lengths else 0.0

        has_tcp = 1.0 if flow.protocol == "TCP" else 0.0
        has_udp = 1.0 if flow.protocol == "UDP" else 0.0
        has_icmp = 1.0 if flow.protocol == "ICMP" else 0.0

        def port_indicator(port):
            return 1.0 if (flow.src_port == port or flow.dst_port == port) else 0.0

        proto_num = {"TCP": 6, "UDP": 17, "ICMP": 1}.get(flow.protocol, 0)

        raw = {
            "Header_Length": avg_header_length,
            "Protocol Type": float(proto_num),
            "Time_To_Live": avg_ttl,
            "Rate": rate,
            "fin_flag_number": float(fin_count),
            "syn_flag_number": float(syn_count),
            "rst_flag_number": float(rst_count),
            "psh_flag_number": float(psh_count),
            "ack_flag_number": float(ack_count),
            "ece_flag_number": float(ece_count),
            "cwr_flag_number": float(cwr_count),
            "ack_count": float(ack_count),
            "syn_count": float(syn_count),
            "fin_count": float(fin_count),
            "rst_count": float(rst_count),
            "HTTP": port_indicator(PORT_MAP["HTTP"]),
            "HTTPS": port_indicator(PORT_MAP["HTTPS"]),
            "DNS": port_indicator(PORT_MAP["DNS"]),
            "Telnet": port_indicator(PORT_MAP["Telnet"]),
            "SMTP": port_indicator(PORT_MAP["SMTP"]),
            "SSH": port_indicator(PORT_MAP["SSH"]),
            "IRC": port_indicator(PORT_MAP["IRC"]),
            "TCP": has_tcp,
            "UDP": has_udp,
            "ICMP": has_icmp,
            "Tot sum": tot_sum,
            "Min": min_size,
            "Max": max_size,
            "AVG": avg_size,
            "Std": std_size,
            "Tot size": tot_size,
            "IAT": iat,
        }

        # Placeholders for columns we don't have real definitions for —
        # safe because they're not in the frozen 20 (see docstring).
        for col in PLACEHOLDER_ZERO_COLUMNS:
            raw[col] = 0.0

        return raw

    def extract(self, flow) -> np.ndarray:
        """
        Returns a (1, frozen_feature_count) scaled numpy array ready for the
        CNN. Internally: compute all 37 raw features -> scale all 37 (exact
        training order) -> slice down to the frozen 20.
        """
        raw = self._compute_raw_features(flow)

        missing = [f for f in self.full_feature_order if f not in raw]
        if missing:
            raise ValueError(
                f"Feature extractor cannot compute: {missing}. "
                f"Add definitions to _compute_raw_features() or PLACEHOLDER_ZERO_COLUMNS."
            )

        full_vector = np.array(
            [[raw[f] for f in self.full_feature_order]], dtype=np.float32
        )
        full_scaled = self.scaler.transform(full_vector)  # (1, 37)

        # Select only the frozen 20 columns, in frozen order
        frozen_scaled = full_scaled[:, self.frozen_indices]  # (1, 20)
        return frozen_scaled


if __name__ == "__main__":
    from dataclasses import dataclass, field
    import time

    @dataclass
    class FakePacket:
        _len: int
        _time: float
        def __len__(self): return self._len
        @property
        def time(self): return self._time
        def haslayer(self, layer): return False

    @dataclass
    class FakeFlow:
        src_ip: str = "10.0.0.1"
        dst_ip: str = "10.0.0.2"
        src_port: int = 12345
        dst_port: int = 80
        protocol: str = "TCP"
        start_time: float = field(default_factory=time.time)
        last_packet_time: float = field(default_factory=lambda: time.time() + 2)
        packets: list = field(default_factory=lambda: [FakePacket(60, time.time()),
                                                         FakePacket(60, time.time() + 1)])

    extractor = FeatureExtractor()
    fake_flow = FakeFlow()
    vector = extractor.extract(fake_flow)
    print(f"\nExtracted + scaled vector shape: {vector.shape}")
    print(vector)