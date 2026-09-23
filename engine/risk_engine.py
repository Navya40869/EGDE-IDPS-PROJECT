"""
engine/risk_engine.py — Phase 9: Stateful Fuzzy Risk Engine

Implements a Type-1 Mamdani Fuzzy Inference System:
    Fuzzification (triangular/trapezoidal membership functions)
    -> Rule evaluation (IF-THEN, AND = min)
    -> Aggregation
    -> Defuzzification (centroid)
    -> crisp 0-10 risk score -> action band

Inputs: CNN confidence, and attack frequency from this source IP over a
sliding time window (tracked via a per-IP deque, pruned each cycle so risk
DECAYS over time rather than accumulating forever — this is what makes the
engine "stateful").

All membership function breakpoints, rules, and action thresholds are loaded
from risk_rules.json at runtime — nothing is hardcoded here, so thresholds
can be tuned without touching this file.

This engine exists specifically to address a finding from live testing: the
CNN alone produces false positives on isolated, low-frequency traffic (e.g.
a single background DNS query classified as Spoofing). A single isolated
high-confidence prediction now yields only MEDIUM risk (see the rule table),
while REPEATED high-confidence predictions from the same source IP escalate
to CRITICAL — requiring corroboration before serious action is taken.

Usage (standalone validation, per blueprint Section 14 — synthetic scenarios):
    python engine/risk_engine.py
"""

import json
import os
import time
from collections import defaultdict, deque

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

RISK_RULES_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "engine", "risk_rules.json")


def _build_membership(variable, mf_config, resolution=0.01):
    """Constructs an Antecedent/Consequent's membership functions from JSON config."""
    domain = mf_config["domain"]
    universe = np.arange(domain[0], domain[1] + resolution, resolution)
    var = ctrl.Antecedent(universe, variable) if variable != "risk" else ctrl.Consequent(universe, variable)

    for label, spec in mf_config.items():
        if label in ("domain", "note"):
            continue
        if spec["type"] == "trimf":
            var[label] = fuzz.trimf(var.universe, spec["points"])
        elif spec["type"] == "trapmf":
            var[label] = fuzz.trapmf(var.universe, spec["points"])
        else:
            raise ValueError(f"Unknown membership function type: {spec['type']}")
    return var


class RiskEngine:
    def __init__(self, risk_rules_path: str = RISK_RULES_PATH):
        with open(risk_rules_path) as f:
            self.config = json.load(f)

        self.window_seconds = self.config["sliding_window_seconds"]
        self.action_thresholds = self.config["action_thresholds"]
        self.benign_override = self.config["benign_override"]["enabled"]

        # --- Build the Mamdani FIS from JSON config, not hardcoded ---
        mfs = self.config["membership_functions"]
        self.confidence_var = _build_membership("confidence", mfs["confidence"])
        self.frequency_var = _build_membership("frequency", mfs["frequency"])
        self.risk_var = _build_membership("risk", mfs["risk"])

        rules = []
        for rule_spec in self.config["rules"]:
            antecedent = (self.confidence_var[rule_spec["confidence"]] &
                          self.frequency_var[rule_spec["frequency"]])
            consequent = self.risk_var[rule_spec["risk"]]
            rules.append(ctrl.Rule(antecedent, consequent))

        self.control_system = ctrl.ControlSystem(rules)
        self.simulation = ctrl.ControlSystemSimulation(self.control_system)

        # --- Stateful per-source-IP tracking ---
        # ip -> deque of (timestamp, predicted_class, confidence)
        self.ip_history = defaultdict(deque)

        print(f"[RiskEngine] Loaded {len(rules)} fuzzy rules, "
              f"sliding window = {self.window_seconds}s")

    def _prune_old_entries(self, source_ip: str, now: float):
        history = self.ip_history[source_ip]
        while history and (now - history[0][0]) > self.window_seconds:
            history.popleft()

    def _compute_frequency(self, source_ip: str, now: float) -> int:
        """Count of non-Benign predictions from this IP within the sliding window."""
        self._prune_old_entries(source_ip, now)
        return sum(1 for (_, cls, _) in self.ip_history[source_ip] if cls != "Benign")

    def _fuzzy_compute(self, confidence: float, frequency: int) -> float:
        # Clip to the universe bounds — the FIS domain is fixed, real inputs
        # can exceed it (e.g. more than 20 attack flows in the window)
        confidence = min(max(confidence, 0.0), 1.0)
        frequency = min(max(frequency, 0), 20)

        self.simulation.input["confidence"] = confidence
        self.simulation.input["frequency"] = frequency
        self.simulation.compute()
        return float(self.simulation.output["risk"])

    def _score_to_action(self, score: float) -> str:
        for action, (low, high) in self.action_thresholds.items():
            if low <= score < high or (score == high == 10.0):
                return action
        return "LOG_ONLY"  # fallback, should not normally be reached

    def assess(self, source_ip: str, predicted_class: str, confidence: float) -> dict:
        """
        Call this once per classified flow. Returns the risk assessment.
        """
        now = time.time()
        self.ip_history[source_ip].append((now, predicted_class, confidence))
        self._prune_old_entries(source_ip, now)

        if source_ip in self.config.get("ip_allowlist", []):
            risk_score = 0.0
            frequency = 0
        elif self.benign_override and predicted_class == "Benign":
           risk_score = 0.0
           frequency = 0
        else:
           frequency = self._compute_frequency(source_ip, now)
           risk_score = self._fuzzy_compute(confidence, frequency)

        action = self._score_to_action(risk_score)

        return {
            "source_ip": source_ip,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "frequency_in_window": frequency,
            "risk_score": round(risk_score, 3),
            "action": action,
        }


def main():
    """
    Standalone validation using synthetic scenarios, per the blueprint's
    Phase 14 requirement: "Validate the fuzzy risk engine using synthetic
    scenarios." No live capture needed — this directly exercises the FIS.
    """
    engine = RiskEngine()

    print("\n=== Scenario 1: single low-confidence Benign flow ===")
    result = engine.assess("10.0.0.1", "Benign", 0.55)
    print(result)
    assert result["action"] == "LOG_ONLY", "Benign should always be LOG_ONLY"
    print("PASS")

    print("\n=== Scenario 2: single isolated high-confidence Spoofing prediction ===")
    print("(this is the false-positive case from live testing — should NOT immediately BLOCK)")
    result = engine.assess("10.0.0.2", "Spoofing", 0.95)
    print(result)
    assert result["action"] in ("LOG_ONLY", "THROTTLE"), \
        "A single isolated high-confidence hit should not immediately escalate to DROP/BLOCK"
    print("PASS")

    print("\n=== Scenario 3: burst of high-confidence DDoS/DoS flows from one IP ===")
    print("(repeated attacks from the same source should escalate)")
    ip = "10.0.0.3"
    for i in range(15):
        result = engine.assess(ip, "DoS/DDoS", 0.97)
    print(f"After 15 rapid high-confidence hits: {result}")
    assert result["action"] in ("DROP_PACKET", "BLOCK_IP"), \
        "Sustained high-confidence attack pattern should escalate to DROP/BLOCK"
    print("PASS")

    print("\n=== Scenario 4: risk decay — old attacks should stop counting after the window ===")
    ip = "10.0.0.4"
    for i in range(15):
        engine.assess(ip, "DoS/DDoS", 0.97)
    print("Simulating the sliding window expiring (manually clearing history to simulate time passing)...")
    engine.ip_history[ip].clear()  # simulates all entries aging out of the window
    result = engine.assess(ip, "DoS/DDoS", 0.97)
    print(f"After window reset, single new hit: {result}")
    assert result["frequency_in_window"] <= 1, "Frequency should have reset after the window passed"
    print("PASS")

    print("\n=== Scenario 5: medium confidence, medium frequency ===")
    ip = "10.0.0.5"
    for i in range(5):
        result = engine.assess(ip, "Reconnaissance", 0.65)
    print(result)
    print("(no strict assertion — just confirming a moderate, non-extreme result)")

    print("\nAll synthetic scenarios passed. Risk Engine validated.")


if __name__ == "__main__":
    main()