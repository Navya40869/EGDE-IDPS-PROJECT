# Edge-IDPS

An edge intrusion detection/prevention pipeline that sniffs live network
traffic, groups packets into flows, classifies each flow with a 1D-CNN
(Benign, DoS/DDoS, Mirai, Reconnaissance, Spoofing), scores the threat through
a stateful fuzzy risk engine with per-source-IP sliding-window escalation, and
takes a firewall action. Flows with too few packets to characterize are gated
as `INSUFFICIENT_EVIDENCE` rather than guessed at. A read-only Streamlit
dashboard visualizes the live telemetry. The `hardware/` tools (Scapy attack
generator + Flask/ESP32 trigger) exist to exercise detection on an **isolated
lab network you control**.

## Prerequisites

- Python 3.11+ with the project dependencies:
  `pip install torch scikit-learn scapy scikit-fuzzy streamlit pandas numpy joblib`
- **Live capture needs raw-socket access**: run the pipeline terminal as
  Administrator (Windows) or with `sudo` (Linux). Windows also needs
  [Npcap](https://npcap.com/) installed.

## Running

The pipeline is the only component that **writes** telemetry
(`logs/live_stream_output.csv`); the dashboard only **reads** it. Run them in
two terminals.

**1. Start the detection pipeline** (elevated terminal). Replace `"Wi-Fi"`
with your capture interface:

```
python engine/stream_pipeline.py --interface "Wi-Fi" --duration 120 --firewall-mode simulation
```

- `--interface` — NIC to sniff (omit to use Scapy's default).
- `--duration` — seconds to run.
- `--firewall-mode` — `simulation` (log actions only) or `windows` (apply real
  firewall rules).

**2. Start the dashboard** (separate terminal):

```
streamlit run dashboard/app.py
```

It opens in your browser and polls `logs/live_stream_output.csv`, showing
active threats, average CNN confidence, risk-score distribution, and a risk
timeline over the last 60 seconds (INSUFFICIENT_EVIDENCE flows are counted
separately, not as threats).

With both running, any traffic on the sniffed interface appears as classified
flows in the dashboard within a few seconds.
