# Edge-IDPS — Project Inspection Report

**Inspection date:** 2026-09-29
**Scope:** Read-only static inspection of `D:\Dev\Edge-IDPS-Project`. No code that captures packets, trains a model, or starts a server was run. `py_compile` was run over all project `.py` files (passed clean, no syntax errors); imports were grepped across project files only; all JSON/config files were read in full; `Stratified_data.csv` (~1.83 GB) and the `.npy` arrays were noted by size only, not parsed.

> **A note on confidence:** where a file's intent is clear from its code and docstrings, this report states it plainly. Where behaviour depends on an artifact trained off-repo (the deployed 5-class model), or on a claim I could not execute (live accuracy, feature correctness against real captures), that is called out explicitly rather than asserted.

---

## A. Project Mechanism (actual end-to-end data flow)

This is reconstructed from imports and function calls, not from the README.

### Live pipeline (the real runtime path)

Orchestrated by `engine/stream_pipeline.py` → `main()`:

1. **Capture** — `PacketSniffer` (`engine/packet_sniffer.py`) runs `scapy.sniff()` in a daemon thread with BPF filter `"ip or arp"`, `store=False`. Every packet is pushed via `put_nowait` onto a thread-safe `queue.Queue` (maxsize 10000). On overflow it drops the newest packet and counts it. Capture never processes — deliberate producer/consumer split.

2. **Flow assembly** — `FlowManager` (`engine/flow_manager.py`) runs two daemon threads:
   - *consumer*: pulls packets, computes a 5-tuple `(src_ip, dst_ip, src_port, dst_port, proto)` via `extract_5tuple()` (handles ARP, IP+TCP/UDP/ICMP; ICMP/ARP ports zeroed), groups into `Flow` dataclass objects, and closes a TCP flow on FIN (`0x01`) or RST (`0x04`).
   - *cleanup*: every 1 s, closes flows on idle timeout (5 s) or max duration (15 s).
   Closed `Flow` objects (with their raw packet list) are pushed onto `closed_flow_queue`.

3. **The evidence gate** — back in `stream_pipeline.main()`, each closed flow is drained. If `len(flow.packets) < 2` (`MIN_PACKETS_FOR_CLASSIFICATION`), it is **not** classified: a telemetry row is written with `predicted_class="INSUFFICIENT_EVIDENCE"`, `action="NO_ACTION"`, and the flow is skipped. This is the documented anti-false-positive gate.

4. **Inference** — `InferenceEngine.predict()` (`models/inference.py`):
   - `FeatureExtractor.extract()` (`engine/feature_extractor.py`) computes **all 37 raw features** in `preprocessing_metadata.json` order → transforms all 37 through the trained `scaler.pkl` → slices out the **frozen 20** columns in `frozen_features.json` order. (This ordering — scale-then-select — replicates the training pipeline; feeding 20 features straight into a 37-column scaler would raise.)
   - The `(1,20)` scaled vector → `EdgeIDPSCNN` (`models/cnn_model.py`) → `softmax` → argmax class + confidence + full class-probability dict.

5. **Risk scoring** — `RiskEngine.assess()` (`engine/risk_engine.py`): a Type-1 Mamdani fuzzy system (scikit-fuzzy) built entirely from `engine/risk_rules.json`. Inputs are **CNN confidence** and **attack frequency** (count of non-Benign predictions from this source IP inside a 60 s sliding window, tracked in a per-IP `deque` pruned each call → risk decays). Short-circuits: IPs in `ip_allowlist` → risk 0; `Benign` → risk 0 (benign override). Otherwise defuzzified to a crisp 0–10 score → mapped to an action band.

6. **Prevention** — `Firewall.act()` (`engine/firewall.py`) executes the action string:
   - `LOG_ONLY` / `THROTTLE` → informational only, no blocklist write (THROTTLE has no real rate-limiting — explicit placeholder).
   - `DROP_PACKET` / `BLOCK_IP` → append a line to `logs/active_blocklist.txt`. In `windows` mode `BLOCK_IP` also runs `netsh advfirewall` to add a real inbound block rule. `linux` mode is a stub (`print` only).

7. **Telemetry** — for every flow, `stream_pipeline` appends one row to `logs/live_stream_output.csv` (15 columns). **This is the only writer of telemetry.**

8. **Dashboard** — `dashboard/app.py` (Streamlit) **reads only** `live_stream_output.csv` and `active_blocklist.txt`, auto-refreshing every 1 s. It never writes. Shows active-threat count, flows/sec, avg confidence, model version, class distribution, risk-score histogram, risk timeline, blocklist table, recent events. INSUFFICIENT_EVIDENCE rows are excluded from threat metrics.

### Attack-generation path (lab only)

`hardware/flask_listener.py` (Flask on `0.0.0.0:5000`) exposes `POST /trigger` and `GET /health`. A trigger (from an ESP32 or `curl`) launches one of four `hardware/scapy_attacker.py` functions in a background thread: `dos_flood` (SYN flood), `arp_spoof`, `recon_scan` (SYN port sweep), `benign_traffic`. These are wired to the listener but are **independent of the detection pipeline** — they generate traffic on the wire that the sniffer may then observe.

### Offline / demo paths (do not require live capture)

- **`engine/replay_dataset.py`** — replays held-out `*_test_5class.npy` rows through model → risk engine → firewall (simulation) → telemetry, tagged `closed_reason="DATASET_REPLAY"`, `flow_id="replay-<idx>"`. Skips `FeatureExtractor` (rows already scaled). **This is what actually produced the current logs** (see §E).
- **Training/prep scripts** — `preprocess.py` → `feature_ranking.py` → `filter_classes.py` → `merge_dos_ddos.py` prepare data; `models/train_cnn.py` trains the **8-class** model. See §C/§E for the important gap: the deployed **5-class** model's training script is *not* in the repo.

---

## B. Tools & Libraries

### Third-party libraries actually imported (project files only)

| Library | Imported in |
|---|---|
| **torch** (`torch`, `torch.nn`, `torch.nn.functional`, `torch.utils.data`) | `models/cnn_model.py`, `models/inference.py`, `models/train_cnn.py`, `engine/replay_dataset.py` |
| **numpy** | `feature_extractor.py`, `risk_engine.py`, `inspect_features.py`, `validate_feature_extraction.py`, `replay_dataset.py`, `feature_ranking.py`, `filter_classes.py`, `merge_dos_ddos.py`, `preprocess.py`, `train_cnn.py` |
| **pandas** | `dashboard/app.py`, `feature_ranking.py`, `validate_feature_extraction.py`, `inspect_dataset.py`, `preprocess.py` |
| **scapy** (`scapy.all`) | `packet_sniffer.py`, `flow_manager.py`, `feature_extractor.py`, `validate_feature_extraction.py`, `hardware/scapy_attacker.py` |
| **scikit-fuzzy** (`skfuzzy`, `skfuzzy.control`) | `engine/risk_engine.py` |
| **scikit-learn** (`sklearn.*`) | `preprocess.py`, `feature_ranking.py`, `train_cnn.py` |
| **joblib** | `feature_extractor.py`, `preprocess.py`, `fix_class_labels.py`, `utils/config.py` |
| **matplotlib** | `feature_ranking.py` |
| **streamlit** + **streamlit-autorefresh** | `dashboard/app.py` |
| **flask** | `hardware/flask_listener.py` |

Standard-library modules used: `argparse, queue, threading, time, os, sys, json, csv, random, subprocess, dataclasses, collections, datetime`.

### Flags

- **No dependency manifest exists** — there is no `requirements.txt`, `pyproject.toml`, or `setup.py`. Dependencies are documented only in the README's single `pip install` line and scattered docstrings.
- **README `pip install` line is incomplete.** It lists `torch scikit-learn scapy scikit-fuzzy streamlit pandas numpy joblib` but omits three libraries that *are* imported: **matplotlib** (`feature_ranking.py`), **streamlit-autorefresh** (`dashboard/app.py` — the dashboard's own docstring does mention it), and **flask** (`hardware/flask_listener.py`). A fresh environment following only the README's main line would fail to run feature ranking, the dashboard, or the hardware listener.
- **No import with a missing/undocumented install was found** among project files — every third-party import maps to a library named somewhere (README or docstring), aside from the three README omissions above.
- **No unused imports** of significance in project files. (`utils/config.py:22` and `:114` flagged by a raw grep are a docstring line and a legitimate function-local `import joblib`, respectively.)

---

## C. Features Implemented vs. Not

| Component | Status | Notes |
|---|---|---|
| Packet capture | **Working** | Threaded Scapy sniffer, queue back-pressure, drop accounting. Requires Admin + Npcap; not executed here. |
| Flow management | **Working** | 5-tuple grouping, FIN/RST + idle/max-duration closure, thread-safe, shutdown drains active flows. |
| Feature extraction | **Working, correctness unverified live** | 37-raw → scale → slice-20 logic is sound and matches training order. *But* 5 of the 37 columns (`DHCP, ARP, IGMP, IPv, LLC`) are hard-coded `0.0` placeholders, and the docstring itself flags that CICIoT2023 feature formulas are "best-effort reconstructions." A dedicated validator (`validate_feature_extraction.py`) exists but needs live traffic to run — I could not confirm it passes. |
| CNN inference | **Working (loads deployed 5-class model)** | `inference.py` loads `edge_idps_model_5class.pth` + `model_metadata_5class.json`, with a defensive feature-count cross-check. Architecture in `cnn_model.py` is complete. |
| Fuzzy risk engine | **Working** | Fully config-driven Mamdani FIS; has a self-contained synthetic-scenario test in `main()` with assertions. The one component with an executable self-test that needs no live capture. |
| Firewall / prevention | **Working (simulation); windows mode plausible but unrun** | Simulation writes blocklist; `windows` mode issues real `netsh` rules (not exercised here); `linux` is a documented stub only. See §E for a DROP_PACKET logging quirk. |
| Dashboard | **Working** | Read-only Streamlit app; complete and internally consistent with the telemetry schema. |
| Attack simulation | **Working (lab only)** | Four Scapy modes; docstrings document empirically-tuned flow shapes (e.g. ~100 SYNs/flow to classify as DoS/DDoS). Not executed. |
| ESP32 / Flask trigger | **Working (server side); ESP32 firmware absent** | Flask listener + `/trigger` + `/health` are implemented. No ESP32 firmware/sketch is in the repo — the hardware side is simulated via `curl`. |
| Graceful shutdown | **Working** | `stop()` on sniffer and flow manager use stop-events + thread joins; flow manager flushes remaining flows on shutdown. Pipeline handles `KeyboardInterrupt`. |
| Logging / telemetry | **Working** | Append-only CSV writer; single-writer discipline enforced by design. |
| **5-class model training** | **Missing from repo** | The deployed model was trained off-repo (Colab). See §E #1. |

---

## D. Configuration State (actual current contents)

### `data/processed/frozen_features.json`
```
version: "1.0", feature_count: 20
feature_order (20): Tot sum, Max, AVG, Tot size, Rate, Time_To_Live,
  Header_Length, ack_flag_number, IAT, Min, Std, ack_count, HTTPS,
  Protocol Type, psh_flag_number, TCP, ICMP, UDP, syn_count, syn_flag_number
```

### `data/processed/class_order_5class.json`
```
class_order: ["DoS/DDoS", "Benign", "Mirai", "Reconnaissance", "Spoofing"]
merged_classes: {"DoS/DDoS": ["DoS","DDoS"]}
dropped_classes: ["Brute Force", "Web-based"]
note: DoS/DDoS merged after 6-class model misclassified 64% of DDoS as DoS
      (per-flow feature-granularity limit); single- vs multi-source flood
      distinction deferred to the risk engine.
```

### `engine/risk_rules.json`
```
version: "1.0", sliding_window_seconds: 60
ip_allowlist: ["192.168.60.73", "192.168.60.74"]     <-- current values
membership_functions:
  confidence [0,1]: low trap[0,0,.3,.5], medium tri[.3,.55,.8], high trap[.65,.85,1,1]
  frequency  [0,20]: low trap[0,0,2,5],  medium tri[2,7,12],    high trap[8,14,20,20]
  risk       [0,10]: low trap[0,0,1.5,3], medium tri[2,4.5,7],
                     high tri[5.5,7.5,9], critical trap[8,9,10,10]
rules: 9 (full confidence×frequency grid → low..critical)
action_thresholds: LOG_ONLY [0,3), THROTTLE [3,6), DROP_PACKET [6,8.5), BLOCK_IP [8.5,10]
benign_override: enabled = true
```

### `data/processed/model_metadata_5class.json` (recorded values — not re-run)
```
model: 1D-CNN            num_classes: 5           feature_count: 20
class_order: ["DoS/DDoS","Benign","Mirai","Reconnaissance","Spoofing"]
scaler: StandardScaler
test_accuracy: 0.98743   macro_f1: 0.85740
per_class_f1: DoS/DDoS 0.9993, Benign 0.7831, Mirai 0.9899,
              Reconnaissance 0.7141, Spoofing 0.8005
class_weights: DoS/DDoS 0.223, Benign 8.560, Mirai 3.568,
               Reconnaissance 13.606, Spoofing 19.328
hyperparameters: lr 0.001, batch 64, max_epochs 80, early_stop_patience 12,
                 loss "CrossEntropyLoss (weighted, no Focal Loss)", seed 42
training_environment: "Google Colab (T4 GPU)"     training_date: 2026-08-14
confusion_matrix (rows=true, cols=pred), order = class_order above:
  DoS/DDoS       [1456247,       9,   1587,    90,    97]
  Benign         [      0,   29774,      0,  6427,  1838]
  Mirai          [    248,       1,  91000,     3,    21]
  Reconnaissance [      1,    5537,      0, 17468,   927]
  Spoofing       [      0,    2679,      0,  1000, 13168]
```

**Reading the confusion matrix:** near-perfect DoS/DDoS and Mirai; the macro-F1 (0.857) is dragged down by **Benign, Reconnaissance, and Spoofing bleeding into each other** — e.g. 6,427 true-Benign flows predicted Reconnaissance, 5,537 true-Recon predicted Benign, and Spoofing losing rows to both. This is precisely the low-frequency false-positive problem the risk engine's frequency-escalation is designed to compensate for downstream.

---

## E. Inconsistencies / Red Flags

1. **The deployed model's training code is not in the repo (significant).** The live pipeline loads `edge_idps_model_5class.pth`, but the only training script present, `models/train_cnn.py`, trains the **8-class** model (`edge_idps_model.pth`) using `utils.config.CLASS_ORDER`, with `max_epochs=50, patience=7`. The 5-class metadata records `max_epochs=80, patience=12, "no Focal Loss"`, `training_environment="Google Colab (T4 GPU)"`. `filter_classes.py` and `merge_dos_ddos.py` end by instructing the user to *"upload … to your Colab Drive folder, then retrain with the 5-class cells."* So the actually-deployed model was trained by a Colab notebook that is **not committed**. The repo cannot reproduce its own production model.

2. **Stale `class_order` in `preprocessing_metadata.json`.** It records `["Benign","DoS","DDoS","Reconnaissance","Spoofing","Brute Force","Web-based","Mirai"]` — the *old, pre-fix "logical"* order — whereas `utils/config.py` `CLASS_ORDER` is now the corrected alphabetical order (`["Benign","Brute Force","DDoS","DoS","Mirai","Reconnaissance","Spoofing","Web-based"]`) that `LabelEncoder` actually produces. `preprocess.py` writes `CLASS_ORDER` into this file, so the artifact predates the config fix. **Impact is cosmetic** for the live path (only `feature_names` from this file is consumed downstream, and that is used consistently), but the `class_order` field is misleading and would confuse anyone reading it as ground truth.

3. **`DROP_PACKET` re-logs the same IP on every flow.** In `firewall.py`, `_block_ip()` adds the IP to `self._blocked_ips` (dedup), but `_drop_packet()` never does. So sustained DROP-band traffic writes a new `active_blocklist.txt` line per flow. The current `logs/active_blocklist.txt` confirms this — `10.10.10.60` appears repeatedly with `DROP_PACKET`. The dashboard's "Active Firewall Blocklist" table therefore shows many duplicate rows for one IP. Whether this is intended (DROP is per-packet, not a persistent block) is arguable, but the "avoid duplicate blocking/log spam" comment on `_blocked_ips` does not hold for DROP_PACKET.

4. **ARP is captured and grouped but has no real feature representation.** The sniffer filter includes `arp`, and `flow_manager.extract_5tuple()` builds `"ARP"` flows — but in `feature_extractor.py`, `ARP` is in `PLACEHOLDER_ZERO_COLUMNS` (forced to `0.0`), protocol number maps to `0`, and `TCP/UDP/ICMP` indicators are all `0`. So an ARP-spoofing flow reaches the CNN with its ARP-ness zeroed out. Given `arp_spoof` is a headline attack mode (→ Spoofing class), online ARP detection has weak feature support even though the plumbing captures it. (The trained model's Spoofing class came from dataset rows, where ARP was likewise a placeholder column.)

5. **Two model generations coexist; the 8-class artifacts are effectively dead code.** `edge_idps_model.pth`, `model_metadata.json`, `train_cnn.py`, `fix_class_labels.py`, and the `*_6class`/8-class `.npy` arrays are legacy — nothing in the live pipeline loads them. `inference.py`'s own docstring says `utils.config.CLASS_ORDER` is "superseded." Not wrong, but a reader must know which generation is live (the 5-class one).

6. **Copy-paste artifact comment.** `flow_manager.py:65` contains a stray `# ... rest of the function stays exactly the same` mid-function — a leftover from an edit. Harmless, but sloppy.

7. **`benign.pcap` is 24 bytes** — a pcap global header with zero packets. It is empty; `.gitignore` ignores `*.pcap` yet this one is tracked. Likely a stray placeholder.

8. **Current logs are dataset-replay, not live capture.** Every row in `logs/live_stream_output.csv` is `closed_reason="DATASET_REPLAY"` from 2026-09-29, source IP `10.10.10.60` (a non-default `--attacker-ip`). No live-capture telemetry is present, so live packet→feature correctness has not been demonstrated in the committed artifacts. `logs/live_stream_output_old.csv` (~864 KB) is a prior run retained alongside.

9. **`linux` firewall mode is half-wired.** `Firewall.__init__` accepts `"linux"` and `_block_ip` dispatches to `_linux_block`, but `_linux_block` only prints; `stream_pipeline`'s `--firewall-mode` choices are limited to `simulation`/`windows`, so `linux` is unreachable from the CLI anyway. Documented as intentional stub.

10. **No test suite / CI.** The "tests" are `if __name__ == "__main__"` self-checks inside modules (`risk_engine.py` has real assertions; others are manual print harnesses that need live capture). No `pytest` files, no automated regression coverage for the pipeline.

**No file was found referencing a path that does not exist** in the live path (all `data/processed/*` and `models/*` artifacts consumed by the pipeline are present). `data/raw/` is named in `.gitignore` but absent from the tree — expected, since raw data is not committed.

---

## F. File Inventory

Legend — **Working**: complete, coherent, consistent with the live path. **Incomplete/untested**: exists but not executable/verifiable here or relies on off-repo pieces. **Legacy/superseded**: functional but not on the deployed path. **Stub**.

| Path | Purpose (one line) | Status |
|---|---|---|
| `engine/packet_sniffer.py` | Threaded Scapy capture → queue, with drop accounting | Working |
| `engine/flow_manager.py` | Group packets into 5-tuple flows; close on FIN/RST/timeout | Working (stray comment L65) |
| `engine/feature_extractor.py` | 37 raw features → scale → slice frozen 20 for the CNN | Working; live correctness unverified (5 placeholder cols) |
| `engine/inference` *(see models/)* | — | — |
| `engine/risk_engine.py` | Config-driven Mamdani fuzzy risk + per-IP sliding window | Working (has self-test w/ assertions) |
| `engine/firewall.py` | Execute LOG/THROTTLE/DROP/BLOCK; sim + windows(netsh) | Working; DROP_PACKET re-logs (§E#3); linux stub |
| `engine/risk_rules.json` | Fuzzy MFs, 9 rules, action bands, allowlist, benign override | Working (config) |
| `engine/stream_pipeline.py` | **Live orchestrator**; sole telemetry writer | Working |
| `engine/replay_dataset.py` | Replay held-out test rows through model→risk→firewall→CSV | Working (untracked at snapshot; wrote current logs) |
| `engine/inspect_features.py` | Diagnostic: per-feature z-scores for real flows | Working (diagnostic; needs live traffic) |
| `engine/validate_feature_extraction.py` | Validate extractor vs independent recompute (Phase 7 gate) | Incomplete/untested (needs live traffic to run) |
| `models/cnn_model.py` | `EdgeIDPSCNN` 1D-CNN architecture | Working |
| `models/inference.py` | Load deployed 5-class model; flow→prediction | Working |
| `models/train_cnn.py` | Train the **8-class** CNN; write model + metadata | Legacy/superseded (not the deployed model) |
| `models/edge_idps_model_5class.pth` | Deployed 5-class weights (34 KB) | Working artifact (trained off-repo) |
| `models/edge_idps_model.pth` | 8-class weights (35 KB) | Legacy/superseded |
| `preprocess.py` | Load CSV → map labels → split → fit scaler → save arrays | Working (produced stale class_order field, §E#2) |
| `feature_ranking.py` | RF + MI ranking → freeze top-20 → `frozen_features.json` | Working |
| `filter_classes.py` | Drop Brute Force/Web-based → 6-class arrays | Working (prep step; hands off to Colab) |
| `merge_dos_ddos.py` | Merge DoS+DDoS → 5-class arrays + class order | Working (prep step; hands off to Colab) |
| `fix_class_labels.py` | Rewrite 8-class metadata per true encoder order | Legacy (one-off fix for 8-class metadata) |
| `inspect_dataset.py` | Ad-hoc CSV class-distribution/nulls inspector | Working (utility) |
| `utils/config.py` | `LABEL_MAP`, alphabetical `CLASS_ORDER`, `map_labels`, verifier | Working (8-class taxonomy; superseded by 5-class at runtime) |
| `dashboard/app.py` | Read-only Streamlit SOC dashboard | Working |
| `hardware/scapy_attacker.py` | Four Scapy attack generators (lab only) | Working (lab only; not run) |
| `hardware/flask_listener.py` | Flask `/trigger` + `/health` → launch attacks | Working (ESP32 firmware absent) |
| `data/processed/frozen_features.json` | The 20 frozen feature names/order | Working (config) |
| `data/processed/class_order_5class.json` | Deployed 5-class order + merge/drop provenance | Working (config) |
| `data/processed/class_order_6class.json` | Intermediate 6-class order | Legacy/intermediate |
| `data/processed/model_metadata_5class.json` | Deployed model architecture + metrics | Working (metadata) |
| `data/processed/model_metadata.json` | 8-class model metadata | Legacy/superseded |
| `data/processed/preprocessing_metadata.json` | 37-feature order, split settings, dataset stats | Working but stale `class_order` (§E#2) |
| `data/processed/scaler.pkl`, `label_encoder.pkl` | Fitted StandardScaler / LabelEncoder | Working artifacts |
| `data/processed/*.npy` (train/test, 8/6/5-class) | Preprocessed feature/label arrays (hundreds MB–~1.9 GB) | Data (not parsed) |
| `data/processed/feature_importance.csv`, `feature_ranking_full.csv`, `feature_ranking.png` | Ranking deliverables | Working (artifacts) |
| `Stratified_data.csv` | Raw CICIoT2023 stratified dataset (~1.83 GB) | Data (existence + size only, not parsed) |
| `benign.pcap` | 24-byte empty pcap (header only) | Stray/empty (§E#7) |
| `logs/live_stream_output.csv` | Current telemetry (all DATASET_REPLAY) | Runtime output |
| `logs/live_stream_output_old.csv` | Prior telemetry run (~864 KB) | Runtime output (retained) |
| `logs/active_blocklist.txt` / `_archive.txt` | Blocklist entries / archive | Runtime output |
| `README.md` | Overview + run instructions | Working (incomplete pip line, §B) |
| `.gitignore`, `.vscode/settings.json` | Ignore rules; analysis extraPaths | Working (config) |

---

## Summary

The **live detection path is coherent and complete end to end** — capture → flow → gate → 5-class CNN → fuzzy risk → firewall → telemetry → read-only dashboard — with genuinely thoughtful design touches (producer/consumer threading, the INSUFFICIENT_EVIDENCE gate, a stateful decaying risk engine that self-tests, strict single-writer telemetry, and scale-then-select feature ordering that matches training). The recorded 5-class model is strong overall (98.7% acc / 0.857 macro-F1), with Benign/Recon/Spoofing confusion being the known weak spot the risk engine is meant to backstop.

The **most consequential gap** is reproducibility: the deployed 5-class model was trained in an uncommitted Colab notebook, so the repo's own `train_cnn.py` produces a *different* (8-class) model. Secondary issues are a stale `class_order` in `preprocessing_metadata.json`, `DROP_PACKET` blocklist re-logging, ARP captured but not feature-represented, and no automated test suite. None of these break the demo path — which the current logs show was exercised via `replay_dataset.py` rather than live capture — but they matter for a viva-grade claim of "reproducible, validated, live" detection.
