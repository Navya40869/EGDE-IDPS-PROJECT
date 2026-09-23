"""
hardware/flask_listener.py — Phase 11: Flask Attack Listener API

Receives an HTTP POST trigger (from the ESP32, or simulated via curl/browser
for testing without the physical hardware) and launches the corresponding
Scapy attack function in a background thread — the HTTP response returns
immediately rather than blocking for the full attack duration.

OPERATIONAL CONSTRAINT — same as scapy_attacker.py: only trigger attacks
against devices on an isolated lab network/VM you control.

Usage:
    python hardware/flask_listener.py
    (runs on http://localhost:5000 by default)

Simulating an ESP32 trigger without the physical hardware, e.g. with curl:
    curl -X POST http://localhost:5000/trigger ^
         -H "Content-Type: application/json" ^
         -d "{\"mode\": \"dos\", \"target\": \"192.168.1.50\", \"duration\": 15}"
"""

import threading

from flask import Flask, request, jsonify

from scapy_attacker import dos_flood, arp_spoof, recon_scan, benign_traffic

app = Flask(__name__)

VALID_MODES = {"dos", "arp_spoof", "recon", "benign"}


@app.route("/trigger", methods=["POST"])
def trigger():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    mode = data.get("mode")
    target = data.get("target")

    if mode not in VALID_MODES:
        return jsonify({"error": f"mode must be one of {sorted(VALID_MODES)}"}), 400
    if not target:
        return jsonify({"error": "target IP is required"}), 400

    duration = int(data.get("duration", 15))
    rate = data.get("rate")  # None -> attack function uses its own default

    print(f"[Flask] Received trigger: mode={mode}, target={target}, "
          f"duration={duration}, rate={rate}")

    if mode == "dos":
        port = int(data.get("port", 80))
        spoof_source = bool(data.get("spoof_source", False))
        packets_per_flow = int(data.get("packets_per_flow", 150))
        thread = threading.Thread(
            target=dos_flood,
            args=(target, port, duration, rate or 100, spoof_source, packets_per_flow),
            daemon=True,
        )
    elif mode == "recon":
        port_start = int(data.get("port_start", 1))
        port_end = int(data.get("port_end", 1024))
        probes_per_port = int(data.get("probes_per_port", 2))
        thread = threading.Thread(
            target=recon_scan,
            args=(target, port_start, port_end, duration, probes_per_port, rate or 100),
            daemon=True,
        )
    elif mode == "arp_spoof":
        spoof_ip = data.get("spoof")
        if not spoof_ip:
            return jsonify({"error": "arp_spoof mode requires a 'spoof' IP "
                                      "(the IP to impersonate, e.g. the gateway)"}), 400
        thread = threading.Thread(
            target=arp_spoof,
            args=(target, spoof_ip, duration, rate or 5),
            daemon=True,
        )
    elif mode == "benign":
        port = int(data.get("port", 80))
        thread = threading.Thread(
            target=benign_traffic,
            args=(target, port, duration, rate or 10),
            daemon=True,
        )

    thread.start()

    return jsonify({
        "status": "triggered",
        "mode": mode,
        "target": target,
        "duration": duration,
    }), 200


@app.route("/health", methods=["GET"])
def health():
    """Simple endpoint to confirm the listener is up, e.g. for the ESP32 to
    check connectivity before sending a real trigger."""
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    print("=" * 60)
    print("Edge-IDPS Flask Attack Listener")
    print("REMINDER: only trigger attacks against an isolated lab network/VM.")
    print("=" * 60)
    print("Listening on http://0.0.0.0:5000")
    print("POST /trigger with JSON: {\"mode\": \"dos\"|\"arp_spoof\"|\"recon\"|\"benign\", "
          "\"target\": \"<ip>\", \"duration\": <seconds>}")
    app.run(host="0.0.0.0", port=5000, debug=False)