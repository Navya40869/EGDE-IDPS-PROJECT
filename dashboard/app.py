"""
dashboard/app.py — Phase 12: Streamlit SOC Dashboard

READ-ONLY — this file only ever reads logs/live_stream_output.csv and
logs/active_blocklist.txt. It never writes to either; all writing happens in
engine/stream_pipeline.py. This separation is deliberate (per the blueprint):
the dashboard polls, the pipeline writes — never the reverse.

Refreshes every second via streamlit_autorefresh.

Run:
    pip install streamlit streamlit-autorefresh pandas
    streamlit run dashboard/app.py

(In a separate terminal, run engine/stream_pipeline.py to actually generate
telemetry for the dashboard to display.)
"""

import os

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
TELEMETRY_PATH = os.path.join(LOGS_DIR, "live_stream_output.csv")
BLOCKLIST_PATH = os.path.join(LOGS_DIR, "active_blocklist.txt")

RECENT_WINDOW_SECONDS = 60  # for "active threat count" and "flows/sec" metrics

st.set_page_config(page_title="Edge-IDPS SOC Dashboard", layout="wide")
st_autorefresh(interval=1000, key="dashboard_refresh")  # 1-second refresh

st.title("Edge-IDPS — Real-Time SOC Dashboard")


def load_telemetry() -> pd.DataFrame:
    if not os.path.exists(TELEMETRY_PATH):
        return pd.DataFrame(columns=[
            "timestamp", "flow_id", "source_ip", "destination_ip",
            "predicted_class", "confidence", "risk_score", "action", "model_version",
        ])
    df = pd.read_csv(TELEMETRY_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def load_blocklist() -> pd.DataFrame:
    if not os.path.exists(BLOCKLIST_PATH):
        return pd.DataFrame(columns=["timestamp", "blocked_ip", "trigger", "predicted_class", "risk_score"])
    return pd.read_csv(
        BLOCKLIST_PATH,
        names=["timestamp", "blocked_ip", "trigger", "predicted_class", "risk_score"],
        header=None,
    )


telemetry_df = load_telemetry()
blocklist_df = load_blocklist()

if telemetry_df.empty:
    st.warning(
        "No telemetry yet. Run `python engine\\stream_pipeline.py --interface \"Wi-Fi\" "
        "--duration 60` in a separate terminal to start generating live data."
    )
else:
    now = pd.Timestamp.now()
    recent_cutoff = now - pd.Timedelta(seconds=RECENT_WINDOW_SECONDS)
    recent_df = telemetry_df[telemetry_df["timestamp"] >= recent_cutoff]

    # A real threat is a classified, non-Benign flow — INSUFFICIENT_EVIDENCE
    # is neither a threat nor a clean pass, it's "we chose not to guess" due
    # to too few packets in the flow (see engine/stream_pipeline.py gate).
    classified_df = recent_df[recent_df["predicted_class"] != "INSUFFICIENT_EVIDENCE"]

    # --- Top metrics row ---
    col1, col2, col3, col4 = st.columns(4)

    active_threats = (classified_df["predicted_class"] != "Benign").sum()
    col1.metric("Active Threats (last 60s)", int(active_threats))

    flows_per_sec = len(recent_df) / RECENT_WINDOW_SECONDS
    col2.metric("Flows / sec (avg, last 60s)", f"{flows_per_sec:.2f}")

    avg_confidence = classified_df["confidence"].mean() if not classified_df.empty else 0.0
    col3.metric("Avg CNN Confidence", f"{avg_confidence:.2%}" if pd.notna(avg_confidence) else "N/A")

    model_version = telemetry_df["model_version"].iloc[-1] if not telemetry_df.empty else "N/A"
    col4.metric("Model Version", model_version)

    insufficient_count = (recent_df["predicted_class"] == "INSUFFICIENT_EVIDENCE").sum()
    st.caption(f"{insufficient_count} of {len(recent_df)} flows in the last 60s were "
               f"INSUFFICIENT_EVIDENCE (too few packets to classify) and are excluded "
               f"from the metrics above.")

    st.divider()

    # --- Charts row ---
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Attack Category Distribution")
        if not recent_df.empty:
            class_counts = recent_df["predicted_class"].value_counts()
            st.bar_chart(class_counts)
        else:
            st.write("No data in the last 60s.")

    with chart_col2:
        st.subheader("Risk Score Distribution")
        if not classified_df.empty:
            binned = pd.cut(classified_df["risk_score"], bins=10)
            counts = binned.value_counts().sort_index()
            # Convert Interval index to plain string labels — passing the
            # raw IntervalIndex straight to st.bar_chart renders garbled
            # axis text (e.g. "(0.441, 1.882]" gets truncated/mangled).
            counts.index = [f"{iv.left:.1f}\u2013{iv.right:.1f}" for iv in counts.index]
            st.bar_chart(counts)
        else:
            st.write("No classified flows in the last 60s.")

    st.subheader("Timeline of Risk Scores")
    if not classified_df.empty:
        timeline = classified_df.sort_values("timestamp").set_index("timestamp")["risk_score"]
        st.line_chart(timeline)
    else:
        st.write("No data in the last 60s.")

    st.divider()

    # --- Tables row ---
    table_col1, table_col2 = st.columns(2)

    with table_col1:
        st.subheader("Active Firewall Blocklist")
        if not blocklist_df.empty:
            st.dataframe(blocklist_df.sort_values("timestamp", ascending=False), use_container_width=True)
        else:
            st.write("No IPs currently blocked.")

    with table_col2:
        st.subheader("Recent Telemetry Events")
        recent_events = telemetry_df.sort_values("timestamp", ascending=False).head(20)
        st.dataframe(recent_events, use_container_width=True)

st.caption("Dashboard is read-only — polls logs/live_stream_output.csv and "
           "logs/active_blocklist.txt every 1s. Never writes to either.")