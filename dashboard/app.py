"""
dashboard/app.py — Phase 12: Streamlit SOC Dashboard (Kali-style theme)

READ-ONLY — this file only ever reads logs/live_stream_output.csv and
logs/active_blocklist.txt. It never writes to either; all writing happens in
engine/stream_pipeline.py. This separation is deliberate: the dashboard
polls, the pipeline writes — never the reverse.

One deliberate exception: the sidebar "Explain a live flow" button runs a
short sniff-only capture to explain a fresh prediction (see the EXPLAIN A LIVE
FLOW section). It only runs on an explicit click, never on auto-refresh, and
still writes nothing. Its sibling "Explain a replayed test row" captures
nothing: it explains a held-out row from data/processed/X_test_5class.npy.

Refreshes every second via streamlit_autorefresh (can be paused from the
sidebar for a static read while explaining something).

AUTHENTICATION REQUIRED — a login gate (streamlit-authenticator, bcrypt
hashes) runs before the sidebar, data loading and the explain button; nothing
else renders until you sign in (see the AUTHENTICATION GATE section).
One-time setup:
    1. python dashboard/generate_password_hash.py --username <name>
    2. Paste its output into dashboard/auth_config.yaml (gitignored; template:
       dashboard/auth_config.example.yaml). Only bcrypt hashes are accepted.

Run:
    pip install streamlit streamlit-autorefresh streamlit-authenticator pandas plotly
    streamlit run dashboard/app.py

(In a separate terminal, run engine/stream_pipeline.py to actually generate
telemetry for the dashboard to display.)
"""

import ipaddress
import json
import os

import pandas as pd
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from streamlit_autorefresh import st_autorefresh

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
TELEMETRY_PATH = os.path.join(LOGS_DIR, "live_stream_output.csv")
BLOCKLIST_PATH = os.path.join(LOGS_DIR, "active_blocklist.txt")

st.set_page_config(page_title="Edge-IDPS SOC Terminal", layout="wide", page_icon="🛡")

# =============================================================================
# THEME — dark terminal / Kali-style look. Pure CSS, no behavior change.
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    }
    .stApp {
        background-color: #0a0e0a;
        color: #00ff41;
    }
    section[data-testid="stSidebar"] {
        background-color: #0d1310;
        border-right: 1px solid #00ff4133;
    }
    h1, h2, h3 {
        color: #00ff41 !important;
        text-shadow: 0 0 8px #00ff4155;
    }
    h1 {
        border-bottom: 2px solid #00ff41;
        padding-bottom: 8px;
    }
    .term-tag {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 3px;
        font-weight: 700;
        font-size: 0.85em;
        letter-spacing: 1px;
    }
    .tag-log      { background:#1a1a1a; color:#888; border:1px solid #444; }
    .tag-throttle { background:#2a2200; color:#ffd400; border:1px solid #ffd400; }
    .tag-drop     { background:#2a1500; color:#ff8c00; border:1px solid #ff8c00; }
    .tag-block    { background:#2a0000; color:#ff2b2b; border:1px solid #ff2b2b;
                     text-shadow:0 0 6px #ff2b2b; }
    .tag-benign   { background:#002a0a; color:#00ff41; border:1px solid #00ff41; }
    .tag-unknown  { background:#1a1a1a; color:#666; border:1px solid #333; }

    .term-box {
        background: #0d1310;
        border: 1px solid #00ff4144;
        border-radius: 4px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }
    .term-header {
        color: #00ff41;
        font-weight: 700;
        letter-spacing: 2px;
        border-bottom: 1px dashed #00ff4155;
        padding-bottom: 6px;
        margin-bottom: 8px;
    }
    .feed-line {
        font-size: 0.86em;
        padding: 3px 0;
        border-bottom: 1px solid #ffffff0a;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    div[data-testid="stMetricValue"] {
        color: #00ff41 !important;
        text-shadow: 0 0 6px #00ff4166;
    }
    div[data-testid="stMetricLabel"] {
        color: #00cc33aa !important;
        letter-spacing: 1px;
    }
    .stDataFrame { border: 1px solid #00ff4133 !important; }
    ::-webkit-scrollbar { width: 8px; background: #0a0e0a; }
    ::-webkit-scrollbar-thumb { background: #00ff4155; border-radius: 4px; }
    /* Holds only the reduced-motion probe below; never takes up space */
    .st-key-motion_probe { display: none; }
</style>
""", unsafe_allow_html=True)

ACTION_TAG = {
    "LOG_ONLY": '<span class="term-tag tag-log">LOG_ONLY</span>',
    "THROTTLE": '<span class="term-tag tag-throttle">THROTTLE</span>',
    "DROP_PACKET": '<span class="term-tag tag-drop">DROP_PACKET</span>',
    "BLOCK_IP": '<span class="term-tag tag-block">BLOCK_IP</span>',
}


def action_tag(action: str) -> str:
    return ACTION_TAG.get(action, f'<span class="term-tag tag-unknown">{action}</span>')


def class_tag(predicted_class: str) -> str:
    if predicted_class == "Benign":
        return f'<span class="term-tag tag-benign">{predicted_class}</span>'
    if predicted_class == "INSUFFICIENT_EVIDENCE":
        return f'<span class="term-tag tag-unknown">INSUFF_EVID</span>'
    return f'<span class="term-tag tag-block">{predicted_class}</span>'


def _valid_ip(text: str) -> bool:
    """Blank (no filter) or a well-formed IPv4/IPv6 address."""
    if not text:
        return True
    try:
        ipaddress.ip_address(text)
        return True
    except ValueError:
        return False


@st.cache_data
def _replay_class_options() -> list:
    # Read from the JSON rather than the model, so the sidebar loads no torch
    with open(os.path.join(PROJECT_ROOT, "data", "processed", "class_order_5class.json")) as f:
        return json.load(f)["class_order"]


# =============================================================================
# AUTHENTICATION GATE — must stay ABOVE the sidebar, data loading and the
# "Explain a live flow" button. Unless the session is authenticated,
# st.stop() ends the script inside this block: no sidebar controls render, no
# auto-refresh timer starts, no log file is read, no capture can be triggered.
# Only the theme CSS and tag helpers above (no data) run before it.
#
# Credentials: dashboard/auth_config.yaml (gitignored), bcrypt hashes only.
# Setup: python dashboard/generate_password_hash.py --username <name>
# Template: dashboard/auth_config.example.yaml
# =============================================================================
AUTH_CONFIG_PATH = os.path.join(PROJECT_ROOT, "dashboard", "auth_config.yaml")
AUTH_SETUP_HINT = ("Run `python dashboard\\generate_password_hash.py --username <name>` and "
                   "paste its output into `dashboard\\auth_config.yaml` "
                   "(see `dashboard\\auth_config.example.yaml`).")


def _load_auth_config():
    """Returns (config, error_message). Rejects any non-bcrypt password."""
    if not os.path.exists(AUTH_CONFIG_PATH):
        return None, f"No login config found at `dashboard\\auth_config.yaml`. {AUTH_SETUP_HINT}"
    try:
        with open(AUTH_CONFIG_PATH, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        users = config["credentials"]["usernames"] or {}
        cookie = config["cookie"]
        cookie_fields = (str(cookie["name"]), str(cookie["key"]), float(cookie["expiry_days"]))
    except (OSError, yaml.YAMLError, KeyError, TypeError, ValueError) as e:
        return None, f"`dashboard\\auth_config.yaml` is malformed ({e!r}). {AUTH_SETUP_HINT}"
    if not users:
        return None, f"`dashboard\\auth_config.yaml` defines no users. {AUTH_SETUP_HINT}"
    not_hashed = [u for u, rec in users.items()
                  if not stauth.Hasher.is_hash(str((rec or {}).get("password", "")))]
    if not_hashed:
        return None, (f"Password for {not_hashed} is not a bcrypt hash — plaintext passwords "
                      f"are refused. {AUTH_SETUP_HINT}")
    if cookie_fields[1].startswith("REPLACE_") or len(cookie_fields[1]) < 16:
        return None, f"`cookie.key` must be a long random secret. {AUTH_SETUP_HINT}"
    return config, None


# --- Login page look: centered "USER LOGIN" card. Reuses the terminal palette
# above (#0a0e0a page, #0d1310 panels, #00ff41 accent). Injected only while
# the login page is showing, so the dashboard CSS is untouched.
LOGIN_CSS = """
<style>
    .st-key-login_card {
        max-width: 460px;
        margin: 6vh auto 0 auto;
        padding: 0 16px;
        gap: 0.9rem;
    }
    .login-avatar {
        width: 112px; height: 112px;
        margin: 0 auto;
        border-radius: 50%;
        background: #0d1310;
        border: 2px solid #00ff41aa;
        box-shadow: 0 0 0 6px #0a0e0a, 0 0 0 8px #00ff4133, 0 0 18px #00ff4133;
        display: flex; align-items: center; justify-content: center;
    }
    .login-title {
        text-align: center;
        margin: 22px 0 2px 0;
        font-size: 1.7rem;
        font-weight: 400;
        letter-spacing: 6px;
        color: #00ff41;
        text-shadow: 0 0 8px #00ff4155;
    }
    .login-subtitle {
        text-align: center;
        font-size: 0.72rem;
        letter-spacing: 1px;
        color: #00cc33aa;
        margin-bottom: 8px;
    }

    /* icon + pill input rows */
    .st-key-login_card [class*="st-key-login_row_"] {
        flex-wrap: nowrap !important;
        align-items: center;
        gap: 14px;
    }
    .st-key-login_card [class*="st-key-login_row_"] > div:first-child {
        flex: 0 0 30px !important;
        width: 30px !important;
    }
    .st-key-login_card [class*="st-key-login_row_"] > div:last-child {
        flex: 1 1 auto !important;
        min-width: 0;
    }
    .st-key-login_card [class*="st-key-login_row_"] [data-testid="stMarkdownContainer"],
    .st-key-login_options [data-testid="stMarkdownContainer"] { margin-bottom: 0 !important; }
    .login-icon { display: flex; align-items: center; justify-content: center; }
    .login-icon svg { stroke: #00cc33aa; transition: stroke 0.15s; }
    .st-key-login_card [class*="st-key-login_row_"]:focus-within .login-icon svg { stroke: #00ff41; }

    .st-key-login_card [data-testid="stTextInputRootElement"] {
        border-radius: 999px !important;
        background: #0d1310 !important;
        border: 1px solid #00ff4155 !important;
        box-shadow: 0 0 0 4px #0a0e0a, 0 0 0 5px #00ff4122;
        padding: 0 10px;
        transition: border-color 0.15s, box-shadow 0.15s;
    }
    .st-key-login_card [data-testid="stTextInputRootElement"]:hover { border-color: #00ff4199 !important; }
    .st-key-login_card [data-testid="stTextInputRootElement"]:focus-within {
        border-color: #00ff41 !important;
        box-shadow: 0 0 0 4px #0a0e0a, 0 0 0 5px #00ff4166, 0 0 12px #00ff4133;
    }
    .st-key-login_card [data-testid="stTextInputField"],
    .st-key-login_card [data-testid="stTextInputRootElement"] button {
        background: transparent !important;
    }
    .st-key-login_card [data-testid="stTextInputField"] {
        color: #00ff41 !important;
        caret-color: #00ff41;
        height: 46px;
        padding-left: 10px;
    }
    .st-key-login_card [data-testid="stTextInputField"]::placeholder { color: #00cc33aa; opacity: 1; }
    .st-key-login_card [data-testid="InputInstructions"] { display: none; }

    /* remember me / forgot password row and LOGIN button sit under the inputs */
    .st-key-login_options, .st-key-login_submit, .st-key-login_status { padding-left: 44px; }
    .st-key-login_options {
        flex-wrap: wrap !important;
        justify-content: space-between !important;
        align-items: center;
        row-gap: 6px;
    }
    .st-key-login_options [data-testid="stCheckbox"] p { color: #00cc33aa; font-size: 0.85rem; }
    .st-key-login_options [data-testid="stCheckbox"] label > div:first-of-type {
        border-radius: 50% !important;
        border-color: #00ff4199 !important;
    }
    .st-key-login_options [data-testid="stCheckbox"] label[data-selected="true"] > div:first-of-type {
        background-color: #00ff41 !important;
        border-color: #00ff41 !important;
    }
    .st-key-login_options [data-testid="stCheckbox"] label[data-selected="true"] > div:first-of-type svg {
        stroke: #0a0e0a;
    }
    .login-forgot { position: relative; font-size: 0.85rem; margin: 0; }
    .login-forgot summary {
        list-style: none;
        cursor: pointer;
        font-style: italic;
        color: #00ff41;
    }
    .login-forgot summary::-webkit-details-marker { display: none; }
    .login-forgot summary:hover, .login-forgot summary:focus-visible { text-decoration: underline; }
    .login-forgot .login-forgot-note {
        position: absolute; right: 0; top: 1.8em; z-index: 10;
        width: min(300px, 80vw);
        background: #0d1310;
        border: 1px solid #00ff4155;
        border-radius: 8px;
        padding: 10px 12px;
        color: #00cc33;
        font-size: 0.78rem;
        line-height: 1.45;
    }

    .st-key-login_card [data-testid="stFormSubmitButton"] button {
        width: 100%;
        min-height: 50px;
        border-radius: 14px;
        background: #00ff41;
        border: 1px solid #00ff41;
        color: #0a0e0a;
        letter-spacing: 4px;
        transition: background 0.15s, box-shadow 0.15s, transform 0.05s;
    }
    .st-key-login_card [data-testid="stFormSubmitButton"] button p { font-size: 1.05rem; font-weight: 700; }
    .st-key-login_card [data-testid="stFormSubmitButton"] button:hover {
        background: #33ff66; border-color: #33ff66; color: #0a0e0a;
        box-shadow: 0 0 14px #00ff4166;
    }
    .st-key-login_card [data-testid="stFormSubmitButton"] button:focus-visible {
        outline: 2px solid #00ff41; outline-offset: 3px; color: #0a0e0a;
    }
    .st-key-login_card [data-testid="stFormSubmitButton"] button:active {
        background: #00cc33; border-color: #00cc33; transform: translateY(1px);
    }

    @media (max-width: 480px) {
        .st-key-login_card { margin-top: 2vh; padding: 0 4px; }
        .login-avatar { width: 92px; height: 92px; }
        .login-title { font-size: 1.35rem; letter-spacing: 4px; }
        .st-key-login_options, .st-key-login_submit, .st-key-login_status { padding-left: 0; }
    }
</style>
"""

_SVG = ('<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke-width="1.4" '
        'stroke-linecap="round" stroke-linejoin="round">{body}</svg>')
LOGIN_ICONS = {
    "employee_id": _SVG.format(s=26, body='<rect x="3" y="5" width="18" height="14" rx="2"/>'
                                          '<circle cx="9" cy="11" r="2"/><path d="M6 16c.6-1.6 1.7-2.4 3-2.4s2.4.8 3 2.4"/>'
                                          '<path d="M14.5 10h3.5M14.5 13.5h3.5"/>'),
    "username": _SVG.format(s=26, body='<circle cx="12" cy="8" r="4"/><path d="M4 21c0-4.4 3.6-7 8-7s8 2.6 8 7"/>'),
    "password": _SVG.format(s=26, body='<rect x="5" y="10" width="14" height="11" rx="2"/>'
                                       '<path d="M8 10V7a4 4 0 0 1 8 0v3"/><circle cx="12" cy="15" r="1.3"/><path d="M12 16.3V18"/>'),
}
LOGIN_AVATAR = ('<svg width="58" height="58" viewBox="0 0 24 24" fill="#00ff41">'
                '<circle cx="12" cy="7.5" r="4.5"/><path d="M3 21.5c0-5 4-8 9-8s9 3 9 8z"/></svg>')
LOGIN_FORGOT_NOTE = ("Passwords are managed by your Edge-IDPS administrator. Ask them to issue a new "
                     "one with <code>generate_password_hash.py</code> and update "
                     "<code>auth_config.yaml</code>.")


def _login_field(key, label, **kwargs):
    with st.container(horizontal=True, vertical_alignment="center", key=f"login_row_{key}"):
        st.markdown(f'<div class="login-icon">{LOGIN_ICONS[key]}</div>', unsafe_allow_html=True)
        return st.text_input(label, placeholder=label, label_visibility="collapsed",
                             autocomplete="off", key=f"login_{key}", **kwargs)


def _login_form(authenticator):
    """
    Renders the USER LOGIN page unless already signed in (session or cookie).

    Credentials are checked by streamlit-authenticator exactly as its own
    login() widget does — this only replaces the widget's layout. Employee ID
    is required-but-unchecked: it is not part of auth_config.yaml and is never
    stored. "Remember me" decides whether the library's re-login cookie is set;
    unticked, the sign-in lasts only for this browser session.
    """
    # Cookie re-login, done by the library's own login() ("unrendered" draws nothing)
    try:
        authenticator.login(location="unrendered", key="edge_idps_login")
    except stauth.LoginError as e:
        st.error(f"Login error: {e}")
    if st.session_state.get("authentication_status") is True:
        return

    st.markdown(LOGIN_CSS, unsafe_allow_html=True)
    slot = st.empty()
    with slot.container(key="login_card"):
        st.markdown(f'<div class="login-avatar">{LOGIN_AVATAR}</div>'
                    '<div class="login-title">USER LOGIN</div>'
                    '<div class="login-subtitle">EDGE-IDPS SOC TERMINAL · authentication required</div>',
                    unsafe_allow_html=True)
        with st.form("edge_idps_login", border=False):
            employee_id = _login_field("employee_id", "Employee ID")
            username = _login_field("username", "Username")
            password = _login_field("password", "Password", type="password")
            with st.container(horizontal=True, vertical_alignment="center", key="login_options"):
                remember = st.checkbox("Remember me", value=True, key="login_remember")
                st.markdown(f'<details class="login-forgot"><summary>Forgot Password?</summary>'
                            f'<div class="login-forgot-note">{LOGIN_FORGOT_NOTE}</div></details>',
                            unsafe_allow_html=True)
            with st.container(key="login_submit"):
                submitted = st.form_submit_button("LOGIN", width="stretch")

        with st.container(key="login_status"):
            if submitted:
                st.session_state.pop("login_notice", None)
                missing = [name for name, value in (("Employee ID", employee_id),
                                                    ("Username", username),
                                                    ("Password", password)) if not value.strip()]
                if missing:
                    st.error(f"{', '.join(missing)} required.")
                    return
                try:
                    authenticator.authentication_controller.login(username, password)
                except stauth.LoginError as e:
                    st.error(f"Login error: {e}")
                    return
            status = st.session_state.get("authentication_status")
            if status is not True and st.session_state.get("login_notice"):
                st.success(st.session_state["login_notice"])
            elif status is False:
                st.error("ACCESS DENIED — incorrect username or password.")

    if st.session_state.get("authentication_status") is True:
        slot.empty()  # signed in on this run: drop the card, the dashboard renders below
        if remember:
            authenticator.cookie_controller.set_cookie()


def _on_logout(_details):
    # Don't carry a previous user's live-capture explanation into the next login
    st.session_state.pop("xai_result", None)
    st.session_state.pop("tour_state", None)
    # Kept in session state, not passed in: deleting the cookie reruns the
    # script, which would otherwise drop the notice straight away
    st.session_state["login_notice"] = "Logged out."


def _pause_auto_refresh():
    # Runs before the rerun, so the toggle below renders already paused: an
    # auto-refresh rerun would otherwise cut the agent off mid-sentence.
    st.session_state["auto_refresh"] = False


def _start_tour():
    _pause_auto_refresh()
    st.session_state["tour_state"] = {}


# Placeholder at the top of the page so gate messages sit above the login form
gate_header = st.container()

auth_config, auth_error = _load_auth_config()
if auth_error:
    with gate_header:
        st.markdown("# 🛡 EDGE-IDPS :: ACCESS CONTROL")
        st.error(auth_error)
    st.stop()

# Credentials are passed as a dict, not a file path: given a path, the
# library rewrites the YAML on every login. auto_hash=False because
# _load_auth_config() has already refused anything that isn't a bcrypt hash.
authenticator = stauth.Authenticate(
    auth_config["credentials"],
    str(auth_config["cookie"]["name"]),
    str(auth_config["cookie"]["key"]),
    float(auth_config["cookie"]["expiry_days"]),
    auto_hash=False,
)

_login_form(authenticator)  # renders nothing if already signed in (session or cookie)

if st.session_state.get("authentication_status") is True:
    st.sidebar.caption(f"`signed in as {st.session_state.get('username')}`")
    authenticator.logout("Logout", "sidebar", key="edge_idps_logout", callback=_on_logout)
    if not st.session_state.get("authentication_status"):
        # logout() does not rerun the script, so without this the dashboard
        # below would still render for this one run. Show the login form now
        # and stop instead.
        _login_form(authenticator)

if st.session_state.get("authentication_status") is not True:
    st.stop()  # the login card above shows any denied / logged-out message


# Reduced motion: the browser's prefers-reduced-motion setting (an OS
# accessibility option). CSS media queries stop the XAI agent's animations on
# their own, but its typewriter is driven from Python, so Python needs the
# setting too. This invisible component reports it — once on load, and again
# if it changes — and only when it differs from what Python already has, so
# it triggers at most one rerun per change.
_motion_probe = st.components.v2.component("reduced_motion_probe", js="""
export default function({ data, setStateValue }) {
  const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
  const report = () => { if (mq.matches !== data) setStateValue("reduce", mq.matches); };
  report();
  mq.addEventListener("change", report);
  return () => mq.removeEventListener("change", report);
}
""")
with st.container(key="motion_probe"):
    reduce_motion = bool(_motion_probe(
        key="reduced_motion_probe", data=st.session_state.get("reduce_motion"),
        on_reduce_change=lambda: None,
    ).reduce)
st.session_state["reduce_motion"] = reduce_motion


# =============================================================================
# SIDEBAR — interface controls
# =============================================================================
with st.sidebar:
    st.markdown("### ⚙ CONTROL PANEL")

    # Default (on) set through session state rather than value=, because the
    # tour button's callback also sets this key — Streamlit warns when both are used
    st.session_state.setdefault("auto_refresh", True)
    auto_refresh = st.toggle("Live auto-refresh", key="auto_refresh",
                              help="Pause this to freeze the view while explaining something")
    refresh_interval_ms = st.select_slider(
        "Refresh interval", options=[500, 1000, 2000, 5000], value=1000,
        format_func=lambda v: f"{v/1000:.1f}s",
    )

    st.markdown("---")

    window_seconds = st.slider(
        "Analysis window (seconds)", min_value=10, max_value=300, value=60, step=10,
        help="How far back the metrics, charts and feed look",
    )

    st.markdown("---")
    st.markdown("**Filter by class**")
    # Populated after data loads (see below); placeholder container here
    class_filter_container = st.container()

    st.markdown("---")
    show_insufficient = st.checkbox(
        "Show INSUFFICIENT_EVIDENCE rows in feed", value=False,
        help="These are flows with too few packets to classify — excluded from metrics regardless",
    )

    st.markdown("---")
    st.markdown("**Explainable AI**")
    xai_interface = st.text_input("Capture interface", value="", placeholder="default",
                                  help="e.g. Wi-Fi — blank uses Scapy's default")
    xai_target_ip = st.text_input(
        "Target IP (optional)", value="", placeholder="any",
        help="Lab/demo use: only explain flows to or from this address, e.g. the "
             "simulated victim IP you pointed scapy_attacker.py at. Leave blank "
             "to consider all traffic.",
    )
    xai_target_ip = xai_target_ip.strip()
    xai_target_ip_valid = _valid_ip(xai_target_ip)
    if not xai_target_ip_valid:
        st.error(f"`{xai_target_ip}` is not a valid IP address.")
    xai_duration = st.slider("Capture window (seconds)", min_value=3, max_value=15, value=8)
    # Disabled while auto-refresh is on: an autorefresh rerun would abort the
    # capture mid-way. With auto-refresh paused there is no rerun timer at all.
    explain_clicked = st.button(
        "Explain a live flow", disabled=auto_refresh or not xai_target_ip_valid,
        width="stretch",
        help="Pause live auto-refresh first. Captures for the whole window, then "
             "explains the flow with the most packets (to/from the target IP, if "
             "set) — so an active attack wins over background traffic. Needs the "
             "dashboard to be running as Administrator.",
    )
    st.caption("Explains the busiest flow in the window"
               + (f" to/from `{xai_target_ip}`." if xai_target_ip else "."))
    xai_replay_classes = _replay_class_options()
    xai_replay_class = st.selectbox(
        "Test-row class", xai_replay_classes,
        index=xai_replay_classes.index("Mirai") if "Mirai" in xai_replay_classes else 0,
        help="Class of the held-out CICIoT2023 test row to explain.",
    )
    replay_clicked = st.button(
        "Explain a replayed test row", width="stretch", on_click=_pause_auto_refresh,
        help="No capture needed: picks a random held-out test row of the chosen "
             "class and explains the CNN's prediction for it, the same way as a "
             "live flow. For attacks this lab can't generate live, like Mirai. "
             "Pauses live auto-refresh while the agent talks.",
    )
    st.caption("New to this dashboard?")
    st.button(
        "What am I looking at?", icon=":material/school:", type="tertiary",
        on_click=_start_tour, key="tour_button",
        help="The XAI agent walks you through what each panel means, in plain "
             "language. Pauses live auto-refresh while it talks.",
    )

    st.markdown("---")
    st.caption("Firewall mode, model version and dataset info are read "
               "directly from the telemetry log — nothing here is editable "
               "from this dashboard (read-only by design).")

if auto_refresh:
    st_autorefresh(interval=refresh_interval_ms, key="dashboard_refresh")

st.markdown("# 🛡 EDGE-IDPS :: SOC TERMINAL")
st.caption("`root@edge-idps:~#` real-time intrusion detection & prevention — read-only telemetry view")


# =============================================================================
# DASHBOARD TOUR — "What am I looking at?" in the sidebar. A fixed, written
# walkthrough of the panels below (dashboard/xai_panel.py DASHBOARD_TOUR),
# told by the XAI agent. Display only: reads no data and captures nothing, so
# it sits above the data loading and works with empty telemetry too.
# =============================================================================
# The agent "visits" each panel: tour_anchor() leaves an empty slot just above
# a panel, and play_tour() — run once the panels are drawn, at the end of the
# script — fills those slots one step at a time. Streamlit redraws the page
# on every rerun rather than keeping a live DOM, so the agent can't glide
# across the screen; instead each step appears beside its panel (with a fade
# and a scroll to it) while the previous step dims. Slots sit in the normal
# page flow, so they stack with their panels on narrow screens.
tour_slots = {}


def tour_anchor(name):
    if "tour_state" in st.session_state:
        tour_slots[name] = st.container()


def play_tour():
    if "tour_state" in st.session_state:
        from xai_panel import render_dashboard_tour  # lazy: plotly loads only when shown
        render_dashboard_tour(st.session_state["tour_state"], tour_slots,
                              reduce_motion=reduce_motion)


if "tour_state" in st.session_state:
    st.markdown('<div class="term-header">[ XAI AGENT :: WHAT AM I LOOKING AT? ]</div>',
                unsafe_allow_html=True)
    tour_anchor("intro")
    tour_anchor("sidebar")
    if st.button("Close tour", key="tour_close"):
        del st.session_state["tour_state"]
        st.rerun()
    st.markdown("---")


# =============================================================================
# EXPLAIN A LIVE FLOW — the ONE deliberate exception to "the dashboard never
# captures, only reads". Telemetry rows can't be explained after the fact (the
# CSV stores predictions, not the 20-feature vectors), so explaining a
# prediction means capturing and classifying a fresh flow right here.
# Scoped so it can never run on its own:
#   - only inside `if explain_clicked:`, and st.button is True for exactly the
#     one rerun caused by the click — auto-refresh reruns see False;
#   - the button is disabled while auto-refresh is on, so no rerun timer can
#     interrupt the capture;
#   - torch/captum/scapy are imported lazily on click, so the normal polling
#     path loads none of them;
#   - the chart is drawn in memory: this never writes to logs/, and never
#     touches live_stream_output.csv or active_blocklist.txt.
# This is sniff-only — it never sends packets — and runs alongside (not
# instead of) engine/stream_pipeline.py.
# Placed above the data loading so it still works when telemetry is empty.
# =============================================================================
@st.cache_resource(show_spinner="Loading CNN for explanation ...")
def _load_explainer():
    import sys
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "engine"))
    import explain_prediction
    return explain_prediction, explain_prediction.load_inference_engine()


if explain_clicked:
    status_lines = []
    scope = f" to/from {xai_target_ip}" if xai_target_ip else ""
    with st.spinner(f"Capturing for {xai_duration}s, then explaining the largest flow{scope} ..."):
        try:
            xp, xai_engine = _load_explainer()
            explanation = xp.capture_and_explain(
                xai_engine, interface=xai_interface.strip() or None,
                duration=xai_duration, on_status=status_lines.append,
                target_ip=xai_target_ip or None,
            )
            st.session_state["xai_result"] = {"explanation": explanation, "status": status_lines}
        except Exception as e:
            st.session_state["xai_result"] = {"explanation": None,
                                              "status": status_lines + [f"ERROR: {e}"]}

# Held-out test row: same model + Integrated Gradients path, no capture. Reads
# data/processed/*_test_5class.npy only (never train rows) and writes nothing.
if replay_clicked:
    with st.spinner(f"Explaining a held-out {xai_replay_class} test row ..."):
        try:
            xp, xai_engine = _load_explainer()
            explanation = xp.explain_test_row(xai_engine, xai_replay_class)
            st.session_state["xai_result"] = {"explanation": explanation, "status": []}
        except Exception as e:
            st.session_state["xai_result"] = {"explanation": None, "status": [f"ERROR: {e}"]}

if "xai_result" in st.session_state:
    xai = st.session_state["xai_result"]
    st.markdown('<div class="term-header">[ EXPLAINABLE AI :: WHY THIS PREDICTION? ]</div>',
                unsafe_allow_html=True)
    for line in xai["status"]:
        st.caption(line)
    explanation = xai["explanation"]
    if explanation is None:
        st.warning("No flow could be explained. See the messages above.")
    else:
        xp, _ = _load_explainer()
        import matplotlib.pyplot as plt
        from xai_panel import render_xai_explanation  # lazy: plotly loads only when shown
        verdict = (f"→ {class_tag(explanation.predicted_class)} "
                   f"confidence **{explanation.confidence:.1%}**")
        if explanation.source == "test_row":
            st.markdown(f"Held-out test row `#{explanation.flow_id[1]}` "
                        f"(true label {class_tag(explanation.true_class)}) {verdict}",
                        unsafe_allow_html=True)
            if explanation.predicted_class != explanation.true_class:
                st.info(f"The model got this row wrong: it is really "
                        f"{explanation.true_class}. The explanation below shows why it "
                        f"leaned {explanation.predicted_class} instead.")
        else:
            st.markdown(f"Flow `{explanation.flow_id}` ({explanation.packet_count} packets) "
                        f"{verdict}", unsafe_allow_html=True)
        render_xai_explanation(explanation, state=xai, reduce_motion=reduce_motion)
        with st.expander("Full attribution detail (toward and away from the prediction)"):
            xai_col1, xai_col2 = st.columns([3, 2])
            with xai_col1:
                fig = xp.build_chart(explanation)
                st.pyplot(fig)
                plt.close(fig)
            with xai_col2:
                st.dataframe(pd.DataFrame(
                    [{"feature": name, "attribution": round(attr, 4), "scaled value": round(val, 3),
                      "direction": xp.direction_label(attr, explanation.predicted_class)}
                     for name, attr, val in explanation.attributions]),
                    width="stretch", hide_index=True, height=560)
        st.caption("Integrated Gradients vs. an average-flow baseline (all-zero scaled vector). "
                   "Scaled value = standard deviations from the training mean. "
                   + ("Explains this held-out dataset row only — not live traffic or telemetry rows."
                      if explanation.source == "test_row" else
                      "Explains this fresh capture only — not historical telemetry rows."))
    if st.button("Clear explanation"):
        del st.session_state["xai_result"]
        st.rerun()
    st.markdown("---")


# =============================================================================
# DATA LOADING — unchanged logic from the working version
# =============================================================================
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
    play_tour()  # the panels below never draw, so every step plays in the tour header
    st.stop()

now = pd.Timestamp.now()
recent_cutoff = now - pd.Timedelta(seconds=window_seconds)
recent_df = telemetry_df[telemetry_df["timestamp"] >= recent_cutoff]

# Stale-telemetry guard. load_telemetry() only returns empty when the CSV is
# missing (handled above). The silent trap is a non-empty file whose newest row
# is older than the analysis window: every metric below then reads a confident
# zero that looks like "no threats" but really means "no fresh data". Surface
# the age and point at the writer (stream_pipeline.py is the only producer).
latest_ts = telemetry_df["timestamp"].max()
if recent_df.empty and pd.notna(latest_ts):
    age_secs = (now - latest_ts).total_seconds()
    if age_secs < 90:
        age_str = f"{int(age_secs)}s"
    elif age_secs < 5400:
        age_str = f"{int(age_secs // 60)}m"
    else:
        age_str = f"{age_secs / 3600:.1f}h"
    st.warning(
        f"No telemetry in the last {window_seconds}s, so every metric below reads "
        f"zero — this is stale data, not an all-clear. The newest row is **{age_str} "
        f"old** (last seen {latest_ts.strftime('%Y-%m-%d %H:%M:%S')}). The dashboard "
        f"only displays what `engine/stream_pipeline.py` writes; start it in a "
        f"separate terminal to see live flows:\n\n"
        f"`python engine\\stream_pipeline.py --interface \"Wi-Fi\" --duration 300`\n\n"
        f"(Or widen the **Analysis window** slider to include older rows.)"
    )

# A real threat is a classified, non-Benign flow — INSUFFICIENT_EVIDENCE is
# neither a threat nor a clean pass, it's "too few packets to classify".
classified_df = recent_df[recent_df["predicted_class"] != "INSUFFICIENT_EVIDENCE"]

# Populate the sidebar class filter now that we know what classes exist
available_classes = sorted(classified_df["predicted_class"].unique().tolist())
with class_filter_container:
    selected_classes = st.multiselect(
        "Classes shown in charts/feed", options=available_classes,
        default=available_classes, label_visibility="collapsed",
    )
filtered_df = classified_df[classified_df["predicted_class"].isin(selected_classes)] if selected_classes else classified_df


# =============================================================================
# TOP METRICS
# =============================================================================
tour_anchor("metrics")
col1, col2, col3, col4, col5 = st.columns(5)

active_threats = (classified_df["predicted_class"] != "Benign").sum()
col1.metric("ACTIVE THREATS", int(active_threats))

flows_per_sec = len(recent_df) / window_seconds
col2.metric("FLOWS/SEC", f"{flows_per_sec:.2f}")

avg_confidence = classified_df["confidence"].mean() if not classified_df.empty else 0.0
col3.metric("AVG CONFIDENCE", f"{avg_confidence:.1%}" if pd.notna(avg_confidence) else "N/A")

blocked_count = blocklist_df["blocked_ip"].nunique() if not blocklist_df.empty else 0
col4.metric("IPs BLOCKED", blocked_count)

model_version = telemetry_df["model_version"].iloc[-1] if not telemetry_df.empty else "N/A"
col5.metric("MODEL", model_version)

insufficient_count = (recent_df["predicted_class"] == "INSUFFICIENT_EVIDENCE").sum()
st.caption(f"[i] {insufficient_count} of {len(recent_df)} flows in the last {window_seconds}s were "
           f"INSUFFICIENT_EVIDENCE (too few packets to classify) — excluded from metrics above.")

st.markdown("---")

# =============================================================================
# CHARTS
# =============================================================================
tour_anchor("charts")
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown('<div class="term-header">[ ATTACK CATEGORY DISTRIBUTION ]</div>', unsafe_allow_html=True)
    if not filtered_df.empty:
        st.bar_chart(filtered_df["predicted_class"].value_counts(), color="#00ff41")
    else:
        st.write("no data in current window")

with chart_col2:
    st.markdown('<div class="term-header">[ RISK SCORE DISTRIBUTION ]</div>', unsafe_allow_html=True)
    if not filtered_df.empty:
        binned = pd.cut(filtered_df["risk_score"], bins=10)
        counts = binned.value_counts().sort_index()
        counts.index = [f"{iv.left:.1f}\u2013{iv.right:.1f}" for iv in counts.index]
        st.bar_chart(counts, color="#ff8c00")
    else:
        st.write("no classified flows in current window")

st.markdown('<div class="term-header">[ RISK SCORE TIMELINE ]</div>', unsafe_allow_html=True)
if not filtered_df.empty:
    timeline = filtered_df.sort_values("timestamp").set_index("timestamp")["risk_score"]
    st.line_chart(timeline, color="#00ff41")
else:
    st.write("no data in current window")

st.markdown("---")

# =============================================================================
# LIVE FEED — terminal-style scrolling event log
# =============================================================================
tour_anchor("feed")
st.markdown('<div class="term-header">[ LIVE EVENT FEED ]</div>', unsafe_allow_html=True)

feed_source = recent_df if show_insufficient else classified_df
feed_source = feed_source[feed_source["predicted_class"].isin(
    selected_classes + (["INSUFFICIENT_EVIDENCE"] if show_insufficient else [])
)] if selected_classes else feed_source
feed_rows = feed_source.sort_values("timestamp", ascending=False).head(25)

if feed_rows.empty:
    st.markdown('<div class="term-box">-- no events --</div>', unsafe_allow_html=True)
else:
    lines = []
    for _, row in feed_rows.iterrows():
        ts = row["timestamp"].strftime("%H:%M:%S") if pd.notna(row["timestamp"]) else "--:--:--"
        conf = f"{row['confidence']:.2f}" if pd.notna(row.get("confidence")) else "--"
        risk = f"{row['risk_score']:.2f}" if pd.notna(row.get("risk_score")) else "--"
        action_html = action_tag(row.get("action", "")) if pd.notna(row.get("action")) else ""
        class_html = class_tag(row["predicted_class"])
        lines.append(
            f'<div class="feed-line">'
            f'<span style="color:#00cc33">{ts}</span> '
            f'<span style="color:#fff">{row["source_ip"]:<15}</span> '
            f'{class_html} '
            f'<span style="color:#888">conf={conf}</span> '
            f'<span style="color:#888">risk={risk}</span> '
            f'{action_html}'
            f'</div>'
        )
    st.markdown(f'<div class="term-box">{"".join(lines)}</div>', unsafe_allow_html=True)

st.markdown("---")

# =============================================================================
# TABLES
# =============================================================================
table_col1, table_col2 = st.columns(2)

with table_col1:
    st.markdown('<div class="term-header">[ ACTIVE FIREWALL BLOCKLIST ]</div>', unsafe_allow_html=True)
    if not blocklist_df.empty:
        st.dataframe(blocklist_df.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)
    else:
        st.write("no IPs currently blocked")

with table_col2:
    st.markdown('<div class="term-header">[ RECENT TELEMETRY (RAW) ]</div>', unsafe_allow_html=True)
    recent_events = telemetry_df.sort_values("timestamp", ascending=False).head(20)
    st.dataframe(recent_events, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("`[read-only]` polls logs/live_stream_output.csv and logs/active_blocklist.txt "
           f"every {refresh_interval_ms/1000:.1f}s when auto-refresh is on. Never writes to either file.")

# Last, so every panel the tour talks about is already on the page
play_tour()