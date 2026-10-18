"""
dashboard/xai_panel.py — the "XAI agent": an animated robot that talks the
user through things in chat bubbles.

  - render_xai_explanation(): explains one Explanation that
    engine/explain_prediction.py already produced (Integrated Gradients
    attributions) and changes nothing about how it was computed. Called from
    the "Explain a live flow" block in app.py.
  - render_dashboard_tour(): a fixed, written walkthrough of the dashboard's
    panels for first-time visitors. Called from the "What am I looking at?"
    button in app.py.

The pure helpers (toward_shares, build_summary) have no Streamlit calls, so
they can be unit-tested without a running app.
"""

import time

import plotly.graph_objects as go
import streamlit as st



def robot_avatar_svg(line="#00ff41", face="#0d1310", eye="#16261a"):
    """
    Animated robot head as an SVG string (big binocular eyes, antenna).

    st.chat_message(avatar=...) accepts a raw "<svg ...>" string and shows it as
    an <img> data URI. Browsers run CSS @keyframes inside an SVG loaded that
    way (scripts are blocked, which this doesn't need), so the animation lives
    in the SVG's own <style>: an occasional blink, a slow bob, and an antenna
    twitch that is still most of the time. All motion stops under
    prefers-reduced-motion.
    """
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48">
<style>
  .bob {{ animation: bob 4s ease-in-out infinite alternate; }}
  .eyes {{ transform-box: fill-box; transform-origin: center;
           animation: blink 5.5s ease-in-out infinite; }}
  .ant {{ transform-box: view-box; transform-origin: 24px 11px;
          animation: twitch 7s ease-in-out infinite; }}
  @keyframes bob {{ from {{ transform: translateY(0.6px); }} to {{ transform: translateY(-1.2px); }} }}
  @keyframes blink {{ 0%, 88%, 100% {{ transform: scaleY(1); }}
                      91% {{ transform: scaleY(0.08); }} 94% {{ transform: scaleY(1); }} }}
  @keyframes twitch {{ 0%, 70%, 100% {{ transform: rotate(0deg); }}
                       75% {{ transform: rotate(-12deg); }} 82% {{ transform: rotate(9deg); }}
                       88% {{ transform: rotate(0deg); }} }}
  @media (prefers-reduced-motion: reduce) {{ .bob, .eyes, .ant {{ animation: none; }} }}
</style>
<g class="bob">
  <g class="ant">
    <line x1="24" y1="11" x2="24" y2="5" stroke="{line}" stroke-width="1.6" stroke-linecap="round"/>
    <circle cx="24" cy="4" r="2.2" fill="{line}"/>
  </g>
  <rect x="7" y="11" width="34" height="22" rx="8" fill="{face}" stroke="{line}" stroke-width="1.6"/>
  <g class="eyes">
    <rect x="11" y="15" width="11.5" height="14" rx="5.5" fill="{eye}" stroke="{line}" stroke-width="1.3"/>
    <rect x="25.5" y="15" width="11.5" height="14" rx="5.5" fill="{eye}" stroke="{line}" stroke-width="1.3"/>
    <circle cx="17.5" cy="22.5" r="3.1" fill="{line}"/>
    <circle cx="30.5" cy="22.5" r="3.1" fill="{line}"/>
    <circle cx="18.6" cy="21.3" r="0.9" fill="{face}"/>
    <circle cx="31.6" cy="21.3" r="0.9" fill="{face}"/>
  </g>
  <rect x="15" y="35" width="18" height="8" rx="3" fill="{face}" stroke="{line}" stroke-width="1.4"/>
  <line x1="19" y1="39" x2="29" y2="39" stroke="{line}" stroke-width="1.2" stroke-linecap="round"/>
</g>
</svg>"""


# The default avatar. Anything st.chat_message accepts works in its place — an
# emoji ("🤖"), an image path or URL, or another SVG string — via this
# constant or the avatar= parameter of the render functions.
XAI_AGENT_AVATAR = robot_avatar_svg()

# Mirrors explain_prediction.NEGLIGIBLE_ATTRIBUTION. Not imported, so this module
# never pulls in torch/captum (explain_prediction imports torch at module level).
NEGLIGIBLE_ATTRIBUTION = 1e-3

# Slices smaller than this share of the toward-class push fold into "Other"
MIN_SLICE_SHARE = 0.03
MAX_NAMED_SLICES = 5

# Slice colors, theme green first. Validated as a categorical set against the
# dashboard surface #0a0e0a (dataviz validate_palette.js, dark mode): orange and
# yellow are deliberately kept apart. Slices are also separated by a surface-color
# gap and direct-labelled, so identity never rests on color alone.
SLICE_COLORS = ["#16a33a", "#3987e5", "#d95926", "#9085e9", "#c98500"]
OTHER_COLOR = "#5a5f5a"

# Typewriter pacing: ~1.5 s for the whole message, 15-30 ms per word
TYPEWRITER_TOTAL_S = 1.5
TYPEWRITER_MIN_DELAY_S = 0.015
TYPEWRITER_MAX_DELAY_S = 0.03

# "Typing…" dots shown before each message starts typing
TYPING_INDICATOR_S = 0.5

# Entrance animation for agent bubbles and the typing dots. Scoped to the
# "xai_agent_*" containers this module creates, so other chat elements and
# the rest of the dashboard are unaffected.
AGENT_CSS = """
<style>
    [class*="st-key-xai_agent_"] [data-testid="stChatMessage"] {
        animation: xai-enter 0.35s ease-out both;
    }
    @keyframes xai-enter {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .xai-typing { display: inline-flex; gap: 6px; padding: 6px 0; }
    .xai-typing span {
        width: 7px; height: 7px; border-radius: 50%;
        background: #00ff41;
        animation: xai-dot 1s ease-in-out infinite;
    }
    .xai-typing span:nth-child(2) { animation-delay: 0.15s; }
    .xai-typing span:nth-child(3) { animation-delay: 0.3s; }
    @keyframes xai-dot {
        0%, 100% { opacity: 0.25; transform: scale(0.85); }
        50%      { opacity: 1;    transform: scale(1); }
    }
    @media (prefers-reduced-motion: reduce) {
        [class*="st-key-xai_agent_"] [data-testid="stChatMessage"],
        .xai-typing span { animation: none; }
        .xai-typing span { opacity: 0.7; }
    }
</style>
"""
TYPING_HTML = ('<div class="xai-typing" role="status" aria-label="Agent is typing">'
               '<span></span><span></span><span></span></div>')


def toward_shares(attributions, min_share=MIN_SLICE_SHARE, max_named=MAX_NAMED_SLICES):
    """
    Each feature's share of the total push TOWARD the predicted class.

    attributions: Explanation.attributions — (name, attribution, scaled_value).
    Returns [(label, share, attribution, scaled_value or None)], largest first,
    with small or surplus contributors folded into one trailing "Other" slice.
    Empty if no feature pushed toward the class by a non-negligible amount.
    """
    positive = sorted(((n, a, v) for n, a, v in attributions if a >= NEGLIGIBLE_ATTRIBUTION),
                      key=lambda r: r[1], reverse=True)
    total = sum(a for _, a, _ in positive)
    if total <= 0:
        return []
    named, other_attr, other_count = [], 0.0, 0
    for name, attr, value in positive:
        # The top contributor is always named, however small, so a flow is never "all Other"
        if not named or (len(named) < max_named and attr / total >= min_share):
            named.append((name, attr / total, attr, value))
        else:
            other_attr += attr
            other_count += 1
    if other_count:
        named.append((f"Other ({other_count} features)", other_attr / total, other_attr, None))
    return named


def _describe_value(scaled_value):
    # scaled value = standard deviations from the training-set mean
    if scaled_value is None or abs(scaled_value) < 0.5:
        return "near-average"
    level = "unusually " if abs(scaled_value) >= 2 else ""
    return f"{level}{'high' if scaled_value > 0 else 'low'}"


def _feature_phrase(name, share, value):
    return f"{_describe_value(value)} {name} ({share:.0%})"


def build_summary(explanation, shares):
    """Plain-language message for the agent, built only from the given data."""
    e = explanation
    opening = f"I classified this flow as **{e.predicted_class}** with {e.confidence:.1%} confidence"
    named = [s for s in shares if s[3] is not None]  # "Other" is the slice with no value

    if not named:
        text = (f"{opening}, but no single feature stood out — every feature's attribution "
                f"was negligible, so this flow looked close to the average training flow.")
    elif named[0][1] >= 0.8:
        name, share, _, value = named[0]
        text = f"{opening}, almost entirely because of {_feature_phrase(name, share, value)}."
    else:
        top = [_feature_phrase(n, s, v) for n, s, _, v in named[:3]]
        reasons = top[0] if len(top) == 1 else ", ".join(top[:-1]) + f" and {top[-1]}"
        text = f"{opening}, mainly because of {reasons}."

    away = [(n, a) for n, a, _ in e.attributions if a <= -NEGLIGIBLE_ATTRIBUTION]
    if away:
        strongest = min(away, key=lambda r: r[1])[0]
        text += f" The strongest signal against {e.predicted_class} was {strongest}."
    if named:
        text += " The chart shows how the push toward that call splits across features."
    return text


def _typewriter(text):
    words = text.split(" ")
    delay = min(max(TYPEWRITER_TOTAL_S / max(len(words), 1), TYPEWRITER_MIN_DELAY_S),
                TYPEWRITER_MAX_DELAY_S)
    for i, word in enumerate(words):
        yield word if i == 0 else " " + word
        time.sleep(delay)


def build_donut(explanation, shares):
    colors = [OTHER_COLOR if value is None else SLICE_COLORS[i % len(SLICE_COLORS)]
              for i, (_, _, _, value) in enumerate(shares)]
    fig = go.Figure(go.Pie(
        labels=[s[0] for s in shares],
        values=[s[1] for s in shares],
        customdata=[[s[2]] for s in shares],
        hole=0.55,
        sort=False,
        direction="clockwise",
        marker=dict(colors=colors, line=dict(color="#0a0e0a", width=2)),
        textinfo="percent",
        textfont=dict(color="#0a0e0a", size=12),
        insidetextorientation="horizontal",
        hovertemplate="<b>%{label}</b><br>%{percent} of the push toward "
                      f"{explanation.predicted_class}<br>attribution %{{customdata[0]:+.4f}}"
                      "<extra></extra>",
    ))
    fig.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, Courier New, monospace", color="#c8d6c8", size=12),
        legend=dict(orientation="v", x=1.02, y=0.5, font=dict(color="#c8d6c8")),
        hoverlabel=dict(bgcolor="#0d1310", bordercolor="#00ff41", font=dict(color="#c8d6c8")),
        annotations=[dict(text=f"toward<br><b>{explanation.predicted_class}</b>",
                          showarrow=False, font=dict(color="#00ff41", size=14))],
    )
    return fig


def _say(text, animate):
    """One message: typing dots, then the typewriter — or the finished text at once."""
    slot = st.empty()
    if animate:
        slot.markdown(TYPING_HTML, unsafe_allow_html=True)
        time.sleep(TYPING_INDICATOR_S)
        slot.write_stream(_typewriter(text))
    else:
        slot.markdown(text)


def render_xai_explanation(explanation, state=None, avatar=XAI_AGENT_AVATAR, key="xai"):
    """
    Renders the agent chat bubble (typewriter text) with the contribution donut.

    state: a dict that persists across reruns (e.g. the session-state entry
    holding this explanation). The typing dots and typewriter play once per
    explanation; later reruns — auto-refresh, widget clicks — show the finished
    text instantly.
    """
    shares = toward_shares(explanation.attributions)
    text = build_summary(explanation, shares)
    st.markdown(AGENT_CSS, unsafe_allow_html=True)
    with st.container(key=f"xai_agent_{key}"), st.chat_message("assistant", avatar=avatar):
        st.caption("EDGE-IDPS XAI AGENT")
        animate = state is not None and not state.get("typed")
        _say(text, animate)
        if animate:
            state["typed"] = True
        if shares:
            st.plotly_chart(build_donut(explanation, shares), key=f"{key}_donut",
                            config={"displaylogo": False})


# --- Dashboard tour: fixed text describing the panels in app.py, in the order
# they appear on the page. Written for someone new to networking and AI.
# Keep in step with app.py if panels are added, renamed or removed.
DASHBOARD_TOUR = [
    "Hi! I'm the Edge-IDPS agent — let me show you around. This dashboard watches the "
    "network traffic going in and out of this computer. Traffic travels in small pieces "
    "called **packets**, and I group the packets exchanged between two devices into a "
    "**flow**: think of it as one conversation. An AI model reads every flow and labels it "
    "either **Benign** (normal) or as one of four kinds of attack: **DoS/DDoS** (flooding "
    "a device so it stops responding), **Reconnaissance** (scanning for weak spots), "
    "**Spoofing** (pretending to be a different device) or **Mirai** (malware that hijacks "
    "internet-connected gadgets).",

    "The row of numbers at the top is the quick summary, covering the last minute by "
    "default (you can change that window in the sidebar). **ACTIVE THREATS** counts the "
    "flows labelled as an attack. **FLOWS/SEC** is how many conversations per second I'm "
    "seeing; a sudden jump can mean a flood. **AVG CONFIDENCE** is how sure the AI was of "
    "its labels on average: 95% means very sure. **IPs BLOCKED** is how many devices the "
    "firewall has shut out (an IP address is a device's address on the network), and "
    "**MODEL** is which version of the AI is running. Flows with too few packets to judge "
    "are counted separately as INSUFFICIENT_EVIDENCE and left out of these numbers.",

    "The charts underneath show the bigger picture. **Attack category distribution** shows "
    "how many flows got each label. Every flow also gets a **risk score** from 0 to 10 that "
    "combines how sure the AI was with how often the same sender has looked suspicious in "
    "the last minute: one shaky alert scores low, a confident and repeated one scores high. "
    "**Risk score distribution** groups the flows by that score, and the **risk score "
    "timeline** shows it over time, so spikes are the moments worth a closer look.",

    "The **live event feed** lists the newest flows, one per line: the time, the sender's "
    "address, the label, the AI's confidence, the risk score and what the system did about "
    "it. Those actions get stronger as risk rises: **LOG_ONLY** (just noted), **THROTTLE** "
    "(flagged to be slowed down), **DROP_PACKET** (its traffic is thrown away) and "
    "**BLOCK_IP** (the sender is shut out). Senders that were dropped or blocked appear in "
    "the **firewall blocklist** table, and **recent telemetry** is the raw log this whole "
    "page is built from.",

    "Last, the sidebar on the left. You can pause the live updates, change the time window, "
    "or filter the charts by label. **Explain a live flow** asks me to catch one fresh "
    "conversation and explain *why* the AI labelled it the way it did: I'll name the "
    "measurements that mattered most, and a donut chart shows how much each one pushed the "
    "AI toward its answer. Feel free to click around — this page only displays what the "
    "system is doing; nothing here can change it.",
]


def render_dashboard_tour(state, avatar=XAI_AGENT_AVATAR, key="tour", messages=None):
    """
    Plays the dashboard tour as a sequence of agent bubbles, each with typing
    dots and the typewriter. state["shown"] counts the bubbles already played,
    so a rerun mid-tour (a widget click) replays nothing: finished bubbles show
    at once and the tour carries on from where it was.
    """
    messages = DASHBOARD_TOUR if messages is None else messages
    st.markdown(AGENT_CSS, unsafe_allow_html=True)
    for i, text in enumerate(messages):
        with st.container(key=f"xai_agent_{key}_{i}"), \
                st.chat_message("assistant", avatar=avatar):
            if i == 0:
                st.caption("EDGE-IDPS XAI AGENT · DASHBOARD TOUR")
            animate = i >= state.get("shown", 0)
            _say(text, animate)
            if animate:
                state["shown"] = i + 1
