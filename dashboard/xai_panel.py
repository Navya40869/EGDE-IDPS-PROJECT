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

import re
import time
from collections import namedtuple

import plotly.graph_objects as go
import streamlit as st



def robot_avatar_svg(line="#00ff41", face="#0d1310", eye="#16261a",
                     body="#56655c", shade="#26302b", walking=False):
    """
    Animated full-body robot as an SVG string: binocular eyes on a neck, a
    boxy gunmetal body with a green chest readout, stubby arms and a tread.

    st.chat_message(avatar=...) accepts a raw "<svg ...>" string and shows it as
    an <img> data URI. Browsers run CSS @keyframes inside an SVG loaded that
    way (scripts are blocked, which this doesn't need), so the animation lives
    in the SVG's own <style>: an occasional blink, a slow bob and head tilt,
    and an antenna twitch that is still most of the time.

    walking=True adds a one-shot "arrival" that plays as the image loads: the
    tread rolls, the wheels turn, and the body leans forward and steps twice
    before settling into the idle loop. The page-level CSS slides the image in
    from the left at the same time (AGENT_CSS, "xai_walk_").

    All motion stops under prefers-reduced-motion.
    """
    walk_css = f"""
  .walk {{ transform-box: view-box; transform-origin: 24px 46px;
           animation: lean 1.4s ease-out both; }}
  .track {{ animation: roll 0.35s linear 4; }}
  .wheel {{ transform-box: fill-box; transform-origin: center;
            animation: spin 1.4s ease-out 1; }}
  @keyframes lean {{ 0% {{ transform: rotate(8deg); }}
                     15%, 45% {{ transform: rotate(8deg) translateY(-1.2px); }}
                     30%, 60% {{ transform: rotate(8deg) translateY(0); }}
                     100% {{ transform: rotate(0deg); }} }}
  @keyframes roll {{ to {{ stroke-dashoffset: -3.2; }} }}
  @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
  @media (prefers-reduced-motion: reduce) {{ .walk, .track, .wheel {{ animation: none; }} }}""" if walking else ""
    wheels = "".join(
        f'<g class="wheel"><circle cx="{x}" cy="42.5" r="2.3" fill="{face}" stroke="{body}" '
        f'stroke-width="0.7"/><line x1="{x}" y1="40.6" x2="{x}" y2="44.4" stroke="{line}" '
        f'stroke-opacity="0.55" stroke-width="0.7"/></g>'
        for x in (14, 24, 34))
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48">
<defs>
  <linearGradient id="metal" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="{body}"/><stop offset="1" stop-color="{shade}"/>
  </linearGradient>
</defs>
<style>
  .bob {{ animation: bob 4s ease-in-out infinite alternate; }}
  .head {{ transform-box: view-box; transform-origin: 24px 22px;
           animation: tilt 9s ease-in-out infinite; }}
  .eyes {{ transform-box: fill-box; transform-origin: center;
           animation: blink 5.5s ease-in-out infinite; }}
  .ant {{ transform-box: view-box; transform-origin: 32.5px 6.5px;
          animation: twitch 7s ease-in-out infinite; }}
  @keyframes bob {{ from {{ transform: translateY(0.4px); }} to {{ transform: translateY(-0.9px); }} }}
  @keyframes tilt {{ 0%, 20%, 100% {{ transform: rotate(0deg); }}
                     35%, 45% {{ transform: rotate(-5deg); }} 65%, 75% {{ transform: rotate(3deg); }} }}
  @keyframes blink {{ 0%, 88%, 100% {{ transform: scaleY(1); }}
                      91% {{ transform: scaleY(0.08); }} 94% {{ transform: scaleY(1); }} }}
  @keyframes twitch {{ 0%, 70%, 100% {{ transform: rotate(0deg); }}
                       75% {{ transform: rotate(-12deg); }} 82% {{ transform: rotate(9deg); }}
                       88% {{ transform: rotate(0deg); }} }}
  @media (prefers-reduced-motion: reduce) {{ .bob, .head, .eyes, .ant {{ animation: none; }} }}{walk_css}
</style>
<g class="walk">
  <rect x="8.5" y="38.5" width="31" height="8" rx="4" fill="{shade}"/>
  <rect class="track" x="9.1" y="39.1" width="29.8" height="6.8" rx="3.4" fill="none" stroke="{face}"
        stroke-width="1.2" stroke-dasharray="1.6 1.6"/>
  {wheels}
  <g class="bob">
    <rect x="7.5" y="26" width="4" height="8" rx="1.6" fill="url(#metal)"/>
    <rect x="36.5" y="26" width="4" height="8" rx="1.6" fill="url(#metal)"/>
    <rect x="11.5" y="22.5" width="25" height="17" rx="2.5" fill="url(#metal)" stroke="{face}" stroke-width="0.8"/>
    <rect x="15" y="26" width="11" height="6.5" rx="1" fill="{face}"/>
    <rect x="16.3" y="28.3" width="4.5" height="1.9" rx="0.5" fill="{line}"/>
    <rect x="21.6" y="28.3" width="2.8" height="1.9" rx="0.5" fill="{line}" fill-opacity="0.45"/>
    <circle cx="31" cy="28.5" r="1.4" fill="{line}"/>
    <line x1="11.5" y1="35.5" x2="36.5" y2="35.5" stroke="{face}" stroke-width="0.7"/>
    <g class="head">
      <rect x="22.6" y="17" width="2.8" height="6" fill="{shade}"/>
      <rect x="20.5" y="11" width="7" height="2.6" rx="1" fill="{shade}"/>
      <g class="ant">
        <line x1="32.5" y1="6.5" x2="34.5" y2="2.6" stroke="{body}" stroke-width="1" stroke-linecap="round"/>
        <circle cx="34.8" cy="2.2" r="1.3" fill="{line}"/>
      </g>
      <rect x="9.5" y="5.5" width="13" height="12.5" rx="5" fill="url(#metal)" stroke="{face}" stroke-width="0.8"/>
      <rect x="25.5" y="5.5" width="13" height="12.5" rx="5" fill="url(#metal)" stroke="{face}" stroke-width="0.8"/>
      <rect x="11.4" y="7.4" width="9.2" height="8.7" rx="3.8" fill="{eye}" stroke="{line}" stroke-width="0.8"/>
      <rect x="27.4" y="7.4" width="9.2" height="8.7" rx="3.8" fill="{eye}" stroke="{line}" stroke-width="0.8"/>
      <g class="eyes">
        <circle cx="16" cy="11.75" r="2.7" fill="{line}"/>
        <circle cx="32" cy="11.75" r="2.7" fill="{line}"/>
        <circle cx="17" cy="10.7" r="0.85" fill="{face}"/>
        <circle cx="33" cy="10.7" r="0.85" fill="{face}"/>
      </g>
    </g>
  </g>
</g>
</svg>"""


# The default avatar. Anything st.chat_message accepts works in its place — an
# emoji ("🤖"), an image path or URL, or another SVG string — via this
# constant or the avatar= parameter of the render functions.
XAI_AGENT_AVATAR = robot_avatar_svg()
# The tour's arriving agent: the same robot, rolling in to its next panel
XAI_AGENT_WALKING_AVATAR = robot_avatar_svg(walking=True)

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

# Typewriter pacing, slow enough to read along with: 180 ms per word (~330
# words a minute) plus a short breath after clauses and a longer one after
# sentences, the way someone talking would pause.
TYPEWRITER_WORD_S = 0.18
TYPEWRITER_CLAUSE_PAUSE_S = 0.2
TYPEWRITER_SENTENCE_PAUSE_S = 0.45

# "Typing…" dots shown before each message starts typing
TYPING_INDICATOR_S = 1.2

# Entrance animation for agent bubbles and the typing dots. Scoped to the
# "xai_agent_*" containers this module creates, so other chat elements and
# the rest of the dashboard are unaffected.
AGENT_CSS = """
<style>
    [class*="st-key-xai_agent_"] [data-testid="stChatMessage"] {
        animation: xai-enter 0.35s ease-out both;
    }
    /* The tour's arriving agent rolls in from the left while the SVG itself
       leans and turns its tread (robot_avatar_svg(walking=True)) */
    [class*="st-key-xai_walk_"] [data-testid="stChatMessage"] img[alt="assistant avatar"] {
        animation: xai-walk-in 1.4s cubic-bezier(0.25, 0.7, 0.35, 1) both;
    }
    @keyframes xai-walk-in {
        from { transform: translateX(-2.5rem); opacity: 0; }
        25%  { opacity: 1; }
        to   { transform: translateX(0); }
    }
    /* Bigger robot: Streamlit draws chat avatars at 2rem; 3.5rem (1.75x) keeps
       it legible beside the caption without crowding the bubble text. An SVG
       avatar renders as <img alt="assistant avatar"> with no test id. */
    [class*="st-key-xai_agent_"] [data-testid="stChatMessage"] img[alt="assistant avatar"] {
        width: 3.5rem; height: 3.5rem; object-fit: contain;
    }
    @media (max-width: 640px) {
        [class*="st-key-xai_agent_"] [data-testid="stChatMessage"] img[alt="assistant avatar"] {
            width: 2.75rem; height: 2.75rem;
        }
    }
    /* A tour step the agent has moved on from: kept readable, but dimmed and
       without the robot, so the agent reads as having left for the next panel. */
    [class*="st-key-xai_tourpast_"] {
        opacity: 0.6; border-left: 2px solid #1f4d2a; padding-left: 0.75rem;
        transition: opacity 0.3s ease-out;
    }
    [class*="st-key-xai_tourpast_"] p { font-size: 0.85rem; }
    /* Holds only the scroll-into-view script; never takes up space */
    [class*="st-key-xai_tourscroll_"] { display: none; }
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
        [class*="st-key-xai_walk_"] [data-testid="stChatMessage"] img[alt="assistant avatar"],
        .xai-typing span { animation: none; }
        .xai-typing span { opacity: 0.7; }
        [class*="st-key-xai_tourpast_"] { transition: none; }
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


def _words(text):
    # Each word with the whitespace before it, so "".join() gives text back
    # exactly, paragraph breaks included
    return re.findall(r"\s*\S+", text)


def _word_delay(word):
    end = word.strip().rstrip("*_)\"'")  # look past closing markdown and quotes
    if end.endswith((".", "!", "?", ":")):
        return TYPEWRITER_WORD_S + TYPEWRITER_SENTENCE_PAUSE_S
    if end.endswith((",", ";", "—")):
        return TYPEWRITER_WORD_S + TYPEWRITER_CLAUSE_PAUSE_S
    return TYPEWRITER_WORD_S


def reading_time(text):
    """Seconds the typewriter takes over `text` (typing dots not included)."""
    return sum(_word_delay(w) for w in _words(text))


def _typewriter(text):
    for word in _words(text):
        yield word
        time.sleep(_word_delay(word))


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


def render_xai_explanation(explanation, state=None, avatar=XAI_AGENT_AVATAR, key="xai",
                           reduce_motion=False):
    """
    Renders the agent chat bubble (typewriter text) with the contribution donut.

    state: a dict that persists across reruns (e.g. the session-state entry
    holding this explanation). The typing dots and typewriter play once per
    explanation; later reruns — auto-refresh, widget clicks, even one that
    cuts the typewriter off — show the finished text instantly.

    reduce_motion: the browser asks for reduced motion (app.py reads it). The
    text then appears whole at once; CSS media queries can't do this part,
    since Python drives the typewriter.
    """
    shares = toward_shares(explanation.attributions)
    text = build_summary(explanation, shares)
    st.markdown(AGENT_CSS, unsafe_allow_html=True)
    with st.container(key=f"xai_agent_{key}"), st.chat_message("assistant", avatar=avatar):
        st.caption("EDGE-IDPS XAI AGENT")
        animate = state is not None and not state.get("typed")
        if animate:
            state["typed"] = True  # before typing, so an interrupted run isn't retyped
        _say(text, animate and not reduce_motion)
        if shares:
            st.plotly_chart(build_donut(explanation, shares), key=f"{key}_donut",
                            config={"displaylogo": False})


# --- Dashboard tour: fixed text describing the panels in app.py, in the order
# they appear on the page. Written for someone new to networking and AI.
# Keep in step with app.py if panels are added, renamed or removed.
#
# Each step names the panel it talks about (its anchor). app.py puts an empty
# slot just above each panel (tour_anchor("metrics"), ...) and the step plays
# in that slot, so the agent shows up beside whatever it is describing. "intro"
# and "sidebar" both sit in the tour header at the top of the page, which is
# also right next to the sidebar.
TourStep = namedtuple("TourStep", "anchor text")
TOUR_ANCHORS = ("intro", "metrics", "charts", "feed", "sidebar")
TOUR_FALLBACK_ANCHOR = "intro"

DASHBOARD_TOUR = [
    # The greeting is deliberately the longest, unhurried bubble: about a
    # minute at the typewriter's pace (reading_time()).
    TourStep("intro",
             "Hi there, and welcome! I'm the Edge-IDPS XAI agent, a small robot who lives "
             "inside this dashboard, and I'm really glad you stopped by. My job is to explain "
             "things. This system keeps watch over the network traffic around this computer, "
             "and an AI model decides, many times every second, whether that traffic looks "
             "normal or looks like an attack. That AI is quick, but on its own it never tells "
             "you *why* it decided anything. That's where I come in: I put what it's doing into "
             "plain words, so you can understand its decisions, and feel free to question them "
             "when something doesn't look right.\n\n"
             "Over the next few minutes I'll walk you around the page, one panel at a time. "
             "We'll start with a few basics about how networks talk, then roll over to the "
             "summary numbers at the top, the charts that show the bigger picture, the live feed "
             "of events, and finally the controls in the sidebar, where you can ask me to "
             "explain a single decision in detail.\n\n"
             "You don't need any background in networking or AI to follow along. I'll explain "
             "every term as we go, and nothing on this tour changes how the system behaves: "
             "it's all look, no touch. If you've seen enough at any point, the **Close tour** "
             "button at the top ends it straight away. So get comfortable, take your time, and "
             "let's go and have a look around together."),

    TourStep("intro",
             "First, the basics. This dashboard watches the network traffic going in and out "
             "of this computer. Traffic travels in small pieces called **packets**, and I group "
             "the packets exchanged between two devices into a **flow**: think of it as one "
             "conversation. An AI model reads every flow and labels it either **Benign** "
             "(normal) or as one of four kinds of attack: **DoS/DDoS** (flooding a device so it "
             "stops responding), **Reconnaissance** (scanning for weak spots), **Spoofing** "
             "(pretending to be a different device) or **Mirai** (malware that hijacks "
             "internet-connected gadgets)."),

    TourStep("metrics",
             "This row of numbers is the quick summary, covering the last minute by default "
             "(you can change that window in the sidebar). **ACTIVE THREATS** counts the flows "
             "labelled as an attack. **FLOWS/SEC** is how many conversations per second I'm "
             "seeing; a sudden jump can mean a flood. **AVG CONFIDENCE** is how sure the AI was "
             "of its labels on average: 95% means very sure. **IPs BLOCKED** is how many devices "
             "the firewall has shut out (an IP address is a device's address on the network), "
             "and **MODEL** is which version of the AI is running. Flows with too few packets to "
             "judge are counted separately as INSUFFICIENT_EVIDENCE and left out of these numbers."),

    TourStep("charts",
             "These charts show the bigger picture. **Attack category distribution** shows how "
             "many flows got each label. Every flow also gets a **risk score** from 0 to 10 that "
             "combines how sure the AI was with how often the same sender has looked suspicious "
             "in the last minute: one shaky alert scores low, a confident and repeated one scores "
             "high. **Risk score distribution** groups the flows by that score, and the **risk "
             "score timeline** shows it over time, so spikes are the moments worth a closer look."),

    TourStep("feed",
             "The **live event feed** lists the newest flows, one per line: the time, the "
             "sender's address, the label, the AI's confidence, the risk score and what the "
             "system did about it. Those actions get stronger as risk rises: **LOG_ONLY** (just "
             "noted), **THROTTLE** (flagged to be slowed down), **DROP_PACKET** (its traffic is "
             "thrown away) and **BLOCK_IP** (the sender is shut out). Senders that were dropped "
             "or blocked appear in the **firewall blocklist** table below, and **recent "
             "telemetry** is the raw log this whole page is built from."),

    TourStep("sidebar",
             "Last, the sidebar on the left. You can pause the live updates, change the time "
             "window, or filter the charts by label. **Explain a live flow** asks me to watch "
             "the network for a few seconds and explain *why* the AI labelled the busiest "
             "conversation the way it did; **Explain a replayed test row** does the same for a "
             "saved example of an attack, like Mirai. I'll name the measurements that mattered "
             "most, and a donut chart shows how much each one pushed the AI toward its answer. "
             "Nothing on this page can change the system — it only displays it."),
]


def tour_scroll_script(container_key):
    """
    Scrolls the parent page to the tour step in `container_key`. Streamlit
    component iframes are same-origin, so the script can reach the page
    around it. With prefers-reduced-motion it jumps instead of gliding.
    """
    return f"""<script>
const doc = window.parent.document;
const el = doc.querySelector(".st-key-{container_key}");
if (el) {{
  const reduce = window.parent.matchMedia("(prefers-reduced-motion: reduce)").matches;
  el.scrollIntoView({{behavior: reduce ? "auto" : "smooth", block: "center"}});
}}
</script>"""


def _tour_step_active(text, i, key, avatar, animate, reduce_motion=False):
    # Arriving (animate) the agent walks in; on a redraw it is already standing there
    walk = animate and not reduce_motion
    with st.container(key=f"xai_agent_{key}_{i}"), \
            st.container(key=f"xai_walk_{key}_{i}" if walk else None), \
            st.chat_message("assistant", avatar=avatar if walk else _standing(avatar)):
        st.caption("EDGE-IDPS XAI AGENT · DASHBOARD TOUR")
        if animate:
            # Fixed script built here (no user input), so a same-origin iframe is safe
            with st.container(key=f"xai_tourscroll_{key}_{i}"):
                st.iframe(tour_scroll_script(f"xai_agent_{key}_{i}"), height=1)
        _say(text, walk)


def _standing(avatar):
    return XAI_AGENT_AVATAR if avatar == XAI_AGENT_WALKING_AVATAR else avatar


def _tour_step_past(text, i, key):
    with st.container(key=f"xai_tourpast_{key}_{i}"):
        st.markdown(text)


def render_dashboard_tour(state, slots, avatar=XAI_AGENT_WALKING_AVATAR, key="tour", steps=None,
                          reduce_motion=False):
    """
    Plays the dashboard tour, each step beside the panel it describes.

    slots: {anchor: st.container()} that app.py placed above each panel. A
    step whose anchor has no slot (e.g. no telemetry yet, so the panels never
    drew) plays in the TOUR_FALLBACK_ANCHOR slot instead.

    Only the current step shows the agent (robot walking in, typing dots,
    typewriter, and a scroll to bring it into view). When the next step
    starts, the previous one is redrawn as a dimmed note without the robot,
    so the agent appears to walk from panel to panel. state["shown"] counts
    the steps already started, so a rerun mid-tour (a widget click) replays
    nothing: those steps show finished at once and the tour carries on with
    the next one.

    reduce_motion: the browser asks for reduced motion (app.py reads it). Each
    step then appears whole, with no walk, typing dots or typewriter, and
    stays for as long as the typewriter would have taken before the agent
    moves on — so the tour still paces itself, it just doesn't animate.

    Must be called after every panel is drawn, since the slots are filled
    from here, late in the script.
    """
    steps = DASHBOARD_TOUR if steps is None else steps
    st.markdown(AGENT_CSS, unsafe_allow_html=True)

    placeholders = []
    for step in steps:
        with slots.get(step.anchor, slots[TOUR_FALLBACK_ANCHOR]):
            placeholders.append(st.empty())

    shown = min(state.get("shown", 0), len(steps))
    for i in range(shown):
        with placeholders[i].container():
            if i == shown - 1:
                _tour_step_active(steps[i].text, i, key, avatar, animate=False)
            else:
                _tour_step_past(steps[i].text, i, key)

    for i in range(shown, len(steps)):
        if i > 0:
            with placeholders[i - 1].container():
                _tour_step_past(steps[i - 1].text, i - 1, key)
        state["shown"] = i + 1  # before playing, so an interrupted step isn't replayed
        with placeholders[i].container():
            _tour_step_active(steps[i].text, i, key, avatar, animate=True,
                              reduce_motion=reduce_motion)
        if reduce_motion and i < len(steps) - 1:
            time.sleep(reading_time(steps[i].text))
