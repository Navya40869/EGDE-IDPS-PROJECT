"""
tests/test_xai_panel.py — dashboard XAI agent panel, no Streamlit app or model.

Feeds hand-built Explanations (same fields as explain_prediction.Explanation)
through the panel's pure helpers and the Plotly donut builder.

Run from the project root:
    python -m unittest tests.test_xai_panel -v
"""

import os
import re
import sys
import unittest
from types import SimpleNamespace

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "dashboard"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "engine"))

import xai_panel  # noqa: E402


def make_explanation(attributions, predicted_class="DoS/DDoS", confidence=0.97):
    return SimpleNamespace(predicted_class=predicted_class, confidence=confidence,
                           attributions=attributions, flow_id=("10.0.0.5", 1), packet_count=12)


class TowardShares(unittest.TestCase):
    def test_shares_are_of_toward_push_only_and_sum_to_one(self):
        shares = xai_panel.toward_shares([("Flow Pkts/s", 0.62, 3.1), ("Tot Fwd Pkts", 0.28, 1.4),
                                          ("Init Win", -0.5, -1.0), ("Dst Port", 0.10, 0.2)])
        self.assertEqual([s[0] for s in shares], ["Flow Pkts/s", "Tot Fwd Pkts", "Dst Port"])
        self.assertAlmostEqual(sum(s[1] for s in shares), 1.0)
        self.assertAlmostEqual(shares[0][1], 0.62)

    def test_no_contributing_features_gives_no_slices(self):
        self.assertEqual(xai_panel.toward_shares([("a", 0.0, 0.0), ("b", -0.3, 1.0),
                                                  ("c", 0.0004, 0.1)]), [])
        self.assertEqual(xai_panel.toward_shares([]), [])

    def test_single_dominant_feature(self):
        shares = xai_panel.toward_shares([("Flow Pkts/s", 1.2, 4.0), ("b", -0.1, 0.0)])
        self.assertEqual(shares, [("Flow Pkts/s", 1.0, 1.2, 4.0)])

    def test_small_and_surplus_contributors_fold_into_other(self):
        attributions = [(f"f{i}", 1.0, 0.0) for i in range(7)] + [("tiny", 0.05, 0.0)]
        shares = xai_panel.toward_shares(attributions)
        self.assertEqual(len(shares), xai_panel.MAX_NAMED_SLICES + 1)
        self.assertEqual(shares[-1][0], "Other (3 features)")
        self.assertIsNone(shares[-1][3])
        self.assertAlmostEqual(sum(s[1] for s in shares), 1.0)

    def test_top_feature_is_named_even_when_below_min_share(self):
        shares = xai_panel.toward_shares([(f"f{i}", 1.0, 0.0) for i in range(40)])
        self.assertEqual(shares[0][0], "f0")
        self.assertTrue(shares[-1][0].startswith("Other ("))

    def test_negligible_threshold_matches_engine(self):
        import explain_prediction
        self.assertEqual(xai_panel.NEGLIGIBLE_ATTRIBUTION,
                         explain_prediction.NEGLIGIBLE_ATTRIBUTION)


class Summary(unittest.TestCase):
    def summary(self, attributions, **kwargs):
        e = make_explanation(attributions, **kwargs)
        return xai_panel.build_summary(e, xai_panel.toward_shares(e.attributions))

    def test_mainly_because_of_top_features_and_names_counter_signal(self):
        text = self.summary([("Flow Pkts/s", 0.62, 3.1), ("Tot Fwd Pkts", 0.28, 1.4),
                             ("Init Win", -0.5, -1.0), ("Dst Port", 0.10, 0.2)])
        self.assertIn("**DoS/DDoS** with 97.0% confidence", text)
        self.assertIn("mainly because of unusually high Flow Pkts/s (62%), "
                      "high Tot Fwd Pkts (28%) and near-average Dst Port (10%)", text)
        self.assertIn("strongest signal against DoS/DDoS was Init Win", text)

    def test_single_dominant(self):
        text = self.summary([("Flow Pkts/s", 1.2, -4.0)])
        self.assertIn("almost entirely because of unusually low Flow Pkts/s (100%)", text)
        self.assertNotIn("against", text)

    def test_zero_contributors(self):
        text = self.summary([("a", 0.0, 0.0), ("b", 0.0, 0.0)], predicted_class="Benign")
        self.assertIn("**Benign**", text)
        self.assertIn("no single feature stood out", text)
        self.assertNotIn("chart", text)

    def test_typewriter_reassembles_text_and_fits_pacing(self):
        text = self.summary([("Flow Pkts/s", 0.62, 3.1), ("Tot Fwd Pkts", 0.28, 1.4)])
        xai_panel_time = xai_panel.time
        delays = []
        xai_panel.time = SimpleNamespace(sleep=delays.append)
        try:
            self.assertEqual("".join(xai_panel._typewriter(text)), text)
        finally:
            xai_panel.time = xai_panel_time
        word, sentence = xai_panel.TYPEWRITER_WORD_S, xai_panel.TYPEWRITER_SENTENCE_PAUSE_S
        self.assertTrue(all(word <= d <= word + sentence for d in delays))
        self.assertAlmostEqual(sum(delays), xai_panel.reading_time(text))
        # Read-along pace: no faster than ~400 words a minute
        self.assertGreaterEqual(sum(delays), len(text.split()) * 60 / 400)

    def test_typewriter_pauses_at_sentence_ends_and_keeps_paragraphs(self):
        text = "One, two **three.**\n\nFour"
        self.assertEqual("".join(xai_panel._words(text)), text)
        delays = [xai_panel._word_delay(w) for w in xai_panel._words(text)]
        self.assertGreater(delays[0], delays[1])      # "One," breathes
        self.assertEqual(max(delays), delays[2])      # "**three.**" ends a sentence
        self.assertEqual(delays[3], xai_panel.TYPEWRITER_WORD_S)

    def test_typing_dots_hold_for_about_a_second(self):
        self.assertTrue(1.0 <= xai_panel.TYPING_INDICATOR_S <= 1.5)


class Donut(unittest.TestCase):
    def test_other_slice_is_gray_and_named_slices_use_palette(self):
        attributions = [(f"f{i}", 1.0, 0.0) for i in range(7)]
        e = make_explanation(attributions)
        fig = xai_panel.build_donut(e, xai_panel.toward_shares(attributions))
        colors = list(fig.data[0].marker.colors)
        self.assertEqual(colors[:-1], xai_panel.SLICE_COLORS)
        self.assertEqual(colors[-1], xai_panel.OTHER_COLOR)


class Avatar(unittest.TestCase):
    def test_default_avatar_is_wellformed_animated_svg(self):
        import xml.etree.ElementTree as ET
        svg = xai_panel.XAI_AGENT_AVATAR
        root = ET.fromstring(svg)  # raises if malformed
        self.assertTrue(root.tag.endswith("svg"))
        for name in ("blink", "bob", "tilt", "twitch", "prefers-reduced-motion"):
            self.assertIn(name, svg)
        self.assertNotIn("@keyframes roll", svg)  # standing still: no walk

    def test_walking_avatar_is_wellformed_and_walk_stops_under_reduced_motion(self):
        import xml.etree.ElementTree as ET
        svg = xai_panel.XAI_AGENT_WALKING_AVATAR
        ET.fromstring(svg)
        for name in ("@keyframes lean", "@keyframes roll", "@keyframes spin", "blink", "tilt"):
            self.assertIn(name, svg)
        reduced = svg.split("@keyframes spin")[1]
        self.assertIn("prefers-reduced-motion: reduce", reduced)
        self.assertIn(".walk, .track, .wheel { animation: none; }", reduced)
        # One-shot arrival, not an endless walk
        self.assertNotRegex(svg, r"(lean|roll|spin) [^;]*infinite")

    def test_avatars_stay_lightweight(self):
        self.assertLess(len(xai_panel.XAI_AGENT_WALKING_AVATAR.encode()), 6000)

    def test_streamlit_treats_it_as_an_svg_image(self):
        # Same check Streamlit's image_to_url uses to recognise an SVG string
        import re
        for svg in (xai_panel.XAI_AGENT_AVATAR, xai_panel.XAI_AGENT_WALKING_AVATAR):
            self.assertRegex(svg, re.compile(r"(^\s?(<\?xml[\s\S]*<svg\s)|^\s?<svg\s|^\s?<svg>\s)"))

    def test_colors_are_themeable(self):
        svg = xai_panel.robot_avatar_svg(line="#123456", body="#abcdef")
        self.assertIn("#123456", svg)
        self.assertIn("#abcdef", svg)

    def test_redrawn_tour_step_uses_the_standing_robot(self):
        self.assertEqual(xai_panel._standing(xai_panel.XAI_AGENT_WALKING_AVATAR),
                         xai_panel.XAI_AGENT_AVATAR)
        self.assertEqual(xai_panel._standing("🤖"), "🤖")


class DashboardTour(unittest.TestCase):
    def test_tour_covers_every_metric_and_panel_in_app(self):
        import re
        with open(os.path.join(PROJECT_ROOT, "dashboard", "app.py"), encoding="utf-8") as f:
            app = f.read()
        tour = " ".join(step.text for step in xai_panel.DASHBOARD_TOUR).lower()
        metrics = re.findall(r'col\d\.metric\("([^"]+)"', app)
        panels = [h for h in re.findall(r'term-header">\[ ([^\]]+?) \]', app)
                  if not h.startswith(("EXPLAINABLE AI", "XAI AGENT"))]
        self.assertGreaterEqual(len(metrics), 5)
        self.assertGreaterEqual(len(panels), 6)
        for label in metrics + panels:
            words = label.replace("(RAW)", "").replace("ACTIVE FIREWALL", "FIREWALL").strip()
            self.assertIn(words.lower(), tour, f"tour doesn't mention {label!r}")

    def test_tour_is_a_few_short_bubbles_after_the_greeting(self):
        self.assertTrue(4 <= len(xai_panel.DASHBOARD_TOUR) <= 6)
        for step in xai_panel.DASHBOARD_TOUR[1:]:
            self.assertLess(len(step.text.split()), 120)
            self.assertLess(xai_panel.reading_time(step.text), 30)

    def test_tour_opens_with_an_unhurried_greeting_and_greets_only_once(self):
        greeting, *rest = [s.text for s in xai_panel.DASHBOARD_TOUR]
        self.assertTrue(greeting.startswith("Hi"))
        # About a minute to type out, dots included
        self.assertTrue(45 <= xai_panel.reading_time(greeting) + xai_panel.TYPING_INDICATOR_S <= 75)
        self.assertGreater(len(greeting.split()), max(len(t.split()) for t in rest))
        for text in rest:
            self.assertFalse(text.lower().startswith(("hi", "hello")))

    def test_explain_panel_has_no_greeting(self):
        e = SimpleNamespace(predicted_class="Mirai", confidence=0.99,
                            attributions=[("Rate", 1.0, 3.0)])
        text = xai_panel.build_summary(e, xai_panel.toward_shares(e.attributions))
        self.assertTrue(text.startswith("I classified"))

    def test_every_anchor_is_known_and_placed_in_app(self):
        with open(os.path.join(PROJECT_ROOT, "dashboard", "app.py"), encoding="utf-8") as f:
            app = f.read()
        placed = set(re.findall(r'tour_anchor\("(\w+)"\)', app))
        self.assertIn(xai_panel.TOUR_FALLBACK_ANCHOR, placed)
        for step in xai_panel.DASHBOARD_TOUR:
            self.assertIn(step.anchor, xai_panel.TOUR_ANCHORS)
            self.assertIn(step.anchor, placed, f"app.py has no slot for {step.anchor!r}")
        # Panels draw in this order, so the agent walks down the page
        order = [m for m in re.findall(r'tour_anchor\("(\w+)"\)', app) if m not in ("intro", "sidebar")]
        self.assertEqual(order, ["metrics", "charts", "feed"])

    def test_scroll_respects_reduced_motion(self):
        js = xai_panel.tour_scroll_script("xai_agent_tour_2")
        self.assertIn(".st-key-xai_agent_tour_2", js)
        self.assertIn("prefers-reduced-motion: reduce", js)
        self.assertIn('reduce ? "auto" : "smooth"', js)


class AgentCss(unittest.TestCase):
    def test_avatar_is_enlarged_with_a_smaller_size_on_narrow_screens(self):
        css = xai_panel.AGENT_CSS
        self.assertIn('img[alt="assistant avatar"]', css)
        sizes = [float(x) for x in re.findall(r"width: ([\d.]+)rem; height", css)]
        self.assertEqual(sizes[0], 3.5)   # 1.75x Streamlit's 2rem default
        self.assertTrue(2 < sizes[1] < sizes[0])
        self.assertIn("max-width: 640px", css)

    def test_reduced_motion_disables_tour_transitions(self):
        reduced = xai_panel.AGENT_CSS.split("prefers-reduced-motion: reduce")[1]
        self.assertIn("xai_tourpast_", reduced)
        self.assertIn("xai_walk_", reduced)


if __name__ == "__main__":
    unittest.main()
