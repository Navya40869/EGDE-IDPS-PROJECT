# Explainability: why did the CNN classify this flow that way?

`engine/explain_prediction.py` captures one live flow, classifies it with the
5-class CNN exactly as the pipeline does, and shows how much each of the 20
frozen features pushed the prediction toward (or away from) the predicted
class. It prints a table and saves a bar chart to `logs/last_explanation.png`.

## What Integrated Gradients is

Integrated Gradients (IG) answers "which inputs were responsible for this
output?" It takes a *baseline* input (a reference "nothing special" flow),
draws a straight line from the baseline to the actual flow, and adds up the
model's gradient for the predicted class at each step along that line. Each
feature then gets one number, its **attribution**:

- **Positive (green):** this feature's value pushed the model *toward* the
  predicted class.
- **Negative (red, hatched):** it pushed the model *away* from the predicted
  class. The prediction still won, despite this feature.
- **Size:** how strong the push was, in the model's logit (pre-softmax score)
  units.

IG has a useful guarantee called *completeness*: the attributions add up to
the difference between the predicted class's score on this flow and its
score on the baseline. The tool prints that sum and the numerical error
("convergence delta"), so you can check that the explanation accounts for the
whole decision.

## Why IG and not SHAP

- **Fast enough for interactive use.** For this small 1D-CNN, IG costs one
  batched forward/backward pass of 200 interpolated inputs, about 45 ms on
  CPU. Model-agnostic SHAP (KernelSHAP) needs thousands of model evaluations
  per explanation, plus a background dataset, which is too slow to run behind
  a dashboard button.
- **Exact for differentiable models.** The CNN is fully differentiable, so
  IG can use the true gradients instead of approximating the model by
  sampling.
- **Deterministic.** The same flow always gets the same explanation. There is
  no sampling noise to explain away in a demo.

## Why the baseline is all zeros

The CNN never sees raw values. It sees StandardScaler output, and
StandardScaler centres every feature on its training-set mean. So **0 in
scaled space means "exactly average"** for that feature, and the all-zero
vector is the *average training flow*. Each attribution therefore reads as:
"how much did this feature being different from an average flow move the
prediction?" A raw-space zero (0 bytes, TTL 0, 0 packets/s) would be an
impossible flow and a misleading reference point.

The table's **scaled value** column is in the same units: standard deviations
from the training mean. For example, `Time_To_Live +4.26` means this flow's
average TTL was 4.26 std-devs above typical.

## Scope limitation: fresh predictions only

`logs/live_stream_output.csv` stores each prediction's *output* (class,
confidence, risk), **not** the 20-feature vector it was computed from.
Historical dashboard rows therefore can't be explained after the fact. This
tool explains a flow it captures and classifies itself, in the same run.

It uses the same `INSUFFICIENT_EVIDENCE` gate as `stream_pipeline.py`
(`MIN_PACKETS_FOR_CLASSIFICATION = 2`). Flows below the gate are never
classified by the pipeline, so they aren't explained either.

Explanations describe what *the model* relied on. They are not proof that a
feature is causally malicious. If a dominant feature is one of the
best-effort reconstructions noted in `engine/feature_extractor.py`, read the
explanation with that in mind.

## Running it

Standalone. Run it from an **Administrator** terminal at the project root
(live capture needs raw-socket access and Npcap):

```
python engine/explain_prediction.py --interface "Wi-Fi" --duration 20
```

- `--interface`: NIC to sniff (omit to use Scapy's default).
- `--duration`: maximum seconds to wait for a flow with at least 2 packets.
  Flows close on FIN/RST or a 5 s idle timeout. At the deadline, any active
  flows are flushed and the first eligible one is used.
- `--output`: chart path (default `logs/last_explanation.png`).

The tool only sniffs. It never sends packets.

**From the dashboard:** pause **Live auto-refresh** in the sidebar, then
click **Explain a live flow**. The button stays disabled while auto-refresh
is on, so a refresh can't interrupt the capture, and it never runs by itself.
The dashboard process must be running as Administrator for the capture to see
packets.

The result appears as a chat message from an "XAI agent": after a brief
"typing…" indicator, a short plain-language summary types itself out once,
then a donut chart
(Plotly, hover for exact values) of each feature's share of the push
*toward* the predicted class. Features that pushed away from it are named in
the message. The full signed bar chart and table are in the **Full
attribution detail** expander below it. The display code is in
`dashboard/xai_panel.py`.

The agent avatar is `XAI_AGENT_AVATAR` there. By default it's an animated SVG
robot with a full body (`robot_avatar_svg()`: binocular eyes, a gunmetal body
with green highlights and a tread; it blinks, bobs, tilts its head and
twitches its antenna). Anything `st.chat_message` accepts can replace it: an
emoji, an image path or URL, or another SVG string.

The text types at a read-along pace (`TYPEWRITER_WORD_S` and the pauses
after it). Under the OS "reduce motion" setting, every animation stops and
the text appears whole instead of typing; `app.py` reads that setting from
the browser, since Python drives the typewriter.

**Dashboard tour:** the sidebar's **What am I looking at?** button has the
same agent walk a first-time visitor through each panel: a roughly
minute-long greeting, then five shorter messages, each placed beside the
panel it describes, with the robot walking in to it. It's fixed text
(`DASHBOARD_TOUR` in `dashboard/xai_panel.py`), not generated, and it pauses
live auto-refresh so it isn't cut off mid-sentence.
`tests/test_xai_panel.py` fails if a metric or panel heading in
`dashboard/app.py` isn't mentioned in the tour, so update the tour when the
layout changes.

**Tests** (no capture, no GPU):

```
python -m unittest tests.test_explain_prediction -v
```
