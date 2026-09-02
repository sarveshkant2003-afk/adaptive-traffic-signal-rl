"""Interactive demo for the adaptive traffic signal RL project.

Deployed app replays precomputed SUMO simulation traces (generated locally by
scripts/generate_traces.py) -- no SUMO needed here, so it runs anywhere Streamlit does.
"""

import gzip
import json
import os
import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

ROOT = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Adaptive Traffic Signal RL",
    page_icon="🚦",
    layout="wide",
)

# Headline results from notebook 2 (mean ± std over 10 held-out seeds).
# If you retrain and re-run notebook 2, update these and the README table together.
RESULTS = {
    "Q-learning (trained)":            {"wait": (5.31, 0.07), "travel": (37.71, 0.13), "trips": 1190.3},
    "Fixed-time equal (30/30s)":       {"wait": (14.64, 0.24), "travel": (43.94, 0.23), "trips": 1190.0},
    "Fixed-time proportional (40/20s)": {"wait": (12.66, 0.18), "travel": (42.12, 0.18), "trips": 1190.6},
}
COLORS = {"Q-learning (trained)": "#2ecc71",
          "Fixed-time equal (30/30s)": "#e74c3c",
          "Fixed-time proportional (40/20s)": "#e67e22"}

TRACE_FILES = {
    "Q-learning (trained)": "qlearning",
    "Fixed-time equal (30/30s)": "fixed_equal",
    "Fixed-time proportional (40/20s)": "fixed_prop",
}


@st.cache_data
def load_trace(key):
    with gzip.open(os.path.join(ROOT, "app", "traces", f"{key}.json.gz"), "rt") as f:
        return json.load(f)


@st.cache_data
def load_qtable():
    with open(os.path.join(ROOT, "qtable.pkl"), "rb") as f:
        q = pickle.load(f)
    return {k: list(map(float, v)) for k, v in q.items()}


@st.cache_data
def load_history():
    path = os.path.join(ROOT, "train_history.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_replay_template():
    with open(os.path.join(ROOT, "app", "replay.html")) as f:
        return f.read()


def discretize(n):  # must mirror the notebooks
    if n == 0:
        return 0
    if n <= 2:
        return 1
    if n <= 5:
        return 2
    return 3


BUCKET_LABELS = {0: "empty (0)", 1: "light (1–2)", 2: "medium (3–5)", 3: "heavy (6+)"}


st.title("🚦 Adaptive Traffic Signal Control with Q-Learning")
st.markdown(
    "A reinforcement-learning agent that controls a traffic light in the [SUMO](https://eclipse.dev/sumo/) "
    "traffic simulator — deciding every 5 seconds whether to **hold** or **switch** the green phase based on "
    "live queue lengths. It cuts queuing by **~64%** vs. naive fixed timing, and **~58%** vs. a "
    "demand-tuned fixed plan."
)

tab_demo, tab_results, tab_policy, tab_how = st.tabs(
    ["🚦 Live demo", "📊 Results", "🧠 Policy explorer", "📖 How it works"])


# ---------------------------------------------------------------- Live demo
with tab_demo:
    st.markdown(
        "Watch the **trained RL agent** (left) against a **fixed-time controller** (right) on the *identical* "
        "traffic — same simulation seed, same vehicle arrivals. Red cars are stopped, blue cars are moving. "
        "Watch the *Total queue-time* counters diverge: the agent gives the busy north–south road the green "
        "time it actually needs, moment to moment."
    )
    baseline = st.selectbox(
        "Compare the RL agent against:",
        ["Fixed-time equal (30/30s)", "Fixed-time proportional (40/20s)"],
    )
    traces = [load_trace("qlearning"), load_trace(TRACE_FILES[baseline])]
    payload = [{"controller": t["controller"], "frames": t["frames"]} for t in traces]
    html = load_replay_template().replace("__TRACES__", json.dumps(payload, separators=(",", ":")))
    components.html(html, height=520)
    st.caption(
        "First 10 minutes of a held-out evaluation episode (seed 1000), replayed from a real SUMO run. "
        "N/S carries 800 veh/h total, E/W 400 veh/h — the asymmetric demand that fixed timing handles badly."
    )


# ---------------------------------------------------------------- Results
with tab_results:
    st.subheader("Benchmark: 10 held-out evaluation seeds")
    names = list(RESULTS)

    col1, col2 = st.columns(2)
    for col, metric, title, ylabel in [
        (col1, "wait", "Queuing (lower is better)", "Avg. halted vehicles per decision step"),
        (col2, "travel", "Travel time (lower is better)", "Mean vehicle travel time (s)"),
    ]:
        fig = go.Figure(go.Bar(
            x=names,
            y=[RESULTS[n][metric][0] for n in names],
            error_y=dict(type="data", array=[RESULTS[n][metric][1] for n in names]),
            marker_color=[COLORS[n] for n in names],
            text=[f"{RESULTS[n][metric][0]:.2f}" for n in names],
            textposition="outside",
        ))
        fig.update_layout(title=title, yaxis_title=ylabel, showlegend=False,
                          margin=dict(t=40, b=10), height=420)
        fig.update_xaxes(tickangle=-15)
        col.plotly_chart(fig, width="stretch")

    eq, pr = RESULTS["Fixed-time equal (30/30s)"], RESULTS["Fixed-time proportional (40/20s)"]
    ql = RESULTS["Q-learning (trained)"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Queuing vs. equal split", f"−{(1 - ql['wait'][0]/eq['wait'][0])*100:.1f}%")
    m2.metric("Queuing vs. proportional split", f"−{(1 - ql['wait'][0]/pr['wait'][0])*100:.1f}%")
    m3.metric("Travel time vs. equal split", f"−{(1 - ql['travel'][0]/eq['travel'][0])*100:.1f}%")
    st.caption(
        "Completed-trip counts are essentially identical across controllers (~1190), so the gains are genuine "
        "scheduling efficiency — not one controller admitting fewer vehicles. The proportional baseline already "
        "knows the true 2:1 average demand ratio; beating it shows the value of reacting to *moment-to-moment* "
        "fluctuations, not just knowing the average."
    )

    history = load_history()
    if history:
        st.subheader("Training: learning curve")
        eps = [h["episode"] for h in history]
        wait = [h["avg_wait_per_step"] for h in history]
        epsilon = [h["epsilon"] for h in history]
        rolling = pd.Series(wait).rolling(15, min_periods=1).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eps, y=wait, mode="lines", name="Avg wait/step",
                                 line=dict(color="#4da3ff", width=1), opacity=0.35))
        fig.add_trace(go.Scatter(x=eps, y=rolling, mode="lines", name="Rolling mean (15 ep)",
                                 line=dict(color="#4da3ff", width=2.5)))
        fig.add_trace(go.Scatter(x=eps, y=epsilon, mode="lines", name="ε (exploration rate)",
                                 line=dict(color="#f1c40f", width=1.5, dash="dot"), yaxis="y2"))
        fig.update_layout(
            xaxis_title="Training episode",
            yaxis=dict(title="Avg halted vehicles per step"),
            yaxis2=dict(title="ε", overlaying="y", side="right", range=[0, 1.05]),
            height=420, margin=dict(t=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Queuing falls as ε decays: early episodes act mostly at random (exploration); by the end the "
            "agent mostly exploits what it has learned. The noise never fully disappears because vehicle "
            "arrivals are random every episode."
        )


# ---------------------------------------------------------------- Policy explorer
with tab_policy:
    st.subheader("Ask the trained agent: what would you do here?")
    st.markdown(
        "The entire trained 'brain' is a **lookup table**: state → value of each action. "
        "Set up a traffic situation below and see exactly what the agent learned to do — no black box."
    )

    Q = load_qtable()
    left, right = st.columns([1, 1.2])

    with left:
        qn = st.slider("🚗 North approach — queued vehicles", 0, 10, 4)
        qs = st.slider("🚗 South approach — queued vehicles", 0, 10, 4)
        qe = st.slider("🚗 East approach — queued vehicles", 0, 10, 0)
        qw = st.slider("🚗 West approach — queued vehicles", 0, 10, 0)
        green = st.radio("Currently green:", ["North–South", "East–West"], index=1, horizontal=True)

    green_idx = 0 if green == "North–South" else 1
    buckets = tuple(discretize(x) for x in (qn, qe, qs, qw))  # order: N, E, S, W (matches INCOMING_EDGES)
    state = buckets + (green_idx,)

    with right:
        st.markdown(
            f"State the agent sees: `{state}`  \n"
            f"(N: *{BUCKET_LABELS[buckets[0]]}*, E: *{BUCKET_LABELS[buckets[1]]}*, "
            f"S: *{BUCKET_LABELS[buckets[2]]}*, W: *{BUCKET_LABELS[buckets[3]]}*, "
            f"green: *{green}*)"
        )
        key = repr(state)
        # qtable keys are tuples; after load_qtable they're still tuples
        qvals = Q.get(state)
        if qvals is None:
            st.warning(
                "The agent **never encountered this state during training** — with the 2:1 N/S-heavy demand "
                "pattern, some situations (like a huge E/W queue) simply never arose. Its policy falls back to "
                "**keep the current phase** (the safe default in the evaluation code). This is the honest "
                "limitation of tabular RL: no generalization to unseen states."
            )
        else:
            action = int(np.argmax(qvals))
            fig = go.Figure(go.Bar(
                x=["Keep current green", "Switch phases"],
                y=qvals,
                marker_color=["#2ecc71" if action == 0 else "#3a3f44",
                              "#2ecc71" if action == 1 else "#3a3f44"],
                text=[f"{v:.1f}" for v in qvals],
                textposition="outside",
            ))
            fig.update_layout(
                yaxis_title="Learned Q-value (expected future reward)",
                height=330, margin=dict(t=30, b=10),
                title="Q-values for this state (higher = better)",
            )
            st.plotly_chart(fig, width="stretch")

            busy_dir = "North–South" if (qn + qs) >= (qe + qw) else "East–West"
            decision = "**KEEP** the current green" if action == 0 else "**SWITCH** to the other phase"
            serves_busy = (green == busy_dir)
            st.success(
                f"Agent's decision: {decision}.  \n"
                + (f"The green is already serving the busier {busy_dir} direction, so holding it clears the "
                   f"larger queue." if action == 0 and serves_busy else
                   f"The busier direction ({busy_dir}) is waiting on red, so switching serves more demand."
                   if action == 1 else
                   "Switching costs 3 s of all-red yellow time — the agent learned it's not worth paying that "
                   "cost for the current queue imbalance.")
            )
            st.caption(
                "Q-values are negative because the reward is *minus* the accumulated queue — the agent picks the "
                "least-bad future. A switch request is also only honoured after 10 s of minimum green time "
                "(a real-world safety constraint built into the environment)."
            )

    with st.expander("Peek at the raw learned Q-table"):
        rows = [{"State (N,E,S,W buckets, green)": str(k), "Q(keep)": v[0], "Q(switch)": v[1],
                 "Learned action": "switch" if v[1] > v[0] else "keep"}
                for k, v in sorted(Q.items())]
        st.dataframe(pd.DataFrame(rows), width="stretch", height=400)
        st.caption(f"{len(Q)} states were actually visited during training (out of 512 possible).")


# ---------------------------------------------------------------- How it works
with tab_how:
    st.subheader("The problem")
    st.markdown(
        "Fixed-time signals give each direction a pre-set green on a repeating cycle — blind to actual traffic. "
        "When a busy road crosses a quiet one, that wastes green time on empty lanes while queues build on the "
        "busy approach. This project trains an agent to make the hold/switch decision **adaptively**, from "
        "live queue measurements."
    )
    st.subheader("The reinforcement-learning formulation")
    st.markdown(
        """
| Piece | Choice | Why |
|---|---|---|
| **State** | Queue length of each approach, bucketed into 4 levels (0 / 1–2 / 3–5 / 6+), + which phase is green | Small enough to enumerate → a lookup table suffices |
| **Actions** | `keep` or `switch` (switch honoured only after 10 s min-green; always passes through 3 s yellow) | Mirrors how real controllers separate scheduling from safety timing |
| **Reward** | −(halted vehicles, summed each second since last decision) | Directly penalizes accumulated queuing |
| **Algorithm** | Tabular Q-learning, ε-greedy (ε: 1.0 → 0.05 over 200 eps), α=0.1, γ=0.9 | Simplest method that is sufficient at this scale — fully inspectable |
| **Training** | 300 episodes × 1 simulated hour in SUMO | ~38 reachable states converge well before this |
        """
    )
    st.markdown(
        "**Update rule** — after every decision, the table entry for (state, action) is nudged toward what "
        "actually happened:"
    )
    st.latex(r"Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]")
    st.subheader("Why this demo needs no simulator")
    st.markdown(
        "SUMO ran **locally, once** — `scripts/generate_traces.py` recorded per-second vehicle positions and "
        "signal states from real evaluation episodes into small compressed JSON files. This page just replays "
        "them (HTML canvas + a bit of JavaScript), so the deployed demo is instant and dependency-free, while "
        "every frame you see comes from a genuine simulation."
    )
    st.markdown(
        "**Code and notebooks:** [github.com/sarveshkant2003-afk/adaptive-traffic-signal-rl](https://github.com/sarveshkant2003-afk/adaptive-traffic-signal-rl)"
    )
