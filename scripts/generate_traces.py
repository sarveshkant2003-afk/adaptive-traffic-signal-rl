"""Generate precomputed replay traces for the Streamlit demo app.

Runs one full simulated hour per controller (Q-learning, fixed 30/30, fixed 40/20)
on a held-out seed, recording per-second vehicle positions and signal state for the
first SHOWCASE_SECONDS so the deployed app can replay the simulation without SUMO.

Run locally (SUMO required):  .venv/bin/python scripts/generate_traces.py
Outputs:  app/traces/{qlearning,fixed_equal,fixed_prop}.json.gz

NOTE: the constants below (TLS_ID, INCOMING_EDGES, GREEN_PHASES, DECISION_INTERVAL,
MIN_GREEN, YELLOW_TIME, discretize) are duplicated from the notebooks on purpose so this script runs on its own.
If you change them in the notebooks, mirror the change here and re-run.
"""

import gzip
import json
import os
import pickle
import shutil
import subprocess
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

# Make the venv's pip-installed SUMO binaries findable even when the venv isn't activated.
venv_bin = os.path.dirname(sys.executable)
os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")

import traci  # noqa: E402  (needs PATH set first so it can find the sumo binary)

# --- constants mirrored from the notebooks (see module docstring) ---
TLS_ID = "center"
INCOMING_EDGES = ["N2C", "E2C", "S2C", "W2C"]
GREEN_PHASES = [0, 2]
DECISION_INTERVAL = 5
MIN_GREEN = 10
YELLOW_TIME = 3

NET_DIR = "network"
SUMOCFG = f"{NET_DIR}/simulation.sumocfg"

SHOWCASE_SEED = 1000       # first held-out eval seed from notebook 2
SHOWCASE_SECONDS = 600     # how much of the hour the app animates
EPISODE_SECONDS = 3600     # summary metrics still cover the full hour
OUT_DIR = "app/traces"


def discretize(n):
    if n == 0:
        return 0
    if n <= 2:
        return 1
    if n <= 5:
        return 2
    return 3


def build_network_if_missing():
    """Recreate network/ exactly as notebook 1 does (cells 6 and 10)."""
    if os.path.exists(SUMOCFG):
        return
    os.makedirs(NET_DIR, exist_ok=True)
    with open(f"{NET_DIR}/nodes.nod.xml", "w") as f:
        f.write('''<nodes>
    <node id="center" x="0" y="0" type="traffic_light"/>
    <node id="north"  x="0" y="200" type="priority"/>
    <node id="south"  x="0" y="-200" type="priority"/>
    <node id="east"   x="200" y="0" type="priority"/>
    <node id="west"   x="-200" y="0" type="priority"/>
</nodes>''')
    with open(f"{NET_DIR}/edges.edg.xml", "w") as f:
        f.write('''<edges>
    <edge id="N2C" from="north" to="center" numLanes="1" speed="13.89"/>
    <edge id="C2N" from="center" to="north" numLanes="1" speed="13.89"/>
    <edge id="S2C" from="south" to="center" numLanes="1" speed="13.89"/>
    <edge id="C2S" from="center" to="south" numLanes="1" speed="13.89"/>
    <edge id="E2C" from="east" to="center" numLanes="1" speed="13.89"/>
    <edge id="C2E" from="center" to="east" numLanes="1" speed="13.89"/>
    <edge id="W2C" from="west" to="center" numLanes="1" speed="13.89"/>
    <edge id="C2W" from="center" to="west" numLanes="1" speed="13.89"/>
</edges>''')
    with open(f"{NET_DIR}/connections.con.xml", "w") as f:
        f.write('''<connections>
    <connection from="N2C" to="C2S" fromLane="0" toLane="0"/>
    <connection from="S2C" to="C2N" fromLane="0" toLane="0"/>
    <connection from="E2C" to="C2W" fromLane="0" toLane="0"/>
    <connection from="W2C" to="C2E" fromLane="0" toLane="0"/>
</connections>''')
    netconvert = shutil.which("netconvert")
    assert netconvert, "netconvert not found on PATH -- is SUMO installed in the venv?"
    result = subprocess.run(
        [netconvert,
         "--node-files", f"{NET_DIR}/nodes.nod.xml",
         "--edge-files", f"{NET_DIR}/edges.edg.xml",
         "--connection-files", f"{NET_DIR}/connections.con.xml",
         "-o", f"{NET_DIR}/intersection.net.xml"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    with open(f"{NET_DIR}/routes.rou.xml", "w") as f:
        f.write('''<routes>
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67"/>

    <route id="r_N2S" edges="N2C C2S"/>
    <route id="r_S2N" edges="S2C C2N"/>
    <route id="r_E2W" edges="E2C C2W"/>
    <route id="r_W2E" edges="W2C C2E"/>

    <flow id="f_N2S" type="car" route="r_N2S" begin="0" end="3600" vehsPerHour="400" departSpeed="max"/>
    <flow id="f_S2N" type="car" route="r_S2N" begin="0" end="3600" vehsPerHour="400" departSpeed="max"/>
    <flow id="f_E2W" type="car" route="r_E2W" begin="0" end="3600" vehsPerHour="200" departSpeed="max"/>
    <flow id="f_W2E" type="car" route="r_W2E" begin="0" end="3600" vehsPerHour="200" departSpeed="max"/>
</routes>''')
    with open(SUMOCFG, "w") as f:
        f.write('''<configuration>
    <input>
        <net-file value="intersection.net.xml"/>
        <route-files value="routes.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
    </time>
</configuration>''')
    print("Rebuilt network/")


# --- controllers, mirroring notebook 2 cell 3 ---

def rl_controller(state, Q, time_in_phase):
    if state in Q:
        return int(np.argmax(Q[state]))
    return 0


def make_fixed_controller(green_ns, green_ew):
    def controller(state, _, time_in_phase):
        green_idx = state[-1]
        target = green_ns if green_idx == 0 else green_ew
        return 1 if time_in_phase >= target else 0
    return controller


def run_and_record(seed, controller, controller_state):
    """Notebook 2's run_episode, plus per-second frame capture for the first SHOWCASE_SECONDS."""
    traci.start(["sumo", "-c", SUMOCFG, "--no-warnings", "--no-step-log", "--seed", str(seed)])
    traci.trafficlight.setPhase(TLS_ID, GREEN_PHASES[0])
    # netconvert shifts the network so coordinates start at (0,0); re-center on the junction
    # so the app can draw with (0,0) at the middle of the intersection.
    cx, cy = traci.junction.getPosition(TLS_ID)
    green_idx = 0
    time_in_phase = 0
    total_wait_proxy = 0
    depart_time = {}
    travel_times = []
    frames = []
    veh_ids = {}  # sumo vehicle id -> small int for compact JSON

    def capture_frame():
        now = int(traci.simulation.getTime())
        if now > SHOWCASE_SECONDS:
            return
        vehicles = []
        for vid in traci.vehicle.getIDList():
            x, y = traci.vehicle.getPosition(vid)
            idx = veh_ids.setdefault(vid, len(veh_ids))
            vehicles.append([idx, round(x - cx, 1), round(y - cy, 1),
                             int(traci.vehicle.getAngle(vid)),
                             1 if traci.vehicle.getSpeed(vid) < 0.1 else 0])
        frames.append({
            "t": now,
            "phase": traci.trafficlight.getPhase(TLS_ID),
            "queues": [traci.edge.getLastStepHaltingNumber(e) for e in INCOMING_EDGES],
            "vehicles": vehicles,
        })

    def sim_step():
        nonlocal total_wait_proxy
        traci.simulationStep()
        total_wait_proxy += sum(traci.edge.getLastStepHaltingNumber(e) for e in INCOMING_EDGES)
        now = traci.simulation.getTime()
        for vid in traci.simulation.getDepartedIDList():
            depart_time[vid] = now
        for vid in traci.simulation.getArrivedIDList():
            if vid in depart_time:
                travel_times.append(now - depart_time[vid])
        capture_frame()

    while traci.simulation.getTime() < EPISODE_SECONDS:
        state = tuple(discretize(traci.edge.getLastStepHaltingNumber(e)) for e in INCOMING_EDGES) + (green_idx,)
        action = controller(state, controller_state, time_in_phase)
        if action == 1 and time_in_phase >= MIN_GREEN:
            traci.trafficlight.setPhase(TLS_ID, GREEN_PHASES[green_idx] + 1)
            for _ in range(YELLOW_TIME):
                sim_step()
            green_idx = 1 - green_idx
            traci.trafficlight.setPhase(TLS_ID, GREEN_PHASES[green_idx])
            time_in_phase = 0
        for _ in range(DECISION_INTERVAL):
            sim_step()
        time_in_phase += DECISION_INTERVAL

    traci.close()
    summary = {
        "avg_wait_per_step": total_wait_proxy / (EPISODE_SECONDS / DECISION_INTERVAL),
        "mean_travel_time": float(np.mean(travel_times)) if travel_times else None,
        "n_completed_trips": len(travel_times),
    }
    return frames, summary


def main():
    build_network_if_missing()

    with open("qtable.pkl", "rb") as f:
        Q = pickle.load(f)
    print(f"Loaded Q-table ({len(Q)} states)")

    jobs = {
        "qlearning": ("Q-learning (trained)", rl_controller, Q),
        "fixed_equal": ("Fixed-time equal (30/30s)", make_fixed_controller(30, 30), None),
        "fixed_prop": ("Fixed-time proportional (40/20s)", make_fixed_controller(40, 20), None),
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    for key, (label, ctrl, ctrl_state) in jobs.items():
        print(f"Simulating: {label} (seed {SHOWCASE_SEED}, full hour) ...")
        frames, summary = run_and_record(SHOWCASE_SEED, ctrl, ctrl_state)
        trace = {
            "controller": label,
            "seed": SHOWCASE_SEED,
            "showcase_seconds": SHOWCASE_SECONDS,
            "episode_seconds": EPISODE_SECONDS,
            "summary": summary,
            "frames": frames,
        }
        path = f"{OUT_DIR}/{key}.json.gz"
        with gzip.open(path, "wt") as f:
            json.dump(trace, f, separators=(",", ":"))
        size_kb = os.path.getsize(path) / 1024
        print(f"  {len(frames)} frames | wait/step {summary['avg_wait_per_step']:.2f} | "
              f"travel {summary['mean_travel_time']:.1f}s | -> {path} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
