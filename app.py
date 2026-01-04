import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import sys
from copy import deepcopy

from Environment import Environment
from Swarm import Swarm
from Configurations import (
    config_num_of_agents,
    config_initial_swarm_positions,
    config_communication_range
)
import Utils


# ===============================
# print 重定向
# ===============================
class StreamlitStdout:
    def __init__(self, buffer):
        self.buffer = buffer

    def write(self, msg):
        msg = msg.rstrip()
        if msg:
            self.buffer.append(msg)

    def flush(self):
        pass


# ===============================
# SimulationController
# ===============================
class SimulationController:
    def __init__(self):
        self.env = None
        self.swarm = None
        self.step_count = 0
        self.max_steps = 450

        self.algo_mode = 5
        self.destroy_num = 50
        self.is_initialized = False

        self.positions = []
        self.remain_list = []
        self.clusters = 0
        self.connected = True
        self.destroyed_set = set()

    def initialize(self):
        print("Initializing simulation...")
        self.env = Environment()
        self.env.reset()

        enable_csds = (self.algo_mode == 0)
        self.swarm = Swarm(
            algorithm_mode=self.algo_mode,
            enable_csds=enable_csds,
            meta_param_use=True
        )
        self.swarm.reset(
            change_algorithm_mode=True,
            algorithm_mode=self.algo_mode
        )

        self.step_count = 0
        self.is_initialized = True
        self.update_data()
        print("Initialization complete.")

    def update_data(self):
        self.positions = deepcopy(self.env.environment_positions)
        self.remain_list = deepcopy(self.env.remain_list)
        self.clusters = self.env.check_the_clusters()
        self.connected = (self.clusters == 1)

    def destroy_now(self, num=None):
        if not self.is_initialized:
            print("Simulation not initialized")
            return False
        n = self.destroy_num if num is None else int(num)
        print(f"Destroying {n} UAVs...")
        _, destroy_list = self.env.stochastic_destroy(mode=2, num_of_destroyed=n)
        self.destroyed_set.update(destroy_list)
        self.swarm.destroy_happens(deepcopy(destroy_list), deepcopy(self.env.environment_positions))
        self.update_data()
        print(f"Destroyed indices: {destroy_list}")
        return True

    def step(self):
        if not self.is_initialized:
            print("Simulation not initialized")
            return False

        if self.step_count >= self.max_steps:
            print("Reached max steps")
            return False

        t0 = time.time()

        if self.step_count == 0:
            print("Executing destruction phase...")
            _, destroy_list = self.env.stochastic_destroy(
                mode=2,
                num_of_destroyed=self.destroy_num
            )
            self.destroyed_set.update(destroy_list)
            self.swarm.destroy_happens(
                deepcopy(destroy_list),
                deepcopy(self.env.environment_positions)
            )
            print("Destruction phase complete.")

        print(f"Step {self.step_count}: Taking actions...")
        actions, _ = self.swarm.take_actions()

        print("Actions taken. Updating state...")
        next_pos = self.env.next_state(deepcopy(actions))
        self.swarm.update_true_positions(next_pos)
        self.env.update()

        self.step_count += 1
        self.update_data()

        dt = time.time() - t0
        print(f"Step {self.step_count - 1} completed in {dt:.2f}s")
        return True


# ===============================
# Streamlit App
# ===============================
st.set_page_config(
    page_title="UAV Swarm Rescue Visualization",
    layout="wide"
)

st.title("🛩️ UAV Swarm Rescue Visualization")

# -------- Session State --------
if "sim" not in st.session_state:
    st.session_state.sim = SimulationController()

if "logs" not in st.session_state:
    st.session_state.logs = []

sim = st.session_state.sim

# 重定向 stdout / stderr
sys.stdout = StreamlitStdout(st.session_state.logs)
sys.stderr = StreamlitStdout(st.session_state.logs)

# ===============================
# Sidebar
# ===============================
with st.sidebar:
    st.header("Parameters")

    algo_mode = st.selectbox(
        "Algorithm",
        options=[0, 1, 2, 3, 4, 5],
        index=5,
        format_func=lambda x: {
            0: "CSDS",
            1: "HERO",
            2: "Centering",
            3: "SIDR",
            4: "GCN 2017",
            5: "CR-MGC (Proposed)"
        }[x]
    )

    destroy_num = st.slider(
        "Destroyed UAVs",
        min_value=0,
        max_value=190,
        value=50
    )

    if st.button("Initialize / Reset"):
        sim.algo_mode = algo_mode
        sim.destroy_num = destroy_num
        st.session_state.logs.clear()
        sim.initialize()
    
    if st.button("Destroy Now"):
        sim.destroy_now(destroy_num)

    if st.button("Single Step"):
        for i in range(5):
            sim.step()

    if st.button("Run All"):
        if not sim.is_initialized:
            print("Simulation not initialized")
        else:
            print("Run All started")
            while sim.step():
                time.sleep(0.05)
            print("Run All finished")


# ===============================
# Main Layout
# ===============================
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Swarm Visualization")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.set_aspect("equal")

    if sim.is_initialized:
        title = f"Step {sim.step_count} - "
        title += "Connected" if sim.connected else f"{sim.clusters} Clusters"
        ax.set_title(title, color="green" if sim.connected else "red")

        xs = [sim.positions[i][0] for i in sim.remain_list]
        ys = [sim.positions[i][1] for i in sim.remain_list]
        ax.scatter(xs, ys, c="blue", s=30)
        if sim.destroyed_set:
            dxs = [config_initial_swarm_positions[i][0] for i in sorted(sim.destroyed_set)]
            dys = [config_initial_swarm_positions[i][1] for i in sorted(sim.destroyed_set)]
            ax.scatter(dxs, dys, c="red", s=30, alpha=0.2)
    else:
        ax.set_title("Waiting for Initialization")

    st.pyplot(fig)

with col2:
    st.subheader("Metrics")
    st.metric("Step", sim.step_count)
    st.metric("Clusters", sim.clusters)
    st.metric(
        "Status",
        "Connected" if sim.connected else "Disconnected"
    )

    st.subheader("Logs")
    st.text_area(
        "Simulation Log",
        value="\n".join(st.session_state.logs[-500:]),
        height=100   # 👈 缩短高度
    )

    if st.button("Clear Logs"):
        st.session_state.logs.clear()
