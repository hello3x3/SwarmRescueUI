import flet as ft
import matplotlib
# Must set backend before importing pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import base64
import io
import threading
import time
import asyncio
import traceback
import sys
from collections import deque
from copy import deepcopy

# Import project modules
# Ensure these are in the python path
from Environment import Environment
from Swarm import Swarm
from Configurations import config_num_of_agents, config_initial_swarm_positions, config_communication_range
import Utils

# Use Agg backend for non-interactive plotting
matplotlib.use('Agg')

class SimulationController:
    def __init__(self):
        self.env = None
        self.swarm = None
        self.running = False
        self.step_count = 0
        self.max_steps = 450
        self.lock = threading.Lock()
        
        # Default parameters
        self.algo_mode = 5  # CR-MGC
        self.destroy_num = 50
        self.is_initialized = False
        
        # Data for visualization
        self.positions = []
        self.remain_list = []
        self.clusters = 0
        self.connected = True
        
        # Logger callback (set by UI)
        self.log = lambda msg: None
        
    def initialize(self):
        print("Initializing simulation...")
        try:
            with self.lock:
                self.env = Environment()
                self.env.reset()
                
                enable_csds = (self.algo_mode == 0)
                print(f"Algorithm Mode: {self.algo_mode}, CSDS: {enable_csds}")
                # Use meta params if available (set to True as per default in Experiment script)
                self.swarm = Swarm(algorithm_mode=self.algo_mode, enable_csds=enable_csds, meta_param_use=True)
                self.swarm.reset(change_algorithm_mode=True, algorithm_mode=self.algo_mode)
                
                self.step_count = 0
                self.running = False
                self.is_initialized = True
                self.update_data()
                print("Initialization complete.")
        except Exception as e:
            print(f"Error during initialization: {e}")
            traceback.print_exc()
            self.is_initialized = False
            
    def update_data(self):
        self.positions = deepcopy(self.env.environment_positions)
        self.remain_list = deepcopy(self.env.remain_list)
        self.clusters = self.env.check_the_clusters()
        self.connected = (self.clusters == 1)
        
    def step(self):
        if not self.is_initialized:
            return False
            
        with self.lock:
            if self.step_count >= self.max_steps:
                self.running = False
                return False

            t_start = time.time()
            if self.step_count == 0:
                msg = "Executing destruction phase..."
                print(msg)
                self.log(msg)
                # Destruction phase
                _, destroy_list = self.env.stochastic_destroy(mode=2, num_of_destroyed=self.destroy_num)
                self.swarm.destroy_happens(deepcopy(destroy_list), deepcopy(self.env.environment_positions))
                msg = "Destruction phase complete."
                print(msg)
                self.log(msg)
            
            # Action phase
            msg = f"Step {self.step_count}: Taking actions..."
            print(msg)
            self.log(msg)
            try:
                actions, _ = self.swarm.take_actions()
            except Exception as e:
                err_msg = f"Error in take_actions: {e}"
                print(err_msg)
                self.log(err_msg)
                traceback.print_exc()
                return False
                
            print(f"Step {self.step_count}: Actions taken. Updating state...")
            self.log('Actions taken. Updating state...')
            next_pos = self.env.next_state(deepcopy(actions))
            self.swarm.update_true_positions(next_pos)
            self.env.update()
            
            self.step_count += 1
            self.update_data()
            
            dt = time.time() - t_start
            msg = f"Step {self.step_count-1} completed in {dt:.4f}s"
            print(msg)
            self.log(msg)
            return True

async def main(page: ft.Page):
    page.title = "UAV Swarm Rescue Visualization"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.window.width = 1200
    page.window.height = 900
    
    sim = SimulationController()
    
    # Route logs to UI via PubSub
    sim.log = lambda msg: page.pubsub.send_all('log:' + str(msg))
    
    # --- UI Components ---
    
    # Status Indicators
    step_text = ft.Text("Step: 0", size=20, weight=ft.FontWeight.BOLD)
    cluster_text = ft.Text("Clusters: -", size=20)
    status_text = ft.Text("Status: Not Initialized", color=ft.Colors.GREY, size=20)
    clock_text = ft.Text("", size=16, weight=ft.FontWeight.BOLD)
    
    # Log view
    log_view = ft.TextField(value="", multiline=True, read_only=True, min_lines=10, expand=True)
    
    # Plot Image
    plot_image = ft.Image(
        src="",
        width=800,
        height=600,
        fit=ft.ImageFit.CONTAIN,
        gapless_playback=True,
    )
    
    def generate_plot():
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Set limits based on config
        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 1000)
        ax.set_aspect('equal')
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        
        if sim.is_initialized:
            title = f"Step {sim.step_count} - " + ("Connected" if sim.connected else f"Disconnected ({sim.clusters} Clusters)")
            ax.set_title(title, color='green' if sim.connected else 'red')
            
            # Helper to get x, y of active agents
            xs = [sim.positions[i][0] for i in sim.remain_list]
            ys = [sim.positions[i][1] for i in sim.remain_list]
            
            # Plot active agents
            ax.scatter(xs, ys, c='blue', s=30, alpha=0.7, label='Active UAV')
            
        else:
            ax.set_title("Waiting for Initialization...")
            ax.text(500, 500, "Please Set Parameters and Initialize", ha='center', va='center', fontsize=12)
            
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Save to buffer
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def update_ui():
        print(f"Refreshing UI: step={sim.step_count}, clusters={sim.clusters}, connected={sim.connected}")
        step_text.value = f"Step: {sim.step_count}"
        cluster_text.value = f"Clusters: {sim.clusters}"
        status_text.value = "Status: Connected" if sim.connected else "Status: Disconnected"
        status_text.color = ft.Colors.GREEN if sim.connected else ft.Colors.RED
        b64 = generate_plot()
        print(f"Generated image size: {len(b64)} bytes (base64)")
        plot_image.src = "data:image/png;base64," + b64
        page.update()

    ps = page.pubsub
    def on_msg(msg):
        if msg == 'tick':
            update_ui()
        elif msg.startswith('btn:'):
            set_start_button_text(msg.split(':', 1)[1])
        elif msg.startswith('log:'):
            log_view.value = (log_view.value + ('\n' if log_view.value else '') + msg[4:])
            log_view.update()
        elif msg.startswith('clock:'):
            clock_text.value = msg[6:]
            clock_text.update()
    ps.subscribe(on_msg)

    log_queue = deque(maxlen=10000)
    msg_queue = deque(maxlen=10000)

    def run_log_dispatcher():
        while True:
            try:
                if log_queue:
                    batch = []
                    for _ in range(min(50, len(log_queue))):
                        batch.append(log_queue.popleft())
                    msg_queue.append('log:' + '\n'.join(batch))
                time.sleep(0.2)
            except:
                break
    threading.Thread(target=run_log_dispatcher, daemon=True).start()

    def run_msg_sender():
        while True:
            try:
                if msg_queue:
                    ps.send_all(msg_queue.popleft())
                time.sleep(0.05)
            except:
                break
    threading.Thread(target=run_msg_sender, daemon=True).start()

    def run_clock():
        while True:
            try:
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                msg_queue.append('clock:' + ts)
                time.sleep(1)
            except:
                break
    threading.Thread(target=run_clock, daemon=True).start()

    class UiStdout:
        def __init__(self, ps, original):
            self.ps = ps
            self.original = original
            self.buffer = ""
        def write(self, s):
            try:
                self.original.write(s)
            except:
                pass
            self.buffer += s
            while '\n' in self.buffer:
                line, self.buffer = self.buffer.split('\n', 1)
                if line:
                    log_queue.append(line)
        def flush(self):
            try:
                self.original.flush()
            except:
                pass

    sys.stdout = UiStdout(ps, sys.stdout)
    sys.stderr = UiStdout(ps, sys.stderr)

    # Event Handlers
    def on_init_click(e):
        print("Init button clicked")
        sim.algo_mode = int(algo_dropdown.value)
        sim.destroy_num = int(destroy_slider.value)
        sim.initialize()
        status_text.value = "Initialized"
        status_text.color = ft.Colors.BLUE
        update_ui()
        
    def on_step_click(e):
        print("Single Step clicked")
        if not sim.is_initialized:
            print("Simulation not initialized")
            return
        def run_single_step():
            ok = sim.step()
            msg_queue.append('tick')
            if not ok:
                msg_queue.append('btn:Start')
        threading.Thread(target=run_single_step, daemon=True).start()
        
    def run_simulation():
        print("Simulation thread started.")
        try:
            while sim.running:
                should_continue = sim.step()
                if not should_continue:
                    print("Simulation finished or stopped.")
                    sim.running = False
                    msg_queue.append('btn:Start')
                    break
                msg_queue.append('tick')
                time.sleep(0.05)
        except Exception as e:
            print(f"Error in simulation thread: {e}")
            traceback.print_exc()
            sim.running = False
            msg_queue.append('btn:Start')
            
    def on_start_click(e):
        if not sim.is_initialized:
            return
            
        if not sim.running:
            sim.running = True
            start_btn.text = "Pause"
            start_btn.update()
            threading.Thread(target=run_simulation, daemon=True).start()
        else:
            sim.running = False
            start_btn.text = "Start"
            start_btn.update()

    # Controls
    algo_dropdown = ft.Dropdown(
        label="Algorithm",
        value="5",
        options=[
            ft.dropdown.Option("0", "CSDS"),
            ft.dropdown.Option("1", "HERO"),
            ft.dropdown.Option("2", "Centering"),
            ft.dropdown.Option("3", "SIDR"),
            ft.dropdown.Option("4", "GCN 2017"),
            ft.dropdown.Option("5", "CR-MGC (Proposed)"),
        ],
        width=250
    )
    
    destroy_slider = ft.Slider(
        min=0, max=190, value=50, label="Destroyed UAVs: {value}",
        width=300
    )
    
    init_btn = ft.ElevatedButton("Initialize / Reset", on_click=on_init_click, bgcolor=ft.Colors.BLUE_100, color=ft.Colors.BLACK)
    start_btn = ft.ElevatedButton("Start", on_click=on_start_click, bgcolor=ft.Colors.GREEN_100, color=ft.Colors.BLACK)
    step_btn = ft.ElevatedButton("Single Step", on_click=on_step_click)
    
    # Layout
    
    # Top Control Bar
    control_bar = ft.Container(
        content=ft.Row(
            [
                ft.Column([ft.Text("Parameters", size=16, weight=ft.FontWeight.BOLD), algo_dropdown, destroy_slider], spacing=10),
                ft.VerticalDivider(width=1, color=ft.Colors.GREY_300),
                ft.Column([ft.Text("Actions", size=16, weight=ft.FontWeight.BOLD), init_btn, ft.Row([start_btn, step_btn])], spacing=10),
                ft.VerticalDivider(width=1, color=ft.Colors.GREY_300),
                ft.Column([ft.Text("Metrics", size=16, weight=ft.FontWeight.BOLD), step_text, cluster_text, status_text], spacing=5),
            ],
            alignment=ft.MainAxisAlignment.SPACE_EVENLY,
        ),
        padding=15,
        border=ft.border.all(1, ft.Colors.GREY_300),
        border_radius=10,
        bgcolor=ft.Colors.WHITE,
        margin=ft.margin.only(bottom=20)
    )
    
    clock_bar = ft.Container(content=ft.Row([clock_text], alignment=ft.MainAxisAlignment.START), padding=5)
    page.add(
        clock_bar,
        control_bar,
        ft.Row([
            ft.Container(
                content=plot_image,
                alignment=ft.alignment.center,
                expand=True,
                border=ft.border.all(1, ft.Colors.GREY_200),
                border_radius=10,
                padding=10
            ),
            ft.Container(
                content=log_view,
                alignment=ft.alignment.top_left,
                expand=True,
                border=ft.border.all(1, ft.Colors.GREY_200),
                border_radius=10,
                padding=10
            )
        ], expand=True)
    )

    # Initial render
    plot_image.src = "data:image/png;base64," + generate_plot()
    page.update()

if __name__ == "__main__":
    ft.app(target=main, view=ft.AppView.WEB_BROWSER)
