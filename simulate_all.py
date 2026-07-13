"""
Simulate ALL trained attacker models and generate interactive PyVis graphs.

Loads each .pth model, runs it greedily in the environment (with its
matching defender), and produces an interactive HTML attack graph.

Output:
    sim_random.html        - random-trained agent
    sim_aggressive.html    - aggressive-trained agent
    sim_stealth.html       - stealth-trained agent
    sim_patching.html      - patching-trained agent
    sim_generalized.html   - generalized agent (vs random defender)
    sim_combined.html      - combined agent (no advanced defender)

Usage:
    python simulate_all.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
from pyvis.network import Network
from defender_advanced import defender_step as defender_step_advanced

from cyberbattle._env.cyberbattle_env import CyberBattleEnv
from cyberbattle.samples.toyctf import toy_ctf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Q-NETWORK DEFINITIONS (must match training architectures)
# ============================================================

class QNetwork_7(nn.Module):
    """state_dim=7, action_dim=10 (specialized + combined models)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7 + 10, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))


class QNetwork_8(nn.Module):
    """state_dim=8, action_dim=10 (generalized model with defender_id)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8 + 10, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))


# ============================================================
# HELPERS
# ============================================================
def simplify_observation_7(obs):
    return np.array([
        obs['newly_discovered_nodes_count'],
        obs['lateral_move'],
        obs['escalation'],
        obs['customer_data_found'],
        obs['credential_cache_length'],
        np.sum(obs['nodes_privilegelevel']),
        obs['discovered_node_count']
    ], dtype=np.float32)


def simplify_observation_8(obs, defender_id):
    base = simplify_observation_7(obs)
    return np.append(base, defender_id)


def encode_action(action):
    vec = np.zeros(10)
    if 'local_vulnerability' in action:
        vec[0] = 1
        vec[1:3] = action['local_vulnerability'][:2]
    elif 'remote_vulnerability' in action:
        vec[3] = 1
        vec[4:7] = action['remote_vulnerability'][:3]
    elif 'connect' in action:
        vec[7] = 1
        vec[8:10] = action['connect'][:2]
    return vec.astype(np.float32)


def select_action_greedy(state, env, policy_net):
    """Greedy action selection (epsilon=0), samples 50 candidates."""
    best_q = -1e9
    best_action = None
    for _ in range(50):
        action = env.sample_valid_action()
        action_vec = encode_action(action)
        s = torch.tensor(state, dtype=torch.float32).to(device)
        a = torch.tensor(action_vec, dtype=torch.float32).to(device)
        q_val = policy_net(s, a).item()
        if 'local_vulnerability' in action:
            q_val += 1
        if q_val > best_q:
            best_q = q_val
            best_action = action
    return best_action


def action_label(action):
    """Human-readable action label."""
    if 'local_vulnerability' in action:
        vals = action['local_vulnerability']
        return f"LOCAL[{vals[0]},{vals[1]}]"
    elif 'remote_vulnerability' in action:
        vals = action['remote_vulnerability']
        return f"REMOTE[{vals[0]},{vals[1]},{vals[2]}]"
    elif 'connect' in action:
        vals = action['connect']
        return f"CONNECT[{vals[0]},{vals[1]}]"
    return "UNKNOWN"


# ============================================================
# SIMULATION RUNNER
# ============================================================
def run_simulation(model_path, model_class, obs_fn, defender_mode, output_html, title, max_steps=50):
    """
    Load a model, run it in the environment, and produce a PyVis graph.

    Args:
        model_path:    path to .pth file
        model_class:   QNetwork_7 or QNetwork_8
        obs_fn:        function to convert obs -> state vector
        defender_mode: None (no defender) or one of the 4 modes
        output_html:   filename for the output HTML graph
        title:         display title for the graph
        max_steps:     max steps to run
    """
    if not os.path.exists(model_path):
        print(f"  [SKIP] {model_path} not found. Train it first.")
        return None

    # Load model
    policy_net = model_class().to(device)
    policy_net.load_state_dict(torch.load(model_path, map_location=device))
    policy_net.eval()

    # Setup environment
    env = CyberBattleEnv(initial_environment=toy_ctf.new_environment())
    obs, _ = env.reset()
    state = obs_fn(obs)

    # Track simulation
    step_log = []
    node_colors = {}
    edges = []

    # Color initial nodes
    for node in obs['_discovered_nodes']:
        node_colors[node] = 'blue'

    detected = False
    total_reward = 0

    for step in range(max_steps):
        prev_discovered = set(obs['_discovered_nodes'])

        action = select_action_greedy(state, env, policy_net)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        total_reward += reward

        discovered = obs['_discovered_nodes']
        privileges = obs['nodes_privilegelevel']

        # Update node colors
        for i, node in enumerate(discovered):
            if privileges[i] >= 2:
                node_colors[node] = 'red'
            elif privileges[i] == 1:
                node_colors[node] = 'green'
            else:
                if node not in node_colors:
                    node_colors[node] = 'blue'

        # Track edges (new connections)
        new_discovered = set(discovered)
        newly_found = new_discovered - prev_discovered
        if newly_found:
            # Connect new nodes to the source node of the action
            source_nodes = list(prev_discovered)
            if source_nodes:
                for new_node in newly_found:
                    edges.append((source_nodes[0], new_node))

        # Chain edges for discovered nodes
        for i in range(len(discovered) - 1):
            edge = (discovered[i], discovered[i + 1])
            if edge not in edges:
                edges.append(edge)

        # Defender
        if defender_mode and step % 2 == 0:
            dr = defender_step_advanced(env, defender_mode, step, step_log, obs)
            if dr == "detected":
                detected = True
                done = True

        step_log.append({
            'step': step,
            'action': action_label(action),
            'reward': reward,
            'priv': int(np.sum(privileges)),
            'nodes': len(discovered)
        })

        state = obs_fn(obs)

        if done:
            break

    # ========================
    # BUILD PyVis GRAPH
    # ========================
    net = Network(height="700px", width="100%", bgcolor="#0a0a0a", font_color="white")
    net.barnes_hut(gravity=-5000, central_gravity=0.3, spring_length=150)

    color_labels = {'red': 'ADMIN', 'green': 'USER', 'blue': 'DISCOVERED'}

    for node, color in node_colors.items():
        label = str(node)
        access = color_labels.get(color, 'UNKNOWN')
        hover = f"{label}\nAccess: {access}"
        net.add_node(node, label=label, color=color, title=hover, size=25)

    for u, v in edges:
        if u in node_colors and v in node_colors:
            net.add_edge(u, v, color="#444444")

    # Add info box as a special node
    status_str = "DETECTED" if detected else "SURVIVED"
    info = (
        f"{title}\n"
        f"Steps: {len(step_log)}\n"
        f"Reward: {total_reward:.0f}\n"
        f"Privilege: {step_log[-1]['priv'] if step_log else 0}\n"
        f"Nodes: {step_log[-1]['nodes'] if step_log else 0}\n"
        f"Status: {status_str}"
    )
    net.add_node("__INFO__", label=info, color="#222222", shape="box",
                 font={"color": "white", "size": 12}, size=10, x=-300, y=-300)

    net.save_graph(output_html)

    # Print summary
    final_priv = step_log[-1]['priv'] if step_log else 0
    final_nodes = step_log[-1]['nodes'] if step_log else 0
    print(
        f"  [{title:30s}] "
        f"Steps: {len(step_log):2d} | "
        f"Reward: {total_reward:7.2f} | "
        f"Priv: {final_priv} | "
        f"Nodes: {final_nodes} | "
        f"{status_str:8s} -> {output_html}"
    )

    return step_log


# ============================================================
# MAIN: SIMULATE ALL AGENTS
# ============================================================
if __name__ == "__main__":
    MODES = ["random", "aggressive", "stealth", "patching"]

    print("=" * 70)
    print("  SIMULATING ALL TRAINED ATTACKER AGENTS")
    print("=" * 70)

    # --- Part 1: Specialized agents ---
    print("\n--- SPECIALIZED AGENTS ---")
    for mode in MODES:
        model_path = f"attacker_{mode}.pth"
        output_html = f"sim_{mode}.html"
        title = f"Specialized vs {mode}"

        run_simulation(
            model_path=model_path,
            model_class=QNetwork_7,
            obs_fn=simplify_observation_7,
            defender_mode=mode,
            output_html=output_html,
            title=title
        )

    # --- Part 2: Generalized agent (test vs each defender) ---
    print("\n--- GENERALIZED AGENT ---")
    for mode in MODES:
        defender_id = MODES.index(mode)
        output_html = f"sim_generalized_vs_{mode}.html"
        title = f"Generalized vs {mode}"

        run_simulation(
            model_path="attacker_generalized.pth",
            model_class=QNetwork_8,
            obs_fn=lambda obs, did=defender_id: simplify_observation_8(obs, did),
            defender_mode=mode,
            output_html=output_html,
            title=title
        )

    # --- Part 3: Combined agent (from training_combined.py) ---
    print("\n--- COMBINED AGENT ---")
    run_simulation(
        model_path="attacker_combined.pth",
        model_class=QNetwork_7,
        obs_fn=simplify_observation_7,
        defender_mode=None,
        output_html="sim_combined.html",
        title="Combined (no adv defender)"
    )

    print("\n" + "=" * 70)
    print("  DONE! Open the .html files in your browser.")
    print("=" * 70)
