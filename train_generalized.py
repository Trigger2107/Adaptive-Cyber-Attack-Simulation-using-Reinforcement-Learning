"""
Train a GENERALIZED attacker — one agent across ALL defender modes.

Key difference from specialized training:
    - Defender mode is RANDOMIZED each episode
    - Defender ID is appended to state (state_dim = 8, not 7)
    - Agent learns to adapt strategy based on which defender it faces

Produces:
    attacker_generalized.pth
    rewards_generalized.npy
    privilege_generalized.npy
    discovery_generalized.npy
"""

import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from defender_advanced import defender_step

from cyberbattle._env.cyberbattle_env import CyberBattleEnv
from cyberbattle.samples.toyctf import toy_ctf

# ============================================================
# ENV SETUP
# ============================================================
env = CyberBattleEnv(initial_environment=toy_ctf.new_environment())

# ============================================================
# HYPERPARAMETERS
# ============================================================
MODES = ["random", "aggressive", "stealth", "patching"]

state_dim = 8    # 7 base features + 1 defender_id
action_dim = 10

gamma = 0.99
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995
lr = 1e-3
batch_size = 64
memory_size = 5000
episodes = 300
max_steps = 50
target_update_freq = 10

DETECTION_PENALTY = -50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Q-NETWORK (state_dim = 8 to include defender info)
# ============================================================
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

policy_net = QNetwork().to(device)
target_net = QNetwork().to(device)

model_path = "attacker_generalized.pth"
if os.path.exists(model_path):
    policy_net.load_state_dict(torch.load(model_path))
    print(f"Loaded existing model: {model_path}")

target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=lr)
loss_fn = nn.MSELoss()

# ============================================================
# REPLAY BUFFER
# ============================================================
memory = deque(maxlen=memory_size)

# ============================================================
# STATE REPRESENTATION (8-dim: 7 base + defender_id)
# ============================================================
def simplify_observation(obs, defender_id):
    base = np.array([
        obs['newly_discovered_nodes_count'],
        obs['lateral_move'],
        obs['escalation'],
        obs['customer_data_found'],
        obs['credential_cache_length'],
        np.sum(obs['nodes_privilegelevel']),
        obs['discovered_node_count']
    ], dtype=np.float32)
    return np.append(base, defender_id)

# ============================================================
# ACTION ENCODING (10-dim)
# ============================================================
def encode_action(action):
    vec = np.zeros(action_dim)
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

# ============================================================
# ACTION SELECTION (50 samples + escalation bias)
# ============================================================
def select_action(state, env, epsilon):
    if random.random() < epsilon:
        return env.sample_valid_action()

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

# ============================================================
# TRAINING STEP
# ============================================================
def train_step():
    if len(memory) < batch_size:
        return

    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states_t = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    actions_t = torch.tensor(np.array(actions), dtype=torch.float32).to(device)
    rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
    next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
    dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

    q_values = policy_net(states_t, actions_t)

    next_qs = []
    for ns in next_states_t:
        qs = []
        for _ in range(10):
            a = encode_action(env.sample_valid_action())
            a = torch.tensor(a, dtype=torch.float32).to(device)
            qs.append(target_net(ns, a).item())
        next_qs.append(max(qs))

    next_qs = torch.tensor(next_qs, dtype=torch.float32).unsqueeze(1).to(device)
    target = rewards_t + gamma * next_qs * (1 - dones_t)

    loss = loss_fn(q_values, target.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# ============================================================
# MAIN TRAINING LOOP
# ============================================================
reward_history = []
privilege_history = []
discovery_history = []
mode_history = []

print("=" * 60)
print("  GENERALIZED ATTACKER TRAINING (Multi-Defender)")
print("=" * 60)
print(f"  Defender modes: {MODES}")
print(f"  Episodes: {episodes} | Max Steps: {max_steps}")
print(f"  State dim: {state_dim} (7 + defender_id)")
print(f"  Detection Penalty: {DETECTION_PENALTY}")
print(f"  Device: {device}")
print("=" * 60)

for ep in range(episodes):
    # Randomize defender mode each episode
    mode = random.choice(MODES)
    defender_id = MODES.index(mode)

    obs, _ = env.reset()
    state = simplify_observation(obs, defender_id)

    total_reward = 0
    detected = False
    action_history = []

    for step in range(max_steps):
        prev_priv = obs['nodes_privilegelevel'].copy()
        prev_discovered = set(obs['_discovered_nodes'])

        action = select_action(state, env, epsilon)
        action_vec = encode_action(action)
        action_history.append(action)

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Reward shaping
        new_priv = obs['nodes_privilegelevel']
        new_discovered = set(obs['_discovered_nodes'])

        if np.sum(new_priv) > np.sum(prev_priv):
            reward += 20
        if len(new_discovered) > len(prev_discovered):
            reward += 10
        if reward == 0:
            reward -= 1

        # Defender acts every other step
        if step % 2 == 0:
            defender_result = defender_step(env, mode, step, action_history, obs)
        else:
            defender_result = 0

        if defender_result == "detected":
            done = True
            detected = True
            reward += DETECTION_PENALTY
        elif isinstance(defender_result, (int, float)):
            reward += defender_result

        next_state = simplify_observation(obs, defender_id)
        memory.append((state, action_vec, reward, next_state, done))
        train_step()

        state = next_state
        total_reward += reward

        if done:
            break

    # Track metrics
    reward_history.append(total_reward)
    final_priv = int(np.sum(obs['nodes_privilegelevel']))
    final_disc = len(obs['_discovered_nodes'])
    privilege_history.append(final_priv)
    discovery_history.append(final_disc)
    mode_history.append(mode)

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if ep % target_update_freq == 0:
        target_net.load_state_dict(policy_net.state_dict())

    status = "DETECTED" if detected else "OK"
    print(
        f"Ep {ep+1:3d}/{episodes} | "
        f"Def: {mode:10s} | "
        f"Reward: {total_reward:7.2f} | "
        f"Priv: {final_priv:2d} | "
        f"Nodes: {final_disc:2d} | "
        f"Eps: {epsilon:.3f} | "
        f"{status}"
    )


# ============================================================
# SAVE MODEL + METRICS
# ============================================================
torch.save(policy_net.state_dict(), model_path)
print(f"\nModel saved: {model_path}")

np.save("rewards_generalized.npy", reward_history)
np.save("privilege_generalized.npy", privilege_history)
np.save("discovery_generalized.npy", discovery_history)
print("Metrics saved: rewards_generalized.npy, privilege_generalized.npy, discovery_generalized.npy")


# ============================================================
# TEST AGAINST EACH DEFENDER MODE
# ============================================================
print("\n" + "=" * 60)
print("  TESTING GENERALIZED AGENT vs EACH DEFENDER")
print("=" * 60)

epsilon = 0

for test_mode in MODES:
    test_defender_id = MODES.index(test_mode)
    obs, _ = env.reset()
    state = simplify_observation(obs, test_defender_id)
    total_reward = 0
    test_detected = False

    for step in range(50):
        action = select_action(state, env, epsilon)
        obs, reward, terminated, truncated, _ = env.step(action)

        # Defender during test
        if step % 2 == 0:
            dr = defender_step(env, test_mode, step, [], obs)
            if dr == "detected":
                test_detected = True
                break

        state = simplify_observation(obs, test_defender_id)
        total_reward += reward
        if terminated or truncated:
            break

    det_str = " (DETECTED)" if test_detected else ""
    print(
        f"  vs {test_mode:12s} -> "
        f"Reward: {total_reward:7.2f} | "
        f"Priv: {np.sum(obs['nodes_privilegelevel']):2.0f} | "
        f"Nodes: {len(obs['_discovered_nodes']):2d}"
        f"{det_str}"
    )

print("\nTraining complete.")
