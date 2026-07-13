"""
Train SPECIALIZED attackers — one per defender mode.

Produces:
    attacker_random.pth       + rewards_random.npy, privilege_random.npy, discovery_random.npy
    attacker_aggressive.pth   + rewards_aggressive.npy, ...
    attacker_stealth.pth      + rewards_stealth.npy, ...
    attacker_patching.pth     + rewards_patching.npy, ...

Usage:
    python train_specialized.py                  # trains ALL modes
    python train_specialized.py aggressive       # trains only one mode
"""

import sys
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
# HYPERPARAMETERS
# ============================================================
state_dim = 7
action_dim = 10

gamma = 0.99
lr = 1e-3
batch_size = 64
memory_size = 5000
episodes = 300
max_steps = 50
target_update_freq = 10

DETECTION_PENALTY = -50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALL_MODES = ["random", "aggressive", "stealth", "patching"]

# ============================================================
# Q-NETWORK
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

# ============================================================
# HELPERS
# ============================================================
def simplify_observation(obs):
    return np.array([
        obs['newly_discovered_nodes_count'],
        obs['lateral_move'],
        obs['escalation'],
        obs['customer_data_found'],
        obs['credential_cache_length'],
        np.sum(obs['nodes_privilegelevel']),
        obs['discovered_node_count']
    ], dtype=np.float32)


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
# TRAINING FUNCTION (one mode at a time)
# ============================================================
def train_agent(defender_mode):
    print("\n" + "=" * 60)
    print(f"  TRAINING SPECIALIZED ATTACKER vs [{defender_mode.upper()}] defender")
    print("=" * 60)

    env = CyberBattleEnv(initial_environment=toy_ctf.new_environment())

    policy_net = QNetwork().to(device)
    target_net = QNetwork().to(device)

    model_path = f"attacker_{defender_mode}.pth"
    if os.path.exists(model_path):
        policy_net.load_state_dict(torch.load(model_path))
        print(f"  Resumed from: {model_path}")

    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    memory = deque(maxlen=memory_size)

    epsilon = 1.0

    # --- Action selection ---
    def select_action(state, epsilon):
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

    # --- Train step ---
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

    # --- Episode loop ---
    reward_history = []
    privilege_history = []
    discovery_history = []
    action_history = []   # for stealth defender

    for ep in range(episodes):
        obs, _ = env.reset()
        state = simplify_observation(obs)
        total_reward = 0
        detected = False
        action_history.clear()

        for step in range(max_steps):
            prev_priv = obs['nodes_privilegelevel'].copy()
            prev_discovered = set(obs['_discovered_nodes'])

            action = select_action(state, epsilon)
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
                defender_result = defender_step(env, defender_mode, step, action_history, obs)
            else:
                defender_result = 0

            if defender_result == "detected":
                done = True
                detected = True
                reward += DETECTION_PENALTY
            elif isinstance(defender_result, (int, float)):
                reward += defender_result

            next_state = simplify_observation(obs)
            memory.append((state, action_vec, reward, next_state, done))
            train_step()

            state = next_state
            total_reward += reward

            if done:
                break

        reward_history.append(total_reward)
        final_priv = int(np.sum(obs['nodes_privilegelevel']))
        final_disc = len(obs['_discovered_nodes'])
        privilege_history.append(final_priv)
        discovery_history.append(final_disc)

        epsilon = max(0.05, epsilon * 0.995)

        if ep % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        status = "DETECTED" if detected else "OK"
        print(
            f"  [{defender_mode:10s}] Ep {ep+1:3d}/{episodes} | "
            f"Reward: {total_reward:7.2f} | "
            f"Priv: {final_priv:2d} | "
            f"Nodes: {final_disc:2d} | "
            f"Eps: {epsilon:.3f} | "
            f"{status}"
        )

    # Save
    # ALL_MODES = ["random", "aggressive", "stealth", "patching"]
    torch.save(policy_net.state_dict(), model_path)
    np.save(f"rewards_{defender_mode}.npy", reward_history)
    np.save(f"privilege_{defender_mode}.npy", privilege_history)
    np.save(f"discovery_{defender_mode}.npy", discovery_history)

    print(f"\n  Saved: {model_path}")
    print(f"  Saved: rewards_{defender_mode}.npy, privilege_{defender_mode}.npy, discovery_{defender_mode}.npy")

    # Quick test
    print(f"\n  --- Testing [{defender_mode}] agent (greedy) ---")
    epsilon = 0
    obs, _ = env.reset()
    state = simplify_observation(obs)
    total_reward = 0
    for _ in range(50):
        action = select_action(state, epsilon)
        obs, reward, terminated, truncated, _ = env.step(action)
        state = simplify_observation(obs)
        total_reward += reward
        if terminated or truncated:
            break
    print(f"  Test Reward: {total_reward:.2f} | Priv: {np.sum(obs['nodes_privilegelevel'])} | Nodes: {len(obs['_discovered_nodes'])}")

    return reward_history


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # Allow training a single mode via CLI argument
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode not in ALL_MODES:
            print(f"Unknown mode: {mode}. Choose from: {ALL_MODES}")
            sys.exit(1)
        modes_to_train = [mode]
    else:
        modes_to_train = ALL_MODES

    print("=" * 60)
    print("  SPECIALIZED ATTACKER TRAINING")
    print(f"  Modes: {modes_to_train}")
    print(f"  Episodes per mode: {episodes}")
    print(f"  Device: {device}")
    print("=" * 60)

    all_rewards = {}
    for m in modes_to_train:
        all_rewards[m] = train_agent(m)

    print("\n" + "=" * 60)
    print("  ALL SPECIALIZED TRAINING COMPLETE")
    print("=" * 60)
    for m, rewards in all_rewards.items():
        avg_last_50 = np.mean(rewards[-50:]) if len(rewards) >= 50 else np.mean(rewards)
        print(f"  {m:12s} -> Avg reward (last 50 eps): {avg_last_50:.2f}")
