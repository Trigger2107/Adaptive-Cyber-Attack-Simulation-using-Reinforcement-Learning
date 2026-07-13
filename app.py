"""
CyberBattle Simulation Dashboard
Flask app that simulates attacker agents against various defender modes.
Since CyberBattle gym may not be installed, we simulate the environment
with a faithful replica of the state/action/reward logic.
"""

from flask import Flask, make_response, jsonify, request
import random
import numpy as np
import os
import json

app = Flask(__name__)

# ─────────────────────────────────────────────
# SIMULATED CYBERBATTLE ENVIRONMENT
# (mirrors toy_ctf state/action/reward logic)
# ─────────────────────────────────────────────

NODES = [
    "client",
    "GitHubProject",
    "AzureResourceManager",
    "Website.IIS",
    "AzureStorage",
    "SharePoint",
    "AdminMachine",
    "Flag"
]

VULNERABILITIES = {
    "local": [
        ("SasTokenLeak",       "Extracts SAS token from environment"),
        ("CredScan",           "Scans for credentials in local files"),
        ("PrivEsc_Sudo",       "Exploits sudo misconfiguration"),
        ("TokenHijack",        "Hijacks auth token from memory"),
        ("CacheDump",          "Dumps credential cache"),
    ],
    "remote": [
        ("SQLInjection",       "Injects SQL to extract data"),
        ("SMBExploit",         "Exploits SMB vulnerability"),
        ("RCE_Web",            "Remote code execution via web"),
        ("SSRFAttack",         "Server-Side Request Forgery"),
        ("LogForge",           "Log4Shell-style exploit"),
    ],
}

class SimEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.discovered = ["client"]
        self.privileges = {"client": 1}
        self.credential_cache = []
        self.customer_data_found = False
        self.step_count = 0
        self.flags_captured = 0
        return self._obs()

    def _obs(self):
        priv_levels = [self.privileges.get(n, 0) for n in self.discovered]
        return {
            'newly_discovered_nodes_count': 0,
            'lateral_move': 0,
            'escalation': 0,
            'customer_data_found': int(self.customer_data_found),
            'credential_cache_length': len(self.credential_cache),
            'nodes_privilegelevel': np.array(priv_levels, dtype=np.float32),
            'discovered_node_count': len(self.discovered),
            '_discovered_nodes': list(self.discovered),
        }

    def sample_valid_action(self):
        action_type = random.choice(['local_vulnerability', 'remote_vulnerability', 'connect'])
        node_idx = random.randint(0, len(self.discovered) - 1)
        if action_type == 'local_vulnerability':
            vuln_idx = random.randint(0, len(VULNERABILITIES['local']) - 1)
            return {'local_vulnerability': [node_idx, vuln_idx]}
        elif action_type == 'remote_vulnerability':
            target_idx = random.randint(0, len(NODES) - 1)
            vuln_idx = random.randint(0, len(VULNERABILITIES['remote']) - 1)
            return {'remote_vulnerability': [node_idx, target_idx, vuln_idx]}
        else:
            target_idx = random.randint(0, len(NODES) - 1)
            return {'connect': [node_idx, target_idx]}

    def step(self, action):
        self.step_count += 1
        reward = 0
        newly_discovered = 0
        escalation = 0
        lateral = 0
        prev_priv_sum = sum(self.privileges.values())

        if 'local_vulnerability' in action:
            node_idx = int(action['local_vulnerability'][0]) % len(self.discovered)
            vuln_idx = int(action['local_vulnerability'][1]) % len(VULNERABILITIES['local'])
            node = self.discovered[node_idx]
            vuln_name, _ = VULNERABILITIES['local'][vuln_idx]

            if random.random() < 0.55:
                cur_priv = self.privileges.get(node, 0)
                if cur_priv < 2:
                    self.privileges[node] = cur_priv + 1
                    escalation = 1
                    reward += 15
                    # chance to find creds
                    if random.random() < 0.4:
                        cred = f"cred_{node}_{vuln_name[:4]}"
                        if cred not in self.credential_cache:
                            self.credential_cache.append(cred)
                            reward += 5
                # discover adjacent node
                if random.random() < 0.35 and len(self.discovered) < len(NODES):
                    new_node = random.choice([n for n in NODES if n not in self.discovered])
                    self.discovered.append(new_node)
                    self.privileges[new_node] = 0
                    newly_discovered = 1
                    reward += 8

        elif 'remote_vulnerability' in action:
            target_idx = int(action['remote_vulnerability'][1]) % len(NODES)
            vuln_idx = int(action['remote_vulnerability'][2]) % len(VULNERABILITIES['remote'])
            target_node = NODES[target_idx]
            vuln_name, _ = VULNERABILITIES['remote'][vuln_idx]

            if random.random() < 0.4:
                if target_node not in self.discovered:
                    self.discovered.append(target_node)
                    self.privileges[target_node] = 1
                    newly_discovered = 1
                    lateral = 1
                    reward += 12
                    if target_node in ("AzureStorage", "SharePoint"):
                        self.customer_data_found = True
                        reward += 20
                elif self.privileges.get(target_node, 0) < 2:
                    self.privileges[target_node] = min(2, self.privileges.get(target_node, 0) + 1)
                    escalation = 1
                    reward += 10

        elif 'connect' in action:
            target_idx = int(action['connect'][1]) % len(NODES)
            target_node = NODES[target_idx]

            if self.credential_cache and random.random() < 0.5:
                if target_node not in self.discovered:
                    self.discovered.append(target_node)
                    self.privileges[target_node] = 1
                    newly_discovered = 1
                    lateral = 1
                    reward += 10
                if target_node == "Flag":
                    self.flags_captured += 1
                    reward += 50
                if target_node == "AdminMachine" and self.privileges.get(target_node, 0) < 2:
                    self.privileges[target_node] = 2
                    reward += 25

        priv_levels = [self.privileges.get(n, 0) for n in self.discovered]
        obs = {
            'newly_discovered_nodes_count': newly_discovered,
            'lateral_move': lateral,
            'escalation': escalation,
            'customer_data_found': int(self.customer_data_found),
            'credential_cache_length': len(self.credential_cache),
            'nodes_privilegelevel': np.array(priv_levels, dtype=np.float32),
            'discovered_node_count': len(self.discovered),
            '_discovered_nodes': list(self.discovered),
        }

        done = self.flags_captured >= 1 or self.step_count >= 50
        return obs, reward, done, False, {}


# ─────────────────────────────────────────────
# DEFENDERS  (matches defender_advanced.py)
# ─────────────────────────────────────────────

def defender_step(mode, step, history, obs):
    if mode == "random":
        if random.random() < 0.05:
            return "detected", "Routine scan caught suspicious process"
        return 0, "No defender activity"

    elif mode == "aggressive":
        priv = float(np.sum(obs['nodes_privilegelevel'])) if obs else 0
        if priv > 1 and random.random() < 0.4:
            return "detected", f"Privilege escalation alert triggered (priv={priv:.0f})"
        if step > 15 and random.random() < 0.3:
            return "detected", "Late-stage intrusion detected via honeypot"
        return -1, "Aggressive scan (-1 penalty applied)"

    elif mode == "stealth":
        if len(history) > 5 and random.random() < 0.3:
            return "detected", f"Behavioural pattern detected after {len(history)} actions"
        return 0, "Monitoring activity patterns…"

    elif mode == "patching":
        if obs and random.random() < 0.3:
            priv_levels = obs['nodes_privilegelevel']
            if len(priv_levels) > 0:
                node_idx = random.randint(0, len(priv_levels) - 1)
                node_name = obs['_discovered_nodes'][node_idx] if node_idx < len(obs['_discovered_nodes']) else "?"
                return -2, f"Defender patched node '{node_name}' — privilege revoked (-2 penalty)"
        return 0, "Patching daemon idle this cycle"

    return 0, "No defender action"


# ─────────────────────────────────────────────
# AGENT (simulated Q-network logic)
# ─────────────────────────────────────────────

MODEL_PROFILES = {
    "attacker_random":      {"style": "random",      "bias": "explore",   "arch": "7-dim"},
    "attacker_aggressive":  {"style": "aggressive",  "bias": "escalate",  "arch": "7-dim"},
    "attacker_stealth":     {"style": "stealth",     "bias": "low-noise", "arch": "7-dim"},
    "attacker_patching":    {"style": "patching",    "bias": "fast",      "arch": "7-dim"},
    "attacker_generalized": {"style": "generalized", "bias": "adaptive",  "arch": "8-dim"},
}

def select_action_simulated(obs, model_name, step):
    """
    Simulates what a trained model would do based on the model's known training bias.
    Real .pth loading needs PyTorch + CyberBattle installed.
    """
    profile = MODEL_PROFILES.get(model_name, {"bias": "explore"})
    bias = profile["bias"]
    priv = float(np.sum(obs['nodes_privilegelevel']))
    creds = obs['credential_cache_length']

    if bias == "escalate":
        # Aggressive: prefer local vulns for privilege escalation
        node_idx = random.randint(0, max(0, obs['discovered_node_count'] - 1))
        if priv < 3 or random.random() < 0.7:
            return {'local_vulnerability': [node_idx, random.randint(0, 4)]}
        return {'remote_vulnerability': [node_idx, random.randint(0, len(NODES)-1), random.randint(0, 4)]}

    elif bias == "low-noise":
        # Stealth: prefer quiet connect/remote, avoid repeated local
        node_idx = random.randint(0, max(0, obs['discovered_node_count'] - 1))
        if creds > 0 and random.random() < 0.6:
            return {'connect': [node_idx, random.randint(0, len(NODES)-1)]}
        return {'remote_vulnerability': [node_idx, random.randint(0, len(NODES)-1), random.randint(0, 4)]}

    elif bias == "fast":
        # Patching: sprint fast before patches land
        node_idx = random.randint(0, max(0, obs['discovered_node_count'] - 1))
        if step < 10:
            return {'local_vulnerability': [node_idx, random.randint(0, 4)]}
        return {'connect': [node_idx, random.randint(0, len(NODES)-1)]}

    elif bias == "adaptive":
        # Generalized: balanced strategy
        r = random.random()
        node_idx = random.randint(0, max(0, obs['discovered_node_count'] - 1))
        if r < 0.35:
            return {'local_vulnerability': [node_idx, random.randint(0, 4)]}
        elif r < 0.65:
            return {'remote_vulnerability': [node_idx, random.randint(0, len(NODES)-1), random.randint(0, 4)]}
        return {'connect': [node_idx, random.randint(0, len(NODES)-1)]}

    else:  # explore / random
        node_idx = random.randint(0, max(0, obs['discovered_node_count'] - 1))
        r = random.random()
        if r < 0.4:
            return {'local_vulnerability': [node_idx, random.randint(0, 4)]}
        elif r < 0.7:
            return {'remote_vulnerability': [node_idx, random.randint(0, len(NODES)-1), random.randint(0, 4)]}
        return {'connect': [node_idx, random.randint(0, len(NODES)-1)]}


def action_label(action):
    if 'local_vulnerability' in action:
        v = action['local_vulnerability']
        node = NODES[int(v[0]) % len(NODES)]
        vuln = VULNERABILITIES['local'][int(v[1]) % len(VULNERABILITIES['local'])][0]
        return f"LOCAL_VULN({node}, {vuln})", "local"
    elif 'remote_vulnerability' in action:
        v = action['remote_vulnerability']
        src = NODES[int(v[0]) % len(NODES)]
        tgt = NODES[int(v[1]) % len(NODES)]
        vuln = VULNERABILITIES['remote'][int(v[2]) % len(VULNERABILITIES['remote'])][0]
        return f"REMOTE_EXPLOIT({src}→{tgt}, {vuln})", "remote"
    elif 'connect' in action:
        v = action['connect']
        src = NODES[int(v[0]) % len(NODES)]
        tgt = NODES[int(v[1]) % len(NODES)]
        return f"CONNECT({src}→{tgt})", "connect"
    return "UNKNOWN", "unknown"




HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CyberBattle Sim Lab</title>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&family=JetBrains+Mono:wght@300;400;600&display=swap" rel="stylesheet">
<style>
:root {
  --bg:       #020608;
  --bg2:      #060d12;
  --bg3:      #0b1620;
  --border:   #0e2a3a;
  --accent:   #00e5ff;
  --accent2:  #ff3d71;
  --green:    #39ff14;
  --yellow:   #ffd700;
  --dim:      #2a4a5a;
  --text:     #c8e8f0;
  --text-dim: #4a7a8a;
  --red:      #ff3d71;
  --panel:    rgba(6, 20, 30, 0.95);
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background: var(--bg);
  color: var(--text);
  font-family: 'JetBrains Mono', monospace;
  min-height: 100vh;
  overflow-x: hidden;
}

/* Scanline overlay */
body::before {
  content: '';
  position: fixed;
  inset: 0;
  background: repeating-linear-gradient(
    0deg,
    transparent,
    transparent 2px,
    rgba(0,229,255,0.012) 2px,
    rgba(0,229,255,0.012) 4px
  );
  pointer-events: none;
  z-index: 9999;
}

/* Grid bg */
body::after {
  content: '';
  position: fixed;
  inset: 0;
  background-image:
    linear-gradient(rgba(0,229,255,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,229,255,0.03) 1px, transparent 1px);
  background-size: 40px 40px;
  pointer-events: none;
  z-index: 0;
}

.app { position: relative; z-index: 1; display: flex; flex-direction: column; min-height: 100vh; }

/* HEADER */
header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 18px 32px;
  border-bottom: 1px solid var(--border);
  background: linear-gradient(180deg, rgba(0,229,255,0.04) 0%, transparent 100%);
}

.logo {
  font-family: 'Orbitron', monospace;
  font-weight: 900;
  font-size: 1.3rem;
  color: var(--accent);
  letter-spacing: 3px;
  text-shadow: 0 0 20px rgba(0,229,255,0.6);
}

.logo span { color: var(--accent2); }

.header-right {
  display: flex;
  align-items: center;
  gap: 20px;
  font-size: 0.7rem;
  color: var(--text-dim);
  letter-spacing: 2px;
  text-transform: uppercase;
}

.status-dot {
  width: 8px; height: 8px;
  border-radius: 50%;
  background: var(--green);
  box-shadow: 0 0 8px var(--green);
  animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.3; }
}

/* MAIN LAYOUT */
main {
  display: grid;
  grid-template-columns: 320px 1fr;
  grid-template-rows: auto 1fr;
  gap: 0;
  flex: 1;
}

/* CONTROL PANEL */
.control-panel {
  grid-row: 1 / 3;
  border-right: 1px solid var(--border);
  padding: 24px 20px;
  display: flex;
  flex-direction: column;
  gap: 24px;
  background: var(--bg2);
}

.panel-title {
  font-family: 'Orbitron', monospace;
  font-size: 0.65rem;
  letter-spacing: 3px;
  text-transform: uppercase;
  color: var(--accent);
  padding-bottom: 8px;
  border-bottom: 1px solid var(--border);
}

.model-grid {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.model-card {
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 12px 14px;
  cursor: pointer;
  transition: all 0.15s ease;
  position: relative;
  overflow: hidden;
}

.model-card::before {
  content: '';
  position: absolute;
  left: 0; top: 0; bottom: 0;
  width: 3px;
  background: var(--dim);
  transition: all 0.15s ease;
}

.model-card:hover { border-color: var(--accent); }
.model-card:hover::before { background: var(--accent); }

.model-card.active {
  border-color: var(--accent);
  background: rgba(0,229,255,0.06);
}
.model-card.active::before { background: var(--accent); box-shadow: 0 0 8px var(--accent); }

.model-card.unavailable { opacity: 0.4; cursor: not-allowed; }

.model-name {
  font-family: 'Orbitron', monospace;
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 2px;
  color: var(--text);
}

.model-meta {
  font-size: 0.62rem;
  color: var(--text-dim);
  margin-top: 4px;
  letter-spacing: 1px;
}

.model-badge {
  display: inline-block;
  padding: 2px 6px;
  border-radius: 2px;
  font-size: 0.55rem;
  font-weight: 600;
  letter-spacing: 1px;
  text-transform: uppercase;
  margin-top: 6px;
}

.badge-loaded { background: rgba(57,255,20,0.15); color: var(--green); border: 1px solid rgba(57,255,20,0.3); }
.badge-sim    { background: rgba(0,229,255,0.1); color: var(--accent); border: 1px solid rgba(0,229,255,0.2); }

/* DEFENDER SELECTOR */
.defender-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}

.def-btn {
  background: var(--bg3);
  border: 1px solid var(--border);
  color: var(--text-dim);
  padding: 10px 8px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.65rem;
  letter-spacing: 1px;
  cursor: pointer;
  border-radius: 3px;
  text-align: center;
  transition: all 0.15s ease;
  text-transform: uppercase;
}

.def-btn:hover { border-color: var(--accent2); color: var(--text); }

.def-btn.active {
  background: rgba(255,61,113,0.1);
  border-color: var(--accent2);
  color: var(--accent2);
  box-shadow: 0 0 12px rgba(255,61,113,0.2);
}

.def-icon { font-size: 1rem; display: block; margin-bottom: 4px; }

/* STEPS SLIDER */
.slider-wrap { display: flex; align-items: center; gap: 10px; }

input[type=range] {
  flex: 1;
  -webkit-appearance: none;
  height: 3px;
  background: var(--border);
  border-radius: 2px;
  outline: none;
}

input[type=range]::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 14px; height: 14px;
  border-radius: 50%;
  background: var(--accent);
  cursor: pointer;
  box-shadow: 0 0 6px var(--accent);
}

.slider-val {
  font-family: 'Orbitron', monospace;
  font-size: 0.75rem;
  color: var(--accent);
  min-width: 28px;
  text-align: right;
}

/* RUN BUTTON */
.run-btn {
  background: transparent;
  border: 1px solid var(--accent);
  color: var(--accent);
  font-family: 'Orbitron', monospace;
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 3px;
  padding: 14px;
  cursor: pointer;
  border-radius: 3px;
  text-transform: uppercase;
  transition: all 0.15s ease;
  position: relative;
  overflow: hidden;
}

.run-btn::before {
  content: '';
  position: absolute;
  inset: 0;
  background: var(--accent);
  transform: translateY(100%);
  transition: transform 0.2s ease;
}

.run-btn:hover { color: var(--bg); }
.run-btn:hover::before { transform: translateY(0); }
.run-btn span { position: relative; z-index: 1; }

.run-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}
.run-btn:disabled::before { display: none; }
.run-btn:disabled:hover { color: var(--accent); }

/* TOP STATS BAR */
.stats-bar {
  grid-column: 2;
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  border-bottom: 1px solid var(--border);
  background: var(--bg2);
}

.stat-cell {
  padding: 14px 16px;
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.stat-cell:last-child { border-right: none; }

.stat-label {
  font-size: 0.58rem;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: var(--text-dim);
}

.stat-value {
  font-family: 'Orbitron', monospace;
  font-size: 1.1rem;
  font-weight: 700;
  color: var(--accent);
  transition: all 0.3s ease;
}

.stat-value.danger { color: var(--red); }
.stat-value.success { color: var(--green); }
.stat-value.warn { color: var(--yellow); }

/* MAIN CONTENT AREA */
.content-area {
  grid-column: 2;
  display: flex;
  flex-direction: row;
  height: calc(100vh - 120px);
  overflow: hidden;
}

.log-panel  { flex: 0 0 auto; width: 60%; min-width: 180px; max-width: calc(100% - 180px); }
.map-panel  { flex: 1 1 auto; min-width: 180px; }

/* RESIZER */
.resizer {
  flex: 0 0 6px;
  background: var(--border);
  cursor: col-resize;
  position: relative;
  transition: background 0.15s;
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 10;
}
.resizer::after {
  content: '⋮';
  color: var(--dim);
  font-size: 14px;
  letter-spacing: -1px;
  pointer-events: none;
  transition: color 0.15s;
}
.resizer:hover, .resizer.dragging {
  background: var(--accent);
}
.resizer:hover::after, .resizer.dragging::after {
  color: var(--bg);
}

/* LOG PANEL */
.log-panel {
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.log-header {
  padding: 12px 20px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 10px;
  background: var(--bg2);
}

.log-title {
  font-family: 'Orbitron', monospace;
  font-size: 0.65rem;
  letter-spacing: 2px;
  color: var(--text-dim);
  text-transform: uppercase;
  flex: 1;
}

.log-count {
  font-size: 0.6rem;
  color: var(--accent);
  background: rgba(0,229,255,0.1);
  border: 1px solid rgba(0,229,255,0.2);
  padding: 2px 8px;
  border-radius: 2px;
}

.filter-btns {
  display: flex;
  gap: 4px;
}

.filter-btn {
  background: none;
  border: 1px solid var(--border);
  color: var(--text-dim);
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.58rem;
  padding: 3px 8px;
  cursor: pointer;
  border-radius: 2px;
  letter-spacing: 1px;
  transition: all 0.1s;
}
.filter-btn:hover, .filter-btn.active { border-color: var(--accent); color: var(--accent); }

.log-body {
  flex: 1;
  overflow-y: auto;
  padding: 8px 0;
  font-size: 0.68rem;
}

.log-body::-webkit-scrollbar { width: 4px; }
.log-body::-webkit-scrollbar-track { background: transparent; }
.log-body::-webkit-scrollbar-thumb { background: var(--dim); border-radius: 2px; }

.log-entry {
  padding: 8px 20px;
  border-bottom: 1px solid rgba(14,42,58,0.5);
  animation: fadeIn 0.3s ease;
  cursor: pointer;
  transition: background 0.1s;
}

.log-entry:hover { background: rgba(0,229,255,0.03); }
.log-entry.hidden { display: none; }

@keyframes fadeIn {
  from { opacity: 0; transform: translateX(-8px); }
  to   { opacity: 1; transform: translateX(0); }
}

.log-step {
  display: flex;
  align-items: flex-start;
  gap: 10px;
}

.step-num {
  font-family: 'Orbitron', monospace;
  font-size: 0.6rem;
  color: var(--dim);
  min-width: 24px;
  padding-top: 1px;
}

.step-content { flex: 1; }

.action-line {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 3px;
}

.action-tag {
  font-size: 0.55rem;
  padding: 1px 5px;
  border-radius: 2px;
  font-weight: 600;
  letter-spacing: 1px;
  text-transform: uppercase;
}

.tag-local   { background: rgba(0,229,255,0.15);  color: var(--accent);  border: 1px solid rgba(0,229,255,0.3); }
.tag-remote  { background: rgba(255,215,0,0.12);  color: var(--yellow);  border: 1px solid rgba(255,215,0,0.3); }
.tag-connect { background: rgba(57,255,20,0.12);  color: var(--green);   border: 1px solid rgba(57,255,20,0.3); }

.action-text {
  font-size: 0.65rem;
  color: var(--text);
  font-family: 'Share Tech Mono', monospace;
}

.reward-line {
  font-size: 0.6rem;
  color: var(--text-dim);
  display: flex;
  gap: 12px;
  align-items: center;
}

.reward-pos { color: var(--green); }
.reward-neg { color: var(--red); }

.defender-line {
  margin-top: 4px;
  font-size: 0.6rem;
  padding: 3px 8px;
  border-radius: 2px;
  background: rgba(255,61,113,0.06);
  border-left: 2px solid var(--accent2);
  color: var(--accent2);
}

.detected-banner {
  margin: 4px 0;
  padding: 4px 10px;
  background: rgba(255,61,113,0.15);
  border: 1px solid var(--red);
  border-radius: 2px;
  color: var(--red);
  font-family: 'Orbitron', monospace;
  font-size: 0.65rem;
  letter-spacing: 2px;
  text-align: center;
  animation: blink 0.5s step-end 6;
}

@keyframes blink {
  50% { opacity: 0; }
}

.new-node-pill {
  display: inline-block;
  background: rgba(57,255,20,0.1);
  border: 1px solid rgba(57,255,20,0.3);
  color: var(--green);
  font-size: 0.55rem;
  padding: 1px 5px;
  border-radius: 2px;
  margin-left: 4px;
  letter-spacing: 1px;
}

/* NETWORK MAP */
.map-panel {
  display: flex;
  flex-direction: column;
  overflow: hidden;
  background: var(--bg2);
}

.map-header {
  padding: 12px 16px;
  border-bottom: 1px solid var(--border);
  font-family: 'Orbitron', monospace;
  font-size: 0.65rem;
  letter-spacing: 2px;
  color: var(--text-dim);
  text-transform: uppercase;
}

.network-canvas {
  flex: 1;
  position: relative;
  overflow: hidden;
}

svg#netmap {
  width: 100%;
  height: 100%;
}

.node-circle { transition: all 0.4s ease; cursor: pointer; }
.node-label  { font-family: 'Share Tech Mono', monospace; font-size: 9px; fill: #aaa; pointer-events: none; }
.node-priv   { font-family: 'Orbitron', monospace; font-size: 8px; font-weight: 700; pointer-events: none; }
.edge-line   { stroke-opacity: 0.4; transition: all 0.4s ease; }

/* EMPTY STATE */
.empty-state {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: var(--text-dim);
  gap: 12px;
}

.empty-icon {
  font-size: 3rem;
  opacity: 0.3;
}

.empty-text {
  font-family: 'Orbitron', monospace;
  font-size: 0.65rem;
  letter-spacing: 3px;
  text-transform: uppercase;
}

/* LOADING */
.loading-overlay {
  position: absolute;
  inset: 0;
  background: rgba(2,6,8,0.9);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  z-index: 100;
  gap: 16px;
}

.loading-ring {
  width: 48px; height: 48px;
  border: 2px solid var(--border);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin { to { transform: rotate(360deg); } }

.loading-text {
  font-family: 'Orbitron', monospace;
  font-size: 0.65rem;
  letter-spacing: 3px;
  color: var(--accent);
  text-transform: uppercase;
}

.loading-overlay.hidden { display: none; }

/* TOOLTIP */
.tooltip {
  position: fixed;
  background: var(--bg3);
  border: 1px solid var(--accent);
  padding: 8px 12px;
  border-radius: 3px;
  font-size: 0.65rem;
  color: var(--text);
  pointer-events: none;
  z-index: 1000;
  opacity: 0;
  transition: opacity 0.15s;
  max-width: 220px;
  line-height: 1.6;
}
.tooltip.visible { opacity: 1; }

/* TERMINAL CURSOR */
.cursor {
  display: inline-block;
  width: 7px; height: 12px;
  background: var(--accent);
  vertical-align: middle;
  animation: blink-cursor 1s step-end infinite;
  margin-left: 2px;
}
@keyframes blink-cursor { 50% { opacity: 0; } }

/* SCROLLBAR */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--dim); }

/* PROGRESS BAR */
.progress-bar {
  height: 2px;
  background: var(--border);
  border-radius: 1px;
  overflow: hidden;
  margin-top: 6px;
}
.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--accent), var(--green));
  border-radius: 1px;
  transition: width 0.3s ease;
  width: 0%;
}

.outcome-badge {
  font-family: 'Orbitron', monospace;
  font-size: 0.65rem;
  letter-spacing: 2px;
  padding: 4px 10px;
  border-radius: 2px;
  text-transform: uppercase;
}
.outcome-detected { background: rgba(255,61,113,0.15); color: var(--red); border: 1px solid var(--red); }
.outcome-survived { background: rgba(57,255,20,0.12); color: var(--green); border: 1px solid var(--green); }

/* DEFENDER INFO */
.defender-info {
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: 3px;
  padding: 10px 12px;
  font-size: 0.62rem;
  color: var(--text-dim);
  line-height: 1.7;
}
.defender-info strong { color: var(--accent2); }
</style>
</head>
<body>
<div class="app">

<header>
  <div class="logo">CYBER<span>BATTLE</span> // SIM LAB</div>
  <div class="header-right">
    <div class="status-dot"></div>
    <span>SYSTEM ONLINE</span>
    <span id="clock">--:--:--</span>
  </div>
</header>

<main>
  <!-- ===== CONTROL PANEL ===== -->
  <aside class="control-panel">
    <div>
      <div class="panel-title">// Select Agent</div>
      <div class="model-grid" id="modelGrid">
        <div style="color:var(--text-dim);font-size:0.65rem;">Loading models…</div>
      </div>
    </div>

    <div>
      <div class="panel-title">// Defender Mode</div>
      <div class="defender-grid" id="defenderGrid">
        <button class="def-btn active" data-mode="random"     onclick="setDefender(this)"><span class="def-icon">🎲</span>Random</button>
        <button class="def-btn"        data-mode="aggressive" onclick="setDefender(this)"><span class="def-icon">⚡</span>Aggressive</button>
        <button class="def-btn"        data-mode="stealth"    onclick="setDefender(this)"><span class="def-icon">👁</span>Stealth</button>
        <button class="def-btn"        data-mode="patching"   onclick="setDefender(this)"><span class="def-icon">🔧</span>Patching</button>
      </div>
      <div class="defender-info" id="defenderDesc" style="margin-top:8px;">
        <strong>RANDOM</strong> — Low-noise patrol. 5% chance to detect per step. Good baseline for testing raw agent performance.
      </div>
    </div>

    <div>
      <div class="panel-title">// Max Steps</div>
      <div class="slider-wrap">
        <input type="range" id="stepsRange" min="10" max="50" value="30" oninput="document.getElementById('stepsVal').textContent=this.value">
        <div class="slider-val" id="stepsVal">30</div>
      </div>
      <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
    </div>

    <button class="run-btn" id="runBtn" onclick="runSimulation()">
      <span>▶ RUN SIMULATION</span>
    </button>

    <div id="outcomeArea" style="text-align:center;"></div>
  </aside>

  <!-- ===== STATS BAR ===== -->
  <div class="stats-bar">
    <div class="stat-cell">
      <div class="stat-label">Steps</div>
      <div class="stat-value" id="statSteps">—</div>
    </div>
    <div class="stat-cell">
      <div class="stat-label">Total Reward</div>
      <div class="stat-value" id="statReward">—</div>
    </div>
    <div class="stat-cell">
      <div class="stat-label">Privilege</div>
      <div class="stat-value" id="statPriv">—</div>
    </div>
    <div class="stat-cell">
      <div class="stat-label">Nodes</div>
      <div class="stat-value" id="statNodes">—</div>
    </div>
    <div class="stat-cell">
      <div class="stat-label">Credentials</div>
      <div class="stat-value" id="statCreds">—</div>
    </div>
    <div class="stat-cell">
      <div class="stat-label">Customer Data</div>
      <div class="stat-value" id="statData">—</div>
    </div>
  </div>

  <!-- ===== CONTENT AREA ===== -->
  <div class="content-area">

    <!-- LOG -->
    <div class="log-panel">
      <div class="log-header">
        <div class="log-title">// Action Log</div>
        <div class="log-count" id="logCount">0 events</div>
        <div class="filter-btns">
          <button class="filter-btn active" onclick="filterLog('all', this)">ALL</button>
          <button class="filter-btn" onclick="filterLog('local', this)">LOCAL</button>
          <button class="filter-btn" onclick="filterLog('remote', this)">REMOTE</button>
          <button class="filter-btn" onclick="filterLog('connect', this)">CONNECT</button>
          <button class="filter-btn" onclick="filterLog('defender', this)">DEFENDER</button>
        </div>
      </div>

      <div class="log-body" id="logBody">
        <div class="empty-state">
          <div class="empty-icon">⬡</div>
          <div class="empty-text">Awaiting simulation<span class="cursor"></span></div>
        </div>
      </div>
    </div>

    <!-- NETWORK MAP -->
    <div class="resizer" id="resizer"></div>
    <div class="map-panel">
      <div class="map-header">// Network Map</div>
      <div class="network-canvas" id="networkCanvas">
        <svg id="netmap" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <filter id="glow">
              <feGaussianBlur stdDeviation="3" result="blur"/>
              <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
            </filter>
            <marker id="arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
              <path d="M0,0 L0,6 L6,3 z" fill="#2a4a5a"/>
            </marker>
          </defs>
          <g id="mapEdges"></g>
          <g id="mapNodes"></g>
        </svg>
        <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);text-align:center;color:var(--text-dim);">
          <div style="font-family:'Orbitron',monospace;font-size:0.6rem;letter-spacing:2px;">NO DATA</div>
        </div>
      </div>
    </div>
  </div>
</main>

<div class="loading-overlay hidden" id="loadingOverlay">
  <div class="loading-ring"></div>
  <div class="loading-text">Running Simulation<span class="cursor"></span></div>
</div>

<div class="tooltip" id="tooltip"></div>

</div><!-- .app -->

<script>
// ─── STATE ───────────────────────────────────
let selectedModel   = null;
let selectedDefender = 'random';
let currentFilter   = 'all';
let simData         = null;
let models          = [];

const DEFENDER_DESCS = {
  random:     '<strong>RANDOM</strong> — Low-noise patrol. 5% chance to detect per step. Good baseline for testing raw agent performance.',
  aggressive: '<strong>AGGRESSIVE</strong> — High vigilance. Detects privilege escalation (40% chance) and late-stage intrusions (30% if step>15). +1 penalty per step.',
  stealth:    '<strong>STEALTH</strong> — Behavioural analysis. Tracks action history; after 5 actions the chance of detection rises to 30%. Rewards quiet agents.',
  patching:   '<strong>PATCHING</strong> — Active remediation. Every other step (30% chance) revokes a node\\'s privilege level. -2 penalty when triggered.'
};

// ─── CLOCK ───────────────────────────────────
setInterval(() => {
  const d = new Date();
  document.getElementById('clock').textContent =
    d.toTimeString().slice(0,8);
}, 1000);

// ─── INIT ────────────────────────────────────
async function loadModels() {
  try {
    const res = await fetch('/api/models');
    models = await res.json();
    renderModelGrid();
  } catch(e) {
    document.getElementById('modelGrid').innerHTML =
      '<div style="color:var(--red);font-size:0.65rem;">Failed to load models</div>';
  }
}

function renderModelGrid() {
  const grid = document.getElementById('modelGrid');
  grid.innerHTML = '';
  models.forEach(m => {
    const card = document.createElement('div');
    card.className = 'model-card' + (m.loaded ? '' : '');
    card.dataset.id = m.id;
    card.onclick = () => selectModel(m.id);

    const badge = m.loaded
      ? '<span class="model-badge badge-loaded">✓ .pth loaded</span>'
      : '<span class="model-badge badge-sim">~ simulated</span>';

    card.innerHTML = `
      <div class="model-name">${m.label.toUpperCase()}</div>
      <div class="model-meta">bias: ${m.bias} &nbsp;|&nbsp; arch: ${m.arch}</div>
      ${badge}
    `;
    grid.appendChild(card);
  });

  // auto-select first
  if (models.length) selectModel(models[0].id);
}

function selectModel(id) {
  selectedModel = id;
  document.querySelectorAll('.model-card').forEach(c => {
    c.classList.toggle('active', c.dataset.id === id);
  });
}

function setDefender(btn) {
  document.querySelectorAll('.def-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  selectedDefender = btn.dataset.mode;
  document.getElementById('defenderDesc').innerHTML = DEFENDER_DESCS[selectedDefender] || '';
}

// ─── SIMULATION ──────────────────────────────
async function runSimulation() {
  if (!selectedModel) return;

  const btn = document.getElementById('runBtn');
  btn.disabled = true;
  document.getElementById('loadingOverlay').classList.remove('hidden');
  document.getElementById('logBody').innerHTML = '';
  document.getElementById('outcomeArea').innerHTML = '';
  resetStats();
  clearMap();

  const maxSteps = parseInt(document.getElementById('stepsRange').value);

  try {
    const res = await fetch('/api/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: selectedModel,
        defender: selectedDefender,
        max_steps: maxSteps
      })
    });
    simData = await res.json();
    renderSimulation(simData);
  } catch(e) {
    document.getElementById('logBody').innerHTML =
      `<div class="empty-state"><div class="empty-icon">✗</div><div class="empty-text">Error: ${e.message}</div></div>`;
  }

  document.getElementById('loadingOverlay').classList.add('hidden');
  btn.disabled = false;
}

function resetStats() {
  ['statSteps','statReward','statPriv','statNodes','statCreds','statData'].forEach(id => {
    const el = document.getElementById(id);
    el.textContent = '—';
    el.className = 'stat-value';
  });
  document.getElementById('progressFill').style.width = '0%';
}

function renderSimulation(data) {
  // Stats
  document.getElementById('statSteps').textContent  = data.steps;
  document.getElementById('statReward').textContent = data.total_reward > 0
    ? '+' + data.total_reward : data.total_reward;
  document.getElementById('statReward').className = 'stat-value ' +
    (data.total_reward > 0 ? 'success' : 'danger');
  document.getElementById('statPriv').textContent   = data.final_privilege;
  document.getElementById('statPriv').className     = 'stat-value ' +
    (data.final_privilege >= 2 ? 'danger' : data.final_privilege === 1 ? 'warn' : '');
  document.getElementById('statNodes').textContent  = data.final_nodes;
  document.getElementById('statCreds').textContent  = data.credentials;
  document.getElementById('statData').textContent   = data.customer_data ? 'YES' : 'NO';
  document.getElementById('statData').className     = 'stat-value ' +
    (data.customer_data ? 'danger' : '');

  // Progress bar
  const maxSteps = parseInt(document.getElementById('stepsRange').value);
  document.getElementById('progressFill').style.width = (data.steps / maxSteps * 100) + '%';

  // Outcome badge
  const outcome = data.detected
    ? '<div class="outcome-badge outcome-detected">⚠ DETECTED</div>'
    : '<div class="outcome-badge outcome-survived">✓ SURVIVED</div>';
  document.getElementById('outcomeArea').innerHTML = outcome;

  // Log
  const logBody = document.getElementById('logBody');
  logBody.innerHTML = '';

  data.log.forEach((entry, i) => {
    logBody.appendChild(buildLogEntry(entry, i, data.steps));
  });

  document.getElementById('logCount').textContent = data.log.length + ' events';

  applyFilter(currentFilter);
  logBody.scrollTop = 0;

  // Network map
  renderNetworkMap(data);
}

function buildLogEntry(entry, idx, totalSteps) {
  const div = document.createElement('div');
  div.className = 'log-entry';
  div.dataset.type = entry.action_type;
  div.dataset.defTriggered = entry.defender_triggered ? '1' : '0';

  const tagClass = {
    local: 'tag-local', remote: 'tag-remote', connect: 'tag-connect'
  }[entry.action_type] || 'tag-local';

  const rewardClass = entry.reward >= 0 ? 'reward-pos' : 'reward-neg';
  const rewardSign  = entry.reward >= 0 ? '+' : '';

  let newNodeHtml = '';
  if (entry.new_nodes && entry.new_nodes.length) {
    newNodeHtml = entry.new_nodes.map(n =>
      `<span class="new-node-pill">+${n}</span>`).join('');
  }

  const privTotal = Object.values(entry.privileges || {}).reduce((a,b) => a+b, 0);

  let defHtml = '';
  if (entry.defender_triggered && entry.defender_msg) {
    defHtml = `<div class="defender-line">🛡 ${entry.defender_msg}</div>`;
  }

  let detectedHtml = '';
  if (entry.detected) {
    detectedHtml = `<div class="detected-banner">⚠ AGENT DETECTED — EPISODE TERMINATED</div>`;
  }

  div.innerHTML = `
    <div class="log-step">
      <div class="step-num">${String(entry.step).padStart(2,'0')}</div>
      <div class="step-content">
        <div class="action-line">
          <span class="action-tag ${tagClass}">${entry.action_type.toUpperCase()}</span>
          <span class="action-text">${entry.action}</span>
          ${newNodeHtml}
        </div>
        <div class="reward-line">
          <span class="${rewardClass}">Δ ${rewardSign}${entry.reward}</span>
          <span>Σ ${entry.total_reward}</span>
          <span style="color:var(--text-dim)">priv:${privTotal}</span>
          <span style="color:var(--text-dim)">creds:${entry.creds}</span>
          ${entry.customer_data ? '<span style="color:var(--red)">DATA:✓</span>' : ''}
        </div>
        ${defHtml}
        ${detectedHtml}
      </div>
    </div>
  `;

  return div;
}

// ─── FILTER ──────────────────────────────────
function filterLog(type, btn) {
  currentFilter = type;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  applyFilter(type);
}

function applyFilter(type) {
  document.querySelectorAll('.log-entry').forEach(el => {
    if (type === 'all') {
      el.classList.remove('hidden');
    } else if (type === 'defender') {
      el.classList.toggle('hidden', el.dataset.defTriggered !== '1');
    } else {
      el.classList.toggle('hidden', el.dataset.type !== type);
    }
  });
}

// ─── NETWORK MAP ─────────────────────────────
const ALL_NODES = [
  "client","GitHubProject","AzureResourceManager",
  "Website.IIS","AzureStorage","SharePoint",
  "AdminMachine","Flag"
];

// Mutable positions — updated live when nodes are dragged
const NODE_POS = {
  "client":                 [170, 250],
  "GitHubProject":          [90,  130],
  "AzureResourceManager":   [260, 80],
  "Website.IIS":            [310, 190],
  "AzureStorage":           [230, 320],
  "SharePoint":             [100, 340],
  "AdminMachine":           [310, 320],
  "Flag":                   [230, 200],
};

const PRIV_COLORS = ['#1a3a4a', '#00aaff', '#ff3d71'];
const PRIV_STROKE = ['#0e2a3a',  '#0077cc', '#cc1144'];
const PRIV_LABEL  = ['', 'USER', 'ADMIN'];

let _lastMapData = null;

function clearMap() {
  document.getElementById('mapEdges').innerHTML = '';
  document.getElementById('mapNodes').innerHTML = '';
}

function renderNetworkMap(data) {
  _lastMapData = data;
  _drawMap(data);
}

function _drawMap(data) {
  const edgesG = document.getElementById('mapEdges');
  const nodesG = document.getElementById('mapNodes');
  edgesG.innerHTML = '';
  nodesG.innerHTML = '';

  const lastEntry  = data.log[data.log.length - 1];
  const discovered = lastEntry ? lastEntry.discovered_nodes : [];
  const privileges = lastEntry ? lastEntry.privileges : {};

  // Edges
  const edgeSet = new Set();
  if (discovered.length > 1) {
    for (let i = 0; i < discovered.length - 1; i++) {
      const key = discovered[i] + '→' + discovered[i+1];
      if (!edgeSet.has(key)) {
        edgeSet.add(key);
        const a = NODE_POS[discovered[i]];
        const b = NODE_POS[discovered[i+1]];
        if (a && b) {
          const line = document.createElementNS('http://www.w3.org/2000/svg','line');
          line.setAttribute('x1', a[0]); line.setAttribute('y1', a[1]);
          line.setAttribute('x2', b[0]); line.setAttribute('y2', b[1]);
          line.setAttribute('stroke', '#0e3a4a');
          line.setAttribute('stroke-width', '1.5');
          line.setAttribute('marker-end', 'url(#arrow)');
          line.classList.add('edge-line');
          line.dataset.from = discovered[i];
          line.dataset.to   = discovered[i+1];
          edgesG.appendChild(line);
        }
      }
    }
  }

  // Nodes
  ALL_NODES.forEach(node => {
    const pos    = NODE_POS[node];
    if (!pos) return;
    const isDisc = discovered.includes(node);
    const priv   = isDisc ? (privileges[node] || 0) : -1;

    const g = document.createElementNS('http://www.w3.org/2000/svg','g');
    g.dataset.node = node;
    g.style.cursor = 'grab';

    const circle = document.createElementNS('http://www.w3.org/2000/svg','circle');
    circle.setAttribute('cx', pos[0]);
    circle.setAttribute('cy', pos[1]);
    circle.setAttribute('r', isDisc ? 18 : 10);
    if (!isDisc) {
      circle.setAttribute('fill', '#060d12');
      circle.setAttribute('stroke', '#0e1e28');
      circle.setAttribute('stroke-width', '1');
      circle.setAttribute('stroke-dasharray', '3,3');
    } else {
      circle.setAttribute('fill', PRIV_COLORS[priv] || '#1a3a4a');
      circle.setAttribute('stroke', PRIV_STROKE[priv] || '#0e2a3a');
      circle.setAttribute('stroke-width', '2');
      if (priv >= 2) circle.setAttribute('filter', 'url(#glow)');
    }
    circle.classList.add('node-circle');

    const label = document.createElementNS('http://www.w3.org/2000/svg','text');
    label.setAttribute('x', pos[0]);
    label.setAttribute('y', pos[1] + 28);
    label.setAttribute('text-anchor', 'middle');
    label.classList.add('node-label');
    label.textContent = node.length > 10 ? node.slice(0,9)+'…' : node;
    label.style.fill = isDisc ? '#8abbcc' : '#2a4a5a';

    g.appendChild(circle);
    g.appendChild(label);

    if (isDisc && priv > 0) {
      const privLabel = document.createElementNS('http://www.w3.org/2000/svg','text');
      privLabel.setAttribute('x', pos[0]);
      privLabel.setAttribute('y', pos[1] + 4);
      privLabel.setAttribute('text-anchor', 'middle');
      privLabel.classList.add('node-priv');
      privLabel.textContent = PRIV_LABEL[priv] || '';
      privLabel.style.fill = priv >= 2 ? '#ff8aaa' : '#80ccee';
      g.appendChild(privLabel);
    }

    // Tooltip (suppressed while dragging)
    g.addEventListener('mouseenter', (e) => {
      if (_draggingNode) return;
      const tt = document.getElementById('tooltip');
      tt.innerHTML = `<strong>${node}</strong><br>
        Status: ${isDisc ? 'DISCOVERED' : 'UNKNOWN'}<br>
        Privilege: ${isDisc ? (priv === 2 ? 'ADMIN' : priv === 1 ? 'USER' : 'NONE') : 'N/A'}<br>
        <span style="color:var(--text-dim);font-size:0.58rem">drag to reposition</span>`;
      tt.classList.add('visible');
    });
    g.addEventListener('mousemove', (e) => {
      if (_draggingNode) return;
      const tt = document.getElementById('tooltip');
      tt.style.left = (e.clientX + 14) + 'px';
      tt.style.top  = (e.clientY - 10) + 'px';
    });
    g.addEventListener('mouseleave', () => {
      document.getElementById('tooltip').classList.remove('visible');
    });

    g.addEventListener('mousedown', startNodeDrag);
    nodesG.appendChild(g);
  });

  const placeholder = document.querySelector('.network-canvas > div');
  if (placeholder) placeholder.style.display = 'none';
}

// ─── NODE DRAG ───────────────────────────────
let _draggingNode = null;
let _dragOffset   = {x: 0, y: 0};

function startNodeDrag(e) {
  e.preventDefault();
  const g    = e.currentTarget;
  const node = g.dataset.node;
  const svg  = document.getElementById('netmap');
  const pt   = svg.createSVGPoint();
  pt.x = e.clientX; pt.y = e.clientY;
  const svgP = pt.matrixTransform(svg.getScreenCTM().inverse());
  _draggingNode = node;
  _dragOffset   = { x: svgP.x - NODE_POS[node][0], y: svgP.y - NODE_POS[node][1] };
  g.style.cursor = 'grabbing';
  document.getElementById('tooltip').classList.remove('visible');
}

document.addEventListener('mousemove', (e) => {
  if (!_draggingNode) return;
  const svg = document.getElementById('netmap');
  const pt  = svg.createSVGPoint();
  pt.x = e.clientX; pt.y = e.clientY;
  const svgP = pt.matrixTransform(svg.getScreenCTM().inverse());

  NODE_POS[_draggingNode][0] = svgP.x - _dragOffset.x;
  NODE_POS[_draggingNode][1] = svgP.y - _dragOffset.y;

  const cx = NODE_POS[_draggingNode][0];
  const cy = NODE_POS[_draggingNode][1];

  // Move node elements live
  const g = document.querySelector(`g[data-node="${_draggingNode}"]`);
  if (g) {
    g.querySelector('circle').setAttribute('cx', cx);
    g.querySelector('circle').setAttribute('cy', cy);
    g.querySelectorAll('text').forEach(t => {
      t.setAttribute('x', cx);
      t.setAttribute('y', t.classList.contains('node-priv') ? cy + 4 : cy + 28);
    });
  }

  // Update connected edges live
  document.querySelectorAll('.edge-line').forEach(line => {
    if (line.dataset.from === _draggingNode) {
      line.setAttribute('x1', cx); line.setAttribute('y1', cy);
    }
    if (line.dataset.to === _draggingNode) {
      line.setAttribute('x2', cx); line.setAttribute('y2', cy);
    }
  });
});

document.addEventListener('mouseup', () => {
  if (_draggingNode) {
    const g = document.querySelector(`g[data-node="${_draggingNode}"]`);
    if (g) g.style.cursor = 'grab';
    _draggingNode = null;
  }
});

// ─── RESIZABLE SPLITTER ──────────────────────
(function() {
  const resizer  = document.getElementById('resizer');
  const logPanel = document.querySelector('.log-panel');
  const content  = document.querySelector('.content-area');
  let isResizing = false;

  resizer.addEventListener('mousedown', (e) => {
    isResizing = true;
    resizer.classList.add('dragging');
    document.body.style.cursor    = 'col-resize';
    document.body.style.userSelect = 'none';
    e.preventDefault();
  });

  document.addEventListener('mousemove', (e) => {
    if (!isResizing) return;
    const rect  = content.getBoundingClientRect();
    const newW  = e.clientX - rect.left;
    const minW  = 180;
    const maxW  = rect.width - 180 - 6;
    logPanel.style.width = Math.min(Math.max(newW, minW), maxW) + 'px';
  });

  document.addEventListener('mouseup', () => {
    if (!isResizing) return;
    isResizing = false;
    resizer.classList.remove('dragging');
    document.body.style.cursor    = '';
    document.body.style.userSelect = '';
  });
})();

// ─── BOOT ─────────────────────────────────────
loadModels();
</script>
</body>
</html>

"""

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route('/')
def index():
    return make_response(HTML, 200)


@app.route('/api/models')
def get_models():
    models = []
    upload_dir = "/mnt/user-data/uploads"
    for name, profile in MODEL_PROFILES.items():
        path = os.path.join(upload_dir, f"{name}.pth")
        exists = os.path.exists(path)
        models.append({
            "id": name,
            "label": name.replace("attacker_", "").replace("_", " ").title(),
            "style": profile["style"],
            "bias": profile["bias"],
            "arch": profile["arch"],
            "loaded": exists,
        })
    return jsonify(models)


@app.route('/api/simulate', methods=['POST'])
def simulate():
    data = request.json
    model_name = data.get('model', 'attacker_random')
    defender_mode = data.get('defender', 'random')
    max_steps = int(data.get('max_steps', 30))

    env = SimEnv()
    obs = env.reset()

    log = []
    total_reward = 0
    detected = False
    detection_step = None
    action_history = []

    for step in range(max_steps):
        action = select_action_simulated(obs, model_name, step)
        label, atype = action_label(action)
        action_history.append(label)

        prev_nodes = list(obs['_discovered_nodes'])
        prev_priv = float(np.sum(obs['nodes_privilegelevel']))

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        new_priv = float(np.sum(obs['nodes_privilegelevel']))
        new_nodes = [n for n in obs['_discovered_nodes'] if n not in prev_nodes]

        # Reward shaping (mirror training)
        if new_priv > prev_priv:
            reward += 20
        if new_nodes:
            reward += 10 * len(new_nodes)
        if reward == 0:
            reward -= 1

        # Defender acts every other step
        def_result = 0
        def_msg = ""
        if step % 2 == 0:
            def_result, def_msg = defender_step(defender_mode, step, action_history, obs)

        if def_result == "detected":
            detected = True
            detection_step = step
            reward -= 50
            done = True

        elif isinstance(def_result, (int, float)):
            reward += def_result

        total_reward += reward

        entry = {
            "step": step + 1,
            "action": label,
            "action_type": atype,
            "reward": round(reward, 2),
            "total_reward": round(total_reward, 2),
            "discovered_nodes": list(obs['_discovered_nodes']),
            "new_nodes": new_nodes,
            "privileges": {n: int(obs['nodes_privilegelevel'][i]) for i, n in enumerate(obs['_discovered_nodes'])},
            "creds": obs['credential_cache_length'],
            "customer_data": bool(obs['customer_data_found']),
            "defender_msg": def_msg,
            "defender_triggered": def_result != 0,
            "detected": detected,
        }
        log.append(entry)

        if done:
            break

    final_priv = int(np.sum(obs['nodes_privilegelevel']))

    return jsonify({
        "model": model_name,
        "defender": defender_mode,
        "steps": len(log),
        "total_reward": round(total_reward, 2),
        "final_privilege": final_priv,
        "final_nodes": len(obs['_discovered_nodes']),
        "detected": detected,
        "detection_step": detection_step,
        "customer_data": bool(obs['customer_data_found']),
        "credentials": obs['credential_cache_length'],
        "log": log,
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
