import random
import numpy as np

def defender_step(env, mode, step, history, obs=None):
    """
    Advanced defender with multiple behavior modes.

    Args:
        env      - the CyberBattle environment
        mode     - one of: "random", "aggressive", "stealth", "patching"
        step     - current step number in the episode
        history  - list of actions taken so far (for stealth detection)
        obs      - current observation dict (needed by aggressive/patching)

    Returns:
        "detected" -> attacker caught, episode ends
        int/float  -> reward modifier (negative = defender acted)
    """

    if mode == "random":
        if random.random() < 0.05:
            return "detected"
        return 0

    elif mode == "aggressive":
        if obs is not None:
            priv = np.sum(obs['nodes_privilegelevel'])
        else:
            priv = 0
        # high detection if attacker has escalated
        if priv > 1 and random.random() < 0.4:
            return "detected"
        # high detection late in episode
        if step > 15 and random.random() < 0.3:
            return "detected"
        return -1

    elif mode == "stealth":
        # detects patterns — more actions = more likely caught
        if len(history) > 5 and random.random() < 0.3:
            return "detected"
        return 0

    elif mode == "patching":
        # actively revokes privileges on random nodes
        if obs is not None and random.random() < 0.3:
            priv_levels = obs['nodes_privilegelevel']
            if len(priv_levels) > 0:
                node = random.randint(0, len(priv_levels) - 1)
                obs['nodes_privilegelevel'][node] = max(
                    0, priv_levels[node] - 1
                )
            return -2
        return 0

    # fallback
    return 0
