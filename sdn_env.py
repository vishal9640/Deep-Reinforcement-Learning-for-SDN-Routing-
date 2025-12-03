# sdn_env.py
import gym
import time
import requests
import numpy as np
from gym import spaces

class SDNEnv(gym.Env):
    """
    Gym-style environment that communicates with Ryu via REST.
    Observation: vector of normalized link utilizations + optional metrics
    Action: continuous vector of link weight deltas (range [-action_scale, action_scale])
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, ryu_base='http://127.0.0.1:8080', link_count=10,
                 action_scale=0.5, step_time=0.1, auto_reset=True):
        super(SDNEnv, self).__init__()
        self.ryu_base = ryu_base.rstrip('/')
        self.link_count = link_count
        self.action_scale = float(action_scale)
        self.step_time = float(step_time)
        self.auto_reset = auto_reset

        # Observation: link utilizations (link_count) + avg_delay + throughput_norm
        obs_dim = link_count + 2
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-self.action_scale,
                                       high=self.action_scale,
                                       shape=(self.link_count,), dtype=np.float32)

    def reset(self):
        # Optionally restore baseline routing
        try:
            requests.post(self.ryu_base + '/restore_baseline', timeout=2.0)
        except Exception:
            pass
        time.sleep(0.5)
        return self._get_state()

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # Apply action via Ryu REST API
        payload = {"deltas": action.tolist()}
        try:
            requests.post(self.ryu_base + '/set_link_weights', json=payload, timeout=3.0)
        except Exception as e:
            # handle but continue
            print("Warning: set_link_weights failed:", e)

        # Wait for network to react
        time.sleep(self.step_time)

        obs = self._get_state()
        reward = self._compute_reward(obs)
        done = False
        info = {}
        return obs, float(reward), done, info

    def _get_state(self):
        """
        Query Ryu for stats; we expect JSON with keys:
        { "link_utils": [0..1], "avg_delay": <ms>, "throughput": <Mbps> }
        """
        try:
            r = requests.get(self.ryu_base + '/stats', timeout=3.0)
            data = r.json()
        except Exception as e:
            # Fallback: zeros
            data = {"link_utils": [0.0]*self.link_count, "avg_delay": 0.0, "throughput": 0.0}
        link_utils = np.array(data.get("link_utils", [0.0]*self.link_count), dtype=np.float32)
        # ensure proper length
        if len(link_utils) < self.link_count:
            pad = np.zeros(self.link_count - len(link_utils), dtype=np.float32)
            link_utils = np.concatenate([link_utils, pad])
        avg_delay = float(data.get("avg_delay", 0.0))
        throughput = float(data.get("throughput", 0.0))

        # normalize avg_delay and throughput to [0,1] with reasonable caps
        # assume max_delay cap = 1000 ms, max_throughput cap = 1000 Mbps (adjust to scenario)
        ndelay = min(avg_delay / 1000.0, 1.0)
        nthrough = min(throughput / 1000.0, 1.0)
        obs = np.concatenate([link_utils[:self.link_count], [ndelay, nthrough]])
        obs = obs.astype(np.float32)
        return obs

    def _compute_reward(self, obs):
        # obs: [link_utils..., ndelay, nthrough]
        link_utils = obs[:self.link_count]
        ndelay = obs[self.link_count]
        nthrough = obs[self.link_count+1]

        # Reward: maximize throughput, minimize delay and max link util
        alpha = 1.0  # throughput weight
        beta = 1.0   # delay penalty
        gamma = 1.0  # max-link-util penalty

        reward = alpha * nthrough - beta * ndelay - gamma * float(np.max(link_utils))
        # small action penalty could be added externally
        return reward

    def render(self, mode='human'):
        s = self._get_state()
        print("State (link_utils..., ndelay, nthrough):", s)

    def close(self):
        pass
