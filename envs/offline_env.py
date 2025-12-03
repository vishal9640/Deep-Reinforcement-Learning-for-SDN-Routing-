import gym
import numpy as np
from gym import spaces

class OfflineSDNEnv(gym.Env):
    """A simple offline Gym environment that replays recorded states.

    Expects a .npz file with at least 'states' array of shape (N, S).
    Optional arrays: 'actions' (N, A), 'rewards' (N,)

    step(action) will advance the internal pointer and return the next recorded state.
    If 'rewards' is present in the file it will use those; otherwise it will compute
    a reward using the same heuristic as `sdn_env.SDNEnv._compute_reward` (throughput - delay - max_link_util)

    This environment is useful for offline training or prototyping without Ryu/Mininet.
    """

    def __init__(self, data_path, link_count=None, max_steps=None):
        data = np.load(data_path)
        if 'states' not in data:
            raise ValueError("Offline data must contain 'states' array")
        self.states = data['states'].astype(np.float32)
        self.actions = data['actions'].astype(np.float32) if 'actions' in data else None
        self.rewards = data['rewards'].astype(np.float32) if 'rewards' in data else None

        self.n = self.states.shape[0]
        self.state_dim = self.states.shape[1]
        # infer link_count if not given: last two entries are ndelay, nthrough
        if link_count is None:
            # default assumption: last 2 entries are ndelay and nthrough
            self.link_count = max(0, self.state_dim - 2)
        else:
            self.link_count = link_count

        # action dim from actions if available, else use link_count
        if self.actions is not None:
            self.action_dim = self.actions.shape[1]
        else:
            self.action_dim = self.link_count

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)

        self.ptr = 0
        self.max_steps = max_steps
        self.episode_steps = 0

    def reset(self):
        # start at a random index so samples vary across episodes
        self.ptr = np.random.randint(0, max(1, self.n - 1))
        self.episode_steps = 0
        return self.states[self.ptr].copy()

    def step(self, action):
        # clip action to action space
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # advance pointer
        next_ptr = (self.ptr + 1) % self.n
        obs = self.states[next_ptr].copy()
        if self.rewards is not None:
            reward = float(self.rewards[next_ptr])
        else:
            # compute reward: throughput - delay - max_link_util
            link_utils = obs[:self.link_count]
            ndelay = obs[self.link_count] if self.state_dim > self.link_count else 0.0
            nthrough = obs[self.link_count+1] if self.state_dim > self.link_count+1 else 0.0
            reward = float(nthrough - ndelay - float(np.max(link_utils)))

        done = False
        info = {}

        self.ptr = next_ptr
        self.episode_steps += 1
        if self.max_steps is not None and self.episode_steps >= self.max_steps:
            done = True

        return obs, reward, done, info

    def render(self, mode='human'):
        print('Offline state idx', self.ptr, 'state', self.states[self.ptr])

    def close(self):
        pass
