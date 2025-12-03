# utils/replay_buffer.py
import random
from collections import deque
import numpy as np

class ReplayBuffer:
    def __init__(self, max_size=200000):
        self.max_size = int(max_size)
        self.buffer = deque(maxlen=self.max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def size(self):
        return len(self.buffer)
