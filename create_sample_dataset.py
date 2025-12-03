import numpy as np
import os

# Create a quick sample dataset for testing (without waiting for network)
# Generate synthetic states and actions

n_samples = 100
state_dim = 12  # 10 link utils + 2 metrics (delay, throughput)
action_dim = 10

# Create sample data
states = np.random.uniform(0, 1, (n_samples, state_dim)).astype(np.float32)
actions = np.random.uniform(-1, 1, (n_samples, action_dim)).astype(np.float32)
rewards = np.random.uniform(-10, 10, n_samples).astype(np.float32)

# Save to npz
output_path = 'expert_dataset.npz'
np.savez(output_path, states=states, actions=actions, rewards=rewards)
print(f"✓ Created {output_path}")
print(f"  - States shape: {states.shape}")
print(f"  - Actions shape: {actions.shape}")
print(f"  - Rewards shape: {rewards.shape}")
print(f"  - File size: {os.path.getsize(output_path) / 1024:.1f} KB")
