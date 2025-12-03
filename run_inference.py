# run_inference.py
import argparse
import sys
import os
import numpy as np
import torch

# Ensure project root is on sys.path so imports work from anywhere
project_root = os.path.dirname(__file__)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.ddpg import DDPGAgent
from sdn_env import SDNEnv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/ddpg_best.pth')
    parser.add_argument('--ryu', type=str, default='http://127.0.0.1:8080')
    parser.add_argument('--link_count', type=int, default=10)
    parser.add_argument('--offline_data', type=str, default=None,
                        help='Path to .npz offline dataset for inference (no Ryu)')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--max_steps', type=int, default=50)
    args = parser.parse_args()

    # If offline_data provided, use OfflineSDNEnv
    if args.offline_data is not None:
        try:
            from envs.offline_env import OfflineSDNEnv
            env = OfflineSDNEnv(args.offline_data, link_count=args.link_count, max_steps=args.max_steps)
            print(f"Using OfflineSDNEnv for inference: {args.offline_data}")
        except Exception as e:
            print('Failed to load OfflineSDNEnv:', e)
            print('Falling back to live SDNEnv')
            env = SDNEnv(ryu_base=args.ryu, link_count=args.link_count)
    else:
        env = SDNEnv(ryu_base=args.ryu, link_count=args.link_count)
    sdim = env.observation_space.shape[0]
    adim = env.action_space.shape[0]
    agent = DDPGAgent(sdim, adim, max_action=float(env.action_space.high[0]))
    agent.load(args.model)
    for ep in range(args.episodes):
        s = env.reset()
        ep_reward = 0.0
        for t in range(args.max_steps):
            a = agent.select_action(s, noise=0.0)
            s2, r, done, _ = env.step(a)
            ep_reward += r
            s = s2
        print(f"Inference Episode {ep+1} reward {ep_reward:.3f}")

if __name__ == '__main__':
    main()
