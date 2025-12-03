# train_ddpg.py
import os
import argparse
import sys
import numpy as np
import torch
import copy

# Ensure project root is on sys.path so imports work from anywhere
project_root = os.path.dirname(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.ddpg import DDPGAgent
from utils.replay_buffer import ReplayBuffer
from sdn_env import SDNEnv
import json
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ryu', type=str, default='http://127.0.0.1:8080')
    parser.add_argument('--link_count', type=int, default=10)
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--max_steps', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--save_dir', type=str, default='models/')
    parser.add_argument('--noise_start', type=float, default=0.2)
    parser.add_argument('--noise_end', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--pretrained_actor', type=str, default=None,
                        help='Path to pretrained actor state_dict (from imitation pretrain)')
    parser.add_argument('--offline_data', type=str, default=None,
                        help='Path to .npz offline dataset to use instead of live Ryu env')
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Choose environment: offline replay or live SDNEnv
    if args.offline_data is not None:
        try:
            from envs.offline_env import OfflineSDNEnv
            env = OfflineSDNEnv(args.offline_data, link_count=args.link_count, max_steps=args.max_steps)
            print(f"Using OfflineSDNEnv with data {args.offline_data}")
        except Exception as e:
            print("Failed to load OfflineSDNEnv:", e)
            print("Falling back to live SDNEnv")
            env = SDNEnv(ryu_base=args.ryu, link_count=args.link_count)
    else:
        env = SDNEnv(ryu_base=args.ryu, link_count=args.link_count)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = DDPGAgent(state_dim, action_dim, max_action=float(env.action_space.high[0]))
    # If a pretrained actor is provided (imitation learning), load its weights
    if args.pretrained_actor is not None:
        try:
            actor_state = torch.load(args.pretrained_actor, map_location=torch.device('cpu'))
            # imitation_pretrain saves the actor.state_dict(), so load into agent.actor
            agent.actor.load_state_dict(actor_state)
            agent.actor_target = copy.deepcopy(agent.actor)
            print(f"Loaded pretrained actor from {args.pretrained_actor}")
        except Exception as e:
            print(f"Warning: failed to load pretrained actor {args.pretrained_actor}:", e)
    replay = ReplayBuffer(max_size=200000)

    best_reward = -1e9
    noise = args.noise_start
    noise_decay = (args.noise_start - args.noise_end) / max(1, args.episodes*0.6)

    log_file = os.path.join(args.save_dir, 'training_log.jsonl')
    for ep in range(1, args.episodes + 1):
        s = env.reset()
        ep_reward = 0.0
        for step in range(args.max_steps):
            a = agent.select_action(s, noise=noise)
            s2, r, done, _ = env.step(a)
            replay.add(s, a, r, s2, done)
            info = agent.train(replay, batch_size=args.batch_size)
            s = s2
            ep_reward += r
        # decay noise
        noise = max(args.noise_end, noise - noise_decay)

        # logging
        record = {
            'episode': ep,
            'reward': ep_reward,
            'noise': float(noise),
            'replay_size': replay.size(),
            'time': time.time()
        }
        print(f"Ep {ep} reward {ep_reward:.3f} noise {noise:.3f} replay {replay.size()}")
        with open(log_file, 'a') as fh:
            fh.write(json.dumps(record) + '\n')

        # Save checkpoint if improved
        if ep_reward > best_reward:
            best_reward = ep_reward
            ckpt_path = os.path.join(args.save_dir, f'ddpg_best.pth')
            agent.save(ckpt_path)
            print("Saved best model:", ckpt_path)

        # periodic save
        if ep % 50 == 0:
            agent.save(os.path.join(args.save_dir, f'ddpg_ep{ep}.pth'))

    print("Training completed.")

if __name__ == '__main__':
    main()
