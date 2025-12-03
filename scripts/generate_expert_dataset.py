import argparse
import os
import sys
import numpy as np
import threading
import time

# Ensure project root is on sys.path so imports work from anywhere
project_root = os.path.dirname(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sdn_env import SDNEnv
try:
    from tools.traffic_gen import build_schedule, make_scripts, execute_schedule
except Exception:
    # traffic_gen is optional; if unavailable we'll still generate dataset
    build_schedule = None
    make_scripts = None
    execute_schedule = None


def parse_args():
    parser = argparse.ArgumentParser(description='Generate expert dataset for imitation pretraining')
    parser.add_argument('--out', type=str, default='expert_dataset.npz', help='Output .npz path')
    parser.add_argument('--ryu', type=str, default='http://127.0.0.1:8080', help='Ryu base URL')
    parser.add_argument('--link_count', type=int, default=10, help='Number of links (env link_count)')
    parser.add_argument('--episodes', type=int, default=50, help='Number of episodes to run')
    parser.add_argument('--steps', type=int, default=50, help='Steps per episode')
    parser.add_argument('--policy', type=str, default='zeros', choices=['zeros', 'random'], help='Expert policy to use')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    # Traffic generation options (optional)
    parser.add_argument('--traffic_pairs', type=str, default=None,
                        help='Comma-separated host pairs for traffic (e.g. h1:h2,h2:h3)')
    parser.add_argument('--traffic_duration', type=int, default=10, help='iperf duration seconds')
    parser.add_argument('--traffic_bw', type=str, default='10M', help='iperf bandwidth')
    parser.add_argument('--traffic_start_offset', type=float, default=0.5, help='seconds between flows')
    parser.add_argument('--start_traffic', action='store_true', help='If set, attempt to start traffic using tools/traffic_gen')
    parser.add_argument('--traffic_script_out', type=str, default='traffic_commands.sh', help='Path to write generated traffic script')
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    env = SDNEnv(ryu_base=args.ryu, link_count=args.link_count)
    sdim = env.observation_space.shape[0]
    adim = env.action_space.shape[0]

    states = []
    actions = []

    total_steps = 0

    # If traffic schedule requested, build and optionally start it
    traffic_schedule = None
    if args.traffic_pairs is not None and build_schedule is not None:
        pairs = [p.strip() for p in args.traffic_pairs.split(',') if p.strip()]
        traffic_schedule = build_schedule(pairs, args.traffic_duration, args.traffic_bw, args.traffic_start_offset)
        # write script
        try:
            make_scripts(traffic_schedule, args.traffic_script_out)
            print(f'Wrote traffic script to {args.traffic_script_out}')
        except Exception as e:
            print('Warning: failed to write traffic script:', e)

    traffic_thread = None
    if args.start_traffic and traffic_schedule is not None and execute_schedule is not None:
        # run traffic in a background thread so dataset generation can collect states
        def run_traffic():
            try:
                execute_schedule(traffic_schedule)
            except Exception as e:
                print('Traffic execution error:', e)

        traffic_thread = threading.Thread(target=run_traffic, daemon=True)
        traffic_thread.start()
        # give servers a moment to start
        time.sleep(0.5)
    for ep in range(args.episodes):
        s = env.reset()
        for t in range(args.steps):
            if args.policy == 'zeros':
                a = np.zeros(adim, dtype=np.float32)
            elif args.policy == 'random':
                a = env.action_space.sample().astype(np.float32)
            else:
                a = np.zeros(adim, dtype=np.float32)

            s2, r, done, _ = env.step(a)

            states.append(s.astype(np.float32))
            actions.append(a.astype(np.float32))
            total_steps += 1
            s = s2

    states = np.array(states)
    actions = np.array(actions)

    out_dir = os.path.dirname(args.out) or '.'
    os.makedirs(out_dir, exist_ok=True)
    np.savez(args.out, states=states, actions=actions)
    print(f"Saved expert dataset to {args.out} with {total_steps} samples (state_dim={sdim}, action_dim={adim})")


if __name__ == '__main__':
    main()
