# imitation_pretrain.py
import os
import argparse
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import json

# Ensure project root is on sys.path so imports work from anywhere
project_root = os.path.dirname(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.ddpg import Actor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expert_data', type=str, default='expert_dataset.npz')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_path', type=str, default='models/actor_pretrained.pth')
    return parser.parse_args()

def main():
    args = parse_args()
    data = np.load(args.expert_data)
    states = data['states']  # shape (N, S)
    actions = data['actions']  # shape (N, A)
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    ds = TensorDataset(torch.tensor(states, dtype=torch.float32), torch.tensor(actions, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    state_dim = states.shape[1]
    action_dim = actions.shape[1]
    actor = Actor(state_dim, action_dim).to(device)
    opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    for ep in range(args.epochs):
        total_loss = 0.0
        for s_batch, a_batch in loader:
            s_batch = s_batch.to(device)
            a_batch = a_batch.to(device)
            pred = actor(s_batch)
            loss = loss_fn(pred, a_batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * s_batch.size(0)
        avg_loss = total_loss / len(ds)
        print(f"Epoch {ep+1}/{args.epochs}, Loss: {avg_loss:.6f}")
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(actor.state_dict(), args.save_path)
    print("Saved pretrained actor to", args.save_path)

if __name__ == '__main__':
    main()
