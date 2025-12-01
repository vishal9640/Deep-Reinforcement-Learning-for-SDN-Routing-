# Efficient DRL for SDN Routing - Codebase

## Prerequisites
- Ubuntu 20.04 (recommended)
- Python 3.8+
- Mininet & Ryu installed and running.
- Ryu controller must host endpoints:
  - GET  /stats
  - POST /set_link_weights
  - POST /restore_baseline

## Setup
1. Create virtualenv:
   python3 -m venv venv
   source venv/bin/activate
2. Install:
   pip install -r requirements.txt

## Workflow
1. Prepare Ryu controller (Member 1) and ensure REST endpoints are available.
2. (Optional) Generate expert dataset for imitation learning:
   - Run baseline routing and store state/action pairs into `expert_dataset.npz`.
3. Pretrain actor (imitation):
   python pretrain/imitation_pretrain.py --expert_data expert_dataset.npz --save_path models/actor_pretrained.pth
4. Train DDPG (offline / in Mininet):
   python train/train_ddpg.py --ryu http://127.0.0.1:8080 --link_count 10 --episodes 500
5. Inference/demo:
   python run_inference.py --model models/ddpg_best.pth

## Notes
- Tune `link_count`, `step_time`, and reward scaling to match your Mininet topology and traffic patterns.
- Use small topologies to debug before scaling up.

