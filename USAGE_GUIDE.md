# Complete Usage Guide: DRL-based SDN Routing

## Quick Start (5-minute overview)

This project implements Deep Deterministic Policy Gradient (DDPG) for SDN routing optimization. The workflow is:

```
Setup Python → Start Ryu + Mininet → Generate Traffic → Collect Expert Data → Pretrain Actor → Train DDPG → Run Inference
```

---

## Prerequisites

### Required Software (Linux/VM)
- **Mininet** (SDN network emulator)
- **Ryu** (OpenFlow controller with REST API)
- **iperf/iperf3** (traffic generator)
- **Python 3.8+** with torch, gym, numpy, requests

### Ryu Setup
Your Ryu app must expose these REST endpoints:
- `GET /stats` → returns JSON with `{"link_utils": [...], "avg_delay": <ms>, "throughput": <Mbps>}`
- `POST /set_link_weights` → accepts JSON `{"deltas": [...]}`
- `POST /restore_baseline` → resets to default routing

---

## Step-by-Step Execution Guide

### Step 1: Environment Setup

#### 1a. Create Python Virtual Environment (Windows PowerShell or Linux)

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**On Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Required packages in `requirements.txt`:**
- torch
- gym
- numpy
- requests
- pandas (optional)
- matplotlib (optional)

---

### Step 2: Start Ryu Controller & Mininet

**These commands run on your Linux machine or VM where Mininet is installed.**

#### 2a. Start Ryu (Terminal 1)
```bash
# Start your custom Ryu app with REST endpoints
ryu-manager your_ryu_app.py
```

Expected output:
```
loading app your_ryu_app
instantiating app your_ryu_app of YourAppClass
...
```

Verify Ryu is running:
```bash
curl http://127.0.0.1:8080/stats
# Should return JSON with link/flow stats
```

#### 2b. Start Mininet Topology (Terminal 2)

**Option A: Using provided simple topology**
```bash
sudo python topologies/simple_topo.py --controller-ip 127.0.0.1 --controller-port 6633
```

**Option B: Using standard Mininet CLI**
```bash
sudo mn --custom topologies/simple_topo.py --topo simple --controller=remote,ip=127.0.0.1,port=6633
```

You should see:
```
Mininet started. Controller set to 127.0.0.1:6633
mininet>
```

**Inside Mininet CLI, verify connectivity:**
```
mininet> pingall
h1 -> h2 h3 h4
h2 -> h1 h3 h4
h3 -> h1 h2 h4
h4 -> h1 h2 h3
```

---

### Step 3: Generate Traffic Schedule (Optional but Recommended)

**Run from your project root (Windows PowerShell or Linux shell where venv is active):**

```powershell
python tools\traffic_gen.py --pairs h1:h3,h2:h4 --duration 20 --bw 10M --out traffic_schedule.json --script traffic_commands.sh
```

**Parameters:**
- `--pairs`: comma-separated source:destination pairs (e.g., `h1:h3,h2:h4`)
- `--duration`: how long each iperf flow runs (seconds)
- `--bw`: bandwidth limit for iperf (e.g., `10M`, `100M`)
- `--out`: save JSON schedule here
- `--script`: save executable commands here

**Output:**
- `traffic_schedule.json`: JSON list of traffic flows with timing
- `traffic_commands.sh`: shell script with `iperf -s` and `iperf -c` commands

**To run the traffic:**
- Inside Mininet CLI, copy/paste commands from `traffic_commands.sh`, or
- On Linux (inside Mininet environment), execute:
  ```bash
  bash traffic_commands.sh
  ```

---

### Step 4: Generate Expert Dataset

**This creates state-action pairs for imitation learning pretraining.**

**Basic command (no traffic):**
```powershell
python scripts\generate_expert_dataset.py --episodes 100 --steps 50 --out expert_dataset.npz
```

**With automatic traffic (Linux only):**
```bash
python scripts/generate_expert_dataset.py \
  --episodes 100 --steps 50 --out expert_dataset.npz \
  --traffic_pairs h1:h3,h2:h4 \
  --traffic_duration 20 \
  --traffic_bw 10M \
  --start_traffic
```

**Parameters:**
- `--episodes`: number of episodes to collect
- `--steps`: steps per episode
- `--out`: output .npz file path
- `--ryu`: Ryu base URL (default: `http://127.0.0.1:8080`)
- `--link_count`: number of links in topology (default: 10)
- `--policy`: expert policy - `zeros` (baseline) or `random`
- `--traffic_pairs`: optional traffic schedule
- `--start_traffic`: automatically start traffic while collecting
- `--seed`: random seed for reproducibility

**Output:**
- `expert_dataset.npz`: Contains `states` (N×S) and `actions` (N×A) arrays

**Note:** If Ryu/Mininet not reachable, the script uses fallback zeros; you can still create dummy datasets for testing.

---

### Step 5: Pretrain Actor (Imitation Learning)

**Uses `expert_dataset.npz` to pretrain the actor network via supervised learning.**

```powershell
python pretrain\imitation_pretrain.py --expert_data expert_dataset.npz --epochs 50 --save_path models/actor_pretrained.pth
```

**Parameters:**
- `--expert_data`: path to `expert_dataset.npz`
- `--epochs`: number of training epochs
- `--batch_size`: batch size for training (default: 64)
- `--lr`: learning rate (default: 1e-4)
- `--save_path`: where to save actor weights

**Output:**
- `models/actor_pretrained.pth`: Actor network state_dict (weights)

**Expected output:**
```
Epoch 1/50, Loss: 0.123456
Epoch 2/50, Loss: 0.095234
...
Saved pretrained actor to models/actor_pretrained.pth
```

---

### Step 6: Train DDPG Agent

**Main RL training loop. Uses Mininet+Ryu environment to collect experience and train the actor-critic network.**

**Option A: Train from scratch (no pretraining)**
```powershell
python train\train_ddpg.py --ryu http://127.0.0.1:8080 --link_count 10 --episodes 500 --max_steps 50
```

**Option B: Train with pretrained actor (recommended)**
```powershell
python train\train_ddpg.py --ryu http://127.0.0.1:8080 --link_count 10 --episodes 500 --max_steps 50 --pretrained_actor models/actor_pretrained.pth
```

**Key parameters:**
- `--ryu`: Ryu controller URL
- `--link_count`: number of links to control
- `--episodes`: number of training episodes
- `--max_steps`: steps per episode
- `--batch_size`: replay buffer batch size (default: 64)
- `--save_dir`: directory to save checkpoints (default: `models/`)
- `--pretrained_actor`: path to pretrained actor weights (optional)
- `--noise_start`: initial exploration noise (default: 0.2)
- `--noise_end`: final exploration noise (default: 0.05)
- `--seed`: random seed

**Output:**
- `models/training_log.jsonl`: JSON log of each episode (reward, noise, replay size, time)
- `models/ddpg_best.pth`: Best model checkpoint (actor + critic weights)
- `models/ddpg_ep50.pth`, `ddpg_ep100.pth`, etc.: Periodic checkpoints

**Expected output:**
```
Ep 1 reward 12.456 noise 0.200 replay 50
Ep 2 reward 15.234 noise 0.198 replay 100
...
Saved best model: models/ddpg_best.pth
```

**Training tips:**
- If rewards are flat, increase `--episodes` or reduce `--noise_end`
- If training is slow, reduce `--max_steps` or `--link_count` for quick iterations
- Watch `training_log.jsonl` to detect if training is improving

---

### Step 7: Run Inference/Evaluation

**Load a trained model and run episodes without exploration noise (deterministic).**

```powershell
python run_inference.py --model models/ddpg_best.pth --ryu http://127.0.0.1:8080 --link_count 10 --episodes 10 --max_steps 50
```

**Parameters:**
- `--model`: path to saved agent checkpoint
- `--ryu`: Ryu controller URL
- `--link_count`: number of links
- `--episodes`: number of evaluation episodes
- `--max_steps`: steps per episode
- `--seed`: random seed

**Output:**
```
Inference Episode 1 reward 45.123
Inference Episode 2 reward 48.567
...
Average Episode Reward: 46.845
```

---

## Complete Workflow Example

**Here's a full end-to-end run:**

### Terminal 1 (Start Ryu - on Linux/VM)
```bash
ryu-manager my_ryu_app.py
```

### Terminal 2 (Start Mininet - on Linux/VM)
```bash
sudo python topologies/simple_topo.py
```

### Terminal 3 (Run Python scripts - your machine or VM with venv active)

**1. Generate traffic schedule:**
```powershell
python tools\traffic_gen.py --pairs h1:h3,h2:h4 --duration 20 --bw 10M --script traffic_commands.sh
```

**2. Collect expert data (100 episodes):**
```powershell
python scripts\generate_expert_dataset.py --episodes 100 --steps 50 --out expert_dataset.npz --traffic_pairs h1:h3,h2:h4 --traffic_duration 20
```

**3. Pretrain actor (imitation learning):**
```powershell
python pretrain\imitation_pretrain.py --expert_data expert_dataset.npz --epochs 50 --save_path models/actor_pretrained.pth
```

**4. Train DDPG (500 episodes):**
```powershell
python train\train_ddpg.py --ryu http://127.0.0.1:8080 --link_count 10 --episodes 500 --pretrained_actor models/actor_pretrained.pth
```

**5. Run inference (evaluate best model):**
```powershell
python run_inference.py --model models/ddpg_best.pth --ryu http://127.0.0.1:8080 --link_count 10 --episodes 10
```

---

## File Reference & Purposes

| File | Purpose | How to Use |
|------|---------|-----------|
| `topologies/simple_topo.py` | Mininet network topology (4 hosts, 2 switches) | Start with: `sudo python topologies/simple_topo.py` |
| `tools/traffic_gen.py` | Generate iperf traffic schedule and commands | `python tools/traffic_gen.py --pairs h1:h3,h2:h4 --duration 20 --script traffic.sh` |
| `scripts/generate_expert_dataset.py` | Collect state-action pairs for imitation learning | `python scripts/generate_expert_dataset.py --episodes 100 --out expert_dataset.npz` |
| `pretrain/imitation_pretrain.py` | Pretrain actor using expert dataset | `python pretrain/imitation_pretrain.py --expert_data expert_dataset.npz` |
| `train/train_ddpg.py` | Main DDPG training loop | `python train/train_ddpg.py --episodes 500 --ryu http://127.0.0.1:8080` |
| `run_inference.py` | Load model and run evaluation | `python run_inference.py --model models/ddpg_best.pth --episodes 10` |
| `models/ddpg.py` | DDPG agent class (Actor, Critic, training logic) | Imported by train/run_inference scripts |
| `sdn_env.py` | Gym environment that interfaces with Ryu | Used internally by train/generate_expert_dataset |
| `ryu_client.py` | HTTP client wrapper for Ryu REST endpoints | Optional helper; used by sdn_env |
| `utils/replay_buffer.py` | Experience replay buffer for DDPG | Used internally by DDPGAgent |

---

## Expected Outputs & Artifacts

After running the complete pipeline, you should have:

```
project/
├── expert_dataset.npz              # State-action pairs from expert policy
├── traffic_schedule.json           # Traffic schedule (JSON)
├── traffic_commands.sh             # Traffic commands script
├── models/
│   ├── actor_pretrained.pth        # Pretrained actor (imitation)
│   ├── ddpg_best.pth               # Best trained model (actor + critic)
│   ├── ddpg_ep50.pth               # Checkpoint at episode 50
│   ├── ddpg_ep100.pth              # Checkpoint at episode 100
│   └── training_log.jsonl          # Training logs (reward per episode)
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'sdn_env'` | Ensure you're running from project root and venv is activated |
| `ConnectionError: Failed to connect to http://127.0.0.1:8080` | Check Ryu is running and Mininet is started; verify `/stats` endpoint |
| `No such file or directory: expert_dataset.npz` | Run `generate_expert_dataset.py` first to create it |
| Training reward is flat or negative | Reduce `--noise_end`, increase `--episodes`, or check if Ryu is returning valid stats |
| `Mininet not found` | Install Mininet on Linux/VM; cannot run directly on Windows |
| Slow training | Reduce `--max_steps`, use smaller `--link_count`, or run on same machine as Mininet |

---

## Next Steps (Advanced)

1. **Analyze training** → Use `tools/plot_training.py` (to be added) to visualize `training_log.jsonl`
2. **Improve expert policy** → Create `scripts/expert_policy.py` to record actual Ryu link weights as actions
3. **Transfer learning** → Train on small topology, then fine-tune on larger one
4. **Offline training** → Create `envs/offline_env.py` to replay saved stats for faster iteration
5. **Add robustness** → Implement retries in `ryu_client.py` for network failures

---

## Quick Reference: Command Cheatsheet

```powershell
# Setup
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Generate traffic schedule
python tools\traffic_gen.py --pairs h1:h3,h2:h4 --duration 20 --bw 10M --script traffic.sh

# Collect expert data
python scripts\generate_expert_dataset.py --episodes 100 --steps 50 --out expert_dataset.npz

# Pretrain actor
python pretrain\imitation_pretrain.py --expert_data expert_dataset.npz --save_path models/actor_pretrained.pth

# Train DDPG
python train\train_ddpg.py --episodes 500 --pretrained_actor models/actor_pretrained.pth

# Run inference
python run_inference.py --model models/ddpg_best.pth --episodes 10
```

---

**That's it! You now have a complete pipeline to train and evaluate a DDPG-based SDN routing agent.**
