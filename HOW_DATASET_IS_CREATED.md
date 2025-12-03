# How generate_expert_dataset.py Creates Datasets

## Overview

`generate_expert_dataset.py` is a dataset collection script that records **expert demonstrations** from a network environment. It follows an expert policy (either "zeros" or "random") through multiple episodes and collects state-action pairs into a NumPy compressed archive (`.npz`).

---

## High-Level Process

```
┌──────────────────────────────────────────────────┐
│ generate_expert_dataset.py START                 │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │ Parse command-line arguments│
         │ (episodes, steps, policy...)│
         └────────────┬────────────────┘
                      │
                      ▼
         ┌──────────────────────────────┐
         │ Create SDNEnv environment    │
         │ Connect to Ryu (8080)        │
         └────────────┬─────────────────┘
                      │
                      ▼
      ┌──────────────────────────────────┐
      │ [Optional] Generate traffic      │
      │ schedule and optionally run it   │
      └────────────┬─────────────────────┘
                   │
                   ▼
      ┌────────────────────────────────┐
      │ FOR each episode:               │
      │  1. Reset environment           │
      │  2. FOR each step:              │
      │     - Get current state         │
      │     - Apply expert policy       │
      │     - Generate action           │
      │     - Step environment          │
      │     - Record (state, action)    │
      └────────────┬────────────────────┘
                   │
                   ▼
      ┌────────────────────────────────┐
      │ Collect all states & actions   │
      │ Convert to NumPy arrays        │
      │ Save to .npz file              │
      └────────────┬────────────────────┘
                   │
                   ▼
   ┌─────────────────────────────────┐
   │ Output: expert_dataset.npz      │
   │ Contains: states, actions arrays│
   └─────────────────────────────────┘
```

---

## Detailed Step-by-Step Breakdown

### **Step 1: Parse Arguments**

```python
parser.add_argument('--out', type=str, default='expert_dataset.npz', ...)
parser.add_argument('--ryu', type=str, default='http://127.0.0.1:8080', ...)
parser.add_argument('--link_count', type=int, default=10, ...)
parser.add_argument('--episodes', type=int, default=50, ...)
parser.add_argument('--steps', type=int, default=50, ...)
parser.add_argument('--policy', type=str, choices=['zeros', 'random'], ...)
```

**Example usage:**
```bash
python scripts/generate_expert_dataset.py \
  --out expert_dataset.npz \
  --ryu http://127.0.0.1:8080 \
  --episodes 100 \
  --steps 50 \
  --policy zeros \
  --seed 42
```

**What each argument does:**
| Argument | Default | Meaning |
|----------|---------|---------|
| `--out` | `expert_dataset.npz` | Output file path |
| `--ryu` | `http://127.0.0.1:8080` | Ryu controller URL |
| `--link_count` | 10 | Number of links in topology |
| `--episodes` | 50 | Total episodes to collect |
| `--steps` | 50 | Steps per episode |
| `--policy` | `zeros` | Expert policy: `zeros` (no-change) or `random` (random actions) |
| `--seed` | 0 | Random seed for reproducibility |
| `--traffic_pairs` | None | Optional: traffic to generate (e.g., `h1:h2,h2:h3`) |
| `--start_traffic` | False | If set, run traffic while collecting data |

---

### **Step 2: Set Random Seed**

```python
np.random.seed(args.seed)
```

**Purpose:** Ensures reproducibility. Same seed = same random actions (if using `--policy random`)

---

### **Step 3: Initialize Environment**

```python
env = SDNEnv(ryu_base=args.ryu, link_count=args.link_count)
sdim = env.observation_space.shape[0]  # State dimension (usually 12)
adim = env.action_space.shape[0]       # Action dimension (usually 10)
```

**What `SDNEnv` does:**
- Connects to Ryu REST API at `http://127.0.0.1:8080`
- Queries network state: link utilizations, delay, throughput
- Returns normalized observations (0-1 range)
- Provides action space: link weight adjustments [-1, 1]

**Extracted dimensions:**
```python
sdim = 12  # State dimension
adim = 10  # Action dimension
```

---

### **Step 4: [Optional] Generate Traffic Schedule**

```python
if args.traffic_pairs is not None and build_schedule is not None:
    pairs = [p.strip() for p in args.traffic_pairs.split(',') if p.strip()]
    traffic_schedule = build_schedule(pairs, args.traffic_duration, 
                                      args.traffic_bw, args.traffic_start_offset)
    
    # Write script to file
    make_scripts(traffic_schedule, args.traffic_script_out)
    print(f'Wrote traffic script to {args.traffic_script_out}')
```

**Example:**
```bash
python scripts/generate_expert_dataset.py \
  --traffic_pairs "h1:h2,h2:h3" \
  --traffic_duration 10 \
  --traffic_bw 10M \
  --traffic_start_offset 0.5 \
  --start_traffic  # Actually run the traffic
```

**What happens:**
1. Parse traffic pairs: `["h1:h2", "h2:h3"]`
2. Call `build_schedule()` → generates iperf command schedule
3. Call `make_scripts()` → writes bash script to `traffic_commands.sh`
4. Optionally runs traffic in background via `execute_schedule()`

---

### **Step 5: [Optional] Start Traffic in Background**

```python
traffic_thread = None
if args.start_traffic and traffic_schedule is not None and execute_schedule is not None:
    def run_traffic():
        try:
            execute_schedule(traffic_schedule)
        except Exception as e:
            print('Traffic execution error:', e)

    traffic_thread = threading.Thread(target=run_traffic, daemon=True)
    traffic_thread.start()
    time.sleep(0.5)  # Give servers time to start
```

**Why this matters:**
- Traffic in background → network state becomes "congested"
- Script collects state-action pairs **while traffic is running**
- Results in realistic network conditions (not just idle states)
- Without traffic: all states are near-zero (no congestion)

---

### **Step 6: Main Collection Loop - The Core**

```python
for ep in range(args.episodes):           # FOR each episode
    s = env.reset()                        # Reset to initial state
    
    for t in range(args.steps):            # FOR each step
        # Step 6a: Get current state
        # s = observation from environment (shape: 12,)
        
        # Step 6b: Apply expert policy to generate action
        if args.policy == 'zeros':
            a = np.zeros(adim, dtype=np.float32)  # Action: [0, 0, 0, ...]
        elif args.policy == 'random':
            a = env.action_space.sample().astype(np.float32)  # Random [-1, 1]
        
        # Step 6c: Execute action in environment
        s2, r, done, _ = env.step(a)       # s2: next state, r: reward
        
        # Step 6d: Record state-action pair
        states.append(s.astype(np.float32))
        actions.append(a.astype(np.float32))
        
        total_steps += 1
        s = s2  # Move to next state
```

**Example walkthrough for 1 episode with 3 steps:**

```
Episode 1:
  Reset env → s = [0.1, 0.2, 0.0, ..., 0.05, 0.8]  (initial state)
  
  Step 1:
    Policy='zeros' → a = [0, 0, 0, ..., 0]
    Env.step(a) → s2, r, done
    Record: states[0] = [0.1, 0.2, ...], actions[0] = [0, 0, ...]
  
  Step 2:
    s = s2 (previous next_state)
    Policy='zeros' → a = [0, 0, 0, ..., 0]
    Env.step(a) → s2, r, done
    Record: states[1] = s2_prev, actions[1] = [0, 0, ...]
  
  Step 3:
    s = s2
    Policy='zeros' → a = [0, 0, 0, ..., 0]
    Env.step(a) → s2, r, done
    Record: states[2] = s2_prev, actions[2] = [0, 0, ...]

Total records after Episode 1: 3 state-action pairs
```

**After 50 episodes × 50 steps:**
```
Total records = 50 × 50 = 2,500 state-action pairs
```

---

### **Step 7: Convert to NumPy Arrays**

```python
states = np.array(states)      # (2500, 12) - all states stacked
actions = np.array(actions)    # (2500, 10) - all actions stacked
```

**Data structure:**
```
states[0]  = [0.1, 0.2, 0.0, 0.1, 0.0, 0.0, 0.1, 0.2, 0.0, 0.1, 0.05, 0.8]
states[1]  = [0.12, 0.18, 0.0, 0.11, 0.0, ...]
...
states[2499] = [...]

actions[0]  = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  (policy='zeros')
actions[1]  = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
...
```

---

### **Step 8: Save to .npz File**

```python
out_dir = os.path.dirname(args.out) or '.'
os.makedirs(out_dir, exist_ok=True)          # Create directories if needed
np.savez(args.out, states=states, actions=actions)
```

**What `np.savez()` does:**
1. Takes named arrays: `states`, `actions`
2. Compresses them (lossless compression)
3. Saves to single `.npz` file

**File structure:**
```
expert_dataset.npz
├── states       (2500, 12) float32
├── actions      (2500, 10) float32
└── [compressed & archived]
```

**Output message:**
```
Saved expert dataset to expert_dataset.npz with 2500 samples (state_dim=12, action_dim=10)
```

---

## Two Expert Policy Modes

### **Mode 1: `--policy zeros` (Baseline)**

```python
a = np.zeros(adim, dtype=np.float32)
# a = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

**Meaning:**
- Expert says: "Don't change any link weights"
- This is the **baseline/idle policy**
- Network operates with default equal-cost paths

**Use case:**
- Learn what baseline network states look like
- Baseline for comparison against learned policies
- Quick testing (no randomness = deterministic)

**Result dataset:**
```
All 2500 actions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

---

### **Mode 2: `--policy random` (Random Expert)**

```python
a = env.action_space.sample().astype(np.float32)
# a = [-0.3, 0.7, -0.1, 0.2, 0.0, -0.8, 0.5, 0.1, -0.4, 0.6]  (random in [-1, 1])
```

**Meaning:**
- Expert says: "Random link weight adjustments"
- Explores different routing strategies
- Each step gets a different random action

**Use case:**
- Explore diverse routing behaviors
- Learn what different weight configurations produce
- More varied states than zeros policy

**Result dataset:**
```
actions[0] = [-0.3, 0.7, -0.1, 0.2, 0.0, -0.8, 0.5, 0.1, -0.4, 0.6]
actions[1] = [0.2, -0.5, 0.9, 0.1, -0.2, 0.3, -0.7, 0.4, 0.0, -0.1]
actions[2] = [0.5, 0.1, -0.3, 0.6, 0.2, -0.1, 0.4, -0.5, 0.8, 0.0]
...
```

---

## Reproducibility Via Seed

```python
np.random.seed(args.seed)  # Set seed at start
```

**Same seed → Same dataset (if using random policy):**
```bash
# Run 1:
python scripts/generate_expert_dataset.py --policy random --seed 42
# Generates: expert_dataset_v1.npz

# Run 2 (with same seed):
python scripts/generate_expert_dataset.py --policy random --seed 42
# Generates: expert_dataset_v2.npz (IDENTICAL to v1)
```

**Different seed → Different dataset:**
```bash
# Run 3 (different seed):
python scripts/generate_expert_dataset.py --policy random --seed 123
# Generates: expert_dataset_v3.npz (DIFFERENT actions)
```

---

## Integration with Traffic Generation

### Without Traffic (Quiet Network)

```bash
python scripts/generate_expert_dataset.py \
  --episodes 50 \
  --steps 50
```

**Result:**
- All states near-zero (no congestion)
- Network is idle
- No meaningful signal for learning

---

### With Traffic (Congested Network)

```bash
python scripts/generate_expert_dataset.py \
  --episodes 50 \
  --steps 50 \
  --traffic_pairs "h1:h2,h2:h3,h3:h4" \
  --traffic_duration 10 \
  --traffic_bw 10M \
  --traffic_start_offset 0.5 \
  --start_traffic
```

**What happens:**
1. Script generates traffic schedule (iperf commands)
2. Starts traffic **in background thread**
3. While traffic runs, collects 50 × 50 = 2,500 state-action pairs
4. States reflect **congested network conditions**

**Timeline:**
```
t=0.0s:   Script starts
t=0.5s:   Traffic thread starts (0.5s delay for servers to start)
t=0.5-10.5s: Traffic running (iperf flows active)
t=0.5-10.5s: Dataset collection happening (recording states while traffic active)
t=10.5s:  Traffic ends
t=10.5-11.0s: Collection completes
t=11.0s:  expert_dataset.npz saved
```

---

## Example: Complete Workflow

### Command:
```bash
python scripts/generate_expert_dataset.py \
  --out my_expert_data.npz \
  --ryu http://127.0.0.1:8080 \
  --link_count 10 \
  --episodes 100 \
  --steps 50 \
  --policy zeros \
  --seed 42
```

### Execution Flow:

```
Step 1: Parse args
  out='my_expert_data.npz'
  ryu='http://127.0.0.1:8080'
  episodes=100
  steps=50
  policy='zeros'
  seed=42

Step 2: Set seed
  np.random.seed(42)

Step 3: Create environment
  env = SDNEnv(ryu_base='http://127.0.0.1:8080', link_count=10)
  sdim = 12
  adim = 10

Step 4-5: No traffic (traffic_pairs=None)

Step 6: Main collection loop
  100 episodes × 50 steps = 5,000 iterations
  
  Episode 1:
    Reset env
    Step 1: state=[...], action=[0,0,0,...], append both
    Step 2: state=[...], action=[0,0,0,...], append both
    ...
    Step 50: state=[...], action=[0,0,0,...], append both
  
  Episode 2:
    (repeat)
  
  ...
  
  Episode 100:
    (repeat)

Step 7: Convert to arrays
  states.shape = (5000, 12)
  actions.shape = (5000, 10)

Step 8: Save
  np.savez('my_expert_data.npz', states=states, actions=actions)

Output:
  Saved expert dataset to my_expert_data.npz with 5000 samples (state_dim=12, action_dim=10)
```

### Generated File:
```
my_expert_data.npz (size: ~206 KB)
├── states: (5000, 12) float32
└── actions: (5000, 10) float32
```

---

## Data Collection Performance

### Collection Speed

**Typical timing (without traffic):**
```
50 episodes × 50 steps × 1.5s per step = ~3,750 seconds ≈ 1 hour
```

(The `1.5s per step` is from `SDNEnv` waiting for Ryu REST API response + network state update)

**With traffic generation (optional):**
- Traffic runs in background (minimal overhead)
- Collection speed unchanged (still ~1 hour for above config)

### File Size

**For 2,500 samples (50 episodes × 50 steps):**
```
States: (2500, 12) × 4 bytes (float32) = 120 KB
Actions: (2500, 10) × 4 bytes (float32) = 100 KB
Compressed (.npz): ~43 KB (55% compression)
```

---

## Error Handling

### Connection Failure
```python
env = SDNEnv(ryu_base=args.ryu, link_count=args.link_count)
# If Ryu unreachable → silently returns zero states
```

**Result:** Dataset fills with zeros (not ideal, but won't crash)

### Traffic Generation Optional
```python
try:
    from tools.traffic_gen import build_schedule, make_scripts, execute_schedule
except Exception:
    build_schedule = None
    make_scripts = None
    execute_schedule = None
```

**If traffic tools missing:** Script continues without traffic (graceful fallback)

---

## Summary Table

| Aspect | Details |
|--------|---------|
| **Input** | Ryu REST API connection + environment parameters |
| **Process** | Loop through episodes → steps → states/actions |
| **Expert Policies** | `zeros` (no change) or `random` (random deltas) |
| **Output** | `.npz` file with `states` and `actions` arrays |
| **Data Shape** | States: (N, 12), Actions: (N, 10) |
| **Collection Speed** | ~1 hour for 50 episodes × 50 steps |
| **File Size** | ~43 KB per 2,500 samples (compressed) |
| **Reproducibility** | Controlled via `--seed` parameter |
| **Optional Features** | Traffic generation, background execution |

---

## Next Steps

### 1. Generate a Quick Test Dataset
```bash
python scripts/generate_expert_dataset.py \
  --out test_dataset.npz \
  --episodes 10 \
  --steps 20 \
  --policy zeros
```

### 2. Generate Large Dataset (Production)
```bash
python scripts/generate_expert_dataset.py \
  --out expert_dataset_prod.npz \
  --episodes 100 \
  --steps 100 \
  --policy random \
  --seed 12345
```

### 3. Generate with Traffic (Realistic)
```bash
# Terminal 1: Start Mininet topology
sudo python topologies/simple_topo.py

# Terminal 2: Start Ryu
cd ryu && ryu-manager openflow.ControllerBase

# Terminal 3: Generate dataset with traffic
python scripts/generate_expert_dataset.py \
  --out realistic_dataset.npz \
  --episodes 50 \
  --steps 50 \
  --policy random \
  --traffic_pairs "h1:h2,h2:h3" \
  --start_traffic
```

### 4. Use Dataset for Training
```bash
# Imitation pretraining
python pretrain/imitation_pretrain.py \
  --data test_dataset.npz \
  --epochs 50

# Offline RL training
python train/train_ddpg.py \
  --offline_data test_dataset.npz \
  --episodes 100
```

