# RLagentLudo Components Documentation

Comprehensive documentation for the RLagentLudo project, organized by component type: Agents, Environment, and Reward Network.

---

## Table of Contents
1. [Agents](#agents)
2. [Environment](#environment)
3. [Reward Network](#reward-network)

---

## Agents

The agent system provides various RL algorithms and baseline policies for playing Ludo.

### Base Agent Interface

**File**: `src/rl_agent_ludo/agents/baseAgent.py`

**Purpose**: Abstract base class that defines the interface all agents must implement.

**Key Features**:
- Defines standard agent interface with `act()`, `learn()`, and lifecycle methods
- Supports both on-policy (policy gradient) and off-policy (DQN) learning
- Provides hooks for replay buffer and episode management

**Abstract Methods**:
- `is_on_policy` - Returns True for on-policy algorithms (e.g., PPO)
- `needs_replay_learning` - Returns True for off-policy algorithms (e.g., DQN)
- `act(state: State) -> int` - Select action given current state
- `push_to_replay_buffer()` - Store experience for off-policy learning
- `learn_from_replay()` - Learn from replay buffer
- `learn_from_rollout()` - Learn from on-policy rollout

**Lifecycle Hooks**:
- `on_episode_end()` - Called after each episode (epsilon decay, stats)
- `save(filepath)` - Save agent state to disk
- `load(filepath)` - Load agent state from disk

---

### Baseline Agents

**File**: `src/rl_agent_ludo/agents/baseline_agents.py`

**Purpose**: Simple rule-based agents for benchmarking and testing.

#### RandomAgent

**Description**: Selects random valid actions with bias toward moving.

**Features**:
- Works across all curriculum levels (Level 1-5)
- 80% probability to move, 20% to pass
- Supports action masking for multi-token levels

**Use Cases**:
- Baseline opponent during training
- Curriculum level validation
- Testing environment mechanics

**Example**:
```python
agent = RandomAgent(seed=42)
action = agent.act(observation, info)
```

#### GreedyAgent

**Description**: Always moves forward when possible, never passes unnecessarily.

**Features**:
- Deterministic policy (no exploration)
- Prefers moving most advanced token in multi-token scenarios
- Simple but effective baseline

**Use Cases**:
- Stronger baseline than RandomAgent
- Testing game progression logic
- Sanity checking reward functions

#### AlwaysMoveAgent

**Description**: Always tries to move (action 0), useful for testing invalid action handling.

**Features**:
- Ignores action validity
- Used for environment robustness testing

---

### SimpleDQN Agent

**File**: `src/rl_agent_ludo/agents/simple_dqn.py`

**Purpose**: Clean, minimal DQN implementation with experience replay and target network.

**Architecture**:
- **Network**: Feedforward neural network with configurable hidden layers
- **Default**: 2 hidden layers of 128 units each with ReLU activation
- **Input**: State dimension (varies by level: 4D, 8D, 14D, 16D)
- **Output**: Q-values for each action

**Key Features**:
1. **Experience Replay Buffer**
   - Stores (state, action, reward, next_state, done) tuples
   - Default capacity: 50,000 transitions
   - Random sampling breaks correlation

2. **Target Network**
   - Separate network for computing target Q-values
   - Updated every 1,000 steps (configurable)
   - Stabilizes training

3. **Epsilon-Greedy Exploration**
   - Initial epsilon: 1.0 (pure exploration)
   - Minimum epsilon: 0.05
   - Decay rate: 0.995 per episode

4. **Gradient Clipping**
   - Max norm: 10.0
   - Prevents exploding gradients

**Training Algorithm**:
```
1. Sample batch from replay buffer
2. Compute current Q-values: Q(s, a)
3. Compute target Q-values: r + γ * max Q_target(s', a')
4. Minimize MSE loss between current and target
5. Update target network periodically
```

**Hyperparameters**:
- Learning rate: 1e-4
- Discount factor (γ): 0.99
- Batch size: 128
- Target update frequency: 1,000 steps

**Example Usage**:
```python
agent = SimpleDQNAgent(
    state_dim=16,
    action_dim=3,
    learning_rate=1e-4,
    epsilon=1.0,
    device='cuda'
)

# Training loop
for episode in range(num_episodes):
    state, info = env.reset()
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done, _, info = env.step(action)

        agent.store_experience(state, action, reward, next_state, done)
        agent.train_step()

        state = next_state

    agent.decay_epsilon()
```

**Checkpointing**:
- Saves Q-network, target network, optimizer state, epsilon, step count
- Enables resuming training from checkpoints

---

### T-REX Agent

**File**: `src/rl_agent_ludo/agents/trex_agent.py`

**Purpose**: DQN agent trained with learned reward function from human preferences (Trajectory-ranked Reward Extrapolation).

**Key Innovation**: Replaces sparse environment rewards with dense learned rewards from trajectory preferences.

**Architecture**:
- **Policy Network**: Standard DQN (same as SimpleDQNAgent)
- **Reward Network**: Learned from trajectory preferences
  - Input: State (16D)
  - Hidden: 128 units with dropout (0.1)
  - Output: Scalar reward prediction

**Reward Modes**:

1. **Hybrid Rewards (Recommended)**:
   ```
   total_reward = env_reward + λ * scale * learned_reward
   ```
   - Combines sparse env signals (win/loss) with dense learned feedback
   - Default scale: 10.0
   - Default weight (λ): 0.3
   - Best of both worlds: clear objectives + dense guidance

2. **Pure T-REX**:
   ```
   total_reward = learned_reward
   ```
   - Uses only learned rewards (original T-REX approach)
   - Can fail if learned rewards poorly calibrated

**Key Components**:
1. **Reward Network Loading**
   - Loads pre-trained reward network from checkpoint
   - Freezes reward network (eval mode, no training)
   - Network architecture must match training

2. **Learned Reward Computation**
   - Computes learned reward for each state transition
   - Tracks statistics (average learned reward)
   - Provides interpretability into what agent values

3. **Standard DQN Training**
   - Experience replay with learned rewards
   - Target network updates
   - Epsilon-greedy exploration

**Training Pipeline**:
```
Phase 1: Collect Trajectories
  ├─ Run baseline/trained agents
  └─ Save state sequences with outcomes

Phase 2: Learn Reward Function
  ├─ Rank trajectories by quality
  ├─ Create preference pairs
  └─ Train reward network

Phase 3: Train Policy with Learned Rewards
  ├─ Load trained reward network
  ├─ Use learned rewards during RL training
  └─ Learn policy to maximize learned rewards
```

**Example Usage**:
```python
# Train policy with learned rewards
agent = TREXAgent(
    state_dim=16,
    action_dim=3,
    reward_network_path='checkpoints/level6/reward_network_best.pth',
    use_hybrid_rewards=True,
    learned_reward_scale=10.0,
    learned_reward_weight=0.3
)

# Training loop (same as SimpleDQNAgent)
for episode in range(num_episodes):
    state, info = env.reset()
    done = False

    while not done:
        action = agent.act(state)
        next_state, env_reward, done, _, info = env.step(action)

        # Agent internally computes hybrid reward
        agent.store_experience(state, action, env_reward, next_state, done)
        agent.train_step()

        state = next_state

    agent.decay_epsilon()

    # Monitor learned reward statistics
    avg_learned_reward = agent.get_avg_learned_reward()
```

**Advantages**:
- Dense reward signal (every state gets feedback)
- Learns from human/expert preferences
- Can capture complex objectives difficult to hand-engineer

**Disadvantages**:
- Requires trajectory collection phase
- Reward network quality critical
- May inherit biases from preference data

---

## Environment

The environment system implements a curriculum of progressively complex Ludo variants.

### Curriculum Overview

The project uses 5 levels of increasing complexity to build agent capabilities:

| Level | Players | Tokens | Captures | State Dim | Actions | Challenge |
|-------|---------|--------|----------|-----------|---------|-----------|
| 1 | 2 | 1 each | No | 4D | 2 | Basic movement |
| 2 | 2 | 1 each | Yes | 8D | 2 | Opponent interaction |
| 3 | 2 | 2 each | Yes | 14D | 3 | Token coordination |
| 4 | 2 | 2 each | Yes | 16D | 3 | Stochastic dice |
| 5 | 4 | 2 each | Yes | 16D | 3 | Multi-agent |

---

### Level 1: Simple Ludo

**File**: `src/rl_agent_ludo/environment/level1_simple.py`

**Purpose**: Simplest possible Ludo for learning basic movement.

**Game Rules**:
- 2 players, 1 token each
- Linear track: 60 positions (0=home, 1-59=track, 60=goal)
- No six-to-exit rule (can always leave home)
- No capturing mechanics
- No safe zones
- Win condition: First token to reach goal

**State Space (4D)**:
```python
[
    my_pos_norm,        # My token position / 60 (normalized to [0,1])
    opp_pos_norm,       # Opponent position / 60
    can_i_move,         # Binary: Can I move? (1.0 or 0.0)
    can_opp_move        # Binary: Can opponent move?
]
```

**Action Space**:
- 0: Move token
- 1: Pass (when can't move)

**Reward Function**:
```python
# Dense progress reward
reward = (new_pos - old_pos) * 1.0  # +1 per position moved

# Terminal rewards
if won:
    reward += 100.0
elif lost:
    reward -= 100.0

# Penalties
if invalid_action:
    reward -= 1.0
```

**Learning Objectives**:
1. Move forward consistently
2. Understand when moves are valid
3. Reach goal efficiently

**Expected Performance**: 90%+ win rate vs RandomAgent in 2,500 episodes

---

### Level 2: Interaction

**File**: `src/rl_agent_ludo/environment/level2_interaction.py`

**Purpose**: Introduces capturing mechanics and safe zones.

**Game Rules**:
- 2 players, 1 token each
- Full track (60 positions)
- Capturing enabled: Land on opponent → send them home
- Safe zones at positions [10, 20, 30, 40, 50]
- No capturing on safe zones

**State Space (8D)**:
```python
[
    my_pos_norm,           # My position / 60
    opp_pos_norm,          # Opponent position / 60
    am_i_vulnerable,       # Am I at risk? (not in safe zone)
    is_opp_vulnerable,     # Is opponent at risk?
    am_i_in_safe,          # Am I in safe zone?
    is_opp_in_safe,        # Is opponent in safe zone?
    distance_to_opp_norm,  # Distance to opponent / 60
    my_progress            # Overall progress to goal
]
```

**Action Space**: Same as Level 1 (Move or Pass)

**Reward Function**:
```python
# Progress reward
reward += (new_pos - old_pos) * 1.0

# Capture bonus
if captured_opponent:
    reward += 30.0

# Got captured penalty (implicit - reset to home)
# Applied during opponent's turn

# Safe zone bonus
if entered_safe_zone:
    reward += 5.0

# Terminal rewards
if won:
    reward += 100.0
elif lost:
    reward -= 100.0
```

**Learning Objectives**:
1. Offensive play: Capture opponents when possible
2. Defensive play: Use safe zones to avoid capture
3. Risk assessment: Trade-off between aggression and safety

**Expected Performance**: 85%+ win rate vs RandomAgent in 5,000 episodes

---

### Level 3: Multi-Token

**File**: `src/rl_agent_ludo/environment/level3_multitoken.py`

**Purpose**: Introduces token selection strategy with 2 tokens per player.

**Game Rules**:
- 2 players, 2 tokens each
- Full track with captures and safe zones
- Win condition: Both tokens reach goal
- Token coordination required

**State Space (14D)**:
```python
# My tokens (7D)
[
    avg_position_norm,      # Average position of my tokens
    leading_position_norm,  # Furthest token position
    trailing_position_norm, # Trailing token position
    num_at_home,           # 0, 1, or 2 (normalized to [0,1])
    num_at_goal,           # 0, 1, or 2 (normalized to [0,1])
    num_vulnerable,        # How many at risk?
    num_in_safe            # How many in safe zones?
]

# Opponent tokens (7D) - same structure
[...same features for opponent...]
```

**Action Space**:
- 0: Move token 0
- 1: Move token 1
- 2: Pass
- **Action Masking**: Only valid token moves allowed

**Reward Function**:
```python
# Progress reward
reward += (new_pos - old_pos) * 1.0

# Token completion bonus
if token_reached_goal:
    reward += 50.0

# Capture rewards
if captured_opponent:
    reward += 30.0

# Safe zone bonus
if entered_safe_zone:
    reward += 5.0

# Win/loss
if all_tokens_at_goal:
    reward += 100.0
elif opponent_won:
    reward -= 100.0
```

**Learning Objectives**:
1. Token selection: Which token to move?
2. Multi-objective optimization: Balance both tokens
3. Strategic diversity: When to split vs. focus on one token

**Expected Performance**: 75%+ win rate vs RandomAgent in 7,500 episodes

**Key Challenge**: Agent must learn when to:
- Advance leading token (racing strategy)
- Bring up trailing token (balanced strategy)
- Prioritize threatened tokens (defensive)
- Go for captures (offensive)

---

### Level 4: Stochastic

**File**: `src/rl_agent_ludo/environment/level4_stochastic.py`

**Purpose**: Full dice mechanics with dice roll uncertainty.

**Game Rules**: Same as Level 3 with dice rolls (1-6)

**Key Difference**: Agent must handle dice roll outcomes in state representation.

**State Space (16D)**:
- Same as Level 3 (14D) + dice features (2D)
- Dice one-hot encoding or dice value normalized

**Learning Objectives**:
1. Plan under uncertainty
2. Compute expected values across possible dice outcomes
3. Adjust strategy based on dice probabilities

**Expected Performance**: 62%+ win rate vs RandomAgent in 10,000 episodes

---

### Level 5: Multi-Agent

**File**: `src/rl_agent_ludo/environment/level5_multiagent.py`

**Purpose**: Full 4-player competitive Ludo (final challenge).

**Game Rules**:
- 4 players, 2 tokens each
- Full Ludo rules with captures and safe zones
- Win condition: First player to get both tokens home
- Emergent multi-agent dynamics

**State Space (16D)**: Same as Level 4 (egocentric view)

**Action Space**: Same as Level 3/4

**Learning Objectives**:
1. Handle 3 opponents simultaneously
2. Adapt to emergent multi-agent behaviors
3. Balance aggression across multiple threats
4. Navigate complex interaction dynamics

**Expected Performance**: 52%+ win rate vs 3 RandomAgents in 15,000 episodes

**Key Challenges**:
- Non-stationary environment (3 opponents, each learning)
- Partial observability of opponent intentions
- Complex state space with 8 tokens on board
- Credit assignment across long episodes

---

### Unified Ludo Environment

**File**: `src/rl_agent_ludo/environment/unifiedLudoEnv.py`

**Purpose**: Production-ready Ludo environment with Potential-Based Reward Shaping (PBRS).

**Key Innovation**: Theory-grounded reward shaping that preserves optimal policies while accelerating learning.

**Architecture**:
Two environment classes:
- `UnifiedLudoEnv2Tokens`: For 2 tokens per player (28D state)
- `UnifiedLudoEnv4Tokens`: For 4 tokens per player (46D state)

**State Representation**:

**Global Context (10 floats)**:
```python
[
    dice_one_hot[6],        # Dice value (1-6) as one-hot
    our_score,              # Normalized weighted equity score
    enemy1_score,           # Enemy 1 score
    enemy2_score,           # Enemy 2 score
    enemy3_score            # Enemy 3 score
]
```

**Per-Token Features (9 floats × N tokens)**:
```python
[
    zone_identity[5],       # One-hot: [Home, Globe, Star, Victory, Normal]
    normalized_progress,    # Position / 57 (0.0 to 1.0)
    threat_behind,          # Enemy threat level behind (0.0 to 1.0)
    target_ahead,           # Capture opportunity ahead (0.0 to 1.0)
    action_legal            # Can this token move? (0.0 or 1.0)
]
```

**Total State Dimensions**:
- 2 tokens: 10 + (9 × 2) = 28 floats
- 4 tokens: 10 + (9 × 4) = 46 floats

**PBRS Reward Function**:

The environment uses Potential-Based Reward Shaping (Ng et al., 1999) which guarantees preservation of optimal policy:

```python
reward = R_sparse + F
where F = γ * Φ(s') - Φ(s)
```

**Potential Function Φ(s)**:
```python
Φ(s) = W_PROG * Progress + W_SAFE * Safety + W_LEAD * LeadBonus + W_COLLISION * CollisionRisk

where:
  Progress = Sum of normalized token positions
  Safety = Count of tokens on globes/stars
  LeadBonus = Gap between leading and 2nd token (encourages racing)
  CollisionRisk = Average threat level (encourages defense)
```

**Weights**:
- W_PROGRESS = 50.0 (high - drive forward motion)
- W_SAFE = 10.0 (moderate - prefer safe positions)
- W_LEAD = 15.0 (encourages racing strategy)
- W_COLLISION = -20.0 (penalty for risky positions)

**Sparse Rewards** (Ground Truth Objectives):
```python
# Terminal
WIN_REWARD = 100.0
LOSS_PENALTY = -10.0

# Semantic Events
KILL_REWARD = 10.0          # Captured opponent
GOAL_TOKEN_REWARD = 20.0    # Token reached goal
```

**Why PBRS?**
1. Eliminates reward farming cycles (mathematical guarantee)
2. Provides dense gradient signal for learning
3. Preserves optimal policy (same optimal actions as sparse reward)
4. Accelerates convergence without changing solution

**Board Constants**:
```python
HOME_INDEX = 0
GOAL_INDEX = 57
GLOBE_INDEXES = [1, 9, 22, 35, 48]       # Safe from capture
STAR_INDEXES = [5, 12, 18, 25, 31, 38, 44, 51]  # Safe + jump
HOME_CORRIDOR = [52, 53, 54, 55, 56]     # Final stretch
```

**Example Usage**:
```python
# Create environment
env = UnifiedLudoEnv2Tokens(
    player_id=0,
    num_players=4,
    seed=42,
    gamma=0.99  # Discount factor for PBRS
)

# Training loop
obs, info = env.reset()
action_mask = info['action_mask']  # Valid actions
state = info['state']  # Full State object

action = agent.act(obs)
next_obs, reward, done, truncated, info = env.step(action)
```

**Action Masking**:
The environment provides action masks to prevent invalid actions:
```python
action_mask = [True, False, True, False]  # Can move tokens 0 and 2
```

**Expanded State for Coach**:
```python
# For win probability prediction (Coach training)
expanded_state = env.expand_state_with_relative_features(state)
# Adds relative distance and collision risk features
```

---

## Reward Network

The reward network system implements T-REX (Trajectory-ranked Reward EXtrapolation) for learning reward functions from preferences.

### Overview

**T-REX Pipeline**:
```
1. Trajectory Collection → 2. Preference Ranking → 3. Reward Learning → 4. Policy Training
```

---

### Trajectory Collector

**File**: `src/rl_agent_ludo/preference_learning/trajectory_collector.py`

**Purpose**: Collects full game trajectories from agents for preference learning.

**Trajectory Data Structure**:
```python
{
    'episode_id': int,
    'states': List[np.ndarray],      # Full state sequence
    'actions': List[int],             # Action sequence
    'env_rewards': List[float],       # Environment rewards (ignored by T-REX)
    'outcome': str,                   # 'win' or 'loss'
    'winner': int,                    # Winner ID (-1 if no winner)
    'num_captures': int,              # Captures made
    'got_captured': int,              # Times captured
    'episode_length': int,            # Total steps
    'agent_type': str,                # Agent name
    'final_reward': float             # Sum of env rewards
}
```

**Collection Process**:
```python
collector = TrajectoryCollector(save_dir='checkpoints/level6/trajectories')

# Collect from trained agent
trajectories = collector.collect_batch(
    env=env,
    agent=trained_agent,
    num_episodes=1000,
    batch_name='level5_agent',
    seed_start=42
)

# Collect from baseline
baseline_trajs = collector.collect_batch(
    env=env,
    agent=RandomAgent(),
    num_episodes=500,
    batch_name='random_baseline'
)
```

**Statistics Tracked**:
- Win rate
- Average episode length
- Average captures made/received
- Final cumulative reward

**Saved Format**: Pickle files (`.pkl`) containing list of trajectory dictionaries

**Loading Trajectories**:
```python
# Load specific batch
trajs = collector.load_trajectories('level5_agent')

# Load all saved trajectories
all_trajs = collector.load_all_trajectories()

# Summary statistics
collector.summarize_all()
```

---

### Trajectory Ranker

**File**: `src/rl_agent_ludo/preference_learning/trajectory_ranker.py`

**Purpose**: Ranks trajectories and creates preference pairs for reward learning.

**Ranking Criteria** (in priority order):
1. **Win > Loss** (most important)
2. **Among wins**: More captures > Fewer captures
3. **Among wins**: Shorter episode > Longer episode (efficiency)
4. **Among losses**: Survived longer > Died early

**Preference Pair Format**:
```python
(better_trajectory, worse_trajectory)
```

**Creation Process**:
```python
ranker = TrajectoryRanker()

# Create preference pairs
preference_pairs = ranker.create_preference_pairs(
    trajectories=all_trajectories,
    max_pairs=10000,
    seed=42
)

# Split into train/val
train_pairs, val_pairs = ranker.split_train_val(
    train_ratio=0.8,
    seed=42
)

# Save for later use
ranker.save_pairs('checkpoints/level6/preference_pairs.pkl')
```

**Pair Statistics**:
- Total pairs created
- Win vs. loss pairs (strongest signal)
- Average captures (better vs. worse)
- Average episode length (better vs. worse)

**Example Output**:
```
Created 8,742 preference pairs
  (Attempted 10,000 pairings, 87.4% valid)

Preference Pair Statistics:
  Better trajectories that won: 6,234 (71.3%)
  Worse trajectories that lost: 6,189 (70.8%)
  Win vs Loss pairs: 5,983 (68.5%)

  Better trajectories:
    Avg captures: 1.8
    Avg length: 142.3
  Worse trajectories:
    Avg captures: 0.4
    Avg length: 187.6
```

---

### Reward Network

**File**: `src/rl_agent_ludo/preference_learning/reward_network.py`

**Purpose**: Neural network that learns to predict rewards from state observations.

**Architecture**:
```python
RewardNetwork(
    state_dim=16,        # Input dimension
    hidden_dim=128       # Hidden layer size
)

Network:
  Linear(16, 128) → ReLU → Dropout(0.1)
  Linear(128, 128) → ReLU → Dropout(0.1)
  Linear(128, 1)   # Scalar reward output
```

**Key Methods**:

1. **Forward Pass**:
```python
reward = reward_net(state)  # Returns scalar reward
```

2. **Return Prediction**:
```python
total_return = reward_net.predict_return(
    states=trajectory_states,
    discount=0.99
)
# Returns: Sum of discounted rewards along trajectory
```

**Training Objective**:
The network learns to assign rewards such that better trajectories get higher predicted returns.

---

### Reward Learner

**File**: `src/rl_agent_ludo/preference_learning/reward_network.py` (RewardLearner class)

**Purpose**: Trains reward network from preference pairs using Bradley-Terry model.

**Loss Function** (Bradley-Terry):
```python
P(better > worse) = exp(R_better) / (exp(R_better) + exp(R_worse))

Loss = -log P(better > worse)
     = log(1 + exp(R_worse - R_better))

where:
  R_better = Sum of discounted rewards for better trajectory
  R_worse = Sum of discounted rewards for worse trajectory
```

**Training Process**:
```python
learner = RewardLearner(
    state_dim=16,
    hidden_dim=128,
    learning_rate=3e-4,
    weight_decay=1e-5,
    device='cuda'
)

# Train with early stopping
learner.train(
    train_pairs=train_pairs,
    val_pairs=val_pairs,
    num_epochs=100,
    batch_size=32,
    patience=10,
    verbose=True
)

# Save trained reward network
learner.save('checkpoints/level6/reward_learner_final.pth')
```

**Training Features**:
1. **Mini-batch Training**: Process pairs in batches for efficiency
2. **Early Stopping**: Stop when validation loss stops improving
3. **Gradient Clipping**: Prevent exploding gradients (max norm = 1.0)
4. **Checkpointing**: Save best model based on validation loss

**Metrics Tracked**:
- Training loss (ranking loss)
- Validation loss
- Training accuracy (% correctly ranked pairs)
- Validation accuracy

**Example Training Output**:
```
Training reward network for 100 epochs...
Train pairs: 6,994
Val pairs: 1,748
Device: cuda
Batch size: 32

Epoch   1/100 | Train Loss: 0.6823 Acc: 0.562 | Val Loss: 0.6421 Acc: 0.601
Epoch   2/100 | Train Loss: 0.5834 Acc: 0.678 | Val Loss: 0.5623 Acc: 0.689
Epoch   3/100 | Train Loss: 0.4921 Acc: 0.742 | Val Loss: 0.4892 Acc: 0.751
...
Epoch  45/100 | Train Loss: 0.1234 Acc: 0.951 | Val Loss: 0.1567 Acc: 0.932

Early stopping at epoch 45

Training complete!
Best val loss: 0.1567
Final val accuracy: 0.932
```

**Validation Accuracy Interpretation**:
- 93% accuracy means the reward network correctly ranks 93% of trajectory pairs
- High accuracy indicates the network learned meaningful state values

---

### Complete T-REX Workflow

**Step 1: Collect Trajectories**
```python
from rl_agent_ludo.preference_learning.trajectory_collector import TrajectoryCollector

collector = TrajectoryCollector()

# Collect from multiple agents at different skill levels
collector.collect_batch(env, expert_agent, 500, 'expert')
collector.collect_batch(env, intermediate_agent, 500, 'intermediate')
collector.collect_batch(env, random_agent, 500, 'random')
```

**Step 2: Rank Trajectories**
```python
from rl_agent_ludo.preference_learning.trajectory_ranker import TrajectoryRanker

ranker = TrajectoryRanker()
all_trajs = collector.load_all_trajectories()

preference_pairs = ranker.create_preference_pairs(all_trajs, max_pairs=10000)
train_pairs, val_pairs = ranker.split_train_val(train_ratio=0.8)
```

**Step 3: Learn Reward Function**
```python
from rl_agent_ludo.preference_learning.reward_network import RewardLearner

learner = RewardLearner(state_dim=16, device='cuda')
learner.train(train_pairs, val_pairs, num_epochs=100)
learner.save('checkpoints/level6/reward_network_best.pth')
```

**Step 4: Train Policy with Learned Rewards**
```python
from rl_agent_ludo.agents.trex_agent import TREXAgent

agent = TREXAgent(
    state_dim=16,
    action_dim=3,
    reward_network_path='checkpoints/level6/reward_network_best.pth',
    use_hybrid_rewards=True
)

# Standard RL training loop with learned rewards
for episode in range(num_episodes):
    state, info = env.reset()
    done = False

    while not done:
        action = agent.act(state)
        next_state, env_reward, done, _, info = env.step(action)
        agent.store_experience(state, action, env_reward, next_state, done)
        agent.train_step()
        state = next_state

    agent.decay_epsilon()
```

---

## Integration Example

**Complete Training Pipeline**:

```python
import torch
from rl_agent_ludo.environment.level5_multiagent import Level5MultiAgentLudo
from rl_agent_ludo.agents.simple_dqn import SimpleDQNAgent
from rl_agent_ludo.agents.baseline_agents import RandomAgent

# 1. Create environment
env = Level5MultiAgentLudo(seed=42)

# 2. Initialize agent
agent = SimpleDQNAgent(
    state_dim=16,
    action_dim=3,
    learning_rate=5e-5,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# 3. Training loop
num_episodes = 15000
eval_freq = 1000

for episode in range(num_episodes):
    state, info = env.reset()
    done = False
    episode_reward = 0

    while not done:
        # Agent acts
        action = agent.act(state)

        # Environment step
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward

        # Learn
        agent.store_experience(state, action, reward, next_state, done)
        loss = agent.train_step()

        state = next_state

    # Episode end
    agent.decay_epsilon()

    # Evaluate periodically
    if (episode + 1) % eval_freq == 0:
        eval_wins = evaluate_agent(agent, env, num_games=200)
        print(f"Episode {episode+1}: Win Rate = {eval_wins/200*100:.1f}%")

        # Save checkpoint
        agent.save(f'checkpoints/level5/model_ep{episode+1}.pth')
```

---

## File Reference Summary

**Agents**:
- `baseAgent.py` - Abstract agent interface
- `baseline_agents.py` - Random, Greedy, AlwaysMove agents
- `simple_dqn.py` - DQN with experience replay
- `trex_agent.py` - DQN with learned rewards

**Environment**:
- `level1_simple.py` - Basic movement (4D state)
- `level2_interaction.py` - With captures (8D state)
- `level3_multitoken.py` - Multi-token (14D state)
- `level4_stochastic.py` - Stochastic dice (16D state)
- `level5_multiagent.py` - 4-player game (16D state)
- `unifiedLudoEnv.py` - Production env with PBRS

**Reward Network**:
- `trajectory_collector.py` - Collect game trajectories
- `trajectory_ranker.py` - Rank and create preference pairs
- `reward_network.py` - Neural reward function + learner

---

## References

**Academic Papers**:

1. Ng, A. Y., Harada, D., & Russell, S. (1999). Policy invariance under reward transformations: Theory and application to reward shaping. *ICML*.

2. Brown, D., Goo, W., Nagarajan, P., & Niekum, S. (2019). Extrapolating beyond suboptimal demonstrations via inverse reinforcement learning from observations. *ICML*. (T-REX)

3. Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529-533. (DQN)

4. Wang, Z., Schaul, T., Hessel, M., Van Hasselt, H., Lanctot, M., & Freitas, N. (2016). Dueling network architectures for deep reinforcement learning. *ICML*.

**Software and Libraries**:

5. Towers, M., et al. (2023). Gymnasium: A standard interface for reinforcement learning environments. https://github.com/Farama-Foundation/Gymnasium

6. PyTorch Team. (2023). PyTorch: An open source machine learning framework. https://pytorch.org/

**Code Repositories**:

7. Sangrasi, M. (n.d.). AI-Ludo. https://github.com/MehranSangrasi/AI-Ludo

8. Aurucci, R. (n.d.). Ludo_Game_AI. https://github.com/raffaele-aurucci/Ludo_Game_AI

---

## Quick Reference

**Training a Curriculum Level**:
```bash
python experiments/level1_train.py --episodes 2500 --eval_freq 500
python experiments/level2_train.py --episodes 5000 --eval_freq 1000
python experiments/level5_train.py --episodes 15000 --eval_freq 1000
```

**Testing a Trained Agent**:
```bash
python experiments/test_level5.py --checkpoint checkpoints/level5/best_model.pth --num_eval 400
```

**T-REX Pipeline**:
```bash
# 1. Collect trajectories
python experiments/level6_collect_trajectories.py

# 2. Learn reward function
python experiments/level6_learn_reward.py

# 3. Train policy with learned rewards
python experiments/level6_train_policy.py
```

---

**End of Documentation**
