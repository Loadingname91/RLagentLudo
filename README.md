# Reinforcement Learning for Ludo: A Curriculum-Based Approach

A systematic framework for training deep reinforcement learning agents to master the game of Ludo through progressive difficulty levels. This project implements a 6-level curriculum that builds from basic movement to full 4-player competitive gameplay with preference-based reward learning, achieving **66% win rate** in the final challenge.

<p align="center">
  <img src="assets/demo_gameplay.gif" alt="Agent Playing Ludo" width="600"/>
</p>

<p align="center">
  <em>Trained DQN agent playing 4-player Ludo in real-time</em>
</p>

<p align="center">
  <img src="results/visualizations/dashboard.png" alt="Training Results Dashboard" width="900"/>
</p>

## Overview

This project explores the application of deep reinforcement learning to Ludo, a complex stochastic multi-agent board game. Rather than jumping directly to the full game complexity, we employ a **curriculum learning approach** that incrementally introduces game mechanics:

- **Level 1**: Single token, no opponent interaction (basic movement)
- **Level 2**: Single token with opponent interactions (captures)
- **Level 3**: Multiple tokens per player (token selection strategy)
- **Level 4**: Full stochastic dice mechanics
- **Level 5**: 4-player multi-agent competition
- **Level 6**: T-REX with learned rewards from preferences

This structured approach enables the agent to learn fundamental skills before tackling the full game's strategic depth.

## Results

### Win Rates Across All Levels

<p align="center">
  <img src="results/visualizations/win_rates.png" alt="Win Rates" width="800"/>
</p>

### Performance Metrics

<p align="center">
  <img src="results/visualizations/rewards.png" alt="Average Rewards" width="800"/>
</p>

<p align="center">
  <img src="results/visualizations/episode_lengths.png" alt="Episode Lengths" width="800"/>
</p>

### Results Summary

| Level | Challenge | Achieved | Episodes |
|-------|-----------|----------|----------|
| 1 | Basic Movement | **95%** | 2,500 |
| 2 | Opponent Interaction | **90%** | 5,000 |
| 3 | Multi-Token Strategy | **78%** | 7,500 |
| 4 | Stochastic Dynamics | **67%** | 10,000 |
| 5 | Multi-Agent Chaos | **61%** | 15,000 |
| 6 | T-REX (Learned Rewards) | **66%** | 35,000 |

The agents demonstrate strong performance across all levels, with Level 6 showing **2.6x better than random baseline** (25%) through preference-based learning.

## Key Features

- **Curriculum-Based Training**: 6 progressive difficulty levels with clear success metrics
- **Multiple Architectures**: SimpleDQN and T-REX (preference-based learning)
- **Potential-Based Reward Shaping (PBRS)**: Theory-grounded reward engineering that preserves optimal policies
- **Preference Learning**: T-REX implementation for learning rewards from trajectory rankings
- **Comprehensive Evaluation**: Detailed metrics tracking win rates, captures, game lengths, and learning dynamics
- **Visual Gameplay**: Real-time CV2 visualization of trained agents playing
- **Modular Architecture**: Clean separation between environments, agents, and training logic
- **Reproducibility**: Seed management and hyperparameter tracking for all experiments

## Installation

### Prerequisites

- Python 3.8+
- pip or conda

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd RLagentLudo
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Training

Train an agent on a specific level:

```bash
# Level 1: Basic movement
python experiments/level1_train.py --episodes 2500 --eval_freq 500

# Level 5: Full game (4 players, 2 tokens each)
python experiments/level5_train.py --episodes 15000 --eval_freq 1000

# Level 6: T-REX with learned rewards
python experiments/level6_train_policy.py --episodes 35000
```

### Visual Demo

Watch a trained agent play with graphical visualization:

```bash
# Demo Level 5 agent with CV2 window (interactive)
python experiments/demo_visual.py --level 5 --episodes 3

# Generate animated GIF of gameplay (for README/presentations)
python experiments/generate_demo_gif.py --episodes 1 --max_steps 250 --fps 10
```

This will create `assets/demo_gameplay.gif` showing the agent playing.

### Testing

Evaluate a trained model:

```bash
# Test Level 5 agent
python experiments/test_level5.py --checkpoint checkpoints/level5/best_model.pth --num_eval 400
```

### Evaluation & Visualization

Run comprehensive evaluation and generate visualizations:

```bash
# Evaluate all levels (1-6)
python experiments/evaluate_all_models.py

# Generate visualization plots
python experiments/visualize_results.py --results results/evaluations/all_models_evaluation_*.json
```

## Project Structure

```
RLagentLudo/
├── experiments/              # Training and testing scripts
│   ├── level1_train.py      # Level 1: Basic movement
│   ├── level2_train.py      # Level 2: With captures
│   ├── level3_train.py      # Level 3: Multi-token
│   ├── level4_train.py      # Level 4: Stochastic
│   ├── level5_train.py      # Level 5: Multi-agent
│   ├── level6_train_policy.py  # Level 6: T-REX
│   ├── demo_visual.py       # Visual demo with CV2
│   ├── evaluate_all_models.py  # Comprehensive evaluation
│   └── visualize_results.py # Generate plots
├── src/rl_agent_ludo/
│   ├── agents/              # Agent implementations
│   │   ├── simple_dqn.py           # DQN with experience replay
│   │   ├── trex_agent.py           # T-REX with learned rewards
│   │   └── baseline_agents.py      # Random, Greedy agents
│   ├── environment/         # Environment wrappers
│   │   ├── level1_simple.py        # Level 1 environment
│   │   ├── level2_interaction.py   # Level 2 environment
│   │   ├── level3_multitoken.py    # Level 3 environment
│   │   ├── level4_stochastic.py    # Level 4 environment
│   │   ├── level5_multiagent.py    # Level 5 environment
│   │   └── unifiedLudoEnv.py       # Production env with PBRS
│   ├── preference_learning/ # T-REX components
│   │   ├── trajectory_collector.py  # Collect demonstrations
│   │   ├── trajectory_ranker.py     # Rank trajectories
│   │   └── reward_network.py        # Learn reward function
├── results/                 # Evaluation results and plots
│   └── visualizations/      # Generated PNG plots
└── requirements.txt
```

## Agent Architectures

### SimpleDQN Agent (Levels 1-5)

The primary agent uses a **DQN** architecture with:

1. **Experience Replay Buffer**: Stores transitions, breaks correlation
2. **Target Network**: Separate network for stable Q-value targets
3. **Epsilon-Greedy Exploration**: Decays from 1.0 to 0.05
4. **Gradient Clipping**: Prevents exploding gradients

**Network Architecture:**
- Input: State vector (4D to 16D depending on level)
- Hidden layers: 128x128 (ReLU activation)
- Output: Q-values for each action

### T-REX Agent (Level 6)

The T-REX agent learns from trajectory preferences:

**Innovation**: Instead of hand-crafted rewards, learns reward function from ranked demonstrations

**Pipeline**:
1. **Collect Trajectories**: Run agents and record full game sequences
2. **Rank Trajectories**: Create preference pairs (better vs. worse)
3. **Learn Reward Network**: Train neural network to predict rewards
4. **Train Policy**: Use learned rewards to train DQN

**Hybrid Reward Mode**:
```
total_reward = env_reward + 0.3 × 10.0 × learned_reward
```

This combines sparse environment signals with dense learned feedback for best performance.

## Reward Shaping

The project uses **Potential-Based Reward Shaping (PBRS)** to guide learning while preserving optimal policies:

- **Win/Loss**: +100 (win), -100 (loss)
- **Progress Shaping**: Distance-based potential function
- **Capture Rewards**: +30 (capture), -30 (captured)
- **Goal Completion**: +50 per token

PBRS guarantees that the shaped reward function has the same optimal policy as the original sparse reward, while significantly accelerating learning.

## Curriculum Design

### Level 1: Basic Movement (2,500 episodes)
- **Goal**: Learn to move a single token from start to goal
- **State**: 4D (1 token position, goal flag, distance, progress)
- **Actions**: Move token or pass
- **Challenge**: Basic sequential decision-making

### Level 2: Opponent Interaction (5,000 episodes)
- **Goal**: Learn to capture opponents and avoid being captured
- **State**: 8D (player + opponent token states)
- **Challenge**: Adversarial interaction, risk assessment

### Level 3: Multi-Token Strategy (7,500 episodes)
- **Goal**: Manage 2 tokens simultaneously, strategic token selection
- **State**: 14D (2 tokens × 2 players)
- **Actions**: Move token 0, token 1, or pass
- **Challenge**: Resource allocation, multi-objective optimization

### Level 4: Stochastic Dynamics (10,000 episodes)
- **Goal**: Handle full dice mechanics (1-6 outcomes)
- **State**: 16D
- **Challenge**: Partial observability, long-term planning under uncertainty

### Level 5: Multi-Agent Chaos (15,000 episodes)
- **Goal**: Compete against 3 random opponents simultaneously
- **State**: 16D (egocentric view)
- **Challenge**: Full game complexity, emergent multi-agent dynamics

### Level 6: T-REX Learning (35,000 episodes)
- **Goal**: Improve over Level 5 using learned rewards
- **Innovation**: Preference-based reward learning
- **Result**: 66% win rate (vs Level 5's 61%)

## Evaluation Metrics

Each level tracks:
- **Win Rate**: Primary success metric
- **Average Reward**: Cumulative episode reward
- **Game Length**: Steps per episode
- **Capture Statistics**: Captures made vs. received
- **Epsilon**: Exploration rate (decays from 1.0 to 0.05)
- **Replay Buffer Size**: Experience collected

Evaluations run with 500 test games per level against random opponents.

## Documentation

- **`COMPONENTS_DOCUMENTATION.md`** - Detailed documentation of all agents, environments, and reward networks
- **`LEVEL6_EVALUATION_GUIDE.md`** - Guide for evaluating Level 6 (T-REX)
- **`docs/`** - Additional architecture and methodology docs

## Research Foundation

This project builds upon established research in:
- **Curriculum Learning**: Progressive task difficulty for skill acquisition
- **Reward Shaping**: Potential-based reward shaping (Ng et al., 1999)
- **Deep RL**: DQN architectures (Mnih et al., 2015)
- **Preference Learning**: T-REX for learning from demonstrations (Brown et al., 2019)
- **Multi-Agent RL**: Competitive gameplay and emergent strategies

### Key References

1. Ng, A. Y., Harada, D., & Russell, S. (1999). *Policy invariance under reward transformations.* ICML. (PBRS theory)
2. Brown, D., et al. (2019). *Extrapolating Beyond Suboptimal Demonstrations via Inverse Reinforcement Learning from Observations.* ICML. (T-REX)
3. Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning.* Nature. (DQN)

## Future Work

Potential extensions and improvements:
- **Self-Play Training**: Train against past versions of the agent
- **Multi-Agent Learning**: Simultaneous training of all players
- **Policy Gradient Methods**: PPO, A3C for continuous improvement
- **Opponent Modeling**: Explicit modeling of opponent strategies
- **Human Evaluation**: Testing against human players

## License

See LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{rl_agent_ludo_curriculum,
  title = {Reinforcement Learning for Ludo: A Curriculum-Based Approach},
  author = {Balegar, Hitesh},
  year = {2025},
  url = {https://github.com/yourusername/RLagentLudo},
  note = {Deep RL with progressive curriculum and preference learning for multi-agent board games}
}
```

## Acknowledgments

This project builds upon:
- **LudoPy**: Python implementation of Ludo game mechanics
- **DeepMind**: DQN and Deep RL architectures
- **OpenAI**: Reinforcement learning best practices and methodologies
- Existing research on RL applications to board games and curriculum learning
