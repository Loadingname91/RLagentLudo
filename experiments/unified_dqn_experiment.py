"""
Unified DQN Experiment Script

Trains a Unified DQN agent using the "Egocentric Physics" approach with:
- Unified Feature Vector (46 floats for 4 tokens, 28 floats for 2 tokens)
- Action Masking (Logit Masking) for invalid moves
- Delta-Progress + Event Impulses + ILA Penalty reward structure

CPU Performance Optimizations:
- Large batch size (256) for better CPU core utilization
- Train less frequently (every 16 steps) to reduce overhead
- Multiple gradient steps per training call (8) for better CPU efficiency
- Optimized tensor operations using torch.from_numpy
- PyTorch threading configured to use all available CPU cores
"""

import gymnasium as gym
import numpy as np
import sys
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rl_agent_ludo.environment.unifiedLudoEnv import (
    UnifiedLudoEnv2Tokens,
    UnifiedLudoEnv4Tokens,
)
from rl_agent_ludo.agents.unifiedDQNAgent import (
    create_unified_dqn_agent_2tokens,
    create_unified_dqn_agent_4tokens,
    UnifiedDQNAgent,
)
import torch


def _run_greedy_evaluation(
    agent: UnifiedDQNAgent,
    env,
    num_episodes: int = 10,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Run greedy evaluation episodes with epsilon=0 (no exploration).
    
    This reveals the true strength of the current policy without exploration noise.
    
    Args:
        agent: The agent to evaluate
        env: The environment to evaluate in
        num_episodes: Number of evaluation episodes
        seed: Random seed for evaluation
        verbose: Whether to print progress
        
    Returns:
        Dictionary with evaluation statistics
    """
    # Save current epsilon
    original_epsilon = agent.epsilon
    
    # Set epsilon to 0 for greedy evaluation
    agent.epsilon = 0.0
    
    eval_wins = 0
    eval_rewards = []
    eval_lengths = []
    
    for eval_episode in range(num_episodes):
        obs, info = env.reset(seed=seed + eval_episode)
        state = info["state"]
        action_mask = info["action_mask"]
        done = False
        episode_length = 0
        episode_reward = 0.0
        
        max_steps = 10000
        
        while not done and episode_length < max_steps:
            # Greedy action selection (epsilon=0)
            action = agent.act(state, obs=obs, action_mask=action_mask)
            
            obs, reward, terminated, truncated, info = env.step(action)
            next_state = info["state"]
            next_action_mask = info["action_mask"]
            done = terminated or truncated
            
            # Don't store experiences or train during evaluation
            state = next_state
            action_mask = next_action_mask
            episode_reward += reward
            episode_length += 1
        
        eval_lengths.append(episode_length)
        eval_rewards.append(episode_reward)
        
        # Check if agent won
        if done:
            unwrapped_env = env
            while hasattr(unwrapped_env, 'env'):
                unwrapped_env = unwrapped_env.env
            if hasattr(unwrapped_env, 'game') and unwrapped_env.game:
                winners = unwrapped_env.game.get_winners_of_game()
                if hasattr(unwrapped_env, 'player_id') and unwrapped_env.player_id in winners:
                    eval_wins += 1
    
    # Restore original epsilon
    agent.epsilon = original_epsilon
    
    return {
        "win_rate": eval_wins / num_episodes if num_episodes > 0 else 0.0,
        "wins": eval_wins,
        "num_episodes": num_episodes,
        "avg_reward": float(np.mean(eval_rewards)) if len(eval_rewards) > 0 else 0.0,
        "std_reward": float(np.std(eval_rewards)) if len(eval_rewards) > 0 else 0.0,
        "avg_episode_length": float(np.mean(eval_lengths)) if len(eval_lengths) > 0 else 0.0,
        "std_episode_length": float(np.std(eval_lengths)) if len(eval_lengths) > 0 else 0.0,
    }


def run_experiment(
    num_episodes: int = 10000,
    num_players: int = 4,
    tokens_per_player: int = 4,
    seed: int = 2024,
    verbose: bool = True,
    learning_rate: float = 0.001,
    discount_factor: float = 0.95,
    epsilon: float = 1.0,
    epsilon_decay: float = 0.995,  # Not used when epsilon_schedule is set
    min_epsilon: float = 0.01,
    epsilon_schedule: str = 'exponential',  # 'linear', 'exponential', or 'adaptive'
    epsilon_decay_fraction: float = 0.9,  # Fraction of episodes to decay over (0.9 = 90%)
    batch_size: int = 256,  # Large batch size for better CPU utilization
    replay_buffer_size: int = 150000,
    target_update_frequency: int = 1000,
    train_frequency: int = 32,  # Train every 32 steps (reduced frequency for speed)
    gradient_steps: int = 1,  # Single gradient step per training call (optimized for speed)
    hidden_dims: list = [128, 128, 64],  # Smaller network for faster training
    device: str = None,  # 'cuda', 'cpu', or None for auto-detect
    checkpoint_frequency: int = 10000,  # Save checkpoint every N episodes (0 to disable)
    save_best_model: bool = True,  # Save best model based on recent win rate
    best_model_window: int = 1000,  # Episodes to consider for best model (recent win rate)
    resume_from_checkpoint: str = None,  # Path to checkpoint file to resume from
    eval_frequency: int = 1000,  # Run greedy evaluation every N episodes (0 to disable)
    num_eval_episodes: int = 10,  # Number of episodes to run during evaluation
) -> dict:
    """
    Run a Unified DQN experiment.
    
    This trains the Unified DQN agent using the unified feature vector
    with action masking and the new reward structure.
    """
    # Create environment based on tokens_per_player
    # Pass gamma (discount_factor) to environment for PBRS calculation
    if tokens_per_player == 2:
        env = UnifiedLudoEnv2Tokens(
            player_id=0,
            num_players=num_players,
            seed=seed,
            gamma=discount_factor,  # PBRS requires gamma
        )
    elif tokens_per_player == 4:
        env = UnifiedLudoEnv4Tokens(
            player_id=0,
            num_players=num_players,
            seed=seed,
            gamma=discount_factor,  # PBRS requires gamma
        )
    else:
        raise ValueError(f"tokens_per_player must be 2 or 4, got {tokens_per_player}")

    # Auto-detect device if not specified
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if verbose:
        print(f"Using device: {device}")
        if device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Checkpointing setup (before agent creation for directory structure)
    checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Create subdirectory based on hyperparameters for organization
    hyperparam_str = (
        f"ep{num_episodes//1000}k_"
        f"lr{learning_rate}_"
        f"bs{batch_size}_"
        f"tf{train_frequency}_"
        f"gs{gradient_steps}_"
        f"eps{epsilon_schedule}_"
        f"h{'-'.join(map(str, hidden_dims))}"
    )
    run_checkpoint_dir = checkpoint_dir / f"unified_dqn_{tokens_per_player}t_{num_players}p" / hyperparam_str
    run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose and resume_from_checkpoint is None:
        print(f"Checkpoint directory: {run_checkpoint_dir}")
    
    # Create agent based on tokens_per_player
    # Use epsilon_decay=1.0 to disable automatic decay (we'll use manual scheduling like dqn_selfplay.py)
    if tokens_per_player == 2:
        agent = create_unified_dqn_agent_2tokens(
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon,
            epsilon_decay=1.0,  # Disable automatic decay, use manual scheduling
            min_epsilon=min_epsilon,
            batch_size=batch_size,
            replay_buffer_size=replay_buffer_size,
            target_update_frequency=target_update_frequency,
            train_frequency=train_frequency,
            gradient_steps=gradient_steps,
            hidden_dims=hidden_dims,
            device=device,
            seed=seed,
        )
    else:  # tokens_per_player == 4
        agent = create_unified_dqn_agent_4tokens(
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon,
            epsilon_decay=1.0,  # Disable automatic decay, use manual scheduling
            min_epsilon=min_epsilon,
            batch_size=batch_size,
            replay_buffer_size=replay_buffer_size,
            target_update_frequency=target_update_frequency,
            train_frequency=train_frequency,
            gradient_steps=gradient_steps,
            hidden_dims=hidden_dims,
            device=device,
            seed=seed,
        )
    
    # Load checkpoint if resuming (loads agent weights but starts from episode 0)
    if resume_from_checkpoint and Path(resume_from_checkpoint).exists():
        if verbose:
            print(f"Resuming from checkpoint: {resume_from_checkpoint}")
        agent.load(resume_from_checkpoint)
        loaded_buffer_size = getattr(agent, '_loaded_replay_buffer_size', 0)
        current_buffer_size = len(agent.replay_buffer)
        if verbose:
            print(f"  Loaded agent state: epsilon={agent.epsilon:.4f}, previous episodes={agent.episode_count}, previous steps={agent.step_count}")
            if loaded_buffer_size > 0:
                print(f"  Restored replay buffer: {loaded_buffer_size:,} experiences (current size: {current_buffer_size:,})")
            else:
                print(f"  ⚠️  No replay buffer found in checkpoint (old format or empty). Starting with empty buffer.")
                print(f"  Current replay buffer size: {current_buffer_size:,}")
            print(f"  Starting training from episode 0 (agent weights preserved)")

    wins = 0
    losses = 0
    episode_lengths = []
    rewards = []
    
    # Debug stats for periodic logging (similar to dqn_selfplay.py)
    debug_stats = {
        'win_by_episode': [],
        'reward_by_episode': [],
        'exploration_rate': [],
        'replay_buffer_size': [],
    }
    
    # Periodic stats history (captured at each log interval)
    periodic_stats_history = []
    
    # Evaluation history (greedy evaluation results)
    eval_history = []
    
    # Q-value monitoring
    q_value_stats = {
        'mean_q_values': [],
        'max_q_values': [],
        'min_q_values': [],
        'std_q_values': [],
    }
    
    # Log every N episodes for debugging (more frequent for better monitoring)
    log_interval = max(1, num_episodes // 50)  # Log ~50 times during training
    
    # Best model tracking (using rolling average of evaluations)
    best_rolling_avg_eval_win_rate = 0.0
    best_model_path = None
    rolling_avg_window = 3  # Track rolling average of last 3 evaluations

    print(
        f"Running Unified DQN experiment with {num_episodes} episodes "
        f"({num_players} players, {tokens_per_player} tokens, "
        f"unified state abstraction)..."
    )

    tqdm_bar = tqdm(total=num_episodes, initial=0, desc="Running episodes", disable=not verbose)

    for episode in range(num_episodes):
        # Manual epsilon scheduling (optimized for long training runs)
        decay_duration = int(num_episodes * epsilon_decay_fraction)
        progress = min(1.0, episode / max(decay_duration, 1))
        
        if epsilon_schedule == 'exponential':
            # Exponential decay: ε_t = ε_min + (ε_0 - ε_min) * (decay_rate)^t
            # For long runs: decay_rate calculated to reach min_epsilon at decay_duration
            # Using: decay_rate = (min_epsilon / epsilon)^(1/decay_duration)
            # This gives smooth exponential decay
            if decay_duration > 0:
                decay_rate = (min_epsilon / epsilon) ** (1.0 / decay_duration)
                current_epsilon = min_epsilon + (epsilon - min_epsilon) * (decay_rate ** episode)
            else:
                current_epsilon = epsilon
            current_epsilon = max(min(current_epsilon, epsilon), min_epsilon)
            
        elif epsilon_schedule == 'adaptive':
            # Adaptive decay: Slower decay if performance is improving
            # Use exponential but adjust based on recent win rate
            if decay_duration > 0:
                base_decay_rate = (min_epsilon / epsilon) ** (1.0 / decay_duration)
                # Adjust decay rate based on recent performance (if available)
                if len(debug_stats['win_by_episode']) > 100:
                    recent_window = min(100, len(debug_stats['win_by_episode']))
                    recent_win_rate = sum(debug_stats['win_by_episode'][-recent_window:]) / recent_window
                    # If winning more, decay faster; if struggling, decay slower
                    performance_factor = 0.8 + (recent_win_rate * 0.4)  # Range: 0.8-1.2
                    adjusted_decay_rate = base_decay_rate ** performance_factor
                else:
                    adjusted_decay_rate = base_decay_rate
                current_epsilon = min_epsilon + (epsilon - min_epsilon) * (adjusted_decay_rate ** episode)
            else:
                current_epsilon = epsilon
            current_epsilon = max(min(current_epsilon, epsilon), min_epsilon)
            
        else:  # 'linear' (default, matches dqn_selfplay.py)
            # Linear decay: Simple and predictable
            current_epsilon = epsilon - (epsilon - min_epsilon) * progress
            current_epsilon = max(min(current_epsilon, epsilon), min_epsilon)
        
        agent.epsilon = current_epsilon
        
        obs, info = env.reset(seed=seed + episode)
        state = info["state"]
        action_mask = info["action_mask"]
        done = False
        episode_length = 0
        episode_reward = 0.0

        # Safety cap on steps per episode
        max_steps = 10000

        while not done and episode_length < max_steps:
            prev_state = state
            prev_obs = obs.copy()
            prev_action_mask = action_mask.copy()
            
            # Select action with masking
            action = agent.act(state, obs=obs, action_mask=action_mask)
            
            # Q-value monitoring (sample periodically to avoid overhead)
            if episode_length % 10 == 0:  # Sample every 10 steps
                with torch.no_grad():
                    obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(agent.device)
                    action_mask_tensor = torch.from_numpy(action_mask).bool().unsqueeze(0).to(agent.device)
                    q_values = agent.q_network(obs_tensor, action_mask_tensor)[0].cpu().numpy()
                    # Only track valid actions
                    valid_q_values = q_values[action_mask]
                    if len(valid_q_values) > 0:
                        # Skip NaN values (can occur early in training)
                        valid_q_values_clean = valid_q_values[~np.isnan(valid_q_values)]
                        if len(valid_q_values_clean) > 0:
                            q_value_stats['mean_q_values'].append(float(np.mean(valid_q_values_clean)))
                            q_value_stats['max_q_values'].append(float(np.max(valid_q_values_clean)))
                            q_value_stats['min_q_values'].append(float(np.min(valid_q_values_clean)))
                            q_value_stats['std_q_values'].append(float(np.std(valid_q_values_clean)))

            obs, reward, terminated, truncated, info = env.step(action)
            next_state = info["state"]
            next_action_mask = info["action_mask"]
            done = terminated or truncated

            # Store experience and train
            agent.push_to_replay_buffer(
                prev_state,
                action,
                reward,
                next_state,
                done,
                obs=prev_obs,
                next_obs=obs,
                action_mask=prev_action_mask,
                next_action_mask=next_action_mask,
            )

            state = next_state
            action_mask = next_action_mask

            episode_reward += reward
            episode_length += 1

        episode_lengths.append(episode_length)
        rewards.append(episode_reward)

        # Check actual game outcome (similar to dqn_selfplay.py)
        episode_won = False
        if done:
            # Unwrap environment if it's wrapped (e.g., TimeLimit)
            unwrapped_env = env
            while hasattr(unwrapped_env, 'env'):
                unwrapped_env = unwrapped_env.env
            # Access game object to check winners
            if hasattr(unwrapped_env, 'game') and unwrapped_env.game:
                winners = unwrapped_env.game.get_winners_of_game()
                episode_won = unwrapped_env.player_id in winners if hasattr(unwrapped_env, 'player_id') else False
            # Fallback: check terminal reward if game object not available
            elif reward == 100.0:  # Win reward
                episode_won = True
        
        # Track wins and losses
        if episode_won:
            wins += 1
        elif done:  # Game ended but agent didn't win
            losses += 1
        
        # Store debug stats
        debug_stats['win_by_episode'].append(1 if episode_won else 0)
        debug_stats['reward_by_episode'].append(episode_reward)

        tqdm_bar.update(1)
        tqdm_bar.set_postfix(
            episode=episode,
            reward=episode_reward,
            length=episode_length,
            epsilon=agent.epsilon,
            buffer_size=len(agent.replay_buffer),
        )

        # Debug logging every N episodes (similar to dqn_selfplay.py)
        if (episode + 1) % log_interval == 0 or episode == 0:
            replay_buffer_size = len(agent.replay_buffer)
            debug_stats['exploration_rate'].append(agent.epsilon)
            debug_stats['replay_buffer_size'].append(replay_buffer_size)
            
            # Calculate recent win rate (last 10% of episodes so far, but at least 10 episodes)
            # Ensure we have enough data for accurate recent win rate
            total_episodes_so_far = len(debug_stats['win_by_episode'])
            recent_window = max(10, total_episodes_so_far // 10)
            # Only use episodes we actually have data for
            recent_window = min(recent_window, total_episodes_so_far)
            if recent_window > 0:
                recent_wins = sum(debug_stats['win_by_episode'][-recent_window:])
                recent_win_rate = recent_wins / recent_window
            else:
                recent_win_rate = 0.0
            
            # Calculate recent average reward
            recent_avg_reward = np.mean(debug_stats['reward_by_episode'][-recent_window:]) if recent_window > 0 else 0.0
            
            # Calculate recent average episode length
            recent_avg_length = np.mean(episode_lengths[-recent_window:]) if len(episode_lengths) > 0 and recent_window > 0 else 0.0
            
            # Overall win rate so far
            overall_win_rate = wins / (episode + 1) if episode > 0 else 0.0
            
            # Overall average episode length so far
            overall_avg_length = np.mean(episode_lengths) if len(episode_lengths) > 0 else 0.0
            
            # Capture periodic stats at this interval
            periodic_stats_entry = {
                "episode": episode + 1,
                "epsilon": float(agent.epsilon),
                "replay_buffer_size": int(replay_buffer_size),
                "recent_win_rate": float(recent_win_rate),
                "recent_avg_reward": float(recent_avg_reward),
                "recent_avg_length": float(recent_avg_length),
                "recent_window_size": int(recent_window),
                "overall_win_rate": float(overall_win_rate),
                "overall_avg_length": float(overall_avg_length),
            }
            periodic_stats_history.append(periodic_stats_entry)
            
            # Use tqdm.write() to avoid interfering with progress bar
            tqdm.write(
                f"  Episode {episode+1}/{num_episodes}: "
                f"Replay buffer={replay_buffer_size}, "
                f"ε={agent.epsilon:.3f}, "
                f"Overall win rate={overall_win_rate:.2%}, "
                f"Recent win rate={recent_win_rate:.2%}, "
                f"Recent avg reward={recent_avg_reward:.1f}, "
                f"Recent avg length={recent_avg_length:.1f}, "
                f"Overall avg length={overall_avg_length:.1f}"
            )
        
        # Periodic checkpointing
        if checkpoint_frequency > 0 and (episode + 1) % checkpoint_frequency == 0:
            checkpoint_path = run_checkpoint_dir / f"checkpoint_ep{episode+1:06d}_seed{seed}.pth"
            agent.save(str(checkpoint_path))
            if verbose:
                tqdm.write(f"  💾 Checkpoint saved: {checkpoint_path.name}")
        
        # Best model tracking and saving (using rolling average of evaluations)
        # Only save best model if rolling average of last 3 evaluations beats previous best
        if save_best_model and len(eval_history) >= rolling_avg_window:
            # Calculate rolling average of last N evaluations
            recent_evals = eval_history[-rolling_avg_window:]
            rolling_avg_eval_win_rate = np.mean([e["eval_win_rate"] for e in recent_evals])
            
            if rolling_avg_eval_win_rate > best_rolling_avg_eval_win_rate:
                best_rolling_avg_eval_win_rate = rolling_avg_eval_win_rate
                # Save best model based on rolling average
                best_model_path = run_checkpoint_dir / f"best_model_ep{episode+1:06d}_rollingavg{best_rolling_avg_eval_win_rate:.3f}_seed{seed}.pth"
                agent.save(str(best_model_path))
                if verbose:
                    eval_win_rate_strs = [f"{e['eval_win_rate']:.2%}" for e in recent_evals]
                    tqdm.write(
                        f"  🏆 Best model saved (rolling avg eval win rate: {best_rolling_avg_eval_win_rate:.2%}, "
                        f"last {rolling_avg_window} evals: {eval_win_rate_strs}): "
                        f"{best_model_path.name}"
                    )
        # Periodic Greedy Evaluation (epsilon=0)
        if eval_frequency > 0 and (episode + 1) % eval_frequency == 0:
            eval_results = _run_greedy_evaluation(
                agent=agent,
                env=env,
                num_episodes=num_eval_episodes,
                seed=seed + 1000000 + episode,  # Use different seed for evaluation
                verbose=verbose,
            )
            eval_history.append({
                "episode": episode + 1,
                "eval_win_rate": eval_results["win_rate"],
                "eval_avg_reward": eval_results["avg_reward"],
                "eval_avg_length": eval_results["avg_episode_length"],
                "num_eval_episodes": num_eval_episodes,
            })
            # Calculate rolling average if we have enough evaluations
            rolling_avg_info = ""
            if len(eval_history) >= rolling_avg_window:
                recent_evals = eval_history[-rolling_avg_window:]
                rolling_avg = np.mean([e["eval_win_rate"] for e in recent_evals])
                rolling_avg_info = f", Rolling avg ({rolling_avg_window} evals): {rolling_avg:.2%}"
            
            if verbose:
                tqdm.write(
                    f"  📊 Greedy Eval (ε=0): "
                    f"Win rate={eval_results['win_rate']:.2%}, "
                    f"Avg reward={eval_results['avg_reward']:.1f}, "
                    f"Avg length={eval_results['avg_episode_length']:.1f}"
                    f"{rolling_avg_info}"
                )

    env.close()
    
    # Save final checkpoint
    final_checkpoint_path = run_checkpoint_dir / f"final_model_ep{num_episodes}_seed{seed}.pth"
    agent.save(str(final_checkpoint_path))
    if verbose:
        print(f"\n  💾 Final checkpoint saved: {final_checkpoint_path}")
    
    # Calculate final periodic stats (same as shown in logs)
    final_replay_buffer_size = len(agent.replay_buffer)
    final_epsilon = agent.epsilon
    
    # Calculate recent stats (last 10% of episodes, same as periodic logs)
    recent_window = max(10, num_episodes // 10)
    recent_wins = sum(debug_stats['win_by_episode'][-recent_window:]) if len(debug_stats['win_by_episode']) >= recent_window else sum(debug_stats['win_by_episode'])
    recent_win_rate = recent_wins / recent_window if recent_window > 0 and len(debug_stats['win_by_episode']) >= recent_window else (wins / num_episodes if num_episodes > 0 else 0.0)
    
    recent_avg_reward = float(np.mean(debug_stats['reward_by_episode'][-recent_window:])) if len(debug_stats['reward_by_episode']) >= recent_window else float(np.mean(rewards)) if len(rewards) > 0 else 0.0
    
    recent_avg_length = float(np.mean(episode_lengths[-recent_window:])) if len(episode_lengths) >= recent_window else float(np.mean(episode_lengths)) if len(episode_lengths) > 0 else 0.0
    
    # Overall stats
    overall_win_rate = wins / num_episodes if num_episodes > 0 else 0.0
    overall_avg_length = float(np.mean(episode_lengths)) if len(episode_lengths) > 0 else 0.0
    
    # Replay buffer trend
    replay_buffer_trend = None
    if debug_stats['replay_buffer_size']:
        replay_buffer_trend = {
            "initial": int(debug_stats['replay_buffer_size'][0]),
            "final": int(debug_stats['replay_buffer_size'][-1])
        }
    
    # Final debug summary (similar to dqn_selfplay.py)
    if verbose:
        print(f"\n  Debug Summary:")
        print(f"    Final replay buffer size: {final_replay_buffer_size}")
        print(f"    Final epsilon: {final_epsilon:.4f}")
        if replay_buffer_trend:
            print(f"    Replay buffer trend: {replay_buffer_trend['initial']} → {replay_buffer_trend['final']}")
        if best_model_path:
            print(f"    Best model (rolling avg eval win rate: {best_rolling_avg_eval_win_rate:.2%}): {best_model_path.name}")
        if eval_history:
            final_eval = eval_history[-1]
            print(f"    Final greedy eval (ε=0): Win rate={final_eval['eval_win_rate']:.2%}, Avg reward={final_eval['eval_avg_reward']:.1f}")
        if q_value_stats['mean_q_values']:
            q_summary = {
                "mean": np.mean(q_value_stats['mean_q_values']),
                "max": np.max(q_value_stats['max_q_values']),
                "min": np.min(q_value_stats['min_q_values']),
                "std": np.mean(q_value_stats['std_q_values']),
            }
            print(f"    Q-value stats: Mean={q_summary['mean']:.2f}, Max={q_summary['max']:.2f}, Min={q_summary['min']:.2f}, Std={q_summary['std']:.2f}")
        print(f"    Checkpoint directory: {run_checkpoint_dir}")

    stats = {
        "num_episodes": num_episodes,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / num_episodes if num_episodes > 0 else 0.0,
        "avg_episode_length": float(np.mean(episode_lengths)),
        "avg_reward": float(np.mean(rewards)),
        # Periodic stats history (captured at each log interval) - essential for tracking progress
        "periodic_stats_history": periodic_stats_history,
        # Evaluation history (greedy evaluation with epsilon=0)
        "eval_history": eval_history,
        # Q-value monitoring statistics
        "q_value_stats": {
            "mean_q_values": q_value_stats['mean_q_values'][-1000:] if len(q_value_stats['mean_q_values']) > 1000 else q_value_stats['mean_q_values'],  # Keep last 1000 samples
            "max_q_values": q_value_stats['max_q_values'][-1000:] if len(q_value_stats['max_q_values']) > 1000 else q_value_stats['max_q_values'],
            "min_q_values": q_value_stats['min_q_values'][-1000:] if len(q_value_stats['min_q_values']) > 1000 else q_value_stats['min_q_values'],
            "std_q_values": q_value_stats['std_q_values'][-1000:] if len(q_value_stats['std_q_values']) > 1000 else q_value_stats['std_q_values'],
            "summary": {
                "mean": float(np.mean(q_value_stats['mean_q_values'])) if len(q_value_stats['mean_q_values']) > 0 else None,
                "max": float(np.max(q_value_stats['max_q_values'])) if len(q_value_stats['max_q_values']) > 0 else None,
                "min": float(np.min(q_value_stats['min_q_values'])) if len(q_value_stats['min_q_values']) > 0 else None,
                "std": float(np.mean(q_value_stats['std_q_values'])) if len(q_value_stats['std_q_values']) > 0 else None,
            } if len(q_value_stats['mean_q_values']) > 0 else None,
        },
        # Final periodic stats (for convenience, same as last entry in history)
        "periodic_stats": {
            "final_epsilon": float(final_epsilon),
            "final_replay_buffer_size": int(final_replay_buffer_size),
            "recent_win_rate": float(recent_win_rate),
            "recent_avg_reward": float(recent_avg_reward),
            "recent_avg_length": float(recent_avg_length),
            "best_rolling_avg_eval_win_rate": float(best_rolling_avg_eval_win_rate) if best_model_path else None,
        },
        "config": {
            "num_players": num_players,
            "tokens_per_player": tokens_per_player,
            "seed": seed,
            "learning_rate": learning_rate,
            "discount_factor": discount_factor,
            "epsilon": epsilon,
            "min_epsilon": min_epsilon,
            "epsilon_schedule": epsilon_schedule,
            "epsilon_decay_fraction": epsilon_decay_fraction,
            "batch_size": batch_size,
            "replay_buffer_size": replay_buffer_size,
            "target_update_frequency": target_update_frequency,
            "train_frequency": train_frequency,
            "gradient_steps": gradient_steps,
            "hidden_dims": hidden_dims,
            "device": device,
            "best_model_path": str(best_model_path) if best_model_path else None,
            "final_checkpoint_path": str(final_checkpoint_path),
        },
    }

    return stats


def _convert_to_json_serializable(obj):
    """Recursively convert NumPy types and other non-serializable types to native Python types."""
    import numpy as np
    
    # Check for NumPy integer types (compatible with NumPy 1.x and 2.x)
    if isinstance(obj, np.integer):
        return int(obj)
    # Check for NumPy floating types (compatible with NumPy 1.x and 2.x)
    elif isinstance(obj, np.floating):
        return float(obj)
    # Check for NumPy boolean
    elif isinstance(obj, np.bool_):
        return bool(obj)
    # Check for NumPy arrays
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # Check for dictionaries
    elif isinstance(obj, dict):
        return {key: _convert_to_json_serializable(value) for key, value in obj.items()}
    # Check for lists and tuples
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_json_serializable(item) for item in obj]
    # Check for Path objects
    elif isinstance(obj, (Path,)):
        return str(obj)
    else:
        return obj


def save_results(results: dict, agent_name: str, base_seed: int):
    """Save experiment results to a JSON file."""
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_base = f"{agent_name}_seed{base_seed}_{timestamp}"
    
    filepath_json = results_dir / f"{filename_base}.json"
    data_to_save = {
        "agent_name": agent_name,
        "timestamp": timestamp,
        "base_seed": base_seed,
        "experiments": results,
        "summary": {},
    }

    for config_name, stats in results.items():
        data_to_save["summary"][config_name] = {
            "win_rate": stats["win_rate"],
            "recent_win_rate": stats.get("periodic_stats", {}).get("recent_win_rate", None),
            "avg_reward": stats["avg_reward"],
            "recent_avg_reward": stats.get("periodic_stats", {}).get("recent_avg_reward", None),
            "avg_episode_length": stats["avg_episode_length"],
            "best_rolling_avg_eval_win_rate": stats.get("periodic_stats", {}).get("best_rolling_avg_eval_win_rate", None),
        }
    
    # Convert all NumPy types to native Python types for JSON serialization
    data_to_save = _convert_to_json_serializable(data_to_save)

    with open(filepath_json, "w") as f:
        json.dump(data_to_save, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Results saved to: {filepath_json}")
    print(f"{'=' * 60}")

    return filepath_json


def run_quick_test(
    num_episodes: int = 1000,
    num_players: int = 4,
    tokens_per_player: int = 4,
    seed: int = 42,
) -> dict:
    """
    Quick test function for running a single Unified DQN experiment configuration.
    """
    print("=" * 60)
    print(f"Unified DQN Quick Test: {num_episodes} episodes ({tokens_per_player} tokens)")
    print("=" * 60)
    
    results = {}
    config_name = f"unified_dqn_{num_players}p{tokens_per_player}t_{num_episodes//1000}k"
    results[config_name] = run_experiment(
        num_episodes=num_episodes,
        num_players=num_players,
        tokens_per_player=tokens_per_player,
        seed=seed,
    )
    
    print("\n" + "=" * 60)
    print("QUICK TEST RESULTS")
    print("=" * 60)
    for config, stats in results.items():
        print(f"\n{config}:")
        print(f"  Win Rate: {stats['win_rate']:.2%}")
        print(f"  Wins: {stats['wins']}/{stats['num_episodes']}")
        print(f"  Avg Episode Length: {stats['avg_episode_length']:.1f}")
        print(f"  Avg Reward: {stats['avg_reward']:.3f}")
    
    save_results(results, "UnifiedDQNAgent_QuickTest", seed)
    return results


def main(
    num_episodes: int = 10000,
    num_players: int = 4,
    tokens_per_player: int = 4,
    seed: int = 42,
    device: str = None,
    learning_rate: float = 0.001,
    discount_factor: float = 0.95,
    epsilon: float = 1.0,
    epsilon_decay: float = 0.995,
    min_epsilon: float = 0.01,
    epsilon_schedule: str = 'exponential',  # 'linear', 'exponential', or 'adaptive'
    epsilon_decay_fraction: float = 0.9,  # Fraction of episodes to decay over
    batch_size: int = None,
    replay_buffer_size: int = 100000,
    target_update_frequency: int = 1000,
    train_frequency: int = None,
    gradient_steps: int = None,
    hidden_dims: list = None,
    checkpoint_frequency: int = 10000,  # Save checkpoint every N episodes (0 to disable)
    save_best_model: bool = True,  # Save best model based on recent win rate
    best_model_window: int = 1000,  # Episodes to consider for best model
    resume_from_checkpoint: str = None,  # Path to checkpoint to resume from
    eval_frequency: int = 1000,  # Run greedy evaluation every N episodes (0 to disable)
    num_eval_episodes: int = 10,  # Number of episodes to run during evaluation
):
    """Run Unified DQN experiments with specified configuration."""
    agent_name = "UnifiedDQNAgent"
    
    # Auto-detect device if not specified
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # GPU-optimized vs CPU-optimized hyperparameters (if not specified)
    if batch_size is None:
        batch_size = 256  # Default to 256 for faster training (works well for both CPU and GPU)
    
    if train_frequency is None:
        if device == 'cuda':
            train_frequency = 4
        else:
            train_frequency = 32  # Train less frequently for CPU (optimized for speed)
    
    if gradient_steps is None:
        gradient_steps = 1  # Single gradient step for speed (works well with large batch_size)
    
    if hidden_dims is None:
        hidden_dims = [128, 128, 64]  # Smaller network for faster training

    results = {}

    # Run experiment with specified configuration
    print("=" * 60)
    print(f"Unified DQN Experiment: {tokens_per_player} tokens, {num_players} players (Device: {device})")
    print("=" * 60)
    config_name = f"unified_dqn_{num_players}p{tokens_per_player}t_{num_episodes//1000}k"
    results[config_name] = run_experiment(
        num_episodes=num_episodes,
        num_players=num_players,
        tokens_per_player=tokens_per_player,
        seed=seed,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        min_epsilon=min_epsilon,
        epsilon_schedule=epsilon_schedule,
        epsilon_decay_fraction=epsilon_decay_fraction,
        batch_size=batch_size,
        replay_buffer_size=replay_buffer_size,
        target_update_frequency=target_update_frequency,
        train_frequency=train_frequency,
        gradient_steps=gradient_steps,
        hidden_dims=hidden_dims,
        device=device,
        checkpoint_frequency=checkpoint_frequency,
        save_best_model=save_best_model,
        best_model_window=best_model_window,
        resume_from_checkpoint=resume_from_checkpoint,
        eval_frequency=eval_frequency,
        num_eval_episodes=num_eval_episodes,
    )

    # Save results
    save_results(results, agent_name, seed)

    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    for config, stats in results.items():
        print(f"\n{config}:")
        print(f"  Win Rate: {stats['win_rate']:.2%}")
        print(f"  Wins: {stats['wins']}/{stats['num_episodes']}")
        print(f"  Losses: {stats['losses']}")
        print(f"  Avg Episode Length: {stats['avg_episode_length']:.1f}")
        print(f"  Avg Reward: {stats['avg_reward']:.3f}")
        if stats.get("periodic_stats", {}).get("recent_win_rate") is not None:
            print(f"  Recent Win Rate: {stats['periodic_stats']['recent_win_rate']:.2%}")
            print(f"  Recent Avg Reward: {stats['periodic_stats']['recent_avg_reward']:.1f}")
        if stats.get("periodic_stats", {}).get("best_rolling_avg_eval_win_rate") is not None:
            print(f"  Best Rolling Avg Eval Win Rate: {stats['periodic_stats']['best_rolling_avg_eval_win_rate']:.2%}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Unified DQN experiments")
    parser.add_argument("--quick", action="store_true", help="Run quick test (1000 episodes)")
    parser.add_argument("--episodes", type=int, default=10000, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tokens", type=int, default=4, choices=[2, 4], help="Tokens per player (2 or 4)")
    parser.add_argument("--players", type=int, default=4, choices=[2, 4], help="Number of players (2 or 4)")
    parser.add_argument("--device", type=str, default=None, choices=['cuda', 'cpu'], 
                       help="Device to use ('cuda' for GPU, 'cpu' for CPU, None for auto-detect)")
    
    # Hyperparameters (similar to dqn_selfplay.py)
    parser.add_argument("--alpha", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--epsilon_decay", type=float, default=0.995, help="Epsilon decay rate (not used with --epsilon_schedule)")
    parser.add_argument("--min_epsilon", type=float, default=0.01, help="Minimum epsilon")
    parser.add_argument("--epsilon_schedule", type=str, default='exponential', choices=['linear', 'exponential', 'adaptive'],
                       help="Epsilon decay schedule: 'linear' (simple), 'exponential' (recommended for long runs), 'adaptive' (performance-based)")
    parser.add_argument("--epsilon_decay_fraction", type=float, default=0.9,
                       help="Fraction of episodes to decay epsilon over (0.9 = 90%%, default: 0.9 for long runs)")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size (default: 256 for faster training)")
    parser.add_argument("--replay_buffer_size", type=int, default=10000, help="Replay buffer size")
    parser.add_argument("--target_update_frequency", type=int, default=100, help="Steps between target network updates")
    parser.add_argument("--train_frequency", type=int, default=None, help="Steps between training calls (None for auto)")
    parser.add_argument("--gradient_steps", type=int, default=None, help="Gradient steps per training call (None for auto)")
    parser.add_argument("--hidden_dims", type=str, default="128,128,64", 
                       help="Hidden layer dimensions (comma-separated, e.g., '256,256,128')")
    parser.add_argument("--checkpoint_frequency", type=int, default=10000,
                       help="Save checkpoint every N episodes (0 to disable, default: 10000)")
    parser.add_argument("--save_best_model", action="store_true", default=True,
                       help="Save best model based on recent win rate (default: True)")
    parser.add_argument("--no_save_best_model", dest="save_best_model", action="store_false",
                       help="Disable saving best model")
    parser.add_argument("--best_model_window", type=int, default=1000,
                       help="Episodes to consider for best model evaluation (default: 1000)")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Path to checkpoint file to resume training from")
    parser.add_argument("--eval_frequency", type=int, default=1000,
                       help="Run greedy evaluation every N episodes (0 to disable, default: 1000)")
    parser.add_argument("--num_eval_episodes", type=int, default=10,
                       help="Number of episodes to run during greedy evaluation (default: 10)")
    
    args = parser.parse_args()
    
    # Parse hidden_dims
    hidden_dims = [int(x.strip()) for x in args.hidden_dims.split(',')]
    
    if args.quick:
        results = {}
        config_name = f"unified_dqn_quick_test_{args.tokens}t"
        results[config_name] = run_experiment(
            num_episodes=1000,
            num_players=args.players,
            tokens_per_player=args.tokens,
            seed=args.seed,
            device=args.device,
            learning_rate=args.alpha,
            discount_factor=args.gamma,
            epsilon=args.epsilon,
            epsilon_decay=args.epsilon_decay,
            min_epsilon=args.min_epsilon,
            epsilon_schedule=args.epsilon_schedule,
            epsilon_decay_fraction=args.epsilon_decay_fraction,
            batch_size=args.batch_size,
            replay_buffer_size=args.replay_buffer_size,
            target_update_frequency=args.target_update_frequency,
            train_frequency=args.train_frequency,
            gradient_steps=args.gradient_steps,
            hidden_dims=hidden_dims,
            checkpoint_frequency=args.checkpoint_frequency,
            save_best_model=args.save_best_model,
            best_model_window=args.best_model_window,
            resume_from_checkpoint=args.resume_from_checkpoint,
            eval_frequency=args.eval_frequency,
            num_eval_episodes=args.num_eval_episodes,
        )
        save_results(results, "UnifiedDQNAgent_QuickTest", args.seed)
    else:
        main(
            num_episodes=args.episodes,
            num_players=args.players,
            tokens_per_player=args.tokens,
            seed=args.seed,
            device=args.device,
            learning_rate=args.alpha,
            discount_factor=args.gamma,
            epsilon=args.epsilon,
            epsilon_decay=args.epsilon_decay,
            min_epsilon=args.min_epsilon,
            epsilon_schedule=args.epsilon_schedule,
            epsilon_decay_fraction=args.epsilon_decay_fraction,
            batch_size=args.batch_size,
            replay_buffer_size=args.replay_buffer_size,
            target_update_frequency=args.target_update_frequency,
            train_frequency=args.train_frequency,
            gradient_steps=args.gradient_steps,
            hidden_dims=hidden_dims,
            checkpoint_frequency=args.checkpoint_frequency,
            save_best_model=args.save_best_model,
            best_model_window=args.best_model_window,
            resume_from_checkpoint=args.resume_from_checkpoint,
            eval_frequency=args.eval_frequency,
            num_eval_episodes=args.num_eval_episodes,
        )

