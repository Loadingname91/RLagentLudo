"""
Unified Dueling Double DQN Agent for Unified Ludo Environment.

Implements the "Egocentric Physics" approach with:
- Unified Feature Vector (46 floats for 4 tokens, 28 floats for 2 tokens)
- Action Masking (Logit Masking) for invalid moves
- Mathematically Sound Anti-Farming reward structure
- Dueling Double DQN architecture

Key Features:
1. Dueling Architecture: Separates state value V(s) and action advantages A(s,a)
2. Double DQN: Online network selects action, target network evaluates (reduces overestimation)
3. Action masking: Sets invalid action logits to -inf before argmax
4. Supports both 2-token and 4-token configurations
5. CPU-optimized architecture
"""

import random
import pickle
from typing import Dict, Tuple, Optional, List, Union
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

from rl_agent_ludo.utils.state import State
from rl_agent_ludo.agents.baseAgent import Agent


class UnifiedDuelingDQNNetwork(nn.Module):
    """
    Dueling Double DQN Network for Unified Ludo Environment.
    
    Architecture:
    - Input: Unified feature vector (28 floats for 2 tokens, 46 floats for 4 tokens)
    - Hidden layers: Shared feature extractor
    - Dueling heads:
      - Value stream: V(s) - 1 output (state value)
      - Advantage stream: A(s,a) - 4 outputs (action advantages)
    - Output: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
    
    Double DQN Logic:
    - Online network selects action: a* = argmax_a Q_online(s', a)
    - Target network evaluates: Q_target(s', a*)
    
    Optimized for CPU:
    - Layer normalization for stable training
    - ReLU activations (fast on CPU)
    - Xavier initialization for better convergence
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 256, 128], output_dim: int = 4):
        super(UnifiedDuelingDQNNetwork, self).__init__()
        
        # Shared feature extractor
        shared_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            shared_layers.append(nn.Linear(prev_dim, hidden_dim))
            shared_layers.append(nn.LayerNorm(hidden_dim))
            shared_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*shared_layers)
        
        # Dueling heads
        # Value stream: V(s) - estimates state value
        self.value_head = nn.Linear(prev_dim, 1)
        
        # Advantage stream: A(s,a) - estimates action advantages
        self.advantage_head = nn.Linear(prev_dim, output_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the dueling network with optional action masking.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            action_mask: Optional boolean mask of shape (batch_size, 4) where True = valid action
            
        Returns:
            Q-values tensor of shape (batch_size, 4)
            Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
            If action_mask is provided, invalid actions are set to -inf
        """
        # Shared feature extraction
        features = self.feature_extractor(x)  # Shape: (batch_size, last_hidden_dim)
        
        # Value stream: V(s)
        value = self.value_head(features)  # Shape: (batch_size, 1)
        
        # Advantage stream: A(s,a)
        advantage = self.advantage_head(features)  # Shape: (batch_size, 4)
        
        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        # This ensures that Q(s,a) = V(s) when advantage is zero
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        # Apply action masking: set invalid actions to -inf
        if action_mask is not None:
            # Invert mask: True (valid) -> 0, False (invalid) -> -inf
            mask_tensor = (~action_mask).float() * float('-inf')
            q_values = q_values + mask_tensor
        
        return q_values


class UnifiedDQNAgent(Agent):
    """
    Unified Dueling Double DQN agent for Ludo with Egocentric Physics approach.
    
    Features:
    - Dueling Double DQN architecture (V(s) + A(s,a))
    - Double DQN logic: Online selects, target evaluates (reduces overestimation)
    - Direct observation input (unified feature vector)
    - Action masking (logit masking) for invalid moves
    - Experience replay buffer
    - Target network for stable Q-learning
    - CPU-optimized batch processing
    - Works with UnifiedLudoEnv2Tokens and UnifiedLudoEnv4Tokens
    
    State Representation:
    - 28 floats for 2 tokens: Global Context (10) + Token 0 (9) + Token 1 (9)
    - 46 floats for 4 tokens: Global Context (10) + Token 0-3 (9 each)
    
    Action Masking:
    - Uses action_mask from environment info dict
    - Sets invalid action logits to -inf before argmax
    - Agent never selects invalid actions
    """
    
    def __init__(
        self,
        input_dim: int,  # 28 for 2 tokens, 46 for 4 tokens
        learning_rate: float = 0.001,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01,
        batch_size: int = 128,
        replay_buffer_size: int = 10000,
        target_update_frequency: int = 100,
        train_frequency: int = 32,
        gradient_steps: int = 1,
        hidden_dims: List[int] = [128, 128, 64],
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize Unified DQN agent.
        
        Args:
            input_dim: Input dimension (28 for 2 tokens, 46 for 4 tokens)
            learning_rate: Learning rate for optimizer
            discount_factor: Discount factor (gamma)
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay per episode
            min_epsilon: Minimum epsilon value
            batch_size: Batch size for training
            replay_buffer_size: Size of experience replay buffer
            target_update_frequency: Steps between target network updates
            train_frequency: Steps between training calls
            gradient_steps: Number of gradient steps per training call
            hidden_dims: Hidden layer dimensions
            device: Device to use ('cpu', 'cuda', or None for auto-detect)
            seed: Random seed
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Device setup (force CPU for optimization)
        if device is None:
            device = 'cpu'  # Force CPU for this implementation
        self.device = torch.device(device)
        
        # Set PyTorch CPU optimizations
        if self.device.type == 'cpu':
            # Use fewer threads for better performance (4 threads is optimal for CPU)
            # Reduced from 8 for better efficiency (less thread overhead)
            torch.set_num_threads(4)
            # Enable MKLDNN for better CPU performance
            torch.backends.mkldnn.enabled = True
            # Disable deterministic mode for speed
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = False  # Not applicable to CPU
        
        # Hyperparameters
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.replay_buffer_size = replay_buffer_size
        self.target_update_frequency = target_update_frequency
        self.train_frequency = train_frequency
        self.gradient_steps = gradient_steps
        self.input_dim = input_dim
        
        # Neural networks (Dueling Double DQN)
        self.q_network = UnifiedDuelingDQNNetwork(input_dim, hidden_dims, output_dim=4).to(self.device)
        self.target_network = UnifiedDuelingDQNNetwork(input_dim, hidden_dims, output_dim=4).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network is always in eval mode
        
        # Optimizer (Adam is good for CPU)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay buffer
        # Store: (obs, action, reward, next_obs, done, action_mask, next_action_mask)
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        
        # Training counters
        self.step_count = 0
        self.episode_count = 0
        
        # Last observation/action tracking (for replay buffer)
        self._last_obs: Optional[np.ndarray] = None
        self._last_action: Optional[int] = None
        self._last_action_mask: Optional[np.ndarray] = None
    
    @property
    def is_on_policy(self) -> bool:
        """DQN is off-policy."""
        return False
    
    @property
    def needs_replay_learning(self) -> bool:
        """DQN uses experience replay."""
        return True
    
    def act(self, state: State, obs: Optional[np.ndarray] = None, action_mask: Optional[np.ndarray] = None) -> int:
        """
        Epsilon-greedy action selection with action masking.
        
        Args:
            state: Current State object (for compatibility with Agent interface)
            obs: Optional observation array (unified feature vector)
            action_mask: Optional action mask from environment info dict
            
        Returns:
            Action index (0-3, corresponding to token index)
        """
        # Use provided obs or extract from state if needed
        if obs is None:
            # If obs not provided, we need it from somewhere
            # For now, raise error to ensure obs is provided
            raise ValueError("obs must be provided for UnifiedDQNAgent")
        
        if action_mask is None:
            # If no mask provided, assume all actions are valid (shouldn't happen in practice)
            action_mask = np.ones(4, dtype=bool)
        
        # Filter valid actions
        valid_actions = [i for i in range(4) if action_mask[i]]
        
        if not valid_actions:
            # No valid actions, return 0 as fallback
            return 0
        
        # Exploration
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        # Exploitation: use Q-network with action masking
        # Optimized: use torch.from_numpy directly (faster than torch.tensor for numpy arrays)
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)  # Shape: (1, input_dim)
        action_mask_tensor = torch.from_numpy(action_mask).bool().unsqueeze(0).to(self.device)  # Shape: (1, 4)
        
        with torch.no_grad():
            q_values = self.q_network(obs_tensor, action_mask_tensor)[0].cpu().numpy()  # Shape: (4,) - removed extra indexing
        
        # Select best valid action (masking already applied in network)
        best_action = np.argmax(q_values)
        
        # Verify action is valid (should always be true due to masking)
        if not action_mask[best_action]:
            # Fallback: use first valid action
            best_action = valid_actions[0]
        
        # Store for replay buffer
        self._last_obs = obs.copy()
        self._last_action = best_action
        self._last_action_mask = action_mask.copy()
        
        return best_action
    
    def push_to_replay_buffer(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
        done: bool,
        obs: Optional[np.ndarray] = None,
        next_obs: Optional[np.ndarray] = None,
        action_mask: Optional[np.ndarray] = None,
        next_action_mask: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        """
        Store experience in replay buffer.
        
        Args:
            state: Previous state (for compatibility)
            action: Action taken
            reward: Reward received
            next_state: Next state (for compatibility)
            done: Whether episode ended
            obs: Previous observation (unified feature vector)
            next_obs: Next observation (unified feature vector)
            action_mask: Previous action mask
            next_action_mask: Next action mask
        """
        # Use stored obs/action_mask if available (from act), otherwise use provided
        if self._last_obs is not None:
            obs = self._last_obs
            action_mask = self._last_action_mask
        
        if obs is None or next_obs is None:
            # If observations not provided, we can't store experience
            # This shouldn't happen in practice, but handle gracefully
            return
        
        # Default masks if not provided
        if action_mask is None:
            action_mask = np.ones(4, dtype=bool)
        if next_action_mask is None:
            next_action_mask = np.ones(4, dtype=bool)
        
        # Store experience
        self.replay_buffer.append((
            obs.copy(),
            action,
            reward,
            next_obs.copy(),
            done,
            action_mask.copy(),
            next_action_mask.copy(),
        ))
        
        self.step_count += 1
        
        # Train periodically
        if len(self.replay_buffer) >= self.batch_size and self.step_count % self.train_frequency == 0:
            self.learn_from_replay()
        
        # Update target network periodically
        if self.step_count % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Reset tracking
        if done:
            self._last_obs = None
            self._last_action = None
            self._last_action_mask = None
            self.on_episode_end()
    
    def learn_from_replay(self, *args, **kwargs) -> None:
        """
        Learn from experience replay buffer with action masking.
        
        Optimized for speed with batched tensor operations and action masking.
        """
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Perform gradient steps
        for _ in range(self.gradient_steps):
            # Sample batch
            batch = random.sample(self.replay_buffer, self.batch_size)
            
            # Unpack batch
            obs_batch, actions, rewards, next_obs_batch, dones, action_masks, next_action_masks = zip(*batch)
            
            # Convert to tensors (optimized: use torch.from_numpy for numpy arrays, faster than torch.tensor)
            obs_array = np.array(obs_batch, dtype=np.float32)
            next_obs_array = np.array(next_obs_batch, dtype=np.float32)
            action_masks_array = np.array(action_masks, dtype=bool)
            next_action_masks_array = np.array(next_action_masks, dtype=bool)
            
            obs_tensors = torch.from_numpy(obs_array).to(self.device)  # Shape: (batch_size, input_dim)
            next_obs_tensors = torch.from_numpy(next_obs_array).to(self.device)  # Shape: (batch_size, input_dim)
            action_masks_tensors = torch.from_numpy(action_masks_array).to(self.device)  # Shape: (batch_size, 4)
            next_action_masks_tensors = torch.from_numpy(next_action_masks_array).to(self.device)  # Shape: (batch_size, 4)
            
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            dones_tensor = torch.tensor(dones, dtype=torch.bool, device=self.device)
            actions_tensor = torch.tensor(actions, dtype=torch.int64, device=self.device)
            
            # Get Q-values for current states
            q_values = self.q_network(obs_tensors, action_masks_tensors)  # Shape: (batch_size, 4)
            
            # Get Q-values for taken actions
            q_values_selected = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
            
            # Double DQN: Online network selects action, Target network evaluates
            with torch.no_grad():
                # Online network selects best action for next state
                next_q_values_online = self.q_network(next_obs_tensors, next_action_masks_tensors)  # Shape: (batch_size, 4)
                next_actions = next_q_values_online.argmax(1)  # Shape: (batch_size,)
                
                # Target network evaluates the selected action
                next_q_values_target = self.target_network(next_obs_tensors, next_action_masks_tensors)  # Shape: (batch_size, 4)
                next_q_values_selected = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)  # Shape: (batch_size,)
                
                # Compute target Q-values
                target_q_values = rewards_tensor + (self.gamma * next_q_values_selected * ~dones_tensor)
            
            # Compute loss
            loss = F.mse_loss(q_values_selected, target_q_values)
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
            self.optimizer.step()
    
    def learn_from_rollout(self, *args, **kwargs) -> None:
        """DQN doesn't use rollout learning."""
        pass
    
    def on_episode_end(self) -> None:
        """Update epsilon at end of episode."""
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)
        self.episode_count += 1
    
    def save(self, filepath: str) -> None:
        """Save agent state to file."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'input_dim': self.input_dim,
            'replay_buffer': list(self.replay_buffer),  # Convert deque to list
        }, filepath)
    
    def load(self, filepath: str) -> None:
        """Load agent state from file."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.step_count = checkpoint.get('step_count', 0)
        self.episode_count = checkpoint.get('episode_count', 0)
        self.input_dim = checkpoint.get('input_dim', self.input_dim)
        if 'replay_buffer' in checkpoint:
            saved_buffer = checkpoint['replay_buffer']
            saved_buffer_size = len(saved_buffer) if saved_buffer else 0
            self.replay_buffer = deque(saved_buffer, maxlen=self.replay_buffer_size)
            # Log replay buffer restoration (can be accessed via return or logging)
            self._loaded_replay_buffer_size = saved_buffer_size
        else:
            # No replay buffer in checkpoint - this is an old checkpoint format
            self._loaded_replay_buffer_size = 0


# Convenience factory functions
def create_unified_dqn_agent_2tokens(
    learning_rate: float = 0.001,
    discount_factor: float = 0.95,
    epsilon: float = 1.0,
    epsilon_decay: float = 0.995,
    min_epsilon: float = 0.01,
    batch_size: int = 128,
    replay_buffer_size: int = 10000,
    target_update_frequency: int = 100,
    train_frequency: int = 32,
    gradient_steps: int = 1,
    hidden_dims: List[int] = [128, 128, 64],
    device: Optional[str] = None,
    seed: Optional[int] = None,
) -> UnifiedDQNAgent:
    """
    Create Unified DQN agent for 2-token configuration.
    
    Args:
        All hyperparameters same as UnifiedDQNAgent.__init__
        
    Returns:
        UnifiedDQNAgent with input_dim=28
    """
    return UnifiedDQNAgent(
        input_dim=28,  # 10 (global) + 9*2 (tokens)
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
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


def create_unified_dqn_agent_4tokens(
    learning_rate: float = 0.001,
    discount_factor: float = 0.95,
    epsilon: float = 1.0,
    epsilon_decay: float = 0.995,
    min_epsilon: float = 0.01,
    batch_size: int = 128,
    replay_buffer_size: int = 10000,
    target_update_frequency: int = 100,
    train_frequency: int = 32,
    gradient_steps: int = 1,
    hidden_dims: List[int] = [128, 128, 64],
    device: Optional[str] = None,
    seed: Optional[int] = None,
) -> UnifiedDQNAgent:
    """
    Create Unified DQN agent for 4-token configuration.
    
    Args:
        All hyperparameters same as UnifiedDQNAgent.__init__
        
    Returns:
        UnifiedDQNAgent with input_dim=46
    """
    return UnifiedDQNAgent(
        input_dim=46,  # 10 (global) + 9*4 (tokens)
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
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

