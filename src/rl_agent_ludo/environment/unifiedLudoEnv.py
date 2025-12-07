"""
Unified Ludo Gymnasium Environment with Potential-Based Reward Shaping (PBRS).

Implements the "Egocentric Physics" approach with:
- Unified Feature Vector (46 floats for 4 tokens, 28 floats for 2 tokens)
- Potential-Based Reward Shaping (PBRS) to eliminate farming cycles
- Action Masking support

Two separate environment classes:
- UnifiedLudoEnv2Tokens: For 2 tokens per player (28 floats)
- UnifiedLudoEnv4Tokens: For 4 tokens per player (46 floats)

Both support 2 players and 4 players.

Reference: Ng, A. Y., Harada, D., & Russell, S. (1999). Policy invariance under reward transformations.
"""
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces

from rl_agent_ludo.utils.state import State
from rl_agent_ludo.ludo.game import Game as LudopyGame

# --- Board Constants ---
HOME_INDEX = 0
GOAL_INDEX = 57
GLOBE_INDEXES = [1, 9, 22, 35, 48]
STAR_INDEXES = [5, 12, 18, 25, 31, 38, 44, 51]
HOME_CORRIDOR = list(range(52, 57))

# Maximum possible score for normalization
MAX_SCORE_2_TOKENS = 200.0  # 2 tokens * 100 (goal) = 200
MAX_SCORE_4_TOKENS = 400.0  # 4 tokens * 100 (goal) = 400


class _UnifiedLudoEnvBase(gym.Env):
    """
    Unified Ludo Environment with Potential-Based Reward Shaping (PBRS).
    
    State Representation:
    - Global Context (10 floats): Dice One-Hot (6) + Normalized Scores (4)
    - Per-Token Features (9 floats × N tokens): Zone Identity (5) + Progress (1) + Threat_Behind (1) + Target_Ahead (1) + Action_Legal (1)
    
    Total: 10 + (9 × N) floats where N = tokens_per_player
    
    Reward Function (PBRS):
    - Sparse Rewards: Win (+100.0), Loss (-10.0)
    - Semantic Rewards: Kill (+10.0), Goal Token (+20.0)
    - Potential-Based Shaping: F = γ * Φ(s') - Φ(s)
      where Φ(s) = W_PROG * Progress + W_SAFE * Safety
    
    PBRS eliminates farming cycles by design (Ng et al., 1999).
    The potential function provides dense guidance while preserving optimal policy.
    
    Reference: Ng, A. Y., Harada, D., & Russell, S. (1999). Policy invariance under reward transformations.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        player_id: int = 0,
        num_players: int = 4,
        tokens_per_player: int = 4,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        max_score: float = 400.0,
        gamma: float = 0.99,  # Discount factor for PBRS calculation
    ) -> None:
        super().__init__()

        assert num_players in (2, 3, 4), "num_players must be 2, 3, or 4"
        assert tokens_per_player in (2, 4), "tokens_per_player must be 2 or 4"
        assert 0 <= player_id < num_players, "player_id must be < num_players"

        self.player_id = player_id
        self.num_players = num_players
        self.tokens_per_player = tokens_per_player
        self.render_mode = render_mode
        self._seed = seed
        self.max_score = max_score
        self.gamma = gamma  # Discount factor for PBRS calculation

        # Action space: Discrete(4) for token selection
        self.action_space = spaces.Discrete(4)

        # Observation space: 10 (global) + 9 * tokens_per_player
        obs_dim = 10 + (9 * tokens_per_player)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # Game state
        self.game: Optional[LudopyGame] = None
        self.current_player: int = 0
        self.current_dice: int = 1
        self._last_raw_obs: Optional[Tuple] = None
        self._prev_state: Optional[State] = None
        # Note: We don't store _prev_progress anymore - we calculate potential statelessly

    def _set_seed(self, seed: Optional[int]) -> None:
        """Set random seed."""
        if seed is None:
            return
        np.random.seed(seed)
        random.seed(seed)

    def _ghost_players_for_num_players(self) -> List[int]:
        """Get ghost players list based on num_players."""
        if self.num_players == 4:
            return []
        if self.num_players == 3:
            return [3]
        if self.num_players == 2:
            return [2, 3]
        return []

    def _apply_token_limit(self, pieces: List[int]) -> List[int]:
        """Apply token limit (set unused tokens to 0)."""
        if self.tokens_per_player == 4:
            return pieces
        tokens_on_board = list(pieces)
        for i in range(2, len(tokens_on_board)):
            tokens_on_board[i] = 0
        return tokens_on_board

    def _build_state_from_obs(self, obs: Tuple) -> State:
        """Build State object from raw observation."""
        dice, move_pieces, player_pieces, enemy_pieces, player_is_winner, there_is_winner = obs

        player_pieces = self._apply_token_limit(list(player_pieces))
        enemy_pieces = [self._apply_token_limit(list(ep)) for ep in enemy_pieces]
        move_pieces = [p for p in move_pieces if p < self.tokens_per_player]

        # Use absolute indexing - valid_moves are token indices that can move
        if len(move_pieces) > 0:
            valid_moves = list(move_pieces)
            movable_pieces = list(move_pieces)
        else:
            valid_moves = []
            movable_pieces = []

        return State(
            player_pieces=player_pieces,
            enemy_pieces=enemy_pieces,
            current_player=self.current_player,
            dice_roll=int(dice),
            valid_moves=valid_moves,
            movable_pieces=movable_pieces,
        )

    # ---------- Unified State Representation ----------

    def _state_to_obs(self, state: State) -> np.ndarray:
        """
        Build unified feature vector.
        
        Returns:
            np.ndarray: [Global Context (10), Token 0 (9), ..., Token N-1 (9)]
            where N = tokens_per_player
        """
        # Global Context (10 floats)
        global_features = self._build_global_context(state)
        
        # Per-Token Features (9 floats × tokens_per_player)
        token_features = []
        for token_idx in range(self.tokens_per_player):
            token_feat = self._build_token_features(state, token_idx)
            token_features.append(token_feat)
        
        # Concatenate all features
        obs_vec = np.concatenate([global_features] + token_features, dtype=np.float32)
        return obs_vec

    def _build_global_context(self, state: State) -> np.ndarray:
        """
        Build global context features (10 floats).
        
        Returns:
            [Dice One-Hot (6), Normalized Scores (4)]
        """
        features = np.zeros(10, dtype=np.float32)
        
        # Dice One-Hot (indices 0-5)
        dice_idx = min(state.dice_roll - 1, 5)  # 1-6 -> 0-5
        features[dice_idx] = 1.0
        
        # Normalized Scores (indices 6-9)
        # Score for our player
        our_score = self._get_weighted_equity_score(state.player_pieces)
        features[6] = min(our_score / self.max_score, 1.0)
        
        # Scores for enemies (up to 3 enemies)
        for i, enemy_pieces in enumerate(state.enemy_pieces):
            if i < 3:  # Max 3 enemies
                enemy_score = self._get_weighted_equity_score(enemy_pieces)
                features[7 + i] = min(enemy_score / self.max_score, 1.0)
        
        return features

    def _build_token_features(self, state: State, token_idx: int) -> np.ndarray:
        """
        Build per-token features (9 floats).
        
        Returns:
            [Zone Identity One-Hot (5), Normalized Progress (1), Threat_Behind (1), Target_Ahead (1), Action_Legal (1)]
        """
        features = np.zeros(9, dtype=np.float32)
        pos = state.player_pieces[token_idx]
        
        # 1. Zone Identity One-Hot (indices 0-4)
        zone_idx = self._get_zone_identity(pos)
        features[zone_idx] = 1.0
        
        # 2. Normalized Progress (index 5)
        features[5] = self._get_normalized_progress(pos)
        
        # 3. Threat_Behind (index 6)
        features[6] = self._get_threat_behind(pos, state.enemy_pieces)
        
        # 4. Target_Ahead (index 7)
        features[7] = self._get_target_ahead(pos, state.enemy_pieces, state.dice_roll)
        
        # 5. Action_Legal (index 8)
        features[8] = 1.0 if token_idx in state.valid_moves else 0.0
        
        return features

    def _get_zone_identity(self, pos: int) -> int:
        """
        Get zone identity index.
        
        Returns:
            0: Is_Home
            1: Is_Safe_Globe
            2: Is_Safe_Star
            3: Is_Victory_Path (HOME_CORRIDOR)
            4: Is_Normal_Path
        """
        if pos == HOME_INDEX:
            return 0  # Is_Home
        elif pos in GLOBE_INDEXES:
            return 1  # Is_Safe_Globe
        elif pos in STAR_INDEXES:
            return 2  # Is_Safe_Star
        elif pos in HOME_CORRIDOR:
            return 3  # Is_Victory_Path
        else:
            return 4  # Is_Normal_Path

    def _get_normalized_progress(self, pos: int) -> float:
        """Get normalized progress [0, 1]."""
        if pos == HOME_INDEX:
            return 0.0
        elif pos == GOAL_INDEX:
            return 1.0
        else:
            return pos / 57.0

    def _get_threat_behind(self, token_pos: int, enemy_pieces: List[List[int]]) -> float:
        """
        Calculate threat from behind.
        
        Returns:
            [0, 1] where 1.0 = enemy 1 step behind, 0.0 = no threat
        """
        if token_pos in GLOBE_INDEXES or token_pos in STAR_INDEXES:
            return 0.0  # Safe zones
        if token_pos == HOME_INDEX or token_pos == GOAL_INDEX or token_pos > 51:
            return 0.0  # Home, goal, or victory path
        
        min_threat = 0.0
        for enemy in enemy_pieces:
            for e_pos in enemy:
                if e_pos == HOME_INDEX or e_pos == GOAL_INDEX:
                    continue
                # Calculate distance (enemy behind us)
                dist = (token_pos - e_pos) % 52
                # If enemy is 1-6 steps behind
                if 1 <= (-dist % 52) <= 6:
                    threat_magnitude = (7 - (-dist % 52)) / 7.0
                    min_threat = max(min_threat, threat_magnitude)
        
        return min_threat

    def _get_target_ahead(self, token_pos: int, enemy_pieces: List[List[int]], dice_roll: int) -> float:
        """
        Calculate target opportunity ahead.
        
        Returns:
            [0, 1] where 1.0 = enemy 1 step ahead, 0.0 = no target
        """
        if token_pos == HOME_INDEX or token_pos == GOAL_INDEX:
            return 0.0
        
        # Simulate next position
        next_pos = self._simulate_move(token_pos, dice_roll)
        if next_pos == HOME_INDEX or next_pos == GOAL_INDEX:
            return 0.0
        if next_pos in GLOBE_INDEXES or next_pos in STAR_INDEXES:
            return 0.0  # Can't capture on safe zones
        
        min_target = 0.0
        for enemy in enemy_pieces:
            for e_pos in enemy:
                if e_pos == HOME_INDEX or e_pos == GOAL_INDEX:
                    continue
                # Calculate distance (enemy ahead of us)
                dist = (e_pos - next_pos) % 52
                # If enemy is 1-6 steps ahead
                if 1 <= dist <= 6:
                    target_magnitude = (7 - dist) / 7.0
                    min_target = max(min_target, target_magnitude)
        
        return min_target

    def _simulate_move(self, current_pos: int, dice_roll: int) -> int:
        """Simulate a move from current position."""
        if current_pos == HOME_INDEX:
            if dice_roll == 6:
                return 1
            return HOME_INDEX
        if current_pos == GOAL_INDEX:
            return GOAL_INDEX
        
        next_pos = current_pos + dice_roll
        # Star jump logic
        if next_pos in STAR_INDEXES:
            idx = STAR_INDEXES.index(next_pos)
            if idx < len(STAR_INDEXES) - 1:
                next_pos = STAR_INDEXES[idx + 1]
            else:
                next_pos = GOAL_INDEX
        
        if next_pos > GOAL_INDEX:
            return GOAL_INDEX
        return next_pos

    def _get_weighted_equity_score(self, pieces: List[int]) -> float:
        """Calculate weighted equity score for normalization."""
        score = 0.0
        for pos in pieces:
            if pos == GOAL_INDEX:
                score += 100
            elif pos in HOME_CORRIDOR:
                score += 50 + pos
            elif pos == HOME_INDEX:
                score += 0
            else:
                score += pos
        return score

    # ---------- PBRS Reward Function ----------
    
    # Sparse Rewards (The "True" Objective)
    WIN_REWARD = 100.0  # Standardized to 100
    LOSS_PENALTY = -10.0  # Minor penalty for losing to encourage shorter games if losing
    
    # Semantic Rewards (Events that are unambiguously good/bad)
    KILL_REWARD = 10.0  # Killing is always good (tempo swing)
    GOAL_TOKEN_REWARD = 20.0  # Banking a token is always good
    
    # PBRS Weights
    # Potential = W_PROG * Progress + W_SAFE * Safety
    # Scale: Progress (0-1) * 100 ~ 100. Safety (0-1) * 20 ~ 20.
    W_PROGRESS = 50.0  # High weight to drive forward motion
    W_SAFE = 10.0  # Moderate weight to prefer safe spots

    def _calculate_potential(self, state: State) -> float:
        """
        Calculate the Potential Phi(s) of a state.
        
        Phi(s) = W_PROG * Sum(Progress) + W_SAFE * Sum(IsSafe)
        
        This is computed statelessly from the state object to ensure robustness.
        """
        # 1. Progress Potential
        total_progress = 0.0
        for i in range(self.tokens_per_player):
            pos = state.player_pieces[i]
            # normalized progress is 0.0 to 1.0
            total_progress += self._get_normalized_progress(pos)
        
        progress_potential = total_progress * self.W_PROGRESS
        
        # 2. Safety Potential
        # Count tokens on globes or stars
        safe_count = 0
        for i in range(self.tokens_per_player):
            pos = state.player_pieces[i]
            if pos in GLOBE_INDEXES or pos in STAR_INDEXES:
                safe_count += 1
        
        safety_potential = safe_count * self.W_SAFE
        
        return progress_potential + safety_potential

    def _compute_reward_and_done(
        self, prev_state: Optional[State], current_state: State, action: int
    ) -> Tuple[float, bool]:
        """
        Compute PBRS reward + Sparse Rewards.
        
        Reward = R_sparse + (gamma * Phi(s') - Phi(s))
        
        This implementation is stateless - potentials are calculated on the fly
        from state objects to ensure mathematical consistency.
        """
        if self.game is None:
            return 0.0, False
        
        winners = self.game.get_winners_of_game()
        done = len(winners) > 0
        
        # 1. Terminal Rewards (Sparse)
        if done:
            if self.player_id in winners:
                return self.WIN_REWARD, True
            else:
                return self.LOSS_PENALTY, True
        
        # Skip if not our turn
        if prev_state is None or prev_state.current_player != self.player_id:
            return 0.0, False
        
        reward = 0.0
        
        # 2. Semantic Events (Kill & Goal)
        prev_pos = prev_state.player_pieces[action]
        curr_pos = current_state.player_pieces[action]
        
        if prev_pos != curr_pos:
            # Kill Check
            captured = False
            for enemy in prev_state.enemy_pieces:
                for e_pos in enemy:
                    if e_pos == curr_pos and e_pos not in [HOME_INDEX, GOAL_INDEX, 1]:
                        if e_pos not in GLOBE_INDEXES:
                            reward += self.KILL_REWARD
                            captured = True
                            break
                if captured:
                    break
            
            # Goal Check
            if curr_pos == GOAL_INDEX:
                reward += self.GOAL_TOKEN_REWARD
        
        # 3. Potential-Based Reward Shaping (The Anti-Farming Logic)
        # F = gamma * Phi(s') - Phi(s)
        # We calculate potentials dynamically here using the state objects
        current_potential = self._calculate_potential(current_state)
        prev_potential = self._calculate_potential(prev_state)
        
        shaping_reward = (self.gamma * current_potential) - prev_potential
        reward += shaping_reward
        
        return reward, False


    # ---------- Action Masking ----------

    def _get_action_mask(self, state: State) -> np.ndarray:
        """
        Get action mask for invalid moves.
        
        Returns:
            np.ndarray: Boolean mask where True = valid action, False = invalid
        """
        mask = np.zeros(4, dtype=bool)
        for i in range(4):
            if i < self.tokens_per_player and i in state.valid_moves:
                mask[i] = True
        return mask

    # ---------- Gym API ----------

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment."""
        self._set_seed(seed if seed is not None else self._seed)
        ghost_players = self._ghost_players_for_num_players()
        self.game = LudopyGame(ghost_players=ghost_players)
        self.game.reset()

        raw_obs, self.current_player = self.game.get_observation()
        self._last_raw_obs = raw_obs
        dice, _, _, _, _, _ = raw_obs
        self.current_dice = int(dice)

        state = self._build_state_from_obs(raw_obs)
        obs_vec = self._state_to_obs(state)
        
        # Action mask
        action_mask = self._get_action_mask(state)
        
        info = {
            "raw_obs": raw_obs,
            "state": state,
            "action_mask": action_mask,
        }
        self._prev_state = state
        return obs_vec, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step environment."""
        if self.game is None:
            raise RuntimeError("Reset first")
        
        prev_state = self._prev_state

        if self._last_raw_obs is None:
            raise RuntimeError("No observation available")
        
        _, move_pieces, _, _, _, _ = self._last_raw_obs

        if isinstance(move_pieces, np.ndarray):
            move_pieces = move_pieces.tolist()
        valid_pieces = [p for p in move_pieces if p < self.tokens_per_player]

        if not valid_pieces:
            piece_to_move = move_pieces[0] if move_pieces else -1
        else:
            if action in move_pieces and action in valid_pieces:
                piece_to_move = action
            else:
                piece_to_move = valid_pieces[0] if valid_pieces else (move_pieces[0] if move_pieces else -1)

        if piece_to_move != -1 and piece_to_move not in move_pieces:
            piece_to_move = move_pieces[0] if move_pieces else -1

        _ = self.game.answer_observation(piece_to_move)

        raw_obs, self.current_player = self.game.get_observation()
        self._last_raw_obs = raw_obs
        dice, _, _, _, _, _ = raw_obs
        self.current_dice = int(dice)

        next_state = self._build_state_from_obs(raw_obs)
        obs_vec = self._state_to_obs(next_state)

        # Compute PBRS reward - Passing both states explicitly
        reward, terminated = self._compute_reward_and_done(prev_state, next_state, action)
        
        # Action mask
        action_mask = self._get_action_mask(next_state)

        self._prev_state = next_state
        info = {
            "raw_obs": raw_obs,
            "state": next_state,
            "action_mask": action_mask,
        }
        return obs_vec, reward, terminated, False, info

    def render(self, mode="human"):
        """Render environment."""
        if self.game:
            return self.game.render_environment()
        return None

    def close(self):
        """Close environment."""
        pass


class UnifiedLudoEnv2Tokens(_UnifiedLudoEnvBase):
    """
    Unified Ludo Environment for 2 tokens per player with PBRS.
    
    Observation Space: 28 floats
    - Global Context (10): Dice One-Hot (6) + Normalized Scores (4)
    - Token 0 (9): Zone Identity (5) + Progress (1) + Threat_Behind (1) + Target_Ahead (1) + Action_Legal (1)
    - Token 1 (9): Same as Token 0
    
    Supports 2 players and 4 players.
    """
    
    def __init__(
        self,
        player_id: int = 0,
        num_players: int = 4,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        gamma: float = 0.99,  # Discount factor for PBRS
    ) -> None:
        super().__init__(
            player_id=player_id,
            num_players=num_players,
            tokens_per_player=2,
            render_mode=render_mode,
            seed=seed,
            max_score=MAX_SCORE_2_TOKENS,
            gamma=gamma,
        )


class UnifiedLudoEnv4Tokens(_UnifiedLudoEnvBase):
    """
    Unified Ludo Environment for 4 tokens per player with PBRS.
    
    Observation Space: 46 floats
    - Global Context (10): Dice One-Hot (6) + Normalized Scores (4)
    - Token 0-3 (9 each): Zone Identity (5) + Progress (1) + Threat_Behind (1) + Target_Ahead (1) + Action_Legal (1)
    
    Supports 2 players and 4 players.
    """
    
    def __init__(
        self,
        player_id: int = 0,
        num_players: int = 4,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        gamma: float = 0.99,  # Discount factor for PBRS
    ) -> None:
        super().__init__(
            player_id=player_id,
            num_players=num_players,
            tokens_per_player=4,
            render_mode=render_mode,
            seed=seed,
            max_score=MAX_SCORE_4_TOKENS,
            gamma=gamma,
        )

