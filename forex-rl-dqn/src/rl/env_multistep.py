"""Multi-step Forex trading environment for better learning."""
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ForexTradingEnvMultiStep(gym.Env):
    """Multi-step forex trading environment.
    
    Instead of single-step episodes, allows agent to trade over multiple steps
    within an episode, leading to better credit assignment.
    """
    
    def __init__(
        self,
        windows: np.ndarray,
        prices: np.ndarray,
        window_end_prices: np.ndarray,
        fee_perc: float = 0.0001,
        spread_perc: float = 0.0002,
        episode_length: int = 100,
    ):
        """Initialize multi-step trading environment.
        
        Args:
            windows: Array of shape (n_samples, window_size, n_features).
            prices: Array of shape (n_samples,) with prices.
            window_end_prices: Array of shape (n_samples,) with window end prices.
            fee_perc: Trading fee percentage.
            spread_perc: Bid-ask spread percentage.
            episode_length: Number of steps per episode.
        """
        super().__init__()
        
        self.windows = windows
        self.prices = prices
        self.window_end_prices = window_end_prices
        self.n_samples = len(windows)
        self.window_size = windows.shape[1]
        self.n_features = windows.shape[2]
        
        self.fee_perc = fee_perc
        self.spread_perc = spread_perc
        self.episode_length = episode_length
        
        # Action space: 0=hold, 1=buy, 2=sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, self.n_features),
            dtype=np.float32
        )
        
        # Episode state
        self.current_idx = 0
        self.position = 0
        self.episode_step = 0
        self.start_idx = 0
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset to start of a new episode."""
        super().reset(seed=seed)
        
        # Random starting point (ensuring we have episode_length steps ahead)
        max_start = self.n_samples - self.episode_length - 1
        if max_start <= 0:
            raise ValueError(f"Not enough samples for episode_length={self.episode_length}")
            
        self.start_idx = self.np_random.integers(0, max_start)
        self.current_idx = self.start_idx
        self.position = 0
        self.episode_step = 0
        
        observation = self.windows[self.current_idx]
        
        info = {
            "idx": self.current_idx,
            "position": self.position,
            "episode_step": self.episode_step,
        }
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step."""
        # Map action to position
        action_to_position = {0: 0, 1: 1, 2: -1}
        new_position = action_to_position[action]
        
        # Calculate cost
        position_change_cost = 0.0
        if new_position != self.position:
            position_change_cost = self.fee_perc + self.spread_perc
        
        # Current and next prices
        current_price = self.window_end_prices[self.current_idx]
        next_price = self.prices[self.current_idx]
        
        # Price return
        price_return = (next_price - current_price) / current_price
        
        # Position reward (using NEW position)
        old_position = self.position
        self.position = new_position
        position_reward = price_return * self.position
        
        # Total reward
        reward = position_reward - position_change_cost
        
        # Move to next step
        self.current_idx += 1
        self.episode_step += 1
        
        # Check if episode is done
        terminated = (self.episode_step >= self.episode_length or 
                     self.current_idx >= self.n_samples - 1)
        truncated = False
        
        # Next observation
        if not terminated:
            next_observation = self.windows[self.current_idx]
        else:
            next_observation = self.windows[self.current_idx - 1]
        
        info = {
            "idx": self.current_idx,
            "position": self.position,
            "old_position": old_position,
            "price_return": price_return,
            "position_reward": position_reward,
            "cost": position_change_cost,
            "episode_step": self.episode_step,
        }
        
        return next_observation, reward, terminated, truncated, info
    
    def get_episode_count(self) -> int:
        """Get number of possible episodes."""
        return max(1, (self.n_samples - self.episode_length) // self.episode_length)
