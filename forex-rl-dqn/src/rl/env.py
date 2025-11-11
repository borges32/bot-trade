"""Forex trading environment for Reinforcement Learning."""
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ForexTradingEnv(gym.Env):
    """Forex trading environment with single position (hold/buy/sell).
    
    Actions:
        0: hold (neutral, position = 0)
        1: buy (long, position = +1)
        2: sell (short, position = -1)
    
    State:
        Window of normalized features (window_size, n_features).
    
    Reward:
        Per-step reward based on position and price change, minus costs when changing position.
    """
    
    def __init__(
        self,
        windows: np.ndarray,
        prices: np.ndarray,
        fee_perc: float = 0.0001,
        spread_perc: float = 0.0002,
    ):
        """Initialize the trading environment.
        
        Args:
            windows: Array of shape (n_episodes, window_size, n_features).
            prices: Array of shape (n_episodes,) with next prices after each window.
            fee_perc: Trading fee as percentage (e.g., 0.0001 = 0.01%).
            spread_perc: Bid-ask spread as percentage.
        """
        super().__init__()
        
        self.windows = windows
        self.prices = prices
        self.n_episodes = len(windows)
        self.window_size = windows.shape[1]
        self.n_features = windows.shape[2]
        
        self.fee_perc = fee_perc
        self.spread_perc = spread_perc
        
        # Action space: 0=hold, 1=buy, 2=sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: window of features
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, self.n_features),
            dtype=np.float32
        )
        
        # Episode state
        self.current_step = 0
        self.position = 0  # -1 (short), 0 (neutral), +1 (long)
        self.current_price = 0.0
        self.next_price = 0.0
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to a random episode.
        
        Args:
            seed: Random seed.
            options: Additional options.
            
        Returns:
            Tuple of (observation, info).
        """
        super().reset(seed=seed)
        
        # Randomly select an episode
        self.current_step = self.np_random.integers(0, self.n_episodes)
        self.position = 0
        
        # Get current window and next price
        observation = self.windows[self.current_step]
        self.next_price = self.prices[self.current_step]
        
        # Initialize current price (approximation from window if needed)
        # We assume the last close in the window is the current price
        # Since features are normalized, we use next_price as reference
        self.current_price = self.next_price
        
        info = {
            "step": self.current_step,
            "position": self.position,
        }
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment.
        
        Args:
            action: Action to take (0=hold, 1=buy, 2=sell).
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        # Map action to position
        action_to_position = {0: 0, 1: 1, 2: -1}
        new_position = action_to_position[action]
        
        # Calculate return
        price_return = (self.next_price - self.current_price) / self.current_price
        
        # Calculate reward based on position
        position_reward = price_return * self.position
        
        # Calculate cost if position changed
        position_change_cost = 0.0
        if new_position != self.position:
            position_change_cost = self.fee_perc + self.spread_perc
        
        # Total reward
        reward = position_reward - position_change_cost
        
        # Update position and price
        self.position = new_position
        self.current_price = self.next_price
        
        # Episode ends after one step (single-step episodes)
        terminated = True
        truncated = False
        
        # Next observation (will be ignored since episode terminates)
        next_observation = self.windows[self.current_step]
        
        info = {
            "step": self.current_step,
            "position": self.position,
            "price_return": price_return,
            "position_reward": position_reward,
            "cost": position_change_cost,
        }
        
        return next_observation, reward, terminated, truncated, info
    
    def get_episode_count(self) -> int:
        """Get total number of episodes available.
        
        Returns:
            Number of episodes.
        """
        return self.n_episodes
