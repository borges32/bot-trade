"""Experience replay buffer for DQN."""
from collections import deque
from typing import Tuple

import numpy as np


class ReplayBuffer:
    """Simple replay buffer for storing and sampling experiences."""
    
    def __init__(self, capacity: int):
        """Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store.
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add an experience to the buffer.
        
        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: Whether episode ended.
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample a batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample.
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones).
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        states, actions, rewards, next_states, dones = zip(
            *[self.buffer[idx] for idx in indices]
        )
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )
    
    def __len__(self) -> int:
        """Get current size of the buffer.
        
        Returns:
            Number of experiences in the buffer.
        """
        return len(self.buffer)
