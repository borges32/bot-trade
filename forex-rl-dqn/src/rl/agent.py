"""Dueling Double DQN Agent with LSTM for Forex trading."""
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DuelingDQN(nn.Module):
    """Dueling DQN architecture with LSTM for sequential data."""
    
    def __init__(
        self,
        n_features: int,
        n_actions: int,
        lstm_hidden: int = 128,
        mlp_hidden: int = 256,
        dueling: bool = True,
    ):
        """Initialize the Dueling DQN network.
        
        Args:
            n_features: Number of input features per timestep.
            n_actions: Number of possible actions.
            lstm_hidden: LSTM hidden size.
            mlp_hidden: MLP hidden layer size.
            dueling: Whether to use dueling architecture.
        """
        super().__init__()
        
        self.n_features = n_features
        self.n_actions = n_actions
        self.lstm_hidden = lstm_hidden
        self.dueling = dueling
        
        # LSTM layer to process sequences
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )
        
        if dueling:
            # Dueling architecture: separate value and advantage streams
            self.value_stream = nn.Sequential(
                nn.Linear(lstm_hidden, mlp_hidden),
                nn.ReLU(),
                nn.Linear(mlp_hidden, 1)
            )
            
            self.advantage_stream = nn.Sequential(
                nn.Linear(lstm_hidden, mlp_hidden),
                nn.ReLU(),
                nn.Linear(mlp_hidden, n_actions)
            )
        else:
            # Standard DQN
            self.q_network = nn.Sequential(
                nn.Linear(lstm_hidden, mlp_hidden),
                nn.ReLU(),
                nn.Linear(mlp_hidden, n_actions)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, seq_len, n_features).
            
        Returns:
            Q-values of shape (batch, n_actions).
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take the last output of the sequence
        lstm_last = lstm_out[:, -1, :]
        
        if self.dueling:
            # Dueling: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
            value = self.value_stream(lstm_last)
            advantage = self.advantage_stream(lstm_last)
            
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            # Standard DQN
            q_values = self.q_network(lstm_last)
        
        return q_values


class DQNAgent:
    """Double DQN Agent with experience replay and epsilon-greedy exploration."""
    
    def __init__(
        self,
        n_features: int,
        n_actions: int,
        device: torch.device,
        gamma: float = 0.99,
        lr: float = 0.0001,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 50000,
        target_update_interval: int = 500,
        grad_clip_norm: float = 10.0,
        lstm_hidden: int = 128,
        mlp_hidden: int = 256,
        dueling: bool = True,
    ):
        """Initialize the DQN agent.
        
        Args:
            n_features: Number of input features.
            n_actions: Number of possible actions.
            device: PyTorch device.
            gamma: Discount factor.
            lr: Learning rate.
            epsilon_start: Initial exploration rate.
            epsilon_end: Final exploration rate.
            epsilon_decay_steps: Steps to decay epsilon.
            target_update_interval: Steps between target network updates.
            grad_clip_norm: Gradient clipping norm.
            lstm_hidden: LSTM hidden size.
            mlp_hidden: MLP hidden size.
            dueling: Whether to use dueling architecture.
        """
        self.n_features = n_features
        self.n_actions = n_actions
        self.device = device
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.target_update_interval = target_update_interval
        self.grad_clip_norm = grad_clip_norm
        
        # Networks
        self.q_network = DuelingDQN(
            n_features, n_actions, lstm_hidden, mlp_hidden, dueling
        ).to(device)
        
        self.target_network = DuelingDQN(
            n_features, n_actions, lstm_hidden, mlp_hidden, dueling
        ).to(device)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Training state
        self.steps = 0
        self.update_count = 0
        
    def get_epsilon(self) -> float:
        """Get current epsilon value based on linear decay.
        
        Returns:
            Current epsilon value.
        """
        progress = min(self.steps / self.epsilon_decay_steps, 1.0)
        epsilon = self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress
        return epsilon
    
    def act(self, state: np.ndarray, greedy: bool = False) -> Tuple[int, float]:
        """Select an action using epsilon-greedy policy.
        
        Args:
            state: State observation of shape (window_size, n_features).
            greedy: If True, always select greedy action.
            
        Returns:
            Tuple of (action, confidence).
            - action: Selected action index.
            - confidence: Confidence score (softmax of Q-values).
        """
        epsilon = 0.0 if greedy else self.get_epsilon()
        
        # Epsilon-greedy exploration
        if np.random.random() < epsilon:
            action = np.random.randint(0, self.n_actions)
            confidence = 1.0 / self.n_actions  # Uniform confidence
        else:
            # Greedy action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                action = q_values.argmax(dim=1).item()
                
                # Calculate confidence as softmax probability of selected action
                probs = F.softmax(q_values, dim=1)
                confidence = probs[0, action].item()
        
        return action, confidence
    
    def train_step(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> float:
        """Perform one training step using Double DQN.
        
        Args:
            states: Batch of states (batch, window_size, n_features).
            actions: Batch of actions (batch,).
            rewards: Batch of rewards (batch,).
            next_states: Batch of next states (batch, window_size, n_features).
            dones: Batch of done flags (batch,).
            
        Returns:
            Loss value.
        """
        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        
        # Current Q-values
        q_values = self.q_network(states_t)
        q_values = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        
        # Double DQN: select action with online network, evaluate with target network
        with torch.no_grad():
            next_q_online = self.q_network(next_states_t)
            next_actions = next_q_online.argmax(dim=1)
            
            next_q_target = self.target_network(next_states_t)
            next_q_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            target_q_values = rewards_t + (1 - dones_t) * self.gamma * next_q_values
        
        # Huber loss (smooth L1)
        loss = F.smooth_l1_loss(q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.grad_clip_norm)
        
        self.optimizer.step()
        
        # Update counters
        self.steps += 1
        self.update_count += 1
        
        # Update target network
        if self.update_count % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def save(self, path: Path) -> None:
        """Save agent state.
        
        Args:
            path: Path to save the agent.
        """
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps": self.steps,
            "update_count": self.update_count,
            "n_features": self.n_features,
            "n_actions": self.n_actions,
        }, path)
    
    def load(self, path: Path) -> None:
        """Load agent state.
        
        Args:
            path: Path to load the agent from.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.steps = checkpoint["steps"]
        self.update_count = checkpoint["update_count"]
