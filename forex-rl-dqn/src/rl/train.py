"""Training script for Forex RL DQN agent."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.common.features import FeatureScaler, create_windows, generate_features
from src.common.utils import get_device, set_seed
from src.rl.agent import DQNAgent
from src.rl.env import ForexTradingEnv
from src.rl.replay_buffer import ReplayBuffer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config file.
        
    Returns:
        Configuration dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_and_prepare_data(
    data_path: str,
    config: dict
) -> tuple:
    """Load CSV data and prepare features.
    
    Args:
        data_path: Path to CSV file.
        config: Configuration dictionary.
        
    Returns:
        Tuple of (train_windows, train_prices, val_windows, val_prices, scaler).
    """
    print(f"Loading data from {data_path}...")
    
    # Load CSV
    df = pd.read_csv(data_path)
    
    # Validate required columns
    required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    print(f"Loaded {len(df)} records")
    
    # Generate features
    feature_names = config["env"]["features"]
    print(f"Generating features: {feature_names}")
    
    features_df = generate_features(df, feature_names)
    
    # Combine with original data
    full_df = pd.concat([df, features_df], axis=1)
    
    # Drop NaN rows (from feature calculation)
    full_df = full_df.dropna().reset_index(drop=True)
    
    print(f"After feature generation and NaN removal: {len(full_df)} records")
    
    # Split into train/val
    train_split = config["train"]["train_split"]
    split_idx = int(len(full_df) * train_split)
    
    train_df = full_df.iloc[:split_idx].reset_index(drop=True)
    val_df = full_df.iloc[split_idx:].reset_index(drop=True)
    
    print(f"Train: {len(train_df)} records, Val: {len(val_df)} records")
    
    # Extract features and prices
    train_features = train_df[feature_names].values
    train_prices = train_df["close"].values
    
    val_features = val_df[feature_names].values
    val_prices = val_df["close"].values
    
    # Scale features
    scaler = FeatureScaler()
    
    if config["env"]["scale_features"]:
        print("Fitting scaler on training data...")
        train_features = scaler.fit_transform(train_features, feature_names)
        val_features = scaler.transform(val_features)
    else:
        scaler.fit(train_features, feature_names)
    
    # Create windows
    window_size = config["env"]["window_size"]
    print(f"Creating windows with size {window_size}...")
    
    train_windows, train_next_prices, train_window_end_prices, _ = create_windows(
        train_features, train_prices, window_size
    )
    val_windows, val_next_prices, val_window_end_prices, _ = create_windows(
        val_features, val_prices, window_size
    )
    
    print(f"Train windows: {len(train_windows)}, Val windows: {len(val_windows)}")
    
    return (
        train_windows, train_next_prices, train_window_end_prices,
        val_windows, val_next_prices, val_window_end_prices,
        scaler
    )


def evaluate_agent(
    agent: DQNAgent,
    env: ForexTradingEnv,
    n_episodes: int = 100
) -> dict:
    """Evaluate agent on validation data.
    
    Args:
        agent: DQN agent.
        env: Trading environment.
        n_episodes: Number of episodes to evaluate.
        
    Returns:
        Dictionary with evaluation metrics.
    """
    total_reward = 0.0
    total_position_reward = 0.0
    total_cost = 0.0
    wins = 0
    
    n_episodes = min(n_episodes, env.get_episode_count())
    
    for _ in range(n_episodes):
        state, _ = env.reset()
        action, _ = agent.act(state, greedy=True)
        _, reward, _, _, info = env.step(action)
        
        total_reward += reward
        total_position_reward += info["position_reward"]
        total_cost += info["cost"]
        
        if reward > 0:
            wins += 1
    
    return {
        "avg_reward": total_reward / n_episodes,
        "avg_position_reward": total_position_reward / n_episodes,
        "avg_cost": total_cost / n_episodes,
        "win_rate": wins / n_episodes,
    }


def train(
    config_path: str,
    data_path: str,
    artifacts_dir: str = "artifacts"
) -> None:
    """Main training loop.
    
    Args:
        config_path: Path to configuration file.
        data_path: Path to training data CSV.
        artifacts_dir: Directory to save artifacts.
    """
    # Load config
    config = load_config(config_path)
    
    # Set seed
    set_seed(config["seed"])
    
    # Get device
    device = get_device(config["train"]["device"])
    print(f"Using device: {device}")
    
    # Prepare data
    (train_windows, train_prices, train_window_end_prices,
     val_windows, val_prices, val_window_end_prices,
     scaler) = load_and_prepare_data(data_path, config)
    
    # Create environments
    train_env = ForexTradingEnv(
        train_windows,
        train_prices,
        train_window_end_prices,
        fee_perc=config["env"]["fee_perc"],
        spread_perc=config["env"]["spread_perc"],
    )
    
    val_env = ForexTradingEnv(
        val_windows,
        val_prices,
        val_window_end_prices,
        fee_perc=config["env"]["fee_perc"],
        spread_perc=config["env"]["spread_perc"],
    )
    
    print(f"Train episodes: {train_env.get_episode_count()}")
    print(f"Val episodes: {val_env.get_episode_count()}")
    
    # Create agent
    n_features = train_windows.shape[2]
    n_actions = 3
    
    agent = DQNAgent(
        n_features=n_features,
        n_actions=n_actions,
        device=device,
        gamma=config["agent"]["gamma"],
        lr=config["agent"]["lr"],
        epsilon_start=config["agent"]["epsilon_start"],
        epsilon_end=config["agent"]["epsilon_end"],
        epsilon_decay_steps=config["agent"]["epsilon_decay_steps"],
        target_update_interval=config["agent"]["target_update_interval"],
        grad_clip_norm=config["agent"]["grad_clip_norm"],
        lstm_hidden=config["agent"]["lstm_hidden"],
        mlp_hidden=config["agent"]["mlp_hidden"],
        dueling=config["agent"]["dueling"],
    )
    
    # Create replay buffer
    replay_buffer = ReplayBuffer(capacity=config["agent"]["replay_size"])
    
    # Training loop
    max_steps = config["train"]["max_steps"]
    batch_size = config["agent"]["batch_size"]
    start_training_after = config["agent"]["start_training_after"]
    eval_interval = config["train"]["eval_interval"]
    checkpoint_interval = config["train"]["checkpoint_interval"]
    
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50 + "\n")
    
    step = 0
    episode = 0
    total_reward = 0.0
    
    while step < max_steps:
        # Reset environment
        state, _ = train_env.reset()
        episode += 1
        
        # Select action
        action, _ = agent.act(state)
        
        # Execute action
        next_state, reward, terminated, _, info = train_env.step(action)
        
        # Store in replay buffer
        replay_buffer.push(state, action, reward, next_state, float(terminated))
        
        total_reward += reward
        step += 1
        
        # Train agent
        if len(replay_buffer) >= start_training_after:
            batch = replay_buffer.sample(batch_size)
            loss = agent.train_step(*batch)
            
            if step % 1000 == 0:
                print(
                    f"Step {step}/{max_steps} | Episode {episode} | "
                    f"Loss: {loss:.4f} | Epsilon: {agent.get_epsilon():.3f} | "
                    f"Avg Reward: {total_reward/episode:.4f}"
                )
        
        # Evaluation
        if step % eval_interval == 0 and step > 0:
            print(f"\n--- Evaluation at step {step} ---")
            eval_metrics = evaluate_agent(agent, val_env, n_episodes=100)
            print(f"Avg Reward: {eval_metrics['avg_reward']:.6f}")
            print(f"Avg Position Reward: {eval_metrics['avg_position_reward']:.6f}")
            print(f"Avg Cost: {eval_metrics['avg_cost']:.6f}")
            print(f"Win Rate: {eval_metrics['win_rate']:.2%}")
            print()
        
        # Checkpoint
        if step % checkpoint_interval == 0 and step > 0:
            artifacts_path = Path(artifacts_dir)
            artifacts_path.mkdir(exist_ok=True)
            
            agent.save(artifacts_path / f"dqn_step_{step}.pt")
            print(f"Checkpoint saved at step {step}")
    
    # Final save
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(exist_ok=True)
    
    agent.save(artifacts_path / "dqn.pt")
    scaler.save(artifacts_path / "feature_state.json")
    
    # Save config
    with open(artifacts_path / "config.yaml", "w") as f:
        yaml.dump(config, f)
    
    print("\n" + "="*50)
    print("Training complete!")
    print(f"Model saved to {artifacts_path / 'dqn.pt'}")
    print(f"Scaler saved to {artifacts_path / 'feature_state.json'}")
    print("="*50)
    
    # Final evaluation
    print("\n--- Final Evaluation ---")
    eval_metrics = evaluate_agent(agent, val_env, n_episodes=val_env.get_episode_count())
    print(f"Avg Reward: {eval_metrics['avg_reward']:.6f}")
    print(f"Avg Position Reward: {eval_metrics['avg_position_reward']:.6f}")
    print(f"Avg Cost: {eval_metrics['avg_cost']:.6f}")
    print(f"Win Rate: {eval_metrics['win_rate']:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Forex RL DQN agent")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to CSV data file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--artifacts",
        type=str,
        default="artifacts",
        help="Directory to save artifacts"
    )
    
    args = parser.parse_args()
    
    train(args.config, args.data, args.artifacts)
