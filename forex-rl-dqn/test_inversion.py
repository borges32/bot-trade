"""Test model performance with inverted predictions."""
import argparse
import numpy as np
import pandas as pd
import torch
import yaml
from pathlib import Path

from src.common.features import FeatureScaler, create_windows, generate_features
from src.rl.agent import DQNAgent
from src.rl.env import ForexTradingEnv


def evaluate_model(agent, env, n_episodes=1000, invert=False):
    """Evaluate model with optional action inversion."""
    total_reward = 0.0
    total_position_reward = 0.0
    total_cost = 0.0
    wins = 0
    
    action_counts = {0: 0, 1: 0, 2: 0}
    
    n_episodes = min(n_episodes, env.get_episode_count())
    
    for _ in range(n_episodes):
        state, _ = env.reset()
        
        # Get model prediction
        action, _ = agent.act(state, greedy=True)
        action_counts[action] += 1
        
        # Invert if requested
        if invert:
            if action == 1:  # BUY -> SELL
                action = 2
            elif action == 2:  # SELL -> BUY
                action = 1
            # HOLD stays HOLD
        
        # Execute action
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
        "action_distribution": action_counts,
    }


def main(model_path, data_path, config_path):
    """Run evaluation with and without inversion."""
    
    print("="*70)
    print("TESTE DE INVERSÃO DE PREDIÇÕES")
    print("="*70)
    
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Load data
    print(f"\n📊 Carregando dados: {data_path}")
    df = pd.read_csv(data_path)
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Generate features
    feature_names = config["env"]["features"]
    features_df = generate_features(df, feature_names)
    full_df = pd.concat([df, features_df], axis=1).dropna().reset_index(drop=True)
    
    # Extract and scale
    features = full_df[feature_names].values
    prices = full_df["close"].values
    
    scaler = FeatureScaler()
    features = scaler.fit_transform(features, feature_names)
    
    # Create windows
    window_size = config["env"]["window_size"]
    windows, next_prices, window_end_prices, _ = create_windows(features, prices, window_size)
    
    # Use validation split
    train_split = config["train"]["train_split"]
    split_idx = int(len(windows) * train_split)
    
    val_windows = windows[split_idx:]
    val_next_prices = next_prices[split_idx:]
    val_window_end_prices = window_end_prices[split_idx:]
    
    print(f"Validation samples: {len(val_windows)}")
    
    # Create environment
    env = ForexTradingEnv(
        val_windows,
        val_next_prices,
        val_window_end_prices,
        fee_perc=config["env"]["fee_perc"],
        spread_perc=config["env"]["spread_perc"],
    )
    
    # Load model
    print(f"\n🤖 Carregando modelo: {model_path}")
    
    n_features = windows.shape[2]
    n_actions = 3
    device = torch.device("cpu")
    
    agent = DQNAgent(
        n_features=n_features,
        n_actions=n_actions,
        device=device,
        gamma=config["agent"]["gamma"],
        lr=config["agent"]["lr"],
        epsilon_start=0.0,  # No exploration
        epsilon_end=0.0,
        epsilon_decay_steps=1,
        target_update_interval=config["agent"]["target_update_interval"],
        grad_clip_norm=config["agent"]["grad_clip_norm"],
        lstm_hidden=config["agent"]["lstm_hidden"],
        mlp_hidden=config["agent"]["mlp_hidden"],
        dueling=config["agent"]["dueling"],
    )
    
    agent.load(model_path)
    print("✅ Modelo carregado")
    
    # Evaluate NORMAL
    print("\n" + "="*70)
    print("AVALIAÇÃO NORMAL (sem inversão)")
    print("="*70)
    
    results_normal = evaluate_model(agent, env, n_episodes=1000, invert=False)
    
    print(f"\n📊 RESULTADOS NORMAIS:")
    print(f"  Avg Reward: {results_normal['avg_reward']:.6f} ({results_normal['avg_reward']*100:.4f}%)")
    print(f"  Avg Position Reward: {results_normal['avg_position_reward']:.6f}")
    print(f"  Avg Cost: {results_normal['avg_cost']:.6f}")
    print(f"  Win Rate: {results_normal['win_rate']:.2%}")
    print(f"\n  Distribuição de Ações:")
    print(f"    HOLD: {results_normal['action_distribution'][0]}")
    print(f"    BUY:  {results_normal['action_distribution'][1]}")
    print(f"    SELL: {results_normal['action_distribution'][2]}")
    
    # Evaluate INVERTED
    print("\n" + "="*70)
    print("AVALIAÇÃO INVERTIDA (BUY↔SELL)")
    print("="*70)
    
    results_inverted = evaluate_model(agent, env, n_episodes=1000, invert=True)
    
    print(f"\n📊 RESULTADOS INVERTIDOS:")
    print(f"  Avg Reward: {results_inverted['avg_reward']:.6f} ({results_inverted['avg_reward']*100:.4f}%)")
    print(f"  Avg Position Reward: {results_inverted['avg_position_reward']:.6f}")
    print(f"  Avg Cost: {results_inverted['avg_cost']:.6f}")
    print(f"  Win Rate: {results_inverted['win_rate']:.2%}")
    
    # Comparison
    print("\n" + "="*70)
    print("COMPARAÇÃO")
    print("="*70)
    
    improvement_reward = ((results_inverted['avg_reward'] - results_normal['avg_reward']) / 
                         abs(results_normal['avg_reward']) * 100) if results_normal['avg_reward'] != 0 else 0
    improvement_win_rate = (results_inverted['win_rate'] - results_normal['win_rate']) * 100
    
    print(f"\n📈 Melhoria com inversão:")
    print(f"  Avg Reward: {improvement_reward:+.1f}%")
    print(f"  Win Rate: {improvement_win_rate:+.1f} pontos percentuais")
    
    if results_inverted['win_rate'] > results_normal['win_rate']:
        print(f"\n✅ INVERSÃO MELHORA O DESEMPENHO!")
        print(f"   Usar inversão em produção:")
        print(f"   - Win Rate sobe de {results_normal['win_rate']:.1%} para {results_inverted['win_rate']:.1%}")
        print(f"   - Avg Reward melhora {improvement_reward:+.1f}%")
    else:
        print(f"\n❌ Inversão NÃO ajuda.")
        print(f"   Problema está em outro lugar (dados, features, arquitetura)")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test model with inverted predictions")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model (.pt file)"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to CSV data file"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file"
    )
    
    args = parser.parse_args()
    main(args.model, args.data, args.config)
