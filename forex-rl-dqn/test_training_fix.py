"""Test if the training fix resolves the zero reward problem."""
import numpy as np
import torch
import yaml
from pathlib import Path

from src.common.features import FeatureScaler, create_windows, generate_features
from src.rl.agent import DQNAgent
from src.rl.env import ForexTradingEnv
from src.rl.replay_buffer import ReplayBuffer
import pandas as pd

print("="*70)
print("TESTE DE TREINAMENTO - Verificando correção do bug de reward zero")
print("="*70)

# Load config
with open("config_5m_aggressive.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load small subset of data
df = pd.read_csv("data/usdjpy_history.csv")
df = df.iloc[:10000]  # Use apenas 10k registros para teste rápido
df = df.sort_values("timestamp").reset_index(drop=True)

print(f"\n📊 Dados: {len(df)} registros")

# Generate features
feature_names = config["env"]["features"]
features_df = generate_features(df, feature_names)
full_df = pd.concat([df, features_df], axis=1).dropna().reset_index(drop=True)

print(f"📈 Após features: {len(full_df)} registros")

# Extract features and prices
features = full_df[feature_names].values
prices = full_df["close"].values

# Scale
scaler = FeatureScaler()
features = scaler.fit_transform(features, feature_names)

# Create windows
window_size = config["env"]["window_size"]
windows, next_prices, window_end_prices, _ = create_windows(features, prices, window_size)

print(f"🪟 Windows: {len(windows)}")

# Create environment
env = ForexTradingEnv(
    windows,
    next_prices,
    window_end_prices,
    fee_perc=config["env"]["fee_perc"],
    spread_perc=config["env"]["spread_perc"],
)

# Create agent
n_features = windows.shape[2]
n_actions = 3
device = torch.device("cpu")

agent = DQNAgent(
    n_features=n_features,
    n_actions=n_actions,
    device=device,
    gamma=config["agent"]["gamma"],
    lr=config["agent"]["lr"],
    epsilon_start=1.0,  # Sempre explora para ver diversos rewards
    epsilon_end=1.0,
    epsilon_decay_steps=1,
    target_update_interval=config["agent"]["target_update_interval"],
    grad_clip_norm=config["agent"]["grad_clip_norm"],
    lstm_hidden=config["agent"]["lstm_hidden"],
    mlp_hidden=config["agent"]["mlp_hidden"],
    dueling=config["agent"]["dueling"],
)

# Create replay buffer
replay_buffer = ReplayBuffer(capacity=10000)

print("\n" + "="*70)
print("INICIANDO TESTE DE 2000 STEPS")
print("="*70)

# Training loop
batch_size = config["agent"]["batch_size"]
rewards_collected = []
losses_collected = []

for step in range(2000):
    # Reset
    state, _ = env.reset()
    
    # Random action (epsilon=1.0)
    action, _ = agent.act(state)
    
    # Step
    next_state, reward, done, _, info = env.step(action)
    
    # Store
    replay_buffer.push(state, action, reward, next_state, float(done))
    rewards_collected.append(reward)
    
    # Train
    if len(replay_buffer) >= batch_size:
        batch = replay_buffer.sample(batch_size)
        loss = agent.train_step(*batch)
        losses_collected.append(loss)
        
        if step % 500 == 0 and step > 0:
            avg_reward = np.mean(rewards_collected[-500:])
            avg_loss = np.mean(losses_collected[-100:]) if losses_collected else 0
            
            print(f"Step {step:4d} | "
                  f"Avg Reward (500): {avg_reward:+.6f} | "
                  f"Avg Loss (100): {avg_loss:.6f}")

print("\n" + "="*70)
print("RESULTADOS FINAIS")
print("="*70)

# Análise de rewards
rewards_array = np.array(rewards_collected)
positive_rewards = rewards_array[rewards_array > 0]
negative_rewards = rewards_array[rewards_array < 0]
zero_rewards = rewards_array[rewards_array == 0]

print(f"\n📊 DISTRIBUIÇÃO DE REWARDS:")
print(f"  Total de steps: {len(rewards_array)}")
print(f"  Rewards positivos: {len(positive_rewards)} ({len(positive_rewards)/len(rewards_array)*100:.1f}%)")
print(f"  Rewards negativos: {len(negative_rewards)} ({len(negative_rewards)/len(rewards_array)*100:.1f}%)")
print(f"  Rewards zero: {len(zero_rewards)} ({len(zero_rewards)/len(rewards_array)*100:.1f}%)")
print(f"\n  Reward médio: {rewards_array.mean():+.6f}")
print(f"  Reward std: {rewards_array.std():.6f}")
print(f"  Reward min: {rewards_array.min():+.6f}")
print(f"  Reward max: {rewards_array.max():+.6f}")

# Análise de loss
if losses_collected:
    losses_array = np.array(losses_collected)
    print(f"\n📉 LOSS:")
    print(f"  Loss médio: {losses_array.mean():.6f}")
    print(f"  Loss std: {losses_array.std():.6f}")
    print(f"  Loss min: {losses_array.min():.6f}")
    print(f"  Loss max: {losses_array.max():.6f}")

# Análise de ações
action_counts = {}
for i in range(min(2000, len(replay_buffer))):
    sample = replay_buffer.buffer[i]
    action = sample[1]
    action_counts[action] = action_counts.get(action, 0) + 1

print(f"\n🎯 DISTRIBUIÇÃO DE AÇÕES:")
action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}
for action, count in sorted(action_counts.items()):
    print(f"  {action_names[action]:4s}: {count:4d} ({count/sum(action_counts.values())*100:.1f}%)")

print("\n" + "="*70)

# Verificação final
if abs(rewards_array.mean()) < 1e-9 and rewards_array.std() < 1e-9:
    print("❌ FALHOU: Rewards ainda estão todos zerados!")
    print("   Problema no ambiente ou cálculo de reward.")
elif len(positive_rewards) > 0 and len(negative_rewards) > 0:
    print("✅ SUCESSO: Ambiente gerando rewards variados!")
    print("   O modelo agora pode aprender.")
else:
    print("⚠️  ATENÇÃO: Rewards não estão zerados, mas distribuição suspeita.")
    
if losses_collected and losses_array.mean() > 0:
    print("✅ SUCESSO: Loss está sendo calculado corretamente!")
else:
    print("❌ FALHOU: Loss está zerado ou não foi calculado.")

print("="*70)
