"""Simulate inverted trading strategy on historical data."""
import pandas as pd
import numpy as np
import yaml

print("="*70)
print("SIMULAÇÃO: ESTRATÉGIA NORMAL vs INVERTIDA")
print("="*70)

# Load data - CHANGE HERE FOR DIFFERENT TIMEFRAMES
data_file = 'data/usdjpy_history_15m.csv'  # Change to 15m, 30m, 5m
config_file = 'config_15m.yaml'  # Change to config_15m.yaml, config_30m.yaml, etc

print(f"\n📂 Arquivo de dados: {data_file}")
print(f"⚙️  Config: {config_file}")

df = pd.read_csv(data_file)
df = df.sort_values('timestamp').reset_index(drop=True)

# Calculate returns
df['return'] = df['close'].pct_change()
df['return_next'] = df['return'].shift(-1)

# Remove NaN
df = df.dropna()

# Load config for costs
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

fee = config['env']['fee_perc']
spread = config['env']['spread_perc']
cost = fee + spread

print(f"\n📊 Dados: {len(df)} barras")
print(f"💰 Custo por trade: {cost*100:.4f}%")

# Simulate strategies
np.random.seed(42)

results = {}

# Strategy 1: RANDOM
print("\n" + "="*70)
print("1. ESTRATÉGIA RANDOM (baseline)")
print("="*70)

n_trades = 1000
random_actions = np.random.choice([0, 1, 2], size=n_trades)  # 0=HOLD, 1=BUY, 2=SELL
random_samples = df.sample(n_trades, random_state=42)

rewards = []
for action, (_, row) in zip(random_actions, random_samples.iterrows()):
    ret = row['return_next']
    
    if action == 0:  # HOLD
        reward = 0.0
    elif action == 1:  # BUY
        reward = ret - cost
    else:  # SELL
        reward = -ret - cost
    
    rewards.append(reward)

win_rate = sum(1 for r in rewards if r > 0) / len(rewards)
avg_reward = np.mean(rewards)

print(f"Win Rate: {win_rate:.2%}")
print(f"Avg Reward: {avg_reward:.6f} ({avg_reward*100:.4f}%)")

results['random'] = {'win_rate': win_rate, 'avg_reward': avg_reward}

# Strategy 2: ALWAYS BUY
print("\n" + "="*70)
print("2. ESTRATÉGIA ALWAYS BUY")
print("="*70)

buy_samples = df.sample(n_trades, random_state=43)
buy_rewards = [row['return_next'] - cost for _, row in buy_samples.iterrows()]
buy_win_rate = sum(1 for r in buy_rewards if r > 0) / len(buy_rewards)
buy_avg_reward = np.mean(buy_rewards)

print(f"Win Rate: {buy_win_rate:.2%}")
print(f"Avg Reward: {buy_avg_reward:.6f} ({buy_avg_reward*100:.4f}%)")

results['buy'] = {'win_rate': buy_win_rate, 'avg_reward': buy_avg_reward}

# Strategy 3: ALWAYS SELL
print("\n" + "="*70)
print("3. ESTRATÉGIA ALWAYS SELL")
print("="*70)

sell_samples = df.sample(n_trades, random_state=44)
sell_rewards = [-row['return_next'] - cost for _, row in sell_samples.iterrows()]
sell_win_rate = sum(1 for r in sell_rewards if r > 0) / len(sell_rewards)
sell_avg_reward = np.mean(sell_rewards)

print(f"Win Rate: {sell_win_rate:.2%}")
print(f"Avg Reward: {sell_avg_reward:.6f} ({sell_avg_reward*100:.4f}%)")

results['sell'] = {'win_rate': sell_win_rate, 'avg_reward': sell_avg_reward}

# Strategy 4: Trend Following (BUY if last return > 0, SELL if < 0)
print("\n" + "="*70)
print("4. ESTRATÉGIA TREND FOLLOWING")
print("="*70)

trend_samples = df[df['return'].notna()].sample(n_trades, random_state=45)
trend_rewards = []

for _, row in trend_samples.iterrows():
    if row['return'] > 0:  # Trend up -> BUY
        reward = row['return_next'] - cost
    elif row['return'] < 0:  # Trend down -> SELL
        reward = -row['return_next'] - cost
    else:  # No trend -> HOLD
        reward = 0.0
    
    trend_rewards.append(reward)

trend_win_rate = sum(1 for r in trend_rewards if r > 0) / len(trend_rewards)
trend_avg_reward = np.mean(trend_rewards)

print(f"Win Rate: {trend_win_rate:.2%}")
print(f"Avg Reward: {trend_avg_reward:.6f} ({trend_avg_reward*100:.4f}%)")

results['trend'] = {'win_rate': trend_win_rate, 'avg_reward': trend_avg_reward}

# Strategy 5: INVERTED Trend (BUY if last return < 0, SELL if > 0)
print("\n" + "="*70)
print("5. ESTRATÉGIA MEAN REVERSION (trend invertido)")
print("="*70)

inv_samples = df[df['return'].notna()].sample(n_trades, random_state=46)
inv_rewards = []

for _, row in inv_samples.iterrows():
    if row['return'] > 0:  # Trend up -> SELL (reversal)
        reward = -row['return_next'] - cost
    elif row['return'] < 0:  # Trend down -> BUY (reversal)
        reward = row['return_next'] - cost
    else:  # No trend -> HOLD
        reward = 0.0
    
    inv_rewards.append(reward)

inv_win_rate = sum(1 for r in inv_rewards if r > 0) / len(inv_rewards)
inv_avg_reward = np.mean(inv_rewards)

print(f"Win Rate: {inv_win_rate:.2%}")
print(f"Avg Reward: {inv_avg_reward:.6f} ({inv_avg_reward*100:.4f}%)")

results['mean_reversion'] = {'win_rate': inv_win_rate, 'avg_reward': inv_avg_reward}

# Summary
print("\n" + "="*70)
print("RESUMO COMPARATIVO")
print("="*70)

print(f"\n{'Estratégia':<20} {'Win Rate':<12} {'Avg Reward'}")
print("-" * 50)

for name, res in results.items():
    print(f"{name:<20} {res['win_rate']:>10.2%}  {res['avg_reward']:>+.6f}")

# Find best
best_strategy = max(results.items(), key=lambda x: x[1]['win_rate'])
print(f"\n🏆 MELHOR ESTRATÉGIA: {best_strategy[0]}")
print(f"   Win Rate: {best_strategy[1]['win_rate']:.2%}")
print(f"   Avg Reward: {best_strategy[1]['avg_reward']*100:+.4f}%")

# Análise do resultado do modelo
print("\n" + "="*70)
print("INTERPRETAÇÃO DO MODELO TREINADO")
print("="*70)

model_win_rate = 0.3118  # Do seu resultado

print(f"\n📊 Modelo treinado: Win Rate = {model_win_rate:.2%}")
print(f"📊 Random baseline:  Win Rate = {results['random']['win_rate']:.2%}")
print(f"📊 Trend following:  Win Rate = {results['trend']['win_rate']:.2%}")
print(f"📊 Mean reversion:   Win Rate = {results['mean_reversion']['win_rate']:.2%}")

if model_win_rate < 0.40:
    print(f"\n❌ Win Rate {model_win_rate:.2%} é MUITO BAIXO")
    print(f"   Modelo está PIOR que todas as estratégias simples!")
    print(f"\n💡 POSSÍVEIS CAUSAS:")
    print(f"   1. Dados 30M têm autocorrelação muito baixa (quase random)")
    print(f"   2. Features não capturam padrões úteis")
    print(f"   3. Window size muito pequeno (16 barras = 8h)")
    print(f"   4. Modelo single-step não aprende credit assignment")
    print(f"\n🎯 RECOMENDAÇÕES:")
    print(f"   • Coletar dados de par mais volátil (GBPJPY)")
    print(f"   • Testar timeframe 1H ou 4H")
    print(f"   • Adicionar mais features (volume, ATR, momentum)")
    print(f"   • Usar ambiente multi-step")

print("\n" + "="*70)
