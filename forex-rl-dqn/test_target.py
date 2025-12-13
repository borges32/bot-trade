#!/usr/bin/env python3
"""
Teste rápido do target para garantir que está correto.
"""

import pandas as pd
import numpy as np

# Carrega dados
df = pd.read_csv('data/usdjpy_history_30m.csv')
print(f"📊 Dados: {len(df)} candles\n")

# Testa target simples
horizon = 10
threshold = 0.0003

print(f"🎯 Configuração:")
print(f"   Horizon: {horizon} candles")
print(f"   Threshold: {threshold} ({threshold*100}%)\n")

# Cria target
close = df['close'].values
future_close = pd.Series(close).shift(-horizon)
price_change = (future_close - close) / close

# Aplica threshold
target = (price_change > threshold).astype(int)

# Remove NaN
valid = ~pd.isna(target)
target_valid = target[valid]
price_change_valid = price_change[valid]

print(f"✅ Target criado:")
print(f"   Samples válidos: {len(target_valid)}")
print(f"   Class 0 (DOWN): {(target_valid == 0).sum()} ({(target_valid == 0).sum()/len(target_valid)*100:.2f}%)")
print(f"   Class 1 (UP):   {(target_valid == 1).sum()} ({(target_valid == 1).sum()/len(target_valid)*100:.2f}%)")

# Estatísticas
up_moves = price_change_valid[target_valid == 1]
down_moves = price_change_valid[target_valid == 0]

print(f"\n📈 Estatísticas:")
print(f"   Média UP moves:   {up_moves.mean()*100:+.4f}%")
print(f"   Média DOWN moves: {down_moves.mean()*100:+.4f}%")
print(f"   Separação:        {abs(up_moves.mean() - down_moves.mean())*100:.4f}%")

# Verifica consistência: se target=1, future deve ser > current
print(f"\n🔍 Verificação de Consistência:")

# Pega 10 exemplos aleatórios de cada classe
np.random.seed(42)
up_samples = np.where(target_valid == 1)[0][:10]
down_samples = np.where(target_valid == 0)[0][:10]

print(f"\n   Exemplos CLASS 1 (UP):")
for i in up_samples:
    if i + horizon < len(df):
        current = close[i]
        future = close[i + horizon]
        change = (future - current) / current
        print(f"      [{i:5d}] Current={current:.5f} Future={future:.5f} Change={change*100:+.4f}% ✓" if change > threshold else f"      [{i:5d}] Current={current:.5f} Future={future:.5f} Change={change*100:+.4f}% ✗")

print(f"\n   Exemplos CLASS 0 (DOWN):")
for i in down_samples:
    if i + horizon < len(df):
        current = close[i]
        future = close[i + horizon]
        change = (future - current) / current
        print(f"      [{i:5d}] Current={current:.5f} Future={future:.5f} Change={change*100:+.4f}% ✓" if change <= threshold else f"      [{i:5d}] Current={current:.5f} Future={future:.5f} Change={change*100:+.4f}% ✗")

print(f"\n{'='*70}")
print("CONCLUSÃO:")
print("="*70)
print("Se os exemplos acima estão corretos (UP tem change > threshold,")
print("DOWN tem change <= threshold), então o TARGET ESTÁ CORRETO.")
print("\nSe acurácia é 48%, o problema é:")
print("  1. Features com data leakage (olhando para o futuro)")
print("  2. Overfitting no treino")
print("  3. Split temporal incorreto")
print("="*70)
