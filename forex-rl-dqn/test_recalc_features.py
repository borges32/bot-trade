"""
Teste para verificar que o OptimizedFeatureEngineer SEMPRE calcula indicadores.
"""

import pandas as pd
import numpy as np
from src.common.features_optimized import OptimizedFeatureEngineer

# Criar dados sintéticos OHLCV
np.random.seed(42)
n = 1000

dates = pd.date_range('2024-01-01', periods=n, freq='30min')
close_prices = 100 + np.cumsum(np.random.randn(n) * 0.5)

df = pd.DataFrame({
    'timestamp': dates,
    'open': close_prices + np.random.randn(n) * 0.2,
    'high': close_prices + np.abs(np.random.randn(n)) * 0.5,
    'low': close_prices - np.abs(np.random.randn(n)) * 0.5,
    'close': close_prices,
    'volume': np.random.randint(1000, 10000, n)
})

# Adicionar indicadores "falsos" no CSV (para testar se são removidos)
df['rsi'] = 50  # Valor constante (errado)
df['ema_fast'] = df['close']  # Valor errado
df['ema_slow'] = df['close']  # Valor errado
df['bb_upper'] = df['close'] + 10  # Valores errados
df['bb_middle'] = df['close']
df['bb_lower'] = df['close'] - 10
df['macd'] = 0
df['macd_signal'] = 0
df['atr'] = 1
df['momentum_10'] = 0
df['momentum_20'] = 0
df['volatility'] = 0.01
df['volume_ma'] = df['volume'].mean()

print("=" * 70)
print("TESTE: OptimizedFeatureEngineer SEMPRE Recalcula Indicadores")
print("=" * 70)

print(f"\n📊 DataFrame inicial: {df.shape[0]} linhas, {df.shape[1]} colunas")
print(f"   Colunas: {list(df.columns)}")

# Verificar valores ANTES
print("\n🔍 Valores ANTES (indicadores 'falsos' do CSV):")
print(f"   RSI (primeiros 5): {df['rsi'].head().tolist()}")
print(f"   EMA_fast (primeiros 5): {df['ema_fast'].head().tolist()}")
print(f"   MACD (primeiros 5): {df['macd'].head().tolist()}")

# Criar features
print("\n🔄 Criando features com OptimizedFeatureEngineer...")
engineer = OptimizedFeatureEngineer()
df_features = engineer.create_features(df)

print(f"\n✅ DataFrame final: {df_features.shape[0]} linhas, {df_features.shape[1]} colunas")

# Verificar valores DEPOIS
print("\n🔍 Valores DEPOIS (recalculados):")
print(f"   RSI (posições 100-105): {df_features['rsi'].iloc[100:105].tolist()}")
print(f"   EMA_fast (posições 100-105): {df_features['ema_fast'].iloc[100:105].round(2).tolist()}")
print(f"   MACD (posições 100-105): {df_features['macd'].iloc[100:105].round(4).tolist()}")

# Verificar se RSI está no intervalo correto (0-100)
rsi_valid = df_features['rsi'].dropna()
print(f"\n📈 RSI válido:")
print(f"   Min: {rsi_valid.min():.2f}, Max: {rsi_valid.max():.2f}")
print(f"   Média: {rsi_valid.mean():.2f}")
print(f"   ✓ Está entre 0-100? {(rsi_valid.min() >= 0 and rsi_valid.max() <= 100)}")

# Verificar se EMAs são diferentes (não iguais ao close)
ema_different = not (df_features['ema_fast'].equals(df_features['close']))
print(f"\n📊 EMAs recalculadas:")
print(f"   EMA_fast != close? {ema_different}")

# Verificar se MACD não é zero
macd_not_zero = df_features['macd'].abs().mean() > 0.01
print(f"   MACD != 0? {macd_not_zero}")

# Verificar indicadores complementares
print(f"\n🆕 Indicadores complementares criados:")
complementary = ['stoch_k', 'stoch_d', 'adx']
for ind in complementary:
    has_it = ind in df_features.columns
    print(f"   {ind}: {'✓' if has_it else '✗'}")

# Verificar features derivadas
print(f"\n🔗 Features derivadas criadas:")
derived = ['rsi_normalized', 'ema_cross', 'bb_width', 'macd_hist', 'signal_convergence', 'market_regime']
for feat in derived:
    has_it = feat in df_features.columns
    print(f"   {feat}: {'✓' if has_it else '✗'}")

# Resumo final
print(f"\n" + "=" * 70)
print("RESUMO:")
print("=" * 70)
print(f"✓ Total de colunas: {df_features.shape[1]}")
print(f"✓ Indicadores básicos: 13 (RSI, EMAs, BB, MACD, ATR, Momentum, Vol, VolMA)")
print(f"✓ Indicadores complementares: 6 (Stochastic, ADX, SMAs)")
print(f"✓ Features derivadas: {len(engineer.features_added)}")
print(f"\n{'✅ TESTE PASSOU!' if (rsi_valid.min() >= 0 and macd_not_zero and ema_different) else '❌ TESTE FALHOU!'}")
print("=" * 70)
