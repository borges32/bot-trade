"""Quick test of enhanced features."""
import pandas as pd
from src.common.features import generate_features

print("="*70)
print("TESTE DAS NOVAS FEATURES")
print("="*70)

# Load sample data
df = pd.read_csv('data/usdjpy_history_15m.csv').head(500)

print(f"\n📊 Dados carregados: {len(df)} registros")
print(f"Colunas: {list(df.columns)}")

# Test new features
new_features = [
    'atr_14',
    'momentum_10',
    'momentum_20',
    'volatility_20',
    'volume_ratio',
    'macd',
    'macd_signal',
]

print(f"\n🔍 Testando {len(new_features)} novas features...")

features_df = generate_features(df, new_features)

print(f"\n✅ Features geradas com sucesso!")
print(f"\nResumo estatístico:")
print(features_df.describe())

print(f"\n📈 Valores NaN por feature:")
print(features_df.isnull().sum())

print(f"\n📊 Primeiras 5 linhas:")
print(features_df.head())

# Test all features together
all_features = [
    'rsi_14', 'ema_12', 'ema_26', 'bb_upper_20', 'bb_lower_20',
    'returns_1', 'returns_5',
    'atr_14', 'momentum_10', 'momentum_20', 'volatility_20',
    'volume_ratio', 'macd', 'macd_signal'
]

print(f"\n\n{'='*70}")
print(f"TESTE COMPLETO - {len(all_features)} FEATURES")
print('='*70)

all_features_df = generate_features(df, all_features)
print(f"\n✅ Todas as {len(all_features)} features geradas!")
print(f"Shape: {all_features_df.shape}")
print(f"NaN total: {all_features_df.isnull().sum().sum()}")
print(f"Registros válidos: {len(all_features_df.dropna())}/{len(all_features_df)}")

print("\n" + "="*70)
print("✅ TESTE CONCLUÍDO - Features prontas para treinamento!")
print("="*70)
