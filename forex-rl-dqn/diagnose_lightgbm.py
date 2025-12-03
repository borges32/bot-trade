"""
Diagnóstico do LightGBM - Por que acurácia de 51%?

Possíveis causas:
1. Target muito balanceado (50/50) sem sinal claro
2. Threshold muito pequeno (ruído do mercado)
3. Features não discriminativas
4. Horizonte de previsão muito curto/longo
5. Data leakage ou overfitting nos dados de treino
"""

import pandas as pd
import numpy as np
import yaml
import sys
from pathlib import Path

root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

from src.common.features_optimized import OptimizedFeatureEngineer

# Carrega config
with open('config_hybrid_30m.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Carrega dados
df = pd.read_csv('data/usdjpy_history_30m.csv')
print(f"📊 Dados carregados: {len(df)} candles")
print(f"Colunas: {df.columns.tolist()}\n")

# Cria features
fe = OptimizedFeatureEngineer(config)
df_features = fe.create_features(df)
df_features = df_features.dropna().reset_index(drop=True)

print(f"✅ Features criadas: {df_features.shape}\n")

# Analisa diferentes configurações de target
horizons = [3, 5, 10, 20]
thresholds = [0.0, 0.0001, 0.0003, 0.0005, 0.001]

print("="*80)
print("ANÁLISE DE CONFIGURAÇÕES DE TARGET")
print("="*80)

results = []

for horizon in horizons:
    for threshold in thresholds:
        # Cria target
        future_close = df_features['close'].shift(-horizon)
        price_change = (future_close - df_features['close']) / df_features['close']
        target = (price_change > threshold).astype(int)
        
        # Remove NaN
        valid = ~target.isna()
        target_valid = target[valid]
        
        # Estatísticas
        n_samples = len(target_valid)
        n_up = (target_valid == 1).sum()
        n_down = (target_valid == 0).sum()
        balance = n_up / n_samples if n_samples > 0 else 0
        
        # Magnitude média das mudanças
        price_change_valid = price_change[valid]
        avg_up_move = price_change_valid[target_valid == 1].mean() if n_up > 0 else 0
        avg_down_move = price_change_valid[target_valid == 0].mean() if n_down > 0 else 0
        
        results.append({
            'horizon': horizon,
            'threshold': threshold,
            'n_samples': n_samples,
            'balance': balance,
            'avg_up_move': avg_up_move,
            'avg_down_move': avg_down_move
        })
        
        print(f"\nHorizon: {horizon:2d} candles | Threshold: {threshold:.4f}")
        print(f"  Samples: {n_samples}")
        print(f"  Class 1 (UP):   {n_up:5d} ({n_up/n_samples*100:5.2f}%)")
        print(f"  Class 0 (DOWN): {n_down:5d} ({n_down/n_samples*100:5.2f}%)")
        print(f"  Balance: {balance:.3f}")
        print(f"  Avg UP move:   {avg_up_move*100:+.4f}%")
        print(f"  Avg DOWN move: {avg_down_move*100:+.4f}%")

# Recomendações
print("\n" + "="*80)
print("RECOMENDAÇÕES")
print("="*80)

df_results = pd.DataFrame(results)

# Encontra configuração mais balanceada mas com sinal
# Queremos balance entre 0.45-0.55 mas com movimentos significativos
df_results['signal_strength'] = abs(df_results['avg_up_move'] - df_results['avg_down_move'])
df_results['balance_score'] = 1 - abs(df_results['balance'] - 0.5) * 2  # Penaliza desbalanceamento
df_results['combined_score'] = df_results['signal_strength'] * df_results['balance_score']

best = df_results.nlargest(5, 'combined_score')

print("\n🏆 Top 5 Configurações (melhor sinal + balanceamento):\n")
print(best[['horizon', 'threshold', 'balance', 'signal_strength', 'combined_score']].to_string(index=False))

print("\n📌 Configuração Recomendada:")
best_config = best.iloc[0]
print(f"   Prediction Horizon: {int(best_config['horizon'])} candles")
print(f"   Classification Threshold: {best_config['threshold']:.4f}")
print(f"   Balance: {best_config['balance']:.3f}")
print(f"   Signal Strength: {best_config['signal_strength']*100:.4f}%")

# Análise de correlação das features com o target
print("\n" + "="*80)
print("ANÁLISE DE CORRELAÇÃO DAS FEATURES")
print("="*80)

# Usa a melhor configuração
horizon = int(best_config['horizon'])
threshold = best_config['threshold']

future_close = df_features['close'].shift(-horizon)
price_change = (future_close - df_features['close']) / df_features['close']
target = (price_change > threshold).astype(int)

# Remove NaN
valid = ~target.isna()
df_analysis = df_features[valid].copy()
df_analysis['target'] = target[valid]

# Features (exclui OHLCV e timestamp)
exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'target']
feature_cols = [col for col in df_analysis.columns if col not in exclude_cols]

# Calcula correlação
correlations = []
for col in feature_cols:
    corr = df_analysis[col].corr(df_analysis['target'])
    correlations.append({'feature': col, 'correlation': abs(corr), 'signed_corr': corr})

df_corr = pd.DataFrame(correlations).sort_values('correlation', ascending=False)

print("\n🔝 Top 20 Features Mais Correlacionadas com Target:\n")
print(df_corr.head(20)[['feature', 'signed_corr', 'correlation']].to_string(index=False))

print("\n⚠️ Bottom 10 Features (menos correlacionadas):\n")
print(df_corr.tail(10)[['feature', 'signed_corr', 'correlation']].to_string(index=False))

# Sugestões finais
print("\n" + "="*80)
print("DIAGNÓSTICO E SOLUÇÕES")
print("="*80)

print("\n🔍 Problemas Identificados:")
print("   1. Threshold muito pequeno captura ruído do mercado")
print("   2. Horizonte pode ser inadequado para o timeframe")
print("   3. Features podem ter baixa correlação com target")

print("\n💡 Soluções Propostas:")
print("   1. Aumentar threshold para capturar movimentos reais")
print("   2. Ajustar horizonte baseado no timeframe (30m → 6-10 candles)")
print("   3. Feature selection: remover features com corr < 0.01")
print("   4. Feature engineering: adicionar features de tendência/volatilidade")
print("   5. Class weights: balancear classes minoritárias")
print("   6. Usar regressão em vez de classificação")

print("\n✅ Próximos Passos:")
print("   1. Atualizar config_hybrid_30m.yaml com novos valores")
print("   2. Retreinar LightGBM")
print("   3. Validar acurácia > 55% no teste")

print("\n" + "="*80)
