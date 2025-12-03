"""
Exemplo de uso do sistema com dados do cTrader que já contêm indicadores.

Este script demonstra como usar seus dados históricos que já vêm com
indicadores pré-calculados (RSI, EMAs, Bollinger Bands, MACD, etc.)
"""

import sys
from pathlib import Path
import pandas as pd

# Adiciona path raiz
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

from src.common.features_optimized import OptimizedFeatureEngineer


def main():
    """Demonstra uso do FeatureEngineer otimizado."""
    
    print("=" * 80)
    print("EXEMPLO: Usando dados do cTrader com indicadores pré-calculados")
    print("=" * 80)
    
    # Simula dados do seu CSV
    # Na prática, você faria: df = pd.read_csv('data/usdjpy_history_15m.csv')
    print("\n1. Estrutura esperada do CSV:")
    print("-" * 80)
    expected_columns = [
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'rsi', 'ema_fast', 'ema_slow',
        'bb_upper', 'bb_middle', 'bb_lower',
        'atr', 'momentum_10', 'momentum_20',
        'volatility', 'volume_ma',
        'macd', 'macd_signal'
    ]
    
    for i, col in enumerate(expected_columns, 1):
        print(f"   {i:2d}. {col}")
    
    print("\n2. Carregando dados de exemplo...")
    print("-" * 80)
    
    # Carrega dados reais (ajuste o path conforme necessário)
    data_file = root_dir / 'data' / 'usdjpy_history_15m.csv'
    
    if not data_file.exists():
        print(f"❌ Arquivo não encontrado: {data_file}")
        print("\n📝 Para usar este exemplo:")
        print("   1. Coloque seu arquivo CSV em: data/usdjpy_history_15m.csv")
        print("   2. Certifique-se que tem as colunas listadas acima")
        print("   3. Execute novamente este script")
        return
    
    df = pd.read_csv(data_file)
    print(f"✓ Carregados {len(df)} candles")
    print(f"✓ Período: {df['timestamp'].min()} até {df['timestamp'].max()}")
    
    print("\n3. Colunas encontradas no CSV:")
    print("-" * 80)
    for i, col in enumerate(df.columns, 1):
        print(f"   {i:2d}. {col}")
    
    print("\n4. Criando features otimizadas...")
    print("-" * 80)
    
    # Cria feature engineer otimizado
    fe = OptimizedFeatureEngineer()
    
    # Cria features (rápido pois usa indicadores pré-calculados)
    df_features = fe.create_features(df)
    
    print(f"✓ Features criadas com sucesso!")
    print(f"✓ Indicadores pré-calculados usados: {len(fe.precomputed_found)}")
    print(f"✓ Features novas adicionadas: {len(fe.features_added)}")
    print(f"✓ Total de colunas: {len(df_features.columns)}")
    
    print("\n5. Indicadores pré-calculados detectados:")
    print("-" * 80)
    for i, ind in enumerate(fe.precomputed_found, 1):
        desc = OptimizedFeatureEngineer.EXPECTED_PRECOMPUTED.get(ind, ind)
        print(f"   {i:2d}. {ind:20s} - {desc}")
    
    print("\n6. Features derivadas criadas:")
    print("-" * 80)
    # Mostra primeiras 20 features
    for i, feat in enumerate(fe.features_added[:20], 1):
        print(f"   {i:2d}. {feat}")
    
    if len(fe.features_added) > 20:
        print(f"   ... e mais {len(fe.features_added) - 20} features")
    
    print("\n7. Exemplo de dados processados:")
    print("-" * 80)
    
    # Mostra últimas 5 linhas com algumas features importantes
    important_cols = [
        'timestamp', 'close', 'rsi', 'rsi_normalized',
        'ema_cross', 'bb_position', 'macd_hist', 'volume_ratio'
    ]
    
    # Filtra apenas colunas que existem
    available_cols = [col for col in important_cols if col in df_features.columns]
    
    print(df_features[available_cols].tail())
    
    print("\n8. Estatísticas das features:")
    print("-" * 80)
    
    # Estatísticas de algumas features derivadas
    stats_cols = ['rsi_normalized', 'bb_position', 'ema_cross', 'volume_ratio']
    stats_cols = [col for col in stats_cols if col in df_features.columns]
    
    if stats_cols:
        print(df_features[stats_cols].describe())
    
    print("\n9. Verificação de qualidade:")
    print("-" * 80)
    
    # Verifica NaN
    nan_counts = df_features.isna().sum()
    cols_with_nan = nan_counts[nan_counts > 0]
    
    if len(cols_with_nan) > 0:
        print(f"⚠ Encontradas {len(cols_with_nan)} colunas com valores NaN:")
        for col, count in cols_with_nan.head(10).items():
            pct = (count / len(df_features)) * 100
            print(f"   - {col}: {count} ({pct:.2f}%)")
    else:
        print("✓ Nenhum valor NaN encontrado!")
    
    # Verifica infinitos
    inf_counts = df_features.select_dtypes(include=['float64', 'float32']).apply(
        lambda x: np.isinf(x).sum()
    )
    cols_with_inf = inf_counts[inf_counts > 0]
    
    if len(cols_with_inf) > 0:
        print(f"\n⚠ Encontradas {len(cols_with_inf)} colunas com valores infinitos:")
        for col, count in cols_with_inf.head(10).items():
            print(f"   - {col}: {count}")
    else:
        print("✓ Nenhum valor infinito encontrado!")
    
    print("\n10. Salvando dados processados:")
    print("-" * 80)
    
    output_file = root_dir / 'data' / 'processed_features.csv'
    df_features.to_csv(output_file, index=False)
    print(f"✓ Dados salvos em: {output_file}")
    print(f"✓ Tamanho do arquivo: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    print("\n" + "=" * 80)
    print("✓ CONCLUÍDO!")
    print("=" * 80)
    print("\n📊 Próximos passos:")
    print("   1. Use os dados processados para treinar o LightGBM:")
    print("      python src/training/train_lightgbm.py")
    print("\n   2. Depois treine o PPO:")
    print("      python src/training/train_ppo.py")
    print("\n   3. Ou use o script automático:")
    print("      ./train_hybrid.sh")


if __name__ == '__main__':
    import numpy as np
    main()
