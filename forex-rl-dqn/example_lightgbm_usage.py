#!/usr/bin/env python3
"""
Exemplo de uso do modelo LightGBM para gerar sinais de trading.

Este script demonstra como:
1. Carregar dados históricos
2. Criar features técnicas
3. Fazer predições com LightGBM
4. Interpretar os sinais
"""

import sys
from pathlib import Path
import pandas as pd
import yaml

# Adiciona path raiz
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

from src.inference.predictor import TradingPredictor


def main():
    print("=" * 80)
    print("EXEMPLO: Usando LightGBM para Sinais de Trading")
    print("=" * 80)
    
    # Carrega configuração
    config_file = root_dir / 'config_hybrid_30m.yaml'
    print(f"\n📋 Carregando configuração: {config_file}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Inicializa preditor
    models_dir = root_dir / config['general']['models_dir']
    lightgbm_path = models_dir / 'lightgbm_model'
    
    print(f"🔧 Carregando modelo: {lightgbm_path}")
    
    predictor = TradingPredictor(
        lightgbm_path=str(lightgbm_path),
        config=config
    )
    
    print("✓ Modelo carregado com sucesso!\n")
    
    # Carrega dados de teste
    data_file = root_dir / config['data']['train_file']
    print(f"📊 Carregando dados: {data_file}")
    
    df = pd.read_csv(data_file)
    print(f"✓ Carregados {len(df)} candles\n")
    
    # Pega últimos 100 candles para fazer predição
    recent_candles = df.tail(100).copy()
    
    print("🔮 Fazendo predição...")
    result = predictor.predict(recent_candles)
    
    # Exibe resultado
    print("\n" + "=" * 80)
    print("RESULTADO DA PREDIÇÃO")
    print("=" * 80)
    print(f"🎯 Sinal:              {result['signal']}")
    print(f"📈 Retorno esperado:   {result['predicted_return']:.4%}")
    print(f"💪 Confiança:          {result['confidence']:.2%}")
    print(f"💵 Preço atual:        {result['current_price']:.5f}")
    
    # Interpretação
    print("\n" + "=" * 80)
    print("INTERPRETAÇÃO")
    print("=" * 80)
    
    if result['signal'] == 'BUY':
        print("✅ COMPRAR - O modelo prevê uma alta no preço")
        print(f"   Retorno esperado: +{result['predicted_return']:.4%}")
    elif result['signal'] == 'SELL':
        print("❌ VENDER - O modelo prevê uma queda no preço")
        print(f"   Retorno esperado: {result['predicted_return']:.4%}")
    else:
        print("⏸️  NEUTRO - Sinal não é forte o suficiente")
        print(f"   Confiança abaixo do mínimo ({config['inference']['min_confidence']:.0%})")
    
    # Batch prediction nos últimos dias
    print("\n" + "=" * 80)
    print("PREDIÇÕES EM BATCH (últimos 50 candles)")
    print("=" * 80)
    
    batch_results = predictor.batch_predict(df.tail(50))
    
    # Conta sinais
    signal_counts = batch_results['signal'].value_counts()
    
    print(f"\n📊 Distribuição de sinais:")
    for signal, count in signal_counts.items():
        pct = count / len(batch_results) * 100
        print(f"   {signal:8s}: {count:3d} ({pct:5.1f}%)")
    
    # Últimos 5 sinais
    print(f"\n🕒 Últimos 5 sinais:")
    print("-" * 80)
    for _, row in batch_results.tail(5).iterrows():
        ts = row['timestamp'] if 'timestamp' in row else "N/A"
        print(f"   {ts:20s} | {row['signal']:8s} | "
              f"Retorno: {row['predicted_return']:7.4%} | "
              f"Conf: {row['confidence']:5.2%}")
    
    print("\n" + "=" * 80)
    print("✓ EXEMPLO CONCLUÍDO!")
    print("=" * 80)
    
    print("\n💡 Próximos passos:")
    print("   1. Integrar com sua plataforma de trading (cTrader, MT5, etc)")
    print("   2. Iniciar API REST: python -m src.api.main --config config_hybrid_30m.yaml")
    print("   3. Retreinar com novos dados: ./retrain_lightgbm_30m.sh")


if __name__ == '__main__':
    main()
