"""
Script para executar predições e salvar no Redis.
Pode ser executado manualmente ou via cron/scheduler.
"""

import yaml
import pandas as pd
import sys
from pathlib import Path
from src.inference.predictor import TradingPredictor

def main():
    """Executa predição e salva no Redis."""
    
    # 1. Carrega configuração
    config_path = 'config_30m_optimized.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 2. Define caminho do modelo
    model_path = 'models/hybrid_30m/lightgbm_model.txt'
    
    if not Path(model_path).exists():
        print(f"❌ Modelo não encontrado: {model_path}")
        print("💡 Execute o treinamento primeiro ou ajuste o caminho do modelo")
        sys.exit(1)
    
    # 3. Inicializa predictor (com Redis habilitado)
    print("🔧 Inicializando predictor...")
    predictor = TradingPredictor(
        lightgbm_path=model_path,
        config=config,
        enable_redis=True
    )
    
    # 4. Carrega dados históricos
    data_path = 'data/usdjpy_history_30m.csv'
    if not Path(data_path).exists():
        print(f"❌ Dados não encontrados: {data_path}")
        sys.exit(1)
    
    print(f"📊 Carregando dados de {data_path}...")
    df = pd.read_csv(data_path)
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 5. Pega últimos 100 candles
    recent_candles = df.tail(100)
    
    # 6. Faz predição
    print("🔮 Fazendo predição...")
    result = predictor.predict(recent_candles)
    
    # 7. Exibe resultado
    print("\n" + "="*70)
    print("📊 Predição Salva no Redis")
    print("="*70)
    print(f"  🎯 Sinal: {result['signal']}")
    print(f"  📈 Retorno Previsto: {result['predicted_return']:.4f}% ({result['predicted_return']*100:.2f} basis points)")
    print(f"  📊 Acurácia Base do Modelo: {result.get('base_accuracy', result['confidence']):.2%}")
    print(f"  💯 Confiança Ajustada: {result['confidence']:.2%}")
    print(f"  💰 Preço Atual: {result['current_price']:.5f}")
    print("="*70)
    
    base_acc = result.get('base_accuracy', result['confidence'])
    if result['signal'] == 'BUY':
        print(f"\n✅ COMPRAR - Modelo prevê alta de ~{result['predicted_return']*100:.2f}%")
        print(f"📈 Probabilidade de acerto: {base_acc:.1%} (histórico do modelo)")
    elif result['signal'] == 'SELL':
        print(f"\n❌ VENDER - Modelo prevê queda de ~{abs(result['predicted_return'])*100:.2f}%")
        print(f"📉 Probabilidade de acerto: {base_acc:.1%} (histórico do modelo)")
    else:
        print(f"\n⏸️  NEUTRO - Confiança insuficiente ({result['confidence']:.1%} < threshold)")
    
    print("\n✅ Predição salva no Redis com sucesso!")
    print("🌐 Acesse o frontend em: http://localhost:3000")
    print("🔗 API endpoint: http://localhost:8000/api/prediction/latest")

if __name__ == '__main__':
    main()
