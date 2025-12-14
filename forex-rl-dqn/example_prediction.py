"""
Exemplo de uso do TradingPredictor para fazer predições com LightGBM.
"""

import yaml
import pandas as pd
from pathlib import Path
from src.inference.predictor import TradingPredictor

def load_config(config_path: str):
    """Carrega arquivo de configuração."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def example_single_prediction():
    """
    Exemplo 1: Predição única a partir de dados históricos.
    """
    print("=" * 70)
    print("EXEMPLO 1: Predição Única")
    print("=" * 70)
    
    # 1. Carrega configuração
    config = load_config('config_30m_optimized.yaml')
    
    # 2. Define caminho do modelo treinado
    # Ajuste para o modelo que você treinou
    model_path = 'models/hybrid_30m/lightgbm_model.txt'
    
    # 3. Inicializa o predictor
    predictor = TradingPredictor(
        lightgbm_path=model_path,
        config=config
    )
    
    # 4. Carrega dados históricos recentes
    # Você precisa de pelo menos ~50 candles para features técnicas
    df = pd.read_csv('data/usdjpy_history_30m.csv')
    
    # Converte timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Pega últimos 100 candles (mais do que suficiente para features)
    recent_candles = df.tail(100)
    
    # 5. Faz a predição
    result = predictor.predict(recent_candles)
    
    
    # 6. Exibe resultado
    print(f"\n📊 Resultado da Predição:")
    print(f"  🎯 Sinal: {result['signal']}")
    print(f"  📈 Retorno Previsto: {result['predicted_return']:.4f}% ({result['predicted_return']*100:.2f} basis points)")
    print(f"  � Acurácia Base do Modelo: {result.get('base_accuracy', result['confidence']):.2%}")
    print(f"  💯 Confiança Ajustada: {result['confidence']:.2%}")
    print(f"  💰 Preço Atual: {result['current_price']:.5f}")
    
    # 7. Interpreta o resultado
    print(f"\n📝 Interpretação:")
    base_acc = result.get('base_accuracy', result['confidence'])
    if result['signal'] == 'BUY':
        print(f"  ✅ COMPRAR - Modelo prevê alta de ~{result['predicted_return']*100:.2f}%")
        print(f"  📈 Probabilidade de acerto: {base_acc:.1%} (histórico do modelo)")
        print(f"  🎯 Força do sinal: {result['confidence']/base_acc:.1%}" if base_acc > 0 else "")
    elif result['signal'] == 'SELL':
        print(f"  ❌ VENDER - Modelo prevê queda de ~{abs(result['predicted_return'])*100:.2f}%")
        print(f"  📉 Probabilidade de acerto: {base_acc:.1%} (histórico do modelo)")
        print(f"  🎯 Força do sinal: {result['confidence']/base_acc:.1%}" if base_acc > 0 else "")
    else:
        print(f"  ⏸️  NEUTRO - Confiança insuficiente ({result['confidence']:.1%} < threshold)")
        print(f"  ℹ️  Acurácia base: {base_acc:.1%}, mas retorno previsto muito pequeno")
    
    return result

def example_from_dict():
    """
    Exemplo 2: Predição a partir de lista de dicionários.
    Útil quando você recebe dados de uma API.
    """
    print("\n" + "=" * 70)
    print("EXEMPLO 2: Predição a partir de API/Dicionários")
    print("=" * 70)
    
    # 1. Setup
    config = load_config('config_30m_optimized.yaml')
    model_path = 'models/hybrid_30m/lightgbm_model.txt'
    predictor = TradingPredictor(lightgbm_path=model_path, config=config)
    
    # 2. Simula dados recebidos de uma API
    # Em produção, você receberia isso de seu broker (MetaTrader, cTrader, etc)
    recent_data = [
        {
            'timestamp': '2024-01-01 00:00:00',
            'open': 148.50,
            'high': 148.75,
            'low': 148.40,
            'close': 148.65,
            'volume': 1000
        },
        {
            'timestamp': '2024-01-01 00:30:00',
            'open': 148.65,
            'high': 148.80,
            'low': 148.60,
            'close': 148.70,
            'volume': 1200
        },
        # ... adicione pelo menos 50 candles
    ]
    
    # Carrega dados reais do CSV para ter histórico suficiente
    df = pd.read_csv('data/usdjpy_history_30m.csv')
    recent_data = df.tail(100).to_dict('records')
    
    # 3. Faz predição
    result = predictor.predict_from_recent_data(recent_data)
    
    base_acc = result.get('base_accuracy', result['confidence'])
    print(f"\n📊 Resultado: {result['signal']}")
    print(f"   Acurácia Base: {base_acc:.1%} | Confiança Ajustada: {result['confidence']:.1%}")
    
    return result

def example_batch_prediction():
    """
    Exemplo 3: Predições em batch para backtesting.
    """
    print("\n" + "=" * 70)
    print("EXEMPLO 3: Predições em Batch (Backtesting)")
    print("=" * 70)
    
    # 1. Setup
    config = load_config('config_30m_optimized.yaml')
    model_path = 'models/hybrid_30m/lightgbm_model.txt'
    predictor = TradingPredictor(lightgbm_path=model_path, config=config)
    
    # 2. Carrega dados históricos
    df = pd.read_csv('data/usdjpy_history_30m.csv')
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 3. Faz predições em batch para os últimos 200 candles
    test_data = df.tail(200)
    predictions = predictor.batch_predict(test_data)
    
    # 4. Analisa resultados
    print(f"\n📊 Estatísticas das Predições:")
    print(f"  Total de predições: {len(predictions)}")
    print(f"  BUY signals: {(predictions['signal'] == 'BUY').sum()} ({(predictions['signal'] == 'BUY').sum()/len(predictions)*100:.1f}%)")
    print(f"  SELL signals: {(predictions['signal'] == 'SELL').sum()} ({(predictions['signal'] == 'SELL').sum()/len(predictions)*100:.1f}%)")
    print(f"  NEUTRAL signals: {(predictions['signal'] == 'NEUTRAL').sum()} ({(predictions['signal'] == 'NEUTRAL').sum()/len(predictions)*100:.1f}%)")
    print(f"  Confiança média: {predictions['confidence'].mean():.2%}")
    print(f"  Retorno previsto médio: {predictions['predicted_return'].mean():.4f}%")
    print(f"\n  ℹ️  Nota: A acurácia base do modelo (~{predictor.test_direction_acc:.1%}) é ajustada")
    print(f"     pela magnitude do retorno para gerar a confiança final.")
    
    # 5. Mostra últimas 10 predições
    print(f"\n📈 Últimas 10 predições:")
    print(predictions.tail(10).to_string(index=False))
    
    return predictions

def example_real_time_simulation():
    """
    Exemplo 4: Simulação de trading em tempo real.
    """
    print("\n" + "=" * 70)
    print("EXEMPLO 4: Simulação de Trading em Tempo Real")
    print("=" * 70)
    
    # 1. Setup
    config = load_config('config_30m_optimized.yaml')
    model_path = 'models/hybrid_30m/lightgbm_model.txt'
    predictor = TradingPredictor(lightgbm_path=model_path, config=config)
    
    # 2. Carrega dados
    df = pd.read_csv('data/usdjpy_history_30m.csv')
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 3. Simula 10 momentos diferentes no tempo
    print("\n🔄 Simulando decisões de trading ao longo do tempo...\n")
    
    window_size = 100  # Janela de histórico
    for i in range(10):
        # Pega janela deslizante
        start_idx = -(window_size + 10 - i)
        end_idx = -10 + i if i < 9 else None
        
        candles = df.iloc[start_idx:end_idx]
        
        # Faz predição
        result = predictor.predict(candles)
        
        # Exibe decisão
        timestamp = candles.iloc[-1]['timestamp'] if 'timestamp' in candles.columns else f"T-{10-i}"
        symbol = "🟢" if result['signal'] == 'BUY' else "🔴" if result['signal'] == 'SELL' else "⚪"
        base_acc = result.get('base_accuracy', result['confidence'])
        
        print(f"{symbol} {timestamp} | {result['signal']:7s} | "
              f"Base: {base_acc:5.1%} | Conf: {result['confidence']:5.1%} | "
              f"Retorno: {result['predicted_return']:+.4f}%")

def example_confidence_explanation():
    """
    Exemplo 5: Explicação da diferença entre acurácia base e confiança ajustada.
    """
    print("\n" + "=" * 70)
    print("EXEMPLO 5: Entendendo Acurácia Base vs Confiança Ajustada")
    print("=" * 70)
    
    # 1. Setup
    config = load_config('config_30m_optimized.yaml')
    model_path = 'models/hybrid_30m/lightgbm_model.txt'
    predictor = TradingPredictor(lightgbm_path=model_path, config=config)
    
    # 2. Mostra a acurácia base do modelo
    base_acc = getattr(predictor, 'test_direction_acc', 0.55)
    print(f"\n📊 Acurácia Base do Modelo: {base_acc:.2%}")
    print(f"   Isso significa que o modelo acerta a direção em {base_acc:.1%} dos casos")
    print(f"   (baseado nos dados de teste durante o treinamento)")
    
    # 3. Simula diferentes cenários
    print(f"\n📈 Como a Confiança é Ajustada pela Magnitude do Retorno:\n")
    print(f"{'Retorno Previsto':>18} | {'Magnitude':>10} | {'Conf. Ajustada':>15} | {'Interpretação':>30}")
    print("-" * 80)
    
    scenarios = [
        (0.0001, "Muito Pequeno"),
        (0.0005, "Pequeno"),
        (0.0010, "Médio"),
        (0.0020, "Grande"),
        (0.0050, "Muito Grande"),
    ]
    
    for ret, desc in scenarios:
        # Simula o cálculo de confiança (como será no predictor atualizado)
        magnitude_factor = min(abs(ret) * 100, 1.0)
        adjusted_conf = base_acc * magnitude_factor
        
        # Determina se seria um sinal válido
        min_conf_threshold = 0.40  # 40%
        valid = "✅ TRADE" if adjusted_conf >= min_conf_threshold else "❌ IGNORAR"
        
        print(f"{ret:>+18.4f} | {desc:>10s} | {adjusted_conf:>14.1%} | {valid:>30s}")
    
    print(f"\n💡 Interpretação:")
    print(f"  • Base Accuracy ({base_acc:.1%}) = Probabilidade histórica de acerto")
    print(f"  • Magnitude Factor = Quão forte é o movimento previsto")
    print(f"  • Confiança Ajustada = Base × Magnitude")
    print(f"  • Só opera quando Confiança Ajustada ≥ Threshold (ex: 40%)")
    
    print(f"\n🎯 Exemplo Prático:")
    print(f"  Se o modelo prevê retorno de +0.0020 (0.2%):")
    print(f"  • Magnitude Factor = min(0.002 × 100, 1.0) = 0.20 = 20%")
    print(f"  • Confiança = {base_acc:.2%} × 20% = {base_acc * 0.20:.1%}")
    print(f"  • Conclusão: {'OPERAR' if base_acc * 0.20 >= 0.40 else 'NÃO OPERAR'} (threshold 40%)")

def main():
    """Executa todos os exemplos."""
    try:
        # Exemplo 1: Predição única
        example_single_prediction()
        
        # Exemplo 2: Predição de dicionários (API)
        example_from_dict()
        
        # Exemplo 3: Batch prediction
        example_batch_prediction()
        
        # Exemplo 4: Simulação tempo real
        example_real_time_simulation()
        
        # Exemplo 5: Explicação de confiança
        example_confidence_explanation()
        
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        print("\n💡 Dicas:")
        print("  1. Verifique se o modelo existe em 'models/hybrid_30m/lightgbm_model.txt'")
        print("  2. Verifique se os dados existem em 'data/usdjpy_history_30m.csv'")
        print("  3. Ajuste os caminhos conforme necessário")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
