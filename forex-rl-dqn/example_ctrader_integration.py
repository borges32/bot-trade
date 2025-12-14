"""
Exemplo de integração com cTrader para enviar candles e receber predições.

Este script demonstra como enviar dados de candles em tempo real
para a API e receber predições de trading.
"""

import requests
import json
from datetime import datetime, timedelta

# URL da API
API_URL = "http://localhost:8000"

def send_candles_for_prediction(candles_data, current_price=None):
    """
    Envia candles para a API e recebe predição.
    
    Args:
        candles_data: Lista de dicionários com dados dos candles
        current_price: Preço atual (opcional)
    
    Returns:
        Resposta da API com a predição
    """
    endpoint = f"{API_URL}/api/prediction"
    
    payload = {
        "candles": candles_data,
    }
    
    if current_price is not None:
        payload["current_price"] = current_price
    
    try:
        print(f"📤 Enviando {len(candles_data)} candles para predição...")
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        print("\n✅ Predição Recebida:")
        print(f"  🎯 Sinal: {result['signal']}")
        print(f"  📈 Retorno Previsto: {result['predicted_return']:.4f}%")
        print(f"  📊 Acurácia Base: {result.get('base_accuracy', 0):.2%}")
        print(f"  💯 Confiança Ajustada: {result['confidence']:.2%}")
        print(f"  💰 Preço Atual: {result['current_price']:.5f}")
        print(f"  ⏰ Timestamp: {result.get('timestamp', 'N/A')}")
        
        return result
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Erro na requisição: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"   Detalhes: {e.response.text}")
        return None


def example_with_dummy_data():
    """
    Exemplo 1: Enviando dados de exemplo.
    """
    print("=" * 70)
    print("EXEMPLO 1: Dados de Exemplo")
    print("=" * 70)
    
    # Gera 100 candles de exemplo
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    base_price = 148.50
    
    candles = []
    for i in range(100):
        # Simula variação de preço
        import random
        variation = random.uniform(-0.5, 0.5)
        
        open_price = base_price + variation
        high_price = open_price + random.uniform(0, 0.3)
        low_price = open_price - random.uniform(0, 0.3)
        close_price = open_price + random.uniform(-0.2, 0.2)
        
        candle = {
            "timestamp": (base_time + timedelta(minutes=30 * i)).isoformat(),
            "open": round(open_price, 5),
            "high": round(high_price, 5),
            "low": round(low_price, 5),
            "close": round(close_price, 5),
            "volume": random.randint(500, 2000)
        }
        candles.append(candle)
        base_price = close_price
    
    # Envia para API
    result = send_candles_for_prediction(candles, current_price=base_price)
    
    return result


def example_from_csv():
    """
    Exemplo 2: Carregando dados reais de CSV.
    """
    print("\n" + "=" * 70)
    print("EXEMPLO 2: Dados Reais do CSV")
    print("=" * 70)
    
    import pandas as pd
    
    # Carrega dados históricos
    df = pd.read_csv('data/usdjpy_history_30m.csv')
    
    # Pega últimos 100 candles
    recent_df = df.tail(100)
    
    # Converte para formato da API
    candles = []
    for _, row in recent_df.iterrows():
        candle = {
            "timestamp": row['timestamp'],
            "open": float(row['open']),
            "high": float(row['high']),
            "low": float(row['low']),
            "close": float(row['close']),
            "volume": float(row.get('volume', 0))
        }
        candles.append(candle)
    
    # Usa close do último candle como preço atual
    current_price = candles[-1]['close']
    
    # Envia para API
    result = send_candles_for_prediction(candles, current_price=current_price)
    
    return result


def example_ctrader_format():
    """
    Exemplo 3: Formato típico do cTrader.
    
    Este é o formato que você deve usar ao integrar com cTrader.
    """
    print("\n" + "=" * 70)
    print("EXEMPLO 3: Formato cTrader")
    print("=" * 70)
    
    # Formato típico que você receberia do cTrader via API/webhook
    ctrader_data = {
        "symbol": "USDJPY",
        "timeframe": "M30",
        "candles": [
            # Últimos 100 candles do par USDJPY 30M
            # Normalmente você pegaria isso da API do cTrader
            {
                "timestamp": "2024-12-14T00:00:00Z",
                "open": 148.500,
                "high": 148.750,
                "low": 148.400,
                "close": 148.650,
                "volume": 1234
            },
            {
                "timestamp": "2024-12-14T00:30:00Z",
                "open": 148.650,
                "high": 148.800,
                "low": 148.600,
                "close": 148.720,
                "volume": 1456
            },
            # ... adicione mais candles aqui (mínimo 50, recomendado 100)
        ]
    }
    
    # Para este exemplo, vamos usar dados reais do CSV
    import pandas as pd
    df = pd.read_csv('data/usdjpy_history_30m.csv')
    recent_df = df.tail(100)
    
    ctrader_data["candles"] = []
    for _, row in recent_df.iterrows():
        ctrader_data["candles"].append({
            "timestamp": row['timestamp'],
            "open": float(row['open']),
            "high": float(row['high']),
            "low": float(row['low']),
            "close": float(row['close']),
            "volume": float(row.get('volume', 0))
        })
    
    print(f"📊 Símbolo: {ctrader_data['symbol']}")
    print(f"⏱️  Timeframe: {ctrader_data['timeframe']}")
    print(f"📈 Candles: {len(ctrader_data['candles'])}")
    
    # Envia apenas os candles (API não precisa do símbolo e timeframe)
    result = send_candles_for_prediction(ctrader_data["candles"])
    
    return result


def check_latest_prediction():
    """
    Consulta a última predição salva no Redis.
    """
    print("\n" + "=" * 70)
    print("Consultando Última Predição no Redis")
    print("=" * 70)
    
    try:
        response = requests.get(f"{API_URL}/api/prediction/latest")
        response.raise_for_status()
        
        result = response.json()
        
        print(f"  🎯 Sinal: {result['signal']}")
        print(f"  📈 Retorno: {result['predicted_return']:.4f}%")
        print(f"  💯 Confiança: {result['confidence']:.2%}")
        print(f"  💰 Preço: {result['current_price']:.5f}")
        
        return result
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Erro: {e}")
        return None


def main():
    """Executa exemplos de integração."""
    
    # Exemplo 1: Dados simulados
    example_with_dummy_data()
    
    # Exemplo 2: Dados reais do CSV
    try:
        example_from_csv()
    except FileNotFoundError:
        print("\n⚠️  Arquivo CSV não encontrado, pulando exemplo 2")
    
    # Exemplo 3: Formato cTrader
    try:
        example_ctrader_format()
    except FileNotFoundError:
        print("\n⚠️  Arquivo CSV não encontrado, pulando exemplo 3")
    
    # Consulta última predição
    check_latest_prediction()
    
    print("\n" + "=" * 70)
    print("✅ Exemplos concluídos!")
    print("=" * 70)
    print("\n💡 Integração cTrader:")
    print("  1. Configure webhook no cTrader para chamar http://seu-servidor:8000/api/prediction")
    print("  2. Envie JSON com lista de candles (mínimo 50)")
    print("  3. Receba predição em tempo real")
    print("  4. Dashboard atualiza automaticamente em http://localhost:3000")


if __name__ == '__main__':
    main()
