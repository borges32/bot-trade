"""
Script para executar predições e salvar no Redis.
Pode ser executado manualmente ou via cron/scheduler.
"""

import yaml
import pandas as pd
import sys
import requests
from pathlib import Path

def main():
    """Executa predição usando a API."""
    
    # 1. Carrega dados históricos
    data_path = 'data/usdjpy_history_30m.csv'
    if not Path(data_path).exists():
        print(f"❌ Dados não encontrados: {data_path}")
        sys.exit(1)
    
    print(f"📊 Carregando dados de {data_path}...")
    df = pd.read_csv(data_path)
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 2. Pega últimos 100 candles
    recent_candles = df.tail(100)
    
    # 3. Converte para formato da API
    candles_list = []
    for _, row in recent_candles.iterrows():
        candle = {
            "timestamp": str(row['timestamp']),
            "open": float(row['open']),
            "high": float(row['high']),
            "low": float(row['low']),
            "close": float(row['close']),
            "volume": float(row.get('volume', 0))
        }
        candles_list.append(candle)
    
    # 4. Envia para API
    print("🔮 Enviando dados para API...")
    
    api_url = "http://localhost:8000/api/prediction"
    payload = {
        "candles": candles_list,
        "current_price": candles_list[-1]['close']
    }
    
    try:
        response = requests.post(api_url, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        # 5. Exibe resultado
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
        print("🌐 Acesse o dashboard em: http://localhost:3000")
        print("🔗 API endpoint: http://localhost:8000/api/prediction/latest")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Erro: Não foi possível conectar à API")
        print("💡 Certifique-se de que a API está rodando:")
        print("   docker-compose up -d")
        print("   ou")
        print("   python api_server.py")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"\n❌ Erro HTTP: {e}")
        if hasattr(e, 'response'):
            print(f"   Detalhes: {e.response.text}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
