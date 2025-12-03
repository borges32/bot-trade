"""
Cliente Python para a API de Trading Híbrida.

Demonstra como integrar o sistema em um bot de trading real.
"""

import requests
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradingAPIClient:
    """
    Cliente para a API de trading híbrida LightGBM + PPO.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Inicializa o cliente.
        
        Args:
            base_url: URL base da API
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
        # Verifica se API está disponível
        self._check_health()
    
    def _check_health(self):
        """Verifica se API está saudável."""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            logger.info("✅ API está saudável e respondendo")
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ API não está disponível: {e}")
            raise
    
    def get_signal(
        self,
        candles: List[Dict],
        current_position: int = 0,
        deterministic: bool = True
    ) -> Dict:
        """
        Obtém sinal de trading.
        
        Args:
            candles: Lista de dicionários com OHLCV
            current_position: Posição atual (-1, 0, 1)
            deterministic: Se True, usa política determinística
            
        Returns:
            Dicionário com sinal
        """
        payload = {
            'candles': candles,
            'current_position': current_position,
            'deterministic': deterministic
        }
        
        response = self.session.post(
            f"{self.base_url}/signal",
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        
        return response.json()
    
    def execute_action(self, action: int, price: float) -> Dict:
        """
        Executa uma ação.
        
        Args:
            action: 0=neutro, 1=comprar, 2=vender
            price: Preço de execução
            
        Returns:
            Resultado da execução
        """
        payload = {
            'action': action,
            'price': price
        }
        
        response = self.session.post(
            f"{self.base_url}/execute",
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        
        return response.json()
    
    def get_state(self) -> Dict:
        """
        Obtém estado atual da conta.
        
        Returns:
            Estado atual
        """
        response = self.session.get(f"{self.base_url}/state", timeout=10)
        response.raise_for_status()
        
        return response.json()
    
    def reset_state(self) -> Dict:
        """
        Reseta o estado.
        
        Returns:
            Novo estado
        """
        response = self.session.post(f"{self.base_url}/reset", timeout=10)
        response.raise_for_status()
        
        return response.json()


def load_candles_from_csv(filepath: str, n_candles: int = 100) -> List[Dict]:
    """
    Carrega candles de um CSV.
    
    Args:
        filepath: Caminho do CSV
        n_candles: Número de candles recentes
        
    Returns:
        Lista de dicionários
    """
    df = pd.read_csv(filepath)
    
    # Pega últimos N candles
    recent = df.tail(n_candles)
    
    # Converte para formato esperado
    candles = []
    for _, row in recent.iterrows():
        candle = {
            'timestamp': str(row.get('timestamp', row.name)),
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row.get('volume', 0))
        }
        candles.append(candle)
    
    return candles


def example_single_prediction():
    """Exemplo de predição única."""
    
    print("="*60)
    print("EXEMPLO 1: Predição Única")
    print("="*60)
    print()
    
    # Cria cliente
    client = TradingAPIClient()
    
    # Carrega candles
    print("Carregando dados...")
    candles = load_candles_from_csv('data/usdjpy_history_30m.csv', n_candles=100)
    print(f"✅ {len(candles)} candles carregados")
    print()
    
    # Solicita sinal
    print("Solicitando sinal...")
    signal = client.get_signal(candles, current_position=0)
    
    # Exibe resultado
    print(f"\n📊 SINAL RECEBIDO:")
    print(f"   Ação: {signal['action_name'].upper()}")
    print(f"   Confiança: {signal['confidence']*100:.1f}%")
    print(f"   Sinal LightGBM: {signal['lightgbm_signal']:.4f}")
    
    state = signal['current_state']
    print(f"\n💰 ESTADO:")
    print(f"   Equity: ${state['equity']:.2f}")
    print(f"   Retorno: {state['total_return']*100:.2f}%")
    print()


def example_trading_loop():
    """Exemplo de loop de trading simulado."""
    
    print("="*60)
    print("EXEMPLO 2: Loop de Trading Simulado")
    print("="*60)
    print()
    
    # Cria cliente
    client = TradingAPIClient()
    
    # Reseta estado
    client.reset_state()
    
    # Carrega dados
    df = pd.read_csv('data/usdjpy_history_30m.csv')
    
    # Parâmetros
    MIN_CONFIDENCE = 0.6  # Confiança mínima para operar
    LOOKBACK = 100  # Candles para análise
    
    print(f"Parâmetros:")
    print(f"  Confiança mínima: {MIN_CONFIDENCE*100:.0f}%")
    print(f"  Lookback: {LOOKBACK} candles")
    print()
    
    # Simula 20 decisões
    print("Iniciando loop de trading...\n")
    
    decisions = []
    
    for i in range(20):
        # Janela de candles (simula dados recentes)
        start_idx = len(df) - LOOKBACK - (20 - i) * 5
        end_idx = start_idx + LOOKBACK
        
        if start_idx < 0:
            break
        
        window = df.iloc[start_idx:end_idx]
        candles = []
        
        for _, row in window.iterrows():
            candle = {
                'timestamp': str(row.get('timestamp', row.name)),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row.get('volume', 0))
            }
            candles.append(candle)
        
        # Obtém sinal
        signal = client.get_signal(candles)
        
        # Decide executar baseado em confiança
        execute = signal['confidence'] >= MIN_CONFIDENCE
        
        current_price = candles[-1]['close']
        
        if execute:
            # Executa ação
            exec_result = client.execute_action(signal['action'], current_price)
            
        # Estado atual
        state = client.get_state()
        
        # Log
        icon = "✅" if execute else "⏸️"
        print(f"{icon} Step {i+1:2d}: {signal['action_name']:8s} "
              f"(conf: {signal['confidence']:.2f}) | "
              f"Exec: {str(execute):5s} | "
              f"Equity: ${state['equity']:8.2f} | "
              f"Return: {state['total_return']*100:6.2f}%")
        
        decisions.append({
            'step': i+1,
            'action': signal['action_name'],
            'confidence': signal['confidence'],
            'executed': execute,
            'equity': state['equity'],
            'return': state['total_return']
        })
        
        # Simula delay
        time.sleep(0.1)
    
    # Resultado final
    final_state = client.get_state()
    
    print()
    print("="*60)
    print("RESULTADO FINAL")
    print("="*60)
    print(f"Equity Final: ${final_state['equity']:.2f}")
    print(f"PnL Total: ${final_state['realized_pnl']:.2f}")
    print(f"Retorno: {final_state['total_return']*100:.2f}%")
    print(f"Max Drawdown: {final_state['max_drawdown']*100:.2f}%")
    print()
    
    # Estatísticas
    executed = sum(1 for d in decisions if d['executed'])
    print(f"Estatísticas:")
    print(f"  Total de decisões: {len(decisions)}")
    print(f"  Ações executadas: {executed}")
    print(f"  Taxa de execução: {executed/len(decisions)*100:.1f}%")
    print()


def example_integration_with_broker():
    """
    Exemplo de integração com broker (pseudocódigo).
    
    NOTA: Este é um exemplo conceitual. Você precisará adaptar
    para a API específica do seu broker (cTrader, MT5, etc.)
    """
    
    print("="*60)
    print("EXEMPLO 3: Integração com Broker (Pseudocódigo)")
    print("="*60)
    print()
    
    print("""
    # Pseudocódigo para integração real:
    
    from ctrader_api import CTraderAPI  # Exemplo
    
    # Inicializa APIs
    trading_api = TradingAPIClient('http://localhost:8000')
    broker_api = CTraderAPI(account_id='...', api_key='...')
    
    # Loop de trading
    while True:
        # 1. Obtém candles recentes do broker
        candles = broker_api.get_candles('USDJPY', timeframe='30m', count=100)
        
        # 2. Solicita sinal
        signal = trading_api.get_signal(candles)
        
        # 3. Verifica confiança
        if signal['confidence'] >= 0.7:
            # 4. Obtém posição atual do broker
            position = broker_api.get_position('USDJPY')
            
            # 5. Executa no broker se necessário
            if signal['action_name'] == 'comprar' and position != 1:
                broker_api.close_position('USDJPY')  # Fecha qualquer posição
                broker_api.open_long('USDJPY', volume=0.1)
                
                # Atualiza API local
                current_price = candles[-1]['close']
                trading_api.execute_action(signal['action'], current_price)
                
            elif signal['action_name'] == 'vender' and position != -1:
                broker_api.close_position('USDJPY')
                broker_api.open_short('USDJPY', volume=0.1)
                
                current_price = candles[-1]['close']
                trading_api.execute_action(signal['action'], current_price)
                
            elif signal['action_name'] == 'neutro' and position != 0:
                broker_api.close_position('USDJPY')
                
                current_price = candles[-1]['close']
                trading_api.execute_action(signal['action'], current_price)
        
        # 6. Aguarda próximo candle
        time.sleep(30 * 60)  # 30 minutos
    """)
    print()
    print("⚠️  IMPORTANTE: Adapte para a API do seu broker específico!")
    print()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Cliente da API de Trading')
    parser.add_argument(
        '--example',
        type=int,
        choices=[1, 2, 3],
        default=1,
        help='Exemplo a executar (1=predição única, 2=loop, 3=integração)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.example == 1:
            example_single_prediction()
        elif args.example == 2:
            example_trading_loop()
        elif args.example == 3:
            example_integration_with_broker()
            
    except requests.exceptions.ConnectionError:
        print("\n❌ Erro: API não está rodando!")
        print("\nInicie a API primeiro:")
        print("  cd src/inference")
        print("  python service.py")
        print()
    except FileNotFoundError as e:
        print(f"\n❌ Erro: {e}")
        print("\nVerifique se o arquivo de dados existe em data/")
        print()
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()
