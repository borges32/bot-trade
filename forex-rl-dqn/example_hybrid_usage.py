"""
Exemplo de uso do sistema híbrido em Python.

Demonstra como usar o TradingPredictor para fazer previsões.
"""

import sys
from pathlib import Path
import pandas as pd
import yaml

root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

from src.inference.predictor import TradingPredictor


def example_usage():
    """Exemplo de uso do sistema híbrido."""
    
    print("="*60)
    print("EXEMPLO DE USO - Sistema Híbrido LightGBM + PPO")
    print("="*60)
    print()
    
    # 1. Carrega configuração
    config_file = root_dir / 'config_hybrid.yaml'
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # 2. Inicializa preditor
    print("Carregando modelos...")
    models_dir = root_dir / config['general']['models_dir']
    
    predictor = TradingPredictor(
        lightgbm_path=str(models_dir / 'lightgbm_model'),
        ppo_path=str(models_dir / 'ppo_model'),
        feature_config=config['features'],
        env_config=config['ppo']['env']
    )
    
    print("✅ Modelos carregados\n")
    
    # 3. Carrega dados de exemplo
    print("Carregando dados de exemplo...")
    data_file = root_dir / config['data']['train_file']
    df = pd.read_csv(data_file)
    
    # Renomeia colunas
    column_mapping = {
        config['data'].get('timestamp_col', 'timestamp'): 'timestamp',
        config['data'].get('open_col', 'open'): 'open',
        config['data'].get('high_col', 'high'): 'high',
        config['data'].get('low_col', 'low'): 'low',
        config['data'].get('close_col', 'close'): 'close',
        config['data'].get('volume_col', 'volume'): 'volume',
    }
    rename_map = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=rename_map)
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Pega últimos 100 candles
    recent_candles = df.tail(100)
    print(f"✅ Carregados {len(recent_candles)} candles recentes\n")
    
    # 4. Faz predição
    print("="*60)
    print("FAZENDO PREDIÇÃO")
    print("="*60)
    
    result = predictor.predict(
        candles=recent_candles,
        current_position=0,  # Sem posição
        deterministic=True
    )
    
    print(f"\n📊 RESULTADO DA PREDIÇÃO:")
    print(f"   Ação: {result['action_name'].upper()}")
    print(f"   Confiança: {result['confidence']*100:.1f}%")
    print(f"   Sinal LightGBM: {result['lightgbm_signal']:.4f}")
    print(f"\n💰 ESTADO ATUAL:")
    state = result['current_state']
    print(f"   Posição: {['SHORT', 'FLAT', 'LONG'][state['position']+1]}")
    print(f"   Balance: ${state['balance']:.2f}")
    print(f"   Equity: ${state['equity']:.2f}")
    print(f"   PnL Realizado: ${state['realized_pnl']:.2f}")
    print(f"   Retorno Total: {state['total_return']*100:.2f}%")
    print()
    
    # 5. Simula execução se confiança alta
    if result['confidence'] > 0.6:
        print("="*60)
        print("EXECUTANDO AÇÃO (Confiança > 60%)")
        print("="*60)
        
        current_price = recent_candles.iloc[-1]['close']
        
        exec_result = predictor.execute_action(
            action=result['action'],
            price=current_price
        )
        
        print(f"\n✅ Ação executada!")
        print(f"   Posição anterior: {['SHORT', 'FLAT', 'LONG'][exec_result['previous_position']+1]}")
        print(f"   Nova posição: {['SHORT', 'FLAT', 'LONG'][exec_result['new_position']+1]}")
        print(f"   Trade executado: {exec_result['trade_executed']}")
        if exec_result['trade_executed']:
            print(f"   PnL do trade: ${exec_result['pnl']:.2f}")
        print()
        
        # Estado após execução
        new_state = predictor.get_state()
        print(f"💰 ESTADO APÓS EXECUÇÃO:")
        print(f"   Posição: {['SHORT', 'FLAT', 'LONG'][new_state['position']+1]}")
        print(f"   Equity: ${new_state['equity']:.2f}")
        print(f"   PnL Realizado: ${new_state['realized_pnl']:.2f}")
        print()
    else:
        print("⚠️  Confiança baixa (<60%), não executando\n")
    
    # 6. Loop de simulação
    print("="*60)
    print("SIMULAÇÃO DE 10 DECISÕES")
    print("="*60)
    print()
    
    predictor.reset_state()  # Reseta para nova simulação
    
    # Usa últimos 150 candles e vai avançando
    simulation_data = df.tail(150).reset_index(drop=True)
    
    for i in range(10):
        # Janela de 100 candles
        start_idx = i * 5  # Avança 5 candles por vez
        window = simulation_data.iloc[start_idx:start_idx+100]
        
        if len(window) < 100:
            break
        
        # Predição
        result = predictor.predict(
            candles=window,
            deterministic=True
        )
        
        # Executa se confiança > 0.5
        if result['confidence'] > 0.5:
            current_price = window.iloc[-1]['close']
            predictor.execute_action(result['action'], current_price)
        
        # Mostra resultado
        state = predictor.get_state()
        print(f"Step {i+1:2d}: {result['action_name']:8s} "
              f"(conf: {result['confidence']:.2f}) | "
              f"Pos: {['S', 'F', 'L'][state['position']+1]} | "
              f"Equity: ${state['equity']:8.2f} | "
              f"Return: {state['total_return']*100:6.2f}%")
    
    print()
    
    # Resultado final
    final_state = predictor.get_state()
    print("="*60)
    print("RESULTADO FINAL DA SIMULAÇÃO")
    print("="*60)
    print(f"Balance inicial: ${config['ppo']['env']['initial_balance']:.2f}")
    print(f"Equity final: ${final_state['equity']:.2f}")
    print(f"PnL Total: ${final_state['realized_pnl']:.2f}")
    print(f"Retorno: {final_state['total_return']*100:.2f}%")
    print(f"Max Drawdown: {final_state['max_drawdown']*100:.2f}%")
    print("="*60)


if __name__ == '__main__':
    try:
        example_usage()
    except FileNotFoundError as e:
        print(f"\n❌ Erro: {e}")
        print("\nModelos não encontrados. Execute primeiro:")
        print("  ./train_hybrid.sh")
        print("ou")
        print("  python -m src.training.train_lightgbm")
        print("  python -m src.training.train_ppo")
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()
