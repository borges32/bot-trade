"""
Teste para validar configurações específicas de 15m e 30m.
"""

import yaml
import pandas as pd
import numpy as np
from src.common.features_optimized import OptimizedFeatureEngineer

def load_config(config_path):
    """Carrega arquivo YAML de configuração."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_synthetic_data(n=1000):
    """Cria dados sintéticos OHLCV."""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=n, freq='30min')
    close_prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    
    return pd.DataFrame({
        'timestamp': dates,
        'open': close_prices + np.random.randn(n) * 0.2,
        'high': close_prices + np.abs(np.random.randn(n)) * 0.5,
        'low': close_prices - np.abs(np.random.randn(n)) * 0.5,
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, n)
    })

def test_config(config_path, config_name):
    """Testa uma configuração específica."""
    print(f"\n{'='*70}")
    print(f"TESTE: {config_name}")
    print(f"{'='*70}")
    
    # Carrega config
    config = load_config(config_path)
    print(f"\n📄 Configuração carregada: {config_path}")
    print(f"   Timeframe: {config['general'].get('timeframe', 'N/A')}")
    print(f"   Log dir: {config['general']['log_dir']}")
    print(f"   Models dir: {config['general']['models_dir']}")
    
    # Features config
    features_cfg = config['features']
    print(f"\n🔧 Períodos de Indicadores:")
    print(f"   RSI: {features_cfg.get('rsi_period', 'N/A')}")
    print(f"   EMA Fast: {features_cfg.get('ema_fast', 'N/A')}")
    print(f"   EMA Slow: {features_cfg.get('ema_slow', 'N/A')}")
    print(f"   MACD: {features_cfg.get('macd_fast', 'N/A')}, {features_cfg.get('macd_slow', 'N/A')}, {features_cfg.get('macd_signal', 'N/A')}")
    print(f"   BB: {features_cfg.get('bb_period', 'N/A')} períodos, {features_cfg.get('bb_std', 'N/A')} desvios")
    print(f"   ATR: {features_cfg.get('atr_period', 'N/A')}")
    print(f"   Momentum: {features_cfg.get('momentum_periods', 'N/A')}")
    print(f"   Stochastic: %K={features_cfg.get('stoch_k', 'N/A')}, %D={features_cfg.get('stoch_d', 'N/A')}")
    print(f"   ADX: {features_cfg.get('adx_period', 'N/A')}")
    print(f"   SMAs: {features_cfg.get('sma_periods', 'N/A')}")
    print(f"   Returns: {features_cfg.get('return_periods', 'N/A')}")
    
    # PPO config
    ppo_cfg = config['ppo']
    print(f"\n🎮 Configuração PPO:")
    print(f"   Learning rate: {ppo_cfg['params']['learning_rate']}")
    print(f"   Gamma: {ppo_cfg['params']['gamma']}")
    print(f"   Entropy coef: {ppo_cfg['params']['ent_coef']}")
    print(f"   Stop loss: {ppo_cfg['env']['stop_loss_pct']*100}%")
    print(f"   Take profit: {ppo_cfg['env']['take_profit_pct']*100}%")
    
    # LightGBM config
    lgb_cfg = config['lightgbm']
    print(f"\n🌳 Configuração LightGBM:")
    print(f"   Prediction horizon: {lgb_cfg['prediction_horizon']} candles")
    print(f"   Classification threshold: {lgb_cfg.get('classification_threshold', 0)}%")
    
    # Testa feature engineering
    print(f"\n🔄 Testando Feature Engineering...")
    df = create_synthetic_data(n=500)
    engineer = OptimizedFeatureEngineer(config)
    
    print(f"   DataFrame entrada: {df.shape}")
    df_features = engineer.create_features(df)
    print(f"   DataFrame saída: {df_features.shape}")
    
    # Valida indicadores
    print(f"\n✅ Validação de Indicadores:")
    indicators = ['rsi', 'ema_fast', 'ema_slow', 'bb_upper', 'bb_middle', 'bb_lower',
                  'macd', 'macd_signal', 'atr', 'momentum_10', 'momentum_20',
                  'volatility', 'volume_ma', 'stoch_k', 'adx']
    
    for ind in indicators:
        has_it = ind in df_features.columns
        if has_it:
            valid_count = df_features[ind].notna().sum()
            print(f"   {ind}: ✓ ({valid_count}/{len(df_features)} valores válidos)")
        else:
            print(f"   {ind}: ✗ FALTANDO!")
    
    # Verifica SMAs configuráveis
    print(f"\n📊 SMAs Configuráveis:")
    for period in features_cfg.get('sma_periods', []):
        sma_col = f'sma_{period}'
        has_it = sma_col in df_features.columns
        print(f"   SMA_{period}: {'✓' if has_it else '✗'}")
    
    # Verifica retornos configuráveis
    print(f"\n📈 Retornos Configuráveis:")
    for period in features_cfg.get('return_periods', []):
        ret_col = f'return_{period}'
        has_it = ret_col in df_features.columns
        print(f"   Return_{period}: {'✓' if has_it else '✗'}")
    
    return df_features.shape[1]  # Retorna número de features

# Executa testes
print("\n" + "="*70)
print("TESTE DE CONFIGURAÇÕES ESPECÍFICAS POR TIMEFRAME")
print("="*70)

n_features_15m = test_config('config_hybrid_15m.yaml', 'CONFIG 15 MINUTOS')
n_features_30m = test_config('config_hybrid_30m.yaml', 'CONFIG 30 MINUTOS')

# Resumo comparativo
print(f"\n{'='*70}")
print("RESUMO COMPARATIVO")
print(f"{'='*70}")

print(f"\n📊 15m vs 30m:")
print(f"   Total features 15m: {n_features_15m}")
print(f"   Total features 30m: {n_features_30m}")
print(f"   Diferença: {abs(n_features_15m - n_features_30m)} features")

# Carrega configs para comparação
config_15m = load_config('config_hybrid_15m.yaml')
config_30m = load_config('config_hybrid_30m.yaml')

print(f"\n🔧 Diferenças nos Períodos:")
print(f"   {'Indicador':<20} {'15m':<10} {'30m':<10}")
print(f"   {'-'*40}")

indicators_compare = [
    ('RSI', 'rsi_period'),
    ('EMA Fast', 'ema_fast'),
    ('EMA Slow', 'ema_slow'),
    ('MACD Fast', 'macd_fast'),
    ('MACD Slow', 'macd_slow'),
    ('MACD Signal', 'macd_signal'),
    ('BB Period', 'bb_period'),
    ('ATR', 'atr_period'),
    ('Stochastic %K', 'stoch_k'),
    ('ADX', 'adx_period')
]

for name, key in indicators_compare:
    val_15m = config_15m['features'].get(key, 'N/A')
    val_30m = config_30m['features'].get(key, 'N/A')
    print(f"   {name:<20} {str(val_15m):<10} {str(val_30m):<10}")

print(f"\n🎯 Diferenças PPO:")
ppo_15m = config_15m['ppo']
ppo_30m = config_30m['ppo']

print(f"   {'Parâmetro':<25} {'15m':<15} {'30m':<15}")
print(f"   {'-'*55}")
print(f"   {'Learning Rate':<25} {ppo_15m['params']['learning_rate']:<15} {ppo_30m['params']['learning_rate']:<15}")
print(f"   {'Gamma':<25} {ppo_15m['params']['gamma']:<15} {ppo_30m['params']['gamma']:<15}")
print(f"   {'Entropy Coef':<25} {ppo_15m['params']['ent_coef']:<15} {ppo_30m['params']['ent_coef']:<15}")
print(f"   {'Stop Loss %':<25} {ppo_15m['env']['stop_loss_pct']*100:<15} {ppo_30m['env']['stop_loss_pct']*100:<15}")
print(f"   {'Take Profit %':<25} {ppo_15m['env']['take_profit_pct']*100:<15} {ppo_30m['env']['take_profit_pct']*100:<15}")
print(f"   {'Risk Penalty':<25} {ppo_15m['env']['risk_penalty_lambda']:<15} {ppo_30m['env']['risk_penalty_lambda']:<15}")

print(f"\n🌳 Diferenças LightGBM:")
lgb_15m = config_15m['lightgbm']
lgb_30m = config_30m['lightgbm']

print(f"   Prediction Horizon 15m: {lgb_15m['prediction_horizon']} candles (= 1 hora)")
print(f"   Prediction Horizon 30m: {lgb_30m['prediction_horizon']} candles (= 3 horas)")

print(f"\n{'='*70}")
print("✅ TESTE COMPLETO!")
print(f"{'='*70}")
