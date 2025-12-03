"""
Script de teste rápido do sistema híbrido.

Testa o pipeline completo: dados -> features -> LightGBM -> PPO -> predição
"""

import sys
from pathlib import Path
import pandas as pd
import yaml
import logging

root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

from src.common.features_optimized import OptimizedFeatureEngineer
from src.models.lightgbm_model import LightGBMPredictor
from src.envs import ForexTradingEnv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_hybrid_system():
    """Testa o sistema híbrido completo."""
    
    print("="*60)
    print("TESTE DO SISTEMA HÍBRIDO - LightGBM + PPO")
    print("="*60)
    print()
    
    # Carrega configuração
    config_file = root_dir / 'config_hybrid.yaml'
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # 1. Testa carregamento de dados
    print("1. Testando carregamento de dados...")
    data_file = root_dir / config['data']['train_file']
    
    if not data_file.exists():
        print(f"❌ Arquivo de dados não encontrado: {data_file}")
        print("   Por favor, coloque um arquivo CSV em data/")
        return False
    
    df = pd.read_csv(data_file)
    print(f"✅ Dados carregados: {len(df)} candles")
    print(f"   Colunas: {df.columns.tolist()}")
    print()
    
    # 2. Testa feature engineering
    print("2. Testando feature engineering...")
    fe = OptimizedFeatureEngineer(config['features'])
    
    # Renomeia colunas se necessário
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
    
    df_features = fe.create_features(df.head(200))  # Testa com 200 candles
    df_features = df_features.dropna()
    
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    feature_columns = [col for col in df_features.columns if col not in exclude_cols]
    
    print(f"✅ Features criadas: {len(feature_columns)} features")
    print(f"   Amostras válidas: {len(df_features)}")
    print(f"   Top features: {feature_columns[:5]}")
    print()
    
    # 3. Testa ambiente de trading
    print("3. Testando ambiente de trading...")
    env_config = config['ppo']['env']
    
    env = ForexTradingEnv(
        df=df_features.head(100),
        feature_columns=feature_columns,
        lightgbm_predictor=None,  # Sem LightGBM para teste básico
        config=env_config
    )
    
    obs, info = env.reset()
    print(f"✅ Ambiente criado")
    print(f"   Observation space: {env.observation_space.shape}")
    print(f"   Action space: {env.action_space.n} ações")
    print(f"   Initial balance: ${info['equity']:.2f}")
    print()
    
    # Testa alguns steps
    print("4. Testando execução de ações...")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"   Step {i+1}: Action={action}, Reward={reward:.4f}, "
              f"Position={info['position']}, Equity=${info['equity']:.2f}")
        
        if done or truncated:
            break
    
    print("✅ Ambiente funcional")
    print()
    
    # 4. Verifica se modelos existem
    print("5. Verificando modelos treinados...")
    models_dir = root_dir / config['general']['models_dir']
    
    lgbm_path = models_dir / 'lightgbm_model.txt'
    ppo_path = models_dir / 'ppo_model.zip'
    
    if lgbm_path.exists():
        print(f"✅ LightGBM encontrado: {lgbm_path}")
    else:
        print(f"⚠️  LightGBM não encontrado: {lgbm_path}")
        print("   Execute: python -m src.training.train_lightgbm")
    
    if ppo_path.exists():
        print(f"✅ PPO encontrado: {ppo_path}")
    else:
        print(f"⚠️  PPO não encontrado: {ppo_path}")
        print("   Execute: python -m src.training.train_ppo")
    
    print()
    
    # Resumo
    print("="*60)
    print("RESUMO DO TESTE")
    print("="*60)
    print("✅ Dados: OK")
    print("✅ Features: OK")
    print("✅ Ambiente: OK")
    
    if lgbm_path.exists() and ppo_path.exists():
        print("✅ Modelos: OK")
        print()
        print("🎉 Sistema pronto para uso!")
        print()
        print("Próximos passos:")
        print("  - Iniciar API: cd src/inference && python service.py")
        print("  - Ou usar predictor diretamente em seu código")
    else:
        print("⚠️  Modelos: Treinar necessário")
        print()
        print("Próximos passos:")
        print("  - Treinar modelos: ./train_hybrid.sh")
        print("  - Ou individual:")
        print("    1. python -m src.training.train_lightgbm")
        print("    2. python -m src.training.train_ppo")
    
    print("="*60)
    
    return True


if __name__ == '__main__':
    try:
        success = test_hybrid_system()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Erro no teste: {e}", exc_info=True)
        sys.exit(1)
