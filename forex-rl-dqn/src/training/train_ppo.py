"""
Script de treinamento do agente PPO.

Este script treina um agente PPO para operar no mercado Forex,
usando sinais do LightGBM como parte do estado.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import logging
from datetime import datetime

# Adiciona path raiz ao PYTHONPATH
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from src.common.features_optimized import OptimizedFeatureEngineer
from src.models.lightgbm_model import LightGBMPredictor
from src.models.ppo_agent import PPOTradingAgent, make_trading_env
from src.envs import ForexTradingEnv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = 'config_hybrid.yaml') -> dict:
    """Carrega configuração."""
    config_file = root_dir / config_path
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_and_prepare_data(config: dict) -> pd.DataFrame:
    """Carrega e prepara dados históricos."""
    data_config = config['data']
    
    data_file = root_dir / data_config['train_file']
    logger.info(f"Loading data from {data_file}")
    
    df = pd.read_csv(data_file)
    
    # Renomeia colunas
    column_mapping = {
        data_config.get('timestamp_col', 'timestamp'): 'timestamp',
        data_config.get('open_col', 'open'): 'open',
        data_config.get('high_col', 'high'): 'high',
        data_config.get('low_col', 'low'): 'low',
        data_config.get('close_col', 'close'): 'close',
        data_config.get('volume_col', 'volume'): 'volume',
    }
    
    rename_map = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=rename_map)
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
    
    logger.info(f"Loaded {len(df)} candles")
    
    return df


def create_features(df: pd.DataFrame, config: dict) -> tuple:
    """Cria features técnicas."""
    feature_config = config['features']
    
    logger.info("Creating technical features...")
    
    fe = OptimizedFeatureEngineer(feature_config)
    df_features = fe.create_features(df)
    
    initial_rows = len(df_features)
    df_features = df_features.dropna().reset_index(drop=True)
    logger.info(f"Removed {initial_rows - len(df_features)} rows with NaN")
    
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    feature_columns = [col for col in df_features.columns if col not in exclude_cols]
    
    logger.info(f"Created {len(feature_columns)} features")
    
    return df_features, feature_columns


def split_data(df: pd.DataFrame, config: dict) -> tuple:
    """Divide dados temporalmente."""
    data_config = config['data']
    val_split = data_config.get('val_split', 0.15)
    test_split = data_config.get('test_split', 0.15)
    
    n = len(df)
    
    test_start = int(n * (1 - test_split))
    val_start = int(n * (1 - test_split - val_split))
    
    train_df = df.iloc[:val_start].copy()
    val_df = df.iloc[val_start:test_start].copy()
    test_df = df.iloc[test_start:].copy()
    
    logger.info(f"Data split:")
    logger.info(f"  Train: {len(train_df)} samples")
    logger.info(f"  Val:   {len(val_df)} samples")
    logger.info(f"  Test:  {len(test_df)} samples")
    
    return train_df, val_df, test_df


def load_lightgbm_model(config: dict) -> LightGBMPredictor:
    """
    Carrega modelo LightGBM treinado.
    
    Args:
        config: Configuração
        
    Returns:
        LightGBMPredictor carregado
    """
    models_dir = root_dir / config['general']['models_dir']
    model_path = models_dir / 'lightgbm_model'
    
    if not model_path.with_suffix('.txt').exists():
        logger.warning("LightGBM model not found. PPO will train without LightGBM signals.")
        return None
    
    logger.info(f"Loading LightGBM model from {model_path}")
    
    predictor = LightGBMPredictor(config['lightgbm'])
    predictor.load(model_path)
    
    logger.info("LightGBM model loaded successfully")
    
    return predictor


def train_ppo(config_path: str = 'config_hybrid.yaml') -> dict:
    """
    Função principal de treinamento do PPO.
    
    Args:
        config_path: Caminho do arquivo de configuração
        
    Returns:
        Dicionário com métricas e caminhos
    """
    # Carrega configuração
    config = load_config(config_path)
    
    # Cria diretório de modelos
    models_dir = root_dir / config['general']['models_dir']
    models_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = root_dir / config['general']['log_dir']
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Carrega dados
    df = load_and_prepare_data(config)
    
    # Cria features
    df_features, feature_columns = create_features(df, config)
    
    # Divide dados
    train_df, val_df, test_df = split_data(df_features, config)
    
    # Carrega modelo LightGBM (se disponível)
    lightgbm_predictor = load_lightgbm_model(config)
    
    # Configuração do ambiente
    env_config = config['ppo']['env']
    
    # Cria ambientes
    logger.info("Creating training environment...")
    train_env = make_trading_env(
        df=train_df,
        feature_columns=feature_columns,
        lightgbm_predictor=lightgbm_predictor,
        config=env_config,
        monitor_path=log_dir / 'train'
    )
    
    logger.info("Creating validation environment...")
    val_env = make_trading_env(
        df=val_df,
        feature_columns=feature_columns,
        lightgbm_predictor=lightgbm_predictor,
        config=env_config,
        monitor_path=log_dir / 'val'
    )
    
    # Cria agente PPO
    logger.info("Creating PPO agent...")
    ppo_config = config['ppo']
    ppo_config['seed'] = config['general']['seed']
    
    agent = PPOTradingAgent(train_env, ppo_config)
    
    # Parâmetros de treinamento
    training_config = ppo_config['training']
    
    # Treina agente
    logger.info("Training PPO agent...")
    logger.info(f"Total timesteps: {training_config['total_timesteps']}")
    
    train_metrics = agent.train(
        total_timesteps=training_config['total_timesteps'],
        eval_env=val_env,
        eval_freq=training_config.get('eval_freq', 10000),
        n_eval_episodes=training_config.get('n_eval_episodes', 10),
        save_freq=training_config.get('save_freq', 50000),
        save_path=models_dir
    )
    
    # Salva modelo final
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = models_dir / f'ppo_model_{timestamp}'
    agent.save(model_path)
    
    # Salva também como latest
    latest_path = models_dir / 'ppo_model'
    agent.save(latest_path)
    
    # Avalia em conjunto de teste
    logger.info("Evaluating on test set...")
    test_env = make_trading_env(
        df=test_df,
        feature_columns=feature_columns,
        lightgbm_predictor=lightgbm_predictor,
        config=env_config
    )
    
    test_metrics = agent.evaluate(test_env, n_episodes=10, deterministic=True)
    
    logger.info(f"Test metrics: {test_metrics}")
    
    # Salva métricas
    all_metrics = {
        'train': train_metrics,
        'test': test_metrics
    }
    
    metrics_file = models_dir / f'ppo_metrics_{timestamp}.yaml'
    with open(metrics_file, 'w') as f:
        yaml.dump(all_metrics, f)
    
    logger.info(f"\nTraining completed!")
    logger.info(f"Model saved to: {latest_path}")
    logger.info(f"Metrics saved to: {metrics_file}")
    
    # Limpa ambientes
    train_env.close()
    val_env.close()
    test_env.close()
    
    return {
        'metrics': all_metrics,
        'model_path': str(latest_path),
        'feature_columns': feature_columns
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train PPO agent for Forex trading')
    parser.add_argument(
        '--config',
        type=str,
        default='config_hybrid.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    result = train_ppo(args.config)
    
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Model: {result['model_path']}")
    print(f"\nMetrics:")
    for split, split_metrics in result['metrics'].items():
        print(f"\n{split.upper()}:")
        for metric, value in split_metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
    print("="*60)
