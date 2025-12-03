"""
Script de treinamento do modelo LightGBM.

Este script treina um modelo supervisionado (LightGBM) para prever
a direção ou retorno futuro do preço no mercado Forex.
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
    """
    Carrega e prepara dados históricos.
    
    Args:
        config: Configuração
        
    Returns:
        DataFrame com OHLCV + features
    """
    data_config = config['data']
    
    # Carrega CSV
    data_file = root_dir / data_config['train_file']
    logger.info(f"Loading data from {data_file}")
    
    df = pd.read_csv(data_file)
    
    # Renomeia colunas se necessário
    column_mapping = {
        data_config.get('timestamp_col', 'timestamp'): 'timestamp',
        data_config.get('open_col', 'open'): 'open',
        data_config.get('high_col', 'high'): 'high',
        data_config.get('low_col', 'low'): 'low',
        data_config.get('close_col', 'close'): 'close',
        data_config.get('volume_col', 'volume'): 'volume',
    }
    
    # Só renomeia se a coluna existe
    rename_map = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=rename_map)
    
    # Converte timestamp se necessário
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
    
    logger.info(f"Loaded {len(df)} candles")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    return df


def create_features(df: pd.DataFrame, config: dict) -> tuple:
    """
    Cria features técnicas.
    
    Args:
        df: DataFrame com OHLCV
        config: Configuração
        
    Returns:
        Tuple (df_with_features, feature_columns)
    """
    logger.info("Creating technical features...")
    
    fe = OptimizedFeatureEngineer(config)
    df_features = fe.create_features(df)
    
    # Remove NaN gerados pelos indicadores
    initial_rows = len(df_features)
    df_features = df_features.dropna().reset_index(drop=True)
    logger.info(f"Removed {initial_rows - len(df_features)} rows with NaN")
    
    # Lista de features (tudo exceto OHLCV e timestamp)
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    feature_columns = [col for col in df_features.columns if col not in exclude_cols]
    
    logger.info(f"Created {len(feature_columns)} features")
    
    return df_features, feature_columns


def select_features(
    df: pd.DataFrame, 
    feature_columns: list, 
    config: dict
) -> list:
    """
    Seleciona features mais relevantes baseado em correlação com target.
    
    Args:
        df: DataFrame com features e close price
        feature_columns: Lista de features candidatas
        config: Configuração
        
    Returns:
        Lista de features selecionadas
    """
    lgbm_config = config['lightgbm']
    min_corr = lgbm_config.get('min_correlation_threshold', 0.01)
    
    if min_corr <= 0:
        logger.info("Feature selection disabled (min_correlation_threshold <= 0)")
        return feature_columns
    
    logger.info(f"Selecting features with correlation > {min_corr}")
    
    # Cria target temporário para calcular correlação
    from src.models.lightgbm_model import LightGBMPredictor
    temp_predictor = LightGBMPredictor(lgbm_config)
    target = temp_predictor.create_target(df)
    
    # Remove NaN do target
    valid_idx = ~target.isna()
    df_valid = df[valid_idx].copy()
    target_valid = target[valid_idx]
    
    # Calcula correlação de cada feature
    correlations = {}
    for col in feature_columns:
        corr = abs(df_valid[col].corr(target_valid))
        correlations[col] = corr
    
    # Seleciona features acima do threshold
    selected_features = [col for col, corr in correlations.items() if corr >= min_corr]
    
    # Log das features removidas
    removed_features = set(feature_columns) - set(selected_features)
    if removed_features:
        logger.info(f"Removed {len(removed_features)} features with low correlation:")
        for feat in sorted(removed_features):
            logger.info(f"  - {feat}: {correlations[feat]:.4f}")
    
    logger.info(f"Selected {len(selected_features)}/{len(feature_columns)} features")
    
    # Log top 10 features
    top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:10]
    logger.info("Top 10 features by correlation:")
    for feat, corr in top_features:
        logger.info(f"  {feat}: {corr:.4f}")
    
    return selected_features


def split_data(df: pd.DataFrame, config: dict) -> tuple:
    """
    Divide dados em treino/validação/teste (temporal).
    
    Args:
        df: DataFrame
        config: Configuração
        
    Returns:
        Tuple (train_df, val_df, test_df)
    """
    data_config = config['data']
    val_split = data_config.get('val_split', 0.15)
    test_split = data_config.get('test_split', 0.15)
    
    n = len(df)
    
    # Divide temporalmente (sem shuffle!)
    test_start = int(n * (1 - test_split))
    val_start = int(n * (1 - test_split - val_split))
    
    train_df = df.iloc[:val_start].copy()
    val_df = df.iloc[val_start:test_start].copy()
    test_df = df.iloc[test_start:].copy()
    
    logger.info(f"Data split:")
    logger.info(f"  Train: {len(train_df)} samples ({len(train_df)/n*100:.1f}%)")
    logger.info(f"  Val:   {len(val_df)} samples ({len(val_df)/n*100:.1f}%)")
    logger.info(f"  Test:  {len(test_df)} samples ({len(test_df)/n*100:.1f}%)")
    
    return train_df, val_df, test_df


def train_lightgbm(config_path: str = 'config_hybrid.yaml') -> dict:
    """
    Função principal de treinamento do LightGBM.
    
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
    
    # Carrega dados
    df = load_and_prepare_data(config)
    
    # Cria features
    df_features, feature_columns = create_features(df, config)
    
    # Seleciona features mais relevantes
    selected_features = select_features(df_features, feature_columns, config)
    
    # Divide dados
    train_df, val_df, test_df = split_data(df_features, config)
    
    # Cria modelo
    lgbm_config = config['lightgbm']
    lgbm_config['seed'] = config['general']['seed']
    
    predictor = LightGBMPredictor(lgbm_config)
    
    # Prepara dados de treino
    logger.info("Preparing training data...")
    X_train, y_train = predictor.prepare_data(train_df, selected_features, create_target=True)
    X_val, y_val = predictor.prepare_data(val_df, selected_features, create_target=True)
    X_test, y_test = predictor.prepare_data(test_df, selected_features, create_target=True)
    
    # Treina modelo
    logger.info("Training LightGBM...")
    metrics = predictor.train(X_train, y_train, X_val, y_val)
    
    # Avalia em teste
    logger.info("Evaluating on test set...")
    test_pred = predictor.predict(X_test)
    test_metrics = predictor._compute_metrics(y_test, test_pred, X_test)
    metrics['test'] = test_metrics
    
    logger.info(f"Test metrics: {test_metrics}")
    
    # Salva modelo
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = models_dir / f'lightgbm_model_{timestamp}'
    predictor.save(model_path)
    
    # Salva também como latest
    latest_path = models_dir / 'lightgbm_model'
    predictor.save(latest_path)
    
    # Feature importance
    if predictor.feature_importance_ is not None:
        logger.info("\nTop 20 most important features:")
        print(predictor.get_feature_importance(20).to_string(index=False))
    
    # Salva métricas
    metrics_file = models_dir / f'lightgbm_metrics_{timestamp}.yaml'
    with open(metrics_file, 'w') as f:
        yaml.dump(metrics, f)
    
    logger.info(f"\nTraining completed!")
    logger.info(f"Model saved to: {latest_path}")
    logger.info(f"Metrics saved to: {metrics_file}")
    
    return {
        'metrics': metrics,
        'model_path': str(latest_path),
        'feature_columns': selected_features
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train LightGBM model for Forex prediction')
    parser.add_argument(
        '--config',
        type=str,
        default='config_hybrid.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    result = train_lightgbm(args.config)
    
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
