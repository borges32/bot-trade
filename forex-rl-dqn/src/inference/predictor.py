"""
Módulo de inferência para decisões de trading em tempo real.

Este módulo carrega o modelo LightGBM treinado e fornece
uma interface para fazer predições sobre dados recentes de mercado.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import os

from src.models.lightgbm_model import LightGBMPredictor
from src.common.features_optimized import OptimizedFeatureEngineer

logger = logging.getLogger(__name__)

# Redis client (opcional)
_redis_client = None

def get_redis_client():
    """Obtém cliente Redis se disponível."""
    global _redis_client
    if _redis_client is None:
        try:
            import redis
            redis_host = os.getenv('REDIS_HOST', 'localhost')
            redis_port = int(os.getenv('REDIS_PORT', 6379))
            _redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                decode_responses=True,
                socket_connect_timeout=2
            )
            # Testa conexão
            _redis_client.ping()
            logger.info(f"Redis conectado em {redis_host}:{redis_port}")
        except Exception as e:
            logger.warning(f"Redis não disponível: {e}")
            _redis_client = None
    return _redis_client


class TradingPredictor:
    """
    Preditor para decisões de trading baseado em LightGBM.
    
    Usa LightGBM para prever retornos futuros e gera sinais de trading.
    """
    
    def __init__(
        self,
        lightgbm_path: str,
        config: Dict,
        enable_redis: bool = True
    ):
        """
        Inicializa o preditor.
        
        Args:
            lightgbm_path: Caminho do modelo LightGBM
            config: Configuração completa
            enable_redis: Se True, salva predições no Redis
        """
        self.config = config
        self.enable_redis = enable_redis
        
        # Carrega LightGBM
        logger.info(f"Loading LightGBM model from {lightgbm_path}")
        self.lightgbm = LightGBMPredictor(config.get('lightgbm', {}))
        self.lightgbm.load(lightgbm_path)
        
        # Feature engineer
        self.feature_engineer = OptimizedFeatureEngineer(config)
        
        # Thresholds para decisão
        # TEMPORÁRIO: Reduzido para 2% para testes (originalmente 15%)
        # Permite gerar sinais mesmo com retornos muito pequenos
        self.min_confidence = config.get('inference', {}).get('min_confidence', 0.02)
        
        # Tenta carregar test_direction_acc dos metadados
        try:
            import joblib
            metadata_path = Path(lightgbm_path).with_suffix('.metadata.pkl')
            if metadata_path.exists():
                metadata = joblib.load(metadata_path)
                test_metrics = metadata.get('test_metrics', {})
                self.test_direction_acc = test_metrics.get('direction_accuracy', 0.55)
            else:
                self.test_direction_acc = 0.55
        except Exception as e:
            logger.warning(f"Could not load test metrics: {e}")
            self.test_direction_acc = 0.55
        
        logger.info(f"TradingPredictor initialized successfully")
        logger.info(f"Model test direction accuracy: {self.test_direction_acc:.2%}")
    
    def prepare_features(self, candles: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepara features a partir de dados OHLCV.
        
        Args:
            candles: DataFrame com colunas ['open', 'high', 'low', 'close', 'volume']
            
        Returns:
            Tuple (df_with_features, feature_columns)
        """
        # Cria features técnicas
        df_features = self.feature_engineer.create_features(candles.copy())
        
        # Remove NaN
        df_features = df_features.dropna()
        
        # Lista de features
        exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        feature_columns = [col for col in df_features.columns if col not in exclude_cols]
        
        return df_features, feature_columns
    
    def predict(
        self,
        candles: pd.DataFrame,
        current_price: Optional[float] = None
    ) -> Dict:
        """
        Faz predição de trading.
        
        Args:
            candles: DataFrame com histórico recente de candles (OHLCV)
                     Última linha = candle mais recente
            current_price: Preço atual. Se None, usa close do último candle
            
        Returns:
            Dicionário com:
                - signal: "BUY", "SELL", "NEUTRAL"
                - predicted_return: Retorno esperado (%)
                - confidence: Nível de confiança [0, 1]
                - current_price: Preço usado na predição
        """
        if len(candles) < 50:
            logger.warning("Insufficient candles for reliable prediction. Need at least 50.")
        
        # Prepara features
        df_features, feature_columns = self.prepare_features(candles)
        
        if len(df_features) == 0:
            raise ValueError("No valid data after feature engineering")
        
        # Pega última linha (mais recente)
        last_row = df_features.iloc[-1]
        
        # Preço atual
        if current_price is None:
            current_price = last_row['close']
        
        # Log das features para debug
        logger.info(f"[FEATURES DEBUG] Last candle close: {last_row['close']:.5f}")
        
        # Log de algumas features (com verificação segura)
        sample_features = []
        for feat_name in ['close', 'rsi', 'ema_fast', 'macd']:
            if feat_name in last_row:
                sample_features.append(f"{feat_name}={last_row[feat_name]:.5f}")
        logger.info(f"[FEATURES DEBUG] Sample features: {', '.join(sample_features)}")
        
        # Predição LightGBM
        features_dict = {col: last_row[col] for col in feature_columns}
        logger.info(f"[FEATURES DEBUG] Total features sent to model: {len(features_dict)}")
        
        predicted_return = self.lightgbm.predict_single(features_dict)
        
        # Calcula confiança ajustada
        # Base accuracy (do histórico de teste)
        base_accuracy = self.test_direction_acc
        
        # Magnitude do retorno (predicted_return está em escala decimal)
        # Ex: 0.001 = 0.1% = 10 pips em forex típico
        abs_return = abs(predicted_return)
        
        # Converte para percentual
        return_pct = abs_return * 100
        
        # Fator de magnitude: escala logarítmica para dar mais peso a movimentos maiores
        # - Movimentos de 0.05% (5 pips) → ~0.4 (40% do base)
        # - Movimentos de 0.1% (10 pips) → ~0.6 (60% do base)
        # - Movimentos de 0.2% (20 pips) → ~0.8 (80% do base)
        # - Movimentos >= 0.5% (50 pips) → 1.0 (100% do base)
        magnitude_factor = min(return_pct / 0.5, 1.0)
        
        # Confiança final = base_accuracy × magnitude_factor
        # Se base_accuracy = 0.55 (55%) e return = 0.1% (10 pips)
        # magnitude_factor = 0.1 / 0.5 = 0.2
        # confidence = 0.55 × 0.2 = 0.11 (11%)
        confidence = base_accuracy * magnitude_factor
        
        # Log detalhado para debug
        logger.info(f"[PREDICT DEBUG] predicted_return={predicted_return:.6f} ({return_pct:.4f}%), "
                   f"base_accuracy={base_accuracy:.2%}, magnitude_factor={magnitude_factor:.4f}, "
                   f"confidence={confidence:.2%}, min_confidence={self.min_confidence:.2%}")
        
        # Determina sinal
        # Regra: confidence >= min_confidence E confidence > 2.8%
        if confidence >= self.min_confidence and confidence > 0.028:
            if predicted_return > 0:
                signal = "BUY"
            else:
                signal = "SELL"
        else:
            signal = "NEUTRAL"
        
        result = {
            'signal': signal,
            'predicted_return': float(predicted_return),
            'confidence': float(confidence),
            'base_accuracy': float(base_accuracy),
            'current_price': float(current_price)
        }
        
        # Salva no Redis se habilitado
        if self.enable_redis:
            self._save_to_redis(result)
        
        return result
    
    def _save_to_redis(self, result: Dict):
        """
        Salva resultado da predição no Redis.
        
        Args:
            result: Dicionário com resultado da predição
        """
        try:
            redis_client = get_redis_client()
            if redis_client is not None:
                import json
                from datetime import datetime
                
                # Adiciona timestamp
                result_with_timestamp = result.copy()
                result_with_timestamp['timestamp'] = datetime.utcnow().isoformat()
                
                # Salva no Redis (sobrescreve anterior)
                redis_client.set('latest_prediction', json.dumps(result_with_timestamp))
                logger.debug(f"Prediction saved to Redis: {result['signal']}")
        except Exception as e:
            logger.warning(f"Failed to save to Redis: {e}")
    
    def predict_from_recent_data(
        self,
        recent_candles: List[Dict]
    ) -> Dict:
        """
        Versão conveniente que recebe lista de dicionários.
        
        Args:
            recent_candles: Lista de dicts com keys: timestamp, open, high, low, close, volume
            
        Returns:
            Dicionário com predição (mesmo formato de predict())
        """
        # Converte para DataFrame
        df = pd.DataFrame(recent_candles)
        
        # Converte timestamp se necessário
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
        
        return self.predict(df)
    
    def batch_predict(
        self,
        candles: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Faz predições em batch para múltiplos pontos no tempo.
        
        Args:
            candles: DataFrame com histórico de candles
            
        Returns:
            DataFrame com colunas: timestamp, signal, predicted_return, confidence
        """
        # Prepara features
        df_features, feature_columns = self.prepare_features(candles)
        
        if len(df_features) == 0:
            raise ValueError("No valid data after feature engineering")
        
        # Predições em batch
        predictions = self.lightgbm.predict(df_features[feature_columns])
        
        # Calcula confiança (mesma lógica do predict)
        # Converte para percentual e normaliza
        return_pcts = np.abs(predictions) * 100
        magnitude_factors = np.minimum(return_pcts / 0.5, 1.0)
        confidences = self.test_direction_acc * magnitude_factors
        
        # Determina sinais
        # Regra: confidence >= min_confidence E confidence > 2.8%
        signals = []
        for pred, conf in zip(predictions, confidences):
            if conf >= self.min_confidence and conf > 0.028:
                signals.append("BUY" if pred > 0 else "SELL")
            else:
                signals.append("NEUTRAL")
        
        # Cria DataFrame de resultados
        results = pd.DataFrame({
            'timestamp': df_features['timestamp'] if 'timestamp' in df_features.columns else range(len(predictions)),
            'signal': signals,
            'predicted_return': predictions,
            'confidence': confidences
        })
        
        return results
