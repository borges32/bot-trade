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

from src.models.lightgbm_model import LightGBMPredictor
from src.common.features_optimized import OptimizedFeatureEngineer

logger = logging.getLogger(__name__)


class TradingPredictor:
    """
    Preditor para decisões de trading baseado em LightGBM.
    
    Usa LightGBM para prever retornos futuros e gera sinais de trading.
    """
    
    def __init__(
        self,
        lightgbm_path: str,
        config: Dict
    ):
        """
        Inicializa o preditor.
        
        Args:
            lightgbm_path: Caminho do modelo LightGBM
            config: Configuração completa
        """
        self.config = config
        
        # Carrega LightGBM
        logger.info(f"Loading LightGBM model from {lightgbm_path}")
        self.lightgbm = LightGBMPredictor(config.get('lightgbm', {}))
        self.lightgbm.load(lightgbm_path)
        
        # Feature engineer
        self.feature_engineer = OptimizedFeatureEngineer(config)
        
        # Thresholds para decisão
        self.min_confidence = config.get('inference', {}).get('min_confidence', 0.60)
        
        logger.info("TradingPredictor initialized successfully")
    
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
        
        # Predição LightGBM
        features_dict = {col: last_row[col] for col in feature_columns}
        predicted_return = self.lightgbm.predict_single(features_dict)
        
        # Calcula confiança baseada na magnitude do retorno
        # Usa escala melhor para forex (movimentos típicos 0.1-1%)
        confidence = min(abs(predicted_return) * 500, 1.0)  # 0.002 return = 100% confidence
        
        # Determina sinal
        if confidence >= self.min_confidence:
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
            'current_price': float(current_price)
        }
        
        return result
    
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
        
        # Calcula confiança (mesma escala do predict)
        confidences = np.minimum(np.abs(predictions) * 500, 1.0)
        
        # Determina sinais
        signals = []
        for pred, conf in zip(predictions, confidences):
            if conf >= self.min_confidence:
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
