"""
Módulo de inferência para decisões de trading em tempo real.

Este módulo carrega os modelos treinados (LightGBM + PPO) e fornece
uma interface para fazer predições sobre dados recentes de mercado.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

from src.models.lightgbm_model import LightGBMPredictor
from src.models.ppo_agent import PPOTradingAgent
from src.common.features_optimized import OptimizedFeatureEngineer

logger = logging.getLogger(__name__)


class TradingPredictor:
    """
    Preditor completo para decisões de trading.
    
    Combina LightGBM (sinais de mercado) + PPO (decisão de ação)
    para gerar recomendações de trading.
    """
    
    ACTION_NAMES = {
        0: "neutro",
        1: "comprar",
        2: "vender"
    }
    
    def __init__(
        self,
        lightgbm_path: str,
        ppo_path: str,
        feature_config: Dict,
        env_config: Optional[Dict] = None
    ):
        """
        Inicializa o preditor.
        
        Args:
            lightgbm_path: Caminho do modelo LightGBM
            ppo_path: Caminho do modelo PPO
            feature_config: Configuração de features
            env_config: Configuração do ambiente (para estado)
        """
        self.feature_config = feature_config
        self.env_config = env_config or {}
        
        # Carrega LightGBM
        logger.info(f"Loading LightGBM model from {lightgbm_path}")
        self.lightgbm = LightGBMPredictor({})
        self.lightgbm.load(lightgbm_path)
        
        # Feature engineer
        self.feature_engineer = OptimizedFeatureEngineer(feature_config)
        
        # Carrega PPO
        logger.info(f"Loading PPO model from {ppo_path}")
        from stable_baselines3 import PPO
        self.ppo = PPO.load(ppo_path)
        
        # Estado atual (para tracking de posição)
        self.reset_state()
        
        logger.info("TradingPredictor initialized successfully")
    
    def reset_state(self):
        """Reseta o estado interno do preditor."""
        self.position = 0  # -1=short, 0=flat, 1=long
        self.entry_price = 0.0
        self.balance = self.env_config.get('initial_balance', 10000.0)
        self.equity = self.balance
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.max_equity = self.balance
        self.max_drawdown = 0.0
        
        logger.info("State reset")
    
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
        current_position: Optional[int] = None,
        current_price: Optional[float] = None,
        deterministic: bool = True
    ) -> Dict:
        """
        Faz predição de ação de trading.
        
        Args:
            candles: DataFrame com histórico recente de candles (OHLCV)
                     Última linha = candle mais recente
            current_position: Posição atual (-1, 0, 1). Se None, usa estado interno
            current_price: Preço atual. Se None, usa close do último candle
            deterministic: Se True, usa política determinística
            
        Returns:
            Dicionário com:
                - action: 0 (neutro), 1 (comprar), 2 (vender)
                - action_name: "neutro", "comprar", "vender"
                - lightgbm_signal: Sinal do LightGBM (prob ou retorno esperado)
                - confidence: Confiança na ação (baseada no LightGBM)
                - current_state: Estado atual da conta
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
        
        # Posição atual
        if current_position is not None:
            self.position = current_position
        
        # Atualiza estado da conta
        self._update_state(current_price)
        
        # 1. Predição LightGBM
        features_dict = {col: last_row[col] for col in feature_columns}
        lightgbm_signal = self.lightgbm.predict_single(features_dict)
        
        # 2. Constrói observação para PPO
        observation = self._build_observation(
            lightgbm_signal=lightgbm_signal,
            features=last_row[feature_columns].values,
            current_price=current_price
        )
        
        # 3. Predição PPO
        action, _ = self.ppo.predict(observation, deterministic=deterministic)
        action = int(action)
        
        # 4. Calcula confiança
        # Para classifier: usa probabilidade do LightGBM
        # Quanto mais longe de 0.5, maior a confiança
        if self.lightgbm.model_type == 'classifier':
            confidence = abs(lightgbm_signal - 0.5) * 2  # Normaliza para [0, 1]
        else:
            # Para regressor: usa magnitude do retorno esperado
            confidence = min(abs(lightgbm_signal) * 100, 1.0)  # Cap em 1.0
        
        result = {
            'action': action,
            'action_name': self.ACTION_NAMES[action],
            'lightgbm_signal': float(lightgbm_signal),
            'confidence': float(confidence),
            'current_state': {
                'position': self.position,
                'entry_price': self.entry_price,
                'balance': self.balance,
                'equity': self.equity,
                'unrealized_pnl': self.unrealized_pnl,
                'realized_pnl': self.realized_pnl,
                'total_return': (self.equity - self.env_config.get('initial_balance', 10000.0)) / 
                                self.env_config.get('initial_balance', 10000.0),
                'max_drawdown': self.max_drawdown
            }
        }
        
        return result
    
    def predict_from_recent_data(
        self,
        recent_candles: List[Dict],
        current_position: Optional[int] = None,
        deterministic: bool = True
    ) -> Dict:
        """
        Versão conveniente que recebe lista de dicionários.
        
        Args:
            recent_candles: Lista de dicts com keys: timestamp, open, high, low, close, volume
            current_position: Posição atual (-1, 0, 1)
            deterministic: Se True, usa política determinística
            
        Returns:
            Dicionário com predição (mesmo formato de predict())
        """
        # Converte para DataFrame
        df = pd.DataFrame(recent_candles)
        
        # Converte timestamp se necessário
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
        
        return self.predict(df, current_position=current_position, deterministic=deterministic)
    
    def _build_observation(
        self,
        lightgbm_signal: float,
        features: np.ndarray,
        current_price: float
    ) -> np.ndarray:
        """
        Constrói observação para o PPO.
        
        Args:
            lightgbm_signal: Sinal do LightGBM
            features: Array de features técnicas
            current_price: Preço atual
            
        Returns:
            Array de observação
        """
        initial_balance = self.env_config.get('initial_balance', 10000.0)
        
        # Estado da conta (normalizado)
        account_state = np.array([
            self.position,
            self.unrealized_pnl / initial_balance,
            self.equity / initial_balance,
            self.max_drawdown,
            (self.equity - initial_balance) / initial_balance
        ], dtype=np.float32)
        
        # Concatena: sinal LightGBM + features técnicas + estado da conta
        obs = np.concatenate([
            [lightgbm_signal],
            features.astype(np.float32),
            account_state
        ])
        
        # Remove NaN/Inf
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return obs
    
    def _update_state(self, current_price: float):
        """
        Atualiza estado da conta baseado no preço atual.
        
        Args:
            current_price: Preço atual
        """
        # Calcula PnL não realizado
        if self.position == 0:
            self.unrealized_pnl = 0.0
        elif self.position == 1:  # Long
            position_size = self.balance * self.env_config.get('max_position_size', 1.0)
            self.unrealized_pnl = position_size * (current_price - self.entry_price) / self.entry_price
        else:  # Short
            position_size = self.balance * self.env_config.get('max_position_size', 1.0)
            self.unrealized_pnl = position_size * (self.entry_price - current_price) / self.entry_price
        
        # Atualiza equity
        self.equity = self.balance + self.unrealized_pnl
        
        # Atualiza max equity e drawdown
        if self.equity > self.max_equity:
            self.max_equity = self.equity
        
        current_drawdown = (self.max_equity - self.equity) / self.max_equity if self.max_equity > 0 else 0
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
    
    def execute_action(self, action: int, price: float) -> Dict:
        """
        Executa uma ação e atualiza o estado interno.
        
        Args:
            action: Ação (0=neutro, 1=comprar, 2=vender)
            price: Preço de execução
            
        Returns:
            Dicionário com resultado da execução
        """
        # Mapeia ação para posição desejada
        if action == 0:
            desired_position = 0
        elif action == 1:
            desired_position = 1
        else:
            desired_position = -1
        
        result = {
            'previous_position': self.position,
            'new_position': desired_position,
            'trade_executed': False,
            'pnl': 0.0
        }
        
        # Se não há mudança, retorna
        if desired_position == self.position:
            return result
        
        commission = self.env_config.get('commission', 0.0002)
        
        # Se tinha posição, fecha
        if self.position != 0:
            # Calcula PnL
            position_size = self.balance * self.env_config.get('max_position_size', 1.0)
            
            if self.position == 1:  # Fechando long
                pnl = position_size * (price - self.entry_price) / self.entry_price
            else:  # Fechando short
                pnl = position_size * (self.entry_price - price) / self.entry_price
            
            # Desconta comissão
            trade_cost = position_size * commission
            net_pnl = pnl - trade_cost
            
            self.balance += net_pnl
            self.realized_pnl += net_pnl
            
            result['trade_executed'] = True
            result['pnl'] = net_pnl
            
            logger.info(f"Closed position at {price:.5f}, PnL: ${net_pnl:.2f}")
        
        # Abre nova posição se não for neutra
        if desired_position != 0:
            self.entry_price = price
            self.position = desired_position
            
            # Desconta comissão de abertura
            position_size = self.balance * self.env_config.get('max_position_size', 1.0)
            trade_cost = position_size * commission
            self.balance -= trade_cost
            
            result['trade_executed'] = True
            
            logger.info(f"Opened {self.ACTION_NAMES[action]} position at {price:.5f}")
        else:
            self.position = 0
            self.entry_price = 0.0
        
        self.unrealized_pnl = 0.0
        self._update_state(price)
        
        return result
    
    def get_state(self) -> Dict:
        """
        Retorna estado atual completo.
        
        Returns:
            Dicionário com estado
        """
        return {
            'position': self.position,
            'entry_price': self.entry_price,
            'balance': self.balance,
            'equity': self.equity,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'max_equity': self.max_equity,
            'max_drawdown': self.max_drawdown,
            'total_return': (self.equity - self.env_config.get('initial_balance', 10000.0)) / 
                           self.env_config.get('initial_balance', 10000.0)
        }
