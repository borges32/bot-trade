"""
Feature Engineering Otimizado para Forex Trading.

Esta versão SEMPRE calcula todos os indicadores técnicos do zero,
independentemente de existirem no CSV, garantindo consistência total.

Indicadores calculados:
- RSI (14 períodos)
- EMAs (12, 26 períodos)
- Bollinger Bands (20 períodos, 2 desvios)
- MACD (12, 26, 9)
- ATR (14 períodos)
- Momentum (10, 20 períodos)
- Volatilidade
- Volume MA
- Indicadores complementares (Stochastic, ADX, SMAs)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class OptimizedFeatureEngineer:
    """
    Feature engineer que calcula TODOS os indicadores do zero.
    
    Estratégia:
    1. Calcula SEMPRE todos os indicadores técnicos (RSI, EMA, BB, MACD, ATR, etc.)
    2. Ignora indicadores que possam existir no CSV (garante consistência)
    3. Adiciona features complementares (Stochastic, ADX, SMAs)
    4. Cria features derivadas (ratios, crossovers, regimes de mercado)
    """
    
    # Indicadores que esperamos encontrar pré-calculados
    EXPECTED_PRECOMPUTED = {
        'rsi': 'Relative Strength Index',
        'ema_fast': 'EMA rápida',
        'ema_slow': 'EMA lenta',
        'bb_upper': 'Bollinger Band superior',
        'bb_middle': 'Bollinger Band média',
        'bb_lower': 'Bollinger Band inferior',
        'atr': 'Average True Range',
        'momentum_10': 'Momentum 10 períodos',
        'momentum_20': 'Momentum 20 períodos',
        'volatility': 'Volatilidade',
        'volume_ma': 'Média móvel de volume',
        'macd': 'MACD',
        'macd_signal': 'MACD Signal'
    }
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Inicializa o feature engineer.
        
        Args:
            config: Configuração opcional (usa defaults se não fornecido)
        """
        self.config = config or {}
        self.precomputed_found = []
        self.features_added = []
        
        # Extrai períodos da configuração (com defaults)
        features_cfg = self.config.get('features', {})
        
        # Períodos dos indicadores
        self.rsi_period = features_cfg.get('rsi_period', 14)
        self.ema_fast = features_cfg.get('ema_fast', 12)
        self.ema_slow = features_cfg.get('ema_slow', 26)
        self.macd_fast = features_cfg.get('macd_fast', 12)
        self.macd_slow = features_cfg.get('macd_slow', 26)
        self.macd_signal = features_cfg.get('macd_signal', 9)
        self.bb_period = features_cfg.get('bb_period', 20)
        self.bb_std = features_cfg.get('bb_std', 2.0)
        self.atr_period = features_cfg.get('atr_period', 14)
        self.momentum_periods = features_cfg.get('momentum_periods', [10, 20])
        self.volatility_window = features_cfg.get('volatility_window', 20)
        self.volume_ma_period = features_cfg.get('volume_ma_period', 20)
        self.stoch_k = features_cfg.get('stoch_k', 14)
        self.stoch_d = features_cfg.get('stoch_d', 3)
        self.adx_period = features_cfg.get('adx_period', 14)
        self.sma_periods = features_cfg.get('sma_periods', [20, 50])
        self.return_periods = features_cfg.get('return_periods', [1, 3, 5, 10])
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features a partir dos dados OHLCV, calculando TODOS os indicadores.
        
        Args:
            df: DataFrame com OHLCV (timestamp, open, high, low, close, volume)
            
        Returns:
            DataFrame com todas as features
        """
        df = df.copy()
        
        # Remove indicadores pré-calculados se existirem (para recalcular do zero)
        cols_to_drop = [col for col in self.EXPECTED_PRECOMPUTED.keys() if col in df.columns]
        if cols_to_drop:
            logger.info(f"🗑️  Removendo {len(cols_to_drop)} indicadores do CSV para recalcular: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)
        
        # SEMPRE calcula todos os indicadores do zero
        df = self._calculate_all_indicators(df)
        
        # Adiciona features básicas de preço
        df = self._add_price_features(df)
        
        # Adiciona features derivadas dos indicadores calculados
        df = self._add_derived_features(df)
        
        # Adiciona features complementares (Stochastic, ADX, SMAs)
        df = self._add_complementary_features(df)
        
        # Adiciona features de interação
        df = self._add_interaction_features(df)
        
        # Remove NaN
        df = df.fillna(method='ffill').fillna(0)
        
        logger.info(f"✓ Total de features: {len(df.columns)} colunas")
        logger.info(f"✓ Indicadores calculados: {len(self.EXPECTED_PRECOMPUTED)}")
        logger.info(f"✓ Features adicionais: {len(self.features_added)}")
        
        return df
    
    def _calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula TODOS os indicadores técnicos do zero usando períodos configuráveis."""
        logger.info("🔄 Calculando todos os indicadores técnicos...")
        
        # RSI (período configurável)
        df['rsi'] = self._calculate_rsi(df, period=self.rsi_period)
        
        # EMAs (períodos configuráveis)
        df['ema_fast'] = df['close'].ewm(span=self.ema_fast, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.ema_slow, adjust=False).mean()
        
        # Bollinger Bands (período e desvio configuráveis)
        df['bb_middle'] = df['close'].rolling(window=self.bb_period).mean()
        rolling_std = df['close'].rolling(window=self.bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (self.bb_std * rolling_std)
        df['bb_lower'] = df['bb_middle'] - (self.bb_std * rolling_std)
        
        # MACD (períodos configuráveis)
        ema_fast = df['close'].ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.macd_slow, adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=self.macd_signal, adjust=False).mean()
        
        # ATR (período configurável)
        df['atr'] = self._calculate_atr(df, period=self.atr_period)
        
        # Momentum (períodos configuráveis)
        if len(self.momentum_periods) >= 2:
            df['momentum_10'] = df['close'].diff(self.momentum_periods[0])
            df['momentum_20'] = df['close'].diff(self.momentum_periods[1])
        else:
            df['momentum_10'] = df['close'].diff(10)
            df['momentum_20'] = df['close'].diff(20)
        
        # Volatilidade (período configurável)
        df['volatility'] = df['close'].pct_change().rolling(window=self.volatility_window).std()
        
        # Volume MA (período configurável)
        if 'volume' in df.columns:
            df['volume_ma'] = df['volume'].rolling(window=self.volume_ma_period).mean()
        else:
            df['volume_ma'] = 0
        
        logger.info(f"✓ Calculados {len(self.EXPECTED_PRECOMPUTED)} indicadores técnicos")
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features básicas de preço."""
        # Range
        df['range'] = df['high'] - df['low']
        df['range_pct'] = df['range'] / df['close']
        
        # Body e shadows (candlestick)
        df['body'] = abs(df['close'] - df['open'])
        df['body_pct'] = df['body'] / df['close']
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # Direção
        df['is_bullish'] = (df['close'] > df['open']).astype(int)
        
        # Retornos (múltiplos períodos configuráveis)
        for period in self.return_periods:
            df[f'return_{period}'] = df['close'].pct_change(period)
            self.features_added.append(f'return_{period}')
        
        # Log returns
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        self.features_added.extend([
            'range', 'range_pct', 'body', 'body_pct',
            'upper_shadow', 'lower_shadow', 'is_bullish', 'log_return'
        ])
        
        return df
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features derivadas dos indicadores calculados."""
        
        # === FEATURES DE RSI ===
        df['rsi_normalized'] = (df['rsi'] - 50) / 50  # [-1, 1]
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_divergence'] = df['rsi'].diff()
        self.features_added.extend([
            'rsi_normalized', 'rsi_overbought', 'rsi_oversold', 'rsi_divergence'
        ])
        
        # === FEATURES DE EMA ===
        df['ema_cross'] = df['ema_fast'] - df['ema_slow']
        df['ema_cross_pct'] = df['ema_cross'] / df['close']
        df['ema_cross_signal'] = (df['ema_cross'] > 0).astype(int)
        
        # Distância do preço para EMAs
        df['price_ema_fast_dist'] = (df['close'] - df['ema_fast']) / df['close']
        df['price_ema_slow_dist'] = (df['close'] - df['ema_slow']) / df['close']
        
        self.features_added.extend([
            'ema_cross', 'ema_cross_pct', 'ema_cross_signal',
            'price_ema_fast_dist', 'price_ema_slow_dist'
        ])
        
        # === FEATURES DE BOLLINGER BANDS ===
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
        df['bb_position_normalized'] = (df['bb_position'] - 0.5) * 2  # [-1, 1]
        
        # Distância
        df['price_bb_upper_dist'] = (df['bb_upper'] - df['close']) / df['close']
        df['price_bb_lower_dist'] = (df['close'] - df['bb_lower']) / df['close']
        
        # Breakout flags
        df['bb_breakout_upper'] = (df['close'] > df['bb_upper']).astype(int)
        df['bb_breakout_lower'] = (df['close'] < df['bb_lower']).astype(int)
        
        self.features_added.extend([
            'bb_width', 'bb_position', 'bb_position_normalized',
            'price_bb_upper_dist', 'price_bb_lower_dist',
            'bb_breakout_upper', 'bb_breakout_lower'
        ])
        
        # === FEATURES DE MACD ===
        # Histogram
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['macd_hist_change'] = df['macd_hist'].diff()
        
        # Crossover
        df['macd_cross_signal'] = (df['macd'] > df['macd_signal']).astype(int)
        
        # Strength (média móvel do histogram)
        df['macd_strength'] = df['macd_hist'].rolling(5).mean()
        
        self.features_added.extend([
            'macd_hist', 'macd_hist_change', 'macd_cross_signal', 'macd_strength'
        ])
        
        # === FEATURES DE ATR ===
        df['atr_normalized'] = df['atr'] / df['close']
        
        # ATR relativo (comparado com média)
        df['atr_ma'] = df['atr'].rolling(20).mean()
        df['atr_ratio'] = df['atr'] / (df['atr_ma'] + 1e-8)
        
        self.features_added.extend(['atr_normalized', 'atr_ma', 'atr_ratio'])
        
        # === FEATURES DE MOMENTUM ===
        # Ratio entre momentums
        df['momentum_ratio'] = df['momentum_10'] / (df['momentum_20'] + 1e-8)
        
        # Convergência/divergência
        df['momentum_convergence'] = df['momentum_10'] - df['momentum_20']
        
        self.features_added.extend(['momentum_ratio', 'momentum_convergence'])
        
        # === FEATURES DE VOLUME ===
        # Volume relativo
        df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-8)
        df['volume_spike'] = (df['volume_ratio'] > 2.0).astype(int)
        
        # Mudança de volume
        df['volume_change'] = df['volume'].pct_change()
        
        self.features_added.extend(['volume_ratio', 'volume_spike', 'volume_change'])
        
        # === FEATURES DE VOLATILIDADE ===
        # Volatilidade relativa
        df['volatility_ma'] = df['volatility'].rolling(20).mean()
        df['volatility_ratio'] = df['volatility'] / (df['volatility_ma'] + 1e-8)
        
        # Regime de volatilidade
        df['high_volatility'] = (df['volatility_ratio'] > 1.5).astype(int)
        df['low_volatility'] = (df['volatility_ratio'] < 0.5).astype(int)
        
        self.features_added.extend([
            'volatility_ma', 'volatility_ratio',
            'high_volatility', 'low_volatility'
        ])
        
        return df
    
    def _add_complementary_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features complementares que não vêm no CSV."""
        
        # Stochastic (períodos configuráveis)
        df['stoch_k'] = self._calculate_stochastic(df, period=self.stoch_k)
        df['stoch_d'] = df['stoch_k'].rolling(self.stoch_d).mean()
        df['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)
        df['stoch_oversold'] = (df['stoch_k'] < 20).astype(int)
        
        # ADX (período configurável)
        df['adx'] = self._calculate_adx(df, period=self.adx_period)
        df['strong_trend'] = (df['adx'] > 25).astype(int)
        
        # SMAs adicionais (períodos configuráveis)
        for period in self.sma_periods:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'price_sma_{period}_dist'] = (df['close'] - df[f'sma_{period}']) / df['close']
            self.features_added.append(f'sma_{period}')
            self.features_added.append(f'price_sma_{period}_dist')
        
        self.features_added.extend([
            'stoch_k', 'stoch_d', 'stoch_overbought', 'stoch_oversold',
            'adx', 'strong_trend'
        ])
        
        return df
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features de interação entre indicadores."""
        
        # Convergência de sinais
        signals = ['ema_cross_signal', 'macd_cross_signal', 'rsi_oversold', 'rsi_overbought']
        df['signal_convergence'] = df[signals].sum(axis=1)
        self.features_added.append('signal_convergence')
        
        # Regime de mercado (combinação de volatilidade e trend)
        regime_score = df['volatility_ratio'] * df['adx']
        regime_score = regime_score.fillna(regime_score.median())
        df['market_regime'] = pd.cut(
            regime_score,
            bins=[-np.inf, 10, 30, np.inf],
            labels=[0, 1, 2]  # 0=calm, 1=normal, 2=volatile
        ).astype(int)
        self.features_added.append('market_regime')
        
        return df
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calcula Relative Strength Index."""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calcula Average True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def _calculate_stochastic(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calcula Stochastic %K."""
        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()
        
        stoch_k = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-8)
        return stoch_k
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calcula Average Directional Index."""
        # True Range
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Directional Movement
        up_move = df['high'] - df['high'].shift()
        down_move = df['low'].shift() - df['low']
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed indicators
        atr = true_range.rolling(window=period).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / atr
        
        # ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def get_feature_names(self) -> List[str]:
        """Retorna lista de nomes de todas as features criadas."""
        return self.features_added
    
    def get_precomputed_names(self) -> List[str]:
        """Retorna lista de indicadores pré-calculados encontrados."""
        return self.precomputed_found


def create_optimized_features(df: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Função helper para criar features otimizadas.
    
    Args:
        df: DataFrame com OHLCV e indicadores pré-calculados
        config: Configuração opcional
        
    Returns:
        DataFrame com features
    """
    engineer = OptimizedFeatureEngineer(config)
    return engineer.create_features(df)
