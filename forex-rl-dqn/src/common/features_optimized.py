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
        
        # Extrai períodos e flags da configuração (com defaults)
        features_cfg = self.config.get('features', {})
        
        # Flags de ativação (defaults = True para compatibilidade)
        self.use_rsi = features_cfg.get('use_rsi', True)
        self.use_ema = features_cfg.get('use_ema', True)
        self.use_macd = features_cfg.get('use_macd', True)
        self.use_bollinger = features_cfg.get('use_bollinger', True)
        self.use_atr = features_cfg.get('use_atr', True)
        self.use_momentum = features_cfg.get('use_momentum', True)
        self.use_sma = features_cfg.get('use_sma', True)
        self.use_stochastic = features_cfg.get('use_stochastic', True)
        self.use_adx = features_cfg.get('use_adx', True)
        self.use_volatility = features_cfg.get('use_volatility', True)
        self.use_volume_features = features_cfg.get('use_volume_features', True)
        self.use_returns = features_cfg.get('use_returns', True)
        
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
        """Calcula indicadores técnicos baseado nos flags use_*."""
        logger.info("🔄 Calculando indicadores técnicos (respeitando flags)...")
        
        enabled_indicators = []
        disabled_indicators = []
        
        # RSI (se habilitado)
        if self.use_rsi:
            df['rsi'] = self._calculate_rsi(df, period=self.rsi_period)
            enabled_indicators.append(f'RSI (period={self.rsi_period})')
        else:
            disabled_indicators.append('RSI')
        
        # EMAs (se habilitado)
        if self.use_ema:
            df['ema_fast'] = df['close'].ewm(span=self.ema_fast, adjust=False).mean()
            df['ema_slow'] = df['close'].ewm(span=self.ema_slow, adjust=False).mean()
            enabled_indicators.append(f'EMA (fast={self.ema_fast}, slow={self.ema_slow})')
        else:
            disabled_indicators.append('EMA')
        
        # Bollinger Bands (se habilitado)
        if self.use_bollinger:
            df['bb_middle'] = df['close'].rolling(window=self.bb_period).mean()
            rolling_std = df['close'].rolling(window=self.bb_period).std()
            df['bb_upper'] = df['bb_middle'] + (self.bb_std * rolling_std)
            df['bb_lower'] = df['bb_middle'] - (self.bb_std * rolling_std)
            enabled_indicators.append(f'Bollinger Bands (period={self.bb_period}, std={self.bb_std})')
        else:
            disabled_indicators.append('Bollinger Bands')
        
        # MACD (se habilitado)
        if self.use_macd:
            ema_fast = df['close'].ewm(span=self.macd_fast, adjust=False).mean()
            ema_slow = df['close'].ewm(span=self.macd_slow, adjust=False).mean()
            df['macd'] = ema_fast - ema_slow
            df['macd_signal'] = df['macd'].ewm(span=self.macd_signal, adjust=False).mean()
            enabled_indicators.append(f'MACD (fast={self.macd_fast}, slow={self.macd_slow}, signal={self.macd_signal})')
        else:
            disabled_indicators.append('MACD')
        
        # ATR (se habilitado)
        if self.use_atr:
            df['atr'] = self._calculate_atr(df, period=self.atr_period)
            enabled_indicators.append(f'ATR (period={self.atr_period})')
        else:
            disabled_indicators.append('ATR')
        
        # Momentum (se habilitado)
        if self.use_momentum:
            if len(self.momentum_periods) >= 2:
                df['momentum_10'] = df['close'].diff(self.momentum_periods[0])
                df['momentum_20'] = df['close'].diff(self.momentum_periods[1])
            else:
                df['momentum_10'] = df['close'].diff(10)
                df['momentum_20'] = df['close'].diff(20)
            enabled_indicators.append(f'Momentum (periods={self.momentum_periods})')
        else:
            disabled_indicators.append('Momentum')
        
        # Volatilidade (se habilitado)
        if self.use_volatility:
            df['volatility'] = df['close'].pct_change().rolling(window=self.volatility_window).std()
            enabled_indicators.append(f'Volatility (window={self.volatility_window})')
        else:
            disabled_indicators.append('Volatility')
        
        # Volume MA (se habilitado)
        if self.use_volume_features:
            if 'volume' in df.columns:
                df['volume_ma'] = df['volume'].rolling(window=self.volume_ma_period).mean()
            else:
                df['volume_ma'] = 0
            enabled_indicators.append(f'Volume MA (period={self.volume_ma_period})')
        else:
            disabled_indicators.append('Volume Features')
        
        # Log detalhado
        logger.info(f"✅ INDICADORES ATIVOS ({len(enabled_indicators)}):")
        for ind in enabled_indicators:
            logger.info(f"   ✓ {ind}")
        
        if disabled_indicators:
            logger.info(f"❌ INDICADORES DESATIVADOS ({len(disabled_indicators)}):")
            for ind in disabled_indicators:
                logger.info(f"   ✗ {ind}")
        
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
        
        # Retornos (se habilitado)
        if self.use_returns:
            for period in self.return_periods:
                df[f'return_{period}'] = df['close'].pct_change(period)
                self.features_added.append(f'return_{period}')
            
            # Log returns
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))
            self.features_added.append('log_return')
        
        self.features_added.extend([
            'range', 'range_pct', 'body', 'body_pct',
            'upper_shadow', 'lower_shadow', 'is_bullish'
        ])
        
        return df
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features derivadas dos indicadores calculados (respeitando flags)."""
        
        # === FEATURES DE RSI (se RSI habilitado) ===
        if self.use_rsi and 'rsi' in df.columns:
            df['rsi_normalized'] = (df['rsi'] - 50) / 50  # [-1, 1]
            df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
            df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
            df['rsi_divergence'] = df['rsi'].diff()
            self.features_added.extend([
                'rsi_normalized', 'rsi_overbought', 'rsi_oversold', 'rsi_divergence'
            ])
        
        # === FEATURES DE EMA (se EMA habilitado) ===
        if self.use_ema and 'ema_fast' in df.columns and 'ema_slow' in df.columns:
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
        
        # === FEATURES DE BOLLINGER BANDS (se Bollinger habilitado) ===
        if self.use_bollinger and 'bb_upper' in df.columns:
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
        
        # === FEATURES DE MACD (se MACD habilitado) ===
        if self.use_macd and 'macd' in df.columns and 'macd_signal' in df.columns:
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
        
        # === FEATURES DE ATR (se ATR habilitado) ===
        if self.use_atr and 'atr' in df.columns:
            df['atr_normalized'] = df['atr'] / df['close']
            
            # ATR relativo (comparado com média)
            df['atr_ma'] = df['atr'].rolling(20).mean()
            df['atr_ratio'] = df['atr'] / (df['atr_ma'] + 1e-8)
            
            self.features_added.extend(['atr_normalized', 'atr_ma', 'atr_ratio'])
        
        # === FEATURES DE MOMENTUM (se Momentum habilitado) ===
        if self.use_momentum and 'momentum_10' in df.columns and 'momentum_20' in df.columns:
            # Ratio entre momentums
            df['momentum_ratio'] = df['momentum_10'] / (df['momentum_20'] + 1e-8)
            
            # Convergência/divergência
            df['momentum_convergence'] = df['momentum_10'] - df['momentum_20']
            
            self.features_added.extend(['momentum_ratio', 'momentum_convergence'])
        
        # === FEATURES DE VOLUME (se Volume habilitado) ===
        if self.use_volume_features and 'volume_ma' in df.columns:
            # Volume relativo
            df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-8)
            df['volume_spike'] = (df['volume_ratio'] > 2.0).astype(int)
            
            # Mudança de volume
            df['volume_change'] = df['volume'].pct_change()
            
            self.features_added.extend(['volume_ratio', 'volume_spike', 'volume_change'])
        
        # === FEATURES DE VOLATILIDADE (se Volatilidade habilitada) ===
        if self.use_volatility and 'volatility' in df.columns:
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
        """Adiciona features complementares (respeitando flags)."""
        
        # Stochastic (se habilitado)
        if self.use_stochastic:
            df['stoch_k'] = self._calculate_stochastic(df, period=self.stoch_k)
            df['stoch_d'] = df['stoch_k'].rolling(self.stoch_d).mean()
            df['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)
            df['stoch_oversold'] = (df['stoch_k'] < 20).astype(int)
        
        # ADX (se habilitado)
        if self.use_adx:
            df['adx'] = self._calculate_adx(df, period=self.adx_period)
            df['strong_trend'] = (df['adx'] > 25).astype(int)
        
        # SMAs adicionais (se habilitado)
        if self.use_sma:
            for period in self.sma_periods:
                df[f'sma_{period}'] = df['close'].rolling(period).mean()
                df[f'price_sma_{period}_dist'] = (df['close'] - df[f'sma_{period}']) / df['close']
                self.features_added.append(f'sma_{period}')
                self.features_added.append(f'price_sma_{period}_dist')
        
        # Adiciona features criadas (apenas as que existem)
        if self.use_stochastic:
            self.features_added.extend([
                'stoch_k', 'stoch_d', 'stoch_overbought', 'stoch_oversold'
            ])
        if self.use_adx:
            self.features_added.extend(['adx', 'strong_trend'])
        
        return df
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features de interação entre indicadores (apenas se disponíveis)."""
        
        # Convergência de sinais (apenas sinais disponíveis)
        available_signals = []
        if 'ema_cross_signal' in df.columns:
            available_signals.append('ema_cross_signal')
        if 'macd_cross_signal' in df.columns:
            available_signals.append('macd_cross_signal')
        if 'rsi_oversold' in df.columns:
            available_signals.append('rsi_oversold')
        if 'rsi_overbought' in df.columns:
            available_signals.append('rsi_overbought')
        
        if available_signals:
            df['signal_convergence'] = df[available_signals].sum(axis=1)
            self.features_added.append('signal_convergence')
        
        # Regime de mercado (apenas se volatility_ratio e adx existirem)
        if 'volatility_ratio' in df.columns and 'adx' in df.columns:
            regime_score = df['volatility_ratio'] * df['adx']
            regime_score = regime_score.fillna(regime_score.median())
            
            # Preenche NaN antes de converter para int
            df['market_regime'] = pd.cut(
                regime_score,
                bins=[-np.inf, 10, 30, np.inf],
                labels=[0, 1, 2]  # 0=calm, 1=normal, 2=volatile
            )
            df['market_regime'] = df['market_regime'].fillna(1).astype(int)  # Default = normal
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
