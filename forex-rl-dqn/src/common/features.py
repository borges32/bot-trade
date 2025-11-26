"""Feature engineering for OHLCV data without TA-Lib."""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index.
    
    Args:
        close: Close price series.
        period: RSI period.
        
    Returns:
        RSI values (0-100).
    """
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_ema(close: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average.
    
    Args:
        close: Close price series.
        period: EMA period.
        
    Returns:
        EMA values.
    """
    return close.ewm(span=period, adjust=False).mean()


def calculate_bollinger_bands(
    close: pd.Series, period: int = 20, num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands.
    
    Args:
        close: Close price series.
        period: Moving average period.
        num_std: Number of standard deviations for bands.
        
    Returns:
        Tuple of (upper_band, middle_band, lower_band).
    """
    middle = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    return upper, middle, lower


def calculate_returns(close: pd.Series, period: int = 1) -> pd.Series:
    """Calculate percentage returns.
    
    Args:
        close: Close price series.
        period: Period for return calculation.
        
    Returns:
        Percentage returns.
    """
    return close.pct_change(periods=period)


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range.
    
    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        period: ATR period.
        
    Returns:
        ATR values.
    """
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def calculate_momentum(close: pd.Series, period: int = 10) -> pd.Series:
    """Calculate momentum (rate of change).
    
    Args:
        close: Close price series.
        period: Momentum period.
        
    Returns:
        Momentum values (percentage change).
    """
    return close.pct_change(periods=period)


def calculate_volatility(close: pd.Series, period: int = 20) -> pd.Series:
    """Calculate historical volatility (standard deviation of returns).
    
    Args:
        close: Close price series.
        period: Volatility period.
        
    Returns:
        Volatility values.
    """
    returns = close.pct_change()
    volatility = returns.rolling(window=period).std()
    return volatility


def calculate_volume_ma(volume: pd.Series, period: int = 20) -> pd.Series:
    """Calculate volume moving average.
    
    Args:
        volume: Volume series.
        period: MA period.
        
    Returns:
        Volume MA values.
    """
    return volume.rolling(window=period).mean()


def calculate_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
    """Calculate MACD and Signal line.
    
    Args:
        close: Close price series.
        fast: Fast EMA period.
        slow: Slow EMA period.
        signal: Signal line period.
        
    Returns:
        Tuple of (MACD line, Signal line).
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def generate_features(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    """Generate technical features from OHLCV data.
    
    Args:
        df: DataFrame with columns: open, high, low, close, volume.
        feature_names: List of feature names to generate.
        
    Returns:
        DataFrame with generated features.
    """
    features_df = pd.DataFrame(index=df.index)
    
    for feature_name in feature_names:
        if feature_name == "rsi_14":
            features_df["rsi_14"] = calculate_rsi(df["close"], period=14)
        elif feature_name == "ema_12":
            features_df["ema_12"] = calculate_ema(df["close"], period=12)
        elif feature_name == "ema_26":
            features_df["ema_26"] = calculate_ema(df["close"], period=26)
        elif feature_name == "bb_upper_20":
            upper, _, _ = calculate_bollinger_bands(df["close"], period=20)
            features_df["bb_upper_20"] = upper
        elif feature_name == "bb_lower_20":
            _, _, lower = calculate_bollinger_bands(df["close"], period=20)
            features_df["bb_lower_20"] = lower
        elif feature_name == "returns_1":
            features_df["returns_1"] = calculate_returns(df["close"], period=1)
        elif feature_name == "returns_5":
            features_df["returns_5"] = calculate_returns(df["close"], period=5)
        elif feature_name == "atr_14":
            features_df["atr_14"] = calculate_atr(df["high"], df["low"], df["close"], period=14)
        elif feature_name == "momentum_10":
            features_df["momentum_10"] = calculate_momentum(df["close"], period=10)
        elif feature_name == "momentum_20":
            features_df["momentum_20"] = calculate_momentum(df["close"], period=20)
        elif feature_name == "volatility_20":
            features_df["volatility_20"] = calculate_volatility(df["close"], period=20)
        elif feature_name == "volume_ratio":
            volume_ma = calculate_volume_ma(df["volume"], period=20)
            features_df["volume_ratio"] = df["volume"] / volume_ma
        elif feature_name == "macd":
            macd_line, _ = calculate_macd(df["close"])
            features_df["macd"] = macd_line
        elif feature_name == "macd_signal":
            _, signal_line = calculate_macd(df["close"])
            features_df["macd_signal"] = signal_line
        else:
            raise ValueError(f"Unknown feature: {feature_name}")
    
    # Normalize Bollinger Bands as ratio to close price
    if "bb_upper_20" in features_df.columns:
        features_df["bb_upper_20"] = features_df["bb_upper_20"] / df["close"]
    if "bb_lower_20" in features_df.columns:
        features_df["bb_lower_20"] = features_df["bb_lower_20"] / df["close"]
    
    # Normalize EMAs as ratio to close price
    if "ema_12" in features_df.columns:
        features_df["ema_12"] = features_df["ema_12"] / df["close"]
    if "ema_26" in features_df.columns:
        features_df["ema_26"] = features_df["ema_26"] / df["close"]
    
    # Normalize ATR as ratio to close price
    if "atr_14" in features_df.columns:
        features_df["atr_14"] = features_df["atr_14"] / df["close"]
    
    # Normalize MACD as ratio to close price
    if "macd" in features_df.columns:
        features_df["macd"] = features_df["macd"] / df["close"]
    if "macd_signal" in features_df.columns:
        features_df["macd_signal"] = features_df["macd_signal"] / df["close"]
    
    return features_df


class FeatureScaler:
    """Scaler for normalizing features with fit/transform pattern."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.is_fitted = False
        
    def fit(self, features: np.ndarray, feature_names: List[str]) -> "FeatureScaler":
        """Fit the scaler on training data.
        
        Args:
            features: Feature array of shape (n_samples, n_features).
            feature_names: List of feature names.
            
        Returns:
            Self for chaining.
        """
        # Remove NaN rows before fitting
        valid_mask = ~np.isnan(features).any(axis=1)
        valid_features = features[valid_mask]
        
        if len(valid_features) == 0:
            raise ValueError("No valid samples to fit scaler")
            
        self.scaler.fit(valid_features)
        self.feature_names = feature_names
        self.is_fitted = True
        return self
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """Transform features using fitted scaler.
        
        Args:
            features: Feature array of shape (n_samples, n_features).
            
        Returns:
            Scaled features with same shape.
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted before transform")
        return self.scaler.transform(features)
    
    def fit_transform(self, features: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """Fit and transform in one step.
        
        Args:
            features: Feature array of shape (n_samples, n_features).
            feature_names: List of feature names.
            
        Returns:
            Scaled features.
        """
        self.fit(features, feature_names)
        return self.transform(features)
    
    def save(self, path: Path) -> None:
        """Save scaler state to JSON.
        
        Args:
            path: Path to save the scaler state.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted scaler")
            
        state = {
            "feature_names": self.feature_names,
            "mean": self.scaler.mean_.tolist(),
            "scale": self.scaler.scale_.tolist(),
            "var": self.scaler.var_.tolist(),
        }
        
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
    
    def load(self, path: Path) -> "FeatureScaler":
        """Load scaler state from JSON.
        
        Args:
            path: Path to load the scaler state from.
            
        Returns:
            Self for chaining.
        """
        with open(path, "r") as f:
            state = json.load(f)
        
        self.feature_names = state["feature_names"]
        self.scaler.mean_ = np.array(state["mean"])
        self.scaler.scale_ = np.array(state["scale"])
        self.scaler.var_ = np.array(state["var"])
        self.scaler.n_features_in_ = len(self.feature_names)
        self.is_fitted = True
        
        return self


def create_windows(
    features: np.ndarray,
    prices: np.ndarray,
    window_size: int,
    min_valid_ratio: float = 0.8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create sliding windows from features and prices.
    
    Args:
        features: Feature array of shape (n_samples, n_features).
        prices: Price array of shape (n_samples,).
        window_size: Size of the sliding window.
        min_valid_ratio: Minimum ratio of non-NaN values required in a window.
        
    Returns:
        Tuple of (windows, next_prices, window_end_prices, valid_indices).
        - windows: Array of shape (n_windows, window_size, n_features).
        - next_prices: Array of shape (n_windows,) with price after each window.
        - window_end_prices: Array of shape (n_windows,) with price at end of window.
        - valid_indices: Array of valid window indices.
    """
    n_samples = len(features)
    n_features = features.shape[1]
    
    if n_samples < window_size + 1:
        raise ValueError(f"Not enough samples ({n_samples}) for window_size ({window_size})")
    
    windows = []
    next_prices_list = []
    window_end_prices_list = []
    valid_indices = []
    
    for i in range(n_samples - window_size):
        window = features[i : i + window_size]
        
        # Check if window has enough valid values
        valid_ratio = (~np.isnan(window)).sum() / window.size
        
        if valid_ratio >= min_valid_ratio:
            # Fill remaining NaNs with 0 (already scaled)
            window = np.nan_to_num(window, nan=0.0)
            windows.append(window)
            next_prices_list.append(prices[i + window_size])
            window_end_prices_list.append(prices[i + window_size - 1])
            valid_indices.append(i + window_size)
    
    if len(windows) == 0:
        raise ValueError("No valid windows created")
    
    return (
        np.array(windows, dtype=np.float32),
        np.array(next_prices_list, dtype=np.float32),
        np.array(window_end_prices_list, dtype=np.float32),
        np.array(valid_indices, dtype=np.int32)
    )
