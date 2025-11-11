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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create sliding windows from features and prices.
    
    Args:
        features: Feature array of shape (n_samples, n_features).
        prices: Price array of shape (n_samples,).
        window_size: Size of the sliding window.
        min_valid_ratio: Minimum ratio of non-NaN values required in a window.
        
    Returns:
        Tuple of (windows, next_prices, valid_indices).
        - windows: Array of shape (n_windows, window_size, n_features).
        - next_prices: Array of shape (n_windows,) with price after each window.
        - valid_indices: Array of valid window indices.
    """
    n_samples = len(features)
    n_features = features.shape[1]
    
    if n_samples < window_size + 1:
        raise ValueError(f"Not enough samples ({n_samples}) for window_size ({window_size})")
    
    windows = []
    next_prices_list = []
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
            valid_indices.append(i + window_size)
    
    if len(windows) == 0:
        raise ValueError("No valid windows created")
    
    return (
        np.array(windows, dtype=np.float32),
        np.array(next_prices_list, dtype=np.float32),
        np.array(valid_indices, dtype=np.int32)
    )
