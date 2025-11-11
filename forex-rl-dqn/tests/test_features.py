"""Tests for feature engineering."""
import numpy as np
import pandas as pd
import pytest

from src.common.features import (
    FeatureScaler,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_returns,
    calculate_rsi,
    create_windows,
    generate_features,
)


def test_calculate_rsi():
    """Test RSI calculation."""
    # Create sample price data
    prices = pd.Series([100, 102, 101, 103, 104, 102, 105, 107, 106, 108])
    
    rsi = calculate_rsi(prices, period=5)
    
    # RSI should be between 0 and 100
    valid_rsi = rsi[~rsi.isna()]
    assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()
    
    # First values should be NaN due to warmup
    assert rsi.iloc[:5].isna().any()


def test_calculate_ema():
    """Test EMA calculation."""
    prices = pd.Series([100, 102, 101, 103, 104, 102, 105, 107, 106, 108])
    
    ema = calculate_ema(prices, period=5)
    
    # EMA should have same length as input
    assert len(ema) == len(prices)
    
    # Should not have NaN values
    assert not ema.isna().all()


def test_calculate_bollinger_bands():
    """Test Bollinger Bands calculation."""
    prices = pd.Series([100, 102, 101, 103, 104, 102, 105, 107, 106, 108])
    
    upper, middle, lower = calculate_bollinger_bands(prices, period=5, num_std=2.0)
    
    # All should have same length
    assert len(upper) == len(middle) == len(lower) == len(prices)
    
    # Upper should be >= middle >= lower (where not NaN)
    valid_mask = ~(upper.isna() | middle.isna() | lower.isna())
    assert (upper[valid_mask] >= middle[valid_mask]).all()
    assert (middle[valid_mask] >= lower[valid_mask]).all()


def test_calculate_returns():
    """Test returns calculation."""
    prices = pd.Series([100, 102, 101, 103, 104])
    
    returns = calculate_returns(prices, period=1)
    
    # Check first return is NaN
    assert pd.isna(returns.iloc[0])
    
    # Check calculation
    expected_return_1 = (102 - 100) / 100
    assert abs(returns.iloc[1] - expected_return_1) < 1e-6


def test_generate_features():
    """Test feature generation."""
    # Create sample OHLCV data
    df = pd.DataFrame({
        "open": [100, 101, 102, 103, 104] * 20,
        "high": [101, 102, 103, 104, 105] * 20,
        "low": [99, 100, 101, 102, 103] * 20,
        "close": [100.5, 101.5, 102.5, 103.5, 104.5] * 20,
        "volume": [1000, 1100, 1200, 1300, 1400] * 20,
    })
    
    feature_names = ["rsi_14", "ema_12", "returns_1"]
    features_df = generate_features(df, feature_names)
    
    # Check shape
    assert len(features_df) == len(df)
    assert set(features_df.columns) == set(feature_names)
    
    # Check no infinite values
    assert not np.isinf(features_df.values).any()


def test_feature_scaler():
    """Test feature scaler."""
    # Create sample features
    features = np.random.randn(100, 5)
    feature_names = ["f1", "f2", "f3", "f4", "f5"]
    
    scaler = FeatureScaler()
    
    # Fit and transform
    scaled = scaler.fit_transform(features, feature_names)
    
    # Check shape preserved
    assert scaled.shape == features.shape
    
    # Check mean close to 0 and std close to 1
    assert abs(scaled.mean()) < 0.1
    assert abs(scaled.std() - 1.0) < 0.1
    
    # Test transform on new data
    new_features = np.random.randn(10, 5)
    new_scaled = scaler.transform(new_features)
    assert new_scaled.shape == new_features.shape


def test_create_windows():
    """Test window creation."""
    # Create sample features and prices
    n_samples = 100
    n_features = 5
    window_size = 10
    
    features = np.random.randn(n_samples, n_features)
    prices = np.random.randn(n_samples)
    
    windows, next_prices, valid_indices = create_windows(
        features, prices, window_size
    )
    
    # Check shapes
    assert windows.shape[1] == window_size
    assert windows.shape[2] == n_features
    assert len(next_prices) == len(windows)
    assert len(valid_indices) == len(windows)
    
    # Check number of windows
    expected_windows = n_samples - window_size
    assert len(windows) <= expected_windows


def test_create_windows_with_nans():
    """Test window creation with NaN values."""
    n_samples = 100
    n_features = 5
    window_size = 10
    
    features = np.random.randn(n_samples, n_features)
    
    # Introduce some NaNs
    features[5:15, 0] = np.nan
    
    prices = np.random.randn(n_samples)
    
    windows, next_prices, valid_indices = create_windows(
        features, prices, window_size, min_valid_ratio=0.8
    )
    
    # Should still create some windows
    assert len(windows) > 0
    
    # Windows should not have NaN (filled with 0)
    assert not np.isnan(windows).any()
