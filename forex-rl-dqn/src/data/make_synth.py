"""Generate synthetic OHLCV data for testing."""
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def generate_synthetic_ohlcv(
    n_bars: int = 10000,
    initial_price: float = 1.1000,
    volatility: float = 0.0005,
    trend: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data using geometric Brownian motion.
    
    Args:
        n_bars: Number of bars to generate.
        initial_price: Starting price.
        volatility: Price volatility (standard deviation).
        trend: Drift term (positive = uptrend, negative = downtrend).
        seed: Random seed.
        
    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume.
    """
    np.random.seed(seed)
    
    # Generate timestamps (1-minute bars)
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(minutes=i) for i in range(n_bars)]
    
    # Generate close prices using geometric Brownian motion
    returns = np.random.normal(trend, volatility, n_bars)
    log_prices = np.log(initial_price) + np.cumsum(returns)
    close_prices = np.exp(log_prices)
    
    # Generate OHLC from close prices
    data = []
    
    for i in range(n_bars):
        close = close_prices[i]
        
        # Generate high and low around close
        hl_range = abs(np.random.normal(0, volatility * close))
        
        # High is above close
        high = close + np.random.uniform(0, hl_range)
        
        # Low is below close
        low = close - np.random.uniform(0, hl_range)
        
        # Open is between low and high
        open_price = np.random.uniform(low, high)
        
        # Generate volume (random but realistic)
        volume = np.random.uniform(1000, 10000)
        
        data.append({
            "timestamp": timestamps[i].isoformat() + "Z",
            "open": round(open_price, 5),
            "high": round(high, 5),
            "low": round(low, 5),
            "close": round(close, 5),
            "volume": round(volume, 2),
        })
    
    return pd.DataFrame(data)


def main():
    """Generate and save synthetic data."""
    parser = argparse.ArgumentParser(description="Generate synthetic OHLCV data")
    parser.add_argument(
        "--output",
        type=str,
        default="data/ct.csv",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--n-bars",
        type=int,
        default=10000,
        help="Number of bars to generate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    print(f"Generating {args.n_bars} synthetic OHLCV bars...")
    
    df = generate_synthetic_ohlcv(
        n_bars=args.n_bars,
        seed=args.seed,
    )
    
    df.to_csv(args.output, index=False)
    
    print(f"Synthetic data saved to {args.output}")
    print(f"Shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nPrice range: {df['close'].min():.5f} - {df['close'].max():.5f}")


if __name__ == "__main__":
    main()
