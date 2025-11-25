"""Example script for ingesting historical data via API."""
import requests
from datetime import datetime, timedelta

def generate_sample_data(n_bars: int = 100, symbol: str = "EURUSD") -> dict:
    """Generate sample historical data.
    
    Args:
        n_bars: Number of bars to generate.
        symbol: Trading symbol.
        
    Returns:
        Dictionary with symbol and data.
    """
    import random
    
    bars = []
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    base_price = 1.1000
    
    for i in range(n_bars):
        # Simulate price movement
        price_change = random.uniform(-0.0020, 0.0020)
        base_price += price_change
        
        # Generate OHLCV
        open_price = base_price
        close_price = base_price + random.uniform(-0.0010, 0.0010)
        high_price = max(open_price, close_price) + abs(random.uniform(0, 0.0005))
        low_price = min(open_price, close_price) - abs(random.uniform(0, 0.0005))
        volume = random.uniform(1000, 5000)
        
        bar = {
            "timestamp": (base_time + timedelta(minutes=i)).isoformat() + "Z",
            "open": round(open_price, 5),
            "high": round(high_price, 5),
            "low": round(low_price, 5),
            "close": round(close_price, 5),
            "volume": round(volume, 2),
        }
        bars.append(bar)
    
    return {
        "symbol": symbol,
        "data": bars
    }


def ingest_data(api_url: str, symbol: str, n_bars: int = 100):
    """Ingest historical data to API.
    
    Args:
        api_url: Base URL of the API (e.g., http://localhost:8000).
        symbol: Trading symbol.
        n_bars: Number of bars to generate and ingest.
    """
    # Generate data
    print(f"Generating {n_bars} bars for {symbol}...")
    sample_data = generate_sample_data(n_bars=n_bars, symbol=symbol)
    bars = sample_data["data"]
    
    # Send to API (only send the array)
    print(f"Sending data to {api_url}/ingest?symbol={symbol}...")
    response = requests.post(f"{api_url}/ingest?symbol={symbol}", json=bars)
    
    # Check response
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Success!")
        print(f"  Status: {result['status']}")
        print(f"  Records saved: {result['records_saved']}")
        print(f"  File path: {result['file_path']}")
    else:
        print(f"✗ Error: {response.status_code}")
        print(f"  Detail: {response.json()}")


if __name__ == "__main__":
    # Configuration
    API_URL = "http://localhost:8000"
    
    # Example 1: Ingest EURUSD data
    print("=" * 60)
    print("Example 1: Ingesting EURUSD data")
    print("=" * 60)
    ingest_data(API_URL, symbol="EURUSD", n_bars=100)
    
    print("\n" + "=" * 60)
    print("Example 2: Appending more EURUSD data")
    print("=" * 60)
    ingest_data(API_URL, symbol="EURUSD", n_bars=50)
    
    print("\n" + "=" * 60)
    print("Example 3: Ingesting GBPUSD data")
    print("=" * 60)
    ingest_data(API_URL, symbol="GBPUSD", n_bars=75)
    
    print("\n" + "=" * 60)
    print("Done! Check the data/ directory for CSV files.")
    print("=" * 60)
