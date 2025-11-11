"""Example usage of the Forex RL DQN API."""
import json
from datetime import datetime, timedelta

import requests

# API endpoint
API_URL = "http://localhost:8000"


def generate_example_window(n_bars: int = 64, base_price: float = 1.1000):
    """Generate example OHLCV window for testing."""
    window = []
    current_time = datetime.now()
    
    for i in range(n_bars):
        # Simulate price movement
        price = base_price + (i * 0.0001)
        
        bar = {
            "timestamp": (current_time - timedelta(minutes=n_bars-i)).isoformat() + "Z",
            "open": round(price - 0.0002, 5),
            "high": round(price + 0.0003, 5),
            "low": round(price - 0.0003, 5),
            "close": round(price, 5),
            "volume": 1500.0 + (i * 10),
        }
        window.append(bar)
    
    return window


def check_health():
    """Check API health."""
    print("Checking API health...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")
    return response.json()


def get_trading_action(symbol: str = "EURUSD"):
    """Get trading action from the API."""
    print(f"Getting trading action for {symbol}...")
    
    # Generate example window
    window = generate_example_window(64)
    
    # Prepare request
    request_data = {
        "symbol": symbol,
        "window": window,
    }
    
    # Make request
    response = requests.post(
        f"{API_URL}/act",
        json=request_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Action: {result['action'].upper()}")
        print(f"Action ID: {result['action_id']}")
        print(f"Confidence: {result['confidence']:.2%}\n")
        return result
    else:
        print(f"Error: {response.text}\n")
        return None


def main():
    """Run example usage."""
    print("="*60)
    print("Forex RL DQN API - Example Usage")
    print("="*60 + "\n")
    
    # Check health
    health = check_health()
    
    if not health.get("model_loaded"):
        print("⚠️  Warning: Model not loaded!")
        print("Please ensure you have trained the model and artifacts exist.")
        print("Run: python -m src.rl.train --data data/ct.csv --config config.yaml\n")
        return
    
    # Get trading actions for different symbols
    symbols = ["EURUSD", "GBPUSD", "USDJPY"]
    
    for symbol in symbols:
        get_trading_action(symbol)
    
    print("="*60)
    print("Example complete!")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API")
        print("Please ensure the API is running:")
        print("  uvicorn src.api.main:app --host 0.0.0.0 --port 8000")
    except Exception as e:
        print(f"❌ Error: {e}")
