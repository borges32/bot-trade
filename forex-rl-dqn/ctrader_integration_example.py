"""
Example integration with cTrader for collecting and ingesting historical data.

This is a TEMPLATE/PSEUDOCODE showing how to integrate with cTrader API.
You'll need to adapt it to your specific cTrader API implementation.

Requirements:
    pip install requests python-dotenv

Usage:
    1. Set your cTrader API credentials in .env file
    2. Adjust the parameters (symbol, timeframe, etc.)
    3. Run: python ctrader_integration_example.py
"""

import os
import time
from datetime import datetime, timedelta
from typing import List, Dict
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
CTRADER_API_URL = os.getenv("CTRADER_API_URL", "https://api.ctrader.com")
CTRADER_API_KEY = os.getenv("CTRADER_API_KEY", "your-api-key")
CTRADER_API_SECRET = os.getenv("CTRADER_API_SECRET", "your-api-secret")
FOREX_API_URL = os.getenv("FOREX_API_URL", "http://localhost:8000")


class CTraderClient:
    """
    Simplified cTrader API client.
    
    NOTE: This is PSEUDOCODE. You need to implement actual cTrader API calls.
    Refer to cTrader Open API documentation:
    https://help.ctrader.com/open-api/
    """
    
    def __init__(self, api_url: str, api_key: str, api_secret: str):
        self.api_url = api_url
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })
    
    def get_historical_bars(
        self,
        symbol: str,
        timeframe: str = "M1",  # M1, M5, M15, M30, H1, H4, D1, etc.
        start_time: datetime = None,
        end_time: datetime = None,
        count: int = 100
    ) -> List[Dict]:
        """
        Get historical OHLCV bars from cTrader.
        
        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            timeframe: Timeframe code (M1, M5, H1, etc.)
            start_time: Start datetime (optional)
            end_time: End datetime (optional)
            count: Number of bars to retrieve
            
        Returns:
            List of OHLCV bars in the format expected by /ingest endpoint
        """
        # PSEUDOCODE - Replace with actual cTrader API call
        # Example endpoint (this is fictional):
        # endpoint = f"{self.api_url}/v1/symbols/{symbol}/bars"
        
        # params = {
        #     "timeframe": timeframe,
        #     "count": count,
        # }
        # if start_time:
        #     params["start"] = int(start_time.timestamp() * 1000)
        # if end_time:
        #     params["end"] = int(end_time.timestamp() * 1000)
        
        # response = self.session.get(endpoint, params=params)
        # response.raise_for_status()
        # data = response.json()
        
        # For demonstration, return mock data
        bars = []
        base_time = start_time or datetime.now() - timedelta(hours=count)
        base_price = 1.1000
        
        for i in range(count):
            # Simulate realistic OHLCV data
            timestamp = base_time + timedelta(minutes=i)
            open_price = base_price + (i * 0.00001)
            close_price = open_price + 0.00005
            high_price = max(open_price, close_price) + 0.00002
            low_price = min(open_price, close_price) - 0.00002
            volume = 1000 + (i * 10)
            
            bar = {
                "timestamp": timestamp.isoformat() + "Z",
                "open": round(open_price, 5),
                "high": round(high_price, 5),
                "low": round(low_price, 5),
                "close": round(close_price, 5),
                "volume": round(volume, 2)
            }
            bars.append(bar)
        
        return bars


class ForexDataCollector:
    """Collects data from cTrader and sends to Forex RL API."""
    
    def __init__(self, ctrader_client: CTraderClient, forex_api_url: str):
        self.ctrader = ctrader_client
        self.forex_api_url = forex_api_url
    
    def ingest_historical_data(
        self,
        symbol: str,
        timeframe: str = "M1",
        count: int = 100
    ) -> Dict:
        """
        Collect historical data from cTrader and ingest to Forex API.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe code
            count: Number of bars to collect
            
        Returns:
            API response with ingestion status
        """
        print(f"Collecting {count} bars for {symbol} ({timeframe})...")
        
        # Get data from cTrader
        bars = self.ctrader.get_historical_bars(
            symbol=symbol,
            timeframe=timeframe,
            count=count
        )
        
        if not bars:
            raise ValueError("No data received from cTrader")
        
        print(f"Collected {len(bars)} bars")
        
        # Send to Forex API (array directly, symbol in query param)
        print(f"Sending to {self.forex_api_url}/ingest?symbol={symbol}...")
        response = requests.post(
            f"{self.forex_api_url}/ingest?symbol={symbol}",
            json=bars
        )
        response.raise_for_status()
        
        result = response.json()
        print(f"✓ Success! Saved {result['records_saved']} records to {result['file_path']}")
        
        return result
    
    def continuous_collection(
        self,
        symbol: str,
        timeframe: str = "M1",
        interval_seconds: int = 60,
        bars_per_request: int = 10
    ):
        """
        Continuously collect and ingest data at regular intervals.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe code
            interval_seconds: Collection interval in seconds
            bars_per_request: Number of bars to collect per request
        """
        print(f"Starting continuous collection for {symbol}...")
        print(f"Interval: {interval_seconds}s | Bars per request: {bars_per_request}")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                try:
                    # Collect and ingest
                    self.ingest_historical_data(
                        symbol=symbol,
                        timeframe=timeframe,
                        count=bars_per_request
                    )
                    
                    # Wait for next collection
                    print(f"Waiting {interval_seconds} seconds...")
                    time.sleep(interval_seconds)
                    
                except requests.RequestException as e:
                    print(f"✗ Request error: {e}")
                    print("Retrying in 10 seconds...")
                    time.sleep(10)
                    
        except KeyboardInterrupt:
            print("\nStopped by user")


def main():
    """Main execution function."""
    
    # Initialize cTrader client
    print("Initializing cTrader client...")
    ctrader = CTraderClient(
        api_url=CTRADER_API_URL,
        api_key=CTRADER_API_KEY,
        api_secret=CTRADER_API_SECRET
    )
    
    # Initialize data collector
    collector = ForexDataCollector(
        ctrader_client=ctrader,
        forex_api_url=FOREX_API_URL
    )
    
    print("\n" + "=" * 60)
    print("cTrader to Forex RL - Data Collection")
    print("=" * 60)
    
    # Example 1: One-time historical collection
    print("\n[Example 1] Collecting historical data...")
    try:
        collector.ingest_historical_data(
            symbol="EURUSD",
            timeframe="M1",
            count=1000
        )
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 2: Collect multiple symbols
    print("\n[Example 2] Collecting multiple symbols...")
    symbols = ["EURUSD", "GBPUSD", "USDJPY"]
    for symbol in symbols:
        try:
            collector.ingest_historical_data(
                symbol=symbol,
                timeframe="M5",
                count=500
            )
        except Exception as e:
            print(f"Error collecting {symbol}: {e}")
    
    # Example 3: Continuous collection (uncomment to use)
    # print("\n[Example 3] Starting continuous collection...")
    # collector.continuous_collection(
    #     symbol="EURUSD",
    #     timeframe="M1",
    #     interval_seconds=60,
    #     bars_per_request=10
    # )


if __name__ == "__main__":
    # Check if API is running
    try:
        response = requests.get(f"{FOREX_API_URL}/health", timeout=5)
        if response.status_code == 200:
            print(f"✓ Forex API is running at {FOREX_API_URL}")
        else:
            print(f"✗ Forex API returned status {response.status_code}")
            exit(1)
    except requests.RequestException as e:
        print(f"✗ Cannot connect to Forex API at {FOREX_API_URL}")
        print(f"Error: {e}")
        print("\nMake sure the API is running:")
        print("  docker-compose up -d")
        print("  OR")
        print("  uvicorn src.api.main:app --host 0.0.0.0 --port 8000")
        exit(1)
    
    # Run main
    main()
