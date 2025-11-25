"""Tests for API endpoints."""
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Mock the model artifacts for testing
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_model_state():
    """Create mock model state."""
    mock_agent = MagicMock()
    mock_agent.act.return_value = (1, 0.85)  # buy action with 85% confidence
    
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.random.randn(64, 7).astype(np.float32)
    
    mock_config = {
        "env": {
            "window_size": 64,
            "features": ["rsi_14", "ema_12", "ema_26", "bb_upper_20", "bb_lower_20", "returns_1", "returns_5"],
            "scale_features": True,
        }
    }
    
    return {
        "agent": mock_agent,
        "scaler": mock_scaler,
        "config": mock_config,
        "device": "cpu",
        "loaded": True,
    }


@pytest.fixture
def client(mock_model_state):
    """Create test client with mocked model."""
    with patch("src.api.main.model_state", mock_model_state):
        from src.api.main import app
        yield TestClient(app)


def generate_sample_window(n_bars: int = 64):
    """Generate sample OHLCV window."""
    bars = []
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    base_price = 1.1000
    
    for i in range(n_bars):
        price = base_price + np.random.uniform(-0.01, 0.01)
        bars.append({
            "timestamp": (base_time + timedelta(minutes=i)).isoformat() + "Z",
            "open": round(price + np.random.uniform(-0.0005, 0.0005), 5),
            "high": round(price + abs(np.random.uniform(0, 0.001)), 5),
            "low": round(price - abs(np.random.uniform(0, 0.001)), 5),
            "close": round(price, 5),
            "volume": round(np.random.uniform(1000, 5000), 2),
        })
    
    return bars


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_act_endpoint_valid_request(client):
    """Test /act endpoint with valid request."""
    window = generate_sample_window(64)
    
    request_data = {
        "symbol": "EURUSD",
        "window": window,
    }
    
    response = client.post("/act", json=request_data)
    
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert "action" in data
    assert "action_id" in data
    assert "confidence" in data
    
    # Check values
    assert data["action"] in ["hold", "buy", "sell"]
    assert data["action_id"] in [0, 1, 2]
    assert 0.0 <= data["confidence"] <= 1.0


def test_act_endpoint_invalid_window_size(client):
    """Test /act endpoint with wrong window size."""
    window = generate_sample_window(32)  # Wrong size
    
    request_data = {
        "symbol": "EURUSD",
        "window": window,
    }
    
    response = client.post("/act", json=request_data)
    
    assert response.status_code == 400
    assert "Window size must be 64" in response.json()["detail"]


def test_act_endpoint_invalid_ohlc():
    """Test /act endpoint with invalid OHLC (high < low)."""
    # Note: This test doesn't use the fixture since we want to test validation
    from src.api.main import app
    client = TestClient(app)
    
    bars = [{
        "timestamp": "2024-01-01T00:00:00Z",
        "open": 1.1000,
        "high": 1.0990,  # High < Low - invalid!
        "low": 1.1000,
        "close": 1.0995,
        "volume": 1000,
    }]
    
    request_data = {
        "symbol": "EURUSD",
        "window": bars,
    }
    
    response = client.post("/act", json=request_data)
    
    # Should fail validation
    assert response.status_code == 422


def test_act_endpoint_missing_fields():
    """Test /act endpoint with missing fields."""
    from src.api.main import app
    client = TestClient(app)
    
    request_data = {
        "symbol": "EURUSD",
        # Missing window field
    }
    
    response = client.post("/act", json=request_data)
    
    assert response.status_code == 422


def test_act_endpoint_action_mapping(client, mock_model_state):
    """Test that action IDs map correctly to action names."""
    window = generate_sample_window(64)
    
    # Test each action
    for action_id, action_name in [(0, "hold"), (1, "buy"), (2, "sell")]:
        mock_model_state["agent"].act.return_value = (action_id, 0.9)
        
        request_data = {
            "symbol": "EURUSD",
            "window": window,
        }
        
        response = client.post("/act", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["action"] == action_name
        assert data["action_id"] == action_id


def test_act_endpoint_confidence_range(client):
    """Test that confidence is in valid range."""
    window = generate_sample_window(64)
    
    request_data = {
        "symbol": "EURUSD",
        "window": window,
    }
    
    response = client.post("/act", json=request_data)
    
    assert response.status_code == 200
    data = response.json()
    
    # Confidence should be between 0 and 1
    assert 0.0 <= data["confidence"] <= 1.0


def test_ingest_endpoint_valid_request(client):
    """Test /ingest endpoint with valid data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("src.api.main.os.getenv", return_value=tmpdir):
            bars = generate_sample_window(10)
            
            response = client.post("/ingest?symbol=EURUSD", json=bars)
            
            assert response.status_code == 200
            data = response.json()
            
            # Check response structure
            assert data["status"] == "success"
            assert data["records_saved"] == 10
            assert "eurusd_history.csv" in data["file_path"]
            
            # Verify file was created
            csv_path = Path(tmpdir) / "eurusd_history.csv"
            assert csv_path.exists()
            
            # Read and verify content
            import csv
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert len(rows) == 10
                assert rows[0]["timestamp"] == bars[0]["timestamp"]


def test_ingest_endpoint_append_data(client):
    """Test /ingest endpoint appends to existing file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("src.api.main.os.getenv", return_value=tmpdir):
            # First ingestion
            bars1 = generate_sample_window(5)
            response1 = client.post("/ingest?symbol=GBPUSD", json=bars1)
            assert response1.status_code == 200
            assert response1.json()["records_saved"] == 5
            
            # Second ingestion (append)
            bars2 = generate_sample_window(3)
            response2 = client.post("/ingest?symbol=GBPUSD", json=bars2)
            assert response2.status_code == 200
            assert response2.json()["records_saved"] == 3
            
            # Verify total records
            csv_path = Path(tmpdir) / "gbpusd_history.csv"
            import csv
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert len(rows) == 8  # 5 + 3


def test_ingest_endpoint_empty_data(client):
    """Test /ingest endpoint with empty data list."""
    response = client.post("/ingest?symbol=EURUSD", json=[])
    
    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"]


def test_ingest_endpoint_invalid_ohlc(client):
    """Test /ingest endpoint with invalid OHLC data."""
    bars = [{
        "timestamp": "2024-01-01T00:00:00Z",
        "open": 1.1000,
        "high": 1.0990,  # High < Low - invalid!
        "low": 1.1000,
        "close": 1.0995,
        "volume": 1000,
    }]
    
    response = client.post("/ingest?symbol=EURUSD", json=bars)
    
    # Should fail validation
    assert response.status_code == 422


def test_ingest_endpoint_multiple_symbols(client):
    """Test /ingest endpoint creates separate files for different symbols."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("src.api.main.os.getenv", return_value=tmpdir):
            # Ingest EURUSD
            bars_eur = generate_sample_window(5)
            response_eur = client.post("/ingest?symbol=EURUSD", json=bars_eur)
            assert response_eur.status_code == 200
            
            # Ingest GBPUSD
            bars_gbp = generate_sample_window(3)
            response_gbp = client.post("/ingest?symbol=GBPUSD", json=bars_gbp)
            assert response_gbp.status_code == 200
            
            # Verify separate files exist
            eur_csv = Path(tmpdir) / "eurusd_history.csv"
            gbp_csv = Path(tmpdir) / "gbpusd_history.csv"
            
            assert eur_csv.exists()
            assert gbp_csv.exists()
            
            # Verify record counts
            import csv
            with open(eur_csv, 'r') as f:
                assert len(list(csv.DictReader(f))) == 5
            with open(gbp_csv, 'r') as f:
                assert len(list(csv.DictReader(f))) == 3
