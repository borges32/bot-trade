"""Tests for API endpoints."""
import json
from datetime import datetime, timedelta

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
