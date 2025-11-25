"""FastAPI inference service for Forex RL DQN agent."""
import csv
import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.common.features import FeatureScaler, generate_features
from src.common.utils import get_device
from src.rl.agent import DQNAgent

# Initialize FastAPI app
app = FastAPI(
    title="Forex RL DQN API",
    description="Inference API for Forex trading with Reinforcement Learning",
    version="1.0.0",
)

# Global state
model_state = {
    "agent": None,
    "scaler": None,
    "config": None,
    "device": None,
    "loaded": False,
}


class OHLCVBar(BaseModel):
    """Single OHLCV bar."""
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    open: float = Field(..., gt=0, description="Open price")
    high: float = Field(..., gt=0, description="High price")
    low: float = Field(..., gt=0, description="Low price")
    close: float = Field(..., gt=0, description="Close price")
    volume: float = Field(..., ge=0, description="Volume")
    
    @field_validator("high")
    @classmethod
    def validate_high(cls, v: float, info) -> float:
        """Validate high >= low."""
        if "low" in info.data and v < info.data["low"]:
            raise ValueError("high must be >= low")
        return v


class ActRequest(BaseModel):
    """Request for action prediction."""
    symbol: str = Field(..., description="Trading symbol (e.g., EURUSD)")
    window: List[OHLCVBar] = Field(..., min_length=1, description="Window of OHLCV bars")


class ActResponse(BaseModel):
    """Response with action prediction."""
    action: str = Field(..., description="Predicted action: buy, sell, or hold")
    action_id: int = Field(..., description="Action ID: 0=hold, 1=buy, 2=sell")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")


class IngestResponse(BaseModel):
    """Response for data ingestion."""
    status: str = Field(..., description="Status of ingestion")
    records_saved: int = Field(..., description="Number of records saved")
    file_path: str = Field(..., description="Path to the CSV file")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool


def load_model_artifacts(
    model_path: str = "artifacts/dqn.pt",
    scaler_path: str = "artifacts/feature_state.json",
    config_path: str = "artifacts/config.yaml",
) -> None:
    """Load model, scaler, and config.
    
    Args:
        model_path: Path to model checkpoint.
        scaler_path: Path to scaler state.
        config_path: Path to config file.
    """
    global model_state
    
    if model_state["loaded"]:
        return
    
    # Check if files exist
    model_path_obj = Path(model_path)
    scaler_path_obj = Path(scaler_path)
    config_path_obj = Path(config_path)
    
    if not model_path_obj.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not scaler_path_obj.exists():
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    if not config_path_obj.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")
    
    # Load config
    with open(config_path_obj, "r") as f:
        config = yaml.safe_load(f)
    
    # Get device
    device = get_device("cpu")  # Use CPU for inference by default
    
    # Load scaler
    scaler = FeatureScaler()
    scaler.load(scaler_path_obj)
    
    # Determine feature dimensions
    n_features = len(config["env"]["features"])
    n_actions = 3
    
    # Create agent
    agent = DQNAgent(
        n_features=n_features,
        n_actions=n_actions,
        device=device,
        gamma=config["agent"]["gamma"],
        lr=config["agent"]["lr"],
        epsilon_start=0.0,  # No exploration during inference
        epsilon_end=0.0,
        epsilon_decay_steps=1,
        target_update_interval=config["agent"]["target_update_interval"],
        grad_clip_norm=config["agent"]["grad_clip_norm"],
        lstm_hidden=config["agent"]["lstm_hidden"],
        mlp_hidden=config["agent"]["mlp_hidden"],
        dueling=config["agent"]["dueling"],
    )
    
    # Load model weights
    agent.load(model_path_obj)
    agent.q_network.eval()
    
    # Update global state
    model_state["agent"] = agent
    model_state["scaler"] = scaler
    model_state["config"] = config
    model_state["device"] = device
    model_state["loaded"] = True
    
    print(f"Model loaded from {model_path}")


def save_to_csv(data: List[OHLCVBar], symbol: str, data_dir: str = "data") -> tuple[str, int]:
    """Save OHLCV data to CSV file.
    
    Args:
        data: List of OHLCV bars.
        symbol: Trading symbol (used for filename).
        data_dir: Directory to save CSV files.
        
    Returns:
        Tuple of (file_path, records_saved).
    """
    # Create data directory if it doesn't exist
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Determine file path
    csv_filename = f"{symbol.lower()}_history.csv"
    csv_path = data_path / csv_filename
    
    # Convert data to list of dicts
    records = [bar.model_dump() for bar in data]
    
    # Check if file exists
    file_exists = csv_path.exists()
    
    # Define CSV columns
    fieldnames = ["timestamp", "open", "high", "low", "close", "volume"]
    
    # Write to CSV
    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # Write header only if file is new
        if not file_exists:
            writer.writeheader()
        
        # Write data rows
        writer.writerows(records)
    
    return str(csv_path), len(records)


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    try:
        load_model_artifacts()
        print("Model loaded successfully")
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("Model will be loaded on first request")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint.
    
    Returns:
        Health status.
    """
    return HealthResponse(
        status="ok",
        model_loaded=model_state["loaded"]
    )


@app.post("/act", response_model=ActResponse)
async def predict_action(request: ActRequest):
    """Predict trading action for given window.
    
    Args:
        request: Action request with OHLCV window.
        
    Returns:
        Predicted action with confidence.
    """
    # Lazy load model if not loaded
    if not model_state["loaded"]:
        try:
            load_model_artifacts()
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=f"Model not available: {e}")
    
    # Validate window size
    window_size = model_state["config"]["env"]["window_size"]
    if len(request.window) != window_size:
        raise HTTPException(
            status_code=400,
            detail=f"Window size must be {window_size}, got {len(request.window)}"
        )
    
    # Convert window to DataFrame
    window_data = [bar.model_dump() for bar in request.window]
    df = pd.DataFrame(window_data)
    
    # Validate data quality
    if df.isnull().any().any():
        raise HTTPException(status_code=400, detail="Window contains NaN values")
    
    # Generate features
    try:
        feature_names = model_state["config"]["env"]["features"]
        features_df = generate_features(df, feature_names)
        
        # Check for NaNs after feature generation
        if features_df.isnull().any().any():
            raise HTTPException(
                status_code=400,
                detail="Insufficient data to calculate features. Need more historical bars."
            )
        
        features = features_df.values
        
        # Scale features
        if model_state["config"]["env"]["scale_features"]:
            features = model_state["scaler"].transform(features)
        
        # Fill any remaining NaNs with 0
        features = np.nan_to_num(features, nan=0.0)
        
        # Convert to tensor
        state = features.astype(np.float32)
        
        # Get action from agent
        action_id, confidence = model_state["agent"].act(state, greedy=True)
        
        # Map action ID to action name
        action_map = {0: "hold", 1: "buy", 2: "sell"}
        action_name = action_map[action_id]
        
        return ActResponse(
            action=action_name,
            action_id=action_id,
            confidence=float(confidence)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/ingest", response_model=IngestResponse)
async def ingest_historical_data(
    data: List[OHLCVBar],
    symbol: str = "EURUSD"
):
    """Ingest historical OHLCV data and persist to CSV.
    
    Args:
        data: List of OHLCV bars (sent as JSON array in request body).
        symbol: Trading symbol (query parameter, default: EURUSD).
        
    Returns:
        Ingestion status with records saved and file path.
        
    Example:
        POST /ingest?symbol=EURUSD
        Body: [{"timestamp": "2024-01-01T00:00:00Z", "open": 1.1, ...}]
    """
    try:
        # Validate data is not empty
        if not data:
            raise HTTPException(
                status_code=400,
                detail="Data list cannot be empty"
            )
        
        # Save to CSV
        file_path, records_saved = save_to_csv(
            data=data,
            symbol=symbol,
            data_dir=os.getenv("DATA_DIR", "data")
        )
        
        return IngestResponse(
            status="success",
            records_saved=records_saved,
            file_path=file_path
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to ingest data: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
