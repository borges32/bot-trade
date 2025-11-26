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
    """Single OHLCV bar with optional pre-calculated features."""
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    open: float = Field(..., gt=0, description="Open price")
    high: float = Field(..., gt=0, description="High price")
    low: float = Field(..., gt=0, description="Low price")
    close: float = Field(..., gt=0, description="Close price")
    volume: float = Field(..., ge=0, description="Volume")
    
    # Optional pre-calculated features
    rsi: Optional[float] = Field(None, description="RSI indicator")
    ema_fast: Optional[float] = Field(None, description="Fast EMA")
    ema_slow: Optional[float] = Field(None, description="Slow EMA")
    bb_upper: Optional[float] = Field(None, description="Bollinger Band upper")
    bb_middle: Optional[float] = Field(None, description="Bollinger Band middle")
    bb_lower: Optional[float] = Field(None, description="Bollinger Band lower")
    atr: Optional[float] = Field(None, description="Average True Range")
    momentum_10: Optional[float] = Field(None, description="10-period momentum")
    momentum_20: Optional[float] = Field(None, description="20-period momentum")
    volatility: Optional[float] = Field(None, description="Historical volatility")
    volume_ma: Optional[float] = Field(None, description="Volume moving average")
    macd: Optional[float] = Field(None, description="MACD line")
    macd_signal: Optional[float] = Field(None, description="MACD signal line")
    
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
    """Save OHLCV data to CSV file with optional pre-calculated features.
    
    Args:
        data: List of OHLCV bars with optional features.
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
    records = [bar.model_dump(exclude_none=False) for bar in data]
    
    # Check if file exists
    file_exists = csv_path.exists()
    
    # Define all possible CSV columns (OHLCV + all features)
    fieldnames = [
        "timestamp", "open", "high", "low", "close", "volume",
        "rsi", "ema_fast", "ema_slow", "bb_upper", "bb_middle", "bb_lower",
        "atr", "momentum_10", "momentum_20", "volatility", "volume_ma",
        "macd", "macd_signal"
    ]
    
    # Write to CSV
    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        
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


def calculate_features_for_bars(data: List[OHLCVBar]) -> List[OHLCVBar]:
    """Calculate all technical features for OHLCV bars.
    
    Args:
        data: List of OHLCV bars without features.
        
    Returns:
        List of OHLCV bars with calculated features.
    """
    # Convert to DataFrame
    df = pd.DataFrame([{
        'timestamp': bar.timestamp,
        'open': bar.open,
        'high': bar.high,
        'low': bar.low,
        'close': bar.close,
        'volume': bar.volume
    } for bar in data])
    
    # Import feature calculation functions
    from src.common.features import (
        calculate_rsi, calculate_ema, calculate_bollinger_bands,
        calculate_atr, calculate_momentum, calculate_volatility,
        calculate_volume_ma, calculate_macd
    )
    
    # Calculate features
    df['rsi'] = calculate_rsi(df['close'], period=14)
    df['ema_fast'] = calculate_ema(df['close'], period=12)
    df['ema_slow'] = calculate_ema(df['close'], period=26)
    
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df['close'], period=20)
    df['bb_upper'] = bb_upper
    df['bb_middle'] = bb_middle
    df['bb_lower'] = bb_lower
    
    df['atr'] = calculate_atr(df['high'], df['low'], df['close'], period=14)
    df['momentum_10'] = calculate_momentum(df['close'], period=10)
    df['momentum_20'] = calculate_momentum(df['close'], period=20)
    df['volatility'] = calculate_volatility(df['close'], period=20)
    df['volume_ma'] = calculate_volume_ma(df['volume'], period=20)
    
    macd_line, signal_line = calculate_macd(df['close'])
    df['macd'] = macd_line
    df['macd_signal'] = signal_line
    
    # Convert back to OHLCVBar objects
    enriched_bars = []
    for _, row in df.iterrows():
        bar = OHLCVBar(
            timestamp=row['timestamp'],
            open=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['volume'],
            rsi=None if pd.isna(row['rsi']) else float(row['rsi']),
            ema_fast=None if pd.isna(row['ema_fast']) else float(row['ema_fast']),
            ema_slow=None if pd.isna(row['ema_slow']) else float(row['ema_slow']),
            bb_upper=None if pd.isna(row['bb_upper']) else float(row['bb_upper']),
            bb_middle=None if pd.isna(row['bb_middle']) else float(row['bb_middle']),
            bb_lower=None if pd.isna(row['bb_lower']) else float(row['bb_lower']),
            atr=None if pd.isna(row['atr']) else float(row['atr']),
            momentum_10=None if pd.isna(row['momentum_10']) else float(row['momentum_10']),
            momentum_20=None if pd.isna(row['momentum_20']) else float(row['momentum_20']),
            volatility=None if pd.isna(row['volatility']) else float(row['volatility']),
            volume_ma=None if pd.isna(row['volume_ma']) else float(row['volume_ma']),
            macd=None if pd.isna(row['macd']) else float(row['macd']),
            macd_signal=None if pd.isna(row['macd_signal']) else float(row['macd_signal'])
        )
        enriched_bars.append(bar)
    
    return enriched_bars


@app.post("/ingest/calculate", response_model=IngestResponse)
async def ingest_and_calculate_features(
    data: List[OHLCVBar],
    symbol: str = "EURUSD",
    save_count: int = None
):
    """Ingest OHLCV data, calculate all technical features, and persist to CSV.
    
    This endpoint receives only OHLCV data and automatically calculates:
    - RSI, EMA (fast/slow), Bollinger Bands
    - ATR, Momentum (10/20 periods), Volatility
    - Volume MA, MACD (line/signal)
    
    IMPORTANT: Send enough historical data for feature calculation (minimum ~30 bars),
    but only the last N bars will be saved to avoid duplication.
    
    Args:
        data: List of OHLCV bars (features will be calculated).
        symbol: Trading symbol (query parameter, default: EURUSD).
        save_count: Number of last bars to save (default: 1 - only the most recent).
                   Set to -1 to save all bars (useful for initial data load).
        
    Returns:
        Ingestion status with records saved and file path.
        
    Examples:
        # Real-time: Send last 50 bars for context, save only the newest 1
        POST /ingest/calculate?symbol=USDJPY&save_count=1
        
        # Initial load: Send all data and save all
        POST /ingest/calculate?symbol=USDJPY&save_count=-1
        
        # Batch: Send 100 bars, save last 10 new bars
        POST /ingest/calculate?symbol=USDJPY&save_count=10
    """
    try:
        # Validate data is not empty
        if not data:
            raise HTTPException(
                status_code=400,
                detail="Data list cannot be empty"
            )
        
        # Validate minimum bars for feature calculation
        if len(data) < 30:
            raise HTTPException(
                status_code=400,
                detail="Need at least 30 bars for feature calculation. "
                       f"Got {len(data)} bars."
            )
        
        # Calculate features for all bars
        enriched_data = calculate_features_for_bars(data)
        
        # Determine how many bars to save
        if save_count is None:
            save_count = 1  # Default: save only the last bar
        elif save_count == -1:
            save_count = len(enriched_data)  # Save all
        
        # Get the last N bars to save (avoid duplication)
        bars_to_save = enriched_data[-save_count:] if save_count > 0 else enriched_data
        
        # Save to CSV with features
        file_path, records_saved = save_to_csv(
            data=bars_to_save,
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
            detail=f"Failed to ingest and calculate features: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
