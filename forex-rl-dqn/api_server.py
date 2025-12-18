"""
API FastAPI para servir predições de trading.
Consolida endpoints LightGBM e DQN/RL.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any, List
import redis
import json
import logging
from datetime import datetime
import pandas as pd
import yaml
import os
from pathlib import Path
import csv
import numpy as np

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.inference.predictor import TradingPredictor

# Optional DQN imports (only if torch is available)
try:
    import torch
    from src.rl.agent import DQNAgent
    from src.common.features import FeatureScaler, generate_features
    from src.common.utils import get_device
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. DQN endpoints will be disabled.")

app = FastAPI(
    title="Forex Trading API - Completa",
    description="API unificada com predições LightGBM e DQN/RL",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware para logar todas as requisições
from fastapi import Request
import time

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log todas as requisições recebidas."""
    start_time = time.time()
    
    # Log da requisição
    logger.info(f">>> REQUISIÇÃO RECEBIDA: {request.method} {request.url.path}")
    logger.info(f">>> Headers: {dict(request.headers)}")
    logger.info(f">>> Query params: {dict(request.query_params)}")
    
    # Processa a requisição
    response = await call_next(request)
    
    # Log da resposta
    process_time = time.time() - start_time
    logger.info(f"<<< RESPOSTA: Status {response.status_code} | Tempo: {process_time:.3f}s")
    
    return response

# Redis client
redis_client = None

# Predictor instance (LightGBM)
predictor = None

# DQN Model state
dqn_model_state = {
    "agent": None,
    "scaler": None,
    "config": None,
    "device": None,
    "loaded": False,
}


def get_redis_client():
    """Obtém cliente Redis."""
    global redis_client
    if redis_client is None:
        redis_host = os.getenv('REDIS_HOST', 'localhost')
        redis_port = int(os.getenv('REDIS_PORT', 6379))
        redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True
        )
    return redis_client


def get_predictor():
    """Obtém instância do predictor LightGBM."""
    global predictor
    if predictor is None:
        # Carrega configuração do melhor resultado (JSON ou YAML)
        config_base_path = os.getenv('CONFIG_PATH', 'optimization_results/usdjpy_30m')
        config_path_obj = Path(config_base_path)
        
        # Se for diretório, procura best_config.json ou best_config.yaml
        if config_path_obj.is_dir():
            logger.info(f"CONFIG_PATH aponta para diretório: {config_path_obj}")
            # Prioriza JSON (parâmetros otimizados) sobre YAML
            candidate_files = [
                config_path_obj / 'best_config.json',
                config_path_obj / 'best_config.yaml',
            ]
            existing = [c for c in candidate_files if c.exists() and c.is_file()]
            if existing:
                config_path_obj = existing[0]
                logger.info(f"Usando config encontrado: {config_path_obj}")
            else:
                raise FileNotFoundError(
                    f"CONFIG_PATH é um diretório e nenhum arquivo de config foi encontrado em {candidate_files}"
                )
        
        config_path = str(config_path_obj)
        model_path = os.getenv('MODEL_PATH', 'models/hybrid_30m/lightgbm_model.txt')
        
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config não encontrado: {config_path}")
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
        
        # Carrega configuração (JSON ou YAML)
        if config_path.endswith('.json'):
            logger.info(f"Carregando parâmetros otimizados do JSON: {config_path}")
            with open(config_path, 'r') as f:
                best_params = json.load(f)
            
            # Reconstrói config completo a partir dos parâmetros otimizados
            config = {
                'general': {
                    'seed': 42,
                    'verbose': True,
                },
                'data': {
                    'timestamp_col': 'timestamp',
                    'open_col': 'open',
                    'high_col': 'high',
                    'low_col': 'low',
                    'close_col': 'close',
                    'volume_col': 'volume',
                    'lookback_window': 72,
                },
                'features': {
                    # Features booleanas do resultado otimizado
                    'use_rsi': best_params.get('use_rsi', False),
                    'rsi_period': 14,
                    'use_ema': best_params.get('use_ema', False),
                    'ema_periods': [12, 26, 50],
                    'ema_fast': 12,
                    'ema_slow': 26,
                    'use_macd': best_params.get('use_macd', False),
                    'macd_fast': 12,
                    'macd_slow': 26,
                    'macd_signal': 9,
                    'use_bollinger': best_params.get('use_bollinger', False),
                    'bb_period': 20,
                    'bb_std': 2.0,
                    'use_atr': best_params.get('use_atr', False),
                    'atr_period': 14,
                    'use_momentum': best_params.get('use_momentum', False),
                    'momentum_periods': [10, 20],
                    'use_sma': best_params.get('use_sma', False),
                    'sma_periods': [20, 50],
                    'use_stochastic': best_params.get('use_stochastic', False),
                    'stoch_k': 14,
                    'stoch_d': 3,
                    'use_adx': best_params.get('use_adx', False),
                    'adx_period': 14,
                    'use_volatility': best_params.get('use_volatility', False),
                    'volatility_window': 20,
                    'use_volume_features': best_params.get('use_volume_features', False),
                    'volume_ma_period': 20,
                    'use_returns': best_params.get('use_returns', False),
                    'return_periods': [1, 3, 6, 12, 24],
                },
                'lightgbm': {
                    'model_type': 'regressor',
                    'prediction_horizon': int(best_params.get('prediction_horizon', 10)),
                    'params': {
                        'objective': 'regression',
                        'metric': 'rmse',
                        'boosting_type': 'gbdt',
                        'num_leaves': int(best_params.get('num_leaves', 50)),
                        'max_depth': int(best_params.get('max_depth', 6)),
                        'learning_rate': float(best_params.get('learning_rate', 0.03)),
                        'n_estimators': int(best_params.get('n_estimators', 500)),
                        'min_child_samples': int(best_params.get('min_child_samples', 20)),
                        'subsample': float(best_params.get('subsample', 0.8)),
                        'subsample_freq': 1,
                        'colsample_bytree': float(best_params.get('colsample_bytree', 0.8)),
                        'reg_alpha': float(best_params.get('reg_alpha', 0.3)),
                        'reg_lambda': float(best_params.get('reg_lambda', 0.3)),
                        'random_state': 42,
                        'n_jobs': -1,
                        'verbose': -1,
                    },
                    'early_stopping_rounds': 50,
                    'save_feature_importance': True,
                },
                'inference': {
                    'lightgbm_model_path': model_path,
                    'min_confidence': float(best_params.get('min_confidence', 0.02)),
                }
            }
            
            logger.info("Config reconstruído a partir dos parâmetros otimizados (JSON)")
            logger.info(f"  - Features ativas: RSI={config['features']['use_rsi']}, "
                       f"EMA={config['features']['use_ema']}, MACD={config['features']['use_macd']}, "
                       f"Bollinger={config['features']['use_bollinger']}, ATR={config['features']['use_atr']}")
            logger.info(f"  - Prediction Horizon: {config['lightgbm']['prediction_horizon']}")
            logger.info(f"  - Learning Rate: {config['lightgbm']['params']['learning_rate']}")
            logger.info(f"  - Num Leaves: {config['lightgbm']['params']['num_leaves']}")
            
        else:
            # Carrega YAML normalmente
            logger.info(f"Carregando config YAML: {config_path}")
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        
        logger.info(f"Carregando predictor LightGBM com modelo: {model_path}")
        predictor = TradingPredictor(
            lightgbm_path=model_path,
            config=config,
            enable_redis=False  # API gerencia Redis manualmente
        )
        logger.info("Predictor LightGBM carregado com sucesso")
    
    return predictor


def load_dqn_model_artifacts(
    model_path: str = "artifacts/dqn.pt",
    scaler_path: str = "artifacts/feature_state.json",
    config_path: str = "artifacts/config.yaml",
) -> None:
    """Carrega modelo DQN, scaler e config."""
    global dqn_model_state
    
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not available. Cannot load DQN model.")
    
    if dqn_model_state["loaded"]:
        return
    
    # Check if files exist
    model_path_obj = Path(model_path)
    scaler_path_obj = Path(scaler_path)
    config_path_obj = Path(config_path)
    
    if not model_path_obj.exists():
        raise FileNotFoundError(f"DQN Model not found at {model_path}")
    if not scaler_path_obj.exists():
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    if not config_path_obj.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")
    
    # Load config
    with open(config_path_obj, "r") as f:
        config = yaml.safe_load(f)
    
    # Get device
    device = get_device("cpu")
    
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
        epsilon_start=0.0,
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
    dqn_model_state["agent"] = agent
    dqn_model_state["scaler"] = scaler
    dqn_model_state["config"] = config
    dqn_model_state["device"] = device
    dqn_model_state["loaded"] = True
    
    logger.info(f"DQN Model loaded from {model_path}")


# ============================================================================
# MODELOS PYDANTIC
# ============================================================================

class Candle(BaseModel):
    """Modelo de candle individual (LightGBM)."""
    timestamp: str = Field(..., description="Timestamp do candle (ISO format ou string)")
    open: float = Field(..., description="Preço de abertura")
    high: float = Field(..., description="Preço máximo")
    low: float = Field(..., description="Preço mínimo")
    close: float = Field(..., description="Preço de fechamento")
    volume: float = Field(default=0, description="Volume (opcional)")


class OHLCVBar(BaseModel):
    """Single OHLCV bar com features opcionais (DQN)."""
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


class PredictionRequest(BaseModel):
    """Modelo de requisição para predição LightGBM."""
    candles: List[Candle] = Field(..., description="Lista de candles históricos (mínimo 50)", min_length=50)
    current_price: Optional[float] = Field(None, description="Preço atual (opcional)")


class PredictionResponse(BaseModel):
    """Modelo de resposta da predição LightGBM."""
    signal: str
    predicted_return: float
    confidence: float
    base_accuracy: Optional[float] = None
    current_price: float
    timestamp: Optional[str] = None
    status: str = "success"


class ActRequest(BaseModel):
    """Request for action prediction (DQN)."""
    symbol: str = Field(..., description="Trading symbol (e.g., EURUSD)")
    window: List[OHLCVBar] = Field(..., min_length=1, description="Window of OHLCV bars")


class ActResponse(BaseModel):
    """Response with action prediction (DQN)."""
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
    redis: str
    lightgbm_loaded: bool
    dqn_loaded: bool


@app.get("/")
async def root():
    """Endpoint raiz."""
    return {
        "message": "Forex Trading API - Completa",
        "version": "2.0.0",
        "models": {
            "lightgbm": "Predição de retorno e direção",
            "dqn": "Reinforcement Learning para ações de trading"
        },
        "endpoints": {
            "health": "/health",
            "lightgbm_prediction": "/api/prediction (POST)",
            "lightgbm_latest": "/api/prediction/latest (GET)",
            "dqn_action": "/dqn/act (POST)",
            "dqn_ingest": "/dqn/ingest (POST)",
            "dqn_ingest_calculate": "/dqn/ingest/calculate (POST)",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check completo."""
    try:
        r = get_redis_client()
        r.ping()
        redis_status = "healthy"
    except Exception as e:
        redis_status = f"unhealthy: {str(e)}"
    
    return HealthResponse(
        status="healthy",
        redis=redis_status,
        lightgbm_loaded=predictor is not None,
        dqn_loaded=dqn_model_state["loaded"] if TORCH_AVAILABLE else False
    )


# ============================================================================
# ENDPOINTS LIGHTGBM
# ============================================================================

@app.post("/api/prediction", response_model=PredictionResponse, tags=["LightGBM"])
@app.post("/prediction", response_model=PredictionResponse, tags=["LightGBM"], include_in_schema=False)
async def create_prediction(request: PredictionRequest):
    """
    Recebe candles do cTrader, faz predição LightGBM e salva no Redis.
    
    Este é o endpoint principal para integração em tempo real com LightGBM.
    Recebe dados de candles, executa a predição, salva o resultado no Redis e retorna a predição.
    
    Args:
        request: PredictionRequest com lista de candles
    
    Returns:
        PredictionResponse com resultado da predição
    
    Example:
        ```json
        {
            "candles": [
                {
                    "timestamp": "2024-01-01T00:00:00",
                    "open": 148.50,
                    "high": 148.75,
                    "low": 148.40,
                    "close": 148.65,
                    "volume": 1000
                },
                ...
            ],
            "current_price": 148.70
        }
        ```
    """
    # Log de entrada ANTES de qualquer validação
    logger.info("="*80)
    logger.info("[API] Requisição recebida no endpoint /api/prediction")
    logger.info(f"[API] Número de candles na requisição: {len(request.candles)}")
    logger.info(f"[API] Current price: {request.current_price}")
    logger.info("="*80)
    
    try:
        # 1. Valida quantidade de candles
        if len(request.candles) < 50:
            logger.warning(f"[VALIDAÇÃO] Mínimo de 50 candles necessários. Recebidos: {len(request.candles)}")
            raise HTTPException(
                status_code=400,
                detail=f"Mínimo de 50 candles necessários. Recebidos: {len(request.candles)}"
            )
        
        logger.info(f"[LightGBM] Recebidos {len(request.candles)} candles para predição")
        
        # 2. Converte candles para DataFrame
        candles_data = [candle.dict() for candle in request.candles]
        df = pd.DataFrame(candles_data)
        
        # Converte timestamp para datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"[LightGBM] Período: {df['timestamp'].min()} até {df['timestamp'].max()}")
        
        # 3. Obtém predictor
        pred = get_predictor()
        
        # 4. Faz predição
        result = pred.predict(
            candles=df,
            current_price=request.current_price
        )
        
        # 5. Adiciona timestamp
        result['timestamp'] = datetime.utcnow().isoformat()
        result['status'] = 'success'
        
        logger.info(f"[LightGBM] Predição: {result['signal']} | "
                   f"Retorno: {result['predicted_return']:.6f} | "
                   f"Confiança: {result['confidence']:.2%}")
        
        # 6. Salva no Redis
        r = get_redis_client()
        r.set('latest_prediction', json.dumps(result))
        logger.info("[LightGBM] Predição salva no Redis")
        
        return PredictionResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[LightGBM] Erro: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao processar predição: {str(e)}"
        )


@app.get("/api/prediction/latest", response_model=PredictionResponse, tags=["LightGBM"])
@app.get("/prediction/latest", response_model=PredictionResponse, tags=["LightGBM"], include_in_schema=False)
async def get_latest_prediction():
    """
    Retorna a última predição LightGBM armazenada no Redis.
    
    Returns:
        PredictionResponse com dados da última predição
    """
    try:
        r = get_redis_client()
        
        # Busca última predição no Redis
        prediction_data = r.get('latest_prediction')
        
        if prediction_data is None:
            raise HTTPException(
                status_code=404,
                detail="Nenhuma predição disponível. Execute uma predição primeiro."
            )
        
        # Parse JSON
        prediction = json.loads(prediction_data)
        
        return PredictionResponse(**prediction)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[LightGBM] Erro ao buscar predição: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao buscar predição: {str(e)}"
        )


@app.delete("/api/prediction/latest", tags=["LightGBM"])
@app.delete("/prediction/latest", tags=["LightGBM"], include_in_schema=False)
async def delete_latest_prediction():
    """Remove a última predição do Redis."""
    try:
        r = get_redis_client()
        deleted = r.delete('latest_prediction')
        
        if deleted:
            return {"status": "success", "message": "Predição removida"}
        else:
            raise HTTPException(status_code=404, detail="Nenhuma predição encontrada")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[LightGBM] Erro ao deletar: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao deletar predição: {str(e)}"
        )


# ============================================================================
# ENDPOINTS DQN/RL
# ============================================================================

def save_to_csv(data: List[OHLCVBar], symbol: str, data_dir: str = "data") -> tuple[str, int]:
    """Salva dados OHLCV em CSV com features opcionais."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    csv_filename = f"{symbol.lower()}_history.csv"
    csv_path = data_path / csv_filename
    
    records = [bar.model_dump(exclude_none=False) for bar in data]
    file_exists = csv_path.exists()
    
    fieldnames = [
        "timestamp", "open", "high", "low", "close", "volume",
        "rsi", "ema_fast", "ema_slow", "bb_upper", "bb_middle", "bb_lower",
        "atr", "momentum_10", "momentum_20", "volatility", "volume_ma",
        "macd", "macd_signal"
    ]
    
    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        writer.writerows(records)
    
    return str(csv_path), len(records)


def calculate_features_for_bars(data: List[OHLCVBar]) -> List[OHLCVBar]:
    """Calcula todas as features técnicas para OHLCV bars."""
    df = pd.DataFrame([{
        'timestamp': bar.timestamp,
        'open': bar.open,
        'high': bar.high,
        'low': bar.low,
        'close': bar.close,
        'volume': bar.volume
    } for bar in data])
    
    from src.common.features import (
        calculate_rsi, calculate_ema, calculate_bollinger_bands,
        calculate_atr, calculate_momentum, calculate_volatility,
        calculate_volume_ma, calculate_macd
    )
    
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


@app.post("/dqn/act", response_model=ActResponse, tags=["DQN/RL"])
async def predict_action(request: ActRequest):
    """
    Prediz ação de trading usando modelo DQN/RL.
    
    Args:
        request: Requisição com janela de OHLCV bars
        
    Returns:
        Ação prevista (buy, sell, hold) com confiança
    """
    if not TORCH_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="PyTorch não disponível. Instale torch para usar endpoints DQN."
        )
    
    if not dqn_model_state["loaded"]:
        try:
            load_dqn_model_artifacts()
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=f"DQN Model não disponível: {e}")
        except ImportError as e:
            raise HTTPException(status_code=503, detail=str(e))
    
    window_size = dqn_model_state["config"]["env"]["window_size"]
    if len(request.window) != window_size:
        raise HTTPException(
            status_code=400,
            detail=f"Window size deve ser {window_size}, recebido {len(request.window)}"
        )
    
    window_data = [bar.model_dump() for bar in request.window]
    df = pd.DataFrame(window_data)
    
    if df.isnull().any().any():
        raise HTTPException(status_code=400, detail="Window contém valores NaN")
    
    try:
        from src.common.features import generate_features
        
        feature_names = dqn_model_state["config"]["env"]["features"]
        features_df = generate_features(df, feature_names)
        
        if features_df.isnull().any().any():
            raise HTTPException(
                status_code=400,
                detail="Dados insuficientes para calcular features. Envie mais candles históricos."
            )
        
        features = features_df.values
        
        if dqn_model_state["config"]["env"]["scale_features"]:
            features = dqn_model_state["scaler"].transform(features)
        
        features = np.nan_to_num(features, nan=0.0)
        state = features.astype(np.float32)
        
        action_id, confidence = dqn_model_state["agent"].act(state, greedy=True)
        
        action_map = {0: "hold", 1: "buy", 2: "sell"}
        action_name = action_map[action_id]
        
        logger.info(f"[DQN] Ação: {action_name} | Confiança: {confidence:.2%}")
        
        return ActResponse(
            action=action_name,
            action_id=action_id,
            confidence=float(confidence)
        )
        
    except Exception as e:
        logger.error(f"[DQN] Erro na inferência: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro na inferência: {str(e)}")


@app.post("/dqn/ingest", response_model=IngestResponse, tags=["DQN/RL"])
async def ingest_historical_data(
    data: List[OHLCVBar],
    symbol: str = "EURUSD"
):
    """
    Ingere dados históricos OHLCV e persiste em CSV.
    
    Args:
        data: Lista de OHLCV bars
        symbol: Símbolo de trading (padrão: EURUSD)
        
    Returns:
        Status da ingestão com registros salvos e caminho do arquivo
    """
    # Este endpoint não precisa de PyTorch, apenas salva dados
    try:
        if not data:
            raise HTTPException(status_code=400, detail="Lista de dados vazia")
        
        file_path, records_saved = save_to_csv(
            data=data,
            symbol=symbol,
            data_dir=os.getenv("DATA_DIR", "data")
        )
        
        logger.info(f"[DQN] Ingeridos {records_saved} registros para {symbol}")
        
        return IngestResponse(
            status="success",
            records_saved=records_saved,
            file_path=file_path
        )
        
    except Exception as e:
        logger.error(f"[DQN] Erro na ingestão: {e}")
        raise HTTPException(status_code=500, detail=f"Falha ao ingerir dados: {str(e)}")


@app.post("/dqn/ingest/calculate", response_model=IngestResponse, tags=["DQN/RL"])
async def ingest_and_calculate_features(
    data: List[OHLCVBar],
    symbol: str = "EURUSD",
    save_count: int = None
):
    """
    Ingere OHLCV, calcula features técnicas e persiste em CSV.
    
    Calcula automaticamente: RSI, EMA, Bollinger Bands, ATR, Momentum, Volatility, Volume MA, MACD.
    
    IMPORTANTE: Envie dados históricos suficientes (mínimo 30 bars) para cálculo de features,
    mas apenas os últimos N bars serão salvos para evitar duplicação.
    
    Args:
        data: Lista de OHLCV bars (features serão calculadas)
        symbol: Símbolo de trading (padrão: EURUSD)
        save_count: Número de últimos bars a salvar (padrão: 1 - apenas o mais recente)
                   Use -1 para salvar todos (útil para carga inicial)
        
    Returns:
        Status da ingestão
        
    Examples:
        # Tempo real: Envia 50 bars para contexto, salva apenas o mais novo
        POST /dqn/ingest/calculate?symbol=USDJPY&save_count=1
        
        # Carga inicial: Envia todos os dados e salva todos
        POST /dqn/ingest/calculate?symbol=USDJPY&save_count=-1
    """
    try:
        if not data:
            raise HTTPException(status_code=400, detail="Lista de dados vazia")
        
        if len(data) < 30:
            raise HTTPException(
                status_code=400,
                detail=f"Mínimo de 30 bars necessários para cálculo de features. Recebido: {len(data)}"
            )
        
        enriched_data = calculate_features_for_bars(data)
        
        if save_count is None:
            save_count = 1
        elif save_count == -1:
            save_count = len(enriched_data)
        
        bars_to_save = enriched_data[-save_count:] if save_count > 0 else enriched_data
        
        file_path, records_saved = save_to_csv(
            data=bars_to_save,
            symbol=symbol,
            data_dir=os.getenv("DATA_DIR", "data")
        )
        
        logger.info(f"[DQN] Calculadas features e salvos {records_saved} registros para {symbol}")
        
        return IngestResponse(
            status="success",
            records_saved=records_saved,
            file_path=file_path
        )
        
    except Exception as e:
        logger.error(f"[DQN] Erro ao calcular features: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Falha ao ingerir e calcular features: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    # Log de inicialização
    logger.info("="*80)
    logger.info("INICIANDO SERVIDOR API")
    logger.info("="*80)
    logger.info("Endpoints registrados:")
    for route in app.routes:
        if hasattr(route, 'methods') and hasattr(route, 'path'):
            logger.info(f"  {list(route.methods)} {route.path}")
    logger.info("="*80)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
