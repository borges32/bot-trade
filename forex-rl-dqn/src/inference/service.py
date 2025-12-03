"""
API FastAPI para sinais de trading baseados em LightGBM + PPO.

Este serviço expõe endpoints HTTP para receber dados recentes de mercado
e retornar decisões de trading (comprar, vender, neutro).
"""

import sys
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import yaml
import uvicorn

# Adiciona path raiz ao PYTHONPATH
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from src.inference.predictor import TradingPredictor

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Carrega configuração
config_file = root_dir / 'config_hybrid.yaml'
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

# Cria app FastAPI
api_config = config.get('api', {})
app = FastAPI(
    title=api_config.get('title', 'Forex Trading Signal API'),
    description=api_config.get('description', 'API para sinais de trading'),
    version=api_config.get('version', '1.0.0')
)

# Preditor global (carregado na inicialização)
predictor: Optional[TradingPredictor] = None


# ============================================================================
# Modelos Pydantic para request/response
# ============================================================================

class Candle(BaseModel):
    """Representa um candle OHLCV."""
    timestamp: str = Field(..., description="Timestamp do candle (ISO format)")
    open: float = Field(..., description="Preço de abertura", gt=0)
    high: float = Field(..., description="Preço máximo", gt=0)
    low: float = Field(..., description="Preço mínimo", gt=0)
    close: float = Field(..., description="Preço de fechamento", gt=0)
    volume: float = Field(..., description="Volume", ge=0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2024-01-01T12:00:00",
                "open": 1.0950,
                "high": 1.0960,
                "low": 1.0945,
                "close": 1.0955,
                "volume": 1000.0
            }
        }


class SignalRequest(BaseModel):
    """Request para obter sinal de trading."""
    candles: List[Candle] = Field(
        ..., 
        description="Lista de candles recentes (mínimo 50, última = mais recente)",
        min_length=50
    )
    current_position: Optional[int] = Field(
        0,
        description="Posição atual: -1=short, 0=flat, 1=long",
        ge=-1,
        le=1
    )
    deterministic: bool = Field(
        True,
        description="Se True, usa política determinística"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "candles": [
                    {
                        "timestamp": "2024-01-01T12:00:00",
                        "open": 1.0950,
                        "high": 1.0960,
                        "low": 1.0945,
                        "close": 1.0955,
                        "volume": 1000.0
                    }
                ] * 50,  # 50 candles
                "current_position": 0,
                "deterministic": True
            }
        }


class SignalResponse(BaseModel):
    """Response com sinal de trading."""
    action: int = Field(..., description="Ação: 0=neutro, 1=comprar, 2=vender")
    action_name: str = Field(..., description="Nome da ação: neutro/comprar/vender")
    lightgbm_signal: float = Field(..., description="Sinal do LightGBM (probabilidade ou retorno)")
    confidence: float = Field(..., description="Confiança na decisão [0, 1]")
    current_state: Dict = Field(..., description="Estado atual da conta")
    timestamp: str = Field(..., description="Timestamp da predição")
    
    class Config:
        json_schema_extra = {
            "example": {
                "action": 1,
                "action_name": "comprar",
                "lightgbm_signal": 0.65,
                "confidence": 0.80,
                "current_state": {
                    "position": 0,
                    "balance": 10000.0,
                    "equity": 10000.0,
                    "unrealized_pnl": 0.0,
                    "realized_pnl": 0.0,
                    "total_return": 0.0,
                    "max_drawdown": 0.0
                },
                "timestamp": "2024-01-01T12:00:00Z"
            }
        }


class ExecuteActionRequest(BaseModel):
    """Request para executar uma ação."""
    action: int = Field(..., description="Ação: 0=neutro, 1=comprar, 2=vender", ge=0, le=2)
    price: float = Field(..., description="Preço de execução", gt=0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "action": 1,
                "price": 1.0955
            }
        }


class StateResponse(BaseModel):
    """Response com estado atual."""
    position: int
    entry_price: float
    balance: float
    equity: float
    unrealized_pnl: float
    realized_pnl: float
    max_equity: float
    max_drawdown: float
    total_return: float


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Carrega modelos na inicialização."""
    global predictor
    
    logger.info("Starting up API...")
    logger.info("Loading models...")
    
    try:
        models_dir = root_dir / config['general']['models_dir']
        
        lightgbm_path = models_dir / 'lightgbm_model'
        ppo_path = models_dir / 'ppo_model'
        
        # Verifica se modelos existem
        if not lightgbm_path.with_suffix('.txt').exists():
            raise FileNotFoundError(f"LightGBM model not found at {lightgbm_path}")
        
        if not ppo_path.with_suffix('.zip').exists():
            raise FileNotFoundError(f"PPO model not found at {ppo_path}")
        
        # Carrega preditor
        predictor = TradingPredictor(
            lightgbm_path=str(lightgbm_path),
            ppo_path=str(ppo_path),
            feature_config=config['features'],
            env_config=config['ppo']['env']
        )
        
        logger.info("Models loaded successfully!")
        logger.info("API ready to accept requests")
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup na finalização."""
    logger.info("Shutting down API...")


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Endpoint raiz com informações da API."""
    return {
        "name": api_config.get('title', 'Forex Trading Signal API'),
        "version": api_config.get('version', '1.0.0'),
        "status": "running",
        "endpoints": {
            "health": "/health",
            "signal": "/signal (POST)",
            "execute": "/execute (POST)",
            "state": "/state (GET)",
            "reset": "/reset (POST)"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return {
        "status": "healthy",
        "models_loaded": True,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/signal", response_model=SignalResponse)
async def get_signal(request: SignalRequest):
    """
    Obtém sinal de trading baseado em candles recentes.
    
    Recebe dados de mercado recentes e retorna a ação recomendada
    pelo sistema híbrido LightGBM + PPO.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Converte candles para lista de dicts
        candles_data = [candle.model_dump() for candle in request.candles]
        
        # Faz predição
        result = predictor.predict_from_recent_data(
            recent_candles=candles_data,
            current_position=request.current_position,
            deterministic=request.deterministic
        )
        
        # Adiciona timestamp
        result['timestamp'] = datetime.now().isoformat() + 'Z'
        
        logger.info(
            f"Signal generated: {result['action_name']} "
            f"(confidence: {result['confidence']:.2f}, "
            f"lgbm_signal: {result['lightgbm_signal']:.4f})"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error generating signal: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/execute")
async def execute_action(request: ExecuteActionRequest):
    """
    Executa uma ação e atualiza o estado interno.
    
    Use este endpoint após receber um sinal para executar a ação
    e manter o estado sincronizado.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        result = predictor.execute_action(
            action=request.action,
            price=request.price
        )
        
        logger.info(
            f"Action executed: {predictor.ACTION_NAMES.get(request.action, 'unknown')} "
            f"at {request.price:.5f}, PnL: {result.get('pnl', 0):.2f}"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error executing action: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state", response_model=StateResponse)
async def get_state():
    """
    Obtém o estado atual da conta.
    
    Retorna informações sobre posição, PnL, equity, etc.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        state = predictor.get_state()
        return state
        
    except Exception as e:
        logger.error(f"Error getting state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
async def reset_state():
    """
    Reseta o estado interno do preditor.
    
    Use para começar uma nova sessão de trading.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        predictor.reset_state()
        logger.info("State reset")
        
        return {
            "status": "success",
            "message": "State reset successfully",
            "new_state": predictor.get_state()
        }
        
    except Exception as e:
        logger.error(f"Error resetting state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    host = api_config.get('host', '0.0.0.0')
    port = api_config.get('port', 8000)
    
    logger.info(f"Starting API server on {host}:{port}")
    
    uvicorn.run(
        "service:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
