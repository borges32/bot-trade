"""
API FastAPI para servir predições de trading.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import redis
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Forex Trading Predictor API",
    description="API para predições de trading com LightGBM",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis client
redis_client = None

def get_redis_client():
    """Obtém cliente Redis."""
    global redis_client
    if redis_client is None:
        import os
        redis_host = os.getenv('REDIS_HOST', 'localhost')
        redis_port = int(os.getenv('REDIS_PORT', 6379))
        redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True
        )
    return redis_client


class PredictionResponse(BaseModel):
    """Modelo de resposta da predição."""
    signal: str
    predicted_return: float
    confidence: float
    base_accuracy: Optional[float] = None
    current_price: float
    timestamp: Optional[str] = None
    status: str = "success"


@app.get("/")
async def root():
    """Endpoint raiz."""
    return {
        "message": "Forex Trading Predictor API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "prediction": "/api/prediction",
            "latest": "/api/prediction/latest"
        }
    }


@app.get("/health")
async def health():
    """Health check."""
    try:
        r = get_redis_client()
        r.ping()
        redis_status = "healthy"
    except Exception as e:
        redis_status = f"unhealthy: {str(e)}"
    
    return {
        "status": "healthy",
        "redis": redis_status,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/api/prediction/latest", response_model=PredictionResponse)
async def get_latest_prediction():
    """
    Retorna a última predição armazenada no Redis.
    
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
        logger.error(f"Erro ao buscar predição: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao buscar predição: {str(e)}"
        )


@app.post("/api/prediction")
async def create_prediction(data: Dict[str, Any]):
    """
    Salva uma nova predição no Redis.
    
    Args:
        data: Dicionário com dados da predição
    
    Returns:
        Confirmação de salvamento
    """
    try:
        r = get_redis_client()
        
        # Adiciona timestamp se não existir
        if 'timestamp' not in data:
            data['timestamp'] = datetime.utcnow().isoformat()
        
        # Salva no Redis (sobrescreve anterior)
        r.set('latest_prediction', json.dumps(data))
        
        logger.info(f"Predição salva: {data.get('signal')} - {data.get('predicted_return')}")
        
        return {
            "status": "success",
            "message": "Predição salva com sucesso",
            "data": data
        }
    
    except Exception as e:
        logger.error(f"Erro ao salvar predição: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao salvar predição: {str(e)}"
        )


@app.delete("/api/prediction/latest")
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
        logger.error(f"Erro ao deletar predição: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao deletar predição: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
