# 📚 API Consolidada - Guia Completo

A API agora está **consolidada** em um único servidor com endpoints **LightGBM** e **DQN/RL**.

## 🌐 Acesso

- **API**: http://localhost:8000
- **Documentação Interativa**: http://localhost:8000/docs
- **Dashboard**: http://localhost:3000

## 📊 Estrutura da API

```
┌─────────────────────────────────────┐
│     API Consolidada (porta 8000)    │
├─────────────────────────────────────┤
│                                     │
│  📈 LightGBM (Predição de Retorno)  │
│    POST   /api/prediction           │
│    GET    /api/prediction/latest    │
│    DELETE /api/prediction/latest    │
│                                     │
│  🤖 DQN/RL (Ações de Trading)       │
│    POST   /dqn/act                  │
│    POST   /dqn/ingest               │
│    POST   /dqn/ingest/calculate     │
│                                     │
│  🏥 Sistema                          │
│    GET    /health                   │
│    GET    /                         │
└─────────────────────────────────────┘
```

## 🚀 Como Usar

### 1. Iniciar o Servidor

```bash
# Via Docker
docker-compose up -d

# Ou localmente
python api_server.py
```

### 2. Verificar Status

```bash
curl http://localhost:8000/health
```

**Resposta:**
```json
{
  "status": "healthy",
  "redis": "healthy",
  "lightgbm_loaded": true,
  "dqn_loaded": false
}
```

## 📡 Endpoints Detalhados

### 🔹 LightGBM - Predição de Retorno

#### POST /api/prediction
Prediz direção e retorno do preço usando LightGBM.

**Request:**
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
    }
    // ... mínimo 50 candles
  ],
  "current_price": 148.70  // opcional
}
```

**Response:**
```json
{
  "signal": "BUY",
  "predicted_return": 0.0020,
  "confidence": 0.11,
  "base_accuracy": 0.552,
  "current_price": 148.70,
  "timestamp": "2024-12-14T10:30:00.000Z",
  "status": "success"
}
```

**Exemplo:**
```bash
curl -X POST http://localhost:8000/api/prediction \
  -H "Content-Type: application/json" \
  -d @candles.json
```

#### GET /api/prediction/latest
Consulta última predição salva no Redis.

**Response:**
```json
{
  "signal": "BUY",
  "predicted_return": 0.0020,
  "confidence": 0.11,
  "base_accuracy": 0.552,
  "current_price": 148.70,
  "timestamp": "2024-12-14T10:30:00.000Z",
  "status": "success"
}
```

### 🔹 DQN/RL - Ações de Trading

#### POST /dqn/act
Prediz ação (buy/sell/hold) usando modelo DQN.

**Request:**
```json
{
  "symbol": "EURUSD",
  "window": [
    {
      "timestamp": "2024-01-01T00:00:00Z",
      "open": 1.1000,
      "high": 1.1050,
      "low": 1.0950,
      "close": 1.1020,
      "volume": 5000,
      // Features opcionais (calculadas automaticamente se omitidas)
      "rsi": 55.2,
      "ema_fast": 1.1015,
      "ema_slow": 1.1005
    }
    // ... número de candles = window_size do modelo
  ]
}
```

**Response:**
```json
{
  "action": "buy",
  "action_id": 1,
  "confidence": 0.78
}
```

**Mapeamento de Ações:**
- `0` = `hold` (manter)
- `1` = `buy` (comprar)
- `2` = `sell` (vender)

#### POST /dqn/ingest
Ingere dados históricos e salva em CSV.

**Request:**
```json
[
  {
    "timestamp": "2024-01-01T00:00:00Z",
    "open": 1.1000,
    "high": 1.1050,
    "low": 1.0950,
    "close": 1.1020,
    "volume": 5000
  }
  // ... mais candles
]
```

**Query Params:**
- `symbol`: Símbolo (padrão: EURUSD)

**Response:**
```json
{
  "status": "success",
  "records_saved": 100,
  "file_path": "data/eurusd_history.csv"
}
```

#### POST /dqn/ingest/calculate
Ingere dados, **calcula features** automaticamente e salva.

**Request:**
```json
[
  {
    "timestamp": "2024-01-01T00:00:00Z",
    "open": 1.1000,
    "high": 1.1050,
    "low": 1.0950,
    "close": 1.1020,
    "volume": 5000
  }
  // ... mínimo 30 candles para cálculo de features
]
```

**Query Params:**
- `symbol`: Símbolo (padrão: EURUSD)
- `save_count`: Quantos dos últimos bars salvar
  - `1` (padrão): salva apenas o mais recente
  - `-1`: salva todos
  - `N`: salva os últimos N

**Response:**
```json
{
  "status": "success",
  "records_saved": 1,
  "file_path": "data/eurusd_history.csv"
}
```

**Features Calculadas:**
- RSI (14 períodos)
- EMA rápida (12) e lenta (26)
- Bollinger Bands (20 períodos)
- ATR (14 períodos)
- Momentum (10 e 20 períodos)
- Volatilidade (20 períodos)
- Volume MA (20 períodos)
- MACD (linha e signal)

## 🏥 Health Check

### GET /health

**Response:**
```json
{
  "status": "healthy",
  "redis": "healthy",
  "lightgbm_loaded": true,
  "dqn_loaded": false
}
```

## 📊 Documentação Interativa

Acesse http://localhost:8000/docs para:
- ✅ Ver **TODOS** os endpoints em um lugar
- ✅ Testar requisições diretamente no navegador
- ✅ Ver schemas detalhados de request/response
- ✅ Copiar exemplos de código

## 🎯 Casos de Uso

### Caso 1: Predição LightGBM em Tempo Real

```python
import requests

# Pega últimos 100 candles do seu sistema
candles = get_latest_candles(100)

# Envia para API
response = requests.post(
    "http://localhost:8000/api/prediction",
    json={"candles": candles}
)

result = response.json()
print(f"Sinal: {result['signal']}")
print(f"Confiança: {result['confidence']:.2%}")

# Dashboard já mostra automaticamente!
```

### Caso 2: Ação DQN/RL

```python
import requests

# Pega janela de candles (tamanho específico do modelo)
window = get_candle_window(window_size=50)

# Envia para API
response = requests.post(
    "http://localhost:8000/dqn/act",
    json={
        "symbol": "USDJPY",
        "window": window
    }
)

result = response.json()
print(f"Ação: {result['action']}")
print(f"Confiança: {result['confidence']:.2%}")
```

### Caso 3: Ingestão com Cálculo de Features

```python
import requests

# Tempo real: envia contexto, salva apenas o novo
candles = get_latest_candles(50)

response = requests.post(
    "http://localhost:8000/dqn/ingest/calculate?symbol=USDJPY&save_count=1",
    json=candles
)

print(f"Salvos: {response.json()['records_saved']} registros")
```

## 🔧 Configuração

### Variáveis de Ambiente

```yaml
# docker-compose.yml
environment:
  # LightGBM
  - MODEL_PATH=/app/models/hybrid_30m/lightgbm_model.txt
  - CONFIG_PATH=/app/config_30m_optimized.yaml
  
  # DQN (opcional)
  - DQN_MODEL_PATH=/app/artifacts/dqn.pt
  - DQN_SCALER_PATH=/app/artifacts/feature_state.json
  - DQN_CONFIG_PATH=/app/artifacts/config.yaml
  
  # Redis
  - REDIS_HOST=redis
  - REDIS_PORT=6379
  
  # Dados
  - DATA_DIR=/app/data
```

## 🏷️ Tags nos Docs

Os endpoints são organizados por tags:

- **LightGBM**: Predições de retorno com LightGBM
- **DQN/RL**: Ações de trading com Reinforcement Learning

## 💡 Dicas

1. **Use LightGBM** para predições de retorno e direção
2. **Use DQN** para decisões de ação (buy/sell/hold)
3. **Combine ambos** para estratégias híbridas
4. **Monitore** o dashboard em http://localhost:3000
5. **Teste** na documentação interativa em /docs

## ⚠️ Observações

- **LightGBM** precisa de mínimo 50 candles
- **DQN** precisa de window_size específico do modelo
- **Features** são calculadas automaticamente se não fornecidas
- **Redis** armazena apenas a última predição do LightGBM
- **CSV** persiste todos os dados ingeridos via DQN

## 🔗 Links Úteis

- Dashboard: http://localhost:3000
- API Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Health: http://localhost:8000/health
- Latest Prediction: http://localhost:8000/api/prediction/latest

---

✅ **Agora todos os endpoints estão em um único lugar com documentação completa!**
