# 🔌 Integração com cTrader - Guia Completo

## 📋 Visão Geral

Este guia mostra como integrar o sistema de predição com o cTrader para receber sinais de trading em tempo real.

## 🏗️ Arquitetura

```
┌─────────────────┐
│    cTrader      │  Envia candles via HTTP POST
│   (Broker)      │  
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   API Server    │  POST /api/prediction
│   (FastAPI)     │  Processa e prediz
└────────┬────────┘
         │
         ├──────────────┐
         ▼              ▼
┌─────────────┐  ┌──────────────┐
│    Redis    │  │  Dashboard   │
│  (Cache)    │  │  (Frontend)  │
└─────────────┘  └──────────────┘
```

## 🚀 Configuração Rápida

### 1. Inicie os Serviços

```bash
# Sobe todos os containers
docker-compose up -d

# Verifica status
docker-compose ps
```

### 2. Teste a API

```bash
# Health check
curl http://localhost:8000/health

# Documentação interativa
# Abra: http://localhost:8000/docs
```

### 3. Teste com Exemplo

```bash
python example_ctrader_integration.py
```

## 📡 Endpoint Principal

### `POST /api/prediction`

**URL:** `http://localhost:8000/api/prediction`

**Descrição:** Recebe candles do cTrader, executa predição e salva no Redis.

### Request Body

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
    {
      "timestamp": "2024-01-01T00:30:00",
      "open": 148.65,
      "high": 148.80,
      "low": 148.60,
      "close": 148.70,
      "volume": 1200
    }
    // ... mínimo 50 candles, recomendado 100
  ],
  "current_price": 148.70  // opcional
}
```

### Response

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

### Campos do Request

| Campo | Tipo | Obrigatório | Descrição |
|-------|------|-------------|-----------|
| `candles` | Array | ✅ | Lista de candles (mín. 50) |
| `candles[].timestamp` | String | ✅ | Data/hora ISO 8601 |
| `candles[].open` | Float | ✅ | Preço abertura |
| `candles[].high` | Float | ✅ | Preço máximo |
| `candles[].low` | Float | ✅ | Preço mínimo |
| `candles[].close` | Float | ✅ | Preço fechamento |
| `candles[].volume` | Float | ❌ | Volume (opcional) |
| `current_price` | Float | ❌ | Preço atual |

### Campos do Response

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `signal` | String | BUY, SELL ou NEUTRAL |
| `predicted_return` | Float | Retorno previsto (decimal) |
| `confidence` | Float | Confiança ajustada (0-1) |
| `base_accuracy` | Float | Acurácia histórica do modelo |
| `current_price` | Float | Preço usado na predição |
| `timestamp` | String | Timestamp da predição |
| `status` | String | Status da requisição |

## 🔧 Integração com cTrader

### Opção 1: Via cBot (Recomendado)

```csharp
using System;
using System.Net.Http;
using System.Text;
using cAlgo.API;
using Newtonsoft.Json;

[Robot(TimeZone = TimeZones.UTC)]
public class PredictionBot : Robot
{
    private const string API_URL = "http://seu-servidor:8000/api/prediction";
    private HttpClient httpClient;
    
    protected override void OnStart()
    {
        httpClient = new HttpClient();
        
        // Executa predição a cada novo candle
        Bars.BarOpened += OnBarOpened;
    }
    
    private async void OnBarOpened(BarOpenedEventArgs obj)
    {
        try
        {
            // Pega últimos 100 candles
            var candles = new List<object>();
            for (int i = 99; i >= 0; i--)
            {
                var index = Bars.Count - 1 - i;
                candles.Add(new
                {
                    timestamp = Bars.OpenTimes[index].ToString("yyyy-MM-ddTHH:mm:ss"),
                    open = (double)Bars.OpenPrices[index],
                    high = (double)Bars.HighPrices[index],
                    low = (double)Bars.LowPrices[index],
                    close = (double)Bars.ClosePrices[index],
                    volume = (double)Bars.TickVolumes[index]
                });
            }
            
            var request = new
            {
                candles = candles,
                current_price = (double)Symbol.Bid
            };
            
            var json = JsonConvert.SerializeObject(request);
            var content = new StringContent(json, Encoding.UTF8, "application/json");
            
            var response = await httpClient.PostAsync(API_URL, content);
            var resultJson = await response.Content.ReadAsStringAsync();
            var result = JsonConvert.DeserializeObject<PredictionResponse>(resultJson);
            
            Print($"Sinal: {result.signal}, Confiança: {result.confidence:P2}");
            
            // Executa trade baseado no sinal
            if (result.signal == "BUY" && result.confidence >= 0.40)
            {
                ExecuteMarketOrder(TradeType.Buy, SymbolName, 1000);
            }
            else if (result.signal == "SELL" && result.confidence >= 0.40)
            {
                ExecuteMarketOrder(TradeType.Sell, SymbolName, 1000);
            }
        }
        catch (Exception ex)
        {
            Print($"Erro: {ex.Message}");
        }
    }
}

public class PredictionResponse
{
    public string signal { get; set; }
    public double predicted_return { get; set; }
    public double confidence { get; set; }
    public double base_accuracy { get; set; }
    public double current_price { get; set; }
}
```

### Opção 2: Via Python (Wrapper)

```python
import ctrader_open_api as cot
import requests
import time

# Conecta ao cTrader
client = cot.Client("seu_client_id", "seu_secret")
client.connect()

# Função para obter candles
def get_candles(symbol, timeframe, count=100):
    candles = client.get_bars(symbol, timeframe, count)
    
    return [
        {
            "timestamp": c.time.isoformat(),
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume
        }
        for c in candles
    ]

# Loop principal
while True:
    # Pega candles
    candles = get_candles("USDJPY", "M30", 100)
    
    # Envia para API
    response = requests.post(
        "http://localhost:8000/api/prediction",
        json={"candles": candles}
    )
    
    result = response.json()
    print(f"Sinal: {result['signal']}, Confiança: {result['confidence']:.2%}")
    
    # Aguarda próximo candle (30 minutos)
    time.sleep(1800)
```

## 📊 Monitoramento

### Dashboard em Tempo Real

Acesse: **http://localhost:3000**

O dashboard mostra:
- Último sinal gerado
- Confiança e acurácia
- Retorno previsto
- Atualização automática a cada 10s

### Logs da API

```bash
# Ver logs em tempo real
docker-compose logs -f api

# Últimas 100 linhas
docker-compose logs --tail=100 api
```

### Consultar Última Predição

```bash
curl http://localhost:8000/api/prediction/latest
```

## ⚙️ Configurações

### Variáveis de Ambiente

Edite `docker-compose.yml`:

```yaml
environment:
  - MODEL_PATH=/app/models/hybrid_30m/lightgbm_model.txt
  - CONFIG_PATH=/app/config_30m_optimized.yaml
  - REDIS_HOST=redis
  - REDIS_PORT=6379
```

### Threshold de Confiança

Edite `config_30m_optimized.yaml`:

```yaml
inference:
  min_confidence: 0.40  # 40% (ajuste conforme necessário)
```

## 🔒 Segurança em Produção

### 1. Adicione Autenticação

```python
# api_server.py
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

@app.post("/api/prediction")
async def create_prediction(
    request: PredictionRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    # Valida token
    if credentials.credentials != os.getenv("API_TOKEN"):
        raise HTTPException(status_code=401)
    # ... resto do código
```

### 2. Use HTTPS

Configure nginx como proxy reverso:

```nginx
server {
    listen 443 ssl;
    server_name seu-dominio.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:8000;
    }
}
```

### 3. Rate Limiting

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/api/prediction")
@limiter.limit("10/minute")  # 10 requisições por minuto
async def create_prediction(...):
    # ...
```

## 🐛 Troubleshooting

### Erro: "Mínimo de 50 candles necessários"
- Envie pelo menos 50 candles históricos
- Recomendado: 100 candles para melhor precisão

### Erro: "Modelo não encontrado"
- Verifique se o modelo foi treinado
- Confirme o caminho em `MODEL_PATH`

### Predição sempre NEUTRAL
- Ajuste `min_confidence` para valor menor (ex: 0.30)
- Verifique se o modelo está bem treinado

### API não responde
- Verifique se containers estão rodando: `docker-compose ps`
- Veja logs: `docker-compose logs api`

## 📈 Próximos Passos

1. **Backtesting**: Teste a estratégia com dados históricos
2. **Paper Trading**: Opere em conta demo primeiro
3. **Risk Management**: Implemente stop loss e take profit
4. **Multi-timeframe**: Combine sinais de diferentes períodos
5. **Alertas**: Configure notificações (email, Telegram, etc)

## 💡 Dicas

- Use timeframe de 30m para melhores resultados (modelo treinado para isso)
- Sempre envie pelo menos 100 candles para features mais precisas
- Monitore a acurácia real vs prevista
- Não opere apenas com confiança < 40%
- Combine com análise técnica tradicional

## 📞 Suporte

Para problemas ou dúvidas:
- Veja logs: `docker-compose logs`
- Teste com: `python example_ctrader_integration.py`
- Verifique docs: http://localhost:8000/docs
