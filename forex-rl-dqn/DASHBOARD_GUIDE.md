# Sistema de Predição com Dashboard Web

Este sistema permite executar predições de trading e visualizá-las em um dashboard web em tempo real.

## 🏗️ Arquitetura

```
┌─────────────────┐
│   Frontend      │  http://localhost:3000
│   (Nginx)       │  Interface visual
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   API Server    │  http://localhost:8000
│   (FastAPI)     │  Endpoints REST
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     Redis       │  localhost:6379
│   (Cache)       │  Armazena última predição
└─────────────────┘
```

## 🚀 Como Usar

### 1. Iniciar os Serviços

```bash
# Sobe todos os containers (Redis, API, Frontend)
docker-compose up -d

# Verifica se estão rodando
docker-compose ps
```

### 2. Executar uma Predição

```bash
# Instala redis localmente se ainda não tiver
pip install redis

# Executa predição e salva no Redis
python run_prediction.py
```

### 3. Visualizar no Dashboard

Abra seu navegador em: **http://localhost:3000**

O dashboard mostra:
- 🎯 Sinal de trading (BUY/SELL/NEUTRAL)
- 📈 Retorno previsto
- 📊 Acurácia base do modelo
- 💯 Confiança ajustada
- ⚡ Força do sinal
- 💰 Preço atual
- 📝 Interpretação

### 4. Acessar a API Diretamente

```bash
# Endpoint de saúde
curl http://localhost:8000/health

# Buscar última predição
curl http://localhost:8000/api/prediction/latest

# Ver docs interativas
# Abra: http://localhost:8000/docs
```

## 📡 Endpoints da API

### `GET /api/prediction/latest`
Retorna a última predição salva no Redis.

**Resposta:**
```json
{
  "signal": "BUY",
  "predicted_return": 0.0020,
  "confidence": 0.11,
  "base_accuracy": 0.552,
  "current_price": 148.5000,
  "timestamp": "2025-12-14T10:30:00.000Z",
  "status": "success"
}
```

### `POST /api/prediction`
Salva uma nova predição (usado internamente pelo predictor).

### `GET /health`
Verifica saúde da API e conexão com Redis.

## 🔄 Atualização Automática

O dashboard atualiza automaticamente a cada **10 segundos**.

Para atualizar manualmente, clique no botão **"🔄 Atualizar Predição"**.

## 🛠️ Desenvolvimento Local (sem Docker)

### 1. Inicie o Redis
```bash
# Linux/Mac
redis-server

# Ou via Docker apenas Redis
docker run -d -p 6379:6379 redis:7-alpine
```

### 2. Inicie a API
```bash
python api_server.py
# API rodando em http://localhost:8000
```

### 3. Sirva o Frontend
```bash
cd frontend
python -m http.server 3000
# Frontend em http://localhost:3000
```

### 4. Execute Predições
```bash
python run_prediction.py
```

## 📁 Estrutura de Arquivos

```
forex-rl-dqn/
├── api_server.py              # API FastAPI
├── run_prediction.py          # Script para executar predições
├── docker-compose.yml         # Orquestração de containers
├── Dockerfile                 # Container da API
├── requirements.txt           # Dependências (inclui redis)
├── frontend/
│   ├── index.html            # Dashboard web
│   ├── Dockerfile            # Container do frontend
│   └── nginx.conf            # Configuração Nginx
└── src/
    └── inference/
        └── predictor.py      # Salva predições no Redis
```

## 🔧 Configuração

### Variáveis de Ambiente

```bash
# Redis
REDIS_HOST=localhost  # ou 'redis' no docker-compose
REDIS_PORT=6379

# API
API_URL=http://localhost:8000
```

### Threshold de Confiança

Ajuste em `config_30m_optimized.yaml`:
```yaml
inference:
  min_confidence: 0.40  # 40%
```

## 🎨 Personalização do Frontend

Edite `frontend/index.html` para:
- Mudar cores e estilo
- Ajustar intervalo de atualização (linha: `setInterval(loadPrediction, 10000)`)
- Adicionar novos gráficos
- Customizar interpretações

## 📊 Logs

### Logs da API
```bash
docker-compose logs -f api
```

### Logs do Frontend
```bash
docker-compose logs -f frontend
```

### Logs do Redis
```bash
docker-compose logs -f redis
```

## 🛑 Parar os Serviços

```bash
# Para todos os containers
docker-compose down

# Para e remove volumes (limpa dados do Redis)
docker-compose down -v
```

## 🔍 Troubleshooting

### Frontend não carrega
- Verifique se a API está rodando: `curl http://localhost:8000/health`
- Verifique CORS no navegador (F12 → Console)

### Erro "Nenhuma predição disponível"
- Execute `python run_prediction.py` primeiro
- Verifique se o Redis está rodando: `docker-compose ps`

### Predição não salva no Redis
- Verifique conexão: `redis-cli ping` (deve retornar PONG)
- Veja logs do predictor
- Certifique-se de que `enable_redis=True` no predictor

## 📈 Próximos Passos

1. **Automação**: Configure cron/scheduler para executar predições periodicamente
2. **Histórico**: Armazene histórico de predições no Redis com timestamps
3. **Gráficos**: Adicione charts com histórico de sinais e performance
4. **Alertas**: Implemente notificações quando houver sinais fortes
5. **Multi-timeframe**: Suporte para múltiplos timeframes (5m, 15m, 30m)

## 💡 Exemplo de Uso em Produção

```python
# Scheduler para executar predições a cada 30 minutos
import schedule
import time

def job():
    os.system('python run_prediction.py')

schedule.every(30).minutes.do(job)

while True:
    schedule.run_pending()
    time.sleep(1)
```
