# Quick Start Guide - Data Ingestion

Este guia mostra como usar o novo endpoint `/ingest` para coletar e armazenar dados históricos do cTrader.

## 🚀 Início Rápido (5 minutos)

### 1. Iniciar a API

```bash
# Opção A: Com Docker (recomendado)
docker-compose up -d

# Opção B: Localmente
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Aguarde a API iniciar e verifique:
```bash
curl http://localhost:8000/health
```

### 2. Testar com Dados de Exemplo

```bash
# Executar script de exemplo
python example_ingest.py
```

Ou manualmente:
```bash
curl -X POST "http://localhost:8000/ingest?symbol=EURUSD" \
  -H "Content-Type: application/json" \
  -d '[
    {
      "timestamp": "2024-01-01T00:00:00Z",
      "open": 1.1000,
      "high": 1.1010,
      "low": 1.0990,
      "close": 1.1005,
      "volume": 1234.56
    }
  ]'
```

### 3. Verificar Dados Salvos

```bash
# Listar arquivos CSV criados
ls -lh data/

# Ver conteúdo
cat data/eurusd_history.csv
```

### 4. Treinar Modelo com os Dados

```bash
python -m src.rl.train --data data/eurusd_history.csv
```

## 📡 Integração com cTrader

### Setup Inicial

1. **Instalar dependências:**
```bash
pip install requests python-dotenv
```

2. **Configurar credenciais:**
```bash
# Copiar arquivo de exemplo
cp .env.example .env

# Editar .env com suas credenciais do cTrader
nano .env
```

3. **Executar integração:**
```bash
python ctrader_integration_example.py
```

## 🔄 Fluxo Completo

```
┌─────────────────┐
│  cTrader API    │  Coleta dados históricos
└────────┬────────┘
         │
         v
┌─────────────────┐
│ POST /ingest    │  Valida e persiste
└────────┬────────┘
         │
         v
┌─────────────────┐
│ data/*.csv      │  Armazena em CSV
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Train Model     │  Treina RL agent
└────────┬────────┘
         │
         v
┌─────────────────┐
│ POST /act       │  Predições em tempo real
└─────────────────┘
```

## 🎯 Casos de Uso

### Caso 1: Coleta Inicial de Dados

```python
import requests

# Preparar barras (apenas o array)
bars = [...]  # 30 dias * 24h * 60min = ~43200 barras M1

# Coletar 1 mês de dados históricos
response = requests.post(
    "http://localhost:8000/ingest?symbol=EURUSD",
    json=bars
)

print(f"Salvos: {response.json()['records_saved']} registros")
```

### Caso 2: Atualização Contínua

```python
import time
from datetime import datetime, timedelta

while True:
    # Coletar últimas N barras
    bars = get_latest_bars(symbol="EURUSD", count=10)
    
    # Enviar para API (apenas o array)
    requests.post(
        "http://localhost:8000/ingest?symbol=EURUSD",
        json=bars
    )
    
    # Aguardar próximo ciclo
    time.sleep(60)  # Atualizar a cada minuto
```

### Caso 3: Múltiplos Símbolos

```python
symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]

for symbol in symbols:
    bars = get_historical_bars(symbol, count=1000)
    
    response = requests.post(
        f"http://localhost:8000/ingest?symbol={symbol}",
        json=bars
    )
    
    print(f"{symbol}: {response.json()['records_saved']} registros")
```

## 📊 Monitoramento

### Ver Estatísticas dos Dados

```bash
# Contar registros por símbolo
for file in data/*_history.csv; do
    count=$(wc -l < "$file")
    echo "$(basename $file): $((count - 1)) registros"
done
```

### Verificar Qualidade dos Dados

```python
import pandas as pd

# Carregar dados
df = pd.read_csv("data/eurusd_history.csv")

print(f"Total de registros: {len(df)}")
print(f"Período: {df['timestamp'].min()} até {df['timestamp'].max()}")
print(f"\nEstatísticas:")
print(df[['open', 'high', 'low', 'close', 'volume']].describe())

# Verificar dados faltantes
print(f"\nDados faltantes: {df.isnull().sum().sum()}")

# Verificar consistência OHLC
inconsistent = df[df['high'] < df['low']]
print(f"Registros inconsistentes (high < low): {len(inconsistent)}")
```

### Logs da API

```bash
# Docker
docker-compose logs -f api

# Local
# Os logs aparecem no terminal onde você executou uvicorn
```

## ⚠️ Pontos de Atenção

1. **Volume Docker**: O diretório `data/` é mapeado entre host e container
   - Arquivos criados no container aparecem no host
   - Dados persistem mesmo após remover o container

2. **Append de Dados**: O endpoint **adiciona** dados ao arquivo existente
   - Não sobrescreve dados anteriores
   - Cuidado com duplicatas (mesmo timestamp)

3. **Validação**: Todos os dados são validados antes de salvar
   - `high >= low`
   - Valores positivos
   - Timestamps no formato ISO 8601

4. **Performance**: Para grandes volumes, envie em lotes
   - Recomendado: 100-1000 registros por request
   - Evite requests muito grandes (> 10000 registros)

## 🛠️ Troubleshooting Rápido

### "Cannot connect to API"
```bash
# Verificar se API está rodando
curl http://localhost:8000/health

# Reiniciar container
docker-compose restart api
```

### "Data list cannot be empty"
```bash
# Certifique-se de que o array 'data' contém pelo menos 1 registro
# Verifique o JSON enviado
```

### "high must be >= low"
```bash
# Verifique os valores OHLC
# high deve sempre ser maior ou igual a low
```

### Arquivo não aparece no host
```bash
# Verificar volume do Docker
docker-compose down
docker-compose up -d

# Verificar permissões
ls -la data/
```

## 📖 Próximos Passos

1. **Leia a documentação completa**: [INGEST_API.md](INGEST_API.md)
2. **Veja exemplos de código**: [example_ingest.py](example_ingest.py)
3. **Integre com cTrader**: [ctrader_integration_example.py](ctrader_integration_example.py)
4. **Treine o modelo**: `python -m src.rl.train --data data/eurusd_history.csv`

## 💡 Dicas

- **Backup**: Faça backup regular do diretório `data/`
- **Git Ignore**: Adicione `data/*.csv` ao `.gitignore` (já configurado)
- **Monitoramento**: Configure alertas para falhas na coleta de dados
- **Validação**: Sempre valide dados antes de treinar o modelo
- **Testes**: Use `example_ingest.py` para testar antes de integrar com cTrader

## 🆘 Suporte

- **Documentação**: [INGEST_API.md](INGEST_API.md)
- **README**: [README.md](README.md)
- **Issues**: Abra uma issue no GitHub
