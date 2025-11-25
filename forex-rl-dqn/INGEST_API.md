# Endpoint de Ingestão de Dados Históricos

## Visão Geral

O endpoint `/ingest` permite receber e persistir dados históricos do cTrader em arquivos CSV. Este endpoint foi desenvolvido para facilitar a coleta e armazenamento de dados de mercado em tempo real.

## Endpoint

**URL:** `POST /ingest?symbol={SYMBOL}`

**Content-Type:** `application/json`

**Query Parameters:**
- `symbol` (string, opcional): Símbolo do par de moedas. Default: "EURUSD"

## Request Body

Envia um **array JSON** diretamente com as barras OHLCV:

```json
[
  {
    "timestamp": "2024-01-01T00:00:00Z",
    "open": 1.1000,
    "high": 1.1010,
    "low": 1.0990,
    "close": 1.1005,
    "volume": 1234.56
  }
]
```

### Campos de Cada Barra

- **timestamp** (string, obrigatório): Timestamp no formato ISO 8601 (ex: "2024-01-01T00:00:00Z")
- **open** (float, obrigatório): Preço de abertura (> 0)
- **high** (float, obrigatório): Preço máximo (≥ low)
- **low** (float, obrigatório): Preço mínimo (> 0)
- **close** (float, obrigatório): Preço de fechamento (> 0)
- **volume** (float, obrigatório): Volume negociado (≥ 0)

## Response

### Sucesso (200 OK)

```json
{
  "status": "success",
  "records_saved": 100,
  "file_path": "data/eurusd_history.csv"
}
```

### Campos do Response

- **status** (string): Status da operação ("success")
- **records_saved** (int): Número de registros salvos
- **file_path** (string): Caminho do arquivo CSV onde os dados foram salvos

### Erros

**400 Bad Request:**
- Lista de dados vazia
- Tamanho de window incorreto
- Dados OHLCV inválidos (high < low)

**422 Unprocessable Entity:**
- Campos obrigatórios ausentes
- Tipos de dados incorretos
- Validação de schema falhou

**500 Internal Server Error:**
- Erro ao gravar arquivo CSV
- Erro interno do servidor

## Comportamento

### Criação de Arquivo

Quando o endpoint recebe dados para um símbolo pela primeira vez:

1. Cria o diretório `data/` se não existir
2. Cria um novo arquivo CSV com nome `{symbol}_history.csv` (ex: `eurusd_history.csv`)
3. Escreve o cabeçalho CSV: `timestamp,open,high,low,close,volume`
4. Adiciona os registros recebidos

### Append de Dados

Quando o endpoint recebe dados para um símbolo já existente:

1. Verifica se o arquivo `{symbol}_history.csv` existe
2. Abre o arquivo em modo append
3. Adiciona os novos registros **sem reescrever o cabeçalho**
4. Preserva todos os dados anteriores

### Múltiplos Símbolos

Cada símbolo possui seu próprio arquivo CSV:

- `data/eurusd_history.csv`
- `data/gbpusd_history.csv`
- `data/usdjpy_history.csv`
- etc.

## Docker e Volumes

### Configuração do Volume

No `docker-compose.yml`, o diretório `data/` está mapeado como volume:

```yaml
volumes:
  - ./data:/app/data
```

Isso significa que:

- O diretório `data/` do host é montado em `/app/data` no container
- Arquivos criados pelo container são acessíveis no host
- Dados persistem mesmo se o container for removido
- **Modo read-write:** O container pode criar e modificar arquivos

### Variável de Ambiente

O endpoint usa a variável de ambiente `DATA_DIR` para determinar onde salvar os arquivos:

```yaml
environment:
  - DATA_DIR=/app/data
```

Se não definida, o padrão é `data/`.

## Exemplos de Uso

### 1. Usando cURL

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

### 2. Usando Python (requests)

```python
import requests

# Preparar dados (array direto)
bars = [
    {
        "timestamp": "2024-01-01T00:00:00Z",
        "open": 1.1000,
        "high": 1.1010,
        "low": 1.0990,
        "close": 1.1005,
        "volume": 1234.56
    }
]

# Enviar para API
response = requests.post(
    "http://localhost:8000/ingest?symbol=EURUSD",
    json=bars
)
print(response.json())
```

### 3. Usando o Script de Exemplo

```bash
# Iniciar a API
docker-compose up -d

# Executar o script de exemplo
python example_ingest.py
```

### 4. Integrando com cTrader

```python
# Pseudocódigo de integração com cTrader
def collect_and_send_data(symbol, timeframe, bars_count):
    # Coletar dados do cTrader
    bars = ctrader_api.get_historical_bars(
        symbol=symbol,
        timeframe=timeframe,
        count=bars_count
    )
    
    # Formatar para o padrão esperado (apenas array)
    formatted_bars = [
        {
            "timestamp": bar.time.isoformat() + "Z",
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume
        }
        for bar in bars
    ]
    
    # Enviar para API (array diretamente, símbolo no query param)
    response = requests.post(
        f"http://localhost:8000/ingest?symbol={symbol}",
        json=formatted_bars
    )
    
    return response.json()
```

## Formato do Arquivo CSV

Os dados são salvos em formato CSV padrão:

```csv
timestamp,open,high,low,close,volume
2024-01-01T00:00:00Z,1.1000,1.1010,1.0990,1.1005,1234.56
2024-01-01T00:01:00Z,1.1005,1.1015,1.0995,1.1008,1567.89
2024-01-01T00:02:00Z,1.1008,1.1020,1.1000,1.1012,1890.12
```

Este formato é compatível com:

- Pandas: `pd.read_csv("data/eurusd_history.csv")`
- Excel e Google Sheets
- Sistema de treinamento do modelo (`src.rl.train`)

## Verificação de Dados

Para verificar os dados salvos:

```bash
# Listar arquivos CSV
ls -lh data/*.csv

# Ver primeiras linhas
head -n 10 data/eurusd_history.csv

# Contar registros (subtrair 1 para o header)
wc -l data/eurusd_history.csv
```

## Boas Práticas

1. **Validação de Dados:** Sempre valide que `high >= low` antes de enviar
2. **Ordem Cronológica:** Envie dados em ordem cronológica crescente
3. **Deduplicação:** Evite enviar registros duplicados (mesmo timestamp)
4. **Batch Size:** Envie dados em lotes razoáveis (100-1000 registros por request)
5. **Error Handling:** Implemente retry logic para falhas de rede
6. **Backup:** Faça backup regular do diretório `data/`

## Troubleshooting

### Erro: "Data list cannot be empty"

**Solução:** Certifique-se de que o array `data` contém pelo menos um registro.

### Erro: "high must be >= low"

**Solução:** Verifique os valores de high e low. O high deve sempre ser maior ou igual ao low.

### Arquivo não aparece no host

**Solução:** Verifique se o volume está mapeado corretamente no docker-compose.yml:

```bash
docker-compose down
docker-compose up -d
```

### Permissões de escrita

**Solução:** Certifique-se de que o diretório `data/` tem permissões adequadas:

```bash
chmod 755 data/
```

## Performance

- **Throughput:** ~1000 registros/segundo (depende do hardware)
- **Tamanho do arquivo:** Aproximadamente 100 bytes por registro
- **Memória:** Mínima (streaming write)
- **Concorrência:** Thread-safe para múltiplas requests simultâneas

## Próximos Passos

Após ingerir dados históricos:

1. **Treinar o modelo:**
   ```bash
   python -m src.rl.train --data data/eurusd_history.csv
   ```

2. **Fazer predições:**
   ```bash
   curl -X POST http://localhost:8000/act -H "Content-Type: application/json" -d @request.json
   ```

3. **Monitorar performance:**
   - Acompanhe o tamanho dos arquivos CSV
   - Verifique logs da API: `docker-compose logs -f`

## Referências

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Validation](https://docs.pydantic.dev/)
- [cTrader API Documentation](https://help.ctrader.com/open-api/)
