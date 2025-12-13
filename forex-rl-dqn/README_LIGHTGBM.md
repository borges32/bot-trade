# Sistema de Trading Forex com LightGBM

Sistema de predição de sinais de trading para Forex usando **LightGBM** (Gradient Boosting) para prever retornos futuros.

## 🎯 Características

- **Modelo LightGBM Regressor**: Prevê retorno percentual futuro
- **Features técnicas otimizadas**: RSI, EMAs, MACD, Bollinger Bands, ATR, ADX, etc.
- **Configurações por timeframe**: 15m e 30m otimizados separadamente
- **API REST**: FastAPI para integração fácil
- **Alto desempenho**: 53.72% de acurácia direcional com RMSE de 0.0022

## 📊 Resultados do Modelo (30m)

```
Test Metrics:
  RMSE:                0.0022
  MAE:                 0.0015
  Direction Accuracy:  53.72%

Top Features:
  1. atr_ma
  2. adx
  3. sma_50
  4. bb_upper
  5. volatility
```

## 🚀 Quick Start

### 1. Instalação

```bash
# Clone o repositório
git clone <repo-url>
cd forex-rl-dqn

# Instale dependências
pip install -r requirements.txt
```

### 2. Preparar Dados

Seus dados CSV devem ter as colunas OHLCV:
- `timestamp` (ou configurar nome em config)
- `open`, `high`, `low`, `close`, `volume`

```bash
# Coloque seu arquivo CSV em data/
cp seu_arquivo.csv data/usdjpy_history_30m.csv
```

### 3. Treinar Modelo

```bash
# Para timeframe de 30 minutos
python -m src.training.train_lightgbm --config config_hybrid_30m.yaml

# Para timeframe de 15 minutos
python -m src.training.train_lightgbm --config config_hybrid_15m.yaml
```

### 4. Usar Predições

```python
from src.inference.predictor import TradingPredictor
import yaml
import pandas as pd

# Carrega config
with open('config_hybrid_30m.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Inicializa preditor
predictor = TradingPredictor(
    lightgbm_path='models/hybrid_30m/lightgbm_model',
    config=config
)

# Carrega dados recentes (mínimo 50 candles)
df = pd.read_csv('data/recent_candles.csv')

# Faz predição
result = predictor.predict(df)

print(f"Sinal: {result['signal']}")  # BUY, SELL, ou NEUTRAL
print(f"Retorno esperado: {result['predicted_return']:.4%}")
print(f"Confiança: {result['confidence']:.2%}")
```

### 5. API REST (Opcional)

```bash
# Inicia servidor
python -m src.api.main --config config_hybrid_30m.yaml

# Em outro terminal, teste
curl -X POST http://localhost:8000/signal \
  -H "Content-Type: application/json" \
  -d @example_request.json
```

## 📁 Estrutura

```
forex-rl-dqn/
├── config_hybrid_15m.yaml      # Config para 15 minutos
├── config_hybrid_30m.yaml      # Config para 30 minutos
├── data/
│   ├── usdjpy_history_15m.csv
│   └── usdjpy_history_30m.csv
├── models/
│   ├── hybrid_15m/
│   │   └── lightgbm_model.txt
│   └── hybrid_30m/
│       └── lightgbm_model.txt
├── src/
│   ├── common/
│   │   ├── features_optimized.py   # Feature engineering
│   │   └── utils.py
│   ├── models/
│   │   └── lightgbm_model.py       # LightGBM wrapper
│   ├── training/
│   │   └── train_lightgbm.py       # Script de treino
│   ├── inference/
│   │   ├── predictor.py            # Preditor
│   │   └── service.py              # API REST
│   └── api/
│       └── main.py
└── example_lightgbm_usage.py       # Exemplo de uso
```

## ⚙️ Configuração

### Timeframes Suportados

#### 30 Minutos (`config_hybrid_30m.yaml`)
- **Horizonte**: 10 candles (5 horas)
- **Períodos**: RSI=14, EMA=[12,26,50], MACD=(12,26,9)
- **Retornos**: [1, 3, 6, 12, 24] candles
- **Uso**: Tendências de médio prazo, menos ruído

#### 15 Minutos (`config_hybrid_15m.yaml`)
- **Horizonte**: 8 candles (2 horas)
- **Períodos**: RSI=10, EMA=[8,21,50], MACD=(8,17,6)
- **Retornos**: [1, 2, 4, 8, 16] candles
- **Uso**: Operações intraday, mais sinais

### Parâmetros Principais

```yaml
lightgbm:
  model_type: "regressor"
  prediction_horizon: 10      # Candles à frente
  
  params:
    objective: "regression"
    metric: "rmse"
    learning_rate: 0.05
    n_estimators: 500
    max_depth: 6
    reg_alpha: 0.3
    reg_lambda: 0.3

inference:
  min_confidence: 0.60        # Threshold para gerar sinais
```

## 🔧 Scripts Úteis

### Retreinar Modelo
```bash
./retrain_lightgbm_30m.sh
```

### Exemplo de Uso
```bash
python example_lightgbm_usage.py
```

### Testar Features
```bash
python test_features.py
```

## 📈 Melhorando o Modelo

### 1. Mais Dados
- Adicione mais histórico (idealmente 1+ ano)
- Use múltiplos pares de moedas
- Inclua diferentes condições de mercado

### 2. Feature Engineering
- Teste novos indicadores técnicos
- Adicione features de microestrutura
- Use combinações de indicadores

### 3. Hyperparameter Tuning
```python
# Use Optuna ou similar
from lightgbm import LGBMRegressor
import optuna

def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        # ...
    }
    # Treinar e avaliar
    return rmse
```

### 4. Ensemble
- Combine modelos de diferentes timeframes
- Use votação ou stacking
- Pondere por confiança

## 🔌 Integração

### cTrader
```csharp
// C# API call
var client = new HttpClient();
var json = JsonConvert.SerializeObject(new {
    candles = recentCandles
});
var response = await client.PostAsync(
    "http://localhost:8000/signal",
    new StringContent(json, Encoding.UTF8, "application/json")
);
```

### MetaTrader 5
```python
# Python com MetaTrader5
import MetaTrader5 as mt5
import requests

# Pega candles
candles = mt5.copy_rates_from_pos("USDJPY", mt5.TIMEFRAME_M30, 0, 100)

# Chama API
response = requests.post('http://localhost:8000/signal', json={
    'candles': candles.tolist()
})
signal = response.json()['signal']
```

## 📊 Monitoramento

### Métricas Importantes

1. **Direction Accuracy**: % de vezes que prevê direção correta
2. **RMSE**: Erro médio quadrático (quanto menor, melhor)
3. **Sharpe Ratio**: Retorno ajustado ao risco em trading real
4. **Max Drawdown**: Maior perda consecutiva

### Logs
```bash
# Logs de treinamento
tail -f logs/hybrid_30m/training.log

# Logs da API
tail -f logs/hybrid_30m/api.log
```

## ❓ FAQ

**P: Qual timeframe usar?**
R: 30m para médio prazo (menos sinais, mais qualidade). 15m para intraday (mais sinais, mais ruído).

**P: Como interpretar confiança?**
R: Baseada na magnitude do retorno previsto. >60% = sinal forte, <60% = neutro.

**P: Preciso de GPU?**
R: Não. LightGBM roda bem em CPU.

**P: Quanto de histórico preciso?**
R: Mínimo 10k candles. Ideal: 20k+ para treino robusto.

## 📝 Licença

MIT License - veja LICENSE para detalhes.

## 🤝 Contribuindo

Pull requests são bem-vindos! Para mudanças grandes, abra uma issue primeiro.

## ⚠️ Disclaimer

Este software é fornecido "como está" para fins educacionais. Trading envolve risco. Sempre teste em conta demo primeiro. Use por sua conta e risco.
