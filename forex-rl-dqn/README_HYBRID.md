# Sistema Híbrido de Trading: LightGBM + PPO

Sistema completo de trading para Forex baseado na combinação de aprendizado supervisionado (LightGBM) e aprendizado por reforço (PPO).

## 🎯 Visão Geral

Este sistema implementa uma arquitetura híbrida onde:
- **LightGBM** aprende padrões históricos de preço e prevê direção/retorno futuro
- **PPO** aprende quando e como operar, usando sinais do LightGBM + contexto de mercado + gestão de risco

## 🏗️ Arquitetura

```
┌─────────────────────────────────────────────────────────────┐
│                    Dados Históricos (CSV)                    │
│                    (OHLCV do cTrader)                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Feature Engineering                             │
│  (RSI, EMAs, MACD, Bollinger, ATR, Volatilidade, etc.)      │
└────────────────────────┬────────────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          │                             │
          ▼                             ▼
┌──────────────────┐          ┌──────────────────┐
│    LightGBM      │          │    Ambiente      │
│  (Supervisionado)│          │    Gym (PPO)     │
│                  │          │                  │
│ Prevê:           │          │ Estado:          │
│ • Direção        │────────> │ • Sinal LightGBM │
│ • Retorno        │          │ • Features       │
│   Futuro         │          │ • Posição atual  │
│                  │          │ • PnL            │
└──────────────────┘          │ • Equity         │
                              │ • Drawdown       │
                              │                  │
                              │ Ações:           │
                              │ 0 = Neutro       │
                              │ 1 = Comprar      │
                              │ 2 = Vender       │
                              │                  │
                              │ Reward:          │
                              │ PnL - custos     │
                              │ - penalização    │
                              └────────┬─────────┘
                                       │
                                       ▼
                              ┌──────────────────┐
                              │   Agente PPO     │
                              │ (Política Neural)│
                              │                  │
                              │ Aprende:         │
                              │ • Timing         │
                              │ • Gestão de risco│
                              │ • Maximizar PnL  │
                              └────────┬─────────┘
                                       │
                                       ▼
                              ┌──────────────────┐
                              │  API FastAPI     │
                              │                  │
                              │ /signal          │
                              │ /execute         │
                              │ /state           │
                              └──────────────────┘
```

## 📦 Instalação

### 1. Clone o repositório
```bash
git clone <repo>
cd forex-rl-dqn
```

### 2. Crie ambiente virtual
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

### 3. Instale dependências
```bash
pip install -r requirements.txt
```

## 📊 Preparação de Dados

### Formato do CSV (cTrader)
O sistema espera um CSV com as seguintes colunas:
```csv
timestamp,open,high,low,close,volume
2024-01-01 00:00:00,1.0950,1.0960,1.0945,1.0955,1000.0
2024-01-01 00:30:00,1.0955,1.0965,1.0950,1.0960,1200.0
...
```

Coloque seus arquivos CSV em `data/`:
- `data/usdjpy_history_30m.csv` (ou configure em `config_hybrid.yaml`)

## ⚙️ Configuração

Edite `config_hybrid.yaml` para ajustar:

### Dados
```yaml
data:
  train_file: "data/usdjpy_history_30m.csv"
  val_split: 0.15
  test_split: 0.15
```

### LightGBM
```yaml
lightgbm:
  model_type: "classifier"  # ou "regressor"
  prediction_horizon: 5  # candles à frente
  params:
    learning_rate: 0.05
    n_estimators: 500
    max_depth: 6
    # ... outros parâmetros
```

### PPO
```yaml
ppo:
  env:
    initial_balance: 10000.0
    leverage: 1.0
    commission: 0.0002  # 0.02%
    stop_loss_pct: 0.02  # 2%
    take_profit_pct: 0.04  # 4%
  params:
    learning_rate: 0.0003
    n_steps: 2048
    batch_size: 64
    # ... outros parâmetros
```

## 🚀 Uso

### 1. Treinar LightGBM
```bash
python -m src.training.train_lightgbm --config config_hybrid.yaml
```

Isso irá:
- Carregar dados históricos
- Criar features técnicas
- Treinar modelo LightGBM
- Salvar em `models/hybrid/lightgbm_model.txt`
- Exibir métricas e feature importance

### 2. Treinar PPO
```bash
python -m src.training.train_ppo --config config_hybrid.yaml
```

Isso irá:
- Carregar LightGBM treinado
- Criar ambiente de trading
- Treinar agente PPO
- Salvar em `models/hybrid/ppo_model.zip`
- Exibir métricas de performance

### 3. Subir API de Inferência
```bash
cd src/inference
python service.py
```

A API estará disponível em `http://localhost:8000`

#### Endpoints Disponíveis:

**GET /** - Informações da API
```bash
curl http://localhost:8000/
```

**GET /health** - Health check
```bash
curl http://localhost:8000/health
```

**POST /signal** - Obter sinal de trading
```bash
curl -X POST http://localhost:8000/signal \
  -H "Content-Type: application/json" \
  -d '{
    "candles": [
      {
        "timestamp": "2024-01-01T00:00:00",
        "open": 1.0950,
        "high": 1.0960,
        "low": 1.0945,
        "close": 1.0955,
        "volume": 1000.0
      },
      ... (mínimo 50 candles)
    ],
    "current_position": 0,
    "deterministic": true
  }'
```

Resposta:
```json
{
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
```

**POST /execute** - Executar ação
```bash
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{
    "action": 1,
    "price": 1.0955
  }'
```

**GET /state** - Obter estado atual
```bash
curl http://localhost:8000/state
```

**POST /reset** - Resetar estado
```bash
curl -X POST http://localhost:8000/reset
```

## 📝 Exemplo de Uso Completo

```python
import requests
import pandas as pd

# 1. Carrega dados recentes (últimos 100 candles)
df = pd.read_csv('data/usdjpy_history_30m.csv').tail(100)

# 2. Converte para formato JSON
candles = df.to_dict('records')

# 3. Solicita sinal
response = requests.post(
    'http://localhost:8000/signal',
    json={
        'candles': candles,
        'current_position': 0,
        'deterministic': True
    }
)

signal = response.json()
print(f"Ação: {signal['action_name']}")
print(f"Confiança: {signal['confidence']:.2f}")

# 4. Se confiança > 0.6, executa
if signal['confidence'] > 0.6:
    execute_response = requests.post(
        'http://localhost:8000/execute',
        json={
            'action': signal['action'],
            'price': candles[-1]['close']
        }
    )
    print(f"Executado: {execute_response.json()}")
```

## 🎛️ Hiperparâmetros Recomendados

### LightGBM (Classificação)
- `learning_rate`: 0.05
- `n_estimators`: 500
- `max_depth`: 6
- `num_leaves`: 31
- `prediction_horizon`: 5 candles

### PPO
- `learning_rate`: 0.0003
- `n_steps`: 2048
- `batch_size`: 64
- `gamma`: 0.99
- `total_timesteps`: 500000

### Ambiente
- `commission`: 0.0002 (0.02% - ajuste para seu broker)
- `slippage`: 0.0001 (0.01%)
- `stop_loss_pct`: 0.02 (2%)
- `take_profit_pct`: 0.04 (4%)

## 🔧 Ajustes para Seu Contexto

### Par de Moedas
- Ajuste `commission` e `slippage` baseado no spread do seu par
- Pares mais voláteis podem precisar de `stop_loss_pct` maior

### Timeframe
- 5M: Use `prediction_horizon: 3-5`
- 15M: Use `prediction_horizon: 5-7`
- 30M: Use `prediction_horizon: 5-10`
- 1H: Use `prediction_horizon: 7-15`

### Custos de Transação
Consulte seu broker e ajuste:
```yaml
ppo:
  env:
    commission: 0.0002  # Spread + comissão
    slippage: 0.0001    # Slippage médio observado
```

### Alavancagem
```yaml
ppo:
  env:
    leverage: 1.0  # 1:1 (conservador)
    # leverage: 10.0  # 1:10 (agressivo - CUIDADO!)
```

## 📈 Monitoramento e Avaliação

### Métricas do LightGBM
- **AUC** (classificação): > 0.60 é razoável, > 0.70 é bom
- **Accuracy**: > 55% já adiciona valor
- **Direction Accuracy**: mais importante que RMSE

### Métricas do PPO
- **Mean Reward**: deve crescer durante treinamento
- **Mean Equity**: deve ser > initial_balance
- **Win Rate**: > 45% é aceitável
- **Sharpe Ratio**: > 1.0 é bom, > 2.0 é excelente
- **Max Drawdown**: < 20% é desejável

### Logs
Treinamento gera logs em:
- `logs/hybrid/train/` - Logs de treino PPO
- `logs/hybrid/val/` - Logs de validação PPO

Visualize com TensorBoard:
```bash
tensorboard --logdir logs/hybrid
```

## 🐛 Troubleshooting

### "LightGBM model not found"
Treine o LightGBM primeiro:
```bash
python -m src.training.train_lightgbm
```

### "Insufficient candles for reliable prediction"
Envie pelo menos 50 candles no request `/signal`

### Performance ruim
1. Aumente dados de treino (mínimo 6 meses de histórico)
2. Ajuste `prediction_horizon` para seu timeframe
3. Revise custos de transação (commission/slippage)
4. Experimente diferentes features técnicas
5. Treine por mais timesteps (PPO)

## 📚 Estrutura do Projeto

```
forex-rl-dqn/
├── config_hybrid.yaml           # Configuração principal
├── data/                        # Dados históricos CSV
├── models/                      # Modelos treinados
│   └── hybrid/
│       ├── lightgbm_model.txt
│       ├── ppo_model.zip
│       └── checkpoints/
├── logs/                        # Logs de treinamento
├── src/
│   ├── common/                  # Utilities
│   │   ├── features.py         # Feature engineering
│   │   └── utils.py
│   ├── models/                  # Modelos
│   │   ├── lightgbm_model.py
│   │   └── ppo_agent.py
│   ├── envs/                    # Ambientes Gym
│   │   └── forex_trading_env.py
│   ├── training/                # Scripts de treino
│   │   ├── train_lightgbm.py
│   │   └── train_ppo.py
│   └── inference/               # Inferência e API
│       ├── predictor.py
│       └── service.py           # API FastAPI
└── requirements.txt
```

## 🔬 Próximos Passos

1. **Backtesting Completo**: Implemente backtest walk-forward
2. **Multi-Timeframe**: Combine sinais de múltiplos timeframes
3. **Ensemble**: Combine múltiplos modelos LightGBM
4. **Ação Contínua**: Experimente PPO com ação contínua (fração do capital)
5. **Meta-Learning**: Adaptação online aos novos dados

## 📄 Licença

[Sua licença aqui]

## 👥 Contribuição

[Instruções de contribuição]

---

**⚠️ AVISO IMPORTANTE**: Trading envolve risco significativo de perda. Este sistema é para fins educacionais e de pesquisa. Sempre teste extensivamente em ambiente de simulação antes de usar capital real. Nunca opere com dinheiro que não pode perder.
