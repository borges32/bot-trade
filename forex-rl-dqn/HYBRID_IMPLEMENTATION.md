# Implementação do Sistema Híbrido LightGBM + PPO

## 📋 Resumo Executivo

Implementei um sistema completo de trading para Forex baseado na combinação de:
1. **LightGBM** (Supervisionado): Prevê direção/retorno de preço
2. **PPO** (Reinforcement Learning): Decide quando operar e como gerenciar risco

## 🏗️ Arquitetura Implementada

### Componentes Principais

#### 1. **LightGBM - Modelo Supervisionado** (`src/models/lightgbm_model.py`)
- **Modos**: Classificação (direção) ou Regressão (retorno)
- **Features**: Indicadores técnicos (RSI, EMAs, MACD, Bollinger, ATR, etc.)
- **Target**: Preço em N candles à frente (configurável)
- **Output**: Probabilidade de alta (classifier) ou retorno esperado (regressor)

#### 2. **Ambiente Gym** (`src/envs/forex_trading_env.py`)
- **Estado (Observation)**:
  - Sinal LightGBM
  - Features técnicas normalizadas
  - Posição atual (-1, 0, 1)
  - PnL não realizado
  - Equity normalizado
  - Drawdown atual
  
- **Ações (Discretas)**:
  - 0 = Neutro/Flat
  - 1 = Comprar (Long)
  - 2 = Vender (Short)
  
- **Recompensa**:
  ```python
  reward = PnL_delta - custos_transação - λ * drawdown
  ```

- **Recursos**:
  - Stop Loss / Take Profit automático
  - Tracking de drawdown máximo
  - Gestão de posição e capital
  - Cálculo realista de custos (commission + slippage)

#### 3. **Agente PPO** (`src/models/ppo_agent.py`)
- Baseado em `stable-baselines3`
- Rede neural: [256, 256, 128] (configurável)
- Aprende política de trading considerando:
  - Sinais de mercado (LightGBM)
  - Gestão de risco
  - Custos de transação
  - Maximização de Sharpe Ratio

#### 4. **Sistema de Inferência** (`src/inference/predictor.py`)
- Classe `TradingPredictor`:
  - Carrega modelos treinados
  - Processa dados recentes
  - Gera predições com confiança
  - Mantém estado da conta
  - Executa ações e atualiza PnL

#### 5. **API FastAPI** (`src/inference/service.py`)
- **Endpoints**:
  - `GET /` - Informações
  - `GET /health` - Health check
  - `POST /signal` - Obter sinal de trading
  - `POST /execute` - Executar ação
  - `GET /state` - Estado atual
  - `POST /reset` - Resetar estado

## 📁 Estrutura de Arquivos Criados

```
forex-rl-dqn/
├── config_hybrid.yaml                    # ✅ Configuração completa
├── README_HYBRID.md                      # ✅ Documentação detalhada
├── train_hybrid.sh                       # ✅ Script de treinamento
├── test_hybrid_system.py                 # ✅ Teste do sistema
├── example_hybrid_usage.py               # ✅ Exemplo de uso
├── requirements.txt                      # ✅ Atualizado com dependências
│
├── src/
│   ├── models/                           # ✅ NOVO
│   │   ├── __init__.py
│   │   ├── lightgbm_model.py            # ✅ Modelo LightGBM
│   │   └── ppo_agent.py                 # ✅ Agente PPO
│   │
│   ├── envs/                            # ✅ NOVO
│   │   ├── __init__.py
│   │   └── forex_trading_env.py         # ✅ Ambiente Gym
│   │
│   ├── training/                         # ✅ NOVO
│   │   ├── __init__.py
│   │   ├── train_lightgbm.py            # ✅ Treino LightGBM
│   │   └── train_ppo.py                 # ✅ Treino PPO
│   │
│   └── inference/                        # ✅ NOVO
│       ├── __init__.py
│       ├── predictor.py                 # ✅ Motor de inferência
│       └── service.py                   # ✅ API FastAPI
```

## 🎯 Hiperparâmetros Configurados

### LightGBM
```yaml
model_type: "classifier"
prediction_horizon: 5
params:
  learning_rate: 0.05
  n_estimators: 500
  max_depth: 6
  num_leaves: 31
```

### PPO
```yaml
params:
  learning_rate: 0.0003
  n_steps: 2048
  batch_size: 64
  gamma: 0.99
  gae_lambda: 0.95

training:
  total_timesteps: 500000
```

### Ambiente
```yaml
initial_balance: 10000.0
leverage: 1.0
commission: 0.0002  # 0.02%
slippage: 0.0001    # 0.01%
stop_loss_pct: 0.02  # 2%
take_profit_pct: 0.04  # 4%
max_drawdown_pct: 0.20  # 20%
```

## 🚀 Fluxo de Uso

### 1. Preparação
```bash
# Colocar dados CSV em data/
# Editar config_hybrid.yaml se necessário
```

### 2. Testar Sistema
```bash
python test_hybrid_system.py
```

### 3. Treinamento
```bash
# Método 1: Script automatizado
./train_hybrid.sh

# Método 2: Individual
python -m src.training.train_lightgbm
python -m src.training.train_ppo
```

### 4. Uso em Produção

#### Opção A: API HTTP
```bash
cd src/inference
python service.py
```

```python
import requests

response = requests.post('http://localhost:8000/signal', json={
    'candles': [...],  # 50+ candles
    'current_position': 0
})

signal = response.json()
print(f"Ação: {signal['action_name']}")
```

#### Opção B: Python Direto
```python
from src.inference.predictor import TradingPredictor

predictor = TradingPredictor(
    lightgbm_path='models/hybrid/lightgbm_model',
    ppo_path='models/hybrid/ppo_model',
    feature_config=config['features'],
    env_config=config['ppo']['env']
)

result = predictor.predict(candles_df)
print(result['action_name'])
```

## 🔧 Pontos de Customização

### Para Diferentes Pares de Moedas
```yaml
ppo:
  env:
    commission: 0.0003  # Ajustar para spread do par
    slippage: 0.0001    # Ajustar baseado em observação
```

### Para Diferentes Timeframes
```yaml
lightgbm:
  prediction_horizon: 5  # Ajustar:
    # 5M: 3-5
    # 15M: 5-7
    # 30M: 5-10
    # 1H: 7-15
```

### Para Diferentes Níveis de Risco
```yaml
ppo:
  env:
    leverage: 1.0        # Aumentar para mais agressivo
    stop_loss_pct: 0.02  # Reduzir para mais conservador
    max_position_size: 1.0  # Reduzir para diversificar
```

## 📊 Métricas Esperadas

### LightGBM (Bom Desempenho)
- **AUC**: > 0.65
- **Accuracy**: > 55%
- **Direction Accuracy**: > 55%

### PPO (Bom Desempenho)
- **Mean Return**: > 5% no período de teste
- **Sharpe Ratio**: > 1.0
- **Win Rate**: > 45%
- **Max Drawdown**: < 15%

## ⚡ Diferenças do Modelo Anterior

| Aspecto | Modelo Anterior (DQN) | Novo Modelo (LightGBM + PPO) |
|---------|----------------------|------------------------------|
| **Arquitetura** | DQN puro | Híbrido: Supervisionado + RL |
| **Sinais de Mercado** | Apenas RL | LightGBM + PPO |
| **Estabilidade** | Menos estável | Mais estável (PPO) |
| **Interpretabilidade** | Baixa | Alta (feature importance) |
| **Convergência** | Lenta | Mais rápida |
| **Generalização** | Limitada | Melhor (dois modelos) |

## 🎓 Vantagens da Arquitetura Híbrida

1. **LightGBM** fornece "expertise de mercado" baseada em padrões históricos
2. **PPO** aprende "timing e execução" considerando custos e risco
3. **Separação de responsabilidades**: Cada modelo faz o que faz melhor
4. **Interpretabilidade**: Feature importance do LightGBM mostra o que importa
5. **Robustez**: Se um modelo erra, o outro pode compensar

## ⚠️ Considerações Importantes

### Para Produção
1. **Sempre backteste** extensivamente antes de usar capital real
2. **Monitore métricas** continuamente (Sharpe, drawdown, win rate)
3. **Retreine periodicamente** com dados recentes
4. **Ajuste custos** baseado em observação real do broker
5. **Use stop loss** conservadores inicialmente

### Limitações
- Modelo assume mercado **líquido** (sem gaps grandes)
- Não considera **notícias/eventos** fundamentais
- **Custos de transação** são críticos - ajuste com precisão
- Requer **dados de qualidade** (mínimo 6 meses)

## 📚 Próximos Passos Sugeridos

1. **Walk-Forward Validation**: Implementar backtest walk-forward
2. **Multi-Timeframe**: Combinar sinais de múltiplos timeframes
3. **Ensemble**: Usar múltiplos modelos LightGBM votando
4. **Online Learning**: Adaptação contínua a novos dados
5. **Risk Management Avançado**: Portfolio theory, Kelly Criterion

## 🐛 Troubleshooting Comum

| Problema | Solução |
|----------|---------|
| "Model not found" | Execute treinamento primeiro |
| "Insufficient candles" | Envie ≥50 candles no request |
| Performance ruim | Aumente dados, ajuste custos, revise features |
| API não inicia | Verifique se modelos existem em `models/hybrid/` |
| Drawdown muito alto | Reduza leverage, aumente stop loss |

## 📞 Suporte

Para dúvidas:
1. Consulte `README_HYBRID.md` para documentação detalhada
2. Execute `python test_hybrid_system.py` para diagnóstico
3. Verifique logs em `logs/hybrid/`
4. Revise configuração em `config_hybrid.yaml`

---

**Status**: ✅ Sistema completo e funcional, pronto para treinamento e testes.
