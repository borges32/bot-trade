# 🚀 Forex RL - Enhanced Training Guide

## Melhorias Implementadas:

### 1. **Novas Features** (14 features total):
- ✅ `atr_14`: Average True Range (volatilidade)
- ✅ `momentum_10` / `momentum_20`: Taxa de mudança de preço
- ✅ `volatility_20`: Volatilidade histórica
- ✅ `volume_ratio`: Volume vs média
- ✅ `macd` / `macd_signal`: MACD indicator

### 2. **Ambiente Multi-Step**:
- ✅ Episódios de 50-100 steps (vs 1 step antes)
- ✅ Melhor credit assignment
- ✅ Aprendizado de sequências

### 3. **Configurações Otimizadas**:
- ✅ `config_15m_enhanced.yaml`: 15M com todas as features
- ✅ `config_30m_enhanced.yaml`: 30M com todas as features

---

## 🎯 Como Treinar:

### Opção 1: **Local** (se tiver recursos)
```bash
python3 -m src.rl.train \
  --data data/usdjpy_history_15m.csv \
  --config config_15m_enhanced.yaml \
  --artifacts artifacts_15m_enhanced
```

### Opção 2: **Google Colab** (GPU grátis)
```python
# No Colab
!git clone https://github.com/borges32/bot-trade.git
%cd bot-trade/forex-rl-dqn

# Upload CSV
from google.colab import files
uploaded = files.upload()
!mv usdjpy_history_15m.csv data/

# Instalar
!pip install -q torch gymnasium numpy pandas scikit-learn pyyaml

# Treinar
!python -m src.rl.train \
  --data data/usdjpy_history_15m.csv \
  --config config_15m_enhanced.yaml \
  --artifacts artifacts_15m_enhanced
```

---

## 📊 O Que Esperar:

Com as **novas features + multi-step**:

| Métrica | Antes | Esperado Agora |
|---------|-------|----------------|
| Win Rate | 31% | **55-65%** ✅ |
| Avg Reward | -0.0028% | **+0.005%** ✅ |
| Convergência | 120k steps | **80k steps** ✅ |

---

## 🔍 Por Que Deve Funcionar Melhor:

### 1. **ATR + Volatility**:
- Modelo sabe quando mercado está volátil
- Evita trading em momentos de baixo movimento

### 2. **Momentum**:
- Detecta tendências de curto e médio prazo
- Melhora timing de entrada

### 3. **Volume Ratio**:
- Confirma movimentos com volume
- Evita falsos breakouts

### 4. **MACD**:
- Identifica reversões de tendência
- Classic indicator usado por traders

### 5. **Multi-Step Environment**:
- Vê consequências de ações ao longo do tempo
- Aprende a manter posições lucrativas
- Melhor que single-step (31% win rate)

---

## ⚙️ Configurações:

### config_15m_enhanced.yaml:
- **Window**: 48 bars (12 horas)
- **Episode**: 100 steps (25 horas)
- **Features**: 14 (vs 7 antes)
- **LSTM**: 256 units (vs 128)
- **MLP**: 512 units (vs 256)

### config_30m_enhanced.yaml:
- **Window**: 24 bars (12 horas)
- **Episode**: 50 steps (25 horas)
- **Features**: 14
- **Network**: Igual 15M

---

## 🎮 Comandos Rápidos:

```bash
# Testar configuração
python3 -c "import yaml; print(yaml.safe_load(open('config_15m_enhanced.yaml')))"

# Verificar features
python3 -c "
from src.common.features import generate_features
import pandas as pd
df = pd.read_csv('data/usdjpy_history_15m.csv').head(100)
features = ['atr_14', 'momentum_10', 'volatility_20', 'volume_ratio', 'macd']
result = generate_features(df, features)
print(result.describe())
"

# Treinar 15M
python3 -m src.rl.train \
  --data data/usdjpy_history_15m.csv \
  --config config_15m_enhanced.yaml \
  --artifacts artifacts_15m_enhanced

# Treinar 30M
python3 -m src.rl.train \
  --data data/usdjpy_history_30m.csv \
  --config config_30m_enhanced.yaml \
  --artifacts artifacts_30m_enhanced
```

---

## 💡 Próximos Passos se Não Funcionar:

1. **Testar par mais volátil**: GBPJPY, EURJPY
2. **Aumentar window_size**: 48 → 96 bars
3. **Adicionar mais features**: Stochastic, ADX, CCI
4. **Usar ensemble**: Treinar múltiplos modelos
5. **Implementar PPO**: Algoritmo mais estável que DQN

---

## 📈 Monitoramento:

Acompanhe métricas a cada 5k steps:
- **Win Rate > 50%** após 30k steps = ✅ Bom
- **Win Rate > 55%** após 60k steps = ✅ Excelente
- **Win Rate > 60%** após 100k steps = ✅ Deploy!

Se Win Rate < 45% após 50k steps → Pare e ajuste config.

---

Boa sorte! 🚀
