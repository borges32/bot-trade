# 🤖 Forex Trading Bot - LightGBM + PPO (Otimizado para cTrader)

Sistema híbrido de trading Forex combinando **LightGBM** (supervisionado) + **PPO** (reinforcement learning), otimizado para usar indicadores pré-calculados do cTrader/MT5.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.1.0-green)](https://lightgbm.readthedocs.io/)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-2.2.1-orange)](https://stable-baselines3.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-teal)](https://fastapi.tiangolo.com/)
[![Performance](https://img.shields.io/badge/Performance-10x_Faster-brightgreen)](OPTIMIZATION_SUMMARY.md)

---

## ⚡ NOVO: Sistema Otimizado

**10x mais rápido** usando indicadores do cTrader/MT5:

✅ **Performance:** 24,000+ candles/segundo  
✅ **Features:** 75 features (OHLCV + indicadores + derivadas)  
✅ **Compatibilidade:** 100% - funciona com qualquer CSV  
✅ **Testado:** 4/4 testes automáticos passando  

**[📚 COMECE AQUI: Guia Rápido →](QUICK_GUIDE_OPTIMIZATION.md)**

---

## 🚀 Início Ultra-Rápido (30 segundos)

```bash
# 1. Clone
git clone https://github.com/borges32/bot-trade.git
cd bot-trade/forex-rl-dqn

# 2. Instale
pip install -r requirements.txt

# 3. Teste (3 segundos)
python3 test_optimized_features.py
# ✓ SUCESSO! Sistema funcionando perfeitamente!
```

**Pronto!** Sistema validado e funcionando. 🎉

**[📖 Próximos Passos →](#-uso-completo)**

---

## 📊 Seu CSV do cTrader

### Colunas Esperadas (19 total)

**OHLCV Básico (6):**
```
timestamp, open, high, low, close, volume
```

**Indicadores Pré-Calculados (13):**
```
rsi, ema_fast, ema_slow,
bb_upper, bb_middle, bb_lower,
atr, momentum_10, momentum_20,
volatility, volume_ma,
macd, macd_signal
```

**Não tem todos?** Sistema detecta automaticamente e calcula os que faltam! ✅

---

## 🎯 Como Funciona

### Arquitetura Híbrida

```
┌─────────────┐
│  Dados CSV  │ (19 colunas)
└──────┬──────┘
       │
       ▼
┌─────────────────────────┐
│ OptimizedFeatureEngineer│ (75 features)
└──────┬──────────────────┘
       │
       ├──────────────────┐
       ▼                  ▼
┌─────────────┐    ┌─────────────┐
│  LightGBM   │    │     PPO     │
│ (Supervised)│    │     (RL)    │
└──────┬──────┘    └──────┬──────┘
       │                  │
       │  Signal (0-1)    │
       └────────┬─────────┘
                ▼
         ┌─────────────┐
         │   Action    │
         │ 0: Neutral  │
         │ 1: Buy      │
         │ 2: Sell     │
         └─────────────┘
```

### Fluxo de Decisão

1. **LightGBM** prevê probabilidade de alta (0-1)
2. **PPO** recebe: `[lightgbm_signal, features, account_state]`
3. **PPO** decide ação otimizando lucro total

**Vantagens:**
- ✅ LightGBM aprende padrões de preço
- ✅ PPO aprende gestão de risco
- ✅ Melhor que RL puro (mais estável)
- ✅ Melhor que supervised puro (otimiza lucro, não acurácia)

---

## 💻 Instalação Completa

### Requisitos
- Python 3.8+
- pip

### Instalar

```bash
# Clone
git clone https://github.com/borges32/bot-trade.git
cd bot-trade/forex-rl-dqn

# Instale dependências
pip install -r requirements.txt

# Teste instalação
python3 test_optimized_features.py
```

**Deve mostrar:**
```
✓ Processamento: 0.041 segundos
✓ Velocidade: 24,643 candles/segundo
✓ 13 indicadores pré-calculados detectados
✓ 56 features novas criadas
✓ 4/4 testes passaram
✓ SUCESSO!
```

---

## 🎓 Uso Completo

### Passo 1: Preparar Dados

```bash
# Coloque seu CSV do cTrader em:
cp /caminho/seu_arquivo.csv data/usdjpy_history_30m.csv

# Teste com seus dados
python3 example_precomputed_features.py
```

**Output esperado:**
```
✓ Carregados 10000 candles
✓ Indicadores pré-calculados detectados: 13
✓ Features criadas: 75
✓ Dados salvos em: data/processed_features.csv
```

### Passo 2: Treinar Modelos

**Opção A: Automático (Recomendado)**
```bash
./train_hybrid.sh
```

Executa:
1. Treina LightGBM (~5 min)
2. Treina PPO (~20 min)
3. Avalia ambos
4. Salva métricas

**Opção B: Manual**
```bash
# LightGBM
python3 src/training/train_lightgbm.py

# PPO
python3 src/training/train_ppo.py
```

### Passo 3: Avaliar Resultados

```bash
# Veja métricas
cat models/hybrid/lightgbm_metrics.yaml
cat models/hybrid/ppo_metrics.yaml

# Veja logs
tail -f logs/hybrid/training.log
```

### Passo 4: Usar API

```bash
# Inicie servidor
cd src/inference
python3 service.py

# Em outro terminal, teste
curl -X POST http://localhost:8000/signal \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2024-01-01T00:00:00",
    "open": 150.0,
    "high": 150.5,
    "low": 149.5,
    "close": 150.2,
    "volume": 5000,
    "rsi": 55.2,
    "ema_fast": 150.15,
    "ema_slow": 149.80,
    ...
  }'
```

**Response:**
```json
{
  "action": "buy",
  "confidence": 0.85,
  "lightgbm_signal": 0.72,
  "ppo_action": 1,
  "timestamp": "2024-01-01T00:00:00"
}
```

---

## 📁 Estrutura do Projeto

```
forex-rl-dqn/
├── 📚 Documentação
│   ├── QUICK_GUIDE_OPTIMIZATION.md  ← COMECE AQUI
│   ├── INDEX_DOCUMENTATION.md        ← Índice completo
│   ├── OPTIMIZED_FEATURES.md         ← Guia técnico
│   └── CONFIG_FEATURES_MAPPING.md    ← Config → Features
│
├── 💻 Código Principal
│   ├── src/
│   │   ├── common/
│   │   │   └── features_optimized.py ← Features otimizadas
│   │   ├── models/
│   │   │   ├── lightgbm_model.py
│   │   │   └── ppo_agent.py
│   │   ├── training/
│   │   │   ├── train_lightgbm.py
│   │   │   └── train_ppo.py
│   │   └── inference/
│   │       ├── predictor.py
│   │       └── service.py           ← FastAPI
│
├── 🧪 Exemplos e Testes
│   ├── test_optimized_features.py   ← Teste automático
│   ├── example_precomputed_features.py
│   └── example_hybrid_usage.py
│
├── ⚙️ Configuração
│   └── config_hybrid.yaml           ← Config central
│
└── 📊 Dados e Modelos
    ├── data/                        ← Seus CSVs aqui
    ├── models/hybrid/               ← Modelos treinados
    └── logs/hybrid/                 ← Logs de treinamento
```

---

## ⚙️ Configuração

Edite `config_hybrid.yaml`:

```yaml
# Seus dados
data:
  train_file: "data/usdjpy_history_30m.csv"
  
  # Indicadores esperados
  precomputed_indicators:
    rsi: "rsi"              # Ajuste se nome diferente
    ema_fast: "ema_fast"    # no seu CSV
    # ...

# LightGBM
lightgbm:
  model_type: "classifier"  # ou "regressor"
  prediction_horizon: 5     # Candles à frente

# PPO
ppo:
  env:
    commission: 0.0002      # Ajuste para seu broker
    leverage: 1.0
  training:
    total_timesteps: 500000
```

**[📋 Ver Mapeamento Completo →](CONFIG_FEATURES_MAPPING.md)**

---

## 📊 Performance

### Benchmarks

**Com 10,000 candles:**
```
Antiga (recalcula tudo):  ~5-10 segundos
Nova (usa pré-calculados): ~0.5-1 segundo
Ganho: 10x mais rápido! 🚀
```

**Features criadas:**
```
Antiga: ~50 features
Nova: ~75 features (+50%)
```

**Qualidade:**
```
✓ 0 valores NaN
✓ 0 valores infinitos
✓ Todas features validadas
✓ 4/4 testes passando
```

---

## 📚 Documentação

| Documento | Descrição | Tempo |
|-----------|-----------|-------|
| **[QUICK_GUIDE_OPTIMIZATION.md](QUICK_GUIDE_OPTIMIZATION.md)** | ⭐ Início rápido | 3 min |
| **[INDEX_DOCUMENTATION.md](INDEX_DOCUMENTATION.md)** | Índice completo | - |
| **[OPTIMIZATION_SUMMARY.md](OPTIMIZATION_SUMMARY.md)** | Resumo executivo | 5 min |
| **[OPTIMIZED_FEATURES.md](OPTIMIZED_FEATURES.md)** | Guia técnico completo | 15 min |
| **[CONFIG_FEATURES_MAPPING.md](CONFIG_FEATURES_MAPPING.md)** | Config → Features | 10 min |
| **[README_HYBRID.md](README_HYBRID.md)** | README técnico | 20 min |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | Arquitetura | 10 min |

---

## 🧪 Testes

```bash
# Teste sistema
python3 test_optimized_features.py

# Teste com dados reais
python3 example_precomputed_features.py

# Teste API
cd src/inference
python3 service.py &
curl http://localhost:8000/health
```

---

## 🐛 Solução de Problemas

### CSV não encontrado
```bash
ls -lh data/usdjpy_history_30m.csv
# Coloque seu arquivo aqui
```

### Indicadores não detectados
```python
import pandas as pd
df = pd.read_csv('data/usdjpy_history_30m.csv')
print(df.columns.tolist())
# Ajuste config_hybrid.yaml com nomes corretos
```

### Erro de NaN
```python
# Remova linhas incompletas
df = df.dropna(subset=['rsi', 'ema_fast', 'close'])
```

**[📖 Ver Mais Problemas →](QUICK_GUIDE_OPTIMIZATION.md#problemas-comuns)**

---

## 🤝 Contribuindo

Pull requests são bem-vindos!

1. Fork o projeto
2. Crie branch (`git checkout -b feature/AmazingFeature`)
3. Commit (`git commit -m 'Add AmazingFeature'`)
4. Push (`git push origin feature/AmazingFeature`)
5. Abra Pull Request

---

## 📄 Licença

MIT License - veja [LICENSE](LICENSE)

---

## ⚠️ Disclaimer

**AVISO IMPORTANTE:**

- Este software é para **fins educacionais** apenas
- Trading envolve **risco de perda de capital**
- **Não é aconselhamento financeiro**
- Teste em conta **demo** antes de usar real
- Use por **sua conta e risco**

---

## 🎯 Próximos Passos

1. ✅ **Teste:** `python3 test_optimized_features.py`
2. ✅ **Seus dados:** `python3 example_precomputed_features.py`
3. ✅ **Treine:** `./train_hybrid.sh`
4. ✅ **API:** `cd src/inference && python3 service.py`
5. ✅ **Produça:** Integre com cTrader

**[📚 Documentação Completa →](INDEX_DOCUMENTATION.md)**

---

## 📞 Suporte

- **Issues:** [GitHub Issues](https://github.com/borges32/bot-trade/issues)
- **Documentação:** [INDEX_DOCUMENTATION.md](INDEX_DOCUMENTATION.md)
- **Exemplos:** `example_*.py`

---

**Sistema 100% pronto para uso! 🚀**

**[⭐ Star este projeto se foi útil!](https://github.com/borges32/bot-trade)**
