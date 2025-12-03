# 🗺️ Mapeamento: Config → Features

## Estrutura do CSV Esperada

### ✅ Colunas Obrigatórias (6)
Definidas em `data.timestamp_col`, `data.open_col`, etc:

```yaml
data:
  timestamp_col: "timestamp"
  open_col: "open"
  high_col: "high"
  low_col: "low"
  close_col: "close"
  volume_col: "volume"
```

**No CSV:**
```
timestamp, open, high, low, close, volume
```

### ✅ Indicadores Pré-Calculados (13)
Definidos em `data.precomputed_indicators`:

```yaml
data:
  precomputed_indicators:
    rsi: "rsi"
    ema_fast: "ema_fast"
    ema_slow: "ema_slow"
    bb_upper: "bb_upper"
    bb_middle: "bb_middle"
    bb_lower: "bb_lower"
    atr: "atr"
    momentum_10: "momentum_10"
    momentum_20: "momentum_20"
    volatility: "volatility"
    volume_ma: "volume_ma"
    macd: "macd"
    macd_signal: "macd_signal"
```

**No CSV:**
```
rsi, ema_fast, ema_slow, bb_upper, bb_middle, bb_lower,
atr, momentum_10, momentum_20, volatility, volume_ma,
macd, macd_signal
```

---

## Como as Features São Criadas

### 1️⃣ OHLCV Básico (6) → Features de Preço (~12)

**Do CSV:**
- `open`, `high`, `low`, `close`, `volume`

**Features criadas automaticamente:**
```python
# Candlestick
range = high - low
range_pct = range / close
body = abs(close - open)
body_pct = body / close
upper_shadow = high - max(open, close)
lower_shadow = min(open, close) - low
is_bullish = 1 if close > open else 0

# Retornos
return_1 = close.pct_change(1)
return_3 = close.pct_change(3)
return_5 = close.pct_change(5)
return_10 = close.pct_change(10)
log_return = log(close / close.shift(1))
```

**Config relacionado:**
```yaml
features:
  use_returns: true
  return_periods: [1, 3, 5, 10]  # ← Define quais retornos calcular
```

---

### 2️⃣ RSI (do CSV) → Features Derivadas (~4)

**Do CSV:**
- `rsi` (já calculado)

**Features derivadas criadas:**
```python
rsi_normalized = (rsi - 50) / 50        # Normaliza para [-1, 1]
rsi_overbought = 1 if rsi > 70 else 0   # Flag overbought
rsi_oversold = 1 if rsi < 30 else 0     # Flag oversold
rsi_divergence = rsi.diff()             # Mudança do RSI
```

**Config relacionado:**
```yaml
features:
  use_rsi: true
  rsi_period: 14  # ← Usado APENAS se 'rsi' não estiver no CSV
```

**Comportamento:**
- ✅ Se CSV tem `rsi` → usa do CSV (mais rápido)
- ✅ Se CSV NÃO tem `rsi` → calcula com período 14

---

### 3️⃣ EMAs (do CSV) → Features Derivadas (~5)

**Do CSV:**
- `ema_fast`, `ema_slow`

**Features derivadas criadas:**
```python
ema_cross = ema_fast - ema_slow              # Diferença (crossover)
ema_cross_pct = ema_cross / close            # Crossover %
ema_cross_signal = 1 if ema_cross > 0 else 0 # Signal binário
price_ema_fast_dist = (close - ema_fast) / close
price_ema_slow_dist = (close - ema_slow) / close
```

**Config relacionado:**
```yaml
features:
  use_ema: true
  ema_periods: [9, 21, 55]  # ← Usado se ema_fast/slow não existirem
```

**Comportamento:**
- ✅ Se CSV tem `ema_fast` e `ema_slow` → usa do CSV
- ✅ Se CSV NÃO tem → calcula EMAs com períodos [9, 21, 55]

---

### 4️⃣ Bollinger Bands (do CSV) → Features Derivadas (~7)

**Do CSV:**
- `bb_upper`, `bb_middle`, `bb_lower`

**Features derivadas criadas:**
```python
bb_width = (bb_upper - bb_lower) / bb_middle
bb_position = (close - bb_lower) / (bb_upper - bb_lower)
bb_position_normalized = (bb_position - 0.5) * 2  # [-1, 1]
price_bb_upper_dist = (bb_upper - close) / close
price_bb_lower_dist = (close - bb_lower) / close
bb_breakout_upper = 1 if close > bb_upper else 0
bb_breakout_lower = 1 if close < bb_lower else 0
```

**Config relacionado:**
```yaml
features:
  use_bollinger: true
  bb_period: 20    # ← Usado se bb_upper não estiver no CSV
  bb_std: 2
```

---

### 5️⃣ MACD (do CSV) → Features Derivadas (~4)

**Do CSV:**
- `macd`, `macd_signal`

**Features derivadas criadas:**
```python
macd_hist = macd - macd_signal
macd_hist_change = macd_hist.diff()
macd_cross_signal = 1 if macd > macd_signal else 0
macd_strength = macd_hist.rolling(5).mean()
```

**Config relacionado:**
```yaml
features:
  use_macd: true
  macd_fast: 12    # ← Usado se 'macd' não estiver no CSV
  macd_slow: 26
  macd_signal: 9
```

---

### 6️⃣ ATR (do CSV) → Features Derivadas (~3)

**Do CSV:**
- `atr`

**Features derivadas criadas:**
```python
atr_normalized = atr / close
atr_ma = atr.rolling(20).mean()
atr_ratio = atr / atr_ma
```

**Config relacionado:**
```yaml
features:
  use_atr: true
  atr_period: 14   # ← Usado se 'atr' não estiver no CSV
```

---

### 7️⃣ Momentum (do CSV) → Features Derivadas (~2)

**Do CSV:**
- `momentum_10`, `momentum_20`

**Features derivadas criadas:**
```python
momentum_ratio = momentum_10 / momentum_20
momentum_convergence = momentum_10 - momentum_20
```

**Comportamento:**
- ✅ Usa do CSV (não há config para calcular momentum)

---

### 8️⃣ Volume (do CSV) → Features Derivadas (~3)

**Do CSV:**
- `volume`, `volume_ma`

**Features derivadas criadas:**
```python
volume_ratio = volume / volume_ma
volume_spike = 1 if volume_ratio > 2.0 else 0
volume_change = volume.pct_change()
```

**Config relacionado:**
```yaml
features:
  use_volume_features: true  # ← Ativa features de volume
```

---

### 9️⃣ Volatilidade (do CSV) → Features Derivadas (~4)

**Do CSV:**
- `volatility`

**Features derivadas criadas:**
```python
volatility_ma = volatility.rolling(20).mean()
volatility_ratio = volatility / volatility_ma
high_volatility = 1 if volatility_ratio > 1.5 else 0
low_volatility = 1 if volatility_ratio < 0.5 else 0
```

**Config relacionado:**
```yaml
features:
  use_volatility: true
  volatility_window: 20  # ← Usado se 'volatility' não estiver no CSV
```

---

### 🔟 Indicadores Complementares (~10)

**Sempre calculados (não vêm no CSV):**

```python
# Stochastic
stoch_k = ...
stoch_d = stoch_k.rolling(3).mean()
stoch_overbought = 1 if stoch_k > 80 else 0
stoch_oversold = 1 if stoch_k < 20 else 0

# ADX
adx = ...
strong_trend = 1 if adx > 25 else 0

# SMAs adicionais
sma_20 = close.rolling(20).mean()
sma_50 = close.rolling(50).mean()
price_sma_20_dist = (close - sma_20) / close
price_sma_50_dist = (close - sma_50) / close
```

**Config relacionado:**
```yaml
features:
  use_stochastic: true
  stoch_k: 14
  stoch_d: 3
  
  use_adx: true
  adx_period: 14
  
  use_sma: true
  sma_periods: [20, 50]  # ← Períodos das SMAs
```

---

### 1️⃣1️⃣ Features de Interação (~2)

**Sempre criadas:**

```python
# Convergência de sinais
signal_convergence = sum of [ema_cross_signal, macd_cross_signal, 
                             rsi_oversold, rsi_overbought]

# Regime de mercado
market_regime = cut(volatility_ratio * adx, bins=[0, 10, 30, inf])
```

---

## 📊 Resumo Total

| Origem | Quantidade | Exemplos |
|--------|------------|----------|
| **CSV OHLCV** | 6 | timestamp, open, high, low, close, volume |
| **CSV Indicadores** | 13 | rsi, ema_fast, bb_upper, atr, macd, etc. |
| **Features de Preço** | 12 | range, body, returns, shadows |
| **Features Derivadas** | 30-40 | rsi_normalized, ema_cross, bb_position |
| **Complementares** | 10 | stoch_k, adx, sma_20, sma_50 |
| **Interações** | 2 | signal_convergence, market_regime |
| **TOTAL** | **~75** | Features completas para treinar os modelos |

---

## 🎯 Exemplo Prático

### Seu CSV:
```csv
timestamp,open,high,low,close,volume,rsi,ema_fast,ema_slow,bb_upper,bb_middle,bb_lower,atr,momentum_10,momentum_20,volatility,volume_ma,macd,macd_signal
2024-01-01 00:00,150.00,150.50,149.50,150.20,5000,55.2,150.15,149.80,151.00,150.00,149.00,0.50,0.30,0.50,0.015,4800,0.35,0.30
```

### Features Criadas (~75):
```
OHLCV (6):
  timestamp, open, high, low, close, volume

Indicadores do CSV (13):
  rsi, ema_fast, ema_slow, bb_upper, bb_middle, bb_lower,
  atr, momentum_10, momentum_20, volatility, volume_ma, macd, macd_signal

Features de Preço (12):
  range, range_pct, body, body_pct, upper_shadow, lower_shadow,
  is_bullish, return_1, return_3, return_5, return_10, log_return

Features Derivadas de RSI (4):
  rsi_normalized, rsi_overbought, rsi_oversold, rsi_divergence

Features Derivadas de EMA (5):
  ema_cross, ema_cross_pct, ema_cross_signal,
  price_ema_fast_dist, price_ema_slow_dist

... e mais ~35 features
```

---

## ⚙️ Customização

### Quer usar nomes diferentes no CSV?

**Exemplo: seu CSV tem `RSI` em vez de `rsi`**

```yaml
data:
  precomputed_indicators:
    rsi: "RSI"              # ← Nome da coluna no SEU CSV
    ema_fast: "EMA_9"       # ← Nome no seu CSV
    ema_slow: "EMA_21"      # ← Nome no seu CSV
```

### Quer calcular indicadores em vez de usar do CSV?

**Remova do `precomputed_indicators`:**

```yaml
data:
  precomputed_indicators:
    # rsi: "rsi"  ← Comentado = será calculado
    ema_fast: "ema_fast"
    ema_slow: "ema_slow"
    # ... resto
```

### Quer adicionar mais SMAs?

```yaml
features:
  use_sma: true
  sma_periods: [10, 20, 50, 100, 200]  # ← Adicione mais períodos
```

### Quer mais retornos?

```yaml
features:
  use_returns: true
  return_periods: [1, 3, 5, 10, 20, 30]  # ← Adicione mais períodos
```

---

## ✅ Validação

Verifique se seu CSV está compatível:

```python
import pandas as pd
import yaml

# Carrega config
with open('config_hybrid.yaml') as f:
    config = yaml.safe_load(f)

# Carrega CSV
df = pd.read_csv('data/usdjpy_history_30m.csv')

# Verifica OHLCV
ohlcv = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
print("OHLCV:", all(col in df.columns for col in ohlcv))

# Verifica indicadores pré-calculados
indicators = list(config['data']['precomputed_indicators'].values())
missing = [ind for ind in indicators if ind not in df.columns]
print(f"Indicadores: {len(indicators) - len(missing)}/{len(indicators)}")
if missing:
    print(f"Faltando: {missing}")
```

**Deve mostrar:**
```
OHLCV: True
Indicadores: 13/13
```

Ou se faltarem alguns:
```
OHLCV: True
Indicadores: 10/13
Faltando: ['momentum_10', 'momentum_20', 'volatility']
```

(Não tem problema! Serão calculados automaticamente)

---

## 📚 Referência Rápida

**Arquivo:** `config_hybrid.yaml`

**Seções importantes:**
- `data.timestamp_col` ... `data.volume_col` → Mapeamento OHLCV
- `data.precomputed_indicators` → Indicadores do CSV
- `features.*` → Configuração de cálculo de indicadores

**Sistema de features:** `src/common/features_optimized.py`

**Testes:** `python3 test_optimized_features.py`

**Exemplo:** `python3 example_precomputed_features.py`
