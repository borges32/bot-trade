# ⚡ GUIA RÁPIDO: Features Otimizadas

## 🎯 O Que Mudou?

O sistema agora **usa os indicadores do seu CSV** em vez de recalcular tudo.

**Resultado:** 10x mais rápido + mais features + mesma qualidade! 🚀

---

## 📋 Checklist Rápido

### ✅ Seu CSV precisa ter:

**Obrigatório (OHLCV):**
- `timestamp`, `open`, `high`, `low`, `close`, `volume`

**Indicadores (já calculados pelo cTrader):**
- `rsi` - RSI
- `ema_fast`, `ema_slow` - EMAs
- `bb_upper`, `bb_middle`, `bb_lower` - Bollinger Bands
- `atr` - Average True Range
- `momentum_10`, `momentum_20` - Momentum
- `volatility` - Volatilidade
- `volume_ma` - Volume MA
- `macd`, `macd_signal` - MACD

---

## 🚀 Como Usar

### 1. Teste Rápido (3 segundos)
```bash
python3 test_optimized_features.py
```

**Deve mostrar:**
```
✓ 13 indicadores pré-calculados detectados
✓ 56 features novas criadas
✓ 75 colunas totais
✓ 4/4 testes passaram
✓ SUCESSO!
```

### 2. Teste com Seu CSV
```bash
# Coloque seu arquivo em: data/usdjpy_history_15m.csv
python3 example_precomputed_features.py
```

**Vai mostrar:**
- Quais indicadores foram detectados
- Quais features foram criadas
- Estatísticas e qualidade dos dados
- Salva resultado em `data/processed_features.csv`

### 3. Treine os Modelos
```bash
./train_hybrid.sh
```

**Ou manualmente:**
```bash
python3 src/training/train_lightgbm.py
python3 src/training/train_ppo.py
```

---

## 📊 Features Finais (75 total)

### Do CSV (19)
- OHLCV básico (6)
- Indicadores pré-calculados (13)

### Criadas (56)
1. **Price features** (12): range, body, shadows, returns
2. **RSI derivadas** (4): normalized, overbought, oversold, divergence
3. **EMA derivadas** (5): cross, cross_pct, signal, distances
4. **BB derivadas** (7): width, position, breakouts, distances
5. **MACD derivadas** (4): histogram, change, cross, strength
6. **ATR derivadas** (3): normalized, ma, ratio
7. **Momentum derivadas** (2): ratio, convergence
8. **Volume derivadas** (3): ratio, spike, change
9. **Volatility derivadas** (4): ma, ratio, high/low flags
10. **Complementares** (10): Stochastic, ADX, SMAs
11. **Interações** (2): signal convergence, market regime

---

## ⚡ Performance

**Com 1000 candles:**
- Tempo: ~0.04 segundos
- Velocidade: ~24,000 candles/seg
- Qualidade: 0 NaN, 0 infinitos

**Com 10,000 candles:**
- Tempo: ~0.5 segundos
- Arquivo: ~5-10 MB
- Uso RAM: ~50-100 MB

---

## 🔍 Verificação Rápida

### Seu CSV está OK?
```python
import pandas as pd

df = pd.read_csv('data/usdjpy_history_15m.csv')

# Tem as colunas obrigatórias?
required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
print("OHLCV:", all(col in df.columns for col in required))

# Tem os indicadores?
indicators = ['rsi', 'ema_fast', 'ema_slow', 'bb_upper', 'macd']
print("Indicadores:", all(col in df.columns for col in indicators))
```

**Deve mostrar:**
```
OHLCV: True
Indicadores: True
```

---

## ❓ Problemas Comuns

### "Arquivo não encontrado"
```bash
# Certifique-se de colocar o CSV no lugar certo:
ls -lh data/usdjpy_history_15m.csv
```

### "Coluna XXX não encontrada"
```python
# Veja quais colunas existem:
import pandas as pd
df = pd.read_csv('seu_arquivo.csv')
print(df.columns.tolist())

# Renomeie se necessário:
df.rename(columns={'Time': 'timestamp', 'RSI': 'rsi'}, inplace=True)
```

### "Muitos NaN"
```python
# Remova linhas com muitos NaN:
df = df.dropna(thresh=len(df.columns) * 0.8)  # Mantém linhas com 80%+ dados
```

---

## 📚 Documentação Completa

- `OPTIMIZATION_SUMMARY.md` → Resumo executivo
- `OPTIMIZED_FEATURES.md` → Guia detalhado
- `example_precomputed_features.py` → Exemplo prático
- `test_optimized_features.py` → Testes automatizados

---

## ✅ Próximos Passos

1. ✅ Rode `python3 test_optimized_features.py` → **Deve passar!**
2. ✅ Coloque seu CSV em `data/usdjpy_history_15m.csv`
3. ✅ Rode `python3 example_precomputed_features.py` → **Veja features**
4. ✅ Rode `./train_hybrid.sh` → **Treine modelos**
5. ✅ Use em produção! 🚀

---

## 🎉 TL;DR

**Antes:**
- Recalculava tudo
- ~5-10 segundos
- ~50 features

**Agora:**
- Usa indicadores do CSV
- ~0.5 segundos (10x mais rápido!)
- ~75 features (50% mais features!)

**Como usar:**
```bash
python3 test_optimized_features.py  # Testa (3 seg)
./train_hybrid.sh                   # Treina (funciona!)
```

**Pronto! 🚀**
