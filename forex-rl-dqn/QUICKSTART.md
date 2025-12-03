# 🚀 Guia Rápido - Sistema Híbrido LightGBM + PPO

## ⚡ Início Rápido (5 passos)

### 1️⃣ Instalar Dependências
```bash
pip install -r requirements.txt
```

### 2️⃣ Preparar Dados
Coloque seu arquivo CSV do cTrader em `data/`:
```bash
# Exemplo: data/usdjpy_history_30m.csv
```

Formato esperado:
```csv
timestamp,open,high,low,close,volume
2024-01-01 00:00:00,1.0950,1.0960,1.0945,1.0955,1000.0
```

### 3️⃣ Configurar (Opcional)
Edite `config_hybrid.yaml` se necessário:
- Par de moedas (custos de transação)
- Timeframe (prediction_horizon)
- Parâmetros de risco

### 4️⃣ Treinar Modelos
```bash
./train_hybrid.sh
```

Ou individualmente:
```bash
# Primeiro LightGBM
python -m src.training.train_lightgbm

# Depois PPO
python -m src.training.train_ppo
```

### 5️⃣ Usar o Sistema

**Opção A: API HTTP**
```bash
cd src/inference
python service.py
```

**Opção B: Python Direto**
```bash
python example_hybrid_usage.py
```

## 📊 Verificação Rápida

### Testar Instalação
```bash
python test_hybrid_system.py
```

Deve mostrar:
```
✅ Dados: OK
✅ Features: OK
✅ Ambiente: OK
✅ Modelos: OK (após treinamento)
```

### Verificar Modelos Treinados
```bash
ls -lh models/hybrid/
```

Deve conter:
- `lightgbm_model.txt`
- `ppo_model.zip`

## 🎯 Exemplo de Uso da API

### Iniciar Servidor
```bash
cd src/inference
python service.py
```

### Fazer Request (Python)
```python
import requests
import pandas as pd

# Carrega candles recentes
df = pd.read_csv('data/usdjpy_history_30m.csv').tail(100)
candles = df.to_dict('records')

# Solicita sinal
response = requests.post('http://localhost:8000/signal', json={
    'candles': candles,
    'current_position': 0
})

signal = response.json()
print(f"Ação: {signal['action_name']}")
print(f"Confiança: {signal['confidence']:.2%}")
```

### Fazer Request (curl)
```bash
curl -X POST http://localhost:8000/signal \
  -H "Content-Type: application/json" \
  -d @candles.json
```

## 📈 Interpretando Resultados

### Sinal de Trading
```json
{
  "action": 1,
  "action_name": "comprar",
  "lightgbm_signal": 0.65,
  "confidence": 0.80
}
```

- **action_name**: "neutro", "comprar" ou "vender"
- **confidence**: 0-1, use threshold (ex: 0.6) para filtrar
- **lightgbm_signal**: 
  - Classifier: 0-1 (probabilidade de alta)
  - Regressor: retorno esperado

### Recomendações
- ✅ **confidence > 0.7**: Sinal forte, considerar executar
- ⚠️ **confidence 0.5-0.7**: Sinal moderado, avaliar contexto
- ❌ **confidence < 0.5**: Sinal fraco, evitar

## ⚙️ Ajustes Rápidos

### Mudar Par de Moedas
```yaml
# config_hybrid.yaml
data:
  train_file: "data/euraud_history_30m.csv"

ppo:
  env:
    commission: 0.0003  # Ajustar para spread do par
```

### Mudar Timeframe
```yaml
lightgbm:
  prediction_horizon: 10  # 5M=3-5, 15M=5-7, 30M=5-10, 1H=7-15
```

### Tornar Mais Conservador
```yaml
ppo:
  env:
    leverage: 1.0  # Sem alavancagem
    stop_loss_pct: 0.015  # 1.5% (mais apertado)
    max_position_size: 0.5  # 50% do capital
```

### Tornar Mais Agressivo
```yaml
ppo:
  env:
    leverage: 5.0  # ⚠️ CUIDADO!
    stop_loss_pct: 0.03  # 3%
    max_position_size: 1.0  # 100% do capital
```

## 🐛 Problemas Comuns

### "FileNotFoundError: lightgbm_model.txt"
**Solução**: Treinar modelos primeiro
```bash
./train_hybrid.sh
```

### "Insufficient candles"
**Solução**: Enviar ≥50 candles no request

### Performance ruim em backtest
**Soluções**:
1. Aumentar dados de treino (≥6 meses)
2. Ajustar custos realistas (commission/slippage)
3. Revisar prediction_horizon para seu timeframe
4. Treinar por mais tempo (total_timesteps)

### API não responde
**Verificar**:
```bash
# Modelos existem?
ls models/hybrid/

# Porta ocupada?
lsof -i :8000

# Logs de erro?
cd src/inference
python service.py  # Ver output
```

## 📚 Documentação Completa

- **README_HYBRID.md**: Documentação detalhada
- **HYBRID_IMPLEMENTATION.md**: Detalhes técnicos
- **config_hybrid.yaml**: Referência de configuração

## 🎓 Próximos Passos

1. ✅ Testar sistema com dados históricos
2. ✅ Ajustar hiperparâmetros
3. ✅ Fazer backtest completo
4. ⚠️ Testar em conta demo
5. ⚠️ Monitorar por 1-2 meses
6. ⚠️ Avaliar usar capital real (com cuidado!)

## ⚠️ Avisos Importantes

- 🔴 **NUNCA** use em conta real sem backtest extensivo
- 🔴 **SEMPRE** teste em conta demo primeiro
- 🔴 **MONITORE** constantemente (métricas podem degradar)
- 🔴 **AJUSTE** custos de transação realisticamente
- 🔴 **COMECE** com capital pequeno

---

**Suporte**: Consulte documentação detalhada em `README_HYBRID.md`
