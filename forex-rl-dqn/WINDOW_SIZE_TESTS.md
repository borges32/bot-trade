# Guia de Testes - Window Size Optimization

## 📋 Objetivo

Testar diferentes tamanhos de janela (window size) para encontrar a configuração ideal que maximize o desempenho do modelo.

## 🧪 Configurações de Teste

### Teste 1: Window 64 (Baseline Otimizado)
- **Arquivo:** `config_optimized.yaml`
- **Window:** 64 barras (~1 hora em M1)
- **Uso:** Day trading, scalping
- **Artifacts:** `artifacts_w64/`

### Teste 2: Window 96 (Recomendado)
- **Arquivo:** `config_window_96.yaml`
- **Window:** 96 barras (~1.5 horas em M1)
- **Uso:** Swing trading intraday
- **Artifacts:** `artifacts_w96/`

### Teste 3: Window 128 (Tendências Longas)
- **Arquivo:** `config_window_128.yaml`
- **Window:** 128 barras (~2 horas em M1)
- **Uso:** Position trading
- **Artifacts:** `artifacts_w128/`

## 🚀 Executando os Testes

### Passo 1: Executar Treinamentos

```bash
# Teste 1: Window 64
python3 -m src.rl.train \
  --data data/usdjpy_history.csv \
  --config config_optimized.yaml \
  --artifacts artifacts_w64

# Teste 2: Window 96
python3 -m src.rl.train \
  --data data/usdjpy_history.csv \
  --config config_window_96.yaml \
  --artifacts artifacts_w96

# Teste 3: Window 128
python3 -m src.rl.train \
  --data data/usdjpy_history.csv \
  --config config_window_128.yaml \
  --artifacts artifacts_w128
```

### Passo 2: Verificar Timeframe dos Dados

Antes de interpretar resultados, confirme o timeframe:

```bash
python3 -c "
import pandas as pd
df = pd.read_csv('data/usdjpy_history.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
diff = (df['timestamp'].iloc[1] - df['timestamp'].iloc[0]).total_seconds() / 60
print(f'Timeframe: {diff:.0f} minutos')
print(f'Window 64 = {64*diff:.0f} minutos ({64*diff/60:.1f} horas)')
print(f'Window 96 = {96*diff:.0f} minutos ({96*diff/60:.1f} horas)')
print(f'Window 128 = {128*diff:.0f} minutos ({128*diff/60:.1f} horas)')
"
```

## 📊 Monitoramento Durante Treinamento

### Métricas a Observar

Durante o treinamento, monitore os logs:

```bash
# Acompanhar em tempo real
tail -f artifacts_w96/training.log
```

**Sinais POSITIVOS:**
- ✅ Loss > 0.001 (oscilando, não zero)
- ✅ Win Rate aumentando gradualmente
- ✅ Avg Position Reward ≠ 0 (modelo fazendo trades)
- ✅ Epsilon decaindo de 1.0 → 0.1

**Sinais NEGATIVOS:**
- ❌ Loss = 0.0000 persistente
- ❌ Win Rate = 0%
- ❌ Avg Position Reward = 0 (apenas HOLD)
- ❌ Avg Reward não muda

### Checkpoints Importantes

Avalie nos seguintes steps:
- **Step 5,000:** Primeiras tendências
- **Step 50,000:** Epsilon ~0.5 (meio da exploração)
- **Step 100,000:** Epsilon ~0.1 (exploração mínima)
- **Step 200,000:** Treinamento completo

## 📈 Comparação de Resultados

### Método 1: Comparar Logs Finais

```bash
echo "=== WINDOW 64 ==="
tail -30 artifacts_w64/*.log 2>/dev/null | grep -A5 "Final Evaluation"

echo -e "\n=== WINDOW 96 ==="
tail -30 artifacts_w96/*.log 2>/dev/null | grep -A5 "Final Evaluation"

echo -e "\n=== WINDOW 128 ==="
tail -30 artifacts_w128/*.log 2>/dev/null | grep -A5 "Final Evaluation"
```

### Método 2: Extrair Métricas Chave

```bash
# Criar script de comparação
cat > compare_results.sh << 'EOF'
#!/bin/bash

echo "Comparação de Resultados - Window Size Tests"
echo "=============================================="
echo ""

for dir in artifacts_w64 artifacts_w96 artifacts_w128; do
    if [ -d "$dir" ]; then
        window=$(grep "window_size" $dir/config.yaml | awk '{print $2}')
        echo "Window Size: $window"
        echo "---"
        
        # Última avaliação
        tail -100 $dir/*.log 2>/dev/null | grep -A4 "Final Evaluation" | tail -5
        echo ""
    fi
done
EOF

chmod +x compare_results.sh
./compare_results.sh
```

### Método 3: Teste em Dados de Validação

```bash
# Criar conjunto de teste separado (últimos 20%)
python3 -c "
import pandas as pd
df = pd.read_csv('data/usdjpy_history.csv')
split = int(len(df) * 0.8)
test_df = df[split:]
test_df.to_csv('data/usdjpy_test.csv', index=False)
print(f'Test set: {len(test_df)} registros')
"

# Avaliar cada modelo no test set
# (Você precisaria criar um script de avaliação)
```

## 📋 Critérios de Decisão

### Escolha a configuração que tiver:

1. **Win Rate:** > 50% (ideal > 55%)
2. **Avg Reward:** Positivo e estável
3. **Avg Position Reward:** > 0 (fazendo trades lucrativos)
4. **Loss:** Convergindo (0.001 - 0.05, não zero)
5. **Avg Cost:** < Avg Position Reward

### Exemplo de Resultado BOM:

```
Window 96:
--- Final Evaluation ---
Avg Reward: 0.003245
Avg Position Reward: 0.004100
Avg Cost: 0.000855
Win Rate: 56.23%
```

### Exemplo de Resultado RUIM:

```
Window 64:
--- Final Evaluation ---
Avg Reward: 0.000000
Avg Position Reward: 0.000000
Avg Cost: 0.000000
Win Rate: 0.00%
```

## 🎯 Matriz de Decisão

| Métrica | Window 64 | Window 96 | Window 128 | Melhor |
|---------|-----------|-----------|------------|--------|
| Win Rate | __%  | __%  | __%  | ? |
| Avg Reward | _____ | _____ | _____ | ? |
| Pos. Reward | _____ | _____ | _____ | ? |
| Training Time | Rápido | Médio | Lento | - |
| Memory Usage | Baixo | Médio | Alto | - |

**Preencha após os testes e escolha a melhor configuração.**

## 🔍 Análise Esperada

### Se Window 96 for melhor:
- ✅ Captura tendências de curto-médio prazo
- ✅ Balance entre contexto e velocidade
- ✅ Ideal para Forex intraday

### Se Window 64 for melhor:
- ✅ Mais responsivo a mudanças rápidas
- ✅ Melhor para scalping
- ✅ Treina mais rápido

### Se Window 128 for melhor:
- ✅ Melhor para swing trading
- ✅ Captura tendências mais longas
- ✅ Menos trades, mais precisão

### Se TODOS falharem (Win Rate ~0%):
- ❌ Problema não é window size
- ❌ Verificar custos (fee/spread)
- ❌ Analisar qualidade dos dados
- ❌ Revisar reward function

## 📝 Registro de Resultados

### Template para documentar:

```
Data do Teste: 25/11/2025
Dataset: usdjpy_history.csv (60,436 registros)
Timeframe: ___ minutos

TESTE 1 - Window 64:
- Win Rate: ___%
- Avg Reward: _____
- Training Time: ___ min
- Observações: _______________

TESTE 2 - Window 96:
- Win Rate: ___%
- Avg Reward: _____
- Training Time: ___ min
- Observações: _______________

TESTE 3 - Window 128:
- Win Rate: ___%
- Avg Reward: _____
- Training Time: ___ min
- Observações: _______________

CONCLUSÃO:
Melhor configuração: Window ___
Razão: _______________
Próximos passos: _______________
```

## 🚨 Troubleshooting

### Problema: Todos os testes com Win Rate = 0%

**Solução:**
1. Verificar custos muito altos (já corrigido nos configs)
2. Coletar mais dados (>100k registros idealmente)
3. Testar par mais volátil (GBPJPY, XAUUSD)

### Problema: Training muito lento

**Solução:**
```bash
# Reduzir max_steps temporariamente para teste
sed -i 's/max_steps: 200000/max_steps: 50000/' config_window_96.yaml

# Ou usar GPU (se disponível)
sed -i 's/device: auto/device: cuda/' config_window_96.yaml
```

### Problema: Out of Memory

**Solução:**
```bash
# Reduzir batch_size
sed -i 's/batch_size: 64/batch_size: 32/' config_window_128.yaml

# Ou reduzir replay_size
sed -i 's/replay_size: 100000/replay_size: 50000/' config_window_128.yaml
```

## ✅ Checklist de Execução

- [ ] Verificar timeframe dos dados
- [ ] Executar teste Window 64
- [ ] Executar teste Window 96
- [ ] Executar teste Window 128
- [ ] Comparar resultados finais
- [ ] Documentar métricas
- [ ] Escolher melhor configuração
- [ ] Re-treinar com config escolhida (se necessário)
- [ ] Validar em dados out-of-sample

## 📚 Próximos Passos

Após identificar a melhor configuração:

1. **Re-treinar** com mais steps (300k-500k)
2. **Validar** em dados novos (out-of-sample)
3. **Backtest** em período diferente
4. **Paper trading** antes de usar real
5. **Monitorar** performance contínua

---

**Criado:** 25/11/2025  
**Última atualização:** 25/11/2025
