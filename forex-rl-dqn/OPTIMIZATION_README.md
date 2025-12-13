# 🔬 Otimização de Hiperparâmetros - LightGBM Forex

Scripts para encontrar automaticamente a melhor combinação de features e hiperparâmetros para trading de Forex.

## 📋 O que é otimizado?

### **Features Técnicas:**
- ✅ EMAs (Exponential Moving Averages)
- ✅ MACD (Moving Average Convergence Divergence)
- ✅ RSI (Relative Strength Index)
- ✅ Bollinger Bands
- ✅ ATR (Average True Range)

### **Hiperparâmetros do LightGBM:**
- `prediction_horizon`: Quantos candles à frente prever
- `learning_rate`: Taxa de aprendizado (0.01 - 0.05)
- `num_leaves`: Número de folhas por árvore (31 - 70)
- `max_depth`: Profundidade máxima das árvores (4 - 8)
- `n_estimators`: Número de árvores (300 - 800)
- `min_child_samples`: Amostras mínimas por folha (10 - 30)
- `subsample`: Fração de amostras por árvore (0.7 - 0.9)
- `colsample_bytree`: Fração de features por árvore (0.7 - 0.9)
- `reg_alpha`: Regularização L1 (0.1 - 0.5)
- `reg_lambda`: Regularização L2 (0.1 - 0.5)

## 🚀 Como Usar

### **1. Otimização para USDJPY 15m**

```bash
# Testa 50 combinações (padrão)
./run_optimization_15m.sh

# Testa 100 combinações
./run_optimization_15m.sh 100

# Testa 20 combinações (mais rápido)
./run_optimization_15m.sh 20
```

### **2. Otimização para USDJPY 30m**

```bash
# Testa 50 combinações (padrão)
./run_optimization_30m.sh

# Testa 100 combinações
./run_optimization_30m.sh 100
```

## 📊 Resultados

Os resultados são salvos em:

```
optimization_results/
├── usdjpy_15m/
│   ├── optimization_results.csv      # Todos os experimentos
│   ├── best_config.json             # Melhor configuração encontrada
│   ├── best_result_explained.txt    # Relatório detalhado do melhor resultado
│   └── analysis_report.txt          # Relatório de análise (via analyze_optimization.py)
└── usdjpy_30m/
    ├── optimization_results.csv
    ├── best_config.json
    ├── best_result_explained.txt
    └── analysis_report.txt
```

### **CSV - Todos os Experimentos**

Contém todas as combinações testadas com suas métricas:

| Coluna | Descrição |
|--------|-----------|
| `combination_id` | ID do experimento |
| `use_ema`, `use_macd`, etc. | Features ativas |
| `prediction_horizon` | Horizonte de previsão |
| `learning_rate`, `num_leaves`, etc. | Hiperparâmetros |
| `test_rmse` | RMSE no conjunto de teste |
| `test_direction_acc` | Acurácia de direção (%) |
| `combined_score` | Score combinado (menor = melhor) |

### **best_result_explained.txt - Relatório Detalhado**

**Gerado automaticamente** ao final de cada otimização, este relatório contém:

1. **Métricas de Performance**
   - Score combinado com interpretação
   - RMSE, MAE, R² com explicações práticas
   - Acurácia direcional (crucial para trading)

2. **Features/Indicadores Ativados**
   - Lista de todos os 12 indicadores técnicos
   - Status (✓ ATIVO ou ✗ Desativado)
   - Descrição do que cada indicador faz

3. **Hiperparâmetros do LightGBM**
   - Todos os valores configurados
   - Explicação do que cada parâmetro controla
   - Impacto de cada valor na performance

4. **Top 5 Melhores Configurações**
   - Alternativas próximas ao melhor resultado
   - Comparativo de métricas e features

5. **Interpretação e Recomendações**
   - Como usar a configuração encontrada
   - Próximos passos sugeridos
   - Avaliação de qualidade (excelente/moderada/baixa)

**Exemplo de uso:**
```bash
# Após rodar a otimização
./run_optimization_30m.sh 100

# Verifique o relatório gerado automaticamente
cat optimization_results/usdjpy_30m/best_result_explained.txt
```

### **JSON - Melhor Config**

Configuração completa do melhor modelo encontrado:

```json
{
  "combined_score": 0.512345,
  "test_rmse": 0.001234,
  "test_direction_acc": 0.5678,
  "use_ema": true,
  "use_macd": true,
  "prediction_horizon": 5,
  "learning_rate": 0.03,
  "num_leaves": 50,
  ...
}
```

## 🔍 Analisar Resultados

```bash
# Análise detalhada para 15m
python3 analyze_optimization.py optimization_results/usdjpy_15m

# Análise detalhada para 30m
python3 analyze_optimization.py optimization_results/usdjpy_30m
```

O script de análise mostra:
- ✅ Estatísticas gerais de todas as métricas
- 🏆 Melhor configuração encontrada
- 🌟 Top 10 melhores configurações
- 🔍 Impacto de cada feature no resultado
- 📊 Correlação dos parâmetros com o score
- 📈 Distribuição de scores (percentis)

## 📈 Métricas Avaliadas

### **RMSE (Root Mean Squared Error)**
- Mede o erro de previsão do retorno
- **Menor é melhor**
- Valores típicos: 0.0005 - 0.0020

### **Direction Accuracy**
- Acurácia em prever a direção do preço (subir/descer)
- **Maior é melhor**
- Valores típicos: 0.50 - 0.55 (50-55%)

### **Combined Score**
- Score combinado: `RMSE + (1 - Direction_Acc)`
- **Menor é melhor**
- Penaliza tanto erro de magnitude quanto erro de direção

## ⚙️ Estratégia de Busca

### **Random Search**
- Mais eficiente que Grid Search para espaços grandes
- Testa combinações aleatórias de hiperparâmetros
- Primeiro testa todas as combinações de features
- Depois faz random search dos outros parâmetros

### **Número de Combinações**
- **20-30**: Busca rápida (1-2 horas)
- **50**: Padrão recomendado (2-4 horas)
- **100**: Busca completa (4-8 horas)

## 🎯 Exemplos de Uso

### **Otimização Rápida (Teste)**
```bash
# 10 combinações para testar o sistema
./run_optimization_15m.sh 10
```

### **Otimização Padrão**
```bash
# 50 combinações balanceadas
./run_optimization_15m.sh 50
./run_optimization_30m.sh 50
```

### **Otimização Completa (Overnight)**
```bash
# 100 combinações - deixe rodando à noite
./run_optimization_15m.sh 100
./run_optimization_30m.sh 100
```

## 📝 Como Interpretar

### **1. Verifique o Combined Score**
- Quanto menor, melhor
- Valores < 0.50 são excelentes
- Valores < 0.52 são bons
- Valores > 0.55 são fracos

### **2. Verifique Direction Accuracy**
- > 52% = Bom (melhor que aleatório)
- > 54% = Muito bom
- > 56% = Excelente
- < 50% = Pior que aleatório (inverter sinais!)

### **3. Analise Features**
- Quais features melhoram o score?
- Combinações de features funcionam melhor?
- Simplicidade vs complexidade

### **4. Analise Hiperparâmetros**
- Learning rate muito alto = overfitting
- Num leaves muito alto = overfitting
- Regularização muito alta = underfitting

## 🔄 Aplicar Melhor Config

Depois de encontrar a melhor configuração:

```bash
# Visualize a melhor config
cat optimization_results/usdjpy_15m/best_config.json

# Aplique manualmente ao config_hybrid_15m.yaml
# Ou use o script de aplicação automática:
# TODO: Criar script apply_best_config.py
```

## ⚠️ Observações

1. **Tempo de Execução**: Cada combinação leva ~2-5 minutos
   - 50 combinações = 2-4 horas
   - 100 combinações = 4-8 horas

2. **Memória**: Cada modelo treina em ~32k samples
   - Precisa de ~2-4 GB de RAM disponível

3. **Resultados Parciais**: São salvos a cada 5 combinações
   - Se interromper, pode continuar de onde parou

4. **Reprodutibilidade**: Usa `random_state=42`
   - Mesmos parâmetros = mesmos resultados

## 🎓 Próximos Passos

Após a otimização:

1. ✅ Analise os resultados com `analyze_optimization.py`
2. ✅ Identifique padrões nas melhores configurações
3. ✅ Teste a melhor config em dados novos (forward testing)
4. ✅ Monitore performance em produção
5. ✅ Re-otimize periodicamente (a cada 1-3 meses)

## 🐛 Troubleshooting

### **Erro: "Config file not found"**
```bash
# Verifique se os configs existem:
ls -la config_hybrid_*.yaml
```

### **Erro: "Module not found"**
```bash
# Instale dependências:
pip install -r requirements.txt
```

### **Performance muito ruim (< 50% accuracy)**
- Dados podem estar com problemas
- Verifique qualidade dos dados
- Tente outros pares de moedas
- Ajuste prediction_horizon

---

**Criado por**: Sistema de Otimização Automática  
**Versão**: 1.0.0  
**Data**: Dezembro 2025
