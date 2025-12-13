#!/bin/bash
# Script para otimizar hiperparâmetros - USDJPY 15m

echo "🚀 Iniciando otimização de hiperparâmetros para USDJPY 15m"
echo "=================================================="
echo ""

# Número de combinações a testar (padrão: 50)
MAX_COMBINATIONS=${1:-50}

echo "Configurações:"
echo "  - Dataset: usdjpy_history_15m.csv"
echo "  - Timeframe: 15 minutos"
echo "  - Max combinações: $MAX_COMBINATIONS"
echo "  - Output: optimization_results/usdjpy_15m/"
echo ""

# Executa otimização
python3 optimize_hyperparams_15m.py \
    --config config_hybrid_15m.yaml \
    --output-dir optimization_results/usdjpy_15m \
    --max-combinations $MAX_COMBINATIONS

echo ""
echo "✅ Otimização concluída!"
echo ""
echo "📊 Resultados salvos em:"
echo "  - CSV: optimization_results/usdjpy_15m/optimization_results.csv"
echo "  - Best Config: optimization_results/usdjpy_15m/best_config.json"
echo ""
echo "Para visualizar os resultados:"
echo "  cat optimization_results/usdjpy_15m/best_config.json"
echo "  head -20 optimization_results/usdjpy_15m/optimization_results.csv"
