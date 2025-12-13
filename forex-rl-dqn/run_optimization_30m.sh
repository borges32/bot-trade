#!/bin/bash
# Script para otimizar hiperparâmetros - USDJPY 30m

echo "🚀 Iniciando otimização de hiperparâmetros para USDJPY 30m"
echo "=================================================="
echo ""

# Número de combinações a testar (padrão: 50)
MAX_COMBINATIONS=${1:-50}

echo "Configurações:"
echo "  - Dataset: usdjpy_history_30m.csv"
echo "  - Timeframe: 30 minutos"
echo "  - Max combinações: $MAX_COMBINATIONS"
echo "  - Output: optimization_results/usdjpy_30m/"
echo ""

# Executa otimização
python3 optimize_hyperparams_30m.py \
    --config config_hybrid_30m.yaml \
    --output-dir optimization_results/usdjpy_30m \
    --max-combinations $MAX_COMBINATIONS

echo ""
echo "✅ Otimização concluída!"
echo ""
echo "📊 Resultados salvos em:"
echo "  - CSV: optimization_results/usdjpy_30m/optimization_results.csv"
echo "  - Best Config: optimization_results/usdjpy_30m/best_config.json"
echo ""
echo "Para visualizar os resultados:"
echo "  cat optimization_results/usdjpy_30m/best_config.json"
echo "  head -20 optimization_results/usdjpy_30m/optimization_results.csv"
