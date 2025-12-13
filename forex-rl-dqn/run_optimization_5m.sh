#!/bin/bash
# Script para executar otimização de hiperparâmetros - USDJPY 5m

echo "🚀 Iniciando otimização de hiperparâmetros para USDJPY 5m..."
echo ""

# Define número de combinações a testar (padrão: 50)
MAX_COMBINATIONS=${1:-50}

# Executa otimização
python3 optimize_hyperparams_5m.py \
    --config config_hybrid_5m.yaml \
    --output-dir optimization_results/usdjpy_5m \
    --max-combinations $MAX_COMBINATIONS

echo ""
echo "✅ Otimização concluída!"
echo ""
echo "📊 Resultados salvos em: optimization_results/usdjpy_5m/"
echo "   - optimization_results.csv: Todos os resultados"
echo "   - best_config.json: Melhor configuração encontrada"
echo "   - best_result_explained.txt: Relatório detalhado"
echo ""
echo "💡 Para treinar o modelo com a melhor configuração:"
echo "   python -m src.training.train_lightgbm optimization_results/usdjpy_5m/best_config.yaml"
