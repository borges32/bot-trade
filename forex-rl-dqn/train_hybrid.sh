#!/bin/bash
# Script completo de treinamento: LightGBM + PPO

set -e  # Sai se houver erro

echo "========================================="
echo "  Sistema Híbrido LightGBM + PPO"
echo "  Treinamento Completo"
echo "========================================="
echo ""

CONFIG_FILE="${1:-config_hybrid_15m.yaml}"

echo "Usando configuração: $CONFIG_FILE"
echo ""

# 1. Treina LightGBM
echo "========================================="
echo "PASSO 1/2: Treinando LightGBM"
echo "========================================="
python3 -m src.training.train_lightgbm --config "$CONFIG_FILE"

if [ $? -ne 0 ]; then
    echo "❌ Erro no treinamento do LightGBM"
    exit 1
fi

echo ""
echo "✅ LightGBM treinado com sucesso!"
echo ""

# 2. Treina PPO
echo "========================================="
echo "PASSO 2/2: Treinando PPO"
echo "========================================="
python3 -m src.training.train_ppo --config "$CONFIG_FILE"

if [ $? -ne 0 ]; then
    echo "❌ Erro no treinamento do PPO"
    exit 1
fi

echo ""
echo "✅ PPO treinado com sucesso!"
echo ""

# Conclusão
echo "========================================="
echo "  ✅ TREINAMENTO COMPLETO!"
echo "========================================="
echo ""
echo "Modelos salvos em: models/hybrid/"
echo ""
echo "Próximos passos:"
echo "  1. Revisar métricas em models/hybrid/*_metrics_*.yaml"
echo "  2. Visualizar logs: tensorboard --logdir logs/hybrid"
echo "  3. Iniciar API: cd src/inference && python service.py"
echo ""
