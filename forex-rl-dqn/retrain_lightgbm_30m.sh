#!/bin/bash

echo "========================================================================"
echo "RETREINAMENTO DO LIGHTGBM - CONFIG 30M OTIMIZADA"
echo "========================================================================"
echo ""
echo "📊 Melhorias implementadas:"
echo "   ✓ Prediction Horizon: 6 → 20 candles (10 horas)"
echo "   ✓ Classification Threshold: 0.00015 → 0.0001 (balance 50.9%)"
echo "   ✓ Class Weight: balanced (equaliza classes)"
echo "   ✓ Feature Selection: remove features com corr < 0.01"
echo ""
echo "🎯 Expectativa: Acurácia > 55% (vs 51.41% anterior)"
echo ""
echo "========================================================================"
echo ""

# Para treinamento anterior se existir
pkill -f train_lightgbm.py 2>/dev/null

# Remove modelos antigos (backup)
if [ -d "models/hybrid_30m" ]; then
    echo "📦 Fazendo backup de modelos antigos..."
    mv models/hybrid_30m models/hybrid_30m_backup_$(date +%Y%m%d_%H%M%S)
fi

# Cria diretório
mkdir -p models/hybrid_30m
mkdir -p logs/hybrid_30m

echo "🚀 Iniciando treinamento..."
echo ""

# Treina LightGBM com nova configuração
python3 -m src.training.train_lightgbm --config config_hybrid_30m.yaml

echo ""
echo "========================================================================"
echo "✅ TREINAMENTO CONCLUÍDO!"
echo "========================================================================"
echo ""
echo "📊 Verifique as métricas acima:"
echo "   - Test Accuracy deve estar > 55%"
echo "   - Test AUC deve estar > 0.60"
echo ""
echo "💡 Próximos passos se acurácia ainda baixa:"
echo "   1. Aumentar prediction_horizon (30 candles)"
echo "   2. Ajustar threshold (0.0003)"
echo "   3. Usar regressão em vez de classificação"
echo ""
