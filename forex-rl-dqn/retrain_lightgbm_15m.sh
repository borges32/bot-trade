#!/bin/bash

echo "========================================================================"
echo "RETREINAMENTO DO LIGHTGBM - MODELO DE REGRESSÃO (15 MINUTOS)"
echo "========================================================================"
echo ""
echo "🔄 Timeframe: 15 MINUTOS"
echo ""
echo "📊 Configuração:"
echo "   ✓ Model Type: REGRESSOR (preve retorno contínuo)"
echo "   ✓ Prediction Horizon: 8 candles (2 horas)"
echo "   ✓ Target: Retorno percentual (ex: +0.15%, -0.08%)"
echo "   ✓ Metric: RMSE (menor é melhor)"
echo "   ✓ Regularização: alpha=0.3, lambda=0.3 (evita overfitting)"
echo ""
echo "🎯 Características do 15m:"
echo "   ✓ Mais sinais (timeframe mais curto)"
echo "   ✓ Indicadores mais sensíveis (RSI=10, EMA=[8,21,50])"
echo "   ✓ Horizonte de predição: 2 horas"
echo "   ✓ Ideal para intraday trading"
echo ""
echo "📈 Métricas Esperadas:"
echo "   - RMSE Train: 0.001-0.003 (0.1-0.3%)"
echo "   - RMSE Test: 0.0015-0.004 (0.15-0.4%)"
echo "   - Direction Accuracy: > 52%"
echo ""
echo "========================================================================"
echo ""

# Para treinamento anterior se existir
pkill -f train_lightgbm.py 2>/dev/null

# Cria diretórios
mkdir -p models/hybrid_15m
mkdir -p logs/hybrid_15m

echo "🚀 Iniciando treinamento..."
echo ""

# Treina LightGBM com regressão
python3 -m src.training.train_lightgbm --config config_hybrid_15m.yaml

echo ""
echo "========================================================================"
echo "✅ TREINAMENTO CONCLUÍDO!"
echo "========================================================================"
echo ""
echo "📊 Interpretação das Métricas de Regressão:"
echo ""
echo "   RMSE (Root Mean Squared Error):"
echo "     - Quanto menor, melhor"
echo "     - RMSE = 0.002 significa erro médio de ±0.2%"
echo ""
echo "   R² Score:"
echo "     - 0.0 = não melhor que média"
echo "     - > 0.05 = captura 5% da variância (BOM para Forex)"
echo "     - > 0.10 = excelente para trading"
echo ""
echo "   MAE (Mean Absolute Error):"
echo "     - Erro médio absoluto das previsões"
echo "     - MAE < RMSE indica poucos outliers"
echo ""
echo "   Direction Accuracy:"
echo "     - % de vezes que prevê direção correta"
echo "     - > 52% = melhor que aleatório"
echo "     - > 55% = excelente para forex"
echo ""
echo "💡 Próximos passos:"
echo "   1. Verificar métricas acima"
echo "   2. Testar predições: python3 example_lightgbm_usage.py"
echo "   3. Ajustar threshold em config_hybrid_15m.yaml se necessário"
echo ""
echo "📁 Modelo salvo em: models/hybrid_15m/lightgbm_model.txt"
echo ""
