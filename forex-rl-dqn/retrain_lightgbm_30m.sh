#!/bin/bash

echo "========================================================================"
echo "RETREINAMENTO DO LIGHTGBM - MODELO DE REGRESSÃO"
echo "========================================================================"
echo ""
echo "🔄 MUDANÇA: Classificação → Regressão"
echo ""
echo "📊 Configuração:"
echo "   ✓ Model Type: REGRESSOR (preve retorno contínuo)"
echo "   ✓ Prediction Horizon: 10 candles (5 horas)"
echo "   ✓ Target: Retorno percentual (ex: +0.15%, -0.08%)"
echo "   ✓ Metric: RMSE (menor é melhor)"
echo "   ✓ Regularização: alpha=0.3, lambda=0.3 (evita overfitting)"
echo ""
echo "🎯 Vantagens da Regressão:"
echo "   ✓ Prevê MAGNITUDE do movimento (não só direção)"
echo "   ✓ Sem problema de threshold (classificação binária)"
echo "   ✓ Informação mais rica para o PPO"
echo "   ✓ Melhor para stops/targets dinâmicos"
echo ""
echo "📈 Métricas Esperadas:"
echo "   - RMSE Train: 0.001-0.003 (0.1-0.3%)"
echo "   - RMSE Test: 0.0015-0.004 (0.15-0.4%)"
echo "   - R² Score: > 0.05 (correlação com movimento real)"
echo ""
echo "========================================================================"
echo ""

# Para treinamento anterior se existir
pkill -f train_lightgbm.py 2>/dev/null

# Cria diretórios
mkdir -p models/hybrid_30m
mkdir -p logs/hybrid_30m

echo "🚀 Iniciando treinamento..."
echo ""

# Treina LightGBM com regressão
python3 -m src.training.train_lightgbm --config config_hybrid_30m.yaml

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
echo "💡 Próximo passo:"
echo "   Se RMSE test < 0.004 → Modelo BOM, treinar PPO"
echo "   Se RMSE test > 0.005 → Ajustar hiperparâmetros"
echo ""
