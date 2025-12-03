#!/bin/bash

echo "============================================"
echo "  REINICIANDO TREINAMENTO PPO"
echo "  Com Hiperparâmetros Otimizados"
echo "============================================"
echo ""
echo "Mudanças aplicadas:"
echo "  ✓ Learning rate: 0.0003 → 0.001 (3x maior)"
echo "  ✓ Entropy coef: 0.01 → 0.03 (3x mais exploração)"
echo "  ✓ Commission: 0.0002 → 0.0001 (50% menor)"
echo "  ✓ Slippage: 0.0001 → 0.00005 (50% menor)"
echo "  ✓ Max drawdown: 20% → 25%"
echo "  ✓ Risk penalty: 0.1 → 0.05 (50% menor)"
echo ""
echo "Objetivo: Permitir mais exploração e aprendizado inicial"
echo ""

# Para treinamento anterior se estiver rodando
pkill -f "train_ppo.py" 2>/dev/null

echo "Iniciando treinamento PPO com novos parâmetros..."
echo ""

python3 -m src.training.train_ppo --config config_hybrid.yaml
