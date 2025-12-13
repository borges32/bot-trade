# Mudanças: Remoção do PPO

## Data: 3 de dezembro de 2025

## Resumo
Sistema simplificado para usar **apenas LightGBM**. Todas as referências ao PPO foram removidas.

## Arquivos Removidos

### Código PPO
- `src/models/ppo_agent.py` - Agente PPO
- `src/training/train_ppo.py` - Script de treino PPO
- `src/envs/` - Diretório completo de ambientes Gym
- `restart_ppo_training.sh` - Script de restart

### Modelos Treinados
- `models/hybrid/ppo_*` - Todos os modelos PPO
- `models/hybrid/checkpoints/` - Checkpoints de treinamento

### Exemplos e Testes
- `example_hybrid_usage.py` - Exemplo híbrido
- `test_timeframe_configs.py` - Testes de config

## Arquivos Modificados

### Configurações
- `config_hybrid_15m.yaml`
  - Removida seção `ppo`
  - Atualizado título: "LightGBM" (não "LightGBM + PPO")
  - Removido `ppo_model_path` de inference
  
- `config_hybrid_30m.yaml`
  - Removida seção `ppo`
  - Atualizado título: "LightGBM" (não "LightGBM + PPO")
  - Removido `ppo_model_path` de inference

### Código-fonte
- `src/training/__init__.py`
  - Removido import de `train_ppo`
  - `__all__ = ['train_lightgbm']`

- `src/models/__init__.py`
  - Removido import de `PPOTradingAgent`
  - `__all__ = ['LightGBMPredictor']`

- `src/inference/predictor.py` - **REESCRITO**
  - Usa apenas LightGBM
  - Retorna sinais: BUY, SELL, NEUTRAL
  - Baseado em magnitude do retorno previsto
  - Sem tracking de estado/posição

- `src/inference/service.py`
  - Removidos endpoints: `/execute`, `/state`, `/reset`
  - Endpoint `/signal` simplificado
  - Carrega apenas LightGBM

- `example_precomputed_features.py`
  - Removida referência a treino PPO

### Dependências
- `requirements.txt`
  - Removido: `torch`, `torchvision`, `torchaudio`
  - Removido: `stable-baselines3`, `gymnasium`, `tensorboard`
  - Mantido apenas: numpy, pandas, scikit-learn, lightgbm, fastapi

## Arquivos Novos

- `example_lightgbm_usage.py` - Exemplo completo de uso do LightGBM
- `README_LIGHTGBM.md` - Documentação atualizada focada em LightGBM
- `CHANGES.md` - Este arquivo

## Sistema Atual

### Arquitetura
```
Dados OHLCV → Features (OptimizedFeatureEngineer) → LightGBM → Sinal (BUY/SELL/NEUTRAL)
```

### Fluxo de Trabalho
1. Treinar: `python -m src.training.train_lightgbm --config config_hybrid_30m.yaml`
2. Predição: `predictor.predict(df)` → `{'signal': 'BUY', 'predicted_return': 0.0015, ...}`
3. API: `POST /signal` com candles → retorna sinal

### Métricas (30m)
- RMSE: 0.0022
- Direction Accuracy: 53.72%
- MAE: 0.0015

## Próximos Passos Sugeridos

1. **Melhorar modelo LightGBM**
   - Hyperparameter tuning (Optuna)
   - Mais features (microestrutura, sentiment)
   - Ensemble de timeframes

2. **Backtesting**
   - Implementar estratégia baseada em sinais
   - Calcular Sharpe, Max DD, Win Rate

3. **Integração**
   - Conectar com cTrader/MT5
   - Auto-trading (com cautela!)

4. **Monitoramento**
   - Dashboard de performance
   - Alertas de degradação do modelo

## Notas Importantes

- ✅ LightGBM treinado e funcionando
- ✅ Código limpo e simplificado
- ✅ API REST funcional
- ✅ Documentação atualizada
- ⚠️  Sistema focado apenas em predição de retornos
- ⚠️  Não há mais gestão de posição/risco (era feito pelo PPO)
- ⚠️  Usuário deve implementar lógica de execução
