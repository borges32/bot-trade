# ✅ Checklist de Verificação - Sistema Híbrido

Use este checklist para garantir que o sistema está configurado e funcionando corretamente.

## 📋 Checklist de Instalação

### 1. Ambiente Python
```bash
# Verificar versão do Python
python --version  # Deve ser 3.8+
```
- [ ] Python 3.8 ou superior instalado
- [ ] pip atualizado (`pip install --upgrade pip`)
- [ ] Ambiente virtual criado (recomendado)

### 2. Dependências
```bash
# Instalar dependências
pip install -r requirements.txt
```
- [ ] Todas as dependências instaladas sem erro
- [ ] LightGBM instalado corretamente
- [ ] Stable-Baselines3 instalado
- [ ] FastAPI instalado
- [ ] PyTorch instalado

Verificação:
```python
import lightgbm as lgb
import stable_baselines3
import fastapi
import torch
print("✅ Tudo OK!")
```

### 3. Estrutura de Diretórios
```bash
# Verificar estrutura
ls -la src/
ls -la data/
mkdir -p models/hybrid logs/hybrid
```
- [ ] Diretório `src/` existe com submódulos
- [ ] Diretório `data/` criado
- [ ] Diretório `models/hybrid/` criado
- [ ] Diretório `logs/hybrid/` criado

## 📊 Checklist de Dados

### 4. Dados de Treinamento
```bash
ls -lh data/
```
- [ ] Arquivo CSV de histórico colocado em `data/`
- [ ] CSV tem colunas: timestamp, open, high, low, close, volume
- [ ] Mínimo 6 meses de dados (recomendado)
- [ ] Dados ordenados por timestamp
- [ ] Sem valores faltantes críticos

Verificação:
```python
import pandas as pd
df = pd.read_csv('data/usdjpy_history_30m.csv')
print(f"Linhas: {len(df)}")
print(f"Colunas: {df.columns.tolist()}")
print(f"Período: {df['timestamp'].min()} até {df['timestamp'].max()}")
```
- [ ] Resultado acima mostra dados válidos

## ⚙️ Checklist de Configuração

### 5. Configuração
```bash
cat config_hybrid.yaml
```
- [ ] Arquivo `config_hybrid.yaml` existe
- [ ] `data.train_file` aponta para CSV correto
- [ ] Custos de transação ajustados (commission, slippage)
- [ ] Hiperparâmetros revisados

### 6. Teste do Sistema
```bash
python test_hybrid_system.py
```
- [ ] Teste roda sem erros
- [ ] Mostra "✅ Dados: OK"
- [ ] Mostra "✅ Features: OK"
- [ ] Mostra "✅ Ambiente: OK"

## 🎓 Checklist de Treinamento

### 7. Treinamento LightGBM
```bash
python -m src.training.train_lightgbm
```
- [ ] Script inicia sem erros
- [ ] Features são criadas corretamente
- [ ] Treinamento completa
- [ ] Modelo salvo em `models/hybrid/lightgbm_model.txt`
- [ ] Métricas salvas em `models/hybrid/lightgbm_metrics_*.yaml`
- [ ] Feature importance salva

Verificação de métricas:
```bash
cat models/hybrid/lightgbm_metrics_*.yaml
```
- [ ] AUC (se classifier) > 0.55
- [ ] Accuracy > 50%
- [ ] Nenhum overfitting grave (train vs test similar)

### 8. Treinamento PPO
```bash
python -m src.training.train_ppo
```
- [ ] Script inicia sem erros
- [ ] LightGBM model carregado
- [ ] Ambiente criado
- [ ] Treinamento progride
- [ ] Modelo salvo em `models/hybrid/ppo_model.zip`
- [ ] Métricas salvas

Verificação:
```bash
ls -lh models/hybrid/
```
- [ ] `lightgbm_model.txt` existe (>1KB)
- [ ] `ppo_model.zip` existe (>1MB)

## 🧪 Checklist de Testes

### 9. Teste de Inferência
```bash
python example_hybrid_usage.py
```
- [ ] Script carrega modelos sem erro
- [ ] Faz predições
- [ ] Mostra resultados
- [ ] Simulação roda até o fim

### 10. Teste da API
```bash
# Terminal 1
cd src/inference
python service.py &

# Terminal 2
curl http://localhost:8000/health
```
- [ ] API inicia sem erro
- [ ] Health check retorna 200 OK
- [ ] Documentação acessível em http://localhost:8000/docs

### 11. Cliente da API
```bash
python api_client_example.py --example 1
```
- [ ] Cliente conecta à API
- [ ] Recebe sinal sem erro
- [ ] Mostra resultado formatado

## 📈 Checklist de Qualidade

### 12. Métricas de Qualidade

**LightGBM:**
```bash
cat models/hybrid/lightgbm_model.importance.csv | head -20
```
- [ ] Features importantes fazem sentido (ex: RSI, EMAs)
- [ ] Importâncias bem distribuídas (não dominadas por 1-2 features)

**PPO:**
```bash
tensorboard --logdir logs/hybrid
```
- [ ] Gráfico de recompensa mostra tendência crescente
- [ ] Não há colapso de aprendizado
- [ ] Métricas estabilizam ao final

### 13. Sanity Checks

```python
from src.inference.predictor import TradingPredictor
import pandas as pd

predictor = TradingPredictor(
    lightgbm_path='models/hybrid/lightgbm_model',
    ppo_path='models/hybrid/ppo_model',
    feature_config=config['features'],
    env_config=config['ppo']['env']
)

# Teste com dados reais
df = pd.read_csv('data/usdjpy_history_30m.csv').tail(100)
result = predictor.predict(df)

print(f"Action: {result['action_name']}")
print(f"Confidence: {result['confidence']}")
```

- [ ] Predição retorna resultado válido
- [ ] Action é "comprar", "vender" ou "neutro"
- [ ] Confidence está entre 0 e 1
- [ ] Não há erros ou warnings

## 🚀 Checklist Pré-Produção

### 14. Backtesting
- [ ] Backtest com walk-forward validation feito
- [ ] Sharpe ratio > 1.0 no período de teste
- [ ] Max drawdown < 20%
- [ ] Win rate > 40%
- [ ] Resultados consistentes em diferentes períodos

### 15. Demo Trading
- [ ] Sistema integrado com conta demo
- [ ] Executando por pelo menos 1 semana
- [ ] Métricas reais próximas do backtest
- [ ] Sem erros de execução
- [ ] Logs sendo gravados corretamente

### 16. Monitoramento
- [ ] Sistema de alertas configurado
- [ ] Logs sendo salvos
- [ ] Métricas sendo rastreadas
- [ ] Backup automático de modelos
- [ ] Plano de retreinamento definido

## 📄 Checklist de Documentação

### 17. Documentação Lida
- [ ] SUMMARY.md lido
- [ ] QUICKSTART.md seguido
- [ ] README_HYBRID.md consultado
- [ ] ARCHITECTURE.md entendido
- [ ] COMMANDS.md como referência

### 18. Configuração Documentada
- [ ] Custos de transação documentados (de onde vieram)
- [ ] Hiperparâmetros escolhidos documentados
- [ ] Mudanças no código documentadas
- [ ] Processo de retreinamento documentado

## 🎯 Checklist Final

### Antes de Usar em Produção:
- [ ] ✅ Todos os testes passam
- [ ] ✅ Métricas de qualidade aceitáveis
- [ ] ✅ Backtest completo realizado
- [ ] ✅ Testado em demo por ≥ 1 mês
- [ ] ✅ Sistema de monitoramento ativo
- [ ] ✅ Plano de contingência definido
- [ ] ✅ Capital de risco definido (que pode perder)
- [ ] ✅ Stop loss global configurado
- [ ] ✅ Time de análise agendado (ex: semanal)

## 🚨 Red Flags - NÃO Use em Produção Se:

- [ ] ❌ AUC < 0.55 (LightGBM)
- [ ] ❌ Sharpe ratio < 0.5 (PPO)
- [ ] ❌ Max drawdown > 30%
- [ ] ❌ Win rate < 35%
- [ ] ❌ Não testou em demo
- [ ] ❌ Dados de treino < 3 meses
- [ ] ❌ Custos de transação não realistas
- [ ] ❌ Não tem backup/contingência
- [ ] ❌ API tem erros frequentes
- [ ] ❌ Não entende como o sistema funciona

## 📊 Scorecard Final

Some os pontos:
- Instalação (6 itens) × 5 pontos = ____ / 30
- Dados (5 itens) × 5 pontos = ____ / 25
- Treinamento (6 itens) × 10 pontos = ____ / 60
- Testes (4 itens) × 5 pontos = ____ / 20
- Qualidade (3 itens) × 10 pontos = ____ / 30
- Pré-Produção (4 itens) × 15 pontos = ____ / 60

**TOTAL: ____ / 225**

### Interpretação:
- **200-225**: ✅ Excelente! Sistema pronto
- **175-199**: ⚠️ Bom, mas revisar itens faltantes
- **150-174**: ⚠️ Precisa melhorias antes de produção
- **< 150**: ❌ Não está pronto para produção

## 📝 Notas e Observações

Use este espaço para anotar problemas encontrados, ajustes feitos, etc:

```
Data: _________

Observações:
_________________________________
_________________________________
_________________________________
_________________________________
_________________________________

Próximos passos:
1. _______________________________
2. _______________________________
3. _______________________________
```

---

**Última atualização**: Sistema implementado
**Versão**: 1.0.0
