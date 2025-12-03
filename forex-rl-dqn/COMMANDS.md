# 📝 Referência Rápida de Comandos

## 🚀 Setup Inicial

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Verificar instalação
python test_hybrid_system.py

# 3. Verificar estrutura
ls -la src/
ls -la models/
```

## 🎓 Treinamento

### Treinamento Completo (Recomendado)
```bash
# Treina LightGBM + PPO em sequência
./train_hybrid.sh

# Com configuração customizada
./train_hybrid.sh config_custom.yaml
```

### Treinamento Individual

**LightGBM**
```bash
# Com config padrão
python -m src.training.train_lightgbm

# Com config customizada
python -m src.training.train_lightgbm --config config_custom.yaml
```

**PPO**
```bash
# Com config padrão
python -m src.training.train_ppo

# Com config customizada
python -m src.training.train_ppo --config config_custom.yaml
```

## 📊 Monitoramento

### TensorBoard
```bash
# Visualizar logs de treinamento
tensorboard --logdir logs/hybrid

# Com porta customizada
tensorboard --logdir logs/hybrid --port 6006
```

### Verificar Modelos
```bash
# Listar modelos salvos
ls -lh models/hybrid/

# Ver métricas
cat models/hybrid/*_metrics_*.yaml

# Ver feature importance
cat models/hybrid/lightgbm_model.importance.csv
```

## 🌐 API

### Iniciar Servidor
```bash
# Método 1: Diretório correto
cd src/inference
python service.py

# Método 2: Módulo
python -m src.inference.service

# Com porta customizada
uvicorn src.inference.service:app --host 0.0.0.0 --port 8080
```

### Testar API

**Health Check**
```bash
curl http://localhost:8000/health
```

**Obter Sinal**
```bash
# Com arquivo JSON
curl -X POST http://localhost:8000/signal \
  -H "Content-Type: application/json" \
  -d @candles.json

# Inline (exemplo mínimo)
curl -X POST http://localhost:8000/signal \
  -H "Content-Type: application/json" \
  -d '{
    "candles": [...],
    "current_position": 0,
    "deterministic": true
  }'
```

**Estado Atual**
```bash
curl http://localhost:8000/state
```

**Resetar Estado**
```bash
curl -X POST http://localhost:8000/reset
```

**Documentação Interativa**
```bash
# Abrir no navegador
xdg-open http://localhost:8000/docs  # Linux
open http://localhost:8000/docs      # Mac
start http://localhost:8000/docs     # Windows
```

## 🧪 Testes e Exemplos

### Teste do Sistema
```bash
# Teste completo
python test_hybrid_system.py
```

### Exemplos de Uso

**Exemplo básico**
```bash
python example_hybrid_usage.py
```

**Cliente API**
```bash
# Exemplo 1: Predição única
python api_client_example.py --example 1

# Exemplo 2: Loop de trading
python api_client_example.py --example 2

# Exemplo 3: Integração com broker (pseudocódigo)
python api_client_example.py --example 3
```

## 🔧 Utilitários

### Feature Engineering (standalone)
```python
from src.common.features import FeatureEngineer
import pandas as pd

df = pd.read_csv('data/usdjpy_history_30m.csv')
fe = FeatureEngineer(config['features'])
df_features = fe.create_features(df)
```

### Inferência Direta
```python
from src.inference.predictor import TradingPredictor

predictor = TradingPredictor(
    lightgbm_path='models/hybrid/lightgbm_model',
    ppo_path='models/hybrid/ppo_model',
    feature_config=config['features'],
    env_config=config['ppo']['env']
)

result = predictor.predict(candles_df)
print(result['action_name'])
```

## 📦 Gestão de Modelos

### Backup de Modelos
```bash
# Criar backup
tar -czf models_backup_$(date +%Y%m%d).tar.gz models/

# Restaurar backup
tar -xzf models_backup_20240101.tar.gz
```

### Versionamento
```bash
# Copiar modelo atual para versão específica
cp models/hybrid/lightgbm_model.txt models/hybrid/lightgbm_model_v1.0.txt
cp models/hybrid/ppo_model.zip models/hybrid/ppo_model_v1.0.zip
```

## 🧹 Limpeza

### Limpar Logs
```bash
# Remover logs antigos
rm -rf logs/hybrid/*

# Manter apenas últimos N dias
find logs/hybrid -type f -mtime +30 -delete
```

### Limpar Checkpoints
```bash
# Remover checkpoints antigos
rm -rf models/hybrid/checkpoints/*

# Manter apenas últimos
ls -t models/hybrid/checkpoints/*.zip | tail -n +6 | xargs rm
```

## 🐛 Debug

### Logs Detalhados
```bash
# Treino com mais verbosidade
python -m src.training.train_lightgbm 2>&1 | tee train_lightgbm.log
python -m src.training.train_ppo 2>&1 | tee train_ppo.log
```

### Python Debugger
```python
# Adicionar breakpoint no código
import pdb; pdb.set_trace()

# Ou usar ipdb (mais amigável)
import ipdb; ipdb.set_trace()
```

### Verificar Configuração
```python
import yaml

with open('config_hybrid.yaml') as f:
    config = yaml.safe_load(f)
    
print(yaml.dump(config, default_flow_style=False))
```

## 📊 Análise de Resultados

### Analisar Métricas de Treinamento
```python
import yaml

with open('models/hybrid/lightgbm_metrics_*.yaml') as f:
    metrics = yaml.safe_load(f)
    
print("LightGBM Metrics:")
print(f"Train AUC: {metrics['train']['auc']:.4f}")
print(f"Val AUC: {metrics['val']['auc']:.4f}")
print(f"Test AUC: {metrics['test']['auc']:.4f}")
```

### Feature Importance
```python
import pandas as pd

importance = pd.read_csv('models/hybrid/lightgbm_model.importance.csv')
print("\nTop 20 Features:")
print(importance.head(20).to_string(index=False))
```

## 🔄 Workflows Comuns

### Workflow 1: Primeiro Uso
```bash
# 1. Setup
pip install -r requirements.txt
python test_hybrid_system.py

# 2. Treinar
./train_hybrid.sh

# 3. Testar
python example_hybrid_usage.py

# 4. API
cd src/inference && python service.py
```

### Workflow 2: Retreinamento
```bash
# 1. Backup modelos antigos
tar -czf models_backup_$(date +%Y%m%d).tar.gz models/

# 2. Retreinar
./train_hybrid.sh

# 3. Comparar métricas
diff <(cat models_backup/lightgbm_metrics.yaml) \
     <(cat models/hybrid/lightgbm_metrics*.yaml)

# 4. Testar novo modelo
python example_hybrid_usage.py

# 5. Se OK, manter. Se não, restaurar backup
```

### Workflow 3: Deploy em Produção
```bash
# 1. Verificar modelos
ls -lh models/hybrid/

# 2. Testar API localmente
cd src/inference && python service.py &
python api_client_example.py --example 1

# 3. Deploy (exemplo com Docker - opcional)
docker build -t forex-trading-api .
docker run -p 8000:8000 forex-trading-api

# 4. Monitorar
curl http://localhost:8000/health
tail -f logs/api.log
```

## 📚 Comandos Python Úteis

### Verificar Versões
```python
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import stable_baselines3 as sb3
import gymnasium as gym

print(f"Python: {sys.version}")
print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"LightGBM: {lgb.__version__}")
print(f"SB3: {sb3.__version__}")
print(f"Gymnasium: {gym.__version__}")
```

### Verificar GPU/CUDA
```python
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## 🎯 Atalhos e Aliases (Opcional)

Adicione ao seu `.bashrc` ou `.zshrc`:

```bash
# Atalhos para o projeto
alias forex-train='./train_hybrid.sh'
alias forex-test='python test_hybrid_system.py'
alias forex-api='cd src/inference && python service.py'
alias forex-example='python example_hybrid_usage.py'
alias forex-logs='tensorboard --logdir logs/hybrid'
alias forex-backup='tar -czf models_backup_$(date +%Y%m%d).tar.gz models/'
```

Depois execute:
```bash
source ~/.bashrc  # ou ~/.zshrc
forex-test
```

## 📞 Ajuda

```bash
# Ajuda dos scripts
python -m src.training.train_lightgbm --help
python -m src.training.train_ppo --help
python api_client_example.py --help

# Documentação
cat QUICKSTART.md
cat README_HYBRID.md
cat ARCHITECTURE.md
```

---

**Dica**: Salve este arquivo como referência rápida!
