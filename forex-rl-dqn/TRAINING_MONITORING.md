# 📊 Guia de Monitoramento do Treinamento PPO

## 🎯 Objetivo das Mudanças

O treinamento estava **piorando** porque:
- ❌ Learning rate muito baixo (0.0003) → política estagnada
- ❌ Pouca exploração (ent_coef=0.01) → convergiu prematuramente
- ❌ Custos muito altos → difícil aprender a lucrar
- ❌ Penalidade de risco alta → agente muito conservador

## ✅ Mudanças Aplicadas

| Parâmetro | Antes | Depois | Motivo |
|-----------|-------|--------|--------|
| `learning_rate` | 0.0003 | **0.001** | Escapar de mínimo local |
| `ent_coef` | 0.01 | **0.03** | Mais exploração |
| `commission` | 0.0002 | **0.0001** | Facilitar lucro inicial |
| `slippage` | 0.0001 | **0.00005** | Reduzir custos |
| `max_drawdown_pct` | 0.20 | **0.25** | Episódios mais longos |
| `risk_penalty_lambda` | 0.1 | **0.05** | Menos conservador |

## 🚀 Como Reiniciar

### Opção 1: Parar atual e reiniciar do zero
```bash
# Para treinamento atual
pkill -f train_ppo.py

# Reinicia com novos parâmetros
./restart_ppo_training.sh
```

### Opção 2: Deixar terminar e começar novo
```bash
# Aguarde o treinamento atual terminar (500k timesteps)
# Depois execute:
./restart_ppo_training.sh
```

### Opção 3: Treinar direto (recomendado)
```bash
# Para treinamento atual
pkill -f train_ppo.py

# Treina PPO do zero
python3 -m src.training.train_ppo --config config_hybrid.yaml
```

## 📈 Métricas Esperadas (Com Novos Parâmetros)

### Primeiros 50k timesteps:
```
ep_rew_mean: -80 a -40  (melhor que -91)
clip_fraction: 0.05-0.15  (>0, significa que está aprendendo)
entropy_loss: -0.8 a -1.0  (explorando mais)
explained_variance: 0.3-0.5  (subindo)
value_loss: 0.5-0.7  (diminuindo)
```

### 100k-200k timesteps:
```
ep_rew_mean: -20 a +10  (começando a lucrar)
clip_fraction: 0.08-0.12
entropy_loss: -0.9 a -1.1
explained_variance: 0.5-0.7
value_loss: 0.3-0.5
```

### 300k-500k timesteps (final):
```
ep_rew_mean: +20 a +80  (lucrando consistentemente)
clip_fraction: 0.05-0.10
entropy_loss: -1.0 a -1.3
explained_variance: 0.7-0.9
value_loss: 0.2-0.4
```

## 🚨 Sinais de Alerta

### ❌ Se após 100k timesteps:
- `ep_rew_mean` ainda < -50 → Aumentar `learning_rate` para 0.002
- `clip_fraction` = 0 → Aumentar `learning_rate` ou `ent_coef`
- `explained_variance` < 0.3 → Problema na value function

### ❌ Se após 200k timesteps:
- `ep_rew_mean` < -20 → Considere:
  - Reduzir mais `commission` (0.00005)
  - Aumentar `ent_coef` (0.05)
  - Aumentar `learning_rate` (0.002)

### ❌ Se `ep_rew_mean` > 0 mas instável:
- Reduzir `learning_rate` (0.0005)
- Reduzir `ent_coef` (0.02)

## 📊 Monitoramento em Tempo Real

### Ver métricas a cada iteração:
```bash
# Terminal 1: Executa treinamento
./restart_ppo_training.sh

# Terminal 2: Monitora logs
tail -f logs/hybrid/training.log
```

### Ver últimas 20 linhas de métricas:
```bash
tail -20 logs/hybrid/training.log | grep -A 15 "rollout"
```

### Verificar progresso:
```bash
# Quantos timesteps já rodaram?
grep "total_timesteps" logs/hybrid/training.log | tail -1
```

## 🎯 Critérios de Sucesso

### ✅ Treinamento BOM:
- `ep_rew_mean` **crescendo** ao longo do tempo
- `explained_variance` **> 0.6** no final
- `value_loss` **diminuindo**
- `clip_fraction` **> 0** (entre 5-15%)
- Final: `ep_rew_mean` **> 0** (lucrando)

### 🟡 Treinamento MÉDIO:
- `ep_rew_mean` **estável** mas negativo (-20 a 0)
- `explained_variance` 0.4-0.6
- Precisa de mais timesteps ou ajuste fino

### ❌ Treinamento RUIM:
- `ep_rew_mean` **piorando** (como estava antes)
- `clip_fraction` = 0 (estagnado)
- `value_loss` **aumentando**
- Precisa **reajustar hiperparâmetros**

## 🔧 Ajustes Finos (Se Necessário)

### Se não melhorar após mudanças:

**Edite `config_hybrid.yaml`:**

```yaml
ppo:
  params:
    learning_rate: 0.002  # Mais agressivo
    ent_coef: 0.05        # Ainda mais exploração
    
  env:
    commission: 0.00005   # Custos mínimos
    reward_scaling: 2.0   # Recompensas maiores
```

**Depois:**
```bash
./restart_ppo_training.sh
```

## 📝 Registro de Testes

Anote aqui os resultados para comparar:

### Teste 1 (Original - FALHOU):
- Config: lr=0.0003, ent=0.01, comm=0.0002
- Resultado (149k): ep_rew=-91.3, clip=0, var=0.416
- Status: ❌ Piorando

### Teste 2 (Ajustado - EM PROGRESSO):
- Config: lr=0.001, ent=0.03, comm=0.0001
- Resultado (50k): ___ (preencher)
- Resultado (100k): ___ (preencher)
- Resultado (200k): ___ (preencher)
- Status final: ___ (preencher)

### Teste 3 (Se necessário):
- Config: ___ (preencher)
- Resultado: ___ (preencher)
- Status: ___ (preencher)

## 💡 Dicas

1. **Paciência**: PPO precisa de tempo para explorar
2. **Monitoramento**: Acompanhe a cada 50k timesteps
3. **Ajuste gradual**: Mude 1-2 parâmetros por vez
4. **Baseline**: Sempre compare com run anterior
5. **Salvamento**: Modelos são salvos a cada 50k timesteps

## 🎓 Entendendo os Parâmetros

**`learning_rate`**: Velocidade de aprendizado
- Baixo (0.0001-0.0003): Lento, estável, pode estagnar
- Médio (0.0005-0.001): Balanceado
- Alto (0.002-0.005): Rápido, mas instável

**`ent_coef`**: Exploração vs Exploitation
- Baixo (0.01): Exploita mais, pode convergir prematuramente
- Médio (0.03-0.05): Balanceado
- Alto (0.1+): Explora muito, pode nunca convergir

**`commission`**: Custo por trade
- Real (0.0002-0.0005): Realista para Forex
- Treino (0.0001): Facilita aprendizado inicial
- Depois ajustar para valor real

## ✅ Checklist Antes de Reiniciar

- [ ] Parou treinamento anterior: `pkill -f train_ppo.py`
- [ ] Conferiu mudanças em `config_hybrid.yaml`
- [ ] Terminal livre para rodar novo treinamento
- [ ] Pronto para monitorar por ~2-3 horas
- [ ] Tem espaço em disco (logs + checkpoints)

## 🚀 Comando de Início

```bash
# Tudo pronto? Execute:
./restart_ppo_training.sh

# Ou manualmente:
python3 -m src.training.train_ppo --config config_hybrid.yaml
```

**Boa sorte! 🍀**
