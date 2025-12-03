# 📚 Índice de Documentação - Sistema Híbrido LightGBM + PPO

## 🎯 Começando

| Documento | Descrição | Quando Usar |
|-----------|-----------|-------------|
| **[SUMMARY.md](SUMMARY.md)** | Resumo executivo completo | Primeira leitura - visão geral |
| **[QUICKSTART.md](QUICKSTART.md)** | Guia rápido de início (5 passos) | Para começar imediatamente |
| **[COMMANDS.md](COMMANDS.md)** | Referência de comandos | Consulta rápida de comandos |

## 📖 Documentação Técnica

| Documento | Descrição | Quando Usar |
|-----------|-----------|-------------|
| **[README_HYBRID.md](README_HYBRID.md)** | Documentação completa e detalhada | Referência principal completa |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | Diagramas e arquitetura do sistema | Entender estrutura técnica |
| **[HYBRID_IMPLEMENTATION.md](HYBRID_IMPLEMENTATION.md)** | Detalhes de implementação | Desenvolvedores/customização |

## 🔧 Configuração

| Arquivo | Descrição | Quando Editar |
|---------|-----------|---------------|
| **[config_hybrid.yaml](config_hybrid.yaml)** | Configuração centralizada | Ajustar hiperparâmetros |
| **[requirements.txt](requirements.txt)** | Dependências Python | Instalar ou adicionar libs |

## 🎓 Exemplos e Tutoriais

| Script | Descrição | Como Usar |
|--------|-----------|-----------|
| **[test_hybrid_system.py](test_hybrid_system.py)** | Teste completo do sistema | `python test_hybrid_system.py` |
| **[example_hybrid_usage.py](example_hybrid_usage.py)** | Exemplo de uso em Python | `python example_hybrid_usage.py` |
| **[api_client_example.py](api_client_example.py)** | Cliente da API com exemplos | `python api_client_example.py --example 1` |

## 🚀 Scripts de Treinamento

| Script | Descrição | Como Usar |
|--------|-----------|-----------|
| **[train_hybrid.sh](train_hybrid.sh)** | Treinamento completo automatizado | `./train_hybrid.sh` |
| **[src/training/train_lightgbm.py](src/training/train_lightgbm.py)** | Treino individual LightGBM | `python -m src.training.train_lightgbm` |
| **[src/training/train_ppo.py](src/training/train_ppo.py)** | Treino individual PPO | `python -m src.training.train_ppo` |

## 🧩 Módulos do Sistema

### Modelos
| Módulo | Descrição | Responsabilidade |
|--------|-----------|------------------|
| **[src/models/lightgbm_model.py](src/models/lightgbm_model.py)** | Modelo LightGBM | Previsão de direção/retorno |
| **[src/models/ppo_agent.py](src/models/ppo_agent.py)** | Agente PPO | Decisão de trading |

### Ambientes
| Módulo | Descrição | Responsabilidade |
|--------|-----------|------------------|
| **[src/envs/forex_trading_env.py](src/envs/forex_trading_env.py)** | Ambiente Gym | Simulação de trading |

### Inferência
| Módulo | Descrição | Responsabilidade |
|--------|-----------|------------------|
| **[src/inference/predictor.py](src/inference/predictor.py)** | Motor de inferência | Predições em tempo real |
| **[src/inference/service.py](src/inference/service.py)** | API FastAPI | Endpoints HTTP |

### Utilitários
| Módulo | Descrição | Responsabilidade |
|--------|-----------|------------------|
| **[src/common/features.py](src/common/features.py)** | Feature engineering | Criação de indicadores |
| **[src/common/utils.py](src/common/utils.py)** | Utilitários gerais | Funções auxiliares |

## 📊 Fluxograma de Navegação

```
NOVO NO PROJETO?
    ├─> Leia: SUMMARY.md
    └─> Execute: python test_hybrid_system.py

QUER COMEÇAR A USAR?
    ├─> Leia: QUICKSTART.md
    └─> Execute: ./train_hybrid.sh

PRECISA DE COMANDOS?
    └─> Consulte: COMMANDS.md

QUER ENTENDER A ARQUITETURA?
    ├─> Leia: ARCHITECTURE.md
    └─> Leia: HYBRID_IMPLEMENTATION.md

QUER DOCUMENTAÇÃO COMPLETA?
    └─> Leia: README_HYBRID.md

PRECISA CUSTOMIZAR?
    ├─> Edite: config_hybrid.yaml
    └─> Leia: HYBRID_IMPLEMENTATION.md

QUER VER EXEMPLOS?
    ├─> Execute: python example_hybrid_usage.py
    └─> Execute: python api_client_example.py

PRONTO PARA PRODUÇÃO?
    ├─> Leia: README_HYBRID.md (seção "API")
    └─> Execute: cd src/inference && python service.py
```

## 🗂️ Estrutura de Diretórios

```
forex-rl-dqn/
│
├─ 📄 Documentação Principal
│   ├─ SUMMARY.md                    ⭐ COMECE AQUI
│   ├─ QUICKSTART.md                 ⚡ Início rápido
│   ├─ README_HYBRID.md              📖 Docs completas
│   ├─ ARCHITECTURE.md               🏗️ Arquitetura
│   ├─ HYBRID_IMPLEMENTATION.md      🔧 Implementação
│   ├─ COMMANDS.md                   💻 Comandos
│   └─ INDEX.md                      📚 Este arquivo
│
├─ ⚙️ Configuração
│   ├─ config_hybrid.yaml            🎛️ Config principal
│   └─ requirements.txt              📦 Dependências
│
├─ 🧪 Scripts de Teste/Exemplo
│   ├─ test_hybrid_system.py         ✅ Teste completo
│   ├─ example_hybrid_usage.py       📝 Exemplo de uso
│   ├─ api_client_example.py         🌐 Cliente API
│   └─ train_hybrid.sh               🚀 Treino automatizado
│
├─ 📂 src/
│   ├─ models/                       🤖 Modelos ML
│   │   ├─ lightgbm_model.py
│   │   └─ ppo_agent.py
│   │
│   ├─ envs/                         🎮 Ambientes
│   │   └─ forex_trading_env.py
│   │
│   ├─ training/                     🎓 Treinamento
│   │   ├─ train_lightgbm.py
│   │   └─ train_ppo.py
│   │
│   ├─ inference/                    🔮 Inferência
│   │   ├─ predictor.py
│   │   └─ service.py
│   │
│   └─ common/                       🛠️ Utilitários
│       ├─ features.py
│       └─ utils.py
│
├─ 📊 data/                          💾 Dados
├─ 🤖 models/                        💼 Modelos treinados
└─ 📈 logs/                          📋 Logs
```

## 🎓 Ordem de Leitura Recomendada

### Para Iniciantes
1. **SUMMARY.md** - Entenda o que é o sistema
2. **QUICKSTART.md** - Configure e rode
3. **example_hybrid_usage.py** - Veja funcionando
4. **README_HYBRID.md** - Aprofunde conforme necessário

### Para Desenvolvedores
1. **ARCHITECTURE.md** - Entenda a estrutura
2. **HYBRID_IMPLEMENTATION.md** - Detalhes técnicos
3. **Código fonte em src/** - Explore implementação
4. **COMMANDS.md** - Referência de comandos

### Para Integração
1. **QUICKSTART.md** - Setup básico
2. **api_client_example.py** - Veja exemplos de integração
3. **README_HYBRID.md** (seção API) - Documentação da API
4. **src/inference/service.py** - Código da API

## 🔍 Busca Rápida por Tópico

### Instalação e Setup
- **QUICKSTART.md** - Seção "Início Rápido"
- **requirements.txt** - Dependências

### Treinamento
- **train_hybrid.sh** - Script automatizado
- **src/training/** - Scripts individuais
- **config_hybrid.yaml** - Hiperparâmetros

### Uso da API
- **src/inference/service.py** - Implementação
- **api_client_example.py** - Exemplos
- **COMMANDS.md** - Comandos da API

### Configuração
- **config_hybrid.yaml** - Arquivo principal
- **HYBRID_IMPLEMENTATION.md** - Pontos de customização
- **README_HYBRID.md** - Seção "Ajustes"

### Modelos
- **src/models/lightgbm_model.py** - LightGBM
- **src/models/ppo_agent.py** - PPO
- **ARCHITECTURE.md** - Arquitetura

### Ambiente de Trading
- **src/envs/forex_trading_env.py** - Implementação
- **ARCHITECTURE.md** - Diagrama do ambiente

### Features
- **src/common/features.py** - Feature engineering
- **config_hybrid.yaml** - Configuração de features

### Troubleshooting
- **QUICKSTART.md** - "Problemas Comuns"
- **README_HYBRID.md** - "Troubleshooting"
- **COMMANDS.md** - "Debug"

## 📞 Onde Encontrar Ajuda?

| Problema | Onde Procurar |
|----------|---------------|
| Não sei por onde começar | SUMMARY.md, QUICKSTART.md |
| Erro de instalação | QUICKSTART.md, requirements.txt |
| Como treinar modelos | train_hybrid.sh, README_HYBRID.md |
| Como usar a API | api_client_example.py, COMMANDS.md |
| Performance ruim | HYBRID_IMPLEMENTATION.md, config_hybrid.yaml |
| Customizar sistema | HYBRID_IMPLEMENTATION.md, código fonte |
| Entender arquitetura | ARCHITECTURE.md |
| Comandos esquecidos | COMMANDS.md |

## ✅ Checklist de Uso

### Primeira Vez
- [ ] Ler SUMMARY.md
- [ ] Ler QUICKSTART.md
- [ ] Executar `pip install -r requirements.txt`
- [ ] Executar `python test_hybrid_system.py`
- [ ] Colocar dados em `data/`
- [ ] Executar `./train_hybrid.sh`
- [ ] Executar `python example_hybrid_usage.py`

### Antes de Produção
- [ ] Ler README_HYBRID.md completo
- [ ] Ajustar config_hybrid.yaml
- [ ] Fazer backtest extensivo
- [ ] Testar API localmente
- [ ] Validar em conta demo
- [ ] Configurar monitoramento

## 🆘 Suporte Adicional

Se após consultar toda a documentação ainda tiver dúvidas:

1. Revise os exemplos em `example_*.py`
2. Execute `python test_hybrid_system.py` para diagnóstico
3. Consulte logs em `logs/`
4. Revise configuração em `config_hybrid.yaml`
5. Abra uma issue no GitHub (se aplicável)

---

**Última atualização**: Sistema completo implementado e documentado
**Versão**: 1.0.0
**Status**: ✅ Pronto para uso
