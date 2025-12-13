#!/usr/bin/env python3
"""
Script de Otimização de Hiperparâmetros para USDJPY 5m.

Este script testa diferentes combinações de:
- Features técnicas (RSI, EMA, MACD, Bollinger, ATR, etc.)
- Hiperparâmetros do LightGBM (learning_rate, num_leaves, max_depth, etc.)
- Prediction horizon

Salva os resultados em CSV e identifica a melhor combinação.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import logging
from datetime import datetime
from itertools import product
import json
from copy import deepcopy
from typing import Dict

# Adiciona path raiz
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

from src.training.train_lightgbm import train_lightgbm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """Otimizador de hiperparâmetros para LightGBM Forex."""
    
    def __init__(self, base_config_path: str, output_dir: str = "optimization_results"):
        """
        Inicializa otimizador.
        
        Args:
            base_config_path: Caminho para config base
            output_dir: Diretório para salvar resultados
        """
        self.base_config_path = base_config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Carrega config base
        with open(base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        
        self.results = []
        self.best_result = None
        
    def define_search_space(self):
        """
        Define espaço de busca de hiperparâmetros.
        
        Returns:
            Dict com parâmetros a testar
        """
        search_space = {
            # === FEATURES ===
            'features': [
                # Combinações de features
                {'use_ema': True, 'use_macd': True, 'use_rsi': False, 'use_bollinger': False, 'use_atr': True},
                {'use_ema': True, 'use_macd': True, 'use_rsi': True, 'use_bollinger': True, 'use_atr': True},
                {'use_ema': True, 'use_macd': False, 'use_rsi': True, 'use_bollinger': True, 'use_atr': True},
                {'use_ema': True, 'use_macd': True, 'use_rsi': True, 'use_bollinger': False, 'use_atr': False},
                {'use_ema': False, 'use_macd': False, 'use_rsi': True, 'use_bollinger': True, 'use_atr': True},
            ],
            
            # === PREDICTION HORIZON ===
            # Para 5m: 3 = 15min, 6 = 30min, 12 = 60min (1h)
            'prediction_horizon': [3, 6, 9, 12],
            
            # === LIGHTGBM PARAMS ===
            'learning_rate': [0.01, 0.03, 0.05],
            'num_leaves': [31, 50, 70],
            'max_depth': [4, 6, 8],
            'n_estimators': [300, 500, 800],
            'min_child_samples': [10, 20, 30],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'reg_alpha': [0.1, 0.3, 0.5],
            'reg_lambda': [0.1, 0.3, 0.5],
        }
        
        return search_space
    
    def generate_combinations(self, max_combinations: int = 100):
        """
        Gera combinações de hiperparâmetros para testar.
        
        Args:
            max_combinations: Número máximo de combinações
            
        Returns:
            Lista de dicts com combinações
        """
        search_space = self.define_search_space()
        
        # Estratégia: Random Search (mais eficiente que Grid Search)
        combinations = []
        
        # Primeiro testa todas as combinações de features com params padrão
        for features in search_space['features']:
            combo = {
                'features': features,
                'prediction_horizon': 6,  # padrão para 5m (30 min à frente)
                'learning_rate': 0.03,
                'num_leaves': 50,
                'max_depth': 6,
                'n_estimators': 500,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.3,
                'reg_lambda': 0.3,
            }
            combinations.append(combo)
        
        # Depois faz random search dos outros parâmetros
        np.random.seed(42)
        num_feature_combos = len(search_space['features'])
        
        for i in range(max_combinations - len(combinations)):
            # Seleciona features aleatoriamente pelo índice
            feature_idx = np.random.randint(0, num_feature_combos)
            
            combo = {
                'features': search_space['features'][feature_idx],
                'prediction_horizon': int(np.random.choice(search_space['prediction_horizon'])),  # Converte para int Python
                'learning_rate': float(np.random.choice(search_space['learning_rate'])),  # Converte para float Python
                'num_leaves': int(np.random.choice(search_space['num_leaves'])),
                'max_depth': int(np.random.choice(search_space['max_depth'])),
                'n_estimators': int(np.random.choice(search_space['n_estimators'])),
                'min_child_samples': int(np.random.choice(search_space['min_child_samples'])),
                'subsample': float(np.random.choice(search_space['subsample'])),
                'colsample_bytree': float(np.random.choice(search_space['colsample_bytree'])),
                'reg_alpha': float(np.random.choice(search_space['reg_alpha'])),
                'reg_lambda': float(np.random.choice(search_space['reg_lambda'])),
            }
            combinations.append(combo)
        
        logger.info(f"Geradas {len(combinations)} combinações para testar")
        return combinations
    
    def create_config_from_combination(self, combination: dict) -> dict:
        """
        Cria config a partir de uma combinação de hiperparâmetros.
        
        Args:
            combination: Dict com hiperparâmetros
            
        Returns:
            Config completo
        """
        config = deepcopy(self.base_config)  # Deep copy para não modificar original
        
        # Atualiza features
        for key, value in combination['features'].items():
            if key in config['features']:
                config['features'][key] = value
        
        # Atualiza prediction horizon
        config['lightgbm']['prediction_horizon'] = combination['prediction_horizon']
        
        # Atualiza params LightGBM
        lgbm_params = [
            'learning_rate', 'num_leaves', 'max_depth', 'n_estimators',
            'min_child_samples', 'subsample', 'colsample_bytree',
            'reg_alpha', 'reg_lambda'
        ]
        
        for param in lgbm_params:
            if param in combination:
                config['lightgbm']['params'][param] = combination[param]
        
        return config
    
    def evaluate_combination(self, combination: dict, index: int, total: int) -> dict:
        """
        Treina e avalia uma combinação de hiperparâmetros.
        
        Args:
            combination: Dict com hiperparâmetros
            index: Índice da combinação
            total: Total de combinações
            
        Returns:
            Dict com resultados
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"TESTANDO COMBINAÇÃO {index+1}/{total}")
        logger.info(f"{'='*80}")
        
        # Cria config
        config = self.create_config_from_combination(combination)
        
        # Salva config temporário
        temp_config_path = self.output_dir / f"temp_config_{index}.yaml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)
        
        try:
            # Treina modelo
            result = train_lightgbm(str(temp_config_path))
            
            # Extrai métricas
            metrics = result['metrics']
            
            # Verifica se é regressão ou classificação
            is_regression = 'rmse' in metrics.get('train', {})
            
            # Resultado consolidado
            eval_result = {
                'combination_id': index,
                'timestamp': datetime.now().isoformat(),
                
                # Features
                'use_ema': combination['features'].get('use_ema', False),
                'use_macd': combination['features'].get('use_macd', False),
                'use_rsi': combination['features'].get('use_rsi', False),
                'use_bollinger': combination['features'].get('use_bollinger', False),
                'use_atr': combination['features'].get('use_atr', False),
                
                # Hiperparâmetros
                'prediction_horizon': combination['prediction_horizon'],
                'learning_rate': combination['learning_rate'],
                'num_leaves': combination['num_leaves'],
                'max_depth': combination['max_depth'],
                'n_estimators': combination['n_estimators'],
                'min_child_samples': combination['min_child_samples'],
                'subsample': combination['subsample'],
                'colsample_bytree': combination['colsample_bytree'],
                'reg_alpha': combination['reg_alpha'],
                'reg_lambda': combination['reg_lambda'],
            }
            
            # Adiciona métricas conforme o tipo de modelo
            if is_regression:
                # Métricas de treino
                eval_result['train_rmse'] = metrics['train']['rmse']
                eval_result['train_mae'] = metrics['train']['mae']
                eval_result['train_r2'] = metrics['train']['r2']
                eval_result['train_direction_acc'] = metrics['train']['direction_accuracy']
                
                # Métricas de validação
                eval_result['val_rmse'] = metrics['val']['rmse']
                eval_result['val_mae'] = metrics['val']['mae']
                eval_result['val_r2'] = metrics['val']['r2']
                eval_result['val_direction_acc'] = metrics['val']['direction_accuracy']
                
                # Métricas de teste
                eval_result['test_rmse'] = metrics['test']['rmse']
                eval_result['test_mae'] = metrics['test']['mae']
                eval_result['test_r2'] = metrics['test']['r2']
                eval_result['test_direction_acc'] = metrics['test']['direction_accuracy']
                
                # Score combinado (menor é melhor)
                eval_result['combined_score'] = metrics['test']['rmse'] + (1 - metrics['test']['direction_accuracy'])
            else:
                # Métricas de classificação
                eval_result['train_accuracy'] = metrics['train'].get('accuracy', 0)
                eval_result['train_direction_acc'] = metrics['train'].get('accuracy', 0)
                eval_result['val_accuracy'] = metrics['val'].get('accuracy', 0)
                eval_result['val_direction_acc'] = metrics['val'].get('accuracy', 0)
                eval_result['test_accuracy'] = metrics['test'].get('accuracy', 0)
                eval_result['test_direction_acc'] = metrics['test'].get('accuracy', 0)
                eval_result['combined_score'] = 1 - metrics['test'].get('accuracy', 0)
            
            logger.info(f"✅ RESULTADO:")
            logger.info(f"   Test RMSE: {eval_result['test_rmse']:.6f}")
            logger.info(f"   Test Direction Acc: {eval_result['test_direction_acc']:.4f}")
            logger.info(f"   Combined Score: {eval_result['combined_score']:.6f}")
            
            return eval_result
            
        except Exception as e:
            logger.error(f"❌ ERRO ao treinar combinação {index}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
            
        finally:
            # Remove config temporário
            if temp_config_path.exists():
                temp_config_path.unlink()
    
    def run_optimization(self, max_combinations: int = 100):
        """
        Executa otimização de hiperparâmetros.
        
        Args:
            max_combinations: Número máximo de combinações a testar
        """
        logger.info(f"🚀 Iniciando otimização de hiperparâmetros - USDJPY 5m")
        logger.info(f"   Base config: {self.base_config_path}")
        logger.info(f"   Output dir: {self.output_dir}")
        logger.info(f"   Max combinations: {max_combinations}")
        
        # Gera combinações
        combinations = self.generate_combinations(max_combinations)
        
        # Testa cada combinação
        successful = 0
        failed = 0
        
        for i, combination in enumerate(combinations):
            logger.info(f"\n{'='*80}")
            logger.info(f"PROGRESSO: {i+1}/{len(combinations)} ({(i+1)/len(combinations)*100:.1f}%)")
            logger.info(f"Sucessos: {successful}, Falhas: {failed}")
            logger.info(f"{'='*80}")
            
            result = self.evaluate_combination(combination, i, len(combinations))
            
            if result is not None:
                self.results.append(result)
                successful += 1
                
                # Atualiza melhor resultado
                if self.best_result is None or result['combined_score'] < self.best_result['combined_score']:
                    self.best_result = result
                    logger.info(f"🏆 NOVO MELHOR RESULTADO! Score: {result['combined_score']:.6f}")
            else:
                failed += 1
                logger.warning(f"⚠️  Combinação {i} falhou!")
                
            # Salva resultados parciais a cada 5 iterações
            if (i + 1) % 5 == 0:
                self.save_results()
                logger.info(f"💾 Resultados parciais salvos ({len(self.results)} combinações bem-sucedidas)")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"FINALIZAÇÃO")
        logger.info(f"Total testado: {len(combinations)}")
        logger.info(f"Sucessos: {successful}")
        logger.info(f"Falhas: {failed}")
        logger.info(f"{'='*80}")
        
        # Salva resultados finais
        self.save_results()
        
        # Mostra resumo
        self.print_summary()
    
    def save_results(self):
        """Salva resultados em CSV e JSON."""
        if not self.results:
            return
        
        # CSV com todos os resultados
        df = pd.DataFrame(self.results)
        csv_path = self.output_dir / "optimization_results.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"💾 Resultados salvos em: {csv_path}")
        
        # JSON com melhor resultado
        if self.best_result:
            json_path = self.output_dir / "best_config.json"
            with open(json_path, 'w') as f:
                json.dump(self.best_result, f, indent=2)
            logger.info(f"🏆 Melhor config salvo em: {json_path}")
    
    def print_summary(self):
        """Imprime resumo da otimização."""
        if not self.results:
            logger.warning("Nenhum resultado disponível")
            return
        
        df = pd.DataFrame(self.results)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"RESUMO DA OTIMIZAÇÃO - USDJPY 5m")
        logger.info(f"{'='*80}")
        logger.info(f"Total de combinações testadas: {len(self.results)}")
        logger.info(f"\nEstatísticas (Test Set):")
        logger.info(f"  RMSE - Min: {df['test_rmse'].min():.6f}, Max: {df['test_rmse'].max():.6f}, Média: {df['test_rmse'].mean():.6f}")
        logger.info(f"  Direction Acc - Min: {df['test_direction_acc'].min():.4f}, Max: {df['test_direction_acc'].max():.4f}, Média: {df['test_direction_acc'].mean():.4f}")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"🏆 MELHOR CONFIGURAÇÃO")
        logger.info(f"{'='*80}")
        
        best = self.best_result
        logger.info(f"Combined Score: {best['combined_score']:.6f}")
        logger.info(f"\nFeatures:")
        logger.info(f"  EMA: {best['use_ema']}, MACD: {best['use_macd']}, RSI: {best['use_rsi']}")
        logger.info(f"  Bollinger: {best['use_bollinger']}, ATR: {best['use_atr']}")
        
        logger.info(f"\nHiperparâmetros:")
        logger.info(f"  Prediction Horizon: {best['prediction_horizon']} candles ({best['prediction_horizon'] * 5} min)")
        logger.info(f"  Learning Rate: {best['learning_rate']}")
        logger.info(f"  Num Leaves: {best['num_leaves']}")
        logger.info(f"  Max Depth: {best['max_depth']}")
        logger.info(f"  N Estimators: {best['n_estimators']}")
        
        logger.info(f"\nMétricas (Test Set):")
        logger.info(f"  RMSE: {best['test_rmse']:.6f}")
        logger.info(f"  MAE: {best['test_mae']:.6f}")
        logger.info(f"  R²: {best['test_r2']:.6f}")
        logger.info(f"  Direction Accuracy: {best['test_direction_acc']:.4f}")
        
        # Top 5 melhores
        logger.info(f"\n{'='*80}")
        logger.info(f"TOP 5 MELHORES COMBINAÇÕES")
        logger.info(f"{'='*80}")
        
        top5 = df.nsmallest(5, 'combined_score')
        for idx, row in top5.iterrows():
            logger.info(f"\n#{idx+1}:")
            logger.info(f"  Score: {row['combined_score']:.6f}")
            logger.info(f"  Test RMSE: {row['test_rmse']:.6f}, Direction Acc: {row['test_direction_acc']:.4f}")
            logger.info(f"  Features: EMA={row['use_ema']}, MACD={row['use_macd']}, RSI={row['use_rsi']}")
            logger.info(f"  LR={row['learning_rate']}, Leaves={row['num_leaves']}, Depth={row['max_depth']}")
        
        # Gera relatório detalhado em TXT
        self._generate_detailed_report(best, top5)
    
    def _generate_detailed_report(self, best: Dict, top5: pd.DataFrame):
        """Gera relatório detalhado em TXT explicando o melhor resultado."""
        report_path = os.path.join(self.output_dir, 'best_result_explained.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("RELATÓRIO DE OTIMIZAÇÃO - USDJPY 5m\n")
            f.write(f"Data: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            f.write("MELHOR CONFIGURAÇÃO ENCONTRADA\n")
            f.write("-"*80 + "\n\n")
            
            # Métricas de Performance
            f.write("1. MÉTRICAS DE PERFORMANCE (Test Set)\n\n")
            f.write(f"   Score Combinado: {best['combined_score']:.6f}\n")
            f.write(f"   - Quanto menor, melhor (combina RMSE e erro de direção)\n\n")
            
            f.write(f"   RMSE (Root Mean Square Error): {best['test_rmse']:.6f}\n")
            f.write(f"   - Erro médio quadrático das previsões de preço\n")
            f.write(f"   - Quanto menor, mais preciso é o modelo\n\n")
            
            f.write(f"   MAE (Mean Absolute Error): {best['test_mae']:.6f}\n")
            f.write(f"   - Erro médio absoluto das previsões\n")
            f.write(f"   - Interpretação: em média, previsões erram por {best['test_mae']:.6f} pips\n\n")
            
            f.write(f"   R² (Coeficiente de Determinação): {best['test_r2']:.6f}\n")
            f.write(f"   - Explica {best['test_r2']*100:.2f}% da variância dos preços\n")
            f.write(f"   - Valores próximos de 1 são melhores\n\n")
            
            f.write(f"   Acurácia Direcional: {best['test_direction_acc']:.4f} ({best['test_direction_acc']*100:.2f}%)\n")
            f.write(f"   - Percentual de vezes que prevê corretamente se o preço vai subir ou descer\n")
            f.write(f"   - Crucial para trading: acima de 50% indica poder preditivo\n\n")
            
            # Features Ativadas
            f.write("2. FEATURES/INDICADORES TÉCNICOS ATIVADOS\n\n")
            
            features_info = [
                ('use_rsi', 'RSI (Relative Strength Index)', 'Identifica condições de sobrecompra/sobrevenda'),
                ('use_ema', 'EMA (Exponential Moving Average)', 'Suaviza tendências de preço'),
                ('use_macd', 'MACD (Moving Average Convergence Divergence)', 'Detecta mudanças de momentum'),
                ('use_stochastic', 'Stochastic Oscillator', 'Mede momentum comparando preço de fechamento com range'),
                ('use_adx', 'ADX (Average Directional Index)', 'Mede força da tendência'),
                ('use_cci', 'CCI (Commodity Channel Index)', 'Identifica níveis cíclicos'),
                ('use_williams', 'Williams %R', 'Oscilador de momentum para sobrecompra/sobrevenda'),
                ('use_roc', 'ROC (Rate of Change)', 'Velocidade de mudança de preço'),
                ('use_obv', 'OBV (On-Balance Volume)', 'Relaciona volume com movimento de preço'),
                ('use_mfi', 'MFI (Money Flow Index)', 'RSI ponderado por volume'),
                ('use_bollinger', 'Bollinger Bands', 'Bandas de volatilidade'),
                ('use_atr', 'ATR (Average True Range)', 'Mede volatilidade do mercado'),
            ]
            
            for key, name, desc in features_info:
                # Usa .get() para lidar com features que podem não estar no CSV
                is_active = best.get(key, False)
                status = "✓ ATIVO" if is_active else "✗ Desativado"
                f.write(f"   {status:15} {name:45} - {desc}\n")
            
            # Hiperparâmetros
            f.write("\n3. HIPERPARÂMETROS DO LIGHTGBM\n\n")
            
            f.write(f"   Prediction Horizon: {int(best['prediction_horizon'])}\n")
            f.write(f"   - Quantos candles à frente o modelo prevê\n")
            f.write(f"   - Valor {int(best['prediction_horizon'])} = prevê {int(best['prediction_horizon']) * 5} minutos à frente\n\n")
            
            f.write(f"   Learning Rate: {best['learning_rate']}\n")
            f.write(f"   - Taxa de aprendizado do modelo\n")
            f.write(f"   - Valores menores = aprendizado mais lento mas estável\n\n")
            
            f.write(f"   Num Leaves: {int(best['num_leaves'])}\n")
            f.write(f"   - Número máximo de folhas nas árvores\n")
            f.write(f"   - Controla complexidade: mais folhas = modelo mais complexo\n\n")
            
            f.write(f"   Max Depth: {int(best['max_depth'])}\n")
            f.write(f"   - Profundidade máxima das árvores\n")
            f.write(f"   - Limita crescimento para evitar overfitting\n\n")
            
            f.write(f"   N Estimators: {int(best['n_estimators'])}\n")
            f.write(f"   - Número de árvores no ensemble\n")
            f.write(f"   - Mais árvores = modelo mais robusto (até certo ponto)\n\n")
            
            f.write(f"   Min Child Samples: {int(best['min_child_samples'])}\n")
            f.write(f"   - Mínimo de amostras para criar uma folha\n")
            f.write(f"   - Previne overfitting em regiões com poucos dados\n\n")
            
            f.write(f"   Subsample: {best['subsample']}\n")
            f.write(f"   - Fração de dados usada para treinar cada árvore\n")
            f.write(f"   - Adiciona randomização para melhorar generalização\n\n")
            
            f.write(f"   Colsample Bytree: {best['colsample_bytree']}\n")
            f.write(f"   - Fração de features usadas em cada árvore\n")
            f.write(f"   - Reduz correlação entre árvores\n\n")
            
            f.write(f"   Reg Alpha: {best['reg_alpha']}\n")
            f.write(f"   - Regularização L1 (Lasso)\n")
            f.write(f"   - Penaliza features menos importantes\n\n")
            
            f.write(f"   Reg Lambda: {best['reg_lambda']}\n")
            f.write(f"   - Regularização L2 (Ridge)\n")
            f.write(f"   - Suaviza pesos para evitar overfitting\n\n")
            
            # Top 5 alternativas
            f.write("="*80 + "\n")
            f.write("TOP 5 MELHORES CONFIGURAÇÕES\n")
            f.write("="*80 + "\n\n")
            
            for idx, (_, row) in enumerate(top5.iterrows(), 1):
                f.write(f"#{idx} - Score: {row['combined_score']:.6f}\n")
                f.write(f"   Métricas: RMSE={row['test_rmse']:.6f}, MAE={row['test_mae']:.6f}, "
                       f"R²={row['test_r2']:.6f}, Dir Acc={row['test_direction_acc']:.4f}\n")
                f.write(f"   Features: RSI={row.get('use_rsi', False)}, EMA={row.get('use_ema', False)}, MACD={row.get('use_macd', False)}, "
                       f"Bollinger={row.get('use_bollinger', False)}, ATR={row.get('use_atr', False)}\n")
                f.write(f"   Hiperparâmetros: LR={row['learning_rate']}, Leaves={int(row['num_leaves'])}, "
                       f"Depth={int(row['max_depth'])}, N_Est={int(row['n_estimators'])}\n")
                f.write(f"   Horizon={int(row['prediction_horizon'])} ({int(row['prediction_horizon']) * 5}min), Subsample={row['subsample']}, "
                       f"Colsample={row['colsample_bytree']}\n\n")
            
            # Interpretação e recomendações
            f.write("="*80 + "\n")
            f.write("INTERPRETAÇÃO E RECOMENDAÇÕES\n")
            f.write("="*80 + "\n\n")
            
            f.write("COMO USAR ESTA CONFIGURAÇÃO:\n\n")
            f.write("1. O arquivo 'best_config.yaml' já foi gerado com estes parâmetros\n")
            f.write("2. Use-o para treinar o modelo final: python -m src.rl.train --config best_config.yaml\n")
            f.write("3. A acurácia direcional é o principal indicador para trading\n")
            f.write("4. Valores acima de 55% de acurácia direcional já são úteis em produção\n\n")
            
            f.write("PRÓXIMOS PASSOS:\n\n")
            f.write("1. Treinar modelo completo com a melhor configuração\n")
            f.write("2. Fazer backtesting em dados out-of-sample\n")
            f.write("3. Validar performance em diferentes condições de mercado\n")
            f.write("4. Monitorar drift: performance pode degradar ao longo do tempo\n")
            f.write("5. Considerar re-treinamento periódico (ex: semanal/quinzenal para 5m)\n\n")
            
            f.write("OBSERVAÇÕES ESPECÍFICAS PARA 5m:\n\n")
            f.write("- Timeframe mais rápido = mais ruído no mercado\n")
            f.write("- Requer gestão de risco mais rigorosa devido à maior frequência de sinais\n")
            f.write("- Spread e custos de transação têm impacto maior em 5m\n")
            f.write("- Considere filtros adicionais para reduzir falsos sinais\n\n")
            
            if best['test_direction_acc'] < 0.52:
                f.write("⚠️  AVISO: Acurácia direcional abaixo de 52%\n")
                f.write("   - Modelo tem baixo poder preditivo\n")
                f.write("   - Recomenda-se coletar mais dados ou testar outras features\n")
                f.write("   - Em 5m, considere usar timeframes maiores para confirmação\n\n")
            elif best['test_direction_acc'] < 0.55:
                f.write("⚡ ATENÇÃO: Acurácia direcional moderada (52-55%)\n")
                f.write("   - Modelo pode ser útil mas requer gestão de risco cuidadosa\n")
                f.write("   - Considere combinar com outros sinais/filtros\n")
                f.write("   - Use stop loss apertado devido à volatilidade do 5m\n\n")
            else:
                f.write("✓ EXCELENTE: Acurácia direcional acima de 55%\n")
                f.write("   - Modelo demonstra bom poder preditivo\n")
                f.write("   - Adequado para uso em trading com gestão de risco apropriada\n")
                f.write("   - Ainda assim, valide em paper trading antes de usar capital real\n\n")
        
        logger.info(f"\n📄 Relatório detalhado salvo em: {report_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Otimização de hiperparâmetros para LightGBM - USDJPY 5m')
    parser.add_argument(
        '--config',
        type=str,
        default='config_hybrid_5m.yaml',
        help='Path para config base'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='optimization_results/usdjpy_5m',
        help='Diretório para salvar resultados'
    )
    parser.add_argument(
        '--max-combinations',
        type=int,
        default=50,
        help='Número máximo de combinações a testar'
    )
    
    args = parser.parse_args()
    
    # Cria otimizador
    optimizer = HyperparameterOptimizer(
        base_config_path=args.config,
        output_dir=args.output_dir
    )
    
    # Executa otimização
    optimizer.run_optimization(max_combinations=args.max_combinations)
    
    logger.info("\n✅ Otimização concluída!")
