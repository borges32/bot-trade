"""
Modelo LightGBM para previsão de direção/retorno de preço no Forex.

Este módulo implementa um preditor supervisionado que pode operar em dois modos:
1. Classificação: Prevê se o preço vai subir ou descer em N candles
2. Regressão: Prevê o retorno percentual em N candles

As saídas deste modelo são usadas como features para o agente PPO.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, Tuple, Optional, Union
from pathlib import Path
import joblib
import logging
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score
)

logger = logging.getLogger(__name__)


class LightGBMPredictor:
    """
    Preditor LightGBM para mercado Forex.
    
    Pode operar em modo classificação (prever direção) ou regressão (prever retorno).
    """
    
    def __init__(self, config: Dict):
        """
        Inicializa o preditor LightGBM.
        
        Args:
            config: Dicionário de configuração com parâmetros do LightGBM
        """
        self.config = config
        self.model_type = config.get('model_type', 'classifier')
        self.prediction_horizon = config.get('prediction_horizon', 5)
        self.classification_threshold = config.get('classification_threshold', 0.0)
        
        # Parâmetros LightGBM
        self.params = config.get('params', {})
        self.early_stopping_rounds = config.get('early_stopping_rounds', 50)
        
        # Modelo e scaler
        self.model: Optional[Union[lgb.LGBMClassifier, lgb.LGBMRegressor]] = None
        self.feature_names: Optional[list] = None
        self.feature_importance_: Optional[pd.DataFrame] = None
        
        logger.info(f"LightGBM Predictor initialized in {self.model_type} mode")
        logger.info(f"Prediction horizon: {self.prediction_horizon} candles")
    
    def create_target(self, df: pd.DataFrame) -> pd.Series:
        """
        Cria a variável target baseada no horizonte de previsão.
        
        Args:
            df: DataFrame com coluna 'close'
            
        Returns:
            Series com os targets
        """
        close = df['close'].values
        
        if self.model_type == 'classifier':
            # Target = 1 se preço sobe, 0 se desce
            future_close = df['close'].shift(-self.prediction_horizon)
            price_change = (future_close - df['close']) / df['close']
            
            # Aplica threshold se configurado
            target = (price_change > self.classification_threshold).astype(int)
            
            logger.info(f"Classification target created. Class distribution:")
            logger.info(f"  Class 0 (down): {(target == 0).sum()}")
            logger.info(f"  Class 1 (up): {(target == 1).sum()}")
            
        else:  # regressor
            # Target = retorno percentual
            future_close = df['close'].shift(-self.prediction_horizon)
            target = (future_close - df['close']) / df['close']
            
            logger.info(f"Regression target created. Stats:")
            logger.info(f"  Mean: {target.mean():.6f}")
            logger.info(f"  Std: {target.std():.6f}")
            logger.info(f"  Min: {target.min():.6f}")
            logger.info(f"  Max: {target.max():.6f}")
        
        return target
    
    def prepare_data(
        self, 
        df: pd.DataFrame, 
        feature_cols: list,
        create_target: bool = True
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Prepara dados para treinamento/predição.
        
        Args:
            df: DataFrame com features
            feature_cols: Lista de colunas a usar como features
            create_target: Se True, cria a variável target
            
        Returns:
            Tuple (X, y) onde X são as features e y o target (ou None)
        """
        # Remove linhas com NaN nas features
        X = df[feature_cols].copy()
        
        if create_target:
            y = self.create_target(df)
            # Remove linhas onde target é NaN (últimas N linhas)
            valid_idx = ~y.isna()
            X = X[valid_idx]
            y = y[valid_idx]
            
            # Remove linhas com NaN em X
            valid_idx = ~X.isna().any(axis=1)
            X = X[valid_idx]
            y = y[valid_idx]
            
            logger.info(f"Prepared {len(X)} samples with {len(feature_cols)} features")
            return X, y
        else:
            # Remove linhas com NaN
            X = X.dropna()
            logger.info(f"Prepared {len(X)} samples with {len(feature_cols)} features")
            return X, None
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict:
        """
        Treina o modelo LightGBM.
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
            X_val: Features de validação (opcional)
            y_val: Target de validação (opcional)
            
        Returns:
            Dicionário com métricas de treinamento
        """
        self.feature_names = list(X_train.columns)
        
        # Cria modelo apropriado
        if self.model_type == 'classifier':
            self.model = lgb.LGBMClassifier(**self.params)
        else:
            self.model = lgb.LGBMRegressor(**self.params)
        
        # Callbacks para early stopping
        callbacks = []
        eval_set = None
        
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))
            callbacks.append(lgb.log_evaluation(period=100))
        
        logger.info("Training LightGBM model...")
        
        # Treina modelo
        self.model.fit(
            X_train, 
            y_train,
            eval_set=eval_set,
            callbacks=callbacks if callbacks else None
        )
        
        # Calcula métricas
        metrics = {}
        
        # Métricas de treino
        train_pred = self.predict(X_train)
        metrics['train'] = self._compute_metrics(y_train, train_pred, X_train)
        
        # Métricas de validação
        if X_val is not None and y_val is not None:
            val_pred = self.predict(X_val)
            metrics['val'] = self._compute_metrics(y_val, val_pred, X_val)
        
        # Feature importance
        self.feature_importance_ = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Armazena métricas de teste para salvar posteriormente (se disponíveis)
        # Nota: test_metrics será adicionado externamente no train_lightgbm.py
        
        logger.info("Training completed!")
        logger.info(f"Train metrics: {metrics['train']}")
        if 'val' in metrics:
            logger.info(f"Val metrics: {metrics['val']}")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Faz predições.
        
        Args:
            X: Features
            
        Returns:
            Array com predições
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Verifica se é um Booster (modelo carregado) ou LGBMClassifier/Regressor (modelo treinado)
        if isinstance(self.model, lgb.Booster):
            # Modelo carregado - usa predict do Booster
            predictions = self.model.predict(X)
            
            # Para classificação, Booster retorna probabilidades brutas
            # Precisamos converter para probabilidades [0, 1]
            if self.model_type == 'classifier':
                # Sigmoid para converter log-odds em probabilidades
                predictions = 1 / (1 + np.exp(-predictions))
            
            return predictions
        else:
            # Modelo sklearn-like (recém treinado)
            if self.model_type == 'classifier':
                # Retorna probabilidade da classe 1 (preço sobe)
                return self.model.predict_proba(X)[:, 1]
            else:
                # Retorna retorno esperado
                return self.model.predict(X)
    
    def predict_single(self, features: Dict[str, float]) -> float:
        """
        Faz predição para um único sample.
        
        Args:
            features: Dicionário com features
            
        Returns:
            Predição (probabilidade ou retorno esperado)
        """
        # Converte para DataFrame
        X = pd.DataFrame([features])[self.feature_names]
        return self.predict(X)[0]
    
    def _compute_metrics(
        self, 
        y_true: pd.Series, 
        y_pred: np.ndarray,
        X: pd.DataFrame
    ) -> Dict:
        """
        Calcula métricas de avaliação.
        
        Args:
            y_true: Valores verdadeiros
            y_pred: Predições
            X: Features (para contar amostras)
            
        Returns:
            Dicionário com métricas
        """
        metrics = {'n_samples': len(y_true)}
        
        if self.model_type == 'classifier':
            # Métricas de classificação
            y_pred_binary = (y_pred > 0.5).astype(int)
            
            metrics['accuracy'] = accuracy_score(y_true, y_pred_binary)
            metrics['precision'] = precision_score(y_true, y_pred_binary, zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred_binary, zero_division=0)
            
            # AUC se temos ambas as classes
            if len(np.unique(y_true)) > 1:
                metrics['auc'] = roc_auc_score(y_true, y_pred)
            else:
                metrics['auc'] = 0.0
                
        else:
            # Métricas de regressão
            metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2'] = r2_score(y_true, y_pred)
            
            # Directional accuracy (acertou a direção?)
            direction_correct = np.sign(y_true) == np.sign(y_pred)
            metrics['direction_accuracy'] = direction_correct.mean()
        
        return metrics
    
    def save(self, path: Union[str, Path]):
        """
        Salva o modelo treinado.
        
        Args:
            path: Caminho para salvar o modelo
        """
        if self.model is None:
            raise ValueError("No model to save. Train first.")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Salva modelo LightGBM
        if self.model_type == 'classifier':
            model_path = path.with_suffix('.txt')
            self.model.booster_.save_model(str(model_path))
        else:
            model_path = path.with_suffix('.txt')
            self.model.booster_.save_model(str(model_path))
        
        # Salva metadados
        metadata = {
            'model_type': self.model_type,
            'prediction_horizon': self.prediction_horizon,
            'classification_threshold': self.classification_threshold,
            'feature_names': self.feature_names,
            'params': self.params,
            'test_metrics': getattr(self, 'test_metrics_', None)  # Salva métricas de teste se disponíveis
        }
        
        metadata_path = path.with_suffix('.metadata.pkl')
        joblib.dump(metadata, metadata_path)
        
        # Salva feature importance
        if self.feature_importance_ is not None:
            importance_path = path.with_suffix('.importance.csv')
            self.feature_importance_.to_csv(importance_path, index=False)
        
        logger.info(f"Model saved to {model_path}")
    
    def load(self, path: Union[str, Path]):
        """
        Carrega modelo treinado.
        
        Args:
            path: Caminho do modelo salvo
        """
        path = Path(path)
        
        # Carrega metadados
        metadata_path = path.with_suffix('.metadata.pkl')
        metadata = joblib.load(metadata_path)
        
        self.model_type = metadata['model_type']
        self.prediction_horizon = metadata['prediction_horizon']
        self.classification_threshold = metadata['classification_threshold']
        self.feature_names = metadata['feature_names']
        self.params = metadata['params']
        
        # Cria modelo apropriado
        if self.model_type == 'classifier':
            self.model = lgb.LGBMClassifier(**self.params)
        else:
            self.model = lgb.LGBMRegressor(**self.params)
        
        # Carrega booster
        model_path = path.with_suffix('.txt')
        self.model = lgb.Booster(model_file=str(model_path))
        
        # Carrega feature importance
        importance_path = path.with_suffix('.importance.csv')
        if importance_path.exists():
            self.feature_importance_ = pd.read_csv(importance_path)
        
        logger.info(f"Model loaded from {model_path}")
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Retorna as N features mais importantes.
        
        Args:
            top_n: Número de features a retornar
            
        Returns:
            DataFrame com features e importâncias
        """
        if self.feature_importance_ is None:
            raise ValueError("Model not trained yet.")
        
        return self.feature_importance_.head(top_n)
