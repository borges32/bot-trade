"""
Agente PPO para trading usando stable-baselines3.

Este módulo encapsula a criação e treinamento do agente PPO
que aprende a operar no mercado Forex.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union, Callable
import logging

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, CheckpointCallback, CallbackList
)
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

logger = logging.getLogger(__name__)


class TradingMetricsCallback(BaseCallback):
    """
    Callback customizado para logar métricas de trading durante o treinamento.
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_equities = []
        self.episode_trades = []
        self.episode_win_rates = []
        
    def _on_step(self) -> bool:
        # Obtém info do ambiente
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                # Episódio terminou
                ep_reward = info['episode']['r']
                ep_length = info['episode']['l']
                
                self.episode_rewards.append(ep_reward)
                
                # Métricas de trading se disponíveis
                if 'equity' in info:
                    self.episode_equities.append(info['equity'])
                if 'total_trades' in info:
                    self.episode_trades.append(info['total_trades'])
                if 'win_rate' in info:
                    self.episode_win_rates.append(info['win_rate'])
                
                if self.verbose > 0:
                    logger.info(f"Episode finished - Reward: {ep_reward:.2f}, "
                              f"Length: {ep_length}, "
                              f"Equity: {info.get('equity', 0):.2f}")
        
        return True
    
    def get_metrics(self) -> Dict:
        """Retorna métricas acumuladas."""
        return {
            'mean_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'std_reward': np.std(self.episode_rewards) if self.episode_rewards else 0.0,
            'mean_equity': np.mean(self.episode_equities) if self.episode_equities else 0.0,
            'mean_trades': np.mean(self.episode_trades) if self.episode_trades else 0.0,
            'mean_win_rate': np.mean(self.episode_win_rates) if self.episode_win_rates else 0.0,
        }


class PPOTradingAgent:
    """
    Agente PPO para trading Forex.
    
    Encapsula o modelo PPO da stable-baselines3 com configurações
    específicas para trading.
    """
    
    def __init__(self, env, config: Dict):
        """
        Inicializa o agente PPO.
        
        Args:
            env: Ambiente de trading
            config: Configuração do PPO
        """
        self.env = env
        self.config = config
        
        # Parâmetros PPO
        ppo_params = config.get('params', {})
        
        # Arquitetura da rede
        policy_config = config.get('policy_network', {})
        net_arch = policy_config.get('net_arch', [256, 256, 128])
        activation = policy_config.get('activation', 'tanh')
        
        # Cria política
        policy_kwargs = {
            'net_arch': net_arch,
            'activation_fn': self._get_activation_fn(activation)
        }
        
        # Device
        device = config.get('device', 'auto')
        
        # Cria modelo PPO
        self.model = PPO(
            policy='MlpPolicy',
            env=env,
            learning_rate=ppo_params.get('learning_rate', 3e-4),
            n_steps=ppo_params.get('n_steps', 2048),
            batch_size=ppo_params.get('batch_size', 64),
            n_epochs=ppo_params.get('n_epochs', 10),
            gamma=ppo_params.get('gamma', 0.99),
            gae_lambda=ppo_params.get('gae_lambda', 0.95),
            clip_range=ppo_params.get('clip_range', 0.2),
            clip_range_vf=ppo_params.get('clip_range_vf', None),
            ent_coef=ppo_params.get('ent_coef', 0.01),
            vf_coef=ppo_params.get('vf_coef', 0.5),
            max_grad_norm=ppo_params.get('max_grad_norm', 0.5),
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=device,
            seed=config.get('seed', None)
        )
        
        logger.info("PPO Agent initialized")
        logger.info(f"Policy network architecture: {net_arch}")
        logger.info(f"Learning rate: {ppo_params.get('learning_rate', 3e-4)}")
        logger.info(f"Device: {device}")
    
    def _get_activation_fn(self, activation: str):
        """Retorna função de ativação."""
        import torch.nn as nn
        
        activations = {
            'tanh': nn.Tanh,
            'relu': nn.ReLU,
            'elu': nn.ELU,
            'leaky_relu': nn.LeakyReLU
        }
        
        return activations.get(activation.lower(), nn.Tanh)
    
    def train(
        self,
        total_timesteps: int,
        eval_env=None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 10,
        save_freq: int = 50000,
        save_path: Optional[Union[str, Path]] = None,
        callbacks: Optional[list] = None
    ) -> Dict:
        """
        Treina o agente PPO.
        
        Args:
            total_timesteps: Número total de timesteps
            eval_env: Ambiente de avaliação (opcional)
            eval_freq: Frequência de avaliação
            n_eval_episodes: Número de episódios de avaliação
            save_freq: Frequência de salvamento
            save_path: Caminho para salvar modelos
            callbacks: Callbacks adicionais
            
        Returns:
            Dicionário com métricas de treinamento
        """
        # Callbacks padrão
        callback_list = callbacks or []
        
        # Callback de métricas
        metrics_callback = TradingMetricsCallback(verbose=1)
        callback_list.append(metrics_callback)
        
        # Callback de checkpoint
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            checkpoint_callback = CheckpointCallback(
                save_freq=save_freq,
                save_path=str(save_path / 'checkpoints'),
                name_prefix='ppo_model'
            )
            callback_list.append(checkpoint_callback)
        
        # Callback de avaliação
        if eval_env is not None:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=str(save_path / 'best_model') if save_path else None,
                log_path=str(save_path / 'eval_logs') if save_path else None,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                render=False
            )
            callback_list.append(eval_callback)
        
        # Combina callbacks
        combined_callback = CallbackList(callback_list)
        
        logger.info(f"Starting PPO training for {total_timesteps} timesteps...")
        
        # Treina
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=combined_callback,
            progress_bar=True
        )
        
        logger.info("Training completed!")
        
        # Retorna métricas
        metrics = metrics_callback.get_metrics()
        return metrics
    
    def predict(
        self, 
        observation: np.ndarray, 
        deterministic: bool = True
    ) -> tuple:
        """
        Faz predição de ação.
        
        Args:
            observation: Observação do ambiente
            deterministic: Se True, usa política determinística
            
        Returns:
            Tuple (ação, estado)
        """
        action, state = self.model.predict(observation, deterministic=deterministic)
        return action, state
    
    def save(self, path: Union[str, Path]):
        """
        Salva o modelo treinado.
        
        Args:
            path: Caminho para salvar
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save(str(path))
        logger.info(f"Model saved to {path}")
    
    def load(self, path: Union[str, Path], env=None):
        """
        Carrega modelo treinado.
        
        Args:
            path: Caminho do modelo
            env: Novo ambiente (opcional)
        """
        path = Path(path)
        
        self.model = PPO.load(str(path), env=env)
        logger.info(f"Model loaded from {path}")
    
    def evaluate(
        self, 
        env, 
        n_episodes: int = 10,
        deterministic: bool = True
    ) -> Dict:
        """
        Avalia o agente em um ambiente.
        
        Args:
            env: Ambiente de avaliação
            n_episodes: Número de episódios
            deterministic: Se True, usa política determinística
            
        Returns:
            Dicionário com métricas de avaliação
        """
        episode_rewards = []
        episode_equities = []
        episode_trades = []
        episode_win_rates = []
        episode_returns = []
        episode_drawdowns = []
        
        for ep in range(n_episodes):
            obs, info = env.reset()
            done = False
            truncated = False
            episode_reward = 0
            
            while not (done or truncated):
                action, _ = self.predict(obs, deterministic=deterministic)
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
            
            # Coleta métricas
            episode_rewards.append(episode_reward)
            
            if 'equity' in info:
                episode_equities.append(info['equity'])
            if 'total_trades' in info:
                episode_trades.append(info['total_trades'])
            if 'win_rate' in info:
                episode_win_rates.append(info['win_rate'])
            if 'total_return' in info:
                episode_returns.append(info['total_return'])
            if 'max_drawdown' in info:
                episode_drawdowns.append(info['max_drawdown'])
        
        # Calcula estatísticas
        metrics = {
            'n_episodes': n_episodes,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
        }
        
        if episode_equities:
            metrics['mean_equity'] = np.mean(episode_equities)
            metrics['std_equity'] = np.std(episode_equities)
        
        if episode_trades:
            metrics['mean_trades'] = np.mean(episode_trades)
        
        if episode_win_rates:
            metrics['mean_win_rate'] = np.mean(episode_win_rates)
        
        if episode_returns:
            metrics['mean_return'] = np.mean(episode_returns)
            metrics['std_return'] = np.std(episode_returns)
        
        if episode_drawdowns:
            metrics['mean_drawdown'] = np.mean(episode_drawdowns)
            metrics['max_drawdown'] = np.max(episode_drawdowns)
        
        return metrics


def make_trading_env(
    df,
    feature_columns: list,
    lightgbm_predictor=None,
    config: Optional[Dict] = None,
    monitor_path: Optional[Union[str, Path]] = None
):
    """
    Cria ambiente de trading com wrappers.
    
    Args:
        df: DataFrame com dados
        feature_columns: Colunas de features
        lightgbm_predictor: Preditor LightGBM
        config: Configuração do ambiente
        monitor_path: Caminho para logs do Monitor
        
    Returns:
        Ambiente wrapped
    """
    from src.envs import ForexTradingEnv
    
    env = ForexTradingEnv(
        df=df,
        feature_columns=feature_columns,
        lightgbm_predictor=lightgbm_predictor,
        config=config
    )
    
    # Adiciona Monitor para logging
    if monitor_path:
        monitor_path = Path(monitor_path)
        monitor_path.mkdir(parents=True, exist_ok=True)
        env = Monitor(env, str(monitor_path))
    
    return env
