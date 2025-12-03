"""
Ambiente Gym para Trading Forex com integração LightGBM + PPO.

Este ambiente simula operações de trading no mercado Forex, permitindo que
um agente PPO aprenda a operar usando sinais do LightGBM e features de mercado.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class ForexTradingEnv(gym.Env):
    """
    Ambiente de trading Forex compatível com Gym/Gymnasium.
    
    Observation Space:
        - Sinal LightGBM (p_up ou retorno esperado)
        - Features técnicas normalizadas (RSI, EMAs, volatilidade, etc.)
        - Estado da posição atual (-1, 0, 1)
        - PnL não realizado
        - Equity normalizado
        - Drawdown atual
        
    Action Space (discreto):
        0 = Neutro/Flat (sem posição)
        1 = Comprar (long)
        2 = Vender (short)
        
    Reward:
        PnL ajustado por risco e custos de transação
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: list,
        lightgbm_predictor=None,
        config: Optional[Dict] = None
    ):
        """
        Inicializa o ambiente de trading.
        
        Args:
            df: DataFrame com dados históricos (OHLCV + features)
            feature_columns: Lista de colunas de features técnicas
            lightgbm_predictor: Modelo LightGBM treinado (opcional)
            config: Configuração do ambiente
        """
        super().__init__()
        
        self.df = df.reset_index(drop=True)
        self.feature_columns = feature_columns
        self.lightgbm_predictor = lightgbm_predictor
        
        # Configuração
        config = config or {}
        self.initial_balance = config.get('initial_balance', 10000.0)
        self.leverage = config.get('leverage', 1.0)
        self.commission = config.get('commission', 0.0002)  # 0.02%
        self.slippage = config.get('slippage', 0.0001)  # 0.01%
        self.max_position_size = config.get('max_position_size', 1.0)
        self.stop_loss_pct = config.get('stop_loss_pct', 0.02)  # 2%
        self.take_profit_pct = config.get('take_profit_pct', 0.04)  # 4%
        self.max_drawdown_pct = config.get('max_drawdown_pct', 0.20)  # 20%
        self.risk_penalty_lambda = config.get('risk_penalty_lambda', 0.1)
        self.reward_scaling = config.get('reward_scaling', 1.0)
        
        # Espaço de ações: 0=neutro, 1=comprar, 2=vender
        self.action_space = spaces.Discrete(3)
        
        # Espaço de observação
        # Tamanho = 1 (sinal LightGBM) + len(features) + 5 (estado da conta)
        n_features = len(feature_columns)
        obs_size = 1 + n_features + 5
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )
        
        # Estado do episódio
        self.current_step = 0
        self.max_steps = len(df) - 1
        
        # Estado da conta
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.position = 0  # -1=short, 0=flat, 1=long
        self.position_size = 0.0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        
        # Métricas
        self.max_equity = self.initial_balance
        self.max_drawdown = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # Histórico para render
        self.equity_history = []
        self.action_history = []
        
        logger.info(f"ForexTradingEnv initialized with {len(df)} steps")
        logger.info(f"Initial balance: ${self.initial_balance:.2f}")
        logger.info(f"Commission: {self.commission*100:.3f}%")
        
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reseta o ambiente para o estado inicial.
        
        Args:
            seed: Seed para reprodutibilidade
            
        Returns:
            Tuple (observation, info)
        """
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.position = 0
        self.position_size = 0.0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        
        self.max_equity = self.initial_balance
        self.max_drawdown = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        self.equity_history = [self.equity]
        self.action_history = []
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Executa uma ação no ambiente.
        
        Args:
            action: Ação a executar (0=neutro, 1=comprar, 2=vender)
            
        Returns:
            Tuple (observation, reward, terminated, truncated, info)
        """
        # Preço atual
        current_price = self.df.loc[self.current_step, 'close']
        
        # Calcula PnL não realizado antes da ação
        old_unrealized_pnl = self._calculate_unrealized_pnl(current_price)
        
        # Executa ação
        trade_cost = self._execute_action(action, current_price)
        
        # Avança o step
        self.current_step += 1
        
        # Verifica se terminou
        terminated = False
        truncated = False
        
        if self.current_step >= self.max_steps:
            truncated = True
            # Fecha posição ao final
            if self.position != 0:
                self._close_position(current_price)
        
        # Atualiza equity
        new_price = self.df.loc[self.current_step, 'close']
        self.unrealized_pnl = self._calculate_unrealized_pnl(new_price)
        self.equity = self.balance + self.unrealized_pnl
        
        # Verifica stop loss / take profit
        if self.position != 0:
            pnl_pct = self.unrealized_pnl / self.balance if self.balance > 0 else 0
            
            if pnl_pct <= -self.stop_loss_pct:
                # Stop loss atingido
                logger.debug(f"Stop loss triggered at step {self.current_step}")
                self._close_position(new_price)
                
            elif pnl_pct >= self.take_profit_pct:
                # Take profit atingido
                logger.debug(f"Take profit triggered at step {self.current_step}")
                self._close_position(new_price)
        
        # Atualiza métricas
        if self.equity > self.max_equity:
            self.max_equity = self.equity
        
        current_drawdown = (self.max_equity - self.equity) / self.max_equity if self.max_equity > 0 else 0
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Verifica drawdown máximo
        if current_drawdown >= self.max_drawdown_pct:
            logger.warning(f"Max drawdown reached: {current_drawdown*100:.2f}%")
            terminated = True
            # Fecha posição
            if self.position != 0:
                self._close_position(new_price)
        
        # Calcula recompensa
        reward = self._calculate_reward(old_unrealized_pnl, trade_cost, current_drawdown)
        
        # Histórico
        self.equity_history.append(self.equity)
        self.action_history.append(action)
        
        # Observação e info
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _execute_action(self, action: int, price: float) -> float:
        """
        Executa a ação de trading.
        
        Args:
            action: Ação (0=neutro, 1=comprar, 2=vender)
            price: Preço atual
            
        Returns:
            Custo da transação
        """
        trade_cost = 0.0
        
        # Mapeia ação para posição desejada
        desired_position = action - 1  # 0->-1, 1->0, 2->1
        # Corrigindo: 0=neutro(0), 1=long(1), 2=short(-1)
        if action == 0:
            desired_position = 0
        elif action == 1:
            desired_position = 1
        else:  # action == 2
            desired_position = -1
        
        # Se posição não muda, não faz nada
        if desired_position == self.position:
            return 0.0
        
        # Se tinha posição, fecha
        if self.position != 0:
            trade_cost += self._close_position(price)
        
        # Se nova posição não é neutra, abre
        if desired_position != 0:
            trade_cost += self._open_position(desired_position, price)
        
        return trade_cost
    
    def _open_position(self, direction: int, price: float) -> float:
        """
        Abre uma posição.
        
        Args:
            direction: 1 para long, -1 para short
            price: Preço de entrada
            
        Returns:
            Custo da transação
        """
        # Considera slippage
        effective_price = price * (1 + self.slippage * direction)
        
        # Tamanho da posição (fração do capital)
        self.position_size = self.equity * self.max_position_size * self.leverage
        
        # Custo de transação
        trade_cost = self.position_size * self.commission
        
        # Atualiza estado
        self.position = direction
        self.entry_price = effective_price
        self.balance -= trade_cost
        self.total_trades += 1
        
        logger.debug(f"Opened {['SHORT', 'FLAT', 'LONG'][direction+1]} position at {effective_price:.5f}")
        
        return trade_cost
    
    def _close_position(self, price: float) -> float:
        """
        Fecha a posição atual.
        
        Args:
            price: Preço de saída
            
        Returns:
            Custo da transação
        """
        if self.position == 0:
            return 0.0
        
        # Considera slippage (na direção oposta)
        effective_price = price * (1 - self.slippage * self.position)
        
        # Calcula PnL
        if self.position == 1:  # Long
            pnl = self.position_size * (effective_price - self.entry_price) / self.entry_price
        else:  # Short
            pnl = self.position_size * (self.entry_price - effective_price) / self.entry_price
        
        # Custo de transação
        trade_cost = self.position_size * self.commission
        
        # PnL líquido
        net_pnl = pnl - trade_cost
        
        # Atualiza conta
        self.balance += net_pnl
        self.realized_pnl += net_pnl
        
        # Atualiza estatísticas
        if net_pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        logger.debug(f"Closed position at {effective_price:.5f}, PnL: ${net_pnl:.2f}")
        
        # Reseta posição
        self.position = 0
        self.position_size = 0.0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        
        return trade_cost
    
    def _calculate_unrealized_pnl(self, current_price: float) -> float:
        """
        Calcula o PnL não realizado da posição atual.
        
        Args:
            current_price: Preço atual
            
        Returns:
            PnL não realizado
        """
        if self.position == 0:
            return 0.0
        
        if self.position == 1:  # Long
            return self.position_size * (current_price - self.entry_price) / self.entry_price
        else:  # Short
            return self.position_size * (self.entry_price - current_price) / self.entry_price
    
    def _calculate_reward(
        self, 
        old_unrealized_pnl: float, 
        trade_cost: float,
        current_drawdown: float
    ) -> float:
        """
        Calcula a recompensa do step.
        
        Args:
            old_unrealized_pnl: PnL não realizado anterior
            trade_cost: Custo da transação
            current_drawdown: Drawdown atual
            
        Returns:
            Recompensa
        """
        # Variação do PnL (realizado + não realizado)
        pnl_delta = (self.realized_pnl + self.unrealized_pnl) - old_unrealized_pnl
        
        # Penaliza custos de transação
        reward = pnl_delta - trade_cost
        
        # Penaliza risco (drawdown)
        risk_penalty = self.risk_penalty_lambda * current_drawdown * self.initial_balance
        reward -= risk_penalty
        
        # Escala recompensa
        reward = reward * self.reward_scaling / self.initial_balance
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """
        Constrói a observação do estado atual.
        
        Returns:
            Array com observação
        """
        row = self.df.iloc[self.current_step]
        
        # Sinal LightGBM
        if self.lightgbm_predictor is not None:
            features = {col: row[col] for col in self.feature_columns}
            lgbm_signal = self.lightgbm_predictor.predict_single(features)
        else:
            lgbm_signal = 0.5  # Neutro se não há preditor
        
        # Features técnicas (normalizadas)
        technical_features = row[self.feature_columns].values.astype(np.float32)
        
        # Estado da conta (normalizado)
        account_state = np.array([
            self.position,  # -1, 0, 1
            self.unrealized_pnl / self.initial_balance,
            self.equity / self.initial_balance,
            self.max_drawdown,
            (self.equity - self.initial_balance) / self.initial_balance  # retorno total
        ], dtype=np.float32)
        
        # Concatena tudo
        obs = np.concatenate([
            [lgbm_signal],
            technical_features,
            account_state
        ])
        
        # Substitui NaN/Inf por 0
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return obs
    
    def _get_info(self) -> Dict:
        """
        Retorna informações adicionais sobre o estado.
        
        Returns:
            Dicionário com informações
        """
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0
        
        return {
            'step': self.current_step,
            'equity': self.equity,
            'balance': self.balance,
            'position': self.position,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_return': (self.equity - self.initial_balance) / self.initial_balance,
            'max_drawdown': self.max_drawdown,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate
        }
    
    def render(self, mode='human'):
        """
        Renderiza o estado atual.
        """
        if mode == 'human':
            info = self._get_info()
            print(f"\n{'='*60}")
            print(f"Step: {info['step']}/{self.max_steps}")
            print(f"Equity: ${info['equity']:.2f} (Return: {info['total_return']*100:.2f}%)")
            print(f"Position: {['SHORT', 'FLAT', 'LONG'][info['position']+1]}")
            print(f"Unrealized PnL: ${info['unrealized_pnl']:.2f}")
            print(f"Realized PnL: ${info['realized_pnl']:.2f}")
            print(f"Max Drawdown: {info['max_drawdown']*100:.2f}%")
            print(f"Trades: {info['total_trades']} (Win Rate: {info['win_rate']*100:.1f}%)")
            print(f"{'='*60}\n")
    
    def get_episode_stats(self) -> Dict:
        """
        Retorna estatísticas completas do episódio.
        
        Returns:
            Dicionário com estatísticas
        """
        info = self._get_info()
        
        equity_series = pd.Series(self.equity_history)
        returns = equity_series.pct_change().dropna()
        
        stats = {
            **info,
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 1 else 0.0,
            'max_equity': self.max_equity,
            'final_equity': self.equity,
        }
        
        return stats
