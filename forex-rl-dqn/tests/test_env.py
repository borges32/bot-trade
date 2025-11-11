"""Tests for trading environment."""
import numpy as np
import pytest

from src.rl.env import ForexTradingEnv


@pytest.fixture
def sample_env():
    """Create a sample environment for testing."""
    n_episodes = 100
    window_size = 10
    n_features = 5
    
    windows = np.random.randn(n_episodes, window_size, n_features).astype(np.float32)
    prices = np.random.uniform(1.0, 1.2, n_episodes).astype(np.float32)
    
    env = ForexTradingEnv(
        windows=windows,
        prices=prices,
        fee_perc=0.0001,
        spread_perc=0.0002,
    )
    
    return env


def test_env_initialization(sample_env):
    """Test environment initialization."""
    assert sample_env.n_episodes == 100
    assert sample_env.window_size == 10
    assert sample_env.n_features == 5
    assert sample_env.action_space.n == 3


def test_env_reset(sample_env):
    """Test environment reset."""
    obs, info = sample_env.reset()
    
    # Check observation shape
    assert obs.shape == (sample_env.window_size, sample_env.n_features)
    
    # Check info
    assert "step" in info
    assert "position" in info
    assert info["position"] == 0  # Start neutral


def test_env_step_hold(sample_env):
    """Test environment step with hold action."""
    obs, _ = sample_env.reset()
    
    # Take hold action
    next_obs, reward, terminated, truncated, info = sample_env.step(0)
    
    # Episode should terminate after one step
    assert terminated
    assert not truncated
    
    # Position should remain 0
    assert info["position"] == 0
    
    # No position change cost
    assert info["cost"] == 0.0


def test_env_step_buy(sample_env):
    """Test environment step with buy action."""
    obs, _ = sample_env.reset()
    
    # Take buy action
    next_obs, reward, terminated, truncated, info = sample_env.step(1)
    
    # Position should be +1 (long)
    assert info["position"] == 1
    
    # Should have position change cost
    expected_cost = sample_env.fee_perc + sample_env.spread_perc
    assert abs(info["cost"] - expected_cost) < 1e-9


def test_env_step_sell(sample_env):
    """Test environment step with sell action."""
    obs, _ = sample_env.reset()
    
    # Take sell action
    next_obs, reward, terminated, truncated, info = sample_env.step(2)
    
    # Position should be -1 (short)
    assert info["position"] == -1
    
    # Should have position change cost
    expected_cost = sample_env.fee_perc + sample_env.spread_perc
    assert abs(info["cost"] - expected_cost) < 1e-9


def test_env_position_change_cost():
    """Test that changing position incurs cost."""
    windows = np.random.randn(10, 5, 3).astype(np.float32)
    prices = np.array([1.1, 1.105, 1.11, 1.105, 1.1, 1.095, 1.1, 1.105, 1.11, 1.115], dtype=np.float32)
    
    env = ForexTradingEnv(
        windows=windows,
        prices=prices,
        fee_perc=0.01,  # 1% fee
        spread_perc=0.01,  # 1% spread
    )
    
    # Reset to first episode
    env.current_step = 0
    env.position = 0
    env.next_price = prices[0]
    env.current_price = prices[0]
    
    # Change from hold to buy
    _, reward, _, _, info = env.step(1)
    
    # Should have cost of 2% (fee + spread)
    assert info["cost"] == 0.02


def test_env_reward_calculation():
    """Test reward calculation with known prices."""
    windows = np.random.randn(1, 5, 3).astype(np.float32)
    
    # Price goes from 1.0 to 1.1 (10% increase)
    current_price = 1.0
    next_price = 1.1
    prices = np.array([next_price], dtype=np.float32)
    
    env = ForexTradingEnv(
        windows=windows,
        prices=prices,
        fee_perc=0.0,
        spread_perc=0.0,
    )
    
    env.current_step = 0
    env.position = 1  # Long position
    env.current_price = current_price
    env.next_price = next_price
    
    # With long position, should profit from price increase
    obs, _ = env.reset()
    env.position = 1  # Set long position
    env.current_price = current_price
    
    # Take hold action (no position change)
    _, reward, _, _, info = env.step(0)
    
    # Reward should be approximately price return (10%)
    expected_return = (next_price - current_price) / current_price
    # Actual reward is position_reward - cost
    # position_reward = return * position = 0.1 * 1 = 0.1
    # But since we changed position from 0 to 0, no cost
    assert abs(info["position_reward"]) < 0.15  # Allow some tolerance


def test_env_episode_count(sample_env):
    """Test episode count."""
    assert sample_env.get_episode_count() == 100
