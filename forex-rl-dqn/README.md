# Forex RL DQN - Reinforcement Learning for Forex Trading

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange)
![License](https://img.shields.io/badge/license-MIT-green)

A complete, production-ready Reinforcement Learning system for Forex trading using **Dueling Double DQN with LSTM architecture**. This project includes offline training with historical data and a REST API for real-time inference.

## ⚠️ **DISCLAIMER**

**THIS SOFTWARE IS FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY.**

- **FINANCIAL RISK:** Trading in foreign exchange (Forex) carries a high level of risk and may not be suitable for all investors. The high degree of leverage can work against you as well as for you.
- **NO WARRANTY:** This software is provided "AS IS" without any warranty. Past performance is not indicative of future results.
- **NOT FINANCIAL ADVICE:** Nothing in this project constitutes financial advice. Always consult with a qualified financial advisor before making investment decisions.
- **POTENTIAL LOSSES:** You may lose some or all of your invested capital. Never invest money you cannot afford to lose.
- **USE AT YOUR OWN RISK:** The developers and contributors are not responsible for any financial losses incurred through the use of this software.

By using this software, you acknowledge and accept all risks associated with algorithmic trading.

---

## 🎯 Features

- **Advanced RL Architecture:** Dueling Double DQN with LSTM for sequential data processing
- **Technical Indicators:** RSI, EMA, Bollinger Bands, Returns (no TA-Lib dependency)
- **Realistic Trading Costs:** Fee and spread modeling
- **Complete Pipeline:** Data loading → Feature engineering → Training → Inference
- **REST API:** FastAPI-based inference service with Pydantic v2 validation
- **Docker Support:** Containerized deployment with docker-compose
- **Comprehensive Tests:** Unit tests for features, environment, and API
- **CI/CD:** GitHub Actions workflow for automated testing and linting

## 🏗️ Architecture

```
Input: OHLCV Window (64 bars)
    ↓
Feature Engineering (RSI, EMA, BB, Returns)
    ↓
Normalization (StandardScaler)
    ↓
LSTM Layer (128 hidden units)
    ↓
Dueling Architecture
    ├─ Value Stream → V(s)
    └─ Advantage Stream → A(s,a)
    ↓
Q(s,a) = V(s) + (A(s,a) - mean(A))
    ↓
Actions: {0: hold, 1: buy, 2: sell}
```

## 📋 Prerequisites

- Python 3.10 or higher
- pip package manager
- (Optional) Docker and docker-compose for containerized deployment
- (Optional) CUDA-capable GPU for faster training

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
cd forex-rl-dqn

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Synthetic Data (for testing)

```bash
python -m src.data.make_synth --output data/ct.csv --n-bars 10000
```

### 3. Prepare Your cTrader Data

If you have real data from cTrader, ensure your CSV has these columns:

- `timestamp` (ISO 8601 format, e.g., "2024-01-01T00:00:00Z")
- `open` (opening price)
- `high` (highest price)
- `low` (lowest price)
- `close` (closing price)
- `volume` (trade volume)

The data should be sorted chronologically.

### 4. Train the Agent

```bash
python -m src.rl.train --data data/ct.csv --config config.yaml --artifacts artifacts
```

**Training Output:**
- `artifacts/dqn.pt` - Trained model weights
- `artifacts/feature_state.json` - Feature scaler parameters
- `artifacts/config.yaml` - Training configuration
- `artifacts/dqn_step_*.pt` - Periodic checkpoints

**Training Progress:**
```
Loading data from data/ct.csv...
Loaded 10000 records
Generating features: ['rsi_14', 'ema_12', ...]
Train: 8000 records, Val: 2000 records
...
Step 5000 | Loss: 0.0234 | Epsilon: 0.850 | Avg Reward: 0.0012
--- Evaluation at step 5000 ---
Avg Reward: 0.001523
Win Rate: 54.23%
```

### 5. Start the API Server

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### 6. Make Predictions

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Get Trading Action:**
```bash
curl -X POST http://localhost:8000/act \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "EURUSD",
    "window": [
      {
        "timestamp": "2024-01-01T00:00:00Z",
        "open": 1.1000,
        "high": 1.1010,
        "low": 1.0990,
        "close": 1.1005,
        "volume": 1234.56
      },
      ... (64 bars total)
    ]
  }'
```

**Response:**
```json
{
  "action": "buy",
  "action_id": 1,
  "confidence": 0.73
}
```

**Actions:**
- `hold` (0): Neutral position
- `buy` (1): Long position
- `sell` (2): Short position

### 7. Ingest Historical Data (New!)

The API now includes an endpoint to receive and persist historical data from cTrader:

```bash
curl -X POST "http://localhost:8000/ingest?symbol=EURUSD" \
  -H "Content-Type: application/json" \
  -d '[
    {
      "timestamp": "2024-01-01T00:00:00Z",
      "open": 1.1000,
      "high": 1.1010,
      "low": 1.0990,
      "close": 1.1005,
      "volume": 1234.56
    },
    {
      "timestamp": "2024-01-01T00:01:00Z",
      "open": 1.1005,
      "high": 1.1015,
      "low": 1.0995,
      "close": 1.1008,
      "volume": 1567.89
    }
  ]'
```

**Response:**
```json
{
  "status": "success",
  "records_saved": 2,
  "file_path": "data/eurusd_history.csv"
}
```

**Features:**
- Receives an array of OHLCV bars directly in the request body
- Symbol specified as query parameter (default: EURUSD)
- Creates CSV file if it doesn't exist
- Appends to existing file if it exists
- Separate files for each symbol (`{symbol}_history.csv`)
- Data persisted in the `data/` directory (accessible from host via Docker volume)

## 🐳 Docker Deployment

### Build and Run

```bash
# Build the Docker image
docker build -t forex-rl-dqn .

# Run with docker-compose
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop
docker-compose down
```

The API will be available at `http://localhost:8000`

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
seed: 42  # Random seed for reproducibility

env:
  window_size: 64  # Number of bars in observation window
  fee_perc: 0.0001  # 0.01% trading fee
  spread_perc: 0.0002  # 0.02% bid-ask spread
  scale_features: true  # Normalize features
  features:  # Technical indicators to use
    - rsi_14
    - ema_12
    - ema_26
    - bb_upper_20
    - bb_lower_20
    - returns_1
    - returns_5

agent:
  gamma: 0.99  # Discount factor
  lr: 0.0001  # Learning rate
  batch_size: 64
  replay_size: 100000
  epsilon_start: 1.0  # Initial exploration
  epsilon_end: 0.05  # Final exploration
  epsilon_decay_steps: 50000
  lstm_hidden: 128  # LSTM hidden units
  mlp_hidden: 256  # MLP hidden units
  dueling: true  # Use dueling architecture

train:
  max_steps: 200000  # Total training steps
  eval_interval: 5000  # Evaluation frequency
  checkpoint_interval: 10000  # Save frequency
  device: auto  # auto, cpu, cuda
```

## 🧪 Testing

Run all tests:

```bash
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ -v --cov=src --cov-report=html
```

Run specific test file:

```bash
pytest tests/test_features.py -v
```

## 📁 Project Structure

```
forex-rl-dqn/
├── src/
│   ├── common/
│   │   ├── __init__.py
│   │   ├── utils.py          # Utilities (seed, device selection)
│   │   └── features.py       # Feature engineering
│   ├── rl/
│   │   ├── __init__.py
│   │   ├── env.py           # Trading environment
│   │   ├── agent.py         # DQN agent
│   │   ├── replay_buffer.py # Experience replay
│   │   └── train.py         # Training script
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py          # FastAPI application
│   └── data/
│       ├── __init__.py
│       └── make_synth.py    # Synthetic data generator
├── tests/
│   ├── __init__.py
│   ├── test_features.py
│   ├── test_env.py
│   └── test_api.py
├── artifacts/               # Generated during training
│   ├── dqn.pt
│   ├── feature_state.json
│   └── config.yaml
├── data/                    # Place your CSV files here
│   └── ct.csv
├── .github/
│   └── workflows/
│       └── ci.yml           # CI/CD pipeline
├── config.yaml              # Main configuration
├── requirements.txt         # Python dependencies
├── pyproject.toml          # Tool configurations
├── example_usage.py         # Example: How to use the trained model
├── example_ingest.py        # Example: How to ingest historical data
├── ctrader_integration_example.py  # Example: cTrader integration
├── INGEST_API.md           # Documentation: /ingest endpoint
├── Dockerfile
├── docker-compose.yml
├── LICENSE
└── README.md
```

## 📚 Additional Documentation

- **[INGEST_API.md](INGEST_API.md)** - Complete documentation for the data ingestion endpoint
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Deployment guide
- **[STRUCTURE.md](STRUCTURE.md)** - Project structure details

## 🔧 Troubleshooting

### Issue: Model not loading in API

**Solution:** Ensure artifacts exist:
```bash
ls -la artifacts/
# Should show: dqn.pt, feature_state.json, config.yaml
```

### Issue: Training is slow

**Solutions:**
- Use GPU: Set `device: cuda` in `config.yaml`
- Reduce `max_steps` or `window_size`
- Increase `batch_size` if you have enough memory

### Issue: Poor performance

**Solutions:**
- Increase training steps (`max_steps`)
- Adjust hyperparameters (learning rate, epsilon decay)
- Add more features
- Use more historical data
- Tune cost parameters (`fee_perc`, `spread_perc`)

### Issue: API validation errors

**Solution:** Ensure window size matches config:
```python
# Check required window size
import yaml
with open('artifacts/config.yaml') as f:
    config = yaml.safe_load(f)
print(f"Required window size: {config['env']['window_size']}")
```

## 📊 Hyperparameter Tuning

Key parameters to experiment with:

1. **Learning Rate** (`lr`): Start with 0.0001, try 0.001 or 0.00005
2. **Window Size** (`window_size`): 32, 64, or 128 bars
3. **LSTM Hidden** (`lstm_hidden`): 64, 128, or 256
4. **Epsilon Decay** (`epsilon_decay_steps`): Faster (25000) or slower (100000)
5. **Features**: Add/remove technical indicators

## 🔬 Feature Engineering

Current features (without TA-Lib):

- **RSI(14)**: Relative Strength Index
- **EMA(12)**: Exponential Moving Average (12-period)
- **EMA(26)**: Exponential Moving Average (26-period)
- **BB Upper(20)**: Bollinger Band upper (20-period, 2σ)
- **BB Lower(20)**: Bollinger Band lower (20-period, 2σ)
- **Returns(1)**: 1-period percentage returns
- **Returns(5)**: 5-period percentage returns

All features are normalized using StandardScaler fitted on training data.

## 📈 Performance Metrics

During training and evaluation, the system tracks:

- **Average Reward**: Mean reward per episode
- **Average Position Reward**: Reward from price movements (excluding costs)
- **Average Cost**: Transaction costs per trade
- **Win Rate**: Percentage of profitable episodes

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI Gymnasium for the RL environment interface
- PyTorch team for the deep learning framework
- FastAPI for the excellent web framework

## 📚 References

- [Double DQN Paper](https://arxiv.org/abs/1509.06461)
- [Dueling DQN Paper](https://arxiv.org/abs/1511.06581)
- [LSTM Networks](https://www.bioinf.jku.at/publications/older/2604.pdf)

---

**Remember:** Always backtest thoroughly and use paper trading before risking real capital!
