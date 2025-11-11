# Forex RL DQN - Deployment Guide

## Quick Start Commands

### 1. Setup (First Time)
```bash
cd forex-rl-dqn
chmod +x setup.sh
./setup.sh
```

### 2. Manual Setup (Alternative)
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Generate synthetic data
python -m src.data.make_synth --output data/ct.csv --n-bars 10000
```

### 3. Train the Model
```bash
# Activate virtual environment if not active
source venv/bin/activate

# Train
python -m src.rl.train --data data/ct.csv --config config.yaml --artifacts artifacts

# Expected output:
# - artifacts/dqn.pt (model weights)
# - artifacts/feature_state.json (scaler)
# - artifacts/config.yaml (config copy)
```

### 4. Start API Server
```bash
# Local development
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Production
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 5. Test the API
```bash
# Health check
curl http://localhost:8000/health

# Example request
python example_usage.py
```

## Docker Deployment

### Build and Run
```bash
# Build image
docker build -t forex-rl-dqn:latest .

# Run container
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/artifacts:/app/artifacts:ro \
  --name forex-rl-api \
  forex-rl-dqn:latest

# Check logs
docker logs -f forex-rl-api
```

### Docker Compose
```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov-report=html

# Specific test file
pytest tests/test_features.py -v

# Specific test
pytest tests/test_env.py::test_env_step_buy -v
```

## Code Quality

```bash
# Format code
black src/ tests/

# Check formatting
black --check src/ tests/

# Lint
flake8 src/ tests/ --max-line-length=100

# Sort imports
isort src/ tests/
```

## Project Validation Checklist

- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Data generated or loaded (`data/ct.csv` exists)
- [ ] Model trained (artifacts in `artifacts/` directory)
- [ ] Tests passing (`pytest tests/ -v`)
- [ ] API running (`curl http://localhost:8000/health`)
- [ ] Example working (`python example_usage.py`)
- [ ] Docker builds (`docker build -t forex-rl-dqn .`)

## Common Issues and Solutions

### Issue: Import errors
**Solution**: Ensure virtual environment is activated
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: Model not found
**Solution**: Train the model first
```bash
python -m src.rl.train --data data/ct.csv --config config.yaml
```

### Issue: API returns 503
**Solution**: Check if model artifacts exist
```bash
ls -la artifacts/
# Should show: dqn.pt, feature_state.json, config.yaml
```

### Issue: Tests fail
**Solution**: Install test dependencies
```bash
pip install pytest pytest-cov httpx
```

## Performance Tuning

### For Faster Training
- Use GPU: Set `device: cuda` in `config.yaml`
- Increase batch size: `batch_size: 128`
- Reduce max steps: `max_steps: 100000`

### For Better Results
- More data: Use larger historical datasets
- More features: Add additional technical indicators
- Longer training: Increase `max_steps`
- Hyperparameter tuning: Adjust learning rate, epsilon decay

### For Production API
- Multiple workers: `--workers 4`
- Use gunicorn: `gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.api.main:app`
- Enable caching for model loading
- Use reverse proxy (nginx)

## Monitoring

### Check API Health
```bash
while true; do curl -s http://localhost:8000/health | jq; sleep 5; done
```

### Monitor Training Progress
```bash
tail -f training.log
```

### Docker Container Stats
```bash
docker stats forex-rl-api
```

## Backup and Restore

### Backup Model
```bash
tar -czf forex-rl-backup-$(date +%Y%m%d).tar.gz artifacts/
```

### Restore Model
```bash
tar -xzf forex-rl-backup-20250110.tar.gz
```

## Next Steps

1. **Collect Real Data**: Export historical data from cTrader
2. **Backtest**: Evaluate performance on historical data
3. **Paper Trade**: Test in simulated environment
4. **Monitor**: Track performance metrics
5. **Iterate**: Adjust hyperparameters based on results

## Resources

- Documentation: `README.md`
- Structure: `STRUCTURE.md`
- Examples: `example_usage.py`
- Config: `config.yaml`
- Tests: `tests/`

---

**Remember**: Always test thoroughly before using with real money!
