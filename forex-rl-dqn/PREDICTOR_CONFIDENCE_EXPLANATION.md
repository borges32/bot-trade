# 📊 Entendendo o Confidence no Predictor

## ❌ Problema Atual

O `confidence` retornado pelo predictor atual **NÃO** é o `test_direction_acc` do treinamento:

### Confidence Atual (INCORRETO):
```python
# Em predictor.py, linha 118
confidence = min(abs(predicted_return) * 500, 1.0)
```

**Problemas:**
- É uma heurística arbitrária baseada na magnitude do retorno previsto
- Não reflete a probabilidade real de acerto do modelo
- Um retorno de 0.002 (0.2%) = 100% de confiança (sem base estatística)

### Test Direction Accuracy (CORRETO):
```python
# Em lightgbm_model.py, linha 298
direction_correct = np.sign(y_true) == np.sign(y_pred)
metrics['direction_accuracy'] = direction_correct.mean()
```

**É a métrica correta porque:**
- Representa a **acurácia real** do modelo em prever a direção
- Exemplo: 55.2% = modelo acerta a direção 55.2% das vezes
- É uma **probabilidade estatisticamente validada**

---

## ✅ Solução: Usar Test Direction Accuracy como Confidence

### Passo 1: Salvar métricas junto com o modelo

Modificar `lightgbm_model.py` para salvar as métricas de teste:

```python
def save(self, path: Union[str, Path], test_metrics: Optional[Dict] = None):
    """Salva o modelo treinado com métricas de teste."""
    # ... código existente ...
    
    metadata = {
        'model_type': self.model_type,
        'prediction_horizon': self.prediction_horizon,
        'classification_threshold': self.classification_threshold,
        'feature_names': self.feature_names,
        'params': self.params,
        'test_metrics': test_metrics  # ← ADICIONAR ISSO
    }
    
    metadata_path = path.with_suffix('.metadata.pkl')
    joblib.dump(metadata, metadata_path)
```

### Passo 2: Carregar métricas no predictor

Modificar `predictor.py` para carregar e usar a acurácia real:

```python
class TradingPredictor:
    def __init__(self, lightgbm_path: str, config: Dict):
        # ... código existente ...
        
        # Carrega métricas de teste salvas
        metadata_path = Path(lightgbm_path).with_suffix('.metadata.pkl')
        if metadata_path.exists():
            import joblib
            metadata = joblib.load(metadata_path)
            self.test_direction_acc = metadata.get('test_metrics', {}).get('direction_accuracy', 0.55)
        else:
            self.test_direction_acc = 0.55  # Valor padrão conservador
        
        logger.info(f"Model test direction accuracy: {self.test_direction_acc:.2%}")
    
    def predict(self, candles: pd.DataFrame, current_price: Optional[float] = None) -> Dict:
        # ... código existente ...
        
        # USA A ACURÁCIA REAL DO MODELO
        confidence = self.test_direction_acc
        
        # Opcional: Ajusta confiança pela magnitude do retorno
        # Quanto maior o retorno previsto, maior a confiança
        magnitude_factor = min(abs(predicted_return) * 100, 1.0)
        adjusted_confidence = confidence * magnitude_factor
        
        result = {
            'signal': signal,
            'predicted_return': float(predicted_return),
            'confidence': float(adjusted_confidence),  # ← Confiança ajustada
            'base_accuracy': float(confidence),  # ← Acurácia base do modelo
            'current_price': float(current_price)
        }
        
        return result
```

---

## 📈 Exemplo de Uso

### Antes (Confiança Incorreta):
```python
result = predictor.predict(candles)
print(result)
# {
#   'signal': 'BUY',
#   'predicted_return': 0.0020,
#   'confidence': 1.0,  # ← 100%? Impossível!
#   'current_price': 148.50
# }
```

### Depois (Confiança Real):
```python
result = predictor.predict(candles)
print(result)
# {
#   'signal': 'BUY',
#   'predicted_return': 0.0020,
#   'confidence': 0.552,  # ← 55.2% (acurácia real do modelo)
#   'base_accuracy': 0.552,
#   'current_price': 148.50
# }
```

---

## 🎯 Interpretação Correta

### Com test_direction_acc = 0.552 (55.2%):

| Retorno Previsto | Confiança Ajustada | Interpretação |
|------------------|-------------------|---------------|
| +0.001 (0.1%) | 55.2% × 0.1 = 5.5% | Sinal fraco, ignorar |
| +0.005 (0.5%) | 55.2% × 0.5 = 27.6% | Sinal médio |
| +0.010 (1.0%) | 55.2% × 1.0 = 55.2% | **Sinal forte** |
| +0.020 (2.0%) | 55.2% × 1.0 = 55.2% | **Sinal muito forte** |

### Threshold Recomendado:
```python
min_confidence = 0.45  # 45% (ajustado pela magnitude)
```

Isso significa: aceitar sinais quando `base_accuracy × magnitude ≥ 45%`

---

## 📊 Dados dos Seus Modelos

Baseado nos resultados de otimização:

```csv
test_direction_acc: 0.5067961165048543 (50.68%)
```

**Interpretação:**
- O modelo acerta a direção em **50.68%** dos casos
- É ligeiramente melhor que chance aleatória (50%)
- Use como `confidence` base, ajustado pela magnitude do retorno

---

## 🔧 Implementação Recomendada

1. **Salvar métricas no treinamento**
2. **Carregar métricas no predictor**
3. **Usar `test_direction_acc` como confiança base**
4. **Ajustar pela magnitude do retorno previsto**
5. **Definir threshold realista** (ex: 40-50%)

Isso dará **probabilidades estatisticamente corretas** para suas decisões de trading.
