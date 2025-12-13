#!/usr/bin/env python3
"""
Script para analisar resultados da otimização de hiperparâmetros.

Gera visualizações e relatórios a partir dos resultados salvos.
"""

import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def analyze_optimization_results(results_dir: str):
    """
    Analisa resultados da otimização.
    
    Args:
        results_dir: Diretório com resultados
    """
    results_dir = Path(results_dir)
    csv_path = results_dir / "optimization_results.csv"
    best_config_path = results_dir / "best_config.json"
    
    if not csv_path.exists():
        print(f"❌ Arquivo não encontrado: {csv_path}")
        return
    
    # Carrega dados
    df = pd.read_csv(csv_path)
    
    print(f"\n{'='*80}")
    print(f"ANÁLISE DE OTIMIZAÇÃO - {results_dir.name.upper()}")
    print(f"{'='*80}")
    print(f"Total de experimentos: {len(df)}")
    print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Estatísticas gerais
    print(f"\n📊 ESTATÍSTICAS GERAIS (Test Set)")
    print(f"{'='*80}")
    
    metrics = ['test_rmse', 'test_mae', 'test_r2', 'test_direction_acc']
    for metric in metrics:
        if metric in df.columns:
            print(f"\n{metric.upper()}:")
            print(f"  Min:    {df[metric].min():.6f}")
            print(f"  Max:    {df[metric].max():.6f}")
            print(f"  Média:  {df[metric].mean():.6f}")
            print(f"  Mediana: {df[metric].median():.6f}")
            print(f"  Std:    {df[metric].std():.6f}")
    
    # Melhor configuração
    if best_config_path.exists():
        with open(best_config_path, 'r') as f:
            best = json.load(f)
        
        print(f"\n{'='*80}")
        print(f"🏆 MELHOR CONFIGURAÇÃO")
        print(f"{'='*80}")
        print(f"Combined Score: {best['combined_score']:.6f}")
        print(f"\n📈 Métricas (Test Set):")
        print(f"  RMSE: {best['test_rmse']:.6f}")
        print(f"  MAE:  {best['test_mae']:.6f}")
        print(f"  R²:   {best['test_r2']:.6f}")
        print(f"  Direction Accuracy: {best['test_direction_acc']:.4f}")
        
        print(f"\n🎯 Features:")
        print(f"  EMA:       {best['use_ema']}")
        print(f"  MACD:      {best['use_macd']}")
        print(f"  RSI:       {best['use_rsi']}")
        print(f"  Bollinger: {best['use_bollinger']}")
        print(f"  ATR:       {best['use_atr']}")
        
        print(f"\n⚙️  Hiperparâmetros:")
        print(f"  Prediction Horizon: {best['prediction_horizon']}")
        print(f"  Learning Rate:      {best['learning_rate']}")
        print(f"  Num Leaves:         {best['num_leaves']}")
        print(f"  Max Depth:          {best['max_depth']}")
        print(f"  N Estimators:       {best['n_estimators']}")
        print(f"  Min Child Samples:  {best['min_child_samples']}")
        print(f"  Subsample:          {best['subsample']}")
        print(f"  Colsample Bytree:   {best['colsample_bytree']}")
        print(f"  Reg Alpha:          {best['reg_alpha']}")
        print(f"  Reg Lambda:         {best['reg_lambda']}")
    
    # Top 10 configurações
    print(f"\n{'='*80}")
    print(f"🌟 TOP 10 MELHORES CONFIGURAÇÕES")
    print(f"{'='*80}")
    
    top10 = df.nsmallest(10, 'combined_score')
    for rank, (idx, row) in enumerate(top10.iterrows(), 1):
        print(f"\n#{rank} - Score: {row['combined_score']:.6f}")
        print(f"  RMSE: {row['test_rmse']:.6f}, Direction Acc: {row['test_direction_acc']:.4f}")
        print(f"  Features: EMA={row['use_ema']}, MACD={row['use_macd']}, RSI={row['use_rsi']}, BB={row['use_bollinger']}, ATR={row['use_atr']}")
        print(f"  Params: LR={row['learning_rate']}, Leaves={row['num_leaves']}, Depth={row['max_depth']}, Est={row['n_estimators']}")
    
    # Análise de features
    print(f"\n{'='*80}")
    print(f"🔍 ANÁLISE DE FEATURES")
    print(f"{'='*80}")
    
    feature_cols = ['use_ema', 'use_macd', 'use_rsi', 'use_bollinger', 'use_atr']
    for feat in feature_cols:
        if feat in df.columns:
            avg_score_true = df[df[feat] == True]['combined_score'].mean()
            avg_score_false = df[df[feat] == False]['combined_score'].mean()
            count_true = (df[feat] == True).sum()
            count_false = (df[feat] == False).sum()
            
            print(f"\n{feat}:")
            print(f"  Com feature:  Score médio = {avg_score_true:.6f} (n={count_true})")
            print(f"  Sem feature:  Score médio = {avg_score_false:.6f} (n={count_false})")
            print(f"  Diferença:    {avg_score_true - avg_score_false:+.6f}")
    
    # Análise de correlação
    print(f"\n{'='*80}")
    print(f"📊 CORRELAÇÃO COM COMBINED SCORE")
    print(f"{'='*80}")
    
    numeric_cols = ['prediction_horizon', 'learning_rate', 'num_leaves', 'max_depth', 
                   'n_estimators', 'min_child_samples', 'subsample', 'colsample_bytree',
                   'reg_alpha', 'reg_lambda']
    
    correlations = []
    for col in numeric_cols:
        if col in df.columns:
            corr = df[col].corr(df['combined_score'])
            correlations.append((col, corr))
    
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print("\nParâmetros mais correlacionados (absoluto):")
    for param, corr in correlations[:5]:
        direction = "↓ Menor é melhor" if corr < 0 else "↑ Maior é melhor"
        print(f"  {param:20s}: {corr:+.4f}  {direction}")
    
    # Distribuição de scores
    print(f"\n{'='*80}")
    print(f"📈 DISTRIBUIÇÃO DE SCORES")
    print(f"{'='*80}")
    
    percentiles = [10, 25, 50, 75, 90]
    print("\nPercentis do Combined Score:")
    for p in percentiles:
        val = df['combined_score'].quantile(p/100)
        print(f"  P{p:2d}: {val:.6f}")
    
    # Salva relatório
    report_path = results_dir / "analysis_report.txt"
    with open(report_path, 'w') as f:
        f.write(f"RELATÓRIO DE ANÁLISE - {results_dir.name.upper()}\n")
        f.write(f"{'='*80}\n")
        f.write(f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total de experimentos: {len(df)}\n\n")
        
        if best_config_path.exists():
            f.write("MELHOR CONFIGURAÇÃO:\n")
            f.write(json.dumps(best, indent=2))
    
    print(f"\n💾 Relatório salvo em: {report_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analisa resultados da otimização')
    parser.add_argument(
        'results_dir',
        type=str,
        help='Diretório com resultados (ex: optimization_results/usdjpy_15m)'
    )
    
    args = parser.parse_args()
    
    analyze_optimization_results(args.results_dir)
    
    print("\n✅ Análise concluída!")
