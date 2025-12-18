import pandas as pd
import os

# Caminho do arquivo de entrada
input_file = 'data/usdjpy_history_30m.csv'
output_file = 'data/usdjpy_history_30m_processed.csv'

# Ler o CSV
df = pd.read_csv(input_file)

# Selecionar apenas as colunas necessárias
df_processed = df[['timestamp', 'open', 'high', 'low', 'close']].copy()

# Adicionar coluna volume com valor 0
df_processed['volume'] = 0

# Salvar o novo CSV
df_processed.to_csv(output_file, index=False)

print(f"Arquivo processado salvo em: {output_file}")
print(f"Total de linhas: {len(df_processed)}")
print(f"\nPrimeiras linhas:")
print(df_processed.head())
