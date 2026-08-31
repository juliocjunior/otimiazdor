# core/data_loader.py
import pandas as pd
import numpy as np

def carregar_dados_mt5(caminho_arquivo, data_inicio=None, data_fim=None):
    print(f"Carregando {caminho_arquivo}...")
    
    # 1. Leitura e Limpeza do Cabeçalho
    df = pd.read_csv(caminho_arquivo, sep='\t')
    df.columns = [col.replace('<', '').replace('>', '').lower().strip() for col in df.columns]
    
    # 2. Indexação de Data/Hora (Crucial para Time Series)
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], cache=True)
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)
    
    # 3. Filtro de período para Treinamento vs Validação (Out-of-Sample)
    if data_inicio:
        df = df.loc[data_inicio:]
    if data_fim:
        df = df.loc[:data_fim]
        
    spread_medio_pontos = int(df['spread'].mode()[0])
    print(f"Período: {df.index[0]} até {df.index[-1]} | Candles: {len(df)} | Spread: {spread_medio_pontos}")
    
    # 4. Extração de Arrays Numpy Nativos (C-Level para o Numba)
    array_opens = df['open'].to_numpy(dtype=np.float32)
    array_highs = df['high'].to_numpy(dtype=np.float32)   # <-- ADICIONADO (MÁXIMAS)
    array_lows = df['low'].to_numpy(dtype=np.float32)     # <-- ADICIONADO (MÍNIMAS)
    array_closes = df['close'].to_numpy(dtype=np.float32)
    
    # 5. Retorna exatamente as 5 variáveis esperadas pelo Motor Genérico
    return array_opens, array_highs, array_lows, array_closes, spread_medio_pontos