# core/data_utils.py
import pandas as pd
import sqlite3
from pathlib import Path
import time

def salvar_resultados_fast(nome_estudo, combinacoes_params, resultados_matriz, db_name="quant_dashboard.db"):
    """
    Salva milhares de resultados no SQLite e rankeia por T-Stat.
    """
    print("\n💾 Salvando resultados no Banco de Dados (Modo Bulk Insert)...")
    inicio_io = time.time()
    
    diretorio_atual = Path(__file__).parent.parent
    db_path = diretorio_atual / db_name
    
    # 1. Converte as listas de parâmetros para DataFrame
    df = pd.DataFrame(combinacoes_params)
    
    # 2. Adiciona as colunas métricas que vieram do Numba
    df['T_Stat'] = resultados_matriz[:, 0]
    df['Lucro_Saldo'] = resultados_matriz[:, 1]
    df['Total_Trades'] = resultados_matriz[:, 2]
    
    # Ordena o DataFrame pelo T-Stat (Do mais confiável para o menos)
    df = df.sort_values(by='T_Stat', ascending=False)
    
    with sqlite3.connect(db_path) as conn:
        df.to_sql(nome_estudo, conn, if_exists='replace', index=False)
        
    fim_io = time.time()
    print(f"✅ {len(df)} simulações salvas em {fim_io - inicio_io:.4f} segundos!")

    # ============================================================
    # EXIBE A MELHOR ESTRATÉGIA NO TERMINAL (RANKING POR T-STAT)
    # ============================================================
    melhor_linha = df.iloc[0] # Como já ordenamos, o índice 0 é o melhor
    
    print("\n================ MELHOR RESULTADO (POR T-STAT) ================")
    print(f"Estatística T : {melhor_linha['T_Stat']:.4f} (Confiança Estatística)")
    print(f"Total Trades  : {int(melhor_linha['Total_Trades'])}")
    print(f"Saldo Final   : ${melhor_linha['Lucro_Saldo']:.2f}")
    print("--- Hiperparâmetros ---")
    for param in combinacoes_params[0].keys():
        print(f"  {param} : {melhor_linha[param]}")
    print("===============================================================\n")