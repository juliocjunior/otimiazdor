# app_dashboard_quant.py
import streamlit as st
import pandas as pd
import optuna
import plotly.express as px
from pathlib import Path
import sqlite3
import numpy as np


# ==========================================
# 1. CONFIGURAÇÃO DA PÁGINA
# ==========================================
st.set_page_config(
    page_title="QuantLab | Análise de Superfície",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔬 QuantLab: Análise Multidimensional de Parâmetros")
st.markdown("Filtre as dimensões ocultas para encontrar Zonas de Estabilidade (Clusters) reais.")

# ==========================================
# 2. MOTOR DE EXTRAÇÃO DE DADOS PANDAS/SQLITE (CACHED)
# ==========================================
@st.cache_data(ttl=60)
def carregar_dados_sqlite(db_name="quant_dashboard.db", nome_tabela=None):
    db_path = Path(__file__).parent / db_name
    
    if not db_path.exists():
        st.error(f"Banco de dados não encontrado: {db_path}")
        return pd.DataFrame(), [], ""

    try:
        conn = sqlite3.connect(db_path)
        
        # Se não passar o nome, pega a última tabela válida (ignora tabelas de sistema do SQLite)
        if nome_tabela is None:
            cursor = conn.cursor()
            # A query abaixo filtra tabelas do sistema que começam com 'sqlite_'
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
            tabelas = [row[0] for row in cursor.fetchall()]
            
            if not tabelas:
                return pd.DataFrame(), [], ""
            nome_tabela = tabelas[-1] # Pega a última inserida (seu backtest mais recente)
            
        df = pd.read_sql_query(f'SELECT * FROM "{nome_tabela}"', conn)
        conn.close()
        
        parametros = [col for col in df.columns if col != 'Lucro/Saldo']
        return df, parametros, nome_tabela
        
    except Exception as e:
        st.error(f"Erro ao ler banco de dados: {e}")
        return pd.DataFrame(), [], ""

# Executa a carga de dados
df_bruto, lista_parametros, nome_estudo = carregar_dados_sqlite()

if df_bruto.empty:
    st.stop()

# ==========================================
# 3. SIDEBAR: FILTROS DINÂMICOS (A MÁGICA)
# ==========================================
st.sidebar.header(f"Estudo: {nome_estudo}")
st.sidebar.markdown("---")
st.sidebar.subheader("🔪 Fatiador N-Dimensional")
st.sidebar.caption("Reduza o Efeito de Média Oculta ajustando as variáveis que não estão no Heatmap.")

df_filtrado = df_bruto.copy()

# Cria um slider dinâmico para CADA parâmetro que existe no DB
filtros = {}
for param in lista_parametros:
    min_val = float(df_bruto[param].min())
    max_val = float(df_bruto[param].max())
    
    # Se min e max forem iguais, não precisa de slider
    if min_val == max_val:
        st.sidebar.info(f"{param}: Fixo em {min_val}")
        continue
        
    step = 1.0 if pd.api.types.is_integer_dtype(df_bruto[param]) else (max_val - min_val) / 100
    
    filtros[param] = st.sidebar.slider(
        f"Filtro: {param}",
        min_value=min_val,
        max_value=max_val,
        value=(min_val, max_val),
        step=step
    )
    
    # Aplica o filtro no dataframe em tempo real
    df_filtrado = df_filtrado[
        (df_filtrado[param] >= filtros[param][0]) & 
        (df_filtrado[param] <= filtros[param][1])
    ]

st.sidebar.markdown("---")
st.sidebar.metric("Simulações Filtradas", f"{len(df_filtrado)} / {len(df_bruto)}")

# ==========================================
# 4. MAIN ÁREA: VISUALIZAÇÕES QUANTITATIVAS
# ==========================================

if df_filtrado.empty:
    st.warning("Filtro muito restritivo. Nenhuma simulação encontrada com esses parâmetros.")
    st.stop()

# --- Encontrando o Verdadeiro "Melhor" (Baseado em Matemática, não em sorte) ---
# Aqui pegamos o índice da linha com o maior T-Stat
idx_melhor = df_filtrado['T_Stat'].idxmax()
linha_melhor = df_filtrado.loc[idx_melhor]

# --- KPI Section ---
col1, col2, col3 = st.columns(3)

col1.metric(
    "Melhor T-Stat (Confiança)", 
    f"{linha_melhor['T_Stat']:.3f}", 
    help="> 1.96 = 95% de confiança. > 2.58 = 99% de confiança."
)
col2.metric(
    "Lucro deste Setup", 
    f"${linha_melhor['Lucro_Saldo']:.2f}"
)
col3.metric(
    "Média de Trades no Cluster", 
    f"{int(df_filtrado['Total_Trades'].mean())}"
)

# --- Melhor Configuração Dinâmica ---
st.success(f"**🎯 Melhor Configuração Matemática (T-Stat: {linha_melhor['T_Stat']:.2f} | Trades: {int(linha_melhor['Total_Trades'])})**")

cols_params = st.columns(len(lista_parametros))
for i, param in enumerate(lista_parametros):
    valor_param = linha_melhor[param]
    if isinstance(valor_param, (int, np.integer)) or (isinstance(valor_param, float) and valor_param.is_integer()):
        cols_params[i].metric(param, f"{int(valor_param)}")
    else:
        cols_params[i].metric(param, f"{valor_param:.2f}")

st.markdown("---")

# --- Heatmap Section (2D) ---
st.subheader("🗺️ Matriz de Superfície: Zonas de Alfa (T-Stat)")
st.caption("Procure por 'clusters' (regiões contíguas) verdes intensas. Elas indicam parâmetros robustos a variações.")
col_x, col_y, col_agg = st.columns([2, 2, 1])

with col_x:
    eixo_x = st.selectbox("Eixo X (Ex: Período)", lista_parametros, index=0)
with col_y:
    # Garante que o eixo Y seja diferente do X se houver mais de um parâmetro
    index_y = 1 if len(lista_parametros) > 1 else 0
    eixo_y = st.selectbox("Eixo Y (Ex: Desvio)", lista_parametros, index=index_y)
with col_agg:
    agg_func = st.selectbox("Agregação da Célula", ['mean', 'max', 'median'], index=0, help="Agregação do T-Stat")

try:
    pivot_df = df_filtrado.pivot_table(
        index=eixo_y, 
        columns=eixo_x, 
        values='T_Stat', # AGORA O MAPA DE CALOR É SOBRE A ESTATÍSTICA T
        aggfunc=agg_func
    )
    
    fig_heat = px.imshow(
        pivot_df,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdYlGn", 
        origin='lower',
        title=f"Heatmap de T-Stat: {eixo_y} vs {eixo_x}"
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    
except Exception as e:
    st.error(f"Erro ao gerar Heatmap. Certifique-se de que os filtros permitem variação nos dois eixos. Erro: {e}")

st.markdown("---")

# --- 1D Slice Section ---
st.subheader("🔪 Fatiamento 1D (Estabilidade do Parâmetro)")
eixo_slice = st.selectbox("Selecione o Parâmetro para análise Isolada:", lista_parametros, index=0)

fig_scatter = px.scatter(
    df_filtrado, 
    x=eixo_slice, 
    y="T_Stat", # AGORA O EIXO Y É O T-STAT
    color="Lucro_Saldo", # Mantemos a cor como Lucro para você ver se alto T-Stat também deu dinheiro financeiro
    color_continuous_scale="RdYlGn",
    hover_data=["Total_Trades"], # Mostra os trades ao passar o mouse
    trendline="lowess", 
    title=f"Confiança Estatística (T-Stat) ao longo de: {eixo_slice}"
)
# Linha de referência de significância estatística (95%)
fig_scatter.add_hline(y=1.96, line_dash="dash", line_color="white", annotation_text="95% Confiança (T > 1.96)")

st.plotly_chart(fig_scatter, use_container_width=True)

# --- Tabela de Dados Brutos ---
with st.expander("Ver Tabela de Dados (Top Resultados Rankeados por T-Stat)"):
    # Ordena pelo T-Stat de forma descendente
    st.dataframe(df_filtrado.sort_values(by="T_Stat", ascending=False).head(100))