# strategies/ea_bollinger_stoch.py
import numpy as np
from numba import njit
import time
import itertools
import random

# ==============================================================================
# 1. PRÉ-CÁLCULO DE INDICADORES (NUMBA C-LEVEL)
# ==============================================================================
@njit(fastmath=True, nogil=True)
def pre_calcular_bandas(closes, max_periodo):
    tamanho = len(closes)
    mid_matrix = np.zeros((max_periodo + 1, tamanho), dtype=np.float32)
    std_matrix = np.zeros((max_periodo + 1, tamanho), dtype=np.float32)

    for p in range(2, max_periodo + 1):
        for i in range(p - 1, tamanho):
            soma = 0.0
            for j in range(p): soma += closes[i - j]
            media = soma / p
            
            soma_sq_diff = 0.0
            for j in range(p): soma_sq_diff += (closes[i - j] - media) ** 2
            
            variancia = soma_sq_diff / (p - 1) if p > 1 else 1 
            mid_matrix[p, i] = media
            std_matrix[p, i] = np.sqrt(variancia)
    return mid_matrix, std_matrix

@njit(fastmath=True, nogil=True)
def pre_calcular_estocastico(highs, lows, closes, max_periodo):
    tamanho = len(closes)
    stoch_matrix = np.zeros((max_periodo + 1, tamanho), dtype=np.float32)
    
    for p in range(2, max_periodo + 1):
        for i in range(p - 1, tamanho):
            highest_high = highs[i]
            lowest_low = lows[i]
            for j in range(1, p):
                if highs[i - j] > highest_high: highest_high = highs[i - j]
                if lows[i - j] < lowest_low: lowest_low = lows[i - j]
            
            diff = highest_high - lowest_low
            if diff == 0:
                stoch_matrix[p, i] = 50.0 
            else:
                stoch_matrix[p, i] = ((closes[i] - lowest_low) / diff) * 100.0
    return stoch_matrix

@njit(fastmath=True, nogil=True)
def pre_calcular_adx(highs, lows, closes, max_periodo):
    """
    Pré-calcula a matriz do ADX usando Wilder's Smoothing.
    """
    tamanho = len(closes)
    adx_matrix = np.zeros((max_periodo + 1, tamanho), dtype=np.float32)
    
    # 1. Calcula TR, +DM e -DM (Vetores base, independem de período)
    tr = np.zeros(tamanho, dtype=np.float32)
    pdm = np.zeros(tamanho, dtype=np.float32)
    ndm = np.zeros(tamanho, dtype=np.float32)
    
    for i in range(1, tamanho):
        tr1 = highs[i] - lows[i]
        tr2 = abs(highs[i] - closes[i-1])
        tr3 = abs(lows[i] - closes[i-1])
        tr[i] = max(tr1, tr2, tr3)
        
        up_move = highs[i] - highs[i-1]
        down_move = lows[i-1] - lows[i]
        
        if up_move > down_move and up_move > 0: pdm[i] = up_move
        else: pdm[i] = 0.0
            
        if down_move > up_move and down_move > 0: ndm[i] = down_move
        else: ndm[i] = 0.0

    # 2. Loop principal de suavização para cada período exigido
    for p in range(2, max_periodo + 1):
        smooth_tr = 0.0
        smooth_pdm = 0.0
        smooth_ndm = 0.0
        
        dx = np.zeros(tamanho, dtype=np.float32)
        
        # Primeiro cálculo (Soma Simples)
        for i in range(1, p + 1):
            smooth_tr += tr[i]
            smooth_pdm += pdm[i]
            smooth_ndm += ndm[i]
            
        # Suavização de Wilder para o restante do array
        for i in range(p + 1, tamanho):
            smooth_tr = smooth_tr - (smooth_tr / p) + tr[i]
            smooth_pdm = smooth_pdm - (smooth_pdm / p) + pdm[i]
            smooth_ndm = smooth_ndm - (smooth_ndm / p) + ndm[i]
            
            if smooth_tr > 0:
                pdi = 100.0 * (smooth_pdm / smooth_tr)
                ndi = 100.0 * (smooth_ndm / smooth_tr)
                di_sum = pdi + ndi
                if di_sum > 0:
                    dx[i] = 100.0 * abs(pdi - ndi) / di_sum

        # Suavização Final do ADX em cima do DX
        adx_temp = 0.0
        for i in range(p + 1, 2 * p + 1):
            adx_temp += dx[i]
            
        adx_temp /= p
        adx_matrix[p, 2 * p] = adx_temp
        
        for i in range(2 * p + 1, tamanho):
            adx_temp = (adx_temp * (p - 1) + dx[i]) / p
            adx_matrix[p, i] = adx_temp

    return adx_matrix

# ==============================================================================
# 2. MOTOR DE BACKTEST C-LEVEL (SIMULADOR NUMBA)
# ==============================================================================
@njit(fastmath=True, nogil=True)
def simulador_numba(opens, closes, grid_p_bb, grid_d_bb, grid_p_st, grid_esp, grid_p_adx, grid_l_adx, mid_matrix, std_matrix, stoch_matrix, adx_matrix, saldo_ini, spread_pontos, min_trades):
    qtd_simulacoes = len(grid_p_bb)
    resultados = np.zeros((qtd_simulacoes, 3), dtype=np.float32)
    
    spread_em_preco = spread_pontos * np.float32(0.00001)
    tamanho_contrato = np.float32(10000.0)

    for idx in range(qtd_simulacoes):
        # Desempacota os hiperparâmetros desta simulação
        p_bb, d_bb = grid_p_bb[idx], grid_d_bb[idx]
        p_st, espacamento = grid_p_st[idx], grid_esp[idx]
        p_adx, l_adx = grid_p_adx[idx], grid_l_adx[idx]
        
        # Pega as linhas pré-calculadas corretas
        mid_band, std_band = mid_matrix[p_bb], std_matrix[p_bb]
        stoch_band = stoch_matrix[p_st]
        adx_band = adx_matrix[p_adx]
        
        nivel_compra = espacamento
        nivel_venda = 100.0 - espacamento
        
        saldo, posicao, preco_entrada = saldo_ini, 0, np.float32(0.0)
        n_trades, soma_retornos, soma_retornos_sq = 0, np.float32(0.0), np.float32(0.0)
        
        # Inicia após o maior período (ADX demora 2*p para ter o primeiro valor válido)
        start_idx = max(max(p_bb, p_st), p_adx * 2) + 1

        for i in range(start_idx, len(closes)):
            c1, c2 = closes[i-1], closes[i-2]
            
            mb1, mb2 = mid_band[i-1], mid_band[i-2]
            sd1, sd2 = std_band[i-1], std_band[i-2]
            
            lb1, lb2 = mb1 - (sd1 * d_bb), mb2 - (sd2 * d_bb)
            ub1, ub2 = mb1 + (sd1 * d_bb), mb2 + (sd2 * d_bb)
            
            stoch1 = stoch_band[i-1]
            adx1 = adx_band[i-1] # Valor do ADX na vela fechada anterior
            
            # --- LÓGICA CORE DO ROBÔ ---
            # Filtro ADX: Mercado não pode estar em forte tendência
            filtro_adx = adx1 < l_adx
            
            # Compra
            gatilho_compra = (c2 > lb2) and (c1 < lb1)
            filtro_compra = stoch1 <= nivel_compra
            sinal_compra = gatilho_compra and filtro_compra and filtro_adx
            
            # Venda
            gatilho_venda = (c2 < ub2) and (c1 > ub1)
            filtro_venda = stoch1 >= nivel_venda
            sinal_venda = gatilho_venda and filtro_venda and filtro_adx
            
            # Saídas (O ADX não impede saídas, apenas entradas)
            sinal_saida_compra = (c2 <= mb2) and (c1 > mb1)
            sinal_saida_venda = (c2 >= mb2) and (c1 < mb1)
            
            bid, ask = opens[i], opens[i] + spread_em_preco

            # GERENCIAMENTO DE SAÍDA E MÉTRICAS
            if posicao == 1 and sinal_saida_compra:
                lucro = bid - preco_entrada
                retorno_pct = lucro / preco_entrada
                saldo += lucro * tamanho_contrato
                n_trades += 1
                soma_retornos += retorno_pct; soma_retornos_sq += retorno_pct ** 2
                posicao = 0
            elif posicao == -1 and sinal_saida_venda:
                lucro = preco_entrada - ask
                retorno_pct = lucro / preco_entrada
                saldo += lucro * tamanho_contrato
                n_trades += 1
                soma_retornos += retorno_pct; soma_retornos_sq += retorno_pct ** 2
                posicao = 0

            # Novas Entradas
            if posicao == 0:
                if sinal_compra: posicao, preco_entrada = 1, ask 
                elif sinal_venda: posicao, preco_entrada = -1, bid 

        # Fechamento Forçado no Fim
        if posicao == 1: 
            lucro = closes[-1] - preco_entrada
            retorno_pct = lucro / preco_entrada
            saldo += lucro * tamanho_contrato
            n_trades += 1
            soma_retornos += retorno_pct; soma_retornos_sq += retorno_pct ** 2
        elif posicao == -1: 
            lucro = preco_entrada - (closes[-1] + spread_em_preco)
            retorno_pct = lucro / preco_entrada
            saldo += lucro * tamanho_contrato
            n_trades += 1
            soma_retornos += retorno_pct; soma_retornos_sq += retorno_pct ** 2

        # PENALIZAÇÃO DE BAIXA AMOSTRAGEM (T-STAT)
        t_stat = np.float32(-5.0) 
        if n_trades >= min_trades:
            media_retornos = soma_retornos / n_trades
            variancia = (soma_retornos_sq - n_trades * (media_retornos ** 2)) / (n_trades - 1)
            if variancia > 1e-10:
                desvio_padrao = np.sqrt(variancia)
                t_stat = (media_retornos / desvio_padrao) * np.sqrt(n_trades)

        resultados[idx, 0] = t_stat
        resultados[idx, 1] = saldo
        resultados[idx, 2] = n_trades

    return resultados

# ==============================================================================
# 3. INTERFACE PADRÃO DO FRAMEWORK
# ==============================================================================
def executar_otimizacao(opens, highs, lows, closes, spread, **kwargs):
    print("  -> Extraindo hiperparâmetros do YAML...")
    
    p_bb_list = kwargs['bb_periodo']
    d_bb_list = [round(x * 0.1, 1) for x in kwargs['bb_desvio']]
    p_st_list = kwargs['st_periodo']
    esp_list = kwargs['st_espacamento']
    p_adx_list = kwargs['adx_periodo']
    l_adx_list = kwargs['adx_linha']
    
    min_trades = np.int32(kwargs.get('min_trades', 50))
    max_backtests = kwargs.get('max_backtests', 100000) # Limite padrão se não tiver no YAML
    
    # 1. Gera TODAS as combinações possíveis na memória (Isso é rápido)
    todas_combinacoes = list(itertools.product(p_bb_list, d_bb_list, p_st_list, esp_list, p_adx_list, l_adx_list))
    total_possivel = len(todas_combinacoes)
    
    # 2. RANDOM SEARCH: Fatiamento aleatório se passar do limite
    if total_possivel > max_backtests:
        print(f"  ⚠️ [ALERTA] Espaço de busca gigante: {total_possivel:,} opções.")
        print(f"  🎲 Aplicando Random Search: Sorteando {max_backtests:,} combinações aleatórias...")
        # random.sample escolhe elementos únicos (sem repetição)
        combinacoes = random.sample(todas_combinacoes, max_backtests)
    else:
        print(f"  ✅ [INFO] Total de combinações ({total_possivel:,}) está dentro do limite. Executando Grid Completo.")
        combinacoes = todas_combinacoes
    
    # Desempacota para arrays do Numpy (Motor Numba)
    grid_p_bb = np.array([c[0] for c in combinacoes], dtype=np.int32)
    grid_d_bb = np.array([c[1] for c in combinacoes], dtype=np.float32)
    grid_p_st = np.array([c[2] for c in combinacoes], dtype=np.int32)
    grid_esp = np.array([c[3] for c in combinacoes], dtype=np.float32)
    grid_p_adx = np.array([c[4] for c in combinacoes], dtype=np.int32)
    grid_l_adx = np.array([c[5] for c in combinacoes], dtype=np.float32)
    
    # O pré-cálculo de matrizes precisa saber o valor máximo de toda a lista original, não só da amostra
    max_bb = max(p_bb_list)
    max_st = max(p_st_list)
    max_adx = max(p_adx_list)
    
    print(f"  -> Pré-calculando matrizes (Max BB: {max_bb} | Max Stoch: {max_st} | Max ADX: {max_adx})...")
    t0 = time.perf_counter()
    mid_matrix, std_matrix = pre_calcular_bandas(closes, max_bb)
    stoch_matrix = pre_calcular_estocastico(highs, lows, closes, max_st)
    adx_matrix = pre_calcular_adx(highs, lows, closes, max_adx)
    t1 = time.perf_counter()
    
    print(f"  -> Motor Numba disparado: Executando {len(combinacoes):,} backtests (Min. Trades: {min_trades})...")
    resultados_matriz = simulador_numba(
        opens, closes, 
        grid_p_bb, grid_d_bb, 
        grid_p_st, grid_esp, 
        grid_p_adx, grid_l_adx, 
        mid_matrix, std_matrix, stoch_matrix, adx_matrix, 
        np.float32(10000.0), spread, min_trades
    )
    t2 = time.perf_counter()
    
    print(f"  [TEMPO] Matrizes: {t1-t0:.2f}s | Backtests: {t2-t1:.4f}s")
    
    # Salva a lista exatamente como foi executada para o Dashboard
    combinacoes_originais = [
        {"bb_periodo": c[0], "bb_desvio": c[1], "st_periodo": c[2], "st_espacamento": c[3], "adx_periodo": c[4], "adx_linha": c[5]} 
        for c in combinacoes
    ]
    
    return combinacoes_originais, resultados_matriz