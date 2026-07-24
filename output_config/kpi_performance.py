import numpy as np

def calculate_global_kpis_old(vars_dict, params, S, T, J, I, P):
    """
    Calcula y muestra los KPIs globales del modelo estocastico.
    
    vars_dict: Diccionario con los valores optimos extraidos (y, x, I_var, y_bar, w).
    params: Objeto o diccionario con los parametros del modelo (pi, B, epsilon, delta, a, b, precios).
    S, T, J, I, P: Listas o conjuntos de indices (escenarios, periodos, productos, componentes, precios).
    """
    
    # 1. Acumuladores para los valores esperados
    expected_assembled_components = 0.0
    expected_total_components_in = 0.0
    expected_inventory_held = 0.0
    expected_revenue = 0.0
    expected_total_assembled_products = 0.0
    expected_real_demand = 0.0
    
    # Extraer estructuras para legibilidad
    pi = params['pi']  # Probabilidades de los escenarios
    B = params['B']    # Matriz Bill of Materials
    
    for s in S:
        prob = pi[s]
        
        # Iteradores por escenario
        scen_assembled_comps = sum(B[i][j] * vars_dict['y'][j, t, s] for t in T for j in J for i in I)
        scen_comps_in = sum(vars_dict['I_0'][i, s] + sum(vars_dict['x'][i, t, s] for t in T) for i in I)
        scen_inv_held = sum(vars_dict['I_var'][i, t, s] for t in T for i in I)
        
        scen_revenue = sum(params['prices'][p] * vars_dict['y_bar'][j, t, p, s] for t in T for j in J for p in P)
        scen_assembled_prods = sum(vars_dict['y'][j, t, s] for t in T for j in J)
        
        # Demanda real bajo los precios seleccionados (w_j,t,p)
        scen_demand = 0.0
        for t in T:
            for j in J:
                selected_price = sum(params['prices'][p] * vars_dict['w'][j, t, p, s] for p in P)
                demand_val = params['epsilon'][s, t] * (params['a'] - params['b'] * selected_price) + params['delta'][s, j, t]
                scen_demand += demand_val
        
        # Ponderacion por la probabilidad del escenario
        expected_assembled_components += prob * scen_assembled_comps
        expected_total_components_in += prob * scen_comps_in
        expected_inventory_held += prob * scen_inv_held
        expected_revenue += prob * scen_revenue
        expected_total_assembled_products += prob * scen_assembled_prods
        expected_real_demand += prob * scen_demand

    # 2. Calculo final de los Ratios (Manejo seguro de division por cero)
    UR = expected_assembled_components / expected_total_components_in if expected_total_components_in > 0 else 0
    ISR = expected_inventory_held / expected_assembled_components if expected_assembled_components > 0 else 0
    VWAP = expected_revenue / expected_total_assembled_products if expected_total_assembled_products > 0 else 0
    FR = expected_total_assembled_products / expected_real_demand if expected_real_demand > 0 else 0

    # 3. Output Global unificado
    # print("\n" + "="*50)
    # print("RESUMEN GLOBAL DE DESEMPENO (Expected Values)")
    # print("="*50)
    # print(f"Fill Rate Global (FR)             : {FR * 100:.2f} %")
    # print(f"Utilizacion de Componentes (UR)   : {UR * 100:.2f} %")
    # print(f"Ratio Inventario/Produccion (ISR) : {ISR:.4f} unid_inv/unid_prod")
    # print(f"Precio Medio Ponderado (VWAP)     : $ {VWAP:.2f}")
    # print("="*50 + "\n")
    
    return {'FR': FR, 'UR': UR, 'ISR': ISR, 'VWAP': VWAP}

def calculate_global_kpis(x_val, price_eff, y_val, I_val, A, mult, add, a, b, pi, I0):
    """
    Calcula y retorna los KPIs globales (Expected Values) del modelo estocástico.
    
    Inputs:
    -----------
    x_val     : ndarray (comp, time, scenarios) - Componentes comprados
    price_eff : ndarray (prod, time, scenarios) - Precios efectivos asignados
    y_val     : ndarray (prod, time, scenarios) - Productos ensamblados
    I_val     : ndarray (comp, time, scenarios) - Inventario de componentes
    A         : list o ndarray (comp, prod)     - Matriz Bill of Materials
    mult      : list/ndarray (scenarios, time)  - Factor multiplicativo de demanda (epsilon)
    add       : list/ndarray (scenarios, prod, time) - Factor aditivo de demanda (delta)
    a, b      : escalares - Parámetros de la demanda lineal
    pi        : list/ndarray (scenarios,)       - Probabilidades de cada escenario
    
    Outputs:
    --------
    dict con los KPIs: FR, UR, ISR, VWAP
    """
    # 0. Acondicionamiento de dimensiones y tipos
    prod, time, scenarios = y_val.shape
    A = np.array(A)
    pi = np.array(pi)
    ypsilon = np.array(mult)
    delta = np.array(add)

    # 1. Reconstrucción vectorizada de la demanda real (D_eff)
    # Replicamos la misma lógica segura implementada previamente en fulfillment.py
    D_eff = np.zeros((prod, time, scenarios))
    for j in range(prod):
        # Transponemos para que las operaciones calcen con (scenarios, time)
        delta_j = np.array([[delta[s][j][t] for t in range(time)] for s in range(scenarios)])
        D_eff[j] = (ypsilon * (a - b * price_eff[j].T) + delta_j).T
        
    D_eff = np.clip(D_eff, 0.0, None)  # Previene demandas negativas por precios muy altos

    # 2. Agrupación y cálculo de Valores Esperados (Expected Values)
    # Las sumatorias colapsan primero en prod y time (axis=(0,1)), dejando un array 1D
    # de tamaño (scenarios,). Luego hacemos el producto punto con pi.
    
    # a) Productos Ensamblados Esperados
    expected_total_assembled_products = np.sum(y_val.sum(axis=(0, 1)) * pi)
    
    # b) Demanda Real Esperada
    expected_real_demand = np.sum(D_eff.sum(axis=(0, 1)) * pi)
    
    # c) Ingresos Totales Esperados (Revenue)
    revenue_matrix = y_val * price_eff
    expected_revenue = np.sum(revenue_matrix.sum(axis=(0, 1)) * pi)
    
    # d) Componentes Ensamblados Esperados
    # Multiplicación matricial (comp, prod) x (prod, time, scenarios) -> (comp, time, scenarios)
    comps_assembled = np.tensordot(A, y_val, axes=([1], [0]))
    expected_assembled_components = np.sum(comps_assembled.sum(axis=(0, 1)) * pi)
    
    # e) Inventario Mantenido Esperado
    expected_inventory_held = np.sum(I_val.sum(axis=(0, 1)) * pi)
    
    # f) Total Componentes Ingresados
    expected_initial_inventory = np.sum(I0)
    expected_total_components_in = (expected_initial_inventory + np.sum(x_val.sum(axis=(0, 1)) * pi)
)

    # 3. Cálculo de los Ratios (KPIs)
    FR = expected_total_assembled_products / expected_real_demand if expected_real_demand > 0 else 1.0
    UR = expected_assembled_components / expected_total_components_in if expected_total_components_in > 0 else 0.0
    ISR = expected_inventory_held / expected_assembled_components if expected_assembled_components > 0 else 0.0
    VWAP = expected_revenue / expected_total_assembled_products if expected_total_assembled_products > 0 else 0.0

    return FR, UR, ISR, VWAP