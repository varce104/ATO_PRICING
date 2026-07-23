def calculate_global_kpis(vars_dict, params, S, T, J, I, P):
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
                demand_val = params['epsilon'][t, s] * (params['a'] - params['b'] * selected_price) + params['delta'][j, t, s]
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