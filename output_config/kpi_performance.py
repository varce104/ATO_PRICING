import numpy as np

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


def calculate_financial_kpis(x_val, price_eff, y_val, I_val, pi, C, H):
    """
    Calcula los Valores Esperados (Expected Values) de las métricas financieras.
    """
    # Acondicionamiento de parámetros vectoriales para broadcasting
    C_np = np.array(C)[:, np.newaxis, np.newaxis] # Dimensión: (comp, 1, 1)
    H_np = np.array(H)[:, np.newaxis, np.newaxis] # Dimensión: (comp, 1, 1)
    pi_np = np.array(pi)
    
    # 1. Ingresos por Venta Esperados (Expected Revenue)
    revenue_matrix = y_val * price_eff
    expected_revenue = np.sum(revenue_matrix.sum(axis=(0, 1)) * pi_np)
    
    # 2. Costos de Compra Esperados (Expected Procurement Cost)
    procurement_matrix = x_val * C_np
    expected_procurement = np.sum(procurement_matrix.sum(axis=(0, 1)) * pi_np)
    
    # 3. Costos de Inventario Esperados (Expected Inventory Cost)
    inventory_matrix = I_val * H_np
    expected_inventory = np.sum(inventory_matrix.sum(axis=(0, 1)) * pi_np)
    
    return expected_revenue, expected_procurement, expected_inventory