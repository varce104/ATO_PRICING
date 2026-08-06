import gurobipy as gp
from gurobipy import GRB


def local_search_first_improvement(ms_model, w_var, initial_w, J, T, S, P, scenario_groups):
    """
    ms_model: Modelo Gurobi del MS_linear.
    w_var: Diccionario de variables de decisión de precio w[j,t,p_idx,s].
    initial_w: Solución binaria inicial usando p_idx como llave.
    J, T, S: Listas de productos, periodos y escenarios.
    P: Lista con los valores nominales de los precios.
    scenario_groups: Diccionario de partición de escenarios (nodos).
    """
    
    # 1. Función auxiliar (Ahora itera sobre p_idx)
    def fix_prices(w_dict):
        for (j, t, p_idx, s), val in w_dict.items():
            w_var[j, t, p_idx, s].lb = val
            w_var[j, t, p_idx, s].ub = val

    # Evaluar solución inicial 
    fix_prices(initial_w)
    ms_model.setParam('OutputFlag', 0)
    ms_model.optimize()
    best_obj = ms_model.ObjVal
    
    current_w = initial_w.copy()
    improvement = True
    iteration = 0

    print(f"--- Iniciando Local Search (First Improvement) | Obj Inicial: {best_obj} ---")

    while improvement and iteration < 20:
        improvement = False
        iteration += 1
        
        for j in J:
            for t in T:
                for node_scenarios in scenario_groups[t]:
                    
                    s_repr = node_scenarios[0] 
                    
                    # CORRECCIÓN: Buscar directamente el índice (p_idx) seleccionado
                    current_p_idx = next(idx for idx in range(len(P)) if current_w[j, t, idx, s_repr] == 1)
                    
                    # Definir vecinos sumando o restando 1 al índice
                    neighbors = []
                    if current_p_idx > 0:
                        neighbors.append(current_p_idx - 1)
                    if current_p_idx < len(P) - 1:
                        neighbors.append(current_p_idx + 1)
                        
                    for new_p_idx in neighbors:
                        # Aplicar movimiento
                        for s in node_scenarios:
                            current_w[j, t, current_p_idx, s] = 0
                            current_w[j, t, new_p_idx, s] = 1
                            
                        # Re-optimizar
                        fix_prices(current_w)
                        ms_model.optimize()
                        
                        if ms_model.Status == GRB.OPTIMAL and ms_model.ObjVal > best_obj:
                            best_obj = ms_model.ObjVal
                            # Usamos P[idx] solo para imprimir el valor real amigable en consola
                            print(f"Iter {iteration}: Mejora! Obj: {best_obj} | Prod {j}, Per {t}, Precio {P[current_p_idx]} -> {P[new_p_idx]}")
                            improvement = True
                            break 
                            
                        else:
                            # Revertir
                            for s in node_scenarios:
                                current_w[j, t, current_p_idx, s] = 1
                                current_w[j, t, new_p_idx, s] = 0
                                
                    if improvement: break 
                if improvement: break 
            if improvement: break 

    print(f"--- Local Search finalizado | Obj Final: {best_obj} ---")
    fix_prices(current_w)
    ms_model.optimize()
    
    return current_w, best_obj