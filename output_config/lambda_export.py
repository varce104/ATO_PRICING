import numpy as np
import pandas as pd



def extract_solution_arrays_affine_w(lambda_vars, prod, time, scenarios, pr, Model=False):
    w_simulada = np.zeros((prod, time, pr, scenarios))
    
    for j in range(prod):
        for t in range(time):
            for s in range(scenarios):
                s_ref = 0 if Model == "TS_linear_affine" else s
                lambda_values = [lambda_vars[j, t, p, s_ref].X for p in range(pr)]
                best_p = np.argmax(lambda_values)
                w_simulada[j, t, best_p, s] = 1
    return w_simulada


def export_solution_to_excel_affine(filename, w_sim, time, scenarios, pr, A, Model):
    comp, prod = len(A), len(A[0])

    w_bin = extract_solution_arrays_affine_w(w_sim, prod, time, scenarios, pr, Model)
    data_w, indices_w = [], []
    for s in range(scenarios):
        for j in range(prod):
            for p in range(pr):
                data_w.append([w_bin[j, t, p, s] for t in range(time)])
                indices_w.append((f"Scenario_{s}", f"Prod_{j}", f"Price_{p}"))
    df_w = pd.DataFrame(data_w, index=pd.MultiIndex.from_tuples(indices_w, names=["Scenario", "Product", "Price"]), columns=[f"T{t}" for t in range(time)])

    data_lam, indices_lam = [], []
    for s in range(scenarios):
        for j in range(prod):
            for p in range(pr):
                s_ref = 0 if Model == "TS_linear_affine" else s
                data_lam.append([w_sim[j, t, p, s_ref].X for t in range(time)])
                indices_lam.append((f"Scenario_{s}", f"Prod_{j}", f"Price_{p}"))
    df_lam = pd.DataFrame(data_lam, index=pd.MultiIndex.from_tuples(indices_lam, names=["Scenario", "Product", "Price"]), columns=[f"T{t}" for t in range(time)])

    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df_w.to_excel(writer, sheet_name='W_sol')
            df_lam.to_excel(writer, sheet_name='Lambda_raw')
        print(f"\n>> Resultados exportados a: {filename}")
    except Exception as e:
        print(f"Error al exportar Excel: {e}")




def fix_w_from_lambda(m, w_vars, w_rec, prod, time, pr, scenarios, epsilon=0.05):
    for j in range(prod):
        for t in range(time):
            for p in range(pr):
                for s in range(scenarios):
                    idx = (f"Scenario_{s}", f"Prod_{j}", f"Price_{p}")
                    col = f"T{t}"
                    
                    val = float(w_rec.loc[idx, col])

                    if val >= 1 - epsilon:
                        w_vars[j, t, p, s].LB = 1.0
                        w_vars[j, t, p, s].UB = 1.0
                        
                    elif val <= epsilon:
                        w_vars[j, t, p, s].LB = 0.0
                        w_vars[j, t, p, s].UB = 0.0
                        
                    else:
                        # Valor fraccionario (ej: 0.4 y 0.6 para dos precios distintos)
                        # No modificamos LB ni UB. Gurobi la mantendrá como GRB.BINARY
                        # y decidirá el valor óptimo durante el Branch & Bound.
                        pass 
                    
                    # w_vars[j, t, p, s].LB = val
                    # w_vars[j, t, p, s].UB = val
                    
    return w_rec


def fix_w_from_lambda_partial(m, w_vars, w_sim, prod, time, pr, scenarios, epsilon=0.05):
    """
    Fija las variables w solo si el valor de lambda_w es claramente 0 o 1.
    Deja libres (como binarias) aquellas que cayeron en valores fraccionarios.
    """
    for j in range(prod):
        for t in range(time):
            for p in range(pr):
                for s in range(scenarios):
                
                    lambda_val = w_sim.loc[(j, t, p), f"Scen_{s}"] 
                    
                    if lambda_val >= 1 - epsilon:
                        w_vars[j, t, p, s].LB = 1.0
                        w_vars[j, t, p, s].UB = 1.0
                        
                    elif lambda_val <= epsilon:
                        w_vars[j, t, p, s].LB = 0.0
                        w_vars[j, t, p, s].UB = 0.0
                        
                    else:
                        # Valor fraccionario (ej: 0.4 y 0.6 para dos precios distintos)
                        # No modificamos LB ni UB. Gurobi la mantendrá como GRB.BINARY
                        # y decidirá el valor óptimo durante el Branch & Bound.
                        pass 
                        
    m.update()