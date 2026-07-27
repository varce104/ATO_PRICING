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

def export_affine_params_to_excel(filename, rho_vars, Gamma_vars, prod, time, pr, K_features):

    # Rho
    data_rho, idx_rho = [], []
    for j in range(prod):
        for p in range(pr):
            data_rho.append([rho_vars[j, t, p].X for t in range(time)])
            idx_rho.append((f"Prod_{j}", f"Price_{p}"))

    df_rho = pd.DataFrame(
        data_rho,
        index=pd.MultiIndex.from_tuples(idx_rho, names=["Product", "Price"]),
        columns=[f"T{t}" for t in range(time)]
    )

    # Gamma
    data_gamma, idx_gamma = [], []
    for j in range(prod):
        for p in range(pr):
            for q in range(K_features):
                data_gamma.append([Gamma_vars[j, t, p, q].X for t in range(time)])
                idx_gamma.append((f"Prod_{j}", f"Price_{p}", f"Feat_{q}"))

    df_gamma = pd.DataFrame(
        data_gamma,
        index=pd.MultiIndex.from_tuples(idx_gamma, names=["Product", "Price", "Feature"]),
        columns=[f"T{t}" for t in range(time)]
    )

    try:
        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            df_rho.to_excel(writer, sheet_name="Rho")
            df_gamma.to_excel(writer, sheet_name="Gamma")
        print(f"\n>> Parámetros afines exportados a: {filename}")
    except Exception as e:
        print(f"Error al exportar parámetros afines: {e}")

    