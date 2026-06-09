import numpy as np
import pandas as pd



def get_values(var_dict, shape):
    import numpy as np
    vals = np.empty(shape, dtype=float)
    for index in var_dict.keys():
        vals[index] = var_dict[index].X  
    return vals


def extract_solution_arrays(x_vars, w_vars, I_vals, y_vals, D_term, price, comp, prod, time, scenarios, pr):
    x_val = np.zeros((comp, time, scenarios))
    for i in range(comp):
        for t in range(time):
            for s in range(scenarios):
                x_val[i, t, s] = x_vars[i, t, s].X

    price_eff = np.zeros((prod, time, scenarios))
    for j in range(prod):
        for t in range(time):
            for s in range(scenarios):
                for p in range(pr):
                    if w_vars[j, t, p, s].X > 0.5:
                        price_eff[j, t, s] = price[p]

    I_vals = get_values(I_vals, (comp, time, scenarios))
    y_vals = get_values(y_vals, (prod, time, scenarios))

    d_vals = np.zeros((prod, time, scenarios))
    for j in range(prod):
        for t in range(time):
            for s in range(scenarios):
                d_vals[j, t, s] = D_term[j, t, s].getValue()

    return x_val, price_eff, I_vals, y_vals, d_vals


def export_solution_to_excel(filename, x_val, p_eff, y_val, I_val, d_val, time, scenarios, A):
    comp, prod = len(A), len(A[0])

    data_x, indices_x = [], []
    for s in range(scenarios):
        for i in range(comp):
            data_x.append([x_val[i, t, s] for t in range(time)])
            indices_x.append((f"Scenario_{s+1}", f"Comp_{i+1}"))
    df_x = pd.DataFrame(data_x, index=pd.MultiIndex.from_tuples(indices_x, names=["Scenario", "Component"]), columns=[f"T{t+1}" for t in range(time)])

    data_w, indices_w = [], []
    for s in range(scenarios):
        for j in range(prod):
            data_w.append([p_eff[j, t, s] for t in range(time)])
            indices_w.append((f"Scenario_{s+1}", f"Prod_{j+1}"))
    df_w = pd.DataFrame(data_w, index=pd.MultiIndex.from_tuples(indices_w, names=["Scenario", "Product"]), columns=[f"T{t+1}" for t in range(time)])

    data_y, indices_y = [], []
    for s in range(scenarios):
        for j in range(prod):
            data_y.append([y_val[j, t, s] for t in range(time)])
            indices_y.append((f"Scenario_{s+1}", f"Prod_{j+1}"))
    df_y = pd.DataFrame(data_y, index=pd.MultiIndex.from_tuples(indices_y, names=["Scenario", "Product"]), columns=[f"T{t+1}" for t in range(time)])

    data_I, indices_I = [], []
    for s in range(scenarios):
        for i in range(comp):
            data_I.append([I_val[i, t, s] for t in range(time)])
            indices_I.append((f"Scenario_{s+1}", f"Comp_{i+1}"))
    df_I = pd.DataFrame(data_I, index=pd.MultiIndex.from_tuples(indices_I, names=["Scenario", "Component"]), columns=[f"T{t+1}" for t in range(time)])

    data_d, indices_d = [], []
    for s in range(scenarios):
        for j in range(prod):
            data_d.append([d_val[j, t, s] for t in range(time)])
            indices_d.append((f"Scenario_{s+1}", f"Prod_{j+1}"))
    df_d = pd.DataFrame(data_d, index=pd.MultiIndex.from_tuples(indices_d, names=["Scenario", "Product"]), columns=[f"T{t+1}" for t in range(time)])

    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df_x.to_excel(writer, sheet_name='X_sol')
            df_w.to_excel(writer, sheet_name='W_sol')
            df_y.to_excel(writer, sheet_name='Y_sol')
            df_I.to_excel(writer, sheet_name='I_sol')
            df_d.to_excel(writer, sheet_name='D_sol')
        print(f"\n>> Resultados exportados a: {filename}")
    except Exception as e:
        print(f"Error al exportar Excel: {e}")