import gurobipy as gp
from gurobipy import GRB
import numpy as np
from itertools import product
from sol_approach.affine_funct_app import MS_linear_affine
from sol_approach.linealization_prop import MS_linear
from output_config.lambda_export import extract_solution_arrays_affine_w

def fix_variables(model_vars, fixed_vals, is_dict=True):
    if is_dict:
        for key, val in fixed_vals.items():
            model_vars[key].LB = val
            model_vars[key].UB = val
    else:
        it = np.nditer(fixed_vals, flags=['multi_index'])
        for val in it:
            model_vars[it.multi_index].LB = val
            model_vars[it.multi_index].UB = val


def iterative_pricing_inventory(seed, time_periods, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, 
                                pi, branching_structure, I0, max_iter=5, tol=1e-2):
    import time

    comp, prod, pr = len(A), len(A[0]), len(price)

    print("\n--------- Iteracion 0: Resolviendo Modelo Afin Inicial ---------\n")
    m_af, x_af, lam_w, y_af, I_af, _, D_af, rho, Gamma = MS_linear_affine(
        seed, time_periods, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi, branching_structure, I0)
    start = time.time()
    
    m_af.setParam('OutputFlag', 1)
    m_af.optimize()

    
    w_bin = extract_solution_arrays_affine_w(lam_w, prod, time_periods, scenarios, pr)
    
    

    for iteration in range(1, max_iter + 1):
        print(f"\n--------- Iteracion {iteration} ---------\n")
        
        m_lin, x_lin, w_lin, y_lin, I_lin, _, D_lin = MS_linear(
            seed, time_periods, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi, branching_structure, I0)
        
        fix_variables(w_lin, w_bin, is_dict=False)
        
        m_lin.setParam('OutputFlag', 1)
        m_lin.optimize()
        
        if m_lin.status != GRB.OPTIMAL:
            print("\n ##### Fase Inventario Infactible/No optima. Abortando #####\n")
            break
            
        obj_lin = m_lin.objVal

        print(f"\nObj Fase Inventario (Precio Fijo): {obj_lin}\n")
        
        x_fixed = {(i, t, s): x_lin[i, t, s].X for i in range(comp) for t in range(time_periods) for s in range(scenarios)}
        
        m_af, x_af, lam_w, y_af, _, _, _, rho, Gamma = MS_linear_affine(
            seed, time_periods, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi, branching_structure, I0)
        
        fix_variables(x_af, x_fixed, is_dict=True)
        
        m_af.setParam('OutputFlag', 1)
        m_af.optimize()
        
        if m_af.status != GRB.OPTIMAL:
            print("\n ##### Fase Precio Infactible/No óptima. Abortando #####\n")
            break
            
        obj_af = m_af.objVal

        print(f"\n--- Obj Fase Precio (Compras Fijas): {obj_af} ---\n")
        
        if abs(obj_af - best_obj) < tol:
            print(f"\n ------ Convergencia alcanzada. Iteracion {iteration} ------\n")
            break
            
        best_obj = obj_af
        
        w_bin = extract_solution_arrays_affine_w(lam_w, prod, time_periods, scenarios, pr)
    end = time.time()
    exec_time = end - start
    return m_lin, obj_lin, exec_time


def build_phi(ypsilon, delta, time, scenarios, prod, comp, I_fixed=None):
    phi = {}
    for s in range(scenarios):
        phi[s] = {}
        for t in range(time):
            phi[s][t] = []
            
            for tau in range(time - 1):
                if tau < t:
                    phi[s][t].append(ypsilon[s][tau])
                    for j in range(prod):
                        phi[s][t].append(delta[s][j][tau])
                else:
                    phi[s][t].append(0)
                    for j in range(prod):
                        phi[s][t].append(0)
                        
            for i in range(comp):
                if I_fixed is None:
                    phi[s][t].append(0.0)
                else:
                    phi[s][t].append(I_fixed.get((i, t, s), 0.0))         
    return phi


def revenue_maximization_model(prod, time, scenarios, pr, price, D_term, pi, extended_phi, K_features, y_fixed):
    m_rev = gp.Model("Revenue_Max")
    m_rev.setParam('OutputFlag', 1)

    rho = m_rev.addVars(prod, time, pr, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="rho")
    Gamma = m_rev.addVars(prod, time, pr, K_features, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="Gamma")
    lambda_w = m_rev.addVars(prod, time, pr, scenarios, vtype=GRB.CONTINUOUS, lb=0, name="lambda_w")

    r = m_rev.addVars(prod, time, pr, scenarios, vtype=GRB.CONTINUOUS, lb=0, name="r")

    obj = gp.quicksum(pi[s] * price[p] * r[j,t,p,s] for j, t, p, s in product(range(prod), range(time), range(pr), range(scenarios)))
    m_rev.setObjective(obj, GRB.MAXIMIZE)

    for j, t in product(range(prod), range(time)):
        m_rev.addConstr(gp.quicksum(rho[j, t, p] for p in range(pr)) == 1)
        for q in range(K_features):
            m_rev.addConstr(gp.quicksum(Gamma[j, t, p, q] for p in range(pr)) == 0)

    for j, t, p, s in product(range(prod), range(time), range(pr), range(scenarios)):
        gamma_dot = gp.quicksum(Gamma[j, t, p, q] * extended_phi[s][t][q] for q in range(K_features))
        m_rev.addConstr(lambda_w[j, t, p, s] == rho[j, t, p] + gamma_dot)
        
        m_rev.addConstr(r[j, t, p, s] <= lambda_w[j, t, p, s] * D_term[j, t, p, s])

    for j, t, s in product(range(prod), range(time), range(scenarios)):
        m_rev.addConstr(gp.quicksum(r[j, t, p, s] for p in range(pr)) == y_fixed[j, t, s])

    m_rev.optimize()
    
    if m_rev.status != GRB.OPTIMAL:
        raise ValueError("El modelo de Revenue Maximization es infactible.")
        
    return m_rev, lambda_w


def Iter_policy(seed, stages, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi, branching_structure, I0, max_iter=15, tol=1e-3):
    comp, prod, pr = len(A), len(A[0]), len(price)

    import time
    start = time.time()


    K_features = (stages - 1) * (1 + prod) + comp 
    
    D_term = {}
    for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios)):
        D_term[j, t, p, s] = ypsilon[s][t] * (a - b * price[p]) + delta[s][j][t]

    print("\n--- PASO 1: Modelo Afín Lineal Base (I = 0) ---")
    phi_init = build_phi(ypsilon, delta, stages, scenarios, prod, comp, I_fixed=None)
    
    m_af, x_af, lam_w, y_af, I_af, _, _, _, _ = MS_linear_affine(
        seed, stages, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi, branching_structure, I0, phi_init, K_features)
    
    m_af.setParam('OutputFlag', 1)
    m_af.optimize()
    
    w_bin = extract_solution_arrays_affine_w(lam_w, prod, stages, scenarios, pr)
    best_obj = -np.inf


    
    for iteration in range(1, max_iter + 1):
        print(f"\n--- ITERACIÓN {iteration} ---")
        
        m_lin, x_lin, w_lin, y_lin, I_lin, _, _ = MS_linear(
            seed, stages, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi, branching_structure, I0)
        
        for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios)):
            val = float(w_bin[j, t, p, s])
            w_lin[j, t, p, s].LB = val
            w_lin[j, t, p, s].UB = val
            
        m_lin.setParam('OutputFlag', 1)
        m_lin.optimize()
        
        obj_lin = m_lin.objVal
        print(f"[*] Evaluación en MS_linear (Operación): {obj_lin:.2f}")
        
        if (obj_lin - best_obj) <= tol:
            print(f">>> Convergencia alcanzada. Sin mejoras marginales. <<<")
            break
        best_obj = obj_lin
        
        # Extraer estados de inventario y decisiones de producción
        I_fixed = {(i, t, s): I_lin[i, t, s].X for i, t, s in product(range(comp), range(stages), range(scenarios))}
        y_fixed = {(j, t, s): y_lin[j, t, s].X for j, t, s in product(range(prod), range(stages), range(scenarios))}
        
        extended_phi = build_phi(ypsilon, delta, stages, scenarios, prod, comp, I_fixed)
        
        m_rev, lam_w_new = revenue_maximization_model(
            prod, stages, scenarios, pr, price, D_term, pi, extended_phi, K_features, y_fixed)
            
        print(f"[*] Modelo Revenue Max (Política) Resuelto: {m_rev.objVal:.2f}")
        
        w_bin = extract_solution_arrays_affine_w(lam_w_new, prod, stages, scenarios, pr)

    end = time.time()
    exec_time = end - start

    return m_lin, w_bin, exec_time