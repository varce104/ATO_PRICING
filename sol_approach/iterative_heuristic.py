import gurobipy as gp
from gurobipy import GRB
import numpy as np
import random
from itertools import product
from models.affine_funct_app import MS_linear_affine
from models.linealization_prop import MS_linear
from output_config.lambda_export import extract_solution_arrays_affine_w



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
                    # Inventario al inicio de t = final del período t-1
                    val = I_fixed.get((i, t - 1, s), 0.0) if t > 0 else 0.0
                    phi[s][t].append(val)
    return phi


def revenue_max(prod, stages, scenarios, pr, price, D_term, pi, extended_phi, K_features, y_fixed):
    import time
    m_rev = gp.Model("Revenue_Max")
    m_rev.setParam('OutputFlag', 1)

    rho = m_rev.addVars(prod, stages, pr, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="rho")
    Gamma = m_rev.addVars(prod, stages, pr, K_features, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="Gamma")
    lambda_w = m_rev.addVars(prod, stages, pr, scenarios, vtype=GRB.CONTINUOUS, lb=0, name="lambda_w")

    y_bar = m_rev.addVars(prod, stages, pr, scenarios, vtype=GRB.CONTINUOUS, lb=0, name="r")

    obj = gp.quicksum(pi[s] * price[p] * y_bar[j,t,p,s] for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios)))
    m_rev.setObjective(obj, GRB.MAXIMIZE)

    for j, t in product(range(prod), range(stages)):
        m_rev.addConstr(gp.quicksum(rho[j, t, p] for p in range(pr)) == 1)
        for q in range(K_features):
            m_rev.addConstr(gp.quicksum(Gamma[j, t, p, q] for p in range(pr)) == 0)

    for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios)):
        gamma_phi = gp.quicksum(Gamma[j, t, p, q] * extended_phi[s][t][q] for q in range(K_features))
        m_rev.addConstr(lambda_w[j, t, p, s] == rho[j, t, p] + gamma_phi)
        
        m_rev.addConstr(y_bar[j, t, p, s] <= lambda_w[j, t, p, s] * D_term[j, t, p, s])

    for j, t, s in product(range(prod), range(stages), range(scenarios)):
        m_rev.addConstr(gp.quicksum(y_bar[j, t, p, s] for p in range(pr)) == y_fixed[j, t, s])

    t0 = time.time()
    m_rev.optimize()
    t1 = time.time()
    
    if m_rev.status != GRB.OPTIMAL:
        raise ValueError("El modelo de Revenue Maximization es infactible.")
        
    lambda_val = {(j, t, p, s): lambda_w[j, t, p, s].X
                 for j, t, p, s in product(range(prod), range(time), range(pr), range(scenarios))}
    
    return m_rev.ObjVal, lambda_val, t1 - t0



def Iter_policy(seed, stages, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi, branching_structure, I0, max_iter=15, tol=1e-3):
    comp, prod, pr = len(A), len(A[0]), len(price)
    import time

    K_features = (stages - 1) * (1 + prod) + comp

    D_term = {}
    for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios)):
        D_term[j, t, p, s] = ypsilon[s][t] * (a - b * price[p]) + delta[s][j][t]

    print("\n--- PASO 1: MS_linear_affine inicial (I = 0 en phi) ---\n")
    phi_init = build_phi(ypsilon, delta, stages, scenarios, prod, comp, I_fixed=None)

    m_af, _, lam_w, y_af, I_af, _, _, _, _ = MS_linear_affine(
        seed, stages, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi, branching_structure, I0, phi_init, K_features)

    m_af.setParam('OutputFlag', 1)

    t0 = time.time()
    m_af.optimize()
    solve_time = time.time() - t0

    if m_af.status != GRB.OPTIMAL:
        raise ValueError("\nMS_linear_affine inicial no es optimo.\n")

    # Extraer I e y de MS_linear_affine
    I_fixed = {(i, t, s): I_af[i, t, s].X 
               for i, t, s in product(range(comp), range(stages), range(scenarios))}
    y_fixed = {(j, t, s): y_af[j, t, s].X 
               for j, t, s in product(range(prod), range(stages), range(scenarios))}

    best_obj = -np.inf
    m_lin = None

    for iteration in range(1, max_iter + 1):
        print(f"\n--- ITERACION {iteration} ---\n")

        # Paso A: construir phi con inventario actual y resolver revenue_max
        extended_phi = build_phi(ypsilon, delta, stages, scenarios, prod, comp, I_fixed)

        m_rev, lam_w_new, t_rev = revenue_max(
            prod, stages, scenarios, pr, price, D_term, pi, extended_phi, K_features, y_fixed)
        solve_time += t_rev

        print(f"\n >>> Modelo Revenue Max. Obj: {m_rev.objVal:.2f} <<<\n")

        m_inv, _, lambda_w, y_lin, I_lin, _, _, _, _ = MS_linear_affine(
            seed, stages, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi,
            branching_structure, I0, phi_init=extended_phi, K_feat=K_features)

        for j, t, p, q in product(range(prod), range(stages), range(pr), range(K_features)):
            lam_w_new[j, t, p, q].LB = lambda_w[j, t, p, q]
            lam_w_new[j, t, p, q].UB = lambda_w[j, t, p, q]

        m_lin.setParam('OutputFlag', 1)

        t0 = time.time()
        m_lin.optimize()
        solve_time += time.time() - t0

        if m_lin.status != GRB.OPTIMAL:
            print("\nMS_linear infactible/no óptimo.\n")
            break

        obj_lin = m_lin.objVal
        print(f"\n[*] MS_linear (operación): {obj_lin:.2f}\n")

        if (obj_lin - best_obj) <= tol:
            print(f"\n>>> Convergencia en iteración {iteration}. <<<\n")
            break
        best_obj = obj_lin

        # Paso D: actualizar I e y para la siguiente iteración
        I_fixed = {(i, t, s): I_lin[i, t, s].X 
                   for i, t, s in product(range(comp), range(stages), range(scenarios))}
        y_fixed = {(j, t, s): y_lin[j, t, s].X 
                   for j, t, s in product(range(prod), range(stages), range(scenarios))}
        
        print(f"\nEffective Solve Time after iteration {iteration}: {solve_time:.2f} seconds\n")

    return m_lin, best_obj, solve_time


