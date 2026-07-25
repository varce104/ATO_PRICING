import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import random
from itertools import product
from sol_approach.affine_funct_app import MS_linear_affine
from models.linealization_prop import MS_linear
from output_config.lambda_export import extract_solution_arrays_affine_w
from data.phi_features import build_phi as build_phi_base


def build_phi(mode, ypsilon, delta, time, scenarios, prod, comp, L=None, L_det=None, I_fixed=None):
    """
    Arma phi[s][t] usando el modo base seleccionado (eps / eps_delta / eps_lt / eps_delta_lt)
    y le agrega, al final de cada vector, el feature de inventario (uno por componente).
    Retorna (phi, K_features) — K_features incluye el aporte del inventario.
    """
    phi, K_base = build_phi_base(mode, time, scenarios, prod, comp, ypsilon, delta, L, L_det)

    for s in range(scenarios):
        for t in range(time):
            for i in range(comp):
                if I_fixed is None:
                    phi[s][t].append(0.0)
                else:
                    # Inventario al inicio de t = final del período t-1
                    val = I_fixed.get((i, t - 1, s), 0.0) if t > 0 else 0.0
                    phi[s][t].append(val)

    return phi, K_base + comp


def revenue_max(prod, stages, scenarios, pr, price, D_term, pi, extended_phi, K_features, y_fixed):
    """
    Revenue formulation that considers y = y[j,t,s]. Only constraints are about pricing
    """
    import time
    m_rev = gp.Model("Revenue_Max")
    m_rev.setParam('OutputFlag', 0)
    m_rev.setParam('BarHomogeneous', 1)

    rho     = m_rev.addVars(prod, stages, pr, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY)
    Gamma   = m_rev.addVars(prod, stages, pr, K_features, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY)
    lambda_w = m_rev.addVars(prod, stages, pr, scenarios, vtype=GRB.CONTINUOUS, lb=0)

    # y_fixed[j, t, s]
    m_rev.setObjective(gp.quicksum(pi[s] * price[p] * lambda_w[j, t, p, s] * y_fixed[j, t, s]
                    for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios))),GRB.MAXIMIZE)

    for j, t in product(range(prod), range(stages)):
        m_rev.addConstr(gp.quicksum(rho[j, t, p] for p in range(pr)) == 1)
        for q in range(K_features):
            m_rev.addConstr(gp.quicksum(Gamma[j, t, p, q] for p in range(pr)) == 0)

    for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios)):
        gamma_phi = gp.quicksum(Gamma[j, t, p, q] * extended_phi[s][t][q] for q in range(K_features))
        m_rev.addConstr(lambda_w[j, t, p, s] == rho[j, t, p] + gamma_phi)

    
    t0 = time.time()
    m_rev.optimize()
    t1 = time.time()

    if m_rev.status != GRB.OPTIMAL:
        raise ValueError(f"revenue_max no óptimo (status={m_rev.status})")

    lambda_val = {(j, t, p, s): lambda_w[j, t, p, s].X
                  for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios))}

    return m_rev.ObjVal, lambda_val, t1 - t0


def Iter_policy(seed, stages, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi,
                branching_structure, I0, max_iter=15, tol=1e-3, phi_mode="eps_delta"):
    comp, prod, pr = len(A), len(A[0]), len(price)
    import time

    D_term = {}
    for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios)):
        D_term[j, t, p, s] = ypsilon[s][t] * (a - b * price[p]) + delta[s][j][t]

    print("--------- Step 0: Solve MS_affine_approxiamtion (I=0 for features vector) ---------\n")
    phi_init, K_features = build_phi(phi_mode, ypsilon, delta, stages, scenarios, prod, comp, L, L_det, I_fixed=None)

    m_af, _, _, y_af, y_bar, I_af, _, _, _, _, _ = MS_linear_affine(
        seed, stages, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi, branching_structure, I0,
        phi_init, K_features)
    
    m_af.setParam('BarHomogeneous', 1)
    m_af.setParam('OutputFlag', 0)

    t0 = time.time()
    m_af.optimize()
    solve_time = time.time() - t0

    if m_af.status != GRB.OPTIMAL:
        return None, None, None, None

    # Extraer I e y de MS_linear_affine
    I_fixed = {(i, t, s): I_af[i, t, s].X for i, t, s in product(range(comp), range(stages), range(scenarios))} # Inventory
    y_fixed = {(j, t, s): y_af[j, t, s].X for j, t, s in product(range(prod), range(stages), range(scenarios))} # Production (y[j,t,s])
    # y_fixed = {(j, t, p, s): y_bar[j, t, p, s].X for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios))} # Production + revenue (y[j,t,p,s])

    best_obj = -np.inf

    Evol = []
    Evol.append({"iteration": 0, "obj": m_af.objVal, "solve_time": solve_time})

    for iteration in range(1, max_iter + 1):
        print(f"\n--- ITERATION {iteration} ---\n")

        # Paso A: construir phi con inventario actual y resolver revenue_max
        extended_phi, K_features = build_phi(phi_mode, ypsilon, delta, stages, scenarios, prod, comp, L, L_det, I_fixed)
        m_rev, lam_w_new, t_rev = revenue_max(prod, stages, scenarios, pr, price, D_term, pi, extended_phi, K_features, y_fixed)
        # m_rev, lam_w_new, t_rev = revenue_max_v0(prod, stages, scenarios, pr, price, D_term, pi, extended_phi, K_features, y_fixed)
        
        solve_time += t_rev
        
        print(f"\n >>> Max Revenue Model. Obj: {m_rev:.2f} <<<\n")

        m_inv, _, lambda_w, y_af, y_bar, I_af, _, _, _, _, _ = MS_linear_affine(
            seed, stages, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi, branching_structure, I0, 
                            extended_phi, K_features, lambda_fix=True)

        for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios)):
            lambda_w[j, t, p, s].LB = lam_w_new[j, t, p, s]
            lambda_w[j, t, p, s].UB = lam_w_new[j, t, p, s]

        m_inv.setParam('BarHomogeneous', 1)
        m_inv.setParam('OutputFlag', 0)

        t0 = time.time()
        m_inv.optimize()
        solve_time += time.time() - t0

        if m_inv.status != GRB.OPTIMAL:
            print("\nMS_linear_affine unfeasible.\n")
            return None, None, None, None

        obj_lin = m_inv.objVal
        print(f"\n >>> MS_linear_affine (ATO problem): {obj_lin:.2f} <<< \n")

        Evol.append({"iteration": iteration, "obj_pricing": m_rev, "obj_ato": obj_lin})


        if (obj_lin - best_obj) <= tol:
            print(f"\nConvergence achieved. iteration: {iteration}")
            break
            
        best_obj = obj_lin

        # Paso D: actualizar I e y para la siguiente iteración
        I_fixed = {(i, t, s): I_af[i, t, s].X for i, t, s in product(range(comp), range(stages), range(scenarios))} # Inventory
        y_fixed = {(j, t, s): y_af[j, t, s].X for j, t, s in product(range(prod), range(stages), range(scenarios))} # Production (y[j,t,s])
        # y_fixed = {(j, t, p, s): y_bar[j, t, p, s].X for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios))} # Production + revenue (y[j,t,p,s])
        
        print(f"\nEffective Solve Time after iteration {iteration}: {solve_time:.2f} seconds\n")


    # --- Evaluación final en MS_linear con política binarizada ---
    print("\n------ Final Eval: Lambda -> MS_linear ------\n")

    w_bin = np.zeros((prod, stages, pr, scenarios), dtype=float)

    for j, t, s in product(range(prod), range(stages), range(scenarios)):
        best_p = max(range(pr), key=lambda p: lambda_w[j, t, p, s].X)
        w_bin[j, t, best_p, s] = 1.0

    m, x_vars, w_vars, y_vars, I_vars, A, D_term = MS_linear(
        seed, stages, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi,
        branching_structure, I0)

    for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios)):
        w_vars[j, t, p, s].LB = w_bin[j, t, p, s]
        w_vars[j, t, p, s].UB = w_bin[j, t, p, s]
    
    return m, x_vars, w_vars, y_vars, I_vars, A, D_term, solve_time, iteration



