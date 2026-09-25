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



def build_and_solve_master(prod, stages, scenarios, pr, K_features, extended_phi, past_cuts_data):
    """
    Problema Maestro de Benders (Reemplaza a revenue_max).
    Único objetivo: Maximizar theta (estimador del beneficio real operativo del sistema).
    Se somete a los cortes de optimalidad almacenados en past_cuts_data.
    """
    import time
    m_master = gp.Model("Benders_Master")
    m_master.setParam('OutputFlag', 0)
    m_master.setParam('BarHomogeneous', 1)

    BIG_M = 1e9

    theta = m_master.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=BIG_M, name="theta")
    rho = m_master.addVars(prod, stages, pr, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="rho")
    Gamma = m_master.addVars(prod, stages, pr, K_features, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="Gamma")
    lambda_w = m_master.addVars(prod, stages, pr, scenarios, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="lambda_w")

    m_master.setObjective(theta, GRB.MAXIMIZE)

    # Restricciones de política afín
    for j, t in product(range(prod), range(stages)):
        m_master.addConstr(gp.quicksum(rho[j, t, p] for p in range(pr)) == 1)
        for q in range(K_features):
            m_master.addConstr(gp.quicksum(Gamma[j, t, p, q] for p in range(pr)) == 0)

    for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios)):
        gamma_phi = gp.quicksum(Gamma[j, t, p, q] * extended_phi[s][t][q] for q in range(K_features))
        m_master.addConstr(lambda_w[j, t, p, s] == rho[j, t, p] + gamma_phi)

    # Inyección de los Cortes de Benders históricos
    for idx, cut_data in enumerate(past_cuts_data):
        if cut_data[0] == 'OPT':
            _, Z_k, lam_k, RC_k = cut_data
            cut_expr = Z_k + gp.quicksum(
                RC_k[(j, t, p, s)] * (lambda_w[j, t, p, s] - lam_k[(j, t, p, s)])
                for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios))
            )
            m_master.addConstr(theta <= cut_expr, name=f"BendersOptCut_{idx}")
        elif cut_data[0] == 'FEAS':
            _, f_const, Farkas_lam = cut_data
            cut_expr = f_const + gp.quicksum(
                Farkas_lam[(j, t, p, s)] * lambda_w[j, t, p, s]
                for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios))
            )
            m_master.addConstr(cut_expr >= 0, name=f"BendersFeasCut_{idx}")

    t0 = time.time()
    m_master.optimize()
    t1 = time.time()

    if m_master.status != GRB.OPTIMAL:
        print(f" Master problem status {m_master.status}.")
        return None, None, None

    lam_new = {(j, t, p, s): lambda_w[j, t, p, s].X for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios))}
    
    return theta.X, lam_new, (t1 - t0)




def Iter_policy(seed, stages, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi,
                branching_structure, I0, max_iter=15, tol=1e-3, phi_mode="eps_delta"):
    comp, prod, pr = len(A), len(A[0]), len(price)
    import time
    solve_time = 0
    max_iter=25

    print("--------- Heuristic Benders Initialization ---------\n")
    
    # 1. Calcular el Profit Estimado para el punto inicial
    # Costo base estimado de componentes por producto
    C_prod = [sum(A[i][j] * C[i] for i in range(comp)) for j in range(prod)]
    # Precio teórico óptimo P*
    P_star = [(a + b * C_prod[j]) / (2 * b) for j in range(prod)]
    
    lam_current = {}
    for j in range(prod):
        # Encontrar el índice del precio discreto más cercano a P_star[j]
        best_p_idx = min(range(pr), key=lambda p_idx: abs(price[p_idx] - P_star[j]))
        for t, p, s in product(range(stages), range(pr), range(scenarios)):
            lam_current[(j, t, p, s)] = 1.0 if p == best_p_idx else 0.0

    phi_current, K_features = build_phi(phi_mode, ypsilon, delta, stages, scenarios, prod, comp, L, L_det, I_fixed=None)

    # 2. Inicializar el Subproblema ATO (se mantiene en memoria)
    m_inv, x_af, lambda_w_sub, y_af, y_bar, I_af, _, D_term, _, _, _ = MS_linear_affine(
        seed, stages, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi, branching_structure, I0, 
        phi_init=phi_current, K_feat=K_features, lambda_fix=True, phi_mode=phi_mode)

    m_inv.setParam('BarHomogeneous', 1)
    m_inv.setParam('OutputFlag', 0)

    # Configurar subproblema para permitir extracción de rayos de Farkas
    m_inv.setParam('InfUnboundedInfo', 1)

    # Crear restricciones de fijación explícitas (antes de iniciar el bucle)
    fix_constrs = {}
    for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios)):
        fix_constrs[(j, t, p, s)] = m_inv.addConstr(lambda_w_sub[j, t, p, s] == 0.0, name=f"fix_lam_{j}_{t}_{p}_{s}")
    fix_constrs_set = set(fix_constrs.values())

    best_obj = -np.inf
    past_cuts_data = []

    for iteration in range(1, max_iter + 1):
        print(f"\n--- BENDERS ITERATION {iteration} ---\n")

        # --- A. SUBPROBLEMA ATO ---
        # Actualizar el lado derecho (RHS) de las restricciones en lugar de los bounds
        for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios)):
            fix_constrs[(j, t, p, s)].RHS = lam_current[(j, t, p, s)]

        t0 = time.time()
        m_inv.optimize()
        solve_time += time.time() - t0

        if m_inv.status == GRB.INFEASIBLE:
            print(" >>> ATO Subproblem Infeasible. Generating Feasibility Cut <<< ")
            # 1. Extraer rayo dual para las variables acopladas
            Farkas_lambda = {k: c.FarkasDual for k, c in fix_constrs.items()}
            # 2. Calcular la constante del rayo dual para todas las demás restricciones
            farkas_constant = sum(c.FarkasDual * c.RHS for c in m_inv.getConstrs() if c not in fix_constrs_set)
            
            past_cuts_data.append(('FEAS', farkas_constant, Farkas_lambda))
            # Omitimos la convergencia de optimalidad porque no hay Z_ato válido

        elif m_inv.status == GRB.OPTIMAL:
            Z_ato = m_inv.objVal
            print(f" >>> ATO Subproblem Obj (Z_ato): {Z_ato:.2f} <<< ")

            if iteration > 1:
                gap = theta_val - Z_ato
                print(f"     Master Theta: {theta_val:.2f} | Gap: {gap:.2f}")
                if gap <= tol or (Z_ato - best_obj <= tol and gap < 1.0):
                    print(f"\nConvergence achieved at iteration: {iteration}")
                    break
            
            best_obj = max(best_obj, Z_ato)

            # Extraer las variables duales (.Pi) de las restricciones de fijación
            RC_lambda = {k: c.Pi for k, c in fix_constrs.items()}
            # Usar .copy() para no sobreescribir las referencias guardadas
            past_cuts_data.append(('OPT', Z_ato, lam_current.copy(), RC_lambda))
        else:
            print(f"\nStatus desconocido ({m_inv.status}). Abortando.\n")
            return None, None, None, None, None, None, None, None, None

        # --- B. PROBLEMA MAESTRO DE PRECIOS ---
        I_fixed = {(i, t, s): I_af[i, t, s].X for i, t, s in product(range(comp), range(stages), range(scenarios))} if m_inv.status == GRB.OPTIMAL else None
        
        # Continuar con el resto de la lógica del maestro...

        # --- B. PROBLEMA MAESTRO DE PRECIOS ---
        # Extraer el inventario para actualizar phi
        I_fixed = {(i, t, s): I_af[i, t, s].X for i, t, s in product(range(comp), range(stages), range(scenarios))}
        phi_current, _ = build_phi(phi_mode, ypsilon, delta, stages, scenarios, prod, comp, L, L_det, I_fixed)

        # Resolver el maestro con los cortes acumulados y el nuevo phi
        theta_val, lam_current, t_master = build_and_solve_master(
            prod, stages, scenarios, pr, K_features, phi_current, past_cuts_data)

        if theta_val is None:
            print("\nFallo en el maestro.")
            return None, None, None, None, None, None, None, None, None
        
        solve_time += t_master

        print(f" >>> Master Problem Solved. Theta projected limit: {theta_val:.2f} <<< ")
        print(f"Effective Solve Time after iteration {iteration}: {solve_time:.2f} seconds")

    # --- Evaluación final en MS_linear con política binarizada ---
    print("\n------ Final Eval: Lambda -> MS_linear ------\n")

    w_bin = np.zeros((prod, stages, pr, scenarios), dtype=float)

    for j, t, s in product(range(prod), range(stages), range(scenarios)):
        best_p = max(range(pr), key=lambda p: lam_current[(j, t, p, s)])
        w_bin[j, t, best_p, s] = 1.0

    m, x_vars, w_vars, y_vars, I_vars, A, D_term = MS_linear(
        seed, stages, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi,
        branching_structure, I0)

    for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios)):
        w_vars[j, t, p, s].LB = w_bin[j, t, p, s]
        w_vars[j, t, p, s].UB = w_bin[j, t, p, s]
    
    return m, x_vars, w_vars, y_vars, I_vars, A, D_term, solve_time, iteration
