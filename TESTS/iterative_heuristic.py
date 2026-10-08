import os
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from itertools import product
from sol_approach.affine_funct_app import MS_linear_affine
from models.linealization_prop import MS_linear
from output_config.lambda_export import extract_solution_arrays_affine_w
from data.phi_features import build_phi as build_phi_base


def plot_benders_bounds(history, tag, out_dir="figures/benders_heuristic"):
    os.makedirs(out_dir, exist_ok=True)
    it       = [h["iteration"] for h in history]
    lb       = [h["LB"] for h in history]
    ub       = [h["UB"] for h in history]
    t_sub    = [h["t_sub"] for h in history]
    t_master = [h["t_master"] for h in history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1.5]})

    ax1.plot(it, ub, marker=".", color="#742626", label="UB (master)")
    ax1.plot(it, lb, marker=".", color="#3e944e", label="LB (best incumbent)")
    ax1.fill_between(it, lb, ub, alpha=0.15, color="gray")
    ax1.set_ylabel("Objective value")
    ax1.set_title("Heuristic Benders: bounds evolution")
    ax1.grid(linestyle="--", alpha=0.4)
    ax1.legend()

    ax2.bar(it, t_sub, color="#3e944e", alpha=0.5, label="Subproblem")
    ax2.bar(it, t_master, bottom=t_sub, color="#742626", alpha=0.5, label="Master")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Solve time (s)")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.legend()

    plt.tight_layout()
    path = f"{out_dir}/bounds_{tag}.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f">> Gráfico de cotas guardado en: {path}")

def build_phi(mode, ypsilon, delta, time, scenarios, prod, comp, L=None, L_det=None, I_fixed=None):
    """
    Arma phi[s][t] usando el modo base seleccionado (eps / eps_delta / eps_lt / eps_delta_lt)
    y le agrega, al final de cada vector, el feature de inventario (uno por componente).
    Retorna (phi, K_features) — K_features incluye el aporte del inventario.
    """
    phi, K_base = build_phi_base(mode, time, scenarios, prod, comp, ypsilon, delta, L, L_det)

    # for s in range(scenarios):
    #     for t in range(time):
    #         for i in range(comp):
    #             if I_fixed is None:
    #                 phi[s][t].append(0.0)
    #             else:
    #                 # Inventario al inicio de t = final del período t-1
    #                 val = I_fixed.get((i, t - 1, s), 0.0) if t > 0 else 0.0
    #                 phi[s][t].append(val)

    # return phi, K_base + comp
    return phi, K_base



def build_and_solve_master_OLD(prod, stages, scenarios, pr, K_features, extended_phi, past_cuts_data):
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
    for idx, (Z_k, lam_k, RC_k) in enumerate(past_cuts_data):
        # NOTA: RC_k ya contiene la escala de probabilidad pi[s] porque Gurobi 
        # extrae el dual directamente de la función objetivo esperada del subproblema ATO.
        cut_expr = Z_k + gp.quicksum(
            RC_k[(j, t, p, s)] * (lambda_w[j, t, p, s] - lam_k[(j, t, p, s)])
            for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios))
        )
        m_master.addConstr(theta <= cut_expr, name=f"BendersCut_{idx}")

    t0 = time.time()
    m_master.optimize()
    t1 = time.time()

    if m_master.status != GRB.OPTIMAL:
        print(f" Master problem status {m_master.status}.")
        return None, None, None

    lam_new = {(j, t, p, s): lambda_w[j, t, p, s].X for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios))}
    
    return theta.X, lam_new, (t1 - t0)

def Iter_policy_OLD(seed, stages, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi,
                branching_structure, I0, max_iter=50, tol=1e-4, phi_mode="eps_delta", plot=True):
    comp, prod, pr = len(A), len(A[0]), len(price)
    import time
    solve_time = 0

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

    best_obj = -np.inf
    past_cuts_data = []

    iteration_stopped = 0 # Variable para guardar en qué iteración frenó si falla

    history = []
    ub=np.inf
    t_master = 0

    for iteration in range(1, max_iter + 1):
        print(f"\n--- BENDERS ITERATION {iteration} ---\n")
        iteration_stopped = iteration
        # --- A. SUBPROBLEMA ATO ---
        # Fijar los lambdas actuales en las cotas del subproblema
        for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios)):
            lambda_w_sub[j, t, p, s].LB = lam_current[(j, t, p, s)]
            lambda_w_sub[j, t, p, s].UB = lam_current[(j, t, p, s)]

        t0 = time.time()
        m_inv.optimize()
        t_sub = time.time() - t0
        solve_time += time.time() - t0

        if m_inv.status != GRB.OPTIMAL:
            print("\nSubproblema ATO infactible o fallido. Abortando iteraciones y evaluando última solución factible.\n")
            break # Usará el lam_current con el que intentó resolver

        Z_ato = m_inv.objVal
        print(f" >>> ATO Subproblem Obj (Z_ato): {Z_ato:.2f} <<< ")

        # Verificar convergencia evaluando el Gap del Maestro en iteraciones previas
        if iteration > 1:
            gap = theta_val - Z_ato
            print(f"     Master Theta: {theta_val:.2f} | Gap: {gap:.2f}")
            if gap <= tol or (Z_ato - best_obj <= tol and gap < 1.0):
                print(f"\nConvergence achieved at iteration: {iteration}")
                break
        
        best_obj = max(best_obj, Z_ato)

        # Extraer los Reduced Costs (gradientes/duales de lambda)
        RC_lambda = {}
        for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios)):
            RC_lambda[(j, t, p, s)] = lambda_w_sub[j, t, p, s].RC

        # Guardar la información para el corte de esta iteración
        past_cuts_data.append((Z_ato, lam_current, RC_lambda))

        # --- B. PROBLEMA MAESTRO DE PRECIOS ---
        # Extraer el inventario para actualizar phi
        I_fixed = {(i, t, s): I_af[i, t, s].X for i, t, s in product(range(comp), range(stages), range(scenarios))}
        phi_current, _ = build_phi(phi_mode, ypsilon, delta, stages, scenarios, prod, comp, L, L_det, I_fixed)

        # Resolver el maestro con los cortes acumulados y el nuevo phi
        theta_val, lam_new, t_master = build_and_solve_master(
            prod, stages, scenarios, pr, K_features, phi_current, past_cuts_data)

        if theta_val is None:
            print("\nFallo en el maestro. Abortando iteraciones y evaluando última solución factible.\n")
            break # El lam_current queda intacto con los valores de la iteración previa

        lam_current = lam_new # Actualizamos de forma segura solo si resolvió
        solve_time += t_master
        ub = min(ub, theta_val)

        history.append({
            "iteration": iteration,
            "LB": best_obj,
            "UB": ub,
            "t_sub": t_sub,
            "t_master": t_master
        })

        rel_gap = (theta_val - Z_ato) / max(1.0, abs(Z_ato))
        print(f"Master Theta: {theta_val:.2f} | rel gap: {rel_gap:.4%}")
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

    if plot and history:
        tag = f"c{comp}_p{prod}_T{stages}_S{scenarios}_seed{seed}_{time.strftime('%H%M%S')}"
        plot_benders_bounds(history, tag)
    
    return m, x_vars, w_vars, y_vars, I_vars, A, D_term, solve_time, iteration_stopped



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

    # 1. Mantienes tu variable afín continua (acotada entre 0 y 1)
    lambda_w = m_master.addVars(prod, stages, pr, scenarios, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="lambda_w")
    # 2. Creas tu nueva variable BINARIA (lambda barra)
    lambda_bar = m_master.addVars(prod, stages, pr, scenarios, vtype=GRB.BINARY, name="lambda_bar")

    m_master.setObjective(theta, GRB.MAXIMIZE)

    # Restricciones de binarización de lambda
    for j in range(prod):
        for t in range(stages):
            for s in range(scenarios):
                
                # a) Solo un precio puede ser 1
                m_master.addConstr(gp.quicksum(lambda_bar[j, t, p, s] for p in range(pr)) == 1)
                
                # b) Restricción Big-M: el lambda_bar == 1 obliga a que ese lambda_w sea el máximo
                for p in range(pr):
                    for p_prime in range(pr):
                        if p != p_prime:
                            m_master.addConstr(lambda_w[j, t, p, s] >= lambda_w[j, t, p_prime, s] - (1 - lambda_bar[j, t, p, s]))

    # Restricciones de política afín
    for j, t in product(range(prod), range(stages)):
        m_master.addConstr(gp.quicksum(rho[j, t, p] for p in range(pr)) == 1)
        for q in range(K_features):
            m_master.addConstr(gp.quicksum(Gamma[j, t, p, q] for p in range(pr)) == 0)

    # Restricciones de lambda_w en función de rho y Gamma
    for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios)):
        gamma_phi = gp.quicksum(Gamma[j, t, p, q] * extended_phi[s][t][q] for q in range(K_features))
        m_master.addConstr(lambda_w[j, t, p, s] == rho[j, t, p] + gamma_phi)

    # INYECCIÓN DE CORTES USANDO lambda_bar
    for idx, (Z_k, lam_bar_k, RC_k) in enumerate(past_cuts_data):
        cut_expr = Z_k + gp.quicksum(
            RC_k[(j, t, p, s)] * (lambda_bar[j, t, p, s] - lam_bar_k[(j, t, p, s)])
            for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios))
        )
        m_master.addConstr(theta <= cut_expr, name=f"BendersCut_{idx}")

    t0 = time.time()
    m_master.optimize()
    t1 = time.time()

    if m_master.status != GRB.OPTIMAL:
        print(f" Master problem status {m_master.status}.")
        return None, None, None
    
    # Extraer la nueva política binaria para el subproblema (redondeo de seguridad por tolerancia de solver)
    lam_bar_new = {(j, t, p, s): round(lambda_bar[j, t, p, s].X) 
                   for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios))}
    
    return theta.X, lam_bar_new, (t1 - t0)

def Iter_policy(seed, stages, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi,
                branching_structure, I0, max_iter=5, tol=1e-4, phi_mode="eps_delta", plot=True):
    comp, prod, pr = len(A), len(A[0]), len(price)
    import time
    solve_time = 0

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

    best_obj = -np.inf
    past_cuts_data = []

    iteration_stopped = 0 # Variable para guardar en qué iteración frenó si falla

    history = []
    ub=np.inf
    t_master = 0

    for iteration in range(1, max_iter + 1):
        print(f"\n--- BENDERS ITERATION {iteration} ---\n")
        iteration_stopped = iteration
        # --- A. SUBPROBLEMA ATO ---
        # Fijar los lambdas actuales en las cotas del subproblema
        for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios)):
            lambda_w_sub[j, t, p, s].LB = lam_current[(j, t, p, s)]
            lambda_w_sub[j, t, p, s].UB = lam_current[(j, t, p, s)]

        t0 = time.time()
        m_inv.optimize()
        t_sub = time.time() - t0
        solve_time += time.time() - t0

        if m_inv.status != GRB.OPTIMAL:
            print("\nSubproblema ATO infactible o fallido. Abortando iteraciones y evaluando última solución factible.\n")
            break # Usará el lam_current con el que intentó resolver

        Z_ato = m_inv.objVal
        print(f" >>> ATO Subproblem Obj (Z_ato): {Z_ato:.2f} <<< ")

        # Actualizar mejor solución incumbente (LB)
        if Z_ato > best_obj:
            best_obj = Z_ato
            best_lam = lam_current.copy()

        RC_lambda = {(j, t, p, s): lambda_w_sub[j, t, p, s].RC
                     for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios))}

        past_cuts_data.append((Z_ato, lam_current, RC_lambda))

        # --- B. PROBLEMA MAESTRO DE PRECIOS ---
        # phi_current debe actualizarse de forma exógena, sin depender de I_fixed para mantener validez de los cortes
        phi_current, _ = build_phi(phi_mode, ypsilon, delta, stages, scenarios, prod, comp, L, L_det, I_fixed=None)

        theta_val, lam_new, t_master = build_and_solve_master(
            prod, stages, scenarios, pr, K_features, phi_current, past_cuts_data)

        if theta_val is None:
            print("\nFallo en el maestro. Abortando.\n")
            break 

        lam_current = lam_new 
        solve_time += t_master
        
        # El valor objetivo del MP actualiza el Upper Bound
        ub = min(ub, theta_val)
        
        rel_gap = (ub - best_obj) / max(1.0, abs(best_obj))

        history.append({
            "iteration": iteration,
            "LB": best_obj,
            "UB": ub,
            "t_sub": t_sub,
            "t_master": t_master
        })

        print(f" Master UB: {ub:.2f} | Best Z (LB): {best_obj:.2f} | rel gap: {rel_gap:.4%}")
        print(f"Effective Solve Time after iteration {iteration}: {solve_time:.2f} seconds")

        if rel_gap <= tol:
            print(f"\nConvergence achieved at iteration: {iteration}")
            break

    print("\n------ Benders Finished. Preparing Return Object ------\n")

    # Fijar el subproblema en la mejor política descubierta (sin re-optimizar)
    for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios)):
        lambda_w_sub[j, t, p, s].LB = best_lam[(j, t, p, s)]
        lambda_w_sub[j, t, p, s].UB = best_lam[(j, t, p, s)]

    if plot and history:
        tag = f"heuristic_c{comp}_p{prod}_T{stages}_S{scenarios}_seed{seed}_{time.strftime('%H%M%S')}"
        plot_benders_bounds(history, tag)
    
    return m_inv, x_af, lambda_w_sub, y_af, I_af, A, D_term, solve_time, iteration_stopped
