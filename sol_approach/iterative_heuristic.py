import os
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

from sol_approach.affine_funct_app import MS_linear_affine
from data.phi_features import build_phi as build_phi_base
from data.demand import initial_price_idx


def plot_benders_bounds(history, tag, out_dir="figures/benders_heuristic"):
    os.makedirs(out_dir, exist_ok=True)
    it       = [h["iteration"] for h in history]
    lb       = [h["Z_ato"] for h in history]
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

    return phi, K_base #+ comp


#========================================================================================
# Last Version: Standard Iterative Heuristic // Heuristic benders. Correction of incorporation of inventory level in phi.
#========================================================================================
def build_phi_inv(mode, ypsilon, delta, time, scenarios, prod, comp, L=None, L_det=None,
              I_fixed=None, I_scale=None):
    """phi base (exógeno) + comp features de inventario normalizado al final de cada vector."""
    phi, K_base = build_phi_base(mode, time, scenarios, prod, comp, ypsilon, delta, L, L_det)
    for s in range(scenarios):
        for t in range(time):
            for i in range(comp):
                if I_fixed is None or t == 0:
                    phi[s][t].append(0.0)
                else:
                    phi[s][t].append(I_fixed[(i, t - 1, s)] / I_scale[i])
    return phi, K_base + comp

def update_master_phi(m_master, phi, K_base, K_features, prod, stages, pr, scenarios):
    """Sobrescribe en sitio solo los coeficientes de los features de inventario."""
    link, Gamma = m_master._link, m_master._Gamma
    for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios)):
        for q in range(K_base, K_features):
            m_master.chgCoeff(link[j, t, p, s], Gamma[j, t, p, q], -phi[s][t][q])
    m_master.update()

def build_heuristic_master_model(prod, stages, scenarios, pr, K_features, extended_phi, cts=False):
    m_master = gp.Model("Benders_Master_Persistent")
    m_master.setParam('OutputFlag', 0)
    m_master.setParam('BarHomogeneous', 1)

    BIG_M = 1e9

    theta = m_master.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=BIG_M, name="theta")
    rho = m_master.addVars(prod, stages, pr, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="rho")
    Gamma = m_master.addVars(prod, stages, pr, K_features, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="Gamma")
    
    lambda_w = m_master.addVars(prod, stages, pr, scenarios, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="lambda_w")
    vtype_bar = GRB.CONTINUOUS if cts else GRB.BINARY
    lambda_bar = m_master.addVars(prod, stages, pr, scenarios, vtype=vtype_bar, lb=0, ub=1, name="lambda_bar")

    m_master.setObjective(theta, GRB.MAXIMIZE)

    # Restricciones de binarización de lambda
    for j in range(prod):
        for t in range(stages):
            for s in range(scenarios):
                m_master.addConstr(gp.quicksum(lambda_bar[j, t, p, s] for p in range(pr)) == 1)
                for p in range(pr):
                    for p_prime in range(pr):
                        if p != p_prime:
                            m_master.addConstr(
                                lambda_w[j, t, p, s] >= lambda_w[j, t, p_prime, s] - (1 - lambda_bar[j, t, p, s])
                            )

    # Restricciones de política afín
    for j, t in product(range(prod), range(stages)):
        m_master.addConstr(gp.quicksum(rho[j, t, p] for p in range(pr)) == 1)
        for q in range(K_features):
            m_master.addConstr(gp.quicksum(Gamma[j, t, p, q] for p in range(pr)) == 0)

    # Restricciones de lambda_w en función de rho y Gamma
    link = {}
    for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios)):
        gamma_phi = gp.quicksum(Gamma[j, t, p, q] * extended_phi[s][t][q] for q in range(K_features))
        link[j, t, p, s] = m_master.addConstr(lambda_w[j, t, p, s] == rho[j, t, p] + gamma_phi)
    m_master._link, m_master._Gamma = link, Gamma

    return m_master, theta, lambda_bar

def Iter_policy(seed, stages, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi,
                branching_structure, I0, max_iter=5, tol=1e-4, phi_mode="eps_delta", plot=True, 
                time_limit=3600, inv_features=True, pipeline=None):
    comp, prod, pr = len(A), len(A[0]), len(price)
    import time
    solve_time = 0

    print("--------- Heuristic Benders Initialization ---------\n")
    
    # Calcular el Profit Estimado para el punto inicial
    # Costo base estimado de componentes por producto
    C_prod = [sum(A[i][j] * C[i] for i in range(comp)) for j in range(prod)]

    for j in range(prod):
        best_p_idx = initial_price_idx(a, b, C_prod, price, j)

    # P_star = [(a + b * C_prod[j]) / (2 * b) for j in range(prod)]
    # Precio teórico óptimo P*
    
    lam_current = {}
    for j in range(prod):
        # Encontrar el índice del precio discreto más cercano a P_star[j]
        # best_p_idx = min(range(pr), key=lambda p_idx: abs(price[p_idx] - P_star[j]))
        for t, p, s in product(range(stages), range(pr), range(scenarios)):
            lam_current[(j, t, p, s)] = 1.0 if p == best_p_idx else 0.0
    best_lam = lam_current.copy()

    # Crear phi constante (fuera del bucle)
    phi_current, K_features = build_phi_inv(phi_mode, ypsilon, delta, stages, scenarios, prod, comp,
                                        L, L_det, I_fixed=None, I_scale=[1.0] * comp)
    K_base = K_features - comp
    I_scale, lam_prev = None, None

    # Construir Maestro Persistente
    m_master, theta_master, lambda_bar_master = build_heuristic_master_model(prod, stages, scenarios, pr, K_features, phi_current)

    # Inicializar el Subproblema ATO
    m_inv, x_af, lambda_w_sub, y_af, y_bar, I_af, _, D_term, _, _, _ = MS_linear_affine(
        seed, stages, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi, branching_structure, I0, 
        phi_init=phi_current, K_feat=K_features, lambda_fix=True, phi_mode=phi_mode, pipeline=pipeline)

    m_inv.setParam('BarHomogeneous', 1)
    m_inv.setParam('OutputFlag', 0)

    best_obj = -np.inf
    past_cuts_data = []

    iteration_stopped = 0 # Variable para guardar en qué iteración frenó si falla

    history = []
    ub=np.inf
    t_master = 0

    global_start_time = time.time()


    for iteration in range(1, max_iter + 1):
        print(f"\n--- BENDERS ITERATION {iteration} ---\n")
        iteration_stopped = iteration

        elapsed_time = time.time() - global_start_time
        if elapsed_time >= time_limit:
            break # Usa break si es el ciclo normal, o el return correspondiente si estás dentro de run_phase()

        remaining_time = time_limit - elapsed_time
        m_inv.setParam('TimeLimit', max(0.0, remaining_time))
        m_master.setParam('TimeLimit', max(0.0, remaining_time))
        #============================================
        # --- A. SUBPROBLEMA ATO ---
        # Fijar los lambdas actuales en las cotas del subproblema
        for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios)):
            lambda_w_sub[j, t, p, s].LB = lam_current[(j, t, p, s)]
            lambda_w_sub[j, t, p, s].UB = lam_current[(j, t, p, s)]

        t0 = time.time()
        m_inv.optimize()
        t_sub = time.time() - t0
        solve_time += time.time() - t0

        if m_inv.status == GRB.TIME_LIMIT:
            print("\n[!] Límite de tiempo alcanzado resolviendo el Subproblema. Cortando iteración...")
            break

        if m_inv.status != GRB.OPTIMAL:
            print("\nSubproblema ATO infactible o fallido. Abortando iteraciones y evaluando última solución factible.\n")
            break # Usará el lam_current con el que intentó resolver

        Z_ato = m_inv.objVal
        print(f" >>> ATO Subproblem Obj (Z_ato): {Z_ato:.2f} <<< ")
        #============================================
        # Actualizar mejor solución incumbente (LB)
        if Z_ato > best_obj:
            best_obj = Z_ato
            best_lam = lam_current.copy()

        RC_lambda = {(j, t, p, s): lambda_w_sub[j, t, p, s].RC
                     for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios))}

        past_cuts_data.append((Z_ato, lam_current, RC_lambda))
        #============================================
        if inv_features:
            I_fixed = {(i, t, s): I_af[i, t, s].X
                       for i, t, s in product(range(comp), range(stages), range(scenarios))}
            if I_scale is None:   # se congela en la 1ª iteración
                I_scale = [max(1.0, max(I_fixed[i, t, s] for t in range(stages) for s in range(scenarios)))
                           for i in range(comp)]
            phi_current, _ = build_phi_inv(phi_mode, ypsilon, delta, stages, scenarios, prod, comp,
                                       L, L_det, I_fixed=I_fixed, I_scale=I_scale)
            update_master_phi(m_master, phi_current, K_base, K_features, prod, stages, pr, scenarios)
        #============================================
        # --- B. INYECCIÓN DEL CORTE EN EL MAESTRO ---
        cut_expr = Z_ato + gp.quicksum(
            RC_lambda[(j, t, p, s)] * (lambda_bar_master[j, t, p, s] - lam_current[(j, t, p, s)])
            for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios))
        )
        m_master.addConstr(theta_master <= cut_expr, name=f"BendersCut_iter{iteration}")
        #============================================
        # --- C. RESOLUCIÓN DEL MAESTRO ---
        t0_master = time.time()
        m_master.optimize()
        t_master = time.time() - t0_master

        if m_master.status == GRB.TIME_LIMIT:
            print("\n[!] Límite de tiempo alcanzado resolviendo el Maestro. Cortando iteración...")
            break

        if m_master.status != GRB.OPTIMAL:
            print("\nFallo en el maestro. Abortando.\n")
            break 

        theta_val = theta_master.X
        #============================================     
        # Extraer nueva política
        lam_prev = lam_current
        lam_current = {(j, t, p, s): round(lambda_bar_master[j, t, p, s].X)
                       for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios))}

        ub = m_master.ObjBound if inv_features else min(ub, m_master.ObjBound)  # ver punto 3
        rel_gap = (ub - best_obj) / max(1.0, abs(best_obj))

        history.append({
            "iteration": iteration,
            "Z_ato": Z_ato,
            "LB": best_obj,
            "UB": ub,
            "t_sub": t_sub,
            "t_master": t_master
        })

        print(f" Master UB: {ub:.2f} | Best Z (LB): {best_obj:.2f} | rel gap: {rel_gap:.4%}")
        print(f"Effective Solve Time after iteration {iteration}: {solve_time:.2f} seconds")

        if inv_features:
            if lam_current == lam_prev:
                print(f"\nPunto fijo alcanzado en la iteración {iteration}")
                break
        elif rel_gap <= tol:
            print(f"\nConvergence achieved at iteration: {iteration}")
            break
        
        if m_inv.status == GRB.TIME_LIMIT or m_master.status == GRB.TIME_LIMIT:
            print("\n[!] Gurobi interrumpió la optimización interna por límite de tiempo.")
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
#========================================================================================

#========================================================================================
# Iterative Heuristic // Heuristic benders + Strong cuts
#========================================================================================
def Iter_policy_strong(seed, stages, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi,
                       branching_structure, I0, max_iter=5, tol=1e-4, phi_mode="eps_delta", plot=True, time_limit=3600,
                       inv_features=True, pipeline=None):
    comp, prod, pr = len(A), len(A[0]), len(price)
    import time
    solve_time = 0

    print("--------- Heuristic Benders Initialization (Strong Cuts) ---------\n")
    
    C_prod = [sum(A[i][j] * C[i] for i in range(comp)) for j in range(prod)]

    for j in range(prod):
        best_p_idx = initial_price_idx(a, b, C_prod, price, j)
    
    # P_star = [(a + b * C_prod[j]) / (2 * b) for j in range(prod)]
    
    lam_current = {}
    lam_core = {}
    for j in range(prod):
        # best_p_idx = min(range(pr), key=lambda p_idx: abs(price[p_idx] - P_star[j]))
        for t, p, s in product(range(stages), range(pr), range(scenarios)):
            lam_current[(j, t, p, s)] = 1.0 if p == best_p_idx else 0.0
            lam_core[(j, t, p, s)] = 1.0 / pr  # Punto núcleo inicial

    best_lam = lam_current.copy()

    phi_current, K_features = build_phi_inv(phi_mode, ypsilon, delta, stages, scenarios, prod, comp, L, L_det, I_fixed=None, I_scale=[1.0] * comp)
    K_base = K_features - comp 
    I_scale = None

    m_master, theta_master, lambda_bar_master = build_heuristic_master_model(prod, stages, scenarios, pr, K_features, phi_current)

    m_inv, x_af, lambda_w_sub, y_af, y_bar, I_af, _, D_term, _, _, _ = MS_linear_affine(
        seed, stages, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi, branching_structure, I0, 
        phi_init=phi_current, K_feat=K_features, lambda_fix=True, phi_mode=phi_mode, pipeline=pipeline)

    m_inv.setParam('BarHomogeneous', 1)
    m_inv.setParam('OutputFlag', 0)

    best_obj = -np.inf
    history = []
    ub = np.inf
    t_master = 0
    iteration_stopped = 0
    global_start_time = time.time()

    for iteration in range(1, max_iter + 1):
        print(f"\n--- STRONG BENDERS ITERATION {iteration} ---\n")
        iteration_stopped = iteration
        elapsed_time = time.time() - global_start_time
        if elapsed_time >= time_limit:
            break
            
        remaining_time = time_limit - elapsed_time
        m_inv.setParam('TimeLimit', max(0.0, remaining_time))
        m_master.setParam('TimeLimit', max(0.0, remaining_time))

        t0_sub = time.time()
        
        # --- A1. Actualizar y Evaluar Punto Núcleo (Corte Fuerte) ---
        for k_key in lam_core:
            lam_core[k_key] = 0.5 * lam_core[k_key] + 0.5 * lam_current[k_key]

        for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios)):
            lambda_w_sub[j, t, p, s].LB = lam_core[(j, t, p, s)]
            lambda_w_sub[j, t, p, s].UB = lam_core[(j, t, p, s)]

        m_inv.optimize()

        if m_inv.status == GRB.TIME_LIMIT or m_inv.status != GRB.OPTIMAL:
            break
            
        Z_core = m_inv.objVal
        RC_core = {(j, t, p, s): lambda_w_sub[j, t, p, s].RC for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios))}

        # --- A2. Evaluar Punto Actual (Corte Estándar y Cotas) ---
        for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios)):
            lambda_w_sub[j, t, p, s].LB = lam_current[(j, t, p, s)]
            lambda_w_sub[j, t, p, s].UB = lam_current[(j, t, p, s)]

        m_inv.optimize()
        t_sub = time.time() - t0_sub
        solve_time += t_sub

        if m_inv.status == GRB.TIME_LIMIT or m_inv.status != GRB.OPTIMAL:
            break

        Z_ato = m_inv.objVal
        print(f" >>> ATO Subproblem Obj (Z_ato): {Z_ato:.2f} <<< ")

        if Z_ato > best_obj:
            best_obj = Z_ato
            best_lam = lam_current.copy()

        RC_std = {(j, t, p, s): lambda_w_sub[j, t, p, s].RC for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios))}

        #============================================
        if inv_features:
            # Extraer el inventario del subproblema
            I_fixed = {(i, t, s): I_af[i, t, s].X
                    for i, t, s in product(range(comp), range(stages), range(scenarios))}
            # Congelar la escala en la primera iteración
            if I_scale is None:   
                I_scale = [max(1.0, max(I_fixed[i, t, s] for t in range(stages) for s in range(scenarios)))for i in range(comp)] 
            # Reconstruir phi actualizando el inventario normalizado
            phi_current, _ = build_phi_inv(phi_mode, ypsilon, delta, stages, scenarios, prod, comp,L, L_det, I_fixed=I_fixed, I_scale=I_scale) 
            # Actualizar coeficientes en el maestro en sitio
            update_master_phi(m_master, phi_current, K_base, K_features, prod, stages, pr, scenarios)
        #============================================
        # --- B. Inyección de Cortes en el Maestro ---
        # 1. Corte fuerte de Papadakos
        cut_expr_strong = Z_core + gp.quicksum(
            RC_core[(j, t, p, s)] * (lambda_bar_master[j, t, p, s] - lam_core[(j, t, p, s)])
            for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios))
        )
        m_master.addConstr(theta_master <= cut_expr_strong, name=f"StrongCut_iter{iteration}")
        #============================================
        # 2. Corte estándar
        cut_expr_std = Z_ato + gp.quicksum(
            RC_std[(j, t, p, s)] * (lambda_bar_master[j, t, p, s] - lam_current[(j, t, p, s)])
            for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios))
        )
        m_master.addConstr(theta_master <= cut_expr_std, name=f"BendersCut_iter{iteration}")
        #============================================
        
        # --- C. Resolución del Maestro ---
        t0_master = time.time()
        m_master.optimize()
        t_master = time.time() - t0_master

        if m_master.status == GRB.TIME_LIMIT or m_master.status != GRB.OPTIMAL:
            break 

        lam_current = {(j, t, p, s): round(lambda_bar_master[j, t, p, s].X) 
                       for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios))}
                       
        solve_time += t_master
        ub = min(ub, m_master.ObjBound)
        rel_gap = (ub - best_obj) / max(1.0, abs(best_obj))

        history.append({"iteration": iteration, "Z_ato": Z_ato, "LB": best_obj, "UB": ub, "t_sub": t_sub, "t_master": t_master})
        print(f" Master UB: {ub:.2f} | Best Z (LB): {best_obj:.2f} | rel gap: {rel_gap:.4%}")

        if rel_gap <= tol:
            print(f"\nConvergence achieved at iteration: {iteration}")
            break

    for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios)):
        lambda_w_sub[j, t, p, s].LB = best_lam[(j, t, p, s)]
        lambda_w_sub[j, t, p, s].UB = best_lam[(j, t, p, s)]

    if plot and history:
        tag = f"heuristic_strong_c{comp}_p{prod}_T{stages}_S{scenarios}_seed{seed}_{time.strftime('%H%M%S')}"
        plot_benders_bounds(history, tag)
    
    m_inv._ih_lb = best_obj
    m_inv._ih_ub = ub
    return m_inv, x_af, lambda_w_sub, y_af, I_af, A, D_term, solve_time, iteration_stopped
#========================================================================================

#========================================================================================
# Iterative Heuristic // Heuristic benders + Two-Phase Relaxation (Solve relax until CV, then binary)
#========================================================================================
def Iter_policy_relaxed(seed, stages, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi,
                        branching_structure, I0, max_iter=10, tol=1e-4, phi_mode="eps_delta", plot=True, time_limit=3600,
                        inv_features=True):
    comp, prod, pr = len(A), len(A[0]), len(price)
    import time
    solve_time = 0

    print("--------- Heuristic Benders Initialization (Two-Phase) ---------\n")
    
    C_prod = [sum(A[i][j] * C[i] for i in range(comp)) for j in range(prod)]

    for j in range(prod):
            best_p_idx = initial_price_idx(a, b, C_prod, price, j)

    # P_star = [(a + b * C_prod[j]) / (2 * b) for j in range(prod)]
    
    lam_current = {}
    for j in range(prod):
        # best_p_idx = min(range(pr), key=lambda p_idx: abs(price[p_idx] - P_star[j]))
        for t, p, s in product(range(stages), range(pr), range(scenarios)):
            lam_current[(j, t, p, s)] = 1.0 if p == best_p_idx else 0.0

    phi_current, K_features = build_phi_inv(phi_mode, ypsilon, delta, stages, scenarios, prod, comp, L, L_det, I_fixed=None, I_scale=[1.0] * comp)
    K_base = K_features - comp 
    I_scale = None

    m_master, theta_master, lambda_bar_master = build_heuristic_master_model(prod, stages, scenarios, pr, K_features, phi_current, cts=True)

    m_inv, x_af, lambda_w_sub, y_af, y_bar, I_af, _, D_term, _, _, _ = MS_linear_affine(
        seed, stages, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi, branching_structure, I0, 
        phi_init=phi_current, K_feat=K_features, lambda_fix=True, phi_mode=phi_mode)

    m_inv.setParam('BarHomogeneous', 1)
    m_inv.setParam('OutputFlag', 0)

    history = []
    global_start_time = time.time()

    def run_phase(phase_name, start_iter, lam_curr, current_ub, current_lb):
        nonlocal solve_time
        best_obj = current_lb
        ub = current_ub
        best_lam = lam_curr.copy()

        for iteration in range(start_iter, start_iter + max_iter):
            print(f"\n--- BENDERS {phase_name.upper()} ITERATION {iteration} ---\n")
            elapsed_time = time.time() - global_start_time
            if elapsed_time >= time_limit:
                return lam_curr, best_lam, best_obj, ub, iteration - 1

            remaining_time = time_limit - elapsed_time
            m_inv.setParam('TimeLimit', max(0.0, remaining_time))
            m_master.setParam('TimeLimit', max(0.0, remaining_time))

            for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios)):
                lambda_w_sub[j, t, p, s].LB = lam_curr[(j, t, p, s)]
                lambda_w_sub[j, t, p, s].UB = lam_curr[(j, t, p, s)]

            t0_sub = time.time()
            m_inv.optimize()
            t_sub = time.time() - t0_sub

            if m_inv.status == GRB.TIME_LIMIT or m_inv.status != GRB.OPTIMAL:
                return lam_curr, best_lam, best_obj, ub, iteration - 1

            Z_ato = m_inv.objVal
            print(f" >>> ATO Subproblem Obj (Z_ato): {Z_ato:.2f} <<< ")

            if Z_ato > best_obj:
                best_obj = Z_ato
                best_lam = lam_curr.copy()

            RC_lambda = {(j, t, p, s): lambda_w_sub[j, t, p, s].RC for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios))}

            #============================================
            if inv_features:
                # Extraer el inventario del subproblema
                I_fixed = {(i, t, s): I_af[i, t, s].X
                        for i, t, s in product(range(comp), range(stages), range(scenarios))}
                # Congelar la escala en la primera iteración
                if I_scale is None:   
                    I_scale = [max(1.0, max(I_fixed[i, t, s] for t in range(stages) for s in range(scenarios)))for i in range(comp)] 
                # Reconstruir phi actualizando el inventario normalizado
                phi_current, _ = build_phi_inv(phi_mode, ypsilon, delta, stages, scenarios, prod, comp,L, L_det, I_fixed=I_fixed, I_scale=I_scale) 
                # Actualizar coeficientes en el maestro en sitio
                update_master_phi(m_master, phi_current, K_base, K_features, prod, stages, pr, scenarios)
                #============================================
            cut_expr = Z_ato + gp.quicksum(
                RC_lambda[(j, t, p, s)] * (lambda_bar_master[j, t, p, s] - lam_curr[(j, t, p, s)])
                for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios))
            )
            m_master.addConstr(theta_master <= cut_expr, name=f"BendersCut_{phase_name}_iter{iteration}")
            solve_time += t_sub

            t0_master = time.time()
            m_master.optimize()
            t_master = time.time() - t0_master

            if m_master.status == GRB.TIME_LIMIT or m_master.status != GRB.OPTIMAL:
                return None, None, None, None, iteration

            if phase_name == "Relaxed":
                lam_new = {(j, t, p, s): lambda_bar_master[j, t, p, s].X for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios))}
                ub = min(ub, theta_master.X)
            else:
                lam_new = {(j, t, p, s): round(lambda_bar_master[j, t, p, s].X) for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios))}
                ub = min(ub, m_master.ObjBound)

            solve_time += t_master
            rel_gap = (ub - best_obj) / max(1.0, abs(best_obj))

            history.append({"Phase": phase_name, "iteration": iteration, "Z_ato": Z_ato, "LB": best_obj, "UB": ub, "t_sub": t_sub, "t_master": t_master})
            print(f" Master UB: {ub:.2f} | Best Z (LB): {best_obj:.2f} | rel gap: {rel_gap:.4%}")

            if rel_gap <= tol or lam_new == lam_curr:
                print(f"\nConvergence achieved at {phase_name} iteration: {iteration}")
                return lam_new, best_lam, best_obj, ub, iteration

            lam_curr = lam_new

        return lam_curr, best_lam, best_obj, ub, start_iter + max_iter - 1

    # Fase 1: Relajada
    lam_curr_rel, best_lam_rel, best_obj_rel, ub_rel, last_iter_rel = run_phase("Relaxed", 1, lam_current, np.inf, -np.inf)

    # Fase 2: Transición a variables binarias
    if lam_curr_rel is not None:
        print("\n   >> Transicionando Maestro a variables BINARIAS (conservando cortes previos)...")
        for key in lambda_bar_master:
            lambda_bar_master[key].VType = GRB.BINARY
        m_master.update()

        m_master.optimize()
        if m_master.status == GRB.OPTIMAL:
            lam_bin_start = {(j, t, p, s): round(lambda_bar_master[j, t, p, s].X) for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios))}
            ub_bin_start = m_master.ObjBound
        else:
            lam_bin_start, ub_bin_start = lam_curr_rel, ub_rel

        lam_curr_bin, best_lam_bin, best_obj_bin, ub_bin, last_iter_bin = run_phase("Binary", last_iter_rel + 1, lam_bin_start, ub_bin_start, -np.inf)
    else:
        best_lam_bin = best_lam_rel

    # Evaluacion Final Subproblema
    for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios)):
        lambda_w_sub[j, t, p, s].LB = best_lam_bin[(j, t, p, s)]
        lambda_w_sub[j, t, p, s].UB = best_lam_bin[(j, t, p, s)]

    if plot and history:
        tag = f"heuristic_twophase_c{comp}_p{prod}_T{stages}_S{scenarios}_seed{seed}_{time.strftime('%H%M%S')}"
        plot_benders_bounds(history, tag)
    
    if lam_curr_rel is not None:
        m_inv._ih_lb = best_obj_bin
        m_inv._ih_ub = ub_bin
    else:
        m_inv._ih_lb = best_obj_rel
        m_inv._ih_ub = ub_rel
    return m_inv, x_af, lambda_w_sub, y_af, I_af, A, D_term, solve_time, last_iter_bin if lam_curr_rel is not None else last_iter_rel
#========================================================================================
