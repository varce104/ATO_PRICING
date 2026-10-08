import os
import pandas as pd
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from itertools import product
import time

from models.linealization_prop import MS_linear


def export_benders_history(history, tag, out_dir="var_results/benders_history"):
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/benders_{tag}.xlsx"
    pd.DataFrame(history).to_excel(path, index=False)
    print(f"\n>> Historial de Benders exportado a: {path}")
    return path

def plot_benders_bounds(history, tag, out_dir="figures/benders"):
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
    ax1.set_title("Benders: bounds evolution")
    ax1.grid(linestyle="--", alpha=0.4)
    ax1.legend()

    ax2.bar(it, t_sub, color="#3e944e", alpha=0.5, label="Subproblem")
    ax2.bar(it, t_master, bottom=t_sub, color="#742626", alpha=0.5, label="Master")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Solve time (s)")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.legend()

    plt.tight_layout()
    path = f"{out_dir}/benders_bounds_{tag}.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f">> Gráfico de cotas guardado en: {path}")


def build_and_solve_master(prod, stages, scenarios, pr, branching_structure, past_cuts_data):
    """
    Problema Maestro de Benders Tradicional.
    Decide directamente la política de precios binaria (w) respetando 
    la no-anticipatividad del árbol de escenarios.
    """
    m_master = gp.Model("Benders_Master_Exact")
    m_master.setParam('OutputFlag', 0)
    m_master.setParam('BarHomogeneous', 1)

    # Variable theta acotada por arriba para evitar unboundedness inicial
    BIG_M = 1e9
    theta = m_master.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=BIG_M, name="theta")
    
    # Variable original de precios (BINARIA)
    w = m_master.addVars(prod, stages, pr, scenarios, vtype=GRB.BINARY, name="w")
    # w = m_master.addVars(prod, stages, pr, scenarios, vtype=GRB.CONTINUOUS, name="w", lb=0, ub=1)

    m_master.setObjective(theta, GRB.MAXIMIZE)

    # 1. Restricción de Selección Única de Precio
    m_master.addConstrs(
        (gp.quicksum(w[j, t, p, s] for p in range(pr)) == 1 
         for j in range(prod) for t in range(stages) for s in range(scenarios)),
        name="Single_Price"
    )

    # 2. Restricciones de No-Anticipatividad para 'w' (copiado del modelo base)
    structure = branching_structure + [1]*(stages - len(branching_structure)) 
    n_groups = 1 
    for t, branch_factor in enumerate(structure):
        if t >= stages: 
            break
        scenarios_per_group_w = int(scenarios / n_groups)
        for g in range(n_groups):
            first = g * scenarios_per_group_w 
            for k in range(1, scenarios_per_group_w):
                s = first + k
                for j in range(prod):
                    m_master.addConstrs(
                        (w[j, t, p, s] == w[j, t, p, first] for p in range(pr)), 
                        name=f"NAC_w_t{t}_g{g}_s{s}"
                    )
        n_groups = n_groups * branch_factor 

    # 3. Inyección de los Cortes de Benders históricos
    for idx, (Z_k, w_k, RC_k) in enumerate(past_cuts_data):
        # El corte tradicional de Benders sumando el Reduced Cost * (w_maestro - w_k)
        cut_expr = Z_k + gp.quicksum(
            RC_k[(j, t, p, s)] * (w[j, t, p, s] - w_k[(j, t, p, s)])
            for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios))
        )
        m_master.addConstr(theta <= cut_expr, name=f"BendersCut_{idx}")

    t0 = time.time()
    m_master.optimize()
    t1 = time.time()

    if m_master.status != GRB.OPTIMAL:
        print(f"  [!] Alerta: Master problem infactible/no acotado (Status {m_master.status}).")
        return None, None, None, (t1 - t0)

    # Extraer la nueva política de precios binaria propuesta
    w_new = {(j, t, p, s): round(w[j, t, p, s].X) 
             for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios))}
    
    return theta.X, m_master.ObjBound, w_new, (t1 - t0)

def benders(seed, stages, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi,
            branching_structure, I0, max_iter=60, tol=1e-4, save_history=True, plot=True, strong_cuts=False):
    
    comp, prod, pr = len(A), len(A[0]), len(price)
    solve_time = 0

    mode_str = "Papadakos Strong Cuts" if strong_cuts else "Traditional Benders"
    print(f"--------- Benders Initialization ({mode_str}) ---------\n")

    C_prod = [sum(A[i][j] * C[i] for i in range(comp)) for j in range(prod)]
    P_star = [(a + b * C_prod[j]) / (2 * b) for j in range(prod)]

    w_current = {}
    for j in range(prod):
        best_p_idx = min(range(pr), key=lambda p_idx: abs(price[p_idx] - P_star[j]))
        for t, p, s in product(range(stages), range(pr), range(scenarios)):
            w_current[(j, t, p, s)] = 1.0 if p == best_p_idx else 0.0

    #========================================================================================
    # Inicialización del punto núcleo w_core para cortes fuertes de Papadakos
    w_core = {}
    if strong_cuts:
        for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios)):
            w_core[(j, t, p, s)] = 1.0 / pr  # Punto ubicado en el interior relativo del simplex
    #========================================================================================

    m_sub, x_sub, w_sub, y_sub, I_sub, _, D_term = MS_linear(
        seed, stages, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi,
        branching_structure, I0, w_cts=True)
    
    m_sub.setParam('BarHomogeneous', 1)
    m_sub.setParam('OutputFlag', 0)

    best_obj = -np.inf          # LB
    best_w = w_current.copy()
    past_cuts_data = []
    ub = np.inf                 # UB
    history = []
    t_cum = 0.0
    iteration = 0

    for iteration in range(1, max_iter + 1):
        print(f"\n--- BENDERS ITERATION {iteration} ---\n")
        t_wall0 = time.time()

        # --- A. SUBPROBLEMA ATO (Evaluación en w_current) ---
        for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios)):
            w_sub[j, t, p, s].LB = w_current[(j, t, p, s)]
            w_sub[j, t, p, s].UB = w_current[(j, t, p, s)]

        t0 = time.time()
        m_sub.optimize()
        t_sub_std = time.time() - t0

        if m_sub.status != GRB.OPTIMAL:
            print(f"\nSubproblema ATO infactible (Status {m_sub.status}).\n")
            return None, None, None, None, None, None, None, None, None

        Z_ato = m_sub.objVal
        print(f" >>> ATO Subproblem Obj (Z_ato): {Z_ato:.2f} <<< ")

        if Z_ato > best_obj:
            best_obj = Z_ato
            best_w = w_current.copy()

        RC_w_std = {(j, t, p, s): w_sub[j, t, p, s].RC
                    for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios))}

        t_sub_total = t_sub_std
    #========================================================================================
    # Master Problem Cuts
    #========================================================================================
        if not strong_cuts:
            # --- BENDERS CUTS ---
            past_cuts_data.append((Z_ato, w_current, RC_w_std))
        else:
            # --- STRONG CUTS ---
            # 1. Actualizar el punto núcleo dinámico w_core (Eq. 28 de Papadakos 2008) // Convex combination (middle point) of the previous core point and the current solution
            for k_key in w_core:
                w_core[k_key] = 0.5 * w_core[k_key] + 0.5 * w_current[k_key]

            # 2. Fijar w_sub en el punto núcleo interior w_core y resolver el subproblema independiente
            for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios)):
                w_sub[j, t, p, s].LB = w_core[(j, t, p, s)]
                w_sub[j, t, p, s].UB = w_core[(j, t, p, s)]

            t0_strong = time.time()
            m_sub.optimize()
            t_sub_strong = time.time() - t0_strong
            t_sub_total += t_sub_strong

            if m_sub.status != GRB.OPTIMAL:
                print(f"\nSubproblema ATO Strong Cut infactible (Status {m_sub.status}).\n")
                return None, None, None, None, None, None, None, None, None

            Z_strong = m_sub.objVal
            RC_w_strong = {(j, t, p, s): w_sub[j, t, p, s].RC
                           for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios))}

            # 3. Inyectar el corte fuerte de Papadakos (y el corte local ajustado)
            past_cuts_data.append((Z_strong, w_core.copy(), RC_w_strong))
            past_cuts_data.append((Z_ato, w_current, RC_w_std))
    #========================================================================================
        solve_time += t_sub_total

        # --- B. PROBLEMA MAESTRO ---
        theta_val, obj_bound, w_new, t_master = build_and_solve_master(
            prod, stages, scenarios, pr, branching_structure, past_cuts_data)

        if theta_val is None:
            print("\nFallo en el maestro. Pasando a la siguiente instancia.")
            return None, None, None, None, None, None, None, None, None

        solve_time += t_master
        ub = min(ub, obj_bound)
        rel_gap = (ub - best_obj) / max(1.0, abs(best_obj))

        t_wall = time.time() - t_wall0
        t_cum += t_wall
        history.append({
            "iteration": iteration,
            "Z_ato": Z_ato,          # Valor en la solución evaluada
            "LB": best_obj,          # Mejor cota inferior (incumbent)
            "UB": ub,                # Mejor cota superior del maestro
            "rel_gap": rel_gap,
            "t_sub": t_sub_total,    # Tiempo acumulado de subproblemas
            "t_master": t_master,    # Tiempo del maestro
            "t_iter": t_wall,        # Tiempo total de iteración
            "t_iter_wall": t_cum,
        })

        print(f" >>> Master UB: {ub:.2f} | Best Z (LB): {best_obj:.2f} | rel gap: {rel_gap:.4%} <<< ")
        print(f"Effective Solve Time after iteration {iteration}: {solve_time:.2f} seconds")

        if rel_gap <= tol:
            print(f"\nConvergence achieved at iteration: {iteration}")
            break

        if w_new == w_current and not strong_cuts:
            print("\nMaster repitió el mismo punto: convergencia.")
            break

        w_current = w_new
    else:
        print(f"\n[!] max_iter ({max_iter}) alcanzado sin cerrar el gap.")

    print("\n------ Benders Finished. Preparing Return Object ------\n")

    # Restaurar la mejor solución en el subproblema
    for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios)):
        w_sub[j, t, p, s].LB = best_w[(j, t, p, s)]
        w_sub[j, t, p, s].UB = best_w[(j, t, p, s)]

    m_sub._benders_lb = best_obj
    m_sub._benders_ub = ub
    m_sub._benders_history = history

    tag = f"c{comp}_p{prod}_T{stages}_S{scenarios}_seed{seed}_{'strong_' if strong_cuts else ''}{time.strftime('%H%M%S')}"
    if save_history:
        export_benders_history(history, tag)
    if plot:
        plot_benders_bounds(history, tag)

    return m_sub, x_sub, w_sub, y_sub, I_sub, A, D_term, solve_time, iteration