import gurobipy as gp
from gurobipy import GRB
import numpy as np
from itertools import product
import time

from models.linealization_prop import MS_linear


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
        return None, None, (t1 - t0)

    # Extraer la nueva política de precios binaria propuesta
    w_new = {(j, t, p, s): round(w[j, t, p, s].X) 
             for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios))}
    
    return theta.X, w_new, (t1 - t0)



def benders(seed, stages, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi,
                branching_structure, I0, max_iter=50, tol=1e-3, phi_mode="eps_delta"):
    comp, prod, pr = len(A), len(A[0]), len(price)
    import time
    solve_time = 0

    print("--------- Traditional Benders Initialization ---------\n")
    
    # 1. Generar la solución binaria inicial usando el "Profit Estimado"
    C_prod = [sum(A[i][j] * C[i] for i in range(comp)) for j in range(prod)]
    P_star = [(a + b * C_prod[j]) / (2 * b) for j in range(prod)]
    
    w_current = {}
    for j in range(prod):
        best_p_idx = min(range(pr), key=lambda p_idx: abs(price[p_idx] - P_star[j]))
        for t, p, s in product(range(stages), range(pr), range(scenarios)):
            w_current[(j, t, p, s)] = 1.0 if p == best_p_idx else 0.0

    # 2. Inicializar el Subproblema ATO como un LP Continuo (w_cts=True)
    # Esto es OBLIGATORIO en Benders para poder extraer variables duales/reduced costs.
    m_sub, x_sub, w_sub, y_sub, I_sub, _, D_term = MS_linear(
        seed, stages, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi,
        branching_structure, I0, w_cts=True
    )

    m_sub.setParam('BarHomogeneous', 1)
    m_sub.setParam('OutputFlag', 0)

    best_obj = -np.inf
    past_cuts_data = []

    for iteration in range(1, max_iter + 1):
        print(f"\n--- BENDERS ITERATION {iteration} ---\n")

        # --- A. SUBPROBLEMA ATO ---
        # Fijar los precios binarios dictados por el maestro o la inicialización
        for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios)):
            w_sub[j, t, p, s].LB = w_current[(j, t, p, s)]
            w_sub[j, t, p, s].UB = w_current[(j, t, p, s)]

        t0 = time.time()
        m_sub.optimize()
        solve_time += time.time() - t0

        if m_sub.status != GRB.OPTIMAL:
            print(f"\nSubproblema ATO infactible (Status {m_sub.status}).\n")
            return None, None, None, None, None, None, None, None, None

        Z_ato = m_sub.objVal
        print(f" >>> ATO Subproblem Obj (Z_ato): {Z_ato:.2f} <<< ")

        # Criterio de Convergencia
        # if iteration > 1:
        #     gap = theta_val - Z_ato
        #     print(f"     Master Theta: {theta_val:.2f} | Gap: {gap:.2f}")
        #     if gap <= tol or (Z_ato - best_obj <= tol and gap < 0.5):
        #         print(f"\nConvergence achieved at iteration: {iteration}")
        #         break
        
        # best_obj = max(best_obj, Z_ato)
        if iteration > 1:
            best_obj = max(best_obj, Z_ato)
            gap = theta_val - best_obj
            print(f"     Master Theta: {theta_val:.2f} | Gap: {gap:.2f}")
            if gap <= tol or (Z_ato - best_obj <= tol and gap < 0.5):
                print(f"\nConvergence achieved at iteration: {iteration}")
                break

        # Extraer los Reduced Costs (gradientes de w)
        RC_w = {}
        for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios)):
            RC_w[(j, t, p, s)] = w_sub[j, t, p, s].RC

        # Guardar la información para el corte
        past_cuts_data.append((Z_ato, w_current, RC_w))

        # --- B. PROBLEMA MAESTRO DE PRECIOS ---
        theta_val, w_current, t_master = build_and_solve_master(
            prod, stages, scenarios, pr, branching_structure, past_cuts_data
        )
        
        if theta_val is None:
            print("\nFallo en el maestro. Pasando a la siguiente instancia.")
            return None, None, None, None, None, None, None, None, None

        solve_time += t_master

        print(f" >>> Master Problem Solved. Theta projected limit: {theta_val:.2f} <<< ")
        print(f"Effective Solve Time after iteration {iteration}: {solve_time:.2f} seconds")

    # --- Evaluación final y Retorno ---
    print("\n------ Benders Finished. Preparing Return Object ------\n")

    # Como Solver espera un modelo que se pueda consultar con .X (binario en w_vars),
    # devolvemos el modelo subproblema con las cotas finales fijadas a la solución óptima binaria
    return m_sub, x_sub, w_sub, y_sub, I_sub, A, D_term, solve_time, iteration