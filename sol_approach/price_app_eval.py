import numpy as np
import gurobipy as gp
from gurobipy import GRB
from itertools import product

from sol_approach.affine_funct_app import MS_linear_affine
from models.linealization_prop import MS_linear
from output_config.lambda_export import extract_solution_arrays_affine_w


def Affine_eval(seed, stages, scenarios, A, price, L, L_det,
                ypsilon, delta, a, b, C, H, pi, branching, I0, phi_mode="eps_delta"):
    prod, pr = len(A[0]), len(price)

    m_af, x_af, lam_w, y_af, y_bar_af, I_af, A_out, D_term, rho_var, Gamma_var, K_feat = MS_linear_affine(
        seed, stages, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi, branching, I0,
        phi_mode=phi_mode)
    
    m_af.setParam('OutputFlag', 1)
    m_af.setParam('BarHomogeneous', 1)
    m_af.setParam('TimeLimit', 500)
    m_af.optimize()
    solve_time = m_af.Runtime


    if m_af.status != GRB.OPTIMAL:
        print("MS_linear_affine no es óptimo.")
        return [None]*10

    # Extraer tensores óptimos de la política afín global
    rho_opt = np.zeros((prod, stages, pr))
    Gamma_opt = np.zeros((prod, stages, pr, K_feat))
    
    for j, t, p in product(range(prod), range(stages), range(pr)):
        rho_opt[j, t, p] = rho_var[j, t, p].X
        for q in range(K_feat):
            Gamma_opt[j, t, p, q] = Gamma_var[j, t, p, q].X

    obj_affine = m_af.objVal
    w_bin = extract_solution_arrays_affine_w(lam_w, prod, stages, scenarios, pr)

    m, x_vars, w_vars, y_vars, I_vars, A_out, D_term = MS_linear(
        seed, stages, scenarios, A, price, L, L_det,
        ypsilon, delta, a, b, C, H, pi, branching, I0)

    for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios)):
        val = float(w_bin[j, t, p, s])
        w_vars[j, t, p, s].LB = val
        w_vars[j, t, p, s].UB = val

    print(f"[Affine eval] Obj afín: {obj_affine:.4f}")
    return m, x_vars, w_vars, y_vars, I_vars, A_out, D_term, solve_time, rho_opt, Gamma_opt

def Relaxed_eval(seed, stages, scenarios, A, price, L, L_det,
                 ypsilon, delta, a, b, C, H, pi, branching, I0):
    """
    Resuelve MS_linear con w continuo, aproxima política a binaria por argmax,
    evalúa en MS_linear con w fijo.
    Retorna el modelo MS_linear resuelto con la misma interfaz que solver.py espera.
    """
    comp, prod, pr = len(A), len(A[0]), len(price)

    m_cts, _, w_cts, _, _, _, _ = MS_linear(
        seed, stages, scenarios, A, price, L, L_det,
        ypsilon, delta, a, b, C, H, pi, branching, I0, w_cts=True)
    m_cts.setParam('OutputFlag', 0)
    m_cts.Params.NonConvex = 2
    m_cts.setParam('TimeLimit', 500)
    m_cts.optimize()
    solve_time = m_cts.Runtime

    if m_cts.status != GRB.OPTIMAL:
        raise ValueError("MS_linear relajado no es óptimo.")

    obj_relaxed = m_cts.objVal

    w_bin = np.zeros((prod, stages, pr, scenarios))
    for j, t, s in product(range(prod), range(stages), range(scenarios)):
        vals = [w_cts[j, t, p, s].X for p in range(pr)]
        w_bin[j, t, int(np.argmax(vals)), s] = 1.0

    m, x_vars, w_vars, y_vars, I_vars, A_out, D_term = MS_linear(
        seed, stages, scenarios, A, price, L, L_det,
        ypsilon, delta, a, b, C, H, pi, branching, I0)

    for j, t, p, s in product(range(prod), range(stages), range(pr), range(scenarios)):
        val = float(w_bin[j, t, p, s])
        w_vars[j, t, p, s].LB = val
        w_vars[j, t, p, s].UB = val

    print(f"[Relaxed eval] Obj relajado: {obj_relaxed:.4f}")
    return m, x_vars, w_vars, y_vars, I_vars, A_out, D_term, solve_time