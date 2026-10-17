import math
import numpy as np
from gurobipy import GRB

from models.linealization_prop import MS_linear
from data.generator import build_scenario_groups


def _lead(L, L_det, i, tau, s):
    return L[i][tau] if L_det else L[i][tau][s]

def _window_data(k, Hh, first, stride, S_w, ypsilon, delta, L, L_det, comp, prod):
    """Recorta datos al subárbol: escenario de ventana ws -> escenario real representante."""
    reps = [first + ws * stride for ws in range(S_w)]
    y_w = [[ypsilon[r][k + lt] for lt in range(Hh)] for r in reps]
    d_w = [[[delta[r][j][k + lt] for lt in range(Hh)] for j in range(prod)] for r in reps]
    if L_det:
        L_w = [[L[i][k + lt] for lt in range(Hh)] for i in range(comp)]
    else:
        L_w = [[[L[i][k + lt][r] for r in reps] for lt in range(Hh)] for i in range(comp)]
    return y_w, d_w, L_w

#========================================================================================
# Rolling horizon applied to MS_linear. Scenario sampling only in full tree.
#========================================================================================
def Rolling_horizon(seed, stages, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi,
                    branching, I0, horizon=3, fix_x=True, mip_gap=1e-3, sub_time_limit=60):
    """
    Implementación de rolling horizon para MS_linear.
    Acá no existe sampling de escenarios, sólo se utiliza el árbol completo de escenarios.
    """
    comp, prod, pr, T = len(A), len(A[0]), len(price), stages
    A_np = np.array(A, dtype=float)
    I0 = list(I0) if I0 is not None else [0.0] * comp
    structure = list(branching) + [1] * (T - len(branching))
    groups = build_scenario_groups(T, scenarios, branching)

    max_L = int(np.max(np.array(L)))
    if horizon <= max_L:
        raise ValueError(f"horizon={horizon} <= lead time maximo ({max_L}): "
                         f"ningún pedido de la ventana llegaría a usarse dentro de ella.")

    x_full = np.zeros((comp, T, scenarios))
    w_full = np.zeros((prod, T, pr, scenarios))
    state = {}            # (k, primer_escenario_del_nodo) -> (x_hist, uso_acumulado) tras decidir
    solve_time = 0.0

    for k in range(T):
        Hh = min(horizon, T - k)
        bw = structure[k:k + Hh - 1]                  # ramificación dentro de la ventana
        S_w = math.prod(bw)
        print(f"[RH] stage {k}: {len(groups[k])} nodos, ventana de {Hh} periodos, {S_w} escenarios")

        for g in groups[k]:
            first = g[0]

            # 1) Estado heredado del nodo padre
            if k == 0:
                x_hist, used = np.zeros((comp, 0)), np.zeros(comp)
            else:
                size_p = scenarios // math.prod(structure[:k - 1])
                x_hist, used = state[(k - 1, (first // size_p) * size_p)]

            # 2) Inventario inicial y pipeline de la ventana
            I_start = np.array(I0, dtype=float) - used
            pipe = np.zeros((comp, Hh))
            for i in range(comp):
                for tau in range(k):
                    arr = tau + _lead(L, L_det, i, tau, first)
                    if arr < k:
                        I_start[i] += x_hist[i, tau]       # ya llegó
                    elif arr - k < Hh:
                        pipe[i, arr - k] += x_hist[i, tau]  # llega dentro de la ventana

            # 3) Subárbol y MILP de la ventana
            y_w, d_w, L_w = _window_data(k, Hh, first, len(g) // S_w, S_w,
                                         ypsilon, delta, L, L_det, comp, prod)
            m, x, w, y, I, _, _ = MS_linear(
                seed, Hh, S_w, A, price, L_w, L_det, y_w, d_w, a, b, C, H,
                [1.0 / S_w] * S_w, bw, I0=list(I_start), w_cts=False, pipeline=pipe)

            # y[0] es una decisión del nodo: forzarla común (ver nota abajo)
            for j in range(prod):
                for s in range(1, S_w):
                    m.addConstr(y[j, 0, s] == y[j, 0, 0])

            m.setParam('OutputFlag', 0)
            m.setParam('Threads', 1)
            m.setParam('TimeLimit', sub_time_limit)
            m.setParam('MIPGap', mip_gap)
            m.optimize()
            if m.SolCount == 0:
                print(f"[RH] ventana sin solucin (stage {k}, nodo {first}).")
                return (None,) * 8
            solve_time += m.Runtime

            # 4) Implementar solo el período local 0
            x0 = np.array([x[i, 0, 0].X for i in range(comp)])
            y0 = np.array([y[j, 0, 0].X for j in range(prod)])
            for j in range(prod):
                p_star = max(range(pr), key=lambda p: w[j, 0, p, 0].X)
                w_full[j, k, p_star, g] = 1.0
            x_full[:, k, g] = x0[:, None]

            # 5) Actualizar estado para los hijos
            state[(k, first)] = (np.hstack([x_hist, x0[:, None]]), used + A_np @ y0)

    # Evaluar la política en el modelo completo
    m_f, x_f, w_f, y_f, I_f, A_out, D_term = MS_linear(
        seed, T, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi, branching, I0)
    for j in range(prod):
        for t in range(T):
            for p in range(pr):
                for s in range(scenarios):
                    v = float(w_full[j, t, p, s])
                    w_f[j, t, p, s].LB = v
                    w_f[j, t, p, s].UB = v
    if fix_x:
        for i in range(comp):
            for t in range(T):
                for s in range(scenarios):
                    x_f[i, t, s].LB = x_full[i, t, s]
                    x_f[i, t, s].UB = x_full[i, t, s]

    return m_f, x_f, w_f, y_f, I_f, A_out, D_term, solve_time
#========================================================================================


#========================================================================================
# Rolling horizon applied to MS_linear. Scenario sampling for every window.
#========================================================================================
from data.generator import build_scenario_groups, epsilon_ms, delta_ms, lead_times_ms

def _window_data_resampled(k, Hh, first, bw, S_w, seed_w, ypsilon, delta, L, L_det,
                           comp, prod, smp):
    """Árbol nuevo (misma topología bw) para los períodos k..k+Hh-1.
    El período local 0 se fija al dato realizado del nodo."""
    y_w = epsilon_ms(smp["inst"], Hh, S_w, bw, seed_w, smp["lb_eps"], smp["ub_eps"])
    d_w = delta_ms(smp["inst"], prod, Hh, S_w, bw, seed_w + 1, smp["mu"], smp["std"])

    for ws in range(S_w):
        y_w[ws][0] = ypsilon[first][k]
        for j in range(prod):
            d_w[ws][j][0] = delta[first][j][k]

    if L_det:
        L_w = [[L[i][k + lt] for lt in range(Hh)] for i in range(comp)]
    else:
        L_w = lead_times_ms(smp["inst"], comp, Hh, S_w, bw, seed_w + 2,
                            smp["lb_L"], smp["ub_L"], det=False)
        for i in range(comp):
            for ws in range(S_w):
                L_w[i][0][ws] = L[i][k][first]
    return y_w, d_w, L_w


def Rolling_horizon(seed, stages, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi,
                    branching, I0, horizon=3, fix_x=True, mip_gap=1e-3, sub_time_limit=60,
                    resample=False, sampling=None):
    """
    Implementación de rolling horizon para MS_linear.
    incorporación de sampling de escenarios: se resuelve un subárbol de escenarios representativos en cada ventana.
    """
    if resample and sampling is None:
        raise ValueError("resample=True requiere `sampling` (parámetros de las distribuciones).")

    comp, prod, pr, T = len(A), len(A[0]), len(price), stages
    A_np = np.array(A, dtype=float)
    I0 = list(I0) if I0 is not None else [0.0] * comp
    structure = list(branching) + [1] * (T - len(branching))
    groups = build_scenario_groups(T, scenarios, branching)

    max_L = int(np.max(np.array(L)))
    if horizon <= max_L:
        raise ValueError(f"horizon={horizon} <= lead time máximo ({max_L}): "
                         f"ningún pedido de la ventana llegaría a usarse dentro de ella.")

    x_full = np.zeros((comp, T, scenarios))
    w_full = np.zeros((prod, T, pr, scenarios))
    state = {}            # (k, primer_escenario_del_nodo) -> (x_hist, uso_acumulado) tras decidir
    solve_time = 0.0

    for k in range(T):
        Hh = min(horizon, T - k)
        bw = structure[k:k + Hh - 1]                  # ramificación dentro de la ventana
        S_w = math.prod(bw)
        print(f"[RH] stage {k}: {len(groups[k])} nodos, ventana de {Hh} períodos, {S_w} escenarios")

        for g in groups[k]:
            first = g[0]

            # 1) Estado heredado del nodo padre
            if k == 0:
                x_hist, used = np.zeros((comp, 0)), np.zeros(comp)
            else:
                size_p = scenarios // math.prod(structure[:k - 1])
                x_hist, used = state[(k - 1, (first // size_p) * size_p)]

            # 2) Inventario inicial y pipeline de la ventana
            I_start = np.array(I0, dtype=float) - used
            pipe = np.zeros((comp, Hh))
            for i in range(comp):
                for tau in range(k):
                    arr = tau + _lead(L, L_det, i, tau, first)
                    if arr < k:
                        I_start[i] += x_hist[i, tau]       # ya llegó
                    elif arr - k < Hh:
                        pipe[i, arr - k] += x_hist[i, tau]  # llega dentro de la ventana

            # 3) Subárbol y MILP de la ventana
            if resample:
                seed_w = seed * 1_000_003 + 10_007 * k + first
                y_w, d_w, L_w = _window_data_resampled(k, Hh, first, bw, S_w, seed_w, ypsilon, delta, L, L_det, comp, prod, sampling)
            else:
                y_w, d_w, L_w = _window_data(k, Hh, first, len(g) // S_w, S_w,ypsilon, delta, L, L_det, comp, prod)

            m, x, w, y, I, _, _ = MS_linear(seed, Hh, S_w, A, price, L_w, L_det, y_w, d_w, a, b, C, H,
                [1.0 / S_w] * S_w, bw, I0=list(I_start), w_cts=False, pipeline=pipe)

            # y[0] es una decisión del nodo: forzarla común (ver nota abajo)
            for j in range(prod):
                for s in range(1, S_w):
                    m.addConstr(y[j, 0, s] == y[j, 0, 0])

            m.setParam('OutputFlag', 0)
            m.setParam('Threads', 1)
            m.setParam('TimeLimit', sub_time_limit)
            m.setParam('MIPGap', mip_gap)
            m.optimize()
            if m.SolCount == 0:
                print(f"[RH] ventana sin solución (stage {k}, nodo {first}).")
                return (None,) * 8
            solve_time += m.Runtime

            # 4) Implementar solo el período local 0
            x0 = np.array([x[i, 0, 0].X for i in range(comp)])
            y0 = np.array([y[j, 0, 0].X for j in range(prod)])
            for j in range(prod):
                p_star = max(range(pr), key=lambda p: w[j, 0, p, 0].X)
                w_full[j, k, p_star, g] = 1.0
            x_full[:, k, g] = x0[:, None]

            # 5) Actualizar estado para los hijos
            state[(k, first)] = (np.hstack([x_hist, x0[:, None]]), used + A_np @ y0)

    # Evaluar la política en el modelo completo
    m_f, x_f, w_f, y_f, I_f, A_out, D_term = MS_linear(
        seed, T, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi, branching, I0)
    for j in range(prod):
        for t in range(T):
            for p in range(pr):
                for s in range(scenarios):
                    v = float(w_full[j, t, p, s])
                    w_f[j, t, p, s].LB = v
                    w_f[j, t, p, s].UB = v
    if fix_x:
        for i in range(comp):
            for t in range(T):
                for s in range(scenarios):
                    x_f[i, t, s].LB = x_full[i, t, s]
                    x_f[i, t, s].UB = x_full[i, t, s]

    return m_f, x_f, w_f, y_f, I_f, A_out, D_term, solve_time
#========================================================================================



