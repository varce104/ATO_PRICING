import numpy as np
from itertools import product as iproduct


def demand_expr(eps, a_j, b_j, price, delta):
    """
    Única expresión de demanda. Sirve con floats, ndarrays o LinExpr de Gurobi.
    """
    return eps * (a_j - b_j * price) + delta


def demand_params(prod, a, b, a_range=None, b_range=None, seed=0, price_lb=None):
    """
    a, b -> listas de largo `prod`. Acepta escalar, lista (prod,) o rangos U[lb, ub].
    """
    rng = np.random.default_rng(seed + 7919)   # no toca el estado global de random/np.random

    def _vec(val, bounds, name):
        if bounds is not None:
            return rng.uniform(bounds[0], bounds[1], size=prod)
        arr = np.asarray(val, dtype=float)
        if arr.ndim == 0:
            return np.full(prod, float(arr))
        if arr.shape == (prod,):
            return arr.copy()
        raise ValueError(f"{name}: shape {arr.shape} inválido (se esperaba escalar o ({prod},))")

    a_vec, b_vec = _vec(a, a_range, "a"), _vec(b, b_range, "b")

    if np.any(b_vec <= 0):
        raise ValueError("b_j debe ser > 0.")
    if price_lb is not None:
        bad = np.where(a_vec - b_vec * price_lb <= 0)[0]
        if len(bad):
            raise ValueError(f"Productos {bad.tolist()} sin demanda positiva ni al precio mínimo.")
    return a_vec.tolist(), b_vec.tolist()


def demand_table(ypsilon, delta, a, b, price, prod, time, scenarios):
    """
    D[j,t,p,s] truncada en 0.
    """
    return {(j, t, p, s): max(0.0, demand_expr(ypsilon[s][t], a[j], b[j], price[p], delta[s][j][t]))
            for j, t, p, s in iproduct(range(prod), range(time), range(len(price)), range(scenarios))}


def initial_price_idx(a, b, C_prod, price, j):
    """
    Índice de la grilla más cercano a P_j* = (a_j + b_j c_j) / (2 b_j).
    """
    p_star = (a[j] + b[j] * C_prod[j]) / (2 * b[j])
    return min(range(len(price)), key=lambda k: abs(price[k] - p_star))