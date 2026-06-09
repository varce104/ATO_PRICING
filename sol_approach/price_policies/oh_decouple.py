"""
Decoupling Heuristic Policy from Oh, Sourirajan & Ettl (2014).
"Joint Pricing and Production Decisions in an Assemble-to-Order System"
M&SOM 16(4):529-543.

CORE LOGIC (adapted):
  1. Solve myopic TS model → base-stock levels ŷ_t, list prices p̂_t
  2. Safety stocks: Δ_t = ŷ_t - A @ q̂_t
  3. Available inventory: e_t = ŷ_t - Δ_t  (= A @ q̂_t when no excess)
  4. Pricing subproblem per period (adapted Eq. 5):
       max  E[D(p)*p] - c_eff * E[D(p)]
       s.t. A @ E[D(p)] <= e_t   (resource constraint)
  5. Fix w and solve MS_linear

KEY ADAPTATIONS:
  - Discrete price set (not continuous)
  - Lost sales, no backlogs (s_t = 0)
  - Stochastic lead times via TS model
  - The coupling across products through shared components is preserved
    in the resource constraint of Step 4
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from itertools import product as iproduct

from sol_approach.twostage_affine import TS_linear_affine
from sol_approach.price_policies.price_heuristic import apply_price_heuristic_to_model


def _expected_demand(j, t, p_val, a, b, ypsilon, delta, pi, scenarios):
    return sum(pi[s] * max(ypsilon[s][t] * (a - b * p_val) + delta[s][j][t], 0.0)
               for s in range(scenarios))


def _list_price_idx(j, t, price, a, b, ypsilon, delta, pi, scenarios, C, H, A):
    """Unconstrained list price: argmax E[D*(p - c_eff)]."""
    c_eff = sum(A[i][j] * (C[i] + H[i]) for i in range(len(A)))
    return max(range(len(price)),
               key=lambda p: _expected_demand(j, t, price[p], a, b, ypsilon, delta, pi, scenarios)
                             * (price[p] - c_eff))


def _pricing_subproblem(t, prod, price, a, b, ypsilon, delta, pi, scenarios, C, H, A, e_t):
    """
    Adapted Eq.(5) of Oh et al. for discrete prices and lost sales.

    max  sum_j E[D_j(p_j)] * (p_j - c_eff_j)
    s.t. sum_j A[i][j] * E[D_j(p_j)] <= e_t[i]   for all i   (resource)
         sum_p w[j,p] == 1                          for all j   (one price)
         w[j,p] in {0,1}

    The resource constraint couples products through shared components,
    which is the key mechanism of the Oh et al. heuristic.
    """
    comp = len(A)
    pr   = len(price)

    # Precompute expected demands to keep the model linear
    ed = {(j, p): _expected_demand(j, t, price[p], a, b, ypsilon, delta, pi, scenarios)
          for j in range(prod) for p in range(pr)}

    c_eff = [sum(A[i][j] * (C[i] + H[i]) for i in range(comp)) for j in range(prod)]

    m = gp.Model("OH_PricingSubproblem")
    m.setParam('OutputFlag', 0)

    w = m.addVars(prod, pr, vtype=GRB.BINARY)

    m.addConstrs(gp.quicksum(w[j, p] for p in range(pr)) == 1 for j in range(prod))

    for i in range(comp):
        m.addConstr(
            gp.quicksum(A[i][j] * gp.quicksum(w[j, p] * ed[j, p] for p in range(pr))
                        for j in range(prod)) <= max(e_t[i], 0.0))

    m.setObjective(
        gp.quicksum(w[j, p] * ed[j, p] * (price[p] - c_eff[j])
                    for j in range(prod) for p in range(pr)),
        GRB.MAXIMIZE)

    m.optimize()

    if m.status == GRB.OPTIMAL:
        return {(j, p): int(w[j, p].X > 0.5) for j in range(prod) for p in range(pr)}
    return None


def oh_decoupling_heuristic(seed, stages, scenarios, A, price, L, L_det,
                             ypsilon, delta, a, b, C, H, pi, I0):
    """
    Full decoupling heuristic from Oh et al. (2014).
    Returns w_fixed {(j,t,p): 0 or 1}.
    """
    comp = len(A)
    prod = len(A[0])
    pr   = len(price)
    A_arr = np.array(A, dtype=float)

    # ── Step 1: Myopic two-stage model ────────────────────────────────────────
    print("\n--- OH Decoupling: solving myopic TS model ---")
    m_ts, _, _, _, I_ts, _, _, _, _ = TS_linear_affine(
        seed, stages, scenarios, A, price, L, L_det,
        ypsilon, delta, a, b, C, H, pi, I0)
    m_ts.setParam('OutputFlag', 0)
    m_ts.optimize()

    if m_ts.status != GRB.OPTIMAL:
        print("  TS model non-optimal. Falling back to unconstrained list price.")
        p_list = {(j, t, p): 0 for j, t, p in iproduct(range(prod), range(stages), range(pr))}
        for j in range(prod):
            for t in range(stages):
                p_list[j, t, _list_price_idx(j, t, price, a, b, ypsilon, delta, pi, scenarios, C, H, A)] = 1
        return p_list

    # ── Step 2: Base-stock levels ŷ_t ≈ E[I_ts[i,t,s]] ───────────────────────
    y_hat = np.array([[sum(pi[s] * I_ts[i, t, s].X for s in range(scenarios))
                       for t in range(stages)]
                      for i in range(comp)])  # (comp, stages)

    # ── Step 3: List prices and safety stocks ─────────────────────────────────
    q_hat     = np.zeros((prod, stages))
    p_list_fb = np.zeros((prod, stages), dtype=int)  # fallback indices

    for j in range(prod):
        for t in range(stages):
            p_idx = _list_price_idx(j, t, price, a, b, ypsilon, delta, pi, scenarios, C, H, A)
            p_list_fb[j, t] = p_idx
            q_hat[j, t]     = _expected_demand(j, t, price[p_idx], a, b, ypsilon, delta, pi, scenarios)

    # Δ_t = ŷ_t - A @ q̂_t  (safety stocks)
    # e_t = ŷ_t - Δ_t = A @ q̂_t  when y_hat == ŷ_t (no excess inventory)
    # When y_hat > ŷ_t → e_t > A@q̂_t → deeper discounts allowed
    # When y_hat < ŷ_t → e_t < A@q̂_t → prices pushed up
    delta_safety = y_hat - A_arr @ q_hat     # (comp, stages)
    e_t          = y_hat - delta_safety       # (comp, stages) = A @ q_hat at baseline

    # ── Step 4: Pricing subproblem per period ─────────────────────────────────
    print("--- OH Decoupling: solving pricing subproblem per period ---")
    w_fixed = {(j, t, p): 0
               for j, t, p in iproduct(range(prod), range(stages), range(pr))}

    for t in range(stages):
        result = _pricing_subproblem(
            t, prod, price, a, b, ypsilon, delta, pi, scenarios, C, H, A, e_t[:, t])

        if result is not None:
            for j in range(prod):
                for p in range(pr):
                    w_fixed[j, t, p] = result[j, p]
        else:
            # Resource constraint infeasible: fall back to list price
            print(f"  Pricing subproblem infeasible at t={t}. Using list price.")
            for j in range(prod):
                w_fixed[j, t, p_list_fb[j, t]] = 1

    return w_fixed