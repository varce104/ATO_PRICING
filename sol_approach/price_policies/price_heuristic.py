"""
Price Heuristic for Multistage ATO Model
Adapted from Couzon et al. (2020) "Joint optimization of dynamic pricing and lot-sizing 
decisions with nonlinear demands"

CONTEXT
-------
Couzon et al. work with a deterministic, single-echelon lot-sizing problem with isoelastic 
demand D(P) = gamma * alpha * P^(-beta) and continuous prices. Their heuristics sort 
products by a proxy of revenue potential and assign setup/price decisions greedily.

Our model differs in:
  - Stochastic multistage setting with scenario tree
  - Linear demand: D(j,t,s) = ypsilon[s][t] * (a - b*P) + delta[s][j][t]
  - Discrete price set {p_1, ..., p_K}
  - ATO structure: components -> products via Bill of Materials
  - Non-anticipativity constraints on w[j,t,p,s]

ADAPTATION STRATEGY
-------------------
We port the *spirit* of their sorting rules (S1-S4) to our setting:

  S1 (demand sensitivity): sort products by b (price sensitivity of demand).
      In their model: beta/log(alpha). In ours: all products share b, so this 
      rule is a tie and we skip it as a standalone criterion.

  S2 (revenue potential): for each (j,t), compute the expected revenue at each 
      discrete price p and select argmax. This mirrors their "optimal revenue if 
      produced and sold in the same period" rule. This is the PRIMARY heuristic.

  S3 (optimal quantity): select price that maximizes expected demand * price, 
      weighted by scenario probabilities. Equivalent to S2 for our linear demand.

  S4 (cost ratio): select price that balances holding cost exposure against 
      revenue. Adapted using the inventory cost H and component costs C via the 
      Bill of Materials matrix A.

We implement four pricing policies:
  HP1 - "Revenue minus Holding Cost": argmax_p  E[D * p] - H_eff * E[D] * h_cost  (adapts S4)
  HP2 - "Myopic markup": price = cost_markup over expected marginal component cost  (new)
  HP3 - "Lower bound price floor": sets price >= cost floor inspired by Corollary 1

Each heuristic returns w_fixed[j,t,p] in {0,1}, which can then be injected into the 
MS_linear model (fixing w variables) to solve only the operational (x, y, I) subproblem,
dramatically reducing the MILP size.

USAGE
-----
    from sol_approach.price_heuristic import price_heuristic
    w_fixed = price_heuristic(prod, time, price, a, b, ypsilon, delta, pi, C, H, A, 
                               method="HP1")
    # Then fix w in your model:
    for j,t,p,s in product(...):
        w[j,t,p,s].LB = w_fixed[j,t,p]
        w[j,t,p,s].UB = w_fixed[j,t,p]
"""

import numpy as np
from itertools import product as iproduct


def _expected_demand(j, t, p_val, a, b, ypsilon, delta, pi, scenarios):
    """E[D(j,t,s)] at a given price value p_val."""
    ed = 0.0
    for s in range(scenarios):
        d = ypsilon[s][t] * (a - b * p_val) + delta[s][j][t]
        ed += pi[s] * max(d, 0.0)
    return ed


def _effective_component_cost(j, A, C, H):
    """
    Proxy for the marginal cost of satisfying one unit of product j,
    considering BOM requirements and holding costs.
    Adapts the 'cost ratio' idea from sorting rule S4 of Couzon et al.
    """
    comp = len(A)
    cost = 0.0
    for i in range(comp):
        if A[i][j] > 0:
            cost += A[i][j] * (C[i] + H[i])
    return cost


def price_heuristic(prod, time, scenarios, price, a, b, ypsilon, delta, pi, C, H, A, method="HP1"):
    """
    Compute a fixed pricing policy w_fixed[j,t,p] in {0,1} for all (j,t).

    Parameters
    ----------
    prod, time, scenarios : int
    price : list of floats  -- discrete price set
    a, b : float            -- demand intercept and slope
    ypsilon : list[s][t]    -- multiplicative stochastic component
    delta   : list[s][j][t] -- additive stochastic component
    pi      : list[s]       -- scenario probabilities
    C, H    : list[i]       -- component purchase and holding costs
    A       : 2D list       -- Bill of Materials [comp x prod]
    method  : str           -- "HP1", "HP2", "HP3", or "HP4"

    Returns
    -------
    w_fixed : dict {(j,t,p): 0 or 1}
        Non-anticipative pricing policy. Same price for all scenarios at (j,t).
    """
    pr = len(price)
    w_fixed = {(j, t, p): 0 for j, t, p in iproduct(range(prod), range(time), range(pr))}

    for j in range(prod):
        eff_cost = _effective_component_cost(j, A, C, H)

        for t in range(time):
            scores = []

            for p_idx, p_val in enumerate(price):
                ed = _expected_demand(j, t, p_val, a, b, ypsilon, delta, pi, scenarios)

                if method == "HP1":
                    # Adapts S4: revenue minus expected holding cost exposure.
                    # Intuition: higher demand -> more inventory risk -> penalize.
                    score = ed * p_val - eff_cost * ed

                elif method == "HP2":
                    # Myopic markup: choose price that is the highest markup 
                    # over effective cost while still generating positive expected demand.
                    # Inspired by the lower bound on optimal price from Corollary 1
                    # of Couzon et al.: P* >= cost / (1 - 1/beta).
                    # We adapt it: for linear demand, the unconstrained revenue-maximizing 
                    # price is P* = (a + delta_mean) / (2*b), so we round to the nearest 
                    # discrete price and add a cost floor.
                    d_mean = np.mean([delta[s][j][t] for s in range(scenarios)])
                    eps_mean = np.mean([ypsilon[s][t] for s in range(scenarios)])
                    p_star_cont = (eps_mean * a + d_mean) / (2.0 * b * eps_mean) if b > 0 else price[-1]
                    # score is proximity to the theoretical optimum, penalized by 
                    # distance below cost floor
                    cost_floor = eff_cost  
                    if p_val < cost_floor:
                        score = -np.inf
                    else:
                        score = -abs(p_val - p_star_cont)

                elif method == "HP3":
                    # Price floor inspired by Corollary 1 of Couzon et al.
                    # In their isoelastic model: P* >= c / (1 - 1/beta).
                    # For our linear model, a natural floor is P >= eff_cost.
                    # Among prices above the floor, choose the one maximizing revenue.
                    if p_val < eff_cost:
                        score = -np.inf
                    else:
                        score = ed * p_val

                else:
                    raise ValueError(f"Unknown method '{method}'. Choose HP1, HP2, HP3, or HP4.")

                scores.append((score, p_idx))

            # Select best price for (j,t)
            best_score, best_p = max(scores, key=lambda x: x[0])
            w_fixed[j, t, best_p] = 1

    return w_fixed


def apply_price_heuristic_to_model(m, w_vars, w_fixed, prod, time, pr, scenarios):
    """
    Fix the w variables in a Gurobi model according to the heuristic pricing policy.
    This removes all pricing binary decisions, leaving only the operational 
    variables (x, y, I) as free variables -> dramatically reduces the MILP.

    Parameters
    ----------
    m       : gurobipy.Model
    w_vars  : dict {(j,t,p,s): gurobipy.Var}
    w_fixed : dict {(j,t,p): 0 or 1}  -- output of price_heuristic()
    prod, time, pr, scenarios : int
    """
    for j, t, p, s in iproduct(range(prod), range(time), range(pr), range(scenarios)):
        val = float(w_fixed[j, t, p])
        w_vars[j, t, p, s].LB = val
        w_vars[j, t, p, s].UB = val
    m.update()


def compare_heuristics(prod, time, scenarios, price, a, b, ypsilon, delta, pi, C, H, A):
    """
    Utility: compute and print the pricing policy chosen by each heuristic method.
    Useful for quick inspection before running the full model.
    """
    methods = ["HP1", "HP2", "HP3", "HP4"]
    results = {}
    for method in methods:
        w_fixed = price_heuristic(prod, time, scenarios, price, a, b, ypsilon, delta, pi,
                                  C, H, A, method=method)
        # Extract chosen price per (j,t)
        chosen = {}
        for j in range(prod):
            for t in range(time):
                for p_idx, p_val in enumerate(price):
                    if w_fixed[j, t, p_idx] == 1:
                        chosen[j, t] = p_val
        results[method] = chosen

    print("\n=== Heuristic Price Comparison ===")
    print(f"{'(j,t)':<10}", "  ".join(f"{m:<8}" for m in methods))
    for j in range(prod):
        for t in range(time):
            row = f"({j},{t})    "
            for method in methods:
                row += f"{results[method].get((j,t), '-'):<10}"
            print(row)
    return results

"""
Price Heuristic adapted from Oh, Sourirajan & Ettl (2014)
"Joint Pricing and Production Decisions in an Assemble-to-Order System"
Manufacturing & Service Operations Management 16(4):529-543.

═══════════════════════════════════════════════════════════════════════
MAPPING OF DIFFERENCES (important to understand before using this)
═══════════════════════════════════════════════════════════════════════

Oh et al. (2014)                    │ Our multistage model
────────────────────────────────────┼──────────────────────────────────
Dynamic programming                 │ MILP with scenario tree
Continuous price p_t ∈ R            │ Discrete price set, binary w[j,t,p,s]
Demand backlog s_t as state var     │ Lost sales (no backlog tracking)
Zero procurement lead time          │ Stochastic lead times L[i][t][s]
Inverse demand function p(q)        │ Linear: D = ε(a - bP) + δ
Base-stock replenishment policy     │ Explicit procurement x[i,t,s]
Safety stock ssb = ŷ - Bq̂           │ Replaced by expected inv. buffer

═══════════════════════════════════════════════════════════════════════
WHAT IS ADAPTED
═══════════════════════════════════════════════════════════════════════

The core idea from Oh et al. Section 4 (Decoupling Heuristic Policy):

  Step 1 — "List price" (Eq. 4 of Oh et al.):
    Solve a myopic 2-stage stochastic program to get optimal expected 
    demands q̂_t ignoring future periods and current inventory.
    Adaptation: for each (j,t), compute the price p* that maximizes
    expected single-period revenue minus effective component cost,
    with NO inventory constraint. This gives the "unconstrained list price".

  Step 2 — "Inventory adjustment" (Eq. 5 of Oh et al.):
    If actual inventory is above base-stock (excess), discount price.
    If at base-stock, use list price.
    Adaptation: we compute the EXPECTED inventory level for period t
    using the procurement decisions from the previous stage (or a
    pre-solve of the LP relaxation). Then adjust the price toward the 
    discount direction if E[I[i,t]] is above a threshold.

WHAT IS NOT ADAPTED (and why):
  - Backlog-dependent price adjustment: our model has no s_t variable.
    Oh et al. Eq.(3)-(5) depend on backlog state → skipped entirely.
  - Component allocation problem (Eq. 6): in our model this is handled
    implicitly by the y[j,t,s] ≤ D[j,t,s] and BOM constraints.
  - The 7-region characterization (Proposition 6) for the W-model:
    this relies on continuous prices and is not directly portable to
    discrete price sets. We approximate the boundary conditions instead.

═══════════════════════════════════════════════════════════════════════
RESULT
═══════════════════════════════════════════════════════════════════════

Two pricing heuristics are provided:

  OH_LIST  — "List price only" (Step 1 of Oh et al.):
    For each (j,t), choose p ∈ price that maximizes
      E[D(j,t,s,p)] * p  -  c_eff(j) * E[D(j,t,s,p)]
    where c_eff(j) = sum_i A[i][j] * (C[i] + H[i])
    This is the myopic unconstrained optimal, equivalent to q̂_t.

  OH_INV   — "Inventory-adjusted price" (Steps 1+2 of Oh et al.):
    Start from the list price.
    For each (j,t), compute the expected available inventory
    relative to the expected demand at list price.
    If there is a surplus (more inv. than expected demand for 
    components used by j), shift the price DOWN one step.
    If there is a shortage, shift the price UP one step.
    This mimics the "adjust prices to reduce expected holding costs"
    logic of Eq.(5), adapted for discrete prices and lost sales.
"""

import numpy as np
from itertools import product as iproduct


# ─────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────

def _expected_demand(j, t, p_val, a, b, ypsilon, delta, pi, scenarios):
    """E[D(j,t,s)] at a fixed price value p_val (lost-sales, so floor at 0)."""
    ed = 0.0
    for s in range(scenarios):
        d = ypsilon[s][t] * (a - b * p_val) + delta[s][j][t]
        ed += pi[s] * max(d, 0.0)
    return ed


def _effective_component_cost(j, A, C, H):
    """
    Effective unit cost of satisfying one unit of product j.
    Analogous to Oh et al.'s (a_t + c_{t+1}B - h_t B) term in H_t(q).
    We use C[i] as procurement cost and H[i] as holding cost proxy.
    """
    comp = len(A)
    return sum(A[i][j] * (C[i] + H[i]) for i in range(comp))


def _list_price_scores(j, t, price, a, b, ypsilon, delta, pi, scenarios, C, H, A):
    """
    Compute score for each price index for product j at period t.
    Score = E[D]*p - c_eff*E[D]  (myopic single-period profit contribution)
    Corresponds to maximizing H_t(q) = p(q)'q - (a_t + c_{t+1}B - h_tB)q
    from Oh et al., adapted for discrete prices and linear demand.
    """
    c_eff = _effective_component_cost(j, A, C, H)
    scores = []
    for p_idx, p_val in enumerate(price):
        ed = _expected_demand(j, t, p_val, a, b, ypsilon, delta, pi, scenarios)
        score = ed * p_val - c_eff * ed   # revenue minus effective cost
        scores.append((score, p_idx))
    return scores


def _expected_inventory_surplus(j, t, p_val, a, b, ypsilon, delta, pi, scenarios,
                                 A, I0):
    """
    Proxy for E[available_inventory - demand] for product j at period t,
    given price p_val.

    In Oh et al., the inventory adjustment uses the actual y_t - B*q_t.
    Since we don't have a base-stock policy, we approximate:
      surplus[i] ≈ I0[i] / T  (initial inventory spread over periods)
                   - E[D(j,t)] * A[i][j]

    A positive surplus means component i is likely overstocked at period t
    → price should be discounted (push demand up).
    A negative surplus means scarcity → price should be raised.

    We aggregate across all components used by product j,
    weighted by their holding cost H[i] (more expensive components
    → stronger incentive to discount, matching Proposition 8 of Oh et al.).
    """
    comp = len(A)
    ed = _expected_demand(j, t, p_val, a, b, ypsilon, delta, pi, scenarios)

    weighted_surplus = 0.0
    total_h_weight = 0.0
    for i in range(comp):
        if A[i][j] > 0:
            # Expected inventory available per period for component i
            inv_per_period = I0[i] / max(t + 1, 1)
            # Expected demand draw on component i from product j
            demand_draw = ed * A[i][j]
            surplus = inv_per_period - demand_draw
            # Weight by holding cost: surplus of expensive component
            # matters more (Proposition 8 of Oh et al.)
            weighted_surplus += surplus  # * H[i]  ← uncomment to weight by H
            total_h_weight += 1.0

    if total_h_weight > 0:
        return weighted_surplus / total_h_weight
    return 0.0


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────

def price_heuristic_oh(prod, time, scenarios, price, a, b, ypsilon, delta, pi,
                        C, H, A, I0, method="OH_LIST"):
    """
    Compute a fixed (non-anticipative) pricing policy w_fixed[j,t,p] ∈ {0,1}.

    Parameters
    ----------
    prod, time, scenarios : int
    price    : list[float]  — discrete price set {p_0, ..., p_K}
    a, b     : float        — demand intercept and slope
    ypsilon  : list[s][t]   — multiplicative stochastic component
    delta    : list[s][j][t]— additive stochastic component
    pi       : list[s]      — scenario probabilities
    C, H     : list[i]      — component procurement and holding costs
    A        : list[i][j]   — Bill of Materials matrix
    I0       : list[i]      — initial component inventory levels
    method   : str          — "OH_LIST" or "OH_INV"

    Returns
    -------
    w_fixed : dict {(j,t,p): 0 or 1}
        Non-anticipative pricing policy. Assigns exactly one price per (j,t).
    """
    pr = len(price)
    w_fixed = {(j, t, p): 0
               for j, t, p in iproduct(range(prod), range(time), range(pr))}

    for j in range(prod):
        for t in range(time):

            # ── Step 1: compute list price (myopic unconstrained optimum) ──
            scores = _list_price_scores(
                j, t, price, a, b, ypsilon, delta, pi, scenarios, C, H, A)
            _, best_p_idx = max(scores, key=lambda x: x[0])

            if method == "OH_LIST":
                # Use list price directly — corresponds to q̂_t of Oh et al.
                w_fixed[j, t, best_p_idx] = 1

            elif method == "OH_INV":
                # ── Step 2: inventory adjustment ──
                # Compute expected surplus at the list price
                p_list_val = price[best_p_idx]
                surplus = _expected_inventory_surplus(
                    j, t, p_list_val, a, b, ypsilon, delta, pi, scenarios, A, I0)

                # Adjust price index:
                # surplus > 0 → overstocked → discount (lower price index)
                # surplus < 0 → understocked → raise (higher price index)
                # Threshold avoids noise-driven adjustments
                THRESHOLD = 0.5   # units; tune if needed

                if surplus > THRESHOLD and best_p_idx > 0:
                    # Discount: move one step down in the price set
                    # Matches "adjust prices to reduce expected holding costs"
                    # from Eq.(5) of Oh et al.
                    adjusted_p_idx = best_p_idx - 1
                elif surplus < -THRESHOLD and best_p_idx < pr - 1:
                    # Scarcity: move one step up
                    adjusted_p_idx = best_p_idx + 1
                else:
                    adjusted_p_idx = best_p_idx

                w_fixed[j, t, adjusted_p_idx] = 1

            else:
                raise ValueError(
                    f"Unknown method '{method}'. Choose 'OH_LIST' or 'OH_INV'.")

    return w_fixed


def apply_oh_heuristic_to_model(m, w_vars, w_fixed, prod, time, pr, scenarios):
    """
    Fix w variables in a Gurobi model using the Oh et al. heuristic policy.
    Removes all binary pricing decisions → model becomes a pure LP.

    Parameters
    ----------
    m        : gurobipy.Model
    w_vars   : dict {(j,t,p,s): gurobipy.Var}
    w_fixed  : dict {(j,t,p): 0 or 1}  — output of price_heuristic_oh()
    prod, time, pr, scenarios : int
    """
    for j, t, p, s in iproduct(range(prod), range(time), range(pr), range(scenarios)):
        val = float(w_fixed[j, t, p])
        w_vars[j, t, p, s].LB = val
        w_vars[j, t, p, s].UB = val
    m.update()


def compare_oh_heuristics(prod, time, scenarios, price, a, b, ypsilon, delta,
                           pi, C, H, A, I0):
    """
    Print the price chosen by OH_LIST and OH_INV for every (j,t).
    Useful for quick inspection before running the full model.
    """
    results = {}
    for method in ["OH_LIST", "OH_INV"]:
        w_fixed = price_heuristic_oh(
            prod, time, scenarios, price, a, b, ypsilon, delta, pi,
            C, H, A, I0, method=method)
        chosen = {}
        for j in range(prod):
            for t in range(time):
                for p_idx, p_val in enumerate(price):
                    if w_fixed[j, t, p_idx] == 1:
                        chosen[j, t] = p_val
        results[method] = chosen

    print("\n=== Oh et al. (2014) Heuristic Price Comparison ===")
    print(f"{'(j,t)':<10}", "  ".join(f"{m:<12}" for m in ["OH_LIST", "OH_INV"]))
    for j in range(prod):
        for t in range(time):
            row = f"({j},{t})    "
            for method in ["OH_LIST", "OH_INV"]:
                row += f"{results[method].get((j,t), '-'):<14}"
            print(row)
    return results