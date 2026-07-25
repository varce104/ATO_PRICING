import gurobipy as gp
from gurobipy import GRB
import random
from itertools import product as iproduct
from data.phi_features import build_phi


def TS_linear_affine(seed, time, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi,
                      I0=None, phi_mode="eps_delta"):
    random.seed(seed)
    comp = len(A)
    prod = len(A[0])
    pr   = len(price)

    m = gp.Model("TS_ATO_Affine")

    phi, K_features = build_phi(phi_mode, time, scenarios, prod, comp, ypsilon, delta, L, L_det)

    # --- First-stage ---
    x       = m.addVars(comp, time, vtype=GRB.CONTINUOUS, lb=0, name="x")
    rho     = m.addVars(prod, time, pr, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="rho")
    Gamma   = m.addVars(prod, time, pr, K_features, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="Gamma")


    # --- Second-stage ---
    I        = m.addVars(comp, time, scenarios, vtype=GRB.CONTINUOUS, lb=0, name="I")
    y        = m.addVars(prod, time, scenarios, vtype=GRB.CONTINUOUS, lb=0, name="y")
    y_bar        = m.addVars(prod, time, pr, scenarios, vtype=GRB.CONTINUOUS, lb=0, name="r")
    lambda_w = m.addVars(prod, time, pr, scenarios, vtype=GRB.CONTINUOUS, lb=0, name="lambda_w")

    # Affine policy constraints
    for j, t in iproduct(range(prod), range(time)):
        m.addConstr(gp.quicksum(rho[j, t, p] for p in range(pr)) == 1,
                    name=f"sum_rho_{j}_{t}")
        for q in range(K_features):
            m.addConstr(gp.quicksum(Gamma[j, t, p, q] for p in range(pr)) == 0,
                        name=f"sum_Gamma_{j}_{t}_{q}")

    for j, t, p, s in iproduct(range(prod), range(time), range(pr), range(scenarios)):
        gp_phi = gp.quicksum(Gamma[j, t, p, q] * phi[s][t][q] for q in range(K_features))
        m.addConstr(lambda_w[j, t, p, s] == rho[j, t, p] + gp_phi,
                    name=f"def_lambda_{j}_{t}_{p}_{s}")

    D_term = {}
    for j, t, p, s in iproduct(range(prod), range(time), range(pr), range(scenarios)):
        D_term[j, t, p, s] = ypsilon[s][t] * (a - b * price[p]) + delta[s][j][t]

    # Objective: x cost is deterministic (no pi[s])
    f = (gp.quicksum(pi[s] * price[p] * y_bar[j, t, p, s] for s in range(scenarios) for j in range(prod) for t in range(time) for p in range(pr))
       - gp.quicksum(pi[s] * H[i] * I[i, t, s] for s in range(scenarios) for i in range(comp) for t in range(time))
       - gp.quicksum(C[i] * x[i, t] for i in range(comp) for t in range(time)))

    m.setObjective(f, GRB.MAXIMIZE)

    m.addConstrs(y_bar[j, t, p, s] <= lambda_w[j, t, p, s] * D_term[j, t, p, s]
                 for j in range(prod) for t in range(time)
                 for p in range(pr) for s in range(scenarios))

    m.addConstrs(y[j, t, s] == gp.quicksum(y_bar[j, t, p, s] for p in range(pr))
                 for j in range(prod) for t in range(time) for s in range(scenarios))

    # Inventory balance — x has no scenario index
    if L_det:
        alpha = {}
        for i in range(comp):
            for tau in range(time):
                lt = L[i][tau]
                for t in range(time):
                    alpha[i, tau, t] = 1 if tau + lt <= t else 0

        base = (I0[i] if I0 is not None else 0)
        if I0 is not None:
            m.addConstrs(
                (gp.quicksum(y[j, tt, s] * A[i][j] for j in range(prod) for tt in range(t + 1))
                 + I[i, t, s] - I0[i] ==
                 gp.quicksum(alpha[i, tau, t] * x[i, tau] for tau in range(time)))
                for i in range(comp) for t in range(time) for s in range(scenarios))
        else:
            m.addConstrs(
                (gp.quicksum(y[j, tt, s] * A[i][j] for j in range(prod) for tt in range(t + 1))
                 + I[i, t, s] ==
                 gp.quicksum(alpha[i, tau, t] * x[i, tau] for tau in range(time)))
                for i in range(comp) for t in range(time) for s in range(scenarios))
    else:
        alpha = {}
        for i in range(comp):
            for s in range(scenarios):
                for tau in range(time):
                    lt = L[i][tau][s]
                    for t in range(time):
                        alpha[i, tau, t, s] = 1 if tau + lt <= t else 0

        if I0 is not None:
            m.addConstrs(
                (gp.quicksum(y[j, tt, s] * A[i][j] for j in range(prod) for tt in range(t + 1))
                 + I[i, t, s] - I0[i] ==
                 gp.quicksum(alpha[i, tau, t, s] * x[i, tau] for tau in range(time)))
                for i in range(comp) for t in range(time) for s in range(scenarios))
        else:
            m.addConstrs(
                (gp.quicksum(y[j, tt, s] * A[i][j] for j in range(prod) for tt in range(t + 1))
                 + I[i, t, s] ==
                 gp.quicksum(alpha[i, tau, t, s] * x[i, tau] for tau in range(time)))
                for i in range(comp) for t in range(time) for s in range(scenarios))
    
    return m, x, lambda_w, y, I, A, D_term, rho, Gamma