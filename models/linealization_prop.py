import numpy as np
import gurobipy as gp
from gurobipy import GRB
import random
from itertools import product

def MS_linear(seed, time, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi, branching_structure, I0=None, w_cts=False): 
    random.seed(seed)
    comp = len(A)
    prod = len(A[0])
    pr = len(price)

    m = gp.Model("Modelo ATO")

    if w_cts:
        w = m.addVars(prod, time, pr, scenarios, vtype=GRB.CONTINUOUS, name = "w", lb=0, ub=1)
    else:
        w = m.addVars(prod, time, pr, scenarios, vtype=GRB.BINARY, name = "w")

    I = m.addVars(comp, time, scenarios, vtype=GRB.CONTINUOUS, name="I", lb=0)
    x = m.addVars(comp, time, scenarios, vtype=GRB.CONTINUOUS, name="x", lb=0)
    y = m.addVars(prod, time, scenarios, vtype=GRB.CONTINUOUS, name="y", lb=0) 
    y_bar = m.addVars(prod, time, pr, scenarios, vtype=GRB.CONTINUOUS, name="r", lb=0)

    D_term = {}
    for j, t, p, s in product(range(prod), range(time), range(pr), range(scenarios)):
        D_term[j, t, p, s] = ypsilon[s][t] * (a - b * price[p]) + delta[s][j][t]


    f = (gp.quicksum(pi[s]*price[p]*y_bar[j,t,p,s] for s in range(scenarios) for j in range(prod) for t in range(time) for p in range(pr)) -
        gp.quicksum(pi[s]*H[i]*I[i,t,s] for s in range(scenarios) for i in range(comp) for t in range(time)) -
        gp.quicksum(pi[s]*C[i]*x[i,t,s] for s in range(scenarios) for i in range(comp) for t in range(time)))

    m.setObjective(f, GRB.MAXIMIZE)

    m.addConstrs(gp.quicksum(w[j,t,p,s] for p in range(pr)) == 1 for j in range(prod) for t in range(time) for s in range(scenarios))

    m.addConstrs(y_bar[j,t,p,s] <= w[j,t,p,s] * D_term[j, t, p, s] for j in range(prod) for t in range(time) for p in range(pr) for s in range(scenarios))

    m.addConstrs(y[j,t,s] == gp.quicksum(y_bar[j,t,p,s] for p in range(pr) ) for j in range(prod) for t in range(time) for s in range(scenarios))

    if L_det:
        alpha = {} 
        for i in range(comp):
            for tau in range(time): 
                lead_time = L[i][tau] 
                for t in range(time): 
                    if tau + lead_time <= t:
                        alpha[i, tau, t] = 1
                    else:
                        alpha[i, tau, t] = 0

        if I0 is not None:
            m.addConstrs(
            (gp.quicksum(y[j, tt, s] * A[i][j] for j in range(prod) for tt in range(t + 1)) + I[i, t, s] - I0[i] ==
            gp.quicksum(alpha[i, tau, t] * x[i, tau, s] for tau in range(time)))
            for i in range(comp) for t in range(time) for s in range(scenarios))
        else:
            m.addConstrs(
            (gp.quicksum(y[j, tt, s] * A[i][j] for j in range(prod) for tt in range(t + 1)) + I[i, t, s] ==
            gp.quicksum(alpha[i, tau, t] * x[i, tau, s] for tau in range(time)))
            for i in range(comp) for t in range(time) for s in range(scenarios))

    else:
        alpha = {} 
        for i in range(comp):
            for s in range(scenarios):
                for tau in range(time): 
                    lead_time = L[i][tau][s] 
                    
                    for t in range(time): 
                        if tau + lead_time <= t:
                            alpha[i, tau, t, s] = 1
                        else:
                            alpha[i, tau, t, s] = 0
        if I0 is not None:
            m.addConstrs(
            (gp.quicksum(y[j, tt, s] * A[i][j] for j in range(prod) for tt in range(t + 1)) + I[i, t, s] - I0[i] ==
            gp.quicksum(alpha[i, tau, t, s] * x[i, tau, s] for tau in range(time)))
            for i in range(comp) for t in range(time) for s in range(scenarios))

        else:
            m.addConstrs(
            (gp.quicksum(y[j, tt, s] * A[i][j] for j in range(prod) for tt in range(t + 1)) + I[i, t, s] ==
            gp.quicksum(alpha[i, tau, t, s] * x[i, tau, s] for tau in range(time)))
            for i in range(comp) for t in range(time) for s in range(scenarios))

#===================================================================================================================================================================================
    structure = branching_structure + [1]*(time - len(branching_structure)) 
    n_groups = 1 
    for t, branch_factor in enumerate(structure):
        if t >= time: 
            break
        # 1. Decisiones Here-and-Now (x, w) ligadas a F_{t-1}
        scenarios_per_group_xw = int(scenarios / n_groups)
        for g in range(n_groups):
            first = g * scenarios_per_group_xw 
            for k in range(1, scenarios_per_group_xw):
                s = first + k
                m.addConstrs((x[i, t, s] == x[i, t, first] for i in range(comp)), name=f"NAC_x_t{t}_g{g}")
                for j in range(prod):
                    m.addConstrs((w[j, t, p, s] == w[j, t, p, first] for p in range(pr)), name=f"NAC_w_t{t}_g{g}") 
        # 2. Revelación de incertidumbre: Actualizamos n_groups para que represente F_t
        n_groups = n_groups * branch_factor 
        # 3. Decisiones Wait-and-See (y) ligadas a F_t
        scenarios_per_group_y = int(scenarios / n_groups)
        for g in range(n_groups):
            first = g * scenarios_per_group_y
            for k in range(1, scenarios_per_group_y):
                s = first + k
                for j in range(prod):
                    m.addConstr(y[j, t, s] == y[j, t, first], name=f"NAC_y_t{t}_g{g}")
#===================================================================================================================================================================================

    return m, x, w, y, I, A, D_term

#===================================================================================================================================================================================
    structure = branching_structure + [1]*(time - len(branching_structure)) 
    n_groups = 1 
    for t, branch_factor in enumerate(structure):
        if t >= time: 
            break
        scenarios_per_group = int(scenarios / n_groups)
        for g in range(n_groups):
            first = g * scenarios_per_group 
            for k in range(1, scenarios_per_group):
                s = first + k
                m.addConstrs((x[i, t, s] == x[i, t, first] for i in range(comp)), name=f"NAC_x_t{t}_g{g}")
                for j in range(prod):
                     m.addConstrs((w[j, t, p, s] == w[j, t, p, first] for p in range(pr)), name=f"NAC_w_t{t}_g{g}") 
                     m.addConstr(y[j, t, s] == y[j, t, first], name=f"NAC_y_t{t}_g{g}")
        n_groups = n_groups * branch_factor 



def TS_linear(seed, time, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi, branching_structure, I0=None, w_cts=False): 
    random.seed(seed)
    comp = len(A)
    prod = len(A[0])
    pr = len(price)

    m = gp.Model("Modelo ATO")

    if w_cts:
        w = m.addVars(prod, time, pr, vtype=GRB.CONTINUOUS, name = "w", lb=0, ub=1)
    else:
        w = m.addVars(prod, time, pr, vtype=GRB.BINARY, name = "w")

    x = m.addVars(comp, time, vtype=GRB.CONTINUOUS, name="x", lb=0)

    I = m.addVars(comp, time, scenarios, vtype=GRB.CONTINUOUS, name="I", lb=0)
    y = m.addVars(prod, time, scenarios, vtype=GRB.CONTINUOUS, name="y", lb=0) 
    r = m.addVars(prod, time, pr, scenarios, vtype=GRB.CONTINUOUS, name="r", lb=0)

    D_term = {}
    for j, t, p, s in product(range(prod), range(time), range(pr), range(scenarios)):
        D_term[j, t, p, s] = ypsilon[s][t] * (a - b * price[p]) + delta[s][j][t]


    f = (gp.quicksum(pi[s]*price[p]*r[j,t,p,s] for s in range(scenarios) for j in range(prod) for t in range(time) for p in range(pr)) -
        gp.quicksum(pi[s]*H[i]*I[i,t,s] for s in range(scenarios) for i in range(comp) for t in range(time)) -
        gp.quicksum(C[i]*x[i,t] for i in range(comp) for t in range(time)))

    m.setObjective(f, GRB.MAXIMIZE)

    m.addConstrs(gp.quicksum(w[j, t, p] for p in range(pr)) == 1 for j in range(prod) for t in range(time))

    m.addConstrs(r[j, t, p, s] <= w[j, t, p] * D_term[j, t, p, s] for j in range(prod) for t in range(time) for p in range(pr) for s in range(scenarios))

    m.addConstrs(y[j, t, s] == gp.quicksum(r[j, t, p, s] for p in range(pr)) for j in range(prod) for t in range(time) for s in range(scenarios))

    if L_det:
        alpha = {} 
        for i in range(comp):
            for tau in range(time): 
                lead_time = L[i][tau] 
                for t in range(time): 
                    if tau + lead_time <= t:
                        alpha[i, tau, t] = 1
                    else:
                        alpha[i, tau, t] = 0

        if I0 is not None:
            m.addConstrs(
            (gp.quicksum(y[j, tt, s] * A[i][j] for j in range(prod) for tt in range(t + 1)) + I[i, t, s] - I0[i] ==
            gp.quicksum(alpha[i, tau, t] * x[i, tau] for tau in range(time)))
            for i in range(comp) for t in range(time) for s in range(scenarios))
        else:
            m.addConstrs(
            (gp.quicksum(y[j, tt, s] * A[i][j] for j in range(prod) for tt in range(t + 1)) + I[i, t, s] ==
            gp.quicksum(alpha[i, tau, t] * x[i, tau] for tau in range(time)))
            for i in range(comp) for t in range(time) for s in range(scenarios))

    else:
        alpha = {} 
        for i in range(comp):
            for s in range(scenarios):
                for tau in range(time): 
                    lead_time = L[i][tau][s] 
                    
                    for t in range(time): 
                        if tau + lead_time <= t:
                            alpha[i, tau, t, s] = 1
                        else:
                            alpha[i, tau, t, s] = 0
        if I0 is not None:
            m.addConstrs(
            (gp.quicksum(y[j, tt, s] * A[i][j] for j in range(prod) for tt in range(t + 1)) + I[i, t, s] - I0[i] ==
            gp.quicksum(alpha[i, tau, t, s] * x[i, tau] for tau in range(time)))
            for i in range(comp) for t in range(time) for s in range(scenarios))

        else:
            m.addConstrs(
            (gp.quicksum(y[j, tt, s] * A[i][j] for j in range(prod) for tt in range(t + 1)) + I[i, t, s] ==
            gp.quicksum(alpha[i, tau, t, s] * x[i, tau] for tau in range(time)))
            for i in range(comp) for t in range(time) for s in range(scenarios))

    return m, x, w, y, I, A, D_term