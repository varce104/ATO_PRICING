import numpy as np
import gurobipy as gp
from gurobipy import GRB
import random
from itertools import product

def MS_linear_affine(seed, time, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi, branching_structure, I0=None, 
                                phi_init=None, K_feat=None, lambda_fix=False): 
    random.seed(seed)
    comp = len(A)
    prod = len(A[0])
    pr = len(price)

    m = gp.Model("Modelo ATO Afín")

    if K_feat is not None:
        K_features = K_feat
    else:
        K_features = (time - 1) * (1 + prod)

    if phi_init is not None:
        phi = phi_init
    else:
        phi = {}
        for s in range(scenarios):
            phi[s] = {}
            for t in range(time):
                phi[s][t] = [] 
                for tau in range(time - 1):
                    if tau < t:
                        phi[s][t].append(ypsilon[s][tau])
                        
                        for j in range(prod):
                            phi[s][t].append(delta[s][j][tau])
                    else:
                        phi[s][t].append(0) 
                        for j in range(prod):
                            phi[s][t].append(0) 
    
    I = m.addVars(comp, time, scenarios, vtype=GRB.CONTINUOUS, name="I", lb=0)
    x = m.addVars(comp, time, scenarios, vtype=GRB.CONTINUOUS, name="x", lb=0)
    y = m.addVars(prod, time, scenarios, vtype=GRB.CONTINUOUS, name="y", lb=0)
    y_bar = m.addVars(prod, time, pr, scenarios, vtype=GRB.CONTINUOUS, name="r", lb=0)

    lambda_w = m.addVars(prod, time, pr, scenarios, vtype=GRB.CONTINUOUS, name="lambda_w", lb=0) 

    D_term = {}
    for j, t, p, s in product(range(prod), range(time), range(pr), range(scenarios)):
        D_term[j, t, p, s] = ypsilon[s][t] * (a - b * price[p]) + delta[s][j][t]

    f = (gp.quicksum(pi[s]*price[p]*y_bar[j,t,p,s] for s in range(scenarios) for j in range(prod) for t in range(time) for p in range(pr)) -
        gp.quicksum(pi[s]*H[i]*I[i,t,s] for s in range(scenarios) for i in range(comp) for t in range(time)) -
        gp.quicksum(pi[s]*C[i]*x[i,t,s] for s in range(scenarios) for i in range(comp) for t in range(time)))

    m.setObjective(f, GRB.MAXIMIZE)

    m.addConstrs(y_bar[j,t,p,s] <= lambda_w[j,t,p,s] * D_term[j,t,p,s] 
                 for j in range(prod) for t in range(time) for p in range(pr) for s in range(scenarios))

    m.addConstrs(y[j,t,s] == gp.quicksum(y_bar[j,t,p,s] for p in range(pr)) 
                 for j in range(prod) for t in range(time) for s in range(scenarios))

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
                    m.addConstr(y[j, t, s] == y[j, t, first], name=f"NAC_y_t{t}_g{g}")
        n_groups = n_groups * branch_factor 



    if lambda_fix == False: # Cuando se fija lambda, se trabaja como parámetro -> rho y Gamma no son necesarios (incluyendo restricciones asociadas)

        rho = m.addVars(prod, time, pr, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="rho")
        Gamma = m.addVars(prod, time, pr, K_features, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="Gamma")
        for j, t in product(range(prod), range(time)):
            m.addConstr(gp.quicksum(rho[j, t, p] for p in range(pr)) == 1, name=f"sum_rho_{j}_{t}")
            for q in range(K_features):
                m.addConstr(gp.quicksum(Gamma[j, t, p, q] for p in range(pr)) == 0, name=f"sum_Gamma_{j}_{t}_{q}")
        for j, t, p, s in product(range(prod), range(time), range(pr), range(scenarios)):
            prod_gamma_phi = gp.quicksum(Gamma[j, t, p, q] * phi[s][t][q] for q in range(K_features))
            m.addConstr(lambda_w[j, t, p, s] == rho[j, t, p] + prod_gamma_phi, name=f"def_lambda_{j}_{t}_{p}_{s}")

        return m, x, lambda_w, y, I, A, D_term, rho, Gamma, K_features
    
    else:       
        return m, x, lambda_w, y, I, A, D_term, None, None, K_features
    


def MS_affine_cts(seed, time, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi, branching_structure, I0=None, 
                                phi_init=None, K_feat=None, lambda_fix=False): 
    random.seed(seed)
    comp = len(A)
    prod = len(A[0])
    pr = len(price)
    lb_p = min(price)
    ub_p = max(price)

    m = gp.Model("Modelo ATO Afín/lambda cts")

    if K_feat is not None:
        K_features = K_feat
    else:
        K_features = (time - 1) * (1 + prod)

    if phi_init is not None:
        phi = phi_init
    else:
        phi = {}
        for s in range(scenarios):
            phi[s] = {}
            for t in range(time):
                phi[s][t] = [] 
                for tau in range(time - 1):
                    if tau < t:
                        phi[s][t].append(ypsilon[s][tau])
                        
                        for j in range(prod):
                            phi[s][t].append(delta[s][j][tau])
                    else:
                        phi[s][t].append(0) 
                        for j in range(prod):
                            phi[s][t].append(0) 
    
    I = m.addVars(comp, time, scenarios, vtype=GRB.CONTINUOUS, name="I", lb=0)
    x = m.addVars(comp, time, scenarios, vtype=GRB.CONTINUOUS, name="x", lb=0)
    y = m.addVars(prod, time, scenarios, vtype=GRB.CONTINUOUS, name="y", lb=0)
    # y_bar = m.addVars(prod, time, scenarios, vtype=GRB.CONTINUOUS, name="r", lb=0) # No need for this variable anymore

    lambda_w = m.addVars(prod, time, scenarios, vtype=GRB.CONTINUOUS, name="lambda_w", lb=lb_p, ub=ub_p) 

    D_term = {}
    for j, t, s in product(range(prod), range(time), range(scenarios)):
        D_term[j, t, s] = ypsilon[s][t] * (a - b * lambda_w[j, t, s]) + delta[s][j][t]

    f = (gp.quicksum(pi[s]*lambda_w[j, t, s]*y[j,t,s] for s in range(scenarios) for j in range(prod) for t in range(time)) -
        gp.quicksum(pi[s]*H[i]*I[i,t,s] for s in range(scenarios) for i in range(comp) for t in range(time)) -
        gp.quicksum(pi[s]*C[i]*x[i,t,s] for s in range(scenarios) for i in range(comp) for t in range(time)))

    m.setObjective(f, GRB.MAXIMIZE)

    m.addConstrs(y[j,t,s] <= lambda_w[j,t,s] * D_term[j,t,s] 
                 for j in range(prod) for t in range(time) for p in range(pr) for s in range(scenarios))

    # m.addConstrs(y[j,t,s] == gp.quicksum(y_bar[j,t,p,s] for p in range(pr)) 
    #              for j in range(prod) for t in range(time) for s in range(scenarios)) # No need for this constraint as well

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
                    m.addConstr(y[j, t, s] == y[j, t, first], name=f"NAC_y_t{t}_g{g}")
        n_groups = n_groups * branch_factor 

    if lambda_fix == False: # Cuando se fija lambda, se trabaja como parámetro -> rho y Gamma no son necesarios (incluyendo restricciones asociadas)

        rho = m.addVars(prod, time, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="rho")
        Gamma = m.addVars(prod, time, K_features, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="Gamma")

        # for j, t in product(range(prod), range(time)):
        #     m.addConstr(gp.quicksum(rho[j, t, p] for p in range(pr)) == 1, name=f"sum_rho_{j}_{t}")
        #     for q in range(K_features):
        #         m.addConstr(gp.quicksum(Gamma[j, t, p, q] for p in range(pr)) == 0, name=f"sum_Gamma_{j}_{t}_{q}") # Also no need for this because lambda no longe moves in [0,1]

        for j, t, s in product(range(prod), range(time), range(scenarios)):
            prod_gamma_phi = gp.quicksum(Gamma[j, t, q] * phi[s][t][q] for q in range(K_features))
            m.addConstr(lambda_w[j, t, s] == rho[j, t] + prod_gamma_phi, name=f"def_lambda_{j}_{t}_{s}")

        return m, x, lambda_w, y, I, A, D_term, rho, Gamma, K_features
    
    else:       
        return m, x, lambda_w, y, I, A, D_term, None, None, K_features