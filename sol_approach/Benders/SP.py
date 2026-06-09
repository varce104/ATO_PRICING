import gurobipy as gp
from gurobipy import GRB
import random
from itertools import product

def solve_subproblem(seed, time, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi, branching_structure, I0,
                    phi, K_features, rho_val, Gamma_val):
    random.seed(seed)
    comp = len(A)
    prod = len(A[0])
    pr = len(price)

    m_sub = gp.Model("Benders_Subproblem")
    m_sub.Params.OutputFlag = 0

    lambda_val = {}
    for j, t, p, s in product(range(prod), range(time), range(pr), range(scenarios)):
        gamma_dot = sum(Gamma_val[j, t, p, q] * phi[s][t][q] for q in range(K_features))
        lambda_val[j, t, p, s] = max(0, rho_val[j, t, p] + gamma_dot) 

    I = m_sub.addVars(comp, time, scenarios, vtype=GRB.CONTINUOUS, lb=0)
    x = m_sub.addVars(comp, time, scenarios, vtype=GRB.CONTINUOUS, lb=0)
    y = m_sub.addVars(prod, time, scenarios, vtype=GRB.CONTINUOUS, lb=0)
    r = m_sub.addVars(prod, time, pr, scenarios, vtype=GRB.CONTINUOUS, lb=0)

    D_term = {}
    for j, t, p, s in product(range(prod), range(time), range(pr), range(scenarios)):
        D_term[j, t, p, s] = ypsilon[s][t] * (a - b * price[p]) + delta[s][j][t]

    f = (gp.quicksum(pi[s]*price[p]*r[j,t,p,s] for s in range(scenarios) for j in range(prod) for t in range(time) for p in range(pr)) -
         gp.quicksum(pi[s]*H[i]*I[i,t,s] for s in range(scenarios) for i in range(comp) for t in range(time)) -
         gp.quicksum(pi[s]*C[i]*x[i,t,s] for s in range(scenarios) for i in range(comp) for t in range(time)))
    
    m_sub.setObjective(f, GRB.MAXIMIZE)

    demand_constrs = {}
    for j, t, p, s in product(range(prod), range(time), range(pr), range(scenarios)):
        rhs = D_term[j, t, p, s] * lambda_val[j, t, p, s]
        demand_constrs[j, t, p, s] = m_sub.addConstr(r[j,t,p,s] <= rhs)

    m_sub.addConstrs(y[j,t,s] == gp.quicksum(r[j,t,p,s] for p in range(pr)) 
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
            m_sub.addConstrs(
            (gp.quicksum(y[j, tt, s] * A[i][j] for j in range(prod) for tt in range(t + 1)) + I[i, t, s] - I0[i] ==
            gp.quicksum(alpha[i, tau, t] * x[i, tau, s] for tau in range(time)))
            for i in range(comp) for t in range(time) for s in range(scenarios))
        else:
            m_sub.addConstrs(
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
            m_sub.addConstrs(
            (gp.quicksum(y[j, tt, s] * A[i][j] for j in range(prod) for tt in range(t + 1)) + I[i, t, s] - I0[i] ==
            gp.quicksum(alpha[i, tau, t, s] * x[i, tau, s] for tau in range(time)))
            for i in range(comp) for t in range(time) for s in range(scenarios))
        else:
            m_sub.addConstrs(
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
                m_sub.addConstrs((x[i, t, s] == x[i, t, first] for i in range(comp)), name=f"NAC_x_t{t}_g{g}")
                m_sub.addConstr(y[j, t, s] == y[j, t, first], name=f"NAC_y_t{t}_g{g}")
        n_groups = n_groups * branch_factor 

    m_sub.optimize()

    if m_sub.status == gp.GRB.INF_OR_UNBD:
        m_sub.Params.DualReductions = 0
        m_sub.optimize()

    if m_sub.status == gp.GRB.INFEASIBLE:
        print("¡Subproblema infactible! Calculando IIS...")
        m_sub.computeIIS()
        m_sub.write("subproblema_infactible.ilp")
        raise ValueError("Detenido por infactibilidad. Revisa el archivo subproblema_infactible.ilp")
        
    elif m_sub.status == gp.GRB.UNBOUNDED:
        raise ValueError("¡El modelo es NO ACOTADO! La función objetivo tiende a infinito. Revisa si hay costos negativos o si falta una cota superior en 'y'.")

    elif m_sub.status != gp.GRB.OPTIMAL:
        raise ValueError(f"Error inesperado. El modelo no resolvió óptimamente. Código de estatus: {m_sub.status}")

    Q_val = m_sub.ObjVal
    mu_val = {}
    for j, t, p, s in product(range(prod), range(time), range(pr), range(scenarios)):
        mu_val[j, t, p, s] = demand_constrs[j, t, p, s].Pi

    return Q_val, mu_val, D_term, lambda_val