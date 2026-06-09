import gurobipy as gp
from gurobipy import GRB
import random



def Multistage_problem(seed, time, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi,
                        branching_structure, I0): # Modelo Estocástico (Demanda)
    random.seed(seed)
    comp = len(A)
    prod = len(A[0])
    pr = len(price)

    m = gp.Model("Modelo ATO")

    w = m.addVars(prod, time, pr, scenarios, vtype=GRB.BINARY, name = "w")
    I = m.addVars(comp, time, scenarios, vtype=GRB.CONTINUOUS, name="I", lb=0)
    x = m.addVars(comp, time, scenarios, vtype=GRB.CONTINUOUS, name="x", lb=0)
    y = m.addVars(prod, time, scenarios, vtype=GRB.CONTINUOUS, name="y", lb=0)


    D_term = {}
    for j in range(prod):
        for t in range(time):
            for s in range(scenarios):
                P_expr = gp.quicksum(price[p] * w[j, t, p, s] for p in range(pr))
                D_term[j, t, s] = ypsilon[s][t] * (a - b * P_expr) + delta[s][j][t]


    f = (gp.quicksum(pi[s]*w[j,t,p,s]*price[p]*y[j,t,s] for s in range(scenarios) for j in range(prod) for t in range(time) for p in range(pr)) -
        gp.quicksum(pi[s]*H[i]*I[i,t,s] for s in range(scenarios) for i in range(comp) for t in range(time)) -
        gp.quicksum(pi[s]*C[i]*x[i,t,s] for s in range(scenarios) for i in range(comp) for t in range(time))
        )

    m.setObjective(f, GRB.MAXIMIZE)

    # restricción de demanda
    m.addConstrs(y[j,t,s] <= D_term[j,t,s] 
                 for j in range(prod) for t in range(time) for s in range(scenarios))

    # restricción de precio: a cada producto se debe asignar un único precio
    m.addConstrs(gp.quicksum(w[j,t,p,s] for p in range(pr)) == 1 for j in range(prod) for t in range(time) for s in range(scenarios))

    # inventory balance constraints
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

        # inventory balance constraints
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
        alpha = {} # Todos los pedidos que han llegado hasta el período t
        for i in range(comp):
            for s in range(scenarios):
                for tau in range(time): # Período en que se hace el pedido (t)
                    lead_time = L[i][tau][s] 
                    
                    for t in range(time): # Período en que se revisa si ya llegó
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

    # Non-anticipativity constraints for x and w
    structure = branching_structure + [1]*(time - len(branching_structure)) # Si branching_structure es más corta que time, se rellena con 1
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
        n_groups = n_groups * branch_factor # tras pasar por el primer nodo, se empieza a hacer branching

    return m, x, w, y, I, A, D_term




# if I0 is not None:
#         m.addConstrs(
#         (gp.quicksum(y[j, tt, s] * A[i][j] for j in range(prod) for tt in range(1,t+1)) + I[i, t, s] == gp.quicksum(alpha[i, tau, t] * x[i, tau, s] for tau in range(time)))
#         for i in range(comp) for t in range(1,time) for s in range(scenarios))

#         m.addConstrs((I[i,0,s] == I0[i] - gp.quicksum(y[j, 0, s] * A[i][j] for j in range(prod))) for i in range(comp) for s in range(scenarios))

#     else:
#         m.addConstrs(
#         (gp.quicksum(y[j, tt, s] * A[i][j] for j in range(prod) for tt in range(t + 1)) + I[i, t, s] ==
#         gp.quicksum(alpha[i, tau, t] * x[i, tau, s] for tau in range(time)))
#         for i in range(comp) for t in range(time) for s in range(scenarios))

#         # restricción de inventario inicial
#         m.addConstrs(I[i,0,s] == 0 for i in range(comp) for s in range(scenarios))