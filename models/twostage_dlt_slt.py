import random
import gurobipy as gp
from gurobipy import GRB



def Recourse_problem(seed, time, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi, I0=None):
    random.seed(seed)
    comp = len(A)
    prod = len(A[0])
    pr = len(price)

    m = gp.Model("Modelo ATO")
    
    x = m.addVars(comp, time, vtype=GRB.CONTINUOUS, name="x", lb=0)
    w = m.addVars(prod, time, pr, vtype=GRB.BINARY, name = "w")
    y = m.addVars(prod, time, scenarios, vtype=GRB.CONTINUOUS, name="y", lb=0)
    I = m.addVars(comp, time, scenarios, vtype=GRB.CONTINUOUS, name="I", lb=0)

    D_term = {}
    for j in range(prod):
        for t in range(time):
            # expresión del precio seleccionado para (j,t) (es una LinExpr en w)
            P_expr = gp.quicksum(price[p] * w[j, t, p] for p in range(pr))
            for s in range(scenarios):
                D_term[j, t, s] = ypsilon[s][t] * (a - b * P_expr) + delta[s][j][t]


    f = (gp.quicksum(pi[s]*w[j,t,p]*price[p]*y[j,t,s] for s in range(scenarios) for j in range(prod) for t in range(time) for p in range(pr)) -
        gp.quicksum(pi[s]*H[i]*I[i,t,s] for s in range(scenarios) for i in range(comp) for t in range(time)) -
        gp.quicksum(C[i]*x[i,t] for i in range(comp) for t in range(time))
        )

    m.setObjective(f, GRB.MAXIMIZE)

    # restricción de demanda
    m.addConstrs(y[j,t,s] <= D_term[j,t,s] 
                 for j in range(prod) for t in range(time) for s in range(scenarios))

    # restricción de precio: a cada producto se debe asignar un único precio
    m.addConstrs(gp.quicksum(w[j,t,p] for p in range(pr)) == 1 for j in range(prod) for t in range(time))

    # restricción de balance de inventario
    # alpha = 1 si la orden hecha en tau ya llegó en t
    if L_det:
        alpha = {} # alpha[i, tau, t] = 1 si el pedido (i, tau) ya llegó en t
        for i in range(comp):
            for tau in range(time): # Período en que se hace el pedido
                
                # ¡Este es el cambio clave!
                # El lead time depende del período 'tau' en que se pide
                lead_time = L[i][tau] 
                
                for t in range(time): # Período en que se revisa si ya llegó
                    if tau + lead_time <= t:
                        alpha[i, tau, t] = 1
                    else:
                        alpha[i, tau, t] = 0

        if I0 is not None:
            m.addConstrs(
            (gp.quicksum(y[j, tt, s] * A[i][j] for j in range(prod) for tt in range(t + 1)) + I[i, t, s] + I0[i] ==
            gp.quicksum(alpha[i, tau, t] * x[i, tau] for tau in range(time)))
            for i in range(comp) for t in range(time) for s in range(scenarios))

        else:
            m.addConstrs(
            (gp.quicksum(y[j, tt, s] * A[i][j] for j in range(prod) for tt in range(t + 1)) + I[i, t, s] ==
            gp.quicksum(alpha[i, tau, t] * x[i, tau] for tau in range(time)))
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
            (gp.quicksum(y[j, tt, s] * A[i][j] for j in range(prod) for tt in range(t + 1)) + I[i, t, s] + I0[i] ==
            gp.quicksum(alpha[i, tau, t, s] * x[i, tau] for tau in range(time)))
            for i in range(comp) for t in range(time) for s in range(scenarios))

        else:
            m.addConstrs(
            (gp.quicksum(y[j, tt, s] * A[i][j] for j in range(prod) for tt in range(t + 1)) + I[i, t, s] ==
            gp.quicksum(alpha[i, tau, t, s] * x[i, tau] for tau in range(time)))
            for i in range(comp) for t in range(time) for s in range(scenarios))
    
    print("\nSolving two-stage model, this shouldn't take long...")
    m.setParam('OutputFlag', 0)
    m.optimize()

    if m.status == GRB.OPTIMAL:

        x_val = {(i, t): x[i, t].X for i in range(comp) for t in range(time)}
        w_val = {(j, t, p): w[j, t, p].X for j in range(prod) for t in range(time) for p in range(len(price))}

        return x_val, w_val
    else:
        print("\nError: El problema de Valor Esperado no es optimo.")
        print("")
        return None, None