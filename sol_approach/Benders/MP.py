import gurobipy as gp
from gurobipy import GRB
from itertools import product

def create_master(prod, time, pr, scenarios, phi, K_features, V_features):
    m_master = gp.Model("Benders_Master")
    m_master.Params.OutputFlag = 0  

    rho = m_master.addVars(prod, time, pr, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="rho")
    Gamma = m_master.addVars(prod, time, pr, K_features, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="Gamma")
    
    BIG_M_REVENUE = 1e9
    theta = m_master.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=BIG_M_REVENUE, name="theta")

    m_master.setObjective(theta, GRB.MAXIMIZE)

    for j, t in product(range(prod), range(time)):
        m_master.addConstr(gp.quicksum(rho[j, t, p] for p in range(pr)) == 1)
        for q in range(K_features):
            m_master.addConstr(gp.quicksum(Gamma[j, t, p, q] for p in range(pr)) == 0)

    for j, t, p, s in product(range(prod), range(time), range(pr), range(scenarios)):
        gamma_dot = gp.quicksum(Gamma[j, t, p, q] * phi[s][t][q] for q in range(K_features))
        m_master.addConstr(rho[j, t, p] + gamma_dot >= 0, name=f"lambda_nonneg_{j}_{t}_{p}_{s}")

    return m_master, rho, Gamma, theta