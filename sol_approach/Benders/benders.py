import gurobipy as gp
import time
from sol_approach.Benders.MP import create_master
from sol_approach.Benders.SP import solve_subproblem
from output_config.lambda_export import export_solution_to_excel_affine
from itertools import product

class MockVar:
    def __init__(self, value):
        self.X = value

def Benders_dec(seed, stages, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi, branching_structure, I0, time_limit):

    comp = len(A)
    prod = len(A[0])
    pr = len(price)
    V_features = None
    K_features = (stages - 1) * (1 + prod) 
    phi = {}
    for s in range(scenarios):
        phi[s] = {}
        for t in range(stages):
            phi[s][t] = []
            for tau in range(stages - 1):
                if tau < t:
                    phi[s][t].append(ypsilon[s][tau])
                    for j in range(prod):
                        phi[s][t].append(delta[s][j][tau])
                else:
                    phi[s][t].append(0)
                    for j in range(prod):
                        phi[s][t].append(0)

    m_master, rho, Gamma, theta = create_master(prod, stages, pr, scenarios, phi, K_features, V_features)

    TOLERANCE = 1e-4
    UB = float('inf')  # Cota superior (del Maestro)
    LB = -float('inf') # Cota inferior (del Subproblema)


    print("<<< Iniciando Benders >>>")

    start_time = time.time()   
    iteration = 0
    last_lambda = None

    while (time.time() - start_time) < time_limit:
        if iteration == 0:
            rho_val = {(j, t, p): 1.0/pr for j, t, p in product(range(prod), range(stages), range(pr))}
            Gamma_val = {(j, t, p, q): 0.0 for j, t, p, q in product(range(prod), range(stages), range(pr), range(K_features))}
            UB = float('inf') 
        else:
            m_master.optimize()
            if m_master.status != gp.GRB.OPTIMAL:
                print("Error en el Maestro")
                break
            
            UB = theta.X
            rho_val = {k: v.X for k, v in rho.items()}
            Gamma_val = {k: v.X for k, v in Gamma.items()}

        Q_val, mu_val, D_term_max, lambda_val = solve_subproblem(seed, stages, scenarios, A, price, L, L_det, ypsilon, delta, a, b, C, H, pi, branching_structure, I0,
                        phi, K_features, rho_val, Gamma_val)

        last_lambda = lambda_val
        
        LB = max(LB, Q_val)
        gap = (UB - LB) / abs(LB + 1e-10)

        print(f"Iter: {iteration} | UB (Maestro): {UB:.2f} | LB (Subprob): {LB:.2f} | Gap: {gap*100:.2f}%")

        if gap < TOLERANCE:
            print("¡Convergencia alcanzada!")
            break


        g_rho = {}
        g_Gamma = {}
        
        for j, t, p in product(range(prod), range(stages), range(pr)):
            g_rho[j, t, p] = sum(mu_val[j, t, p, s] * D_term_max[j, t, p, s] for s in range(scenarios))
            
            for q in range(K_features):
                g_Gamma[j, t, p, q] = sum(phi[s][t][q] * mu_val[j, t, p, s] * D_term_max[j, t, p, s] for s in range(scenarios))

        cut_expr = Q_val
        for j, t, p in product(range(prod), range(stages), range(pr)):
            cut_expr += g_rho[j, t, p] * (rho[j, t, p] - rho_val[j, t, p])
            for q in range(K_features):
                cut_expr += g_Gamma[j, t, p, q] * (Gamma[j, t, p, q] - Gamma_val[j, t, p, q])
                
        m_master.addConstr(theta <= cut_expr, name=f"Cut_{iteration}")

        iteration += 1

    print(f"<< Iteraciones: {iteration} >>")

    if last_lambda is not None:
        print(">> Procesando lambdas de la última iteración de Benders para exportar...")
        w_vars_mock = {}
        
        for j in range(prod):
            for t in range(stages):
                for p in range(pr):
                    for s in range(scenarios):
                        val = last_lambda.get((j, t, p, s), 0.0) 
                        w_vars_mock[j, t, p, s] = MockVar(val)
        
        filename = f"var_data/MS_benders_inst{seed}.xlsx"
        export_solution_to_excel_affine(filename, w_vars_mock, stages, scenarios, pr, A)
    return None