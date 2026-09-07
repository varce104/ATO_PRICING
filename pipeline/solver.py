from data.generator import epsilon_ms, delta_ms, lead_times_ms
from data.params import parametros, bill_of_materials, price_set
from data.extract import extract_params

from models.multistage import Multistage_problem
from models.linealization_prop import MS_linear, TS_linear

from sol_approach.affine_funct_app import MS_linear_affine, MS_affine_cts
from sol_approach.price_app_eval import Affine_eval, Relaxed_eval
from sol_approach.twostage_affine import TS_linear_affine
from sol_approach.iterative_heuristic import Iter_policy
from sol_approach.local_search import local_search_first_improvement
from sol_approach.out_of_sample import Affine_OOS_eval

from uncertainty_analysis.sto_computation import uncertainty_analysis
from output_config.results_output import export
from output_config.lambda_export import fix_w_from_lambda

import pandas as pd
import numpy as np
from itertools import product
import time
import gurobipy as gp
from gurobipy import GRB



def solve(cfg):
    # Único punto donde se toca cfg — el resto de la función queda igual que antes
    C, H, pi, A, price, mult, add, L, a, b, I0 = extract_params(cfg)

    inst, comp, prod = cfg.size.inst, cfg.size.comp, cfg.size.prod
    stages, scenarios, branching = cfg.size.stages, cfg.size.scenarios, cfg.size.branching
    seed, time_limit = cfg.size.seed, cfg.size.time_limit
    det = cfg.lead_times.det

    Model = cfg.run.Model
    W_cts = cfg.run.W_cts

    show_sol = cfg.Output.show_sol
    show_heatmap = cfg.Output.show_heatmap
    show_boxplot = cfg.Output.show_boxplot
    show_candlestick = cfg.Output.show_candlestick
    show_kpis = cfg.Output.show_kpis
    local_search = cfg.run.local_search
    
    if inst is not None:
        comp = len(A); prod = len(A[0])
        print(f"\n Model: {Model} | seed: {seed} | Components: {comp} | Products: {prod} | Stages: {stages} | Scenarios: {scenarios} ")

    solve_time = 0
    rho=0; gamma=0; K_features=0
    iterations=None

    if Model == "MS": #Non-linear model
        m, x_vars, w_vars, y_vars, I_vars, A, D_term = Multistage_problem(
                                                            seed, stages, scenarios, A, price, L, det, mult, add, a, b, C, H, pi, branching, I0)
            

    elif Model == "MS_linear": # Linear model (base model)
        m, x_vars, w_vars, y_vars, I_vars, A, D_term = MS_linear(
                                                            seed, stages, scenarios, A, price, L, det, mult, add, a, b, C, H, pi, branching, I0, W_cts)

    elif Model == "TS_linear": # Two stage version of MS_linear
        m, x_vars, w_vars, y_vars, I_vars, A, D_term = TS_linear(
                                                            seed, stages, scenarios, A, price, L, det, mult, add, a, b, C, H, pi, branching, I0, W_cts)


    elif Model == "MS_linear_affine":
        if W_cts:
            m, x_vars, w_vars, y_vars, I_vars, A, D_term, rho, gamma, K_features = MS_affine_cts(
                seed, stages, scenarios, A, price, L, det, mult, add, a, b, C, H, pi, branching, I0,
                phi_mode=cfg.run.phi_mode)
        else:
            m, x_vars, w_vars, y_vars, y_bar, I_vars, A, D_term, rho, gamma, K_features = MS_linear_affine(
                seed, stages, scenarios, A, price, L, det, mult, add, a, b, C, H, pi, branching, I0,
                phi_mode=cfg.run.phi_mode)

    elif Model == "TS_linear_affine":
        m, x_vars, w_vars, y_vars, I_vars, A, D_term, _, _ = TS_linear_affine(
            seed, stages, scenarios, A, price, L, det, mult, add, a, b, C, H, pi, I0,
            phi_mode=cfg.run.phi_mode)


    elif Model == "AF_EVAL":
        m, x_vars, w_vars, y_vars, I_vars, A, D_term, solve_time, rho_opt, Gamma_opt = Affine_eval(
            seed, stages, scenarios, A, price, L, det, mult, add, a, b, C, H, pi, branching, I0,
            phi_mode=cfg.run.phi_mode)

    elif Model == "REL_EVAL":
        m, x_vars, w_vars, y_vars, I_vars, A, D_term, solve_time = Relaxed_eval(
            seed, stages, scenarios, A, price, L, det, mult, add, a, b, C, H, pi, branching, I0)
        # Relaxed_eval no usa phi (resuelve MS_linear relajado en w, no la política afín) — sin cambios


    elif Model == "IH":
        m, x_vars, w_vars, y_vars, I_vars, A, D_term, solve_time, iterations = Iter_policy(
            seed, stages, scenarios, A, price, L, det, mult, add, a, b, C, H, pi, branching, I0,
            max_iter=cfg.iter, phi_mode=cfg.run.phi_mode)

#=====================================================================================================================================
    else:
        print("\nWrong input, try again...")
        return None
#=====================================================================================================================================
    if m is None: # if a model is unfeasible
        print(f"\n{Model} unfeasible. Returning NaN values\n")
        return {
            "incumbent": None,
            "bestbd": None,
            "gap": None,
            "time": None,
            "iterations (IH)": None,
            "vss": None,
            "evpi": None,
            "vss_ts": None
        }
#=====================================================================================================================================
#   Gurobi Parameters:
    m.setParam('TimeLimit', time_limit)
    m.setParam('OutputFlag', 1)
    m.setParam('BarHomogeneous', 1)   
    # m.setParam('Crossover', 1)       # fuerza crossover a solución de vértice
    # m.setParam('BarConvTol', 1e-9)   # tolerancia de convergencia más estricta
    # m.setParam("MIPGap", 5e-4) # Gap tol: 0.05% // Gurobi base tol: 0.01%/1e-4
    # m.Params.NonConvex = 2
#=====================================================================================================================================
    m.optimize() # ¡Where magic happens! (or not)
#=====================================================================================================================================
    # LOCAL SEARCH SECTION
    # Verifica que exista una solución óptima del paso 3 y que sea un modelo compatible
    if m.status == GRB.OPTIMAL and getattr(cfg.run, 'local_search', False) and Model in ["AF_EVAL", "REL_EVAL", "IH"]:
        
        # 1. Definir los sets iterables para el modelo matemático
        J_list = list(range(prod))
        T_list = list(range(stages))
        S_list = list(range(scenarios))

        # 2. Extraer el w inicial desde las cotas fijadas por AF_EVAL / REL_EVAL
        initial_w = {}
        for j in J_list:
            for t in T_list:
                for p_idx in range(len(price)):
                    for s in S_list:
                        initial_w[(j, t, p_idx, s)] = w_vars[j, t, p_idx, s].lb

        # 3. Necesitas proveer la estructura de partición de información F_{t-1}
        # Debes asegurar que cfg.size.scenario_groups contenga la matriz de nodos no-anticipativos
        scenario_groups = cfg.size.scenario_groups 
        
        # 4. Ejecutar heurística First Improvement
        ls_start = time.time()
        final_w, best_obj = local_search_first_improvement(ms_model=m,w_var=w_vars,initial_w=initial_w,
            J=J_list, T=T_list, S=S_list, P=price,scenario_groups=scenario_groups, 
            ls_iterations=10) # n° iterations for local search. The less has the more tractable it is
        
        ls_time = time.time() - ls_start
        solve_time += ls_time
        
        incumbent = best_obj
        bestbd = m.objBound 
        gap = 0.0
#=====================================================================================================================================
    # OUT-OF-SAMPLE (OOS) SECTION
    if m.status == GRB.OPTIMAL and getattr(cfg.run, 'oos_eval', False) and Model == "AF_EVAL":
        print("\n--- Iniciando evaluación Out-of-Sample (OOS) ---")

        oos_start = time.time()

        oos_seed = seed + 1000
        oos_branching = cfg.size.oos_branching
        oos_scenarios = cfg.size.oos_scenarios

        # 1. Generamos los tensores de incertidumbre respetando la estructura de branching OOS
        ypsilon_oos = epsilon_ms(inst, stages, oos_scenarios, oos_branching, oos_seed,
                                  cfg.demand.lb_epsilon, cfg.demand.ub_epsilon)
        delta_oos = delta_ms(inst, prod, stages, oos_scenarios, oos_branching, oos_seed,
                              cfg.demand.mu_delta, cfg.demand.std_delta)

        if det:
            L_oos = L
        else:
            L_oos = lead_times_ms(inst, comp, stages, oos_scenarios, oos_branching, oos_seed,
                                   cfg.lead_times.lb_L, cfg.lead_times.ub_L, det)

        # 2. Probabilidades reconstruidas para el tamaño del árbol OOS (uniforme)
        pi_oos = [1.0 / oos_scenarios] * oos_scenarios

        # 3. Rollout de la política afín sobre el árbol OOS
        m_oos, x_oos, w_oos, y_oos, I_oos, A_oos, D_oos = Affine_OOS_eval(
            oos_seed, stages, oos_scenarios, A, price, L_oos, det,
            ypsilon_oos, delta_oos, a, b, C, H, pi_oos, oos_branching, I0,
            rho_opt, Gamma_opt, phi_mode=cfg.run.phi_mode
        )

        m_oos.setParam('TimeLimit', time_limit)
        m_oos.setParam('OutputFlag', 1)
        m_oos.optimize()

        oos_time = time.time() - oos_start
        solve_time += oos_time

        if m_oos.status == GRB.OPTIMAL:
            incumbent = m_oos.objVal
            bestbd = m_oos.objBound
            gap = 0.0
            m, x_vars, w_vars, y_vars, I_vars, D_term = m_oos, x_oos, w_oos, y_oos, I_oos, D_oos
            scenarios = oos_scenarios
            pi = pi_oos
            mult = ypsilon_oos
            add = delta_oos
        else:
            print("Evaluación OOS fallida o infactible.")
#=====================================================================================================================================

    if m.status == GRB.OPTIMAL:
        incumbent = m.objVal
        bestbd = m.objBound
        gap = 0.0

    elif m.status == GRB.TIME_LIMIT:
        if m.SolCount > 0:
            incumbent = m.objVal
            bestbd = m.objBound
            gap = m.MIPGap
        else:
            print("====== No solution available ======")
            incumbent = None
            bestbd = m.objBound
            gap = None

    else:
        incumbent = None
        bestbd = None
        gap = None

    opt_time = m.Runtime + solve_time
    vss, evpi, vss_ts = uncertainty_analysis(cfg, incumbent)

    solutions = [seed, x_vars, w_vars, I_vars, y_vars, D_term, price, stages, scenarios, A, det,
                 Model, rho, gamma, K_features, mult, add, a, b, pi, I0, C, H]

    results = {"incumbent": incumbent, "bestbd": bestbd, "gap": gap, "time": opt_time, "iterations (IH)": iterations, "vss": vss, "evpi": evpi, "vss_ts": vss_ts}

    if show_kpis:
        results["Fill Rate"], results["Utilization Rate"], results["Inventory Service Rate"], results["Weighted Avg Price"], results["Exp_Rev"], results["Exp_comp_cost"], results["Exp_Inv_Cost"] = export(
            show_sol, show_heatmap, show_boxplot, show_candlestick, show_kpis, solutions)
        return results
    else:
        return results
