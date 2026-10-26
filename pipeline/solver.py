from data.extract import extract_params

from models.multistage import Multistage_problem
from models.linealization_prop import MS_linear, TS_linear

from sol_approach.affine_funct_app import MS_linear_affine, MS_affine_cts, TS_linear_affine
from sol_approach.price_app_eval import Affine_eval, Relaxed_eval
from sol_approach.iterative_heuristic import Iter_policy, Iter_policy_strong, Iter_policy_relaxed
from sol_approach.local_search import local_search_first_improvement
from sol_approach.benders import benders, benders_rel
from sol_approach.rolling_horizon import Rolling_horizon, Rolling_horizon_aff, Rolling_horizon_rel

from uncertainty_analysis.sto_computation import uncertainty_analysis
from output_config.results_output import export

import time
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
    max_iter = cfg.run.max_iter

    show_sol = cfg.Output.show_sol
    show_heatmap = cfg.Output.show_heatmap
    show_boxplot = cfg.Output.show_boxplot
    show_candlestick = cfg.Output.show_candlestick
    show_kpis = cfg.Output.show_kpis
    
    if inst is not None:
        comp = len(A); prod = len(A[0])
        print(f"\n Model: {Model} | seed: {seed} | Components: {comp} | Products: {prod} | Stages: {stages} | Scenarios: {scenarios} ")

    solve_time = 0
    rho=0; gamma=0; K_features=0
    iterations=None

#=====================================================================================================================================
# Non-linear ATO pricing model
#=====================================================================================================================================
    if Model == "MS": #Non-linear model
        m, x_vars, w_vars, y_vars, I_vars, A, D_term = Multistage_problem(
                                                            seed, stages, scenarios, A, price, L, det, mult, add, a, b, C, H, pi, branching, I0)
#=====================================================================================================================================
# Linearized ATO pricing model (MS and 2S)
#=====================================================================================================================================
    elif Model == "MS_linear": # Linear model (base model)
        m, x_vars, w_vars, y_vars, I_vars, A, D_term = MS_linear(
                                                            seed, stages, scenarios, A, price, L, det, mult, add, a, b, C, H, pi, branching, I0, W_cts)

    elif Model == "TS_linear": # Two stage version of MS_linear
        m, x_vars, w_vars, y_vars, I_vars, A, D_term = TS_linear(
                                                            seed, stages, scenarios, A, price, L, det, mult, add, a, b, C, H, pi, branching, I0, W_cts)
#=====================================================================================================================================
# Rolling Horizon for MS_linear // AF_EVAL // REL_EVAL
#=====================================================================================================================================
    elif Model == "RH":
        sampling = dict(inst=inst,
                        lb_eps=cfg.demand.lb_epsilon, ub_eps=cfg.demand.ub_epsilon,
                        mu=cfg.demand.mu_delta, std=cfg.demand.std_delta,
                        lb_L=cfg.lead_times.lb_L, ub_L=cfg.lead_times.ub_L)
        m, x_vars, w_vars, y_vars, I_vars, A, D_term, solve_time = Rolling_horizon(
            seed, stages, scenarios, A, price, L, det, mult, add, a, b, C, H, pi, branching, I0,
            horizon=cfg.run.rh_horizon, fix_x=cfg.run.rh_fix_x,
            mip_gap=cfg.run.rh_mip_gap, sub_time_limit=cfg.run.rh_sub_time,
            resample=cfg.run.rh_resample, sampling=sampling)

    elif Model == "RH_AF":
        sampling = dict(inst=inst,
                                lb_eps=cfg.demand.lb_epsilon, ub_eps=cfg.demand.ub_epsilon,
                                mu=cfg.demand.mu_delta, std=cfg.demand.std_delta,
                                lb_L=cfg.lead_times.lb_L, ub_L=cfg.lead_times.ub_L)
        m, x_vars, w_vars, y_vars, I_vars, A, D_term, solve_time = Rolling_horizon_aff(
            seed, stages, scenarios, A, price, L, det, mult, add, a, b, C, H, pi, branching, I0,
            horizon=cfg.run.rh_horizon, fix_x=cfg.run.rh_fix_x,
            mip_gap=cfg.run.rh_mip_gap, sub_time_limit=cfg.run.rh_sub_time,
            resample=cfg.run.rh_resample, sampling=sampling, 
            phi_mode=cfg.run.phi_mode)

    elif Model == "RH_REL":
        sampling = dict(inst=inst,
                                lb_eps=cfg.demand.lb_epsilon, ub_eps=cfg.demand.ub_epsilon,
                                mu=cfg.demand.mu_delta, std=cfg.demand.std_delta,
                                lb_L=cfg.lead_times.lb_L, ub_L=cfg.lead_times.ub_L)
        m, x_vars, w_vars, y_vars, I_vars, A, D_term, solve_time = Rolling_horizon_rel(
                seed, stages, scenarios, A, price, L, det, mult, add, a, b, C, H, pi, branching, I0,
                horizon=cfg.run.rh_horizon, fix_x=cfg.run.rh_fix_x,
                mip_gap=cfg.run.rh_mip_gap, sub_time_limit=cfg.run.rh_sub_time,
                resample=cfg.run.rh_resample, sampling=sampling)

#=====================================================================================================================================
# LDR pricing policy approach for MS_linear and TS_linear
#=====================================================================================================================================
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
#=====================================================================================================================================
# Policy evaluation for MS_linear and TS_linear (LDR and binary relaxation)
#=====================================================================================================================================
    elif Model == "AF_EVAL":
        m, x_vars, w_vars, y_vars, I_vars, A, D_term, solve_time, rho_opt, Gamma_opt = Affine_eval(
            time_limit, seed, stages, scenarios, A, price, L, det, mult, add, a, b, C, H, pi, branching, I0,
            phi_mode=cfg.run.phi_mode)

    elif Model == "REL_EVAL":
        m, x_vars, w_vars, y_vars, I_vars, A, D_term, solve_time = Relaxed_eval(
            time_limit, seed, stages, scenarios, A, price, L, det, mult, add, a, b, C, H, pi, branching, I0)
        # Relaxed_eval no usa phi (resuelve MS_linear relajado en w, no la política afín) — sin cambios
#=====================================================================================================================================
# iterative heuristic approaches
#=====================================================================================================================================
    elif Model == "IH":
        m, x_vars, w_vars, y_vars, I_vars, A, D_term, solve_time, iterations = Iter_policy(
            seed, stages, scenarios, A, price, L, det, mult, add, a, b, C, H, pi, branching, I0, max_iter=max_iter, phi_mode=cfg.run.phi_mode, time_limit=time_limit,)

    elif Model == "IH_SC":
        m, x_vars, w_vars, y_vars, I_vars, A, D_term, solve_time, iterations = Iter_policy_strong(
            seed, stages, scenarios, A, price, L, det, mult, add, a, b, C, H, pi, branching, I0, max_iter=max_iter, phi_mode=cfg.run.phi_mode, time_limit=time_limit,)

    elif Model == "IH_REL":
        m, x_vars, w_vars, y_vars, I_vars, A, D_term, solve_time, iterations = Iter_policy_relaxed(
            seed, stages, scenarios, A, price, L, det, mult, add, a, b, C, H, pi, branching, I0, max_iter=max_iter, phi_mode=cfg.run.phi_mode, time_limit=time_limit,)
#=====================================================================================================================================
# Benders approaches
#=====================================================================================================================================
    elif Model == "BENDERS":
        m, x_vars, w_vars, y_vars, I_vars, A, D_term, solve_time, iterations = benders(
            seed, stages, scenarios, A, price, L, det, mult, add, a, b, C, H, pi, branching, I0, max_iter=max_iter, time_limit=time_limit,)

    elif Model == "BENDERS_SC":
        m, x_vars, w_vars, y_vars, I_vars, A, D_term, solve_time, iterations = benders(
            seed, stages, scenarios, A, price, L, det, mult, add, a, b, C, H, pi, branching, I0, max_iter=max_iter, strong_cuts=True, time_limit=time_limit,)

    elif Model == "BENDERS_REL":
        m, x_vars, w_vars, y_vars, I_vars, A, D_term, solve_time, iterations = benders_rel(
            seed, stages, scenarios, A, price, L, det, mult, add, a, b, C, H, pi, branching, I0, max_iter=max_iter, time_limit=time_limit,)
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
    m.setParam('BarHomogeneous', 1)    # homogeneous barrier algorithm
    # m.setParam('Crossover', 1)       # fuerza crossover a solución de vértice
    # m.setParam('BarConvTol', 1e-9)   # tolerancia de convergencia más estricta
    # m.setParam("MIPGap", 5e-4)       # Gap tol: 0.05% // Gurobi base tol: 0.01%/1e-4
    # m.Params.NonConvex = 2           # Allows non-convex quadratic constraints (for MS_linear_affine)
    m.setParam('Threads', 1)           # Use single thread for reproducibility (Gurobi uses multiple threads by default)
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

    if Model in ("BENDERS", "BENDERS_SC", "BENDERS_REL",) and hasattr(m, "_benders_lb"):
        incumbent = m._benders_lb
        bestbd    = m._benders_ub
        gap       = max(0.0, (bestbd - incumbent) / max(1.0, abs(incumbent)))

    if Model in ("IH", "IH_SC", "IH_REL") and hasattr(m, "_ih_lb"):
        incumbent = m._ih_lb
        bestbd    = m._ih_ub
        gap       = max(0.0, (bestbd - incumbent) / max(1.0, abs(incumbent)))

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
