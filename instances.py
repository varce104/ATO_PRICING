from models.solver import solve
from output_config.mean_var import average_excel_solutions
import pandas as pd


def instances(size, bom, costs, price_param, demand, lead_times, show, iter):
    inst, comp, prod, stages, scenarios, branching, seed, time_limit = size
    show_var, _,_,_,_,_,_,_,_,_,_ = show

    if iter == 1:
        solve(size, bom, costs, price_param, demand, lead_times, show)
    else:
        seeds = [seed + i for i in range(iter)]
        results = []
        files = []

        for seed in seeds:
            size = inst, comp, prod, stages, scenarios, branching, seed, time_limit
            res = solve(size, bom, costs, price_param, demand, lead_times, show)
            results.append(res)
            files.append(f"var_results/MS_SL_inst{seed}.xlsx")
        df = pd.DataFrame(results)
        print(df)
        df.to_csv(f"var_results/metrics_inst.csv", index=True)
        if show_var:
            average_excel_solutions(files, output_path="var_results/mean_var_by_inst/avg_sol.xlsx")