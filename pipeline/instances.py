import os
import pandas as pd
from pipeline.solver import solve
from output_config.mean_var import average_excel_solutions

def append_run_results(model_name, results, seeds, PATH="var_results/all_runs.xlsx"):
    df_new = pd.DataFrame(results)
    df_new.insert(0, "seed", seeds)
    df_new.insert(0, "Model", model_name)

    if os.path.exists(PATH):
        df_old = pd.read_excel(PATH)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_excel(PATH, index=False)
    print(f"\n>> Resultados acumulados en: {PATH} ({len(df_all)} corridas totales)\n")
    return df_all


def instances(cfg):
    seeds = [cfg.size.seed + i for i in range(cfg.iter)] if cfg.iter > 1 else [cfg.size.seed]
    results, files = [], []

    for seed in seeds:
        cfg.size.seed = seed
        res = solve(cfg)
        results.append(res)
        files.append(f"var_results/MS_inst_{seed}.xlsx")

    append_run_results(cfg.run.Model, results, seeds)

    if cfg.Output.show_var:
        average_excel_solutions(files, output_path="var_results/mean_var_by_inst/avg_sol.xlsx")

    return results, seeds