import os
import pandas as pd
from pipeline.solver import solve
from output_config.mean_var import average_excel_solutions

def append_run_results_OLD(model_name, results, seeds, PATH="var_results/all_runs.xlsx"):
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

def instances_OLD(cfg):
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

def append_run_results(model_name, results, seeds, PATH="var_results/all_runs.xlsx", param_name=None, param_value=None):
    df_new = pd.DataFrame(results)
    df_new.insert(0, "seed", seeds)
    
    # Inserción dinámica: Coloca una columna titulada "comp" o "prod" en la posición índice 0.
    if param_name and param_value is not None:
        df_new.insert(0, param_name, param_value)
        
    df_new.insert(0, "Model", model_name)

    # Lógica de concatenación: Si el archivo existe (ej. comp_res.xlsx), 
    # se lee el DataFrame antiguo y se fusiona por debajo (axis=0).
    if os.path.exists(PATH):
        df_old = pd.read_excel(PATH)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_excel(PATH, index=False)
    print(f"\n>> Resultados acumulados en: {PATH} ({len(df_all)} corridas totales)\n")
    return df_all

def instances(cfg, tag=None, param_name=None, param_value=None):
    seeds = [cfg.size.seed + i for i in range(cfg.iter)] if cfg.iter > 1 else [cfg.size.seed]
    results, files = [], []

    for seed in seeds:
        cfg.size.seed = seed
        res = solve(cfg)
        results.append(res)
        files.append(f"var_results/MS_inst_{seed}.xlsx")
    # El tag enruta el archivo a su propio Excel, separando por experimentos.
    out_path = f"var_results/{tag}_res.xlsx" if tag else "var_results/all_runs.xlsx"
    append_run_results(cfg.run.Model, results, seeds, PATH=out_path, param_name=param_name, param_value=param_value)
    # if cfg.Output.show_var:
    #     average_excel_solutions(files, output_path="var_results/mean_var_by_inst/avg_sol.xlsx")

    return results, seeds