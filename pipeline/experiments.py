import copy
import os
import math
import pandas as pd
from pipeline.instances import instances

EXPERIMENTS_DIR = "var_results/experiments"


def run_parameter(base_cfg, axis_name, values, apply_fn):
    """
    base_cfg  : ExperimentConfig ya construido con los valores default/base.
    axis_name : etiqueta del eje (usada como nombre de archivo y columna), p.ej. "comp", "branching".
    values    : lista de valores a barrer. Pueden ser escalares (int, str) o estructuras
                (p.ej. listas de branching); apply_fn decide qué hacer con cada uno.
    apply_fn  : function(cfg, value) -> None. Muta cfg IN-PLACE para reflejar ese valor del eje.
                Es responsabilidad de apply_fn mantener la consistencia entre sub-configs
                (p.ej. si cambia `branching`, debe recalcular `scenarios`).
    """
    all_rows = []

    for val in values:
        cfg = copy.deepcopy(base_cfg)
        apply_fn(cfg, val)

        results, seeds = instances(cfg)

        for res, seed in zip(results, seeds):
            row = dict(res)
            row["axis"] = axis_name
            row["axis_value"] = str(val)   # str() porque val puede ser una lista (branching)
            row["Model"] = cfg.run.Model
            row["seed"] = seed
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    out_path = f"{EXPERIMENTS_DIR}/sweep_{axis_name}.xlsx"
    df.to_excel(out_path, index=False)
    print(f"\n>> Sweep '{axis_name}' guardado en: {out_path} ({len(df)} filas, {len(values)} valores x {len(seeds)} seeds)")
    return df


# ------------------------------------------------------------------
# apply_fn de referencia para cada eje descrito en la tabla de la imagen
# ------------------------------------------------------------------

def apply_comp(cfg, comp_val):
    """Varía #componentes, fijando exactamente 3 componentes por producto (BOM aleatoria)."""
    cfg.size.inst = "None"
    cfg.size.comp = comp_val


def apply_prod(cfg, prod_val):
    """Varía #productos, mismo criterio de BOM."""
    cfg.size.inst = "None"
    cfg.size.prod = prod_val

def apply_branching(cfg, branching_val):
    """
    branching_val: lista completa de branching, p.ej. [4,2,2,2,1,1,1].
    Se recalcula scenarios = prod(branching) y se ajusta stages si el largo cambia.
    """
    cfg.size.branching = branching_val
    cfg.size.stages = len(branching_val) + 1
    cfg.size.scenarios = math.prod(branching_val)

def apply_phi_mode(cfg, phi_mode_val):
    cfg.run.phi_mode = phi_mode_val