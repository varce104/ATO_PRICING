# sto_computation.py
from data.generator import epsilon_ms, delta_ms, lead_times_ms
from data.params import parametros, bill_of_materials, price_set
from data.extract import extract_params

from uncertainty_analysis.vss import VSS
from uncertainty_analysis.evpi import EVPI
from uncertainty_analysis.vss_ts import VSS_2S

import numpy as np

def uncertainty_analysis(cfg, incumbent):
    C, H, pi, A, price, mult, add, L, a, b, I0 = extract_params(cfg)
    stages, scenarios, branching, seed = cfg.size.stages, cfg.size.scenarios, cfg.size.branching, cfg.size.seed
    det = cfg.lead_times.det
    vss_ms, evpi_ms, vss_ts = cfg.Output.vss_calc, cfg.Output.evpi_calc, cfg.Output.vss_ts_calc

    # Ahora pasamos 'cfg' como primer argumento a todas las funciones
    if vss_ms:
        vss = VSS(cfg, seed, stages, scenarios, A, price, L, det, mult, add, a, b, C, H, pi, branching, incumbent, I0)
        print(f"Value of Stochastic Solution (VSS): {vss:.4f}%")
    else:
        vss = -1

    if vss_ts:
        vss_2s = VSS_2S(cfg, seed, stages, scenarios, A, price, L, det, mult, add, a, b, C, H, pi, branching, incumbent, I0)
        print(f"Value of Stochastic Solution - Two Stage (VSS_TS): {vss_2s:.4f}%")
    else:
        vss_2s = -1

    if evpi_ms:
        evpi = EVPI(cfg, seed, stages, scenarios, A, price, L, det, mult, add, a, b, C, H, pi, branching, incumbent, I0)
        print(f"Expected Value of Perfect Information (EVPI): {evpi:.4f}%")
    else:
        evpi = -1

    return vss, evpi, vss_2s

