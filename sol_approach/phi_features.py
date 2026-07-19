
PHI_MODES = {
    "eps":          {"delta": False, "lead_time": False},
    "eps_delta":    {"delta": True,  "lead_time": False},  # comportamiento histórico / default
    "eps_lt":       {"delta": False, "lead_time": True},
    "eps_delta_lt": {"delta": True,  "lead_time": True},
}


def phi_dimension(mode, prod, comp):
    if mode not in PHI_MODES:
        raise ValueError(f"phi_mode desconocido: '{mode}'. Opciones: {list(PHI_MODES)}")
    cfg = PHI_MODES[mode]
    n = 1  # ypsilon siempre está presente
    if cfg["delta"]:
        n += prod
    if cfg["lead_time"]:
        n += comp
    return n


def build_phi(mode, time, scenarios, prod, comp, ypsilon, delta, L=None, L_det=None):
    """
    mode: "eps" | "eps_delta" | "eps_lt" | "eps_delta_lt"
    L / L_det son obligatorios si el modo incluye lead times.
    """
    if mode not in PHI_MODES:
        raise ValueError(f"phi_mode unknown: '{mode}'. Options: {list(PHI_MODES)}")
    cfg = PHI_MODES[mode]

    if cfg["lead_time"] and L is None:
        raise ValueError(f"phi_mode='{mode}' require lead times (L).")

    K_features = (time - 1) * phi_dimension(mode, prod, comp)

    phi = {}
    for s in range(scenarios):
        phi[s] = {}
        for t in range(time):
            phi[s][t] = []
            for tau in range(time - 1):
                if tau < t:
                    phi[s][t].append(ypsilon[s][tau])
                    if cfg["delta"]:
                        for j in range(prod):
                            phi[s][t].append(delta[s][j][tau])
                    if cfg["lead_time"]:
                        for i in range(comp):
                            lt_val = L[i][tau] if L_det else L[i][tau][s]
                            phi[s][t].append(lt_val)
                else:
                    phi[s][t].append(0)
                    if cfg["delta"]:
                        for j in range(prod):
                            phi[s][t].append(0)
                    if cfg["lead_time"]:
                        for i in range(comp):
                            phi[s][t].append(0)

    return phi, K_features