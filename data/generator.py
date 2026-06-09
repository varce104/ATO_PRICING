import random
import math

def epsilon_ms(inst, time, scenarios, branching_structure, seed, lb, ub):
    random.seed(seed)

    if inst == "Oh_et_al_1":
        lb = 0.9; ub = 1.1

    elif inst == "Oh_et_al_2":
        lb = 0.8; ub = 1.2
        # data = [[0.0 for t in range(time)] for s in range(scenarios)]
        # for t in range(time):
        #     val = 0.4 * (1 + 0.6 * math.sin((math.pi / 2) * t))
        #     for s in range(scenarios):
        #         data[s][t] = val
        # return data
    
    elif inst == "Oh_et_al_3":
        lb = 0.5; ub = 1.5

    else:
        pass

    data = [[0.0 for t in range(time)] for s in range(scenarios)]
    structure = branching_structure + [1]*(time - len(branching_structure))
    n_groups = 1
    for t in range(time):
        scenarios_per_group = int(scenarios / n_groups)
        for g in range(n_groups):
            val = random.uniform(lb, ub)
            first_scen = g * scenarios_per_group
            for k in range(scenarios_per_group):
                data[first_scen + k][t] = val
        n_groups = n_groups * structure[t] 
    return data


def delta_ms(inst, prod, time, scenarios, branching_structure, seed, mu, sigma):
    random.seed(seed)

    if inst == "Oh_et_al_1":
        mu = 0; sigma = 10

    elif inst == "Oh_et_al_2":
        mu = 0; sigma = 2    

    elif inst == "Oh_et_al_3":
        mu = 0; sigma = 10

    else:
        pass

    data = [[[0.0 for t in range(time)] for j in range(prod)] for s in range(scenarios)]
    structure = branching_structure + [1]*(time - len(branching_structure))
    n_groups = 1
    for t in range(time):
        scenarios_per_group = int(scenarios / n_groups)
        for g in range(n_groups):
            vals_prod = [max(0.0, random.gauss(mu, sigma)) for _ in range(prod)]
            first_scen = g * scenarios_per_group
            for k in range(scenarios_per_group):
                s = first_scen + k
                for j in range(prod):
                    data[s][j][t] = vals_prod[j]    
        n_groups = n_groups * structure[t] 
    return data



def lead_times_ms(inst, comp, time, scenarios, branching_structure, seed, lb=None, ub=None, det=False):
    random.seed(seed)
    
    L_stochastic = [[[0 for s in range(scenarios)] for t in range(time)] for i in range(comp)]
    structure = branching_structure + [1]*(time - len(branching_structure))
    n_groups = 1
    for t in range(time):
        scenarios_per_group = int(scenarios / n_groups)
        for g in range(n_groups):
            vals_comp = [max(0, int(random.randint(lb, ub))) for _ in range(comp)]
            first_scen = g * scenarios_per_group
            for k in range(scenarios_per_group):
                s = first_scen + k
                for i in range(comp):
                    L_stochastic[i][t][s] = vals_comp[i]
        n_groups = n_groups * structure[t] 

    if det:
        L_deterministic = [[0 for t in range(time)] for i in range(comp)]
        for i in range(comp):
            for t in range(time):
                avg_lt = sum(L_stochastic[i][t][s] for s in range(scenarios)) / scenarios
                L_deterministic[i][t] = int(round(avg_lt))
        return L_deterministic

    return L_stochastic