# Assemble-to-order + Pricing decision problem

The following library optimizes a multistage assemble-to-order model, where component replenishment, production and pricing decisions are made. These decisions are taken under an price-sensitive uncertain demand, as well as a positive and uncertain component lead times. 

## Requirements
 
- Python 3.10+
- [Gurobi](https://www.gurobi.com/) (with valid license)
- `gurobipy`, `numpy`, `pandas`, `openpyxl`, `matplotlib`

## Project Structure
 
```
ATO_PRICING/
├── main.py                        # Entry point. All parameters are set here.
│
├── instances.py                   # Runs single or multiple seeds; aggregates results.
├── solver.py                      # Model dispatcher: builds, solves, and routes outputs.
│
├── data/
│   ├── generator.py               # Generates stochastic scenario tree (ε, δ, lead times).
│   └── params.py                  # BOM, cost, and price set definitions. Includes literature instances.
│
├── models/
│   ├── multistage.py              # Standard multistage stochastic model (nonlinear objective).
│   ├── linealization_prop.py      # Linearized multistage model (MILP via auxiliary variable r).
│   ├── twostage_dlt_slt.py        # Two-stage recourse model (deterministic/stochastic lead times).
│   └── affine_funct_app.py        # Multistage affine function approximation (continuous relaxation of w).
│
├── sol_approach/
│   ├── iterative_heuristic.py     # Iterative approach that decouples pricing decision from inventory problem (or ATO problem)
│   ├── price_app_eval.py          # Evaluation of pricing approximation obtained from other models.
│   └── twostage_affine.py         # Two-stage version of the affine approximation.
│
├── uncertainty_analysis/
│   ├── vss.py                     # Value of the Stochastic Solution (VSS).
│   ├── evpi.py                    # Expected Value of Perfect Information (EVPI).
│   ├── vss_ts.py                  # VSS relative to two-stage model.
│   └── sto_computation.py         # Orchestrates uncertainty metrics.
│
├── output_config/
│   ├── results_output.py          # Main export dispatcher.
│   ├── var_export.py              # Extracts and exports solution arrays to Excel.
│   ├── lambda_export.py           # Exports affine policy (λ) results.
│   ├── mean_var.py                # Averages solutions across multiple seeds.
│   └── graphs/
│       ├── boxplot.py
│       ├── candlestick.py
│       └── heatmap.py
│
├── figures/                       # Output figures (generated at runtime).
└── var_results/                   # Output Excel files (generated at runtime).
```
 
---

## Usage
 
All configuration is done directly in `main.py`. Key parameters:
 
**Problem size**
```python
comp    = 3        # number of components
prod    = 2        # number of products
stages  = 9        # number of time periods
branching = [2] * (stages - 1)   # scenario tree branching factor
```
 
**Demand model** — linear: `D(j,t,s) = ε[s][t] * (a - b*P) + δ[s][j][t]`
```python
a, b = 200, 2.0
lb_epsilon, ub_epsilon = 0.5, 1.5   # ε ~ U[lb, ub]
mu_delta, std_delta    = 0, 8        # δ ~ N(mu, sigma)
```
 
**Lead times**
```python
lb_L, ub_L = 1, 2
det = False   # True → deterministic (averaged), False → stochastic
```

**Model selection** — set `Model` to one of:
 
| Key | Description |
|-----|-------------|
| `MS` | Standard multistage (nonlinear) |
| `MS_linear` | Linearized multistage MILP |
| `MS_linear_affine` | Linearized + affine function approximation |
 
**Literature instances** — set `inst` to one of:
 
| Key | Description |
|-----|-------------|
| `Oh_et_al_1` | W-model, 3 components × 2 products |
| `Oh_et_al_2` | 5 components × 4 products |
| `Oh_et_al_3` | 11 components × 11 products |
| `None` | Manual parameter input |

---
 
## Multiple Seeds
 
Set `iter > 1` in `main.py` to run the same instance over several seeds. Results are saved to `var_results/metrics_inst.csv`. If `show_var = True`, averaged solutions are also exported to `var_results/mean_var_by_inst/avg_sol.xlsx`.
 
---

## Uncertainty Analysis
 
Set any of the following flags to `True` in `main.py`:
 
- `vss_calc` — Value of the Stochastic Solution (multistage)
- `evpi_calc` — Expected Value of Perfect Information
- `vss_ts_calc` — VSS relative to the two-stage model
Results are printed to console and included in the CSV output.
 
---

