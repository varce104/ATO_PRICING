import numpy as np
import matplotlib.pyplot as plt

def plot_candlestick(data: np.ndarray, title: str, ylabel: str, filename: str,
                                p_low=10, p_high=90, ylim_bottom=None, ylim_top=None):
    """
    data: shape (time, scenarios)
    Muestra: rango total (sombra tenue), percentiles p_low/p_high (caja), media (línea).
    """
    T = data.shape[0]
    periods = np.arange(1, T + 1)

    mean    = np.mean(data, axis=1)
    p_lo    = np.percentile(data, p_low,  axis=1)
    p_hi    = np.percentile(data, p_high, axis=1)
    v_min   = np.min(data, axis=1)
    v_max   = np.max(data, axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.fill_between(periods, v_min, v_max, alpha=0.15, color="#3e944e", label="Min–Max")

    ax.fill_between(periods, p_lo, p_hi, alpha=0.4, color="#3e944e", label=f"P{p_low}–P{p_high}")

    ax.plot(periods, mean, color="#742626", linewidth=1.5, linestyle="--", label="Mean")

    ax.set_xticks(periods)
    ax.set_xticklabels([f"t={t+1}" for t in range(T)])
    ax.set_xlabel("Period")
    if ylim_bottom is not None or ylim_top is not None:
        ax.set_ylim(bottom=ylim_bottom, top=ylim_top)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    

def plot_candlestick_decisions(x_val, price_eff, I_exp, y_exp, D_exp, Model, seed, shared_ylim=True):
    x_val = np.array(x_val)
    price_eff = np.array(price_eff)
    I_exp = np.array(I_exp)
    y_exp = np.array(y_exp)
    D_exp = np.array(D_exp)

    # Máximo global por grupo de variable (todos los componentes/productos incluidos)
    x_top = float(x_val.max()) if shared_ylim else None
    I_top = float(I_exp.max()) if shared_ylim else None
    w_top = float(price_eff.max()) if shared_ylim else None
    y_top = float(y_exp.max()) if shared_ylim else None
    D_top = float(D_exp.max()) if shared_ylim else None

    for i in range(x_val.shape[0]):
        plot_candlestick(
            x_val[i, :, :],
            title=f"Component {i+1} Distribution Over Time",
            ylabel="Quantity",
            filename=f"figures/candlesticks/{Model}_{seed}_x_{i+1}.png",
            ylim_bottom=0, ylim_top=x_top
            )

        plot_candlestick(
            I_exp[i, :, :],
            title=f"Inventory {i+1} Distribution Over Time",
            ylabel="Inventory Level",
            filename=f"figures/candlesticks/{Model}_{seed}_I_{i+1}.png",
            ylim_bottom=0, ylim_top=I_top
            )

    for j in range(y_exp.shape[0]):
        plot_candlestick(
            price_eff[j, :, :],
            title=f"Effective Price w[{j+1},t] Distribution Over Time",
            ylabel="Price",
            filename=f"figures/candlesticks/{Model}_{seed}_w_{j+1}.png",
            ylim_bottom=0, ylim_top=w_top
            )

        plot_candlestick(
            y_exp[j, :, :],
            title=f"Assembly y[{j+1},t] Distribution Over Time",
            ylabel="Quantity Assembled",
            filename=f"figures/candlesticks/{Model}_{seed}_y_{j+1}.png",
            ylim_bottom=0, ylim_top=y_top
            )

        plot_candlestick(
            D_exp[j, :, :],
            title=f"Effective Demand D[{j+1},t] Distribution Over Time",
            ylabel="Demand Quantity",
            filename=f"figures/candlesticks/{Model}_{seed}_D_{j+1}.png",
            ylim_bottom=0, ylim_top=D_top
            )