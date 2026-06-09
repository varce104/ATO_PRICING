import numpy as np
import matplotlib.pyplot as plt

def plot_candlestick_over_time(data: np.ndarray, title: str, ylabel: str, filename: str,
                                p_low=10, p_high=90):
    """
    data: shape (time, scenarios)
    Muestra: rango total (sombra tenue), percentiles p_low/p_high (caja), mediana (línea).
    """
    T = data.shape[0]
    periods = np.arange(1, T + 1)

    #median  = np.median(data, axis=1)
    mean    = np.mean(data, axis=1)
    p_lo    = np.percentile(data, p_low,  axis=1)
    p_hi    = np.percentile(data, p_high, axis=1)
    v_min   = np.min(data, axis=1)
    v_max   = np.max(data, axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.fill_between(periods, v_min, v_max, alpha=0.15, color="steelblue", label="Min–Max")

    ax.fill_between(periods, p_lo, p_hi, alpha=0.4, color="steelblue",
                    label=f"P{p_low}–P{p_high}")

    #ax.plot(periods, median, color="navy",   linewidth=2,   label="Median")
    ax.plot(periods, mean,   color="tomato", linewidth=1.5, linestyle="--", label="Mean")

    ax.set_xticks(periods)
    ax.set_xticklabels([f"t={t+1}" for t in range(T)])
    ax.set_xlabel("Period")
    ax.set_ylim(bottom=40)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_candlestick(x_val, price_eff, I_exp, y_exp, D_exp):
    x_val = np.array(x_val)
    price_eff = np.array(price_eff)
    I_exp = np.array(I_exp)
    y_exp = np.array(y_exp)
    D_exp = np.array(D_exp)

    for i in range(x_val.shape[0]):
        plot_candlestick_over_time(
            x_val[i, :, :],
            title="Procurement x[i,t] Distribution Over Time",
            ylabel="Quantity",
            filename=f"figures/x_{i+1}_candlestick.png")

        plot_candlestick_over_time(
            I_exp[i, :, :],
            title="Inventory I[i,t] Distribution Over Time",
            ylabel="Inventory Level",
            filename=f"figures/I_{i+1}_candlestick.png")
    
    for j in range(y_exp.shape[0]):

        plot_candlestick_over_time(
            price_eff[j, :, :],
            title="Effective Price w[j,t] Distribution Over Time",
            ylabel="Price",
            filename=f"figures/w_{j+1}_candlestick.png")
        
        plot_candlestick_over_time(
            y_exp[j, :, :],
            title="Assembly y[j,t] Distribution Over Time",
            ylabel="Quantity Assembled",
            filename=f"figures/y_{j+1}_candlestick.png")
        
        plot_candlestick_over_time(
            D_exp[j, :, :],
            title="Effective Demand D[j,t] Distribution Over Time",
            ylabel="Demand Quantity",
            filename=f"figures/D_{j+1}_candlestick.png")