import numpy as np
import matplotlib.pyplot as plt


def plot_candlestick(data: np.ndarray, title: str, ylabel: str, filename: str,
                                p_low=10, p_high=90, ylim_bottom=None):
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
    if ylim_bottom is not None:
        ax.set_ylim(bottom=ylim_bottom)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_fulfillment_candlestick(fill_ts, filename="figures/fill_rate_candlestick.png"):
    plot_candlestick(
        fill_ts,
        title="Demand Fulfillment Rate Over Time",
        ylabel="Fill Rate (%)",
        filename=filename,
        ylim_bottom=0
    )