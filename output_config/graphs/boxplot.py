import numpy as np
import matplotlib.pyplot as plt


def plot_boxplot_over_time(data: np.ndarray, title: str, ylabel: str, filename: str):
    """
    data: shape (time, scenarios)
    Cada columna es un escenario, cada fila es un período.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.boxplot(
        [data[t, :] for t in range(data.shape[0])],
        positions=range(1, data.shape[0] + 1),
        patch_artist=True,
        boxprops=dict(facecolor="steelblue", alpha=0.6),
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(color="gray"),
        capprops=dict(color="gray"),
        flierprops=dict(marker="o", color="gray", alpha=0.4, markersize=4)
    )
    
    ax.set_xticks(range(1, data.shape[0] + 1))
    ax.set_xticklabels([f"t={t+1}" for t in range(data.shape[0])])
    ax.set_xlabel("Period")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_boxplot(x_val, price_eff, I_exp, y_exp, D_exp):
    x_val = np.array(x_val)
    price_eff = np.array(price_eff)
    I_exp = np.array(I_exp)
    y_exp = np.array(y_exp)
    D_exp = np.array(D_exp)

    for i in range(x_val.shape[0]):
        plot_boxplot_over_time(
            x_val[i, :, :],
            title=f"Component {i+1} Distribution Over Time",
            ylabel="Quantity",
            filename=f"figures/x_{i+1}_boxplot.png")

        plot_boxplot_over_time(
            I_exp[i, :, :],
            title=f"Inventory {i+1} Distribution Over Time",
            ylabel="Inventory Level",
            filename=f"figures/I_{i+1}_boxplot.png")
    
    for j in range(y_exp.shape[0]):

        plot_boxplot_over_time(
            price_eff[j, :, :],
            title=f"Effective Price w[{j+1},t] Distribution Over Time",
            ylabel="Price",
            filename=f"figures/w_{j+1}_boxplot.png")

        plot_boxplot_over_time(
            y_exp[j, :, :],
            title=f"Assembly y[{j+1},t] Distribution Over Time",
            ylabel="Quantity Assembled",
            filename=f"figures/y_{j+1}_boxplot.png")

        plot_boxplot_over_time(
            D_exp[j, :, :],
            title=f"Effective Demand D[{j+1},t] Distribution Over Time",
            ylabel="Demand Quantity",
            filename=f"figures/D_{j+1}_boxplot.png")