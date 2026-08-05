import numpy as np
import matplotlib.pyplot as plt

def plot_heatmap(matrix, xlabel, ylabel, title, filename, cmap="viridis"):
    
    fig, ax = plt.subplots(figsize=(8, 5))

    im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    fig.colorbar(im, ax=ax)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Eje Y: etiquetas enteras comenzando en 1
    n_rows = matrix.shape[0]
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(np.arange(1, n_rows + 1))

    n_cols = matrix.shape[1]
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(np.arange(1, n_cols + 1))

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_instance_decisions(x_val, price_eff, I_exp, y_exp, D_exp, Model, seed):
    
    plot_heatmap(
        x_val,
        xlabel="Periods",
        ylabel="Components",
        title=f"Average Procurement x[i,t,s] ",
        filename=f"figures/heatmaps/{Model}_{seed}_x.png"
    )
    
    plot_heatmap(
        price_eff,
        xlabel="Periodos",
        ylabel="Products",
        title=f"Average Effective Price w[j,t,p,s]",
        filename=f"figures/heatmaps/{Model}_{seed}_w.png"
    )

    plot_heatmap(
        I_exp,
        xlabel="Periodos",
        ylabel="Components",
        title=f"Average Inventory I[i,t,s]",
        filename=f"figures/heatmaps/{Model}_{seed}_I.png"
    )

    plot_heatmap(
        y_exp,
        xlabel="Periodos",
        ylabel="Products",
        title=f"Average Assembly y[j,t,s]",
        filename=f"figures/heatmaps/{Model}_{seed}_y.png"
    )

    plot_heatmap(
        D_exp,
        xlabel="Periodos",
        ylabel="Productos",
        title=f"Average Effective Demand D[j,t,p,s]",
        filename=f"figures/heatmaps/{Model}_{seed}_D.png"
    )