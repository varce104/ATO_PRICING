import numpy as np
import matplotlib.pyplot as plt

def plot_heatmap(matrix, xlabel, ylabel, title, filename, cmap="viridis"):
    
    plt.figure(figsize=(8, 5))
    plt.imshow(matrix, aspect="auto", cmap=cmap)
    plt.colorbar()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_instance_decisions(x_val, price_eff, I_exp, y_exp, D_exp):
    
    plot_heatmap(
        x_val,
        xlabel="Periods",
        ylabel="Components",
        title=f"Average Procurement x[i,t] ",
        filename=f"figures/HM_x.png"
    )
    
    plot_heatmap(
        price_eff,
        xlabel="Periodos",
        ylabel="Products",
        title=f"Average Effective Price w[j,t]",
        filename=f"figures/HM_w.png"
    )

    plot_heatmap(
        I_exp,
        xlabel="Periodos",
        ylabel="Components",
        title=f"Average Inventory I[i,t]",
        filename=f"figures/HM_I_avg_DL.png"
    )

    plot_heatmap(
        y_exp,
        xlabel="Periodos",
        ylabel="Products",
        title=f"Average Assembly y[j,t]",
        filename=f"figures/HM_y_avg_DL.png"
    )

    plot_heatmap(
        D_exp,
        xlabel="Periodos",
        ylabel="Productos",
        title=f"Average Effective Demand D[j,t]",
        filename=f"figures/HM_D_avg_DL.png"
    )