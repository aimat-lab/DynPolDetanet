"""Plots predicted vs. original charge-carrier generation rates for the
paper's device-level OPV validation (Section "Integration of the model
into optical device simulations", Figure 6).

This is a plotting script only -- it reads a pre-computed
results/gen_rates/gen_rates.csv (columns: original/predicted generation
rate, relative error), computed elsewhere by the T-matrix/treams/
scattering-matrix optical simulation pipeline (not part of this repo; see
the paper's Methods section for that pipeline's description). It does not
itself run any part of that pipeline.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import os

# 1. Setup file paths
file_path = "../results/gen_rates/gen_rates.csv"
out_dir = "../results/Plotted_Results"
os.makedirs(out_dir, exist_ok=True)

if not os.path.exists(file_path):
    print(f"Error: {file_path} not found.")
else:
    df = pd.read_csv(file_path)

    true_vals = df['gen_rates_original [mA/(cm^2)]']
    pred_vals = df['gen_rates_predicted [mA/(cm^2)]']
    # Relative error |pred - orig| / orig in %, recomputed from the generation rates
    errors = np.abs(pred_vals - true_vals) / true_vals * 100

    # 2. Metrics
    r2 = r2_score(true_vals, pred_vals)
    rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))

    # 3. Styling
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Liberation Sans",
        "mathtext.it": "Liberation Sans:italic",
        "font.size": 14,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })

    # 4. Generate Plot
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained", dpi=300)

    # Use 'inferno' or 'magma' for the heatmap look
    scatter = ax.scatter(
        true_vals, pred_vals, 
        c=errors, 
        norm=colors.LogNorm(vmin=errors.min(), vmax=errors.max()),
        cmap='inferno', 
        s=80, alpha=0.9, edgecolor='k', linewidth=0.5, zorder=3
    )

    # Reference y=x line
    ax.plot([10, 17], [10, 17], "--", color="black", alpha=0.7, linewidth=1.5, zorder=2)

    ax.set_xlim(10, 17)
    ax.set_ylim(10, 17)

    # 5. Manual Colorbar Ticks
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Relative error [%]", fontsize=14)
    
    # ticks: data minimum -> 0.03 -> 0.1 -> 0.3 -> 1 -> 3 -> data maximum
    custom_ticks = [round(errors.min(), 3), 0.03, 0.1, 0.3, 1, 3, round(errors.max(), 3)]
    
    cbar.set_ticks(custom_ticks)
    cbar.set_ticklabels([str(t) for t in custom_ticks])
    cbar.ax.tick_params(labelsize=12)

    # Labels and Title
    ax.set_xlabel(r"Original generation rate [mA/cm$^2$]")
    ax.set_ylabel(r"Predicted generation rate [mA/cm$^2$]")

    ax.grid(True, alpha=0.3, linestyle="--", zorder=1)

    # 6. Export
    plt.savefig(os.path.join(out_dir, "gen_rates.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, "gen_rates.png"), bbox_inches='tight')

    plt.show()