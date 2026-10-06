"""Figures 4 (QM9SPol) and 5 (HOPV15) of the polarizability manuscript.

Reads the saved data (no model needed) and writes one figure per dataset:
(a) real and (b) imaginary parts of the xx, yy, zz components of one molecule,
(c) predicted versus true values over the validation set, and
(d) training history (EMD and MSE). Text is sans-serif, panels carry
upright (a)-(d) labels, and no panel has a title.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
OUT = os.path.join(RES, "Plotted_Results")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Liberation Sans",
    "mathtext.it": "Liberation Sans:italic",
    "font.size": 9,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "0.8",
    "grid.linewidth": 0.4,
})

C_GT_REAL, C_GT_IMAG, C_PRED = "royalblue", "orange", "black"
C_TRAIN_EMD, C_VAL_EMD = "#D4B872", "#A6611A"
C_TRAIN_MSE, C_VAL_MSE = "#80CDC1", "#018571"


def spectra_row(fig, gs_row, freq, gt, pred, part, scale, ylabel, color, ylim, yticks):
    """One row of three components (xx, yy, zz) sharing the y-axis."""
    sub = gs_row.subgridspec(1, 6, wspace=0.35)
    axes = [fig.add_subplot(sub[0, 2 * k:2 * k + 2]) for k in range(3)]
    gt = {k: v / scale for k, v in gt.items()}
    pred = {k: v / scale for k, v in pred.items()}
    for k, (ax, comp) in enumerate(zip(axes, ["xx", "yy", "zz"])):
        ax.scatter(freq, gt[comp], s=6, color=color, alpha=0.7, label="Ground Truth", linewidths=0)
        ax.scatter(freq, pred[comp], s=5, marker="^", color=C_PRED, alpha=0.6, label="Model Prediction", linewidths=0)
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        ax.set_xlim(1.4, 6.4)
        ax.set_xticks([2, 4, 6])
        ax.grid(True)
        ax.set_xlabel("Frequency (eV)")
        if k > 0:
            ax.tick_params(labelleft=False)
    axes[0].set_ylabel(ylabel)
    axes[2].legend(loc="upper right", frameon=True, facecolor="white", edgecolor="0.7", framealpha=0.9, markerscale=1.8)
    return axes


def spectra_frames(prefix_dir, mol):
    real = pd.read_csv(os.path.join(prefix_dir, f"polarizability_real_{mol}.csv"))
    imag = pd.read_csv(os.path.join(prefix_dir, f"polarizability_imag_{mol}.csv"))
    freq = real["frequency_eV"].values
    def cols(df, kind):
        return {c: df[f"{c}_{kind}"].values for c in ["xx", "yy", "zz"]}
    return freq, cols(real, "gt"), cols(real, "pred"), cols(imag, "gt"), cols(imag, "pred")


def loss_panel(ax, emd_csv, mse_csv, prefix, emd_lim, emd_ticks, mse_lim, mse_ticks):
    emd = pd.read_csv(emd_csv).dropna(subset=["epoch"])
    mse = pd.read_csv(mse_csv).dropna(subset=["epoch"])
    l1 = ax.plot(emd["epoch"], emd[f"{prefix} - emd_loss"], color=C_TRAIN_EMD, lw=1.2, marker="o", ms=2, label="Train EMD")
    l2 = ax.plot(emd["epoch"], emd[f"{prefix} - val_emd_loss"], color=C_VAL_EMD, lw=1.2, marker="o", ms=2, label="Val EMD")
    ax.set_xlim(0, 80)
    ax.set_ylim(*emd_lim)
    ax.set_yticks(emd_ticks)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Earth-mover distance (a.u.)", color=C_VAL_EMD)
    ax.yaxis.set_label_coords(-0.14, 0.5)
    ax.tick_params(axis="y", labelcolor=C_VAL_EMD)
    ax2 = ax.twinx()
    l3 = ax2.plot(mse["epoch"], mse[f"{prefix} - loss_per_epoch"], color=C_TRAIN_MSE, lw=1.2, label="Train MSE")
    l4 = ax2.plot(mse["epoch"], mse[f"{prefix} - epoch_val_loss"], color=C_VAL_MSE, lw=1.2, label="Val MSE")
    ax2.set_ylim(*mse_lim)
    ax2.set_yticks(mse_ticks)
    ax2.set_ylabel("Mean squared error (a.u.)", color=C_VAL_MSE)
    ax2.tick_params(axis="y", labelcolor=C_VAL_MSE)
    ax2.grid(False)
    ax2.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    lines = l1 + l2 + l3 + l4
    ax.legend(lines, [l.get_label() for l in lines], loc="upper right", frameon=True, facecolor="white", edgecolor="0.7", framealpha=0.9)


def scatter_panel(ax, csv, scale, lim, power):
    d = pd.read_csv(csv)
    ax.scatter(d["true"] / scale, d["pred"] / scale, s=1.2, color="slateblue", alpha=0.25,
               linewidths=0, label="Dynamic polarizability", rasterized=True)
    ax.plot(lim, lim, "--", color="coral", lw=1, label="$y=x$")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xticks(range(lim[0], lim[1] + 1))
    ax.set_yticks(range(lim[0], lim[1] + 1))
    ax.set_xlabel(f"True value ($\\times10^{{{power}}}$ a.u.)")
    ax.set_ylabel(f"Predicted value ($\\times10^{{{power}}}$ a.u.)")
    ax.grid(True)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="0.7", framealpha=0.9, markerscale=3)


def build(fig_name, mol_dir, mol, scatter_csv, scale, power, lim, emd_csv, mse_csv, prefix, ylabel_scale, cfg):
    freq, gt_r, pr_r, gt_i, pr_i = spectra_frames(mol_dir, mol)
    fig = plt.figure(figsize=(7.1, 8.4))
    gs = GridSpec(3, 6, figure=fig, height_ratios=[1, 1, 1.25], hspace=0.7, wspace=0.9,
                  left=0.11, right=0.89, top=0.93, bottom=0.06)
    row_a = gs[0:1, :]
    row_b = gs[1:2, :]
    rows = {"a": row_a, "b": row_b}
    yr = f"Real part ({ylabel_scale})" if ylabel_scale else "Real part"
    ax_a = spectra_row(fig, row_a, freq, {k: gt_r[k] for k in gt_r}, {k: pr_r[k] for k in pr_r}, "real", scale,
                       f"$\\alpha$-Real ({ylabel_scale})" if ylabel_scale else "$\\alpha$-Real (a.u.)", C_GT_REAL,
                       cfg["real_ylim"], cfg["real_yticks"])
    ax_b = spectra_row(fig, row_b, freq, {k: gt_i[k] for k in gt_i}, {k: pr_i[k] for k in pr_i}, "imag", scale,
                       f"$\\alpha$-Imag ({ylabel_scale})" if ylabel_scale else "$\\alpha$-Imag (a.u.)", C_GT_IMAG,
                       cfg["imag_ylim"], cfg["imag_yticks"])
    ax_c = fig.add_subplot(gs[2, 0:3])
    scatter_panel(ax_c, scatter_csv, scale=10 ** power, lim=lim, power=power)
    ax_d = fig.add_subplot(gs[2, 3:6])
    loss_panel(ax_d, emd_csv, mse_csv, prefix, cfg["emd_lim"], cfg["emd_ticks"], cfg["mse_lim"], cfg["mse_ticks"])
    # align the bottom row with the spectra rows: (c) starts where the first
    # spectra panel starts, (d) ends where the last spectra panel ends
    pa0 = ax_a[0].get_position()
    pa2 = ax_a[2].get_position()
    pc = ax_c.get_position()
    pd = ax_d.get_position()
    ax_c.set_position([pa0.x0, pc.y0, pc.x1 - pa0.x0, pc.height])
    ax_d.set_position([pd.x0, pd.y0, pa2.x1 - pd.x0, pd.height])
    # centre the EMD label in the gap between panel (c) and the tick labels of (d)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    c_right = ax_c.get_window_extent(rend).x1
    tick_left = min(t.get_window_extent(rend).x0 for t in ax_d.get_yticklabels() if t.get_text())
    lbl = ax_d.yaxis.label.get_window_extent(rend)
    target = c_right + (tick_left - c_right - lbl.width) / 2
    d_px = ax_d.get_window_extent(rend)
    cx = ax_d.yaxis.label.get_position()[0] + (target - lbl.x0) / d_px.width
    ax_d.yaxis.set_label_coords(cx, 0.5)
    # upright panel letters, top left of each panel block
    for letter, ax in [("a", ax_a[0]), ("b", ax_b[0]), ("c", ax_c), ("d", ax_d)]:
        bb = ax.get_position()
        fig.text(bb.x0 - 0.075, bb.y1 + 0.012, f"({letter})", fontsize=11, ha="left", va="bottom")
    out = os.path.join(OUT, fig_name)
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    pm = os.path.join(RES, "HOPV15_predictions")
    # Figure 4: QM9SPol, molecule 70289, values in a.u. (real/imag in the same units as the data)
    build("manuscript_Figure4_QM9SPol.pdf", RES, "QM9SPol_70289",
          os.path.join(RES, "QM9Spol_dynamic_polarizability_pred_vs_true_scatter.csv"),
          scale=1.0, power=3, lim=(-2, 4),
          emd_csv=os.path.join(OUT, "emd_OPT_KITQM9_Shifted61spectra90epochs_32batchsize_0.0005lr_6cutoff_6numblock_256features_32att_KITQM9_seed3512.csv"),
          mse_csv=os.path.join(OUT, "mse_OPT_KITQM9_Shifted61spectra90epochs_32batchsize_0.0005lr_6cutoff_6numblock_256features_32att_KITQM9_seed3512.csv"),
          prefix="OPT_KITQM9_Shifted61spectra90epochs_32batchsize_0.0005lr_6cutoff_6numblock_256features_32att_KITQM9_seed3512",
          ylabel_scale="a.u.",
          cfg=dict(real_ylim=(-300, 450), real_yticks=[-200, 0, 200, 400], imag_ylim=(-50, 500), imag_yticks=[0, 100, 200, 300, 400, 500], emd_lim=(0, 12), emd_ticks=[0, 2, 4, 6, 8, 10], mse_lim=(0, 3200), mse_ticks=[0, 500, 1000, 1500, 2000, 2500, 3000]))
    # Figure 5: HOPV15, molecule 136, values shown in units of 10^3 a.u.
    build("manuscript_Figure5_HOPV15.pdf", pm, "HOPV_136",
          os.path.join(RES, "HOPV241_dynamic_polarizability_pred_vs_true_scatter.csv"),
          scale=1e3, power=4, lim=(-2, 3),
          emd_csv=os.path.join(OUT, "HOPV241_emd.csv"),
          mse_csv=os.path.join(OUT, "HOPV241_mse.csv"),
          prefix="HOPV241_Shifted0.2Dropout_241spectra100epochs_16batchsize_0.0005lr_4cutoff_4numblock_512features_64att_HOPV_241pol_seed3512",
          ylabel_scale="$\\times10^3$ a.u.",
          cfg=dict(real_ylim=(-10, 10), real_yticks=[-10, -5, 0, 5, 10], imag_ylim=(-5, 15), imag_yticks=[-5, 0, 5, 10, 15], emd_lim=(0, 180), emd_ticks=[0, 50, 100, 150], mse_lim=(0, 400000), mse_ticks=[0, 100000, 200000, 300000, 400000]))
