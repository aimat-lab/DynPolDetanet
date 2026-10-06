"""Plots training/validation MSE and EMD curves over epochs for both
released models (QM9SPol and HOPV241), from CSV logs exported from wandb
(see the "epoch" / "val_emd_loss" / "loss_per_epoch" fields logged by
Trainer in trainer.py). Produces the training-history panels shown in the
paper's Figures 4d and 5d.
"""
import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. Define file paths
data_dir = "../results/Plotted_Results/"

# QM9 Files
mse_qm9_file = os.path.join(data_dir, "mse_OPT_KITQM9_Shifted61spectra90epochs_32batchsize_0.0005lr_6cutoff_6numblock_256features_32att_KITQM9_seed3512.csv")
emd_qm9_file = os.path.join(data_dir, "emd_OPT_KITQM9_Shifted61spectra90epochs_32batchsize_0.0005lr_6cutoff_6numblock_256features_32att_KITQM9_seed3512.csv")
prefix_qm9 = "OPT_KITQM9_Shifted61spectra90epochs_32batchsize_0.0005lr_6cutoff_6numblock_256features_32att_KITQM9_seed3512"

# HOPV Files
mse_hopv_file = os.path.join(data_dir, "HOPV241_mse.csv")
emd_hopv_file = os.path.join(data_dir, "HOPV241_emd.csv")
prefix_hopv = "HOPV241_Shifted0.2Dropout_241spectra100epochs_16batchsize_0.0005lr_4cutoff_4numblock_512features_64att_HOPV_241pol_seed3512"

# 2. Load the data
df_mse_qm9 = pd.read_csv(mse_qm9_file).dropna(subset=['epoch'])
df_emd_qm9 = pd.read_csv(emd_qm9_file).dropna(subset=['epoch'])

df_mse_hopv = pd.read_csv(mse_hopv_file).dropna(subset=['epoch'])
df_emd_hopv = pd.read_csv(emd_hopv_file).dropna(subset=['epoch'])

# 3. Set up the publication style globally
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
    "font.size": 14,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--"
})

# Colors
c_train_emd = "#D4B872"  
c_val_emd   = "#A6611A"  
c_train_mse = "#80CDC1"  
c_val_mse   = "#018571"  

os.makedirs(data_dir, exist_ok=True)

# ==========================================
# FIGURE 1: QM9 Dataset
# ==========================================
fig1, ax1 = plt.subplots(figsize=(7, 5), dpi=300)

ax1.set_xlabel("Epoch")
ax1.set_ylabel("Earth-Mover Distance", color=c_val_emd)
ax1.tick_params(axis='y', labelcolor=c_val_emd)

l1 = ax1.plot(df_emd_qm9['epoch'], df_emd_qm9[f'{prefix_qm9} - emd_loss'], 
              color=c_train_emd, linewidth=2, marker='o', markersize=4, label="Train EMD")
l2 = ax1.plot(df_emd_qm9['epoch'], df_emd_qm9[f'{prefix_qm9} - val_emd_loss'], 
              color=c_val_emd, linewidth=2, marker='o', markersize=4, label="Val EMD")
ax1.set_ylim(bottom=0)
ax1.set_xlim([0, 80])

ax2 = ax1.twinx()  
ax2.set_ylabel("Mean Squared Error", color=c_val_mse)
ax2.tick_params(axis='y', labelcolor=c_val_mse)
ax2.grid(False) 

l3 = ax2.plot(df_mse_qm9['epoch'], df_mse_qm9[f'{prefix_qm9} - loss_per_epoch'], 
              color=c_train_mse, linewidth=2, label="Train MSE")
l4 = ax2.plot(df_mse_qm9['epoch'], df_mse_qm9[f'{prefix_qm9} - epoch_val_loss'], 
              color=c_val_mse, linewidth=2, label="Val MSE")
ax2.set_ylim(bottom=0)

lines = l1 + l2 + l3 + l4
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper right", frameon=True, facecolor="white", framealpha=0.9, edgecolor="0.8")

plt.tight_layout()
plt.savefig(os.path.join(data_dir, "loss_QM9.pdf"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(data_dir, "loss_QM9.png"), dpi=300, bbox_inches='tight')
plt.close(fig1) # Close to avoid overlap


# ==========================================
# FIGURE 2: HOPV Dataset
# ==========================================
fig2, ax3 = plt.subplots(figsize=(7, 5), dpi=300)

ax3.set_xlabel("Epoch")
ax3.set_ylabel("Earth-Mover Distance", color=c_val_emd)
ax3.tick_params(axis='y', labelcolor=c_val_emd)

l5 = ax3.plot(df_emd_hopv['epoch'], df_emd_hopv[f'{prefix_hopv} - emd_loss'], 
              color=c_train_emd, linewidth=2, marker='o', markersize=4, label="Train EMD")
l6 = ax3.plot(df_emd_hopv['epoch'], df_emd_hopv[f'{prefix_hopv} - val_emd_loss'], 
              color=c_val_emd, linewidth=2, marker='o', markersize=4, label="Val EMD")
ax3.set_ylim(bottom=0)
ax3.set_xlim([0, 80])

ax4 = ax3.twinx()  
ax4.set_ylabel("Mean Squared Error", color=c_val_mse)
ax4.tick_params(axis='y', labelcolor=c_val_mse)
ax4.grid(False) 

l7 = ax4.plot(df_mse_hopv['epoch'], df_mse_hopv[f'{prefix_hopv} - loss_per_epoch'], 
              color=c_train_mse, linewidth=2, label="Train MSE")
l8 = ax4.plot(df_mse_hopv['epoch'], df_mse_hopv[f'{prefix_hopv} - epoch_val_loss'], 
              color=c_val_mse, linewidth=2, label="Val MSE")
ax4.set_ylim(bottom=0)

lines2 = l5 + l6 + l7 + l8
labels2 = [l.get_label() for l in lines2]
ax3.legend(lines2, labels2, loc="upper right", frameon=True, facecolor="white", framealpha=0.9, edgecolor="0.8")

plt.tight_layout()
plt.savefig(os.path.join(data_dir, "loss_HOPV241.pdf"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(data_dir, "loss_HOPV241.png"), dpi=300, bbox_inches='tight')
plt.show() # Display the final generated figure in your IDE