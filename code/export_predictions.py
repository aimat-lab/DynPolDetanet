"""Exports full per-molecule, per-frequency predictions (all 6 unique
tensor components, real and imaginary) from the pretrained HOPV241 model,
as CSVs plus comparison plots, for every molecule in the held-out val
split -- for detailed manual inspection, not for computing summary
metrics (see evaluate.py for that).
"""
import os
import torch
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from detanet_model import model_loader

# 1. SETUP DIRECTORIES
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
data_dir = os.path.join(parent_dir, 'data')
out_dir = "../results/HOPV15_predictions_full"

# 2. LOAD DATASET
print("Loading dataset...")
dataset = torch.load(os.path.join(data_dir, 'HOPV_241pol.pt'), weights_only=False)
print(f"Number of graphs in the dataset: {len(dataset)}")

# 3. PRE-PROCESS DATASET
for data in dataset:
    # Ensure attributes are linked
    data.real_ee = data.real_ee
    data.imag_ee = data.imag_ee
    # Flatten for internal model logic if needed
    data.y = torch.cat([data.real_ee, data.imag_ee], dim=0)
    # Remove static frequency from spectra for feature repeat
    data.spectra = data.spectra[1:]
    data.x = data.spectra.repeat(len(data.z), 1)

# 4. SHUFFLE AND SPLIT
random.seed(3512)
random.shuffle(dataset)
train_frac = 0.9
split_index = int(train_frac * len(dataset))

train_datasets = dataset[:split_index]
val_datasets   = dataset[split_index:]

# 5. INITIALIZE MODEL
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model_loader.hopv241_dynamic_polarizability_model(device=device)
model.to(device)
model.eval()

# 6. EXPORT AND PLOT FUNCTION
def export_and_plot_all(val_datasets, model, base_out_dir):
    """
    Creates a subfolder for each molecule. 
    Exports all 6 symmetric tensor elements (xx, yy, zz, xy, xz, yz) to CSVs.
    Generates 2x3 comparison plots for Real and Imaginary parts.
    """
    os.makedirs(base_out_dir, exist_ok=True)
    
    # Standard components for a symmetric 3x3 tensor
    tensor_components = [
        ("xx", 0, 0), ("yy", 1, 1), ("zz", 2, 2),
        ("xy", 0, 1), ("xz", 0, 2), ("yz", 1, 2)
    ]
    
    # Plotting styles
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.family": "serif", "font.size": 12})
    
    gt_kwargs_real = dict(marker='o', color='royalblue', s=15, alpha=0.6, label="GT")
    gt_kwargs_imag = dict(marker='o', color='orange',   s=15, alpha=0.6, label="GT")
    pred_kwargs    = dict(marker='^', color='black',    s=8,  alpha=0.7, label="Prediction")

    print(f"Processing {len(val_datasets)} molecules...")

    with torch.no_grad():
        for i, data_entry in enumerate(tqdm(val_datasets, desc="Exporting Results")):
            
            # Identify molecule index
            mol_idx = getattr(data_entry, 'idx', i)
            if torch.is_tensor(mol_idx): mol_idx = mol_idx.item()
            
            # Create molecule-specific subfolder
            mol_dir = os.path.join(base_out_dir, f"molecule_{mol_idx}")
            os.makedirs(mol_dir, exist_ok=True)

            # --- Extract Ground Truth ---
            real_ee = data_entry.real_ee.cpu().numpy() if torch.is_tensor(data_entry.real_ee) else np.asarray(data_entry.real_ee)
            imag_ee = data_entry.imag_ee.cpu().numpy() if torch.is_tensor(data_entry.imag_ee) else np.asarray(data_entry.imag_ee)
            
            static_pol = real_ee[0]  # alpha(0)
            real_gt = real_ee[1:]
            imag_gt = imag_ee[1:]
            freqs = data_entry.freqs[1:].cpu().numpy() if torch.is_tensor(data_entry.freqs) else np.asarray(data_entry.freqs[1:])

            # --- Model Prediction ---
            pol_spec = model(
                z=data_entry.z.to(device),
                pos=data_entry.pos.to(device),
                x_features=data_entry.x.to(device)
            ).cpu().numpy()

            half = pol_spec.shape[0] // 2
            pred_real = pol_spec[:half]
            pred_imag = pol_spec[half:]
            # Add static polarizability to real part as per code requirements
            pred_real += static_pol 

            # --- Export CSVs ---
            cols = [f"{lbl}_gt" for lbl,_,_ in tensor_components] + [f"{lbl}_pred" for lbl,_,_ in tensor_components]
            
            def save_csv(gt, pred, name):
                vals = np.stack([gt[:, r, c] for _, r, c in tensor_components] + 
                                [pred[:, r, c] for _, r, c in tensor_components], axis=1)
                df = pd.DataFrame(vals, columns=cols, index=freqs)
                df.to_csv(os.path.join(mol_dir, f"polarizability_{name}.csv"), index_label="frequency_eV")

            save_csv(real_gt, pred_real, "real")
            save_csv(imag_gt, pred_imag, "imag")

            # --- Generate Plots ---
            for part_name, gt_data, pred_data, kws in [("real", real_gt, pred_real, gt_kwargs_real), 
                                                       ("imag", imag_gt, pred_imag, gt_kwargs_imag)]:
                
                fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
                axes = axes.flatten()
                
                for idx, (lbl, r, c) in enumerate(tensor_components):
                    ax = axes[idx]
                    ax.scatter(freqs, gt_data[:, r, c], **kws)
                    ax.scatter(freqs, pred_data[:, r, c], **pred_kwargs)
                    ax.set_title(f"Component: {lbl}")
                    if idx >= 3: ax.set_xlabel("Frequency (eV)")
                    if idx % 3 == 0: ax.set_ylabel(f"alpha-{part_name.capitalize()} (a.u.)")
                
                handles, labels = axes[0].get_legend_handles_labels()
                fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98), ncol=2)
                fig.suptitle(f"Molecule {mol_idx} - {part_name.capitalize()} Polarizability", fontsize=16)
                
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                fig.savefig(os.path.join(mol_dir, f"plot_{part_name}.png"), dpi=150)
                plt.close(fig)

    print(f"\n✅ All CSVs and Plots saved to: {os.path.abspath(base_out_dir)}")

# 7. EXECUTE
export_and_plot_all(val_datasets, model, out_dir)