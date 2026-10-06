"""Evaluate the pretrained dynamic polarizability models on the held-out test split.

Usage (from code/):  python evaluate.py
"""
import os
import random

import torch
from torch_geometric.loader import DataLoader

from detanet_model import model_loader
from detanet_model.metrics import l2loss, R2, loss_emd_per_spectrum

SEED = 3512
TRAIN_FRAC = 0.9

CODE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(os.path.dirname(CODE_DIR), "data")


def load_and_preprocess(dataset_file, x_features):
    """Load a .pt dataset and apply the same preprocessing used at training
    time: center the real part on the static (zero-frequency) polarizability,
    drop the zero-frequency imaginary part (always 0), concatenate real+imag
    into the target `y`, and build the per-atom UV-vis auxiliary feature `x`
    from the sampled spectrum.
    """
    dataset = torch.load(os.path.join(DATA_DIR, dataset_file), weights_only=False)
    for data in dataset:
        data.real_ee = data.real_ee[1:] - data.real_ee[0]
        data.imag_ee = data.imag_ee[1:]
        data.y = torch.cat([data.real_ee, data.imag_ee], dim=0)
        if x_features:
            data.x = data.spectra[1:].repeat(len(data.z), 1)
    return dataset


def held_out_test_split(dataset, seed=SEED, train_frac=TRAIN_FRAC):
    """Reconstruct the exact 90/10 train/test split used for training, by
    replaying the same seeded shuffle. This is what lets us isolate molecules
    the model never saw during training, using only the released dataset
    file and the known seed (no need for a separate saved split file).
    """
    rng_state = random.getstate()
    try:
        random.seed(seed)
        shuffled = list(dataset)
        random.shuffle(shuffled)
    finally:
        random.setstate(rng_state)
    split_index = int(train_frac * len(shuffled))
    return shuffled[:split_index], shuffled[split_index:]


def evaluate_on_test_set(model, test_dataset, num_pol_spectra, S, batch_size=16, device="cpu"):
    """Run the model on the held-out test set and compute MSE, EMD, and R^2,
    using the same metric implementations used during training/evaluation in
    the paper (detanet_model.metrics).
    """
    device = torch.device(device)
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    model.to(device)
    model.eval()

    all_out, all_target = [], []
    with torch.no_grad():
        for batch in loader:
            kwargs = dict(pos=batch.pos.to(device), z=batch.z.to(device), batch=batch.batch.to(device))
            if hasattr(batch, "x") and batch.x is not None:
                kwargs["x_features"] = batch.x.to(device)
            out = model(**kwargs)
            target = batch.y.to(device)
            all_out.append(out.reshape(target.shape))
            all_target.append(target)

    pred = torch.cat(all_out, dim=0)
    targ = torch.cat(all_target, dim=0)

    n_mol = len(test_dataset)
    pred_b = pred.reshape(n_mol, num_pol_spectra, 3, 3)
    targ_b = targ.reshape(n_mol, num_pol_spectra, 3, 3)

    return {
        "n_molecules": n_mol,
        "mse": l2loss(pred, targ).item(),
        "r2": R2(pred, targ).item(),
        "emd": loss_emd_per_spectrum(pred_b, targ_b, S=S).item(),
        "pred": pred_b,
        "target": targ_b,
    }


def evaluate_qm9spol(device="cpu"):
    """Evaluate the QM9SPol model (61 frequencies, UV-vis input) on its held-out test split (516 molecules)."""
    dataset = load_and_preprocess("QM9SPol.pt", x_features=True)
    _, test = held_out_test_split(dataset)
    model = model_loader.qm9s_dynamic_polarizability_model(device=torch.device(device))
    return evaluate_on_test_set(model, test, num_pol_spectra=122, S=61, batch_size=16, device=device)


def evaluate_hopv241(device="cpu"):
    """Evaluate the HOPV15 model (241 frequencies, UV-vis input) on its held-out test split (35 molecules)."""
    dataset = load_and_preprocess("HOPV_241pol.pt", x_features=True)
    _, test = held_out_test_split(dataset)
    model = model_loader.hopv241_dynamic_polarizability_model(device=torch.device(device))
    return evaluate_on_test_set(model, test, num_pol_spectra=482, S=241, batch_size=8, device=device)


def main():
    for name, result in [("QM9SPol", evaluate_qm9spol()), ("HOPV241", evaluate_hopv241())]:
        print(f"{name} ({result['n_molecules']} held-out molecules): "
              f"MSE {result['mse']:.2f}, EMD {result['emd']:.2f}, R2 {result['r2']:.4f}")


if __name__ == "__main__":
    main()
