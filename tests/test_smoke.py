"""Quick smoke test: both released checkpoints load, run on a few held-out
molecules on CPU, and return finite predictions with the expected shapes.

Run from the code/ directory:  python ../tests/test_smoke.py
"""
import sys
import os
import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
from reproduce_findings import load_and_preprocess, held_out_test_split  # noqa: E402
from detanet_model import model_loader  # noqa: E402

N_MOLECULES = 4  # kept small so the test runs in about a minute on CPU


def check(name, dataset_file, model_fn, num_pol_spectra, S, expected_test_size):
    device = torch.device("cpu")  # real and imaginary parts stacked: num_pol_spectra = 2 x frequencies
    dataset = load_and_preprocess(dataset_file, x_features=True)
    _, test = held_out_test_split(dataset)
    assert len(test) == expected_test_size, f"{name}: test split has {len(test)} molecules, expected {expected_test_size}"
    model = model_fn(device=device).eval()
    subset = [test[i] for i in range(N_MOLECULES)]
    batch = next(iter(DataLoader(subset, batch_size=N_MOLECULES, shuffle=False)))
    with torch.no_grad():
        out = model(pos=batch.pos, z=batch.z, batch=batch.batch, x_features=batch.x)
    out = out.reshape(N_MOLECULES, num_pol_spectra, 3, 3)
    target = batch.y.reshape(N_MOLECULES, num_pol_spectra, 3, 3)
    assert out.shape == target.shape, f"{name}: prediction {tuple(out.shape)} vs target {tuple(target.shape)}"
    assert torch.isfinite(out).all(), f"{name}: non-finite predictions"
    rel = ((out - target).abs().mean() / target.abs().mean()).item()
    print(f"[ok] {name}: test split {len(test)} molecules, output {tuple(out.shape)}, "
          f"mean relative error on {N_MOLECULES} molecules = {rel:.3f}")


if __name__ == "__main__":
    check("QM9SPol 61-point model", "QM9SPol.pt", model_loader.qm9s_dynamic_polarizability_model, 122, 61, 516)
    check("HOPV15 241-point model", "HOPV_241pol.pt", model_loader.hopv241_dynamic_polarizability_model, 482, 241, 35)
    print("all smoke checks passed")
