# DynPolDetanet

Machine-learning model that predicts the frequency-dependent, complex-valued dynamic
polarizability tensor of organic molecules from the 3D geometry and a sampled UV–vis
spectrum. It extends DetaNet to a fast surrogate for time-dependent density functional
theory (TD-DFT). This repository accompanies the manuscript
"Accelerating dynamic polarizability calculations of organic molecules using equivariant
graph neural networks".

Code for **"Accelerating dynamic polarizability calculations of organic molecules
using equivariant graph neural networks"** (Metni, Kraus, Schubert, Krstić, Rockstuhl,
Friederich).

The model extends [DetaNet](https://doi.org/10.1038/s43588-023-00550-y) to predict the
frequency-dependent, complex-valued dynamic polarizability tensor of organic molecules
from the 3D geometry and a sampled UV–vis spectrum.

## Contents

```
code/
  detanet_model/                 model architecture, metrics and checkpoint loaders
  trainer.py                     training loop
  spectra_pol.py                 training script (QM9SPol, 61-point model)
  reproduce_findings.py          evaluation of the released checkpoints on the held-out split
  export_predictions.py          export predictions for individual molecules
  gen_rates.py                   Figure 6 (generation rates, 35 HOPV molecules)
  plot_losses.py                 training histories (EMD and MSE)
  plot_manuscript_fig4_fig5.py   Figures 4 and 5 from the saved data
  trained_param/dynamic-polarizability/
      QM9S_dynamic_polarizability.pth      (not in git)
      HOPV241_dynamic_polarizability.pth   (not in git)
data/
  QM9SPol.pt, HOPV.pt, HOPV_241pol.pt    (not in git: see Data and model weights)
  HOPV15.csv
results/                          saved data behind the figures
tests/test_smoke.py               quick checks (about one minute on CPU)
```

## Data and model weights

The datasets and the model weights are not stored in this repository because of their size.
Download them from the Zenodo record (link to be added) and place them as follows:

| File | Put it in |
|---|---|
| `QM9SPol.pt` (about 88 MB) | `data/` |
| `HOPV.pt` (about 6 MB) | `data/` |
| `HOPV_241pol.pt` (about 20 MB) | `data/` |
| `QM9S_dynamic_polarizability.pth` (about 45 MB) | `code/trained_param/dynamic-polarizability/` |
| `HOPV241_dynamic_polarizability.pth` (about 120 MB) | `code/trained_param/dynamic-polarizability/` |

The scripts read these paths relative to the repository, so no other configuration is needed.

## Installation

Python 3.10 is used for all checks. The checks in `tests/` ran on CPU.

```bash
conda create -n detanet_env python=3.10 -y && conda activate detanet_env
pip install -r requirements.txt
pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.4.1+cu124.html
```

Adjust the `-f` index to your PyTorch build; `requirements.txt` pins the versions used here.

## Quick checks

Run from `code/`:

```bash
python ../tests/test_smoke.py                          # both checkpoints load and predict
DETANET_SMOKE=1 python spectra_pol.py                  # one training epoch on 24 molecules, no W&B
python plot_manuscript_fig4_fig5.py                    # Figures 4 and 5 from the saved data
python gen_rates.py                                    # Figure 6
python plot_losses.py                                  # training histories
```

Smoke-training writes its checkpoint to a temporary folder, so the released weights are not overwritten.

## Evaluation on the held-out split

```bash
python reproduce_findings.py     # about 30 s on CPU, evaluates both checkpoints
```

The held-out test sets are 516 molecules (QM9SPol) and 35 molecules (HOPV15), created with
seed 3512 and a 90/10 split. The metrics of the paper are given in Table 1 of the manuscript.

## Training

`code/spectra_pol.py` trains the QM9SPol model. Hyperparameters are module-level
variables at the top of the file. Logging uses Weights & Biases (`wandb`).
The released checkpoints correspond to the configurations in
`code/detanet_model/model_loader.py`.

## License

See `LICENSE`.

## Citation

If you use this code, please cite the manuscript and the DetaNet architecture it builds on.

Manuscript (APA):
Metni, H., Kraus, M., Schubert, M. L., Krstić, M., Rockstuhl, C., & Friederich, P. (2026).
Accelerating dynamic polarizability calculations of organic molecules using equivariant graph
neural networks. *Manuscript in preparation*.

Software (APA):
Metni, H., Kraus, M., Schubert, M. L., Krstić, M., Rockstuhl, C., & Friederich, P. (2026).
*DynPolDetanet* [Computer software]. GitHub. https://github.com/aimat-lab/DynPolDetanet

DetaNet (APA):
Zou, Z., Zhang, Y., Liang, L., Wei, M., Leng, J., Jiang, J., Luo, Y., & Hu, W. (2023).
A deep learning model for predicting selected organic molecular spectra.
*Nature Computational Science*, 3, 957–964. https://doi.org/10.1038/s43588-023-00550-y
