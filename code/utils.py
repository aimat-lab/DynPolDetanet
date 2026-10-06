from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from torch_geometric.data import Data
import json
import csv
import numpy as np
import pandas as pd
import csv
from functools import partial
import math

# Function to process SMILES into atomic numbers and 3D positionn
def smiles_to_graph(smiles):
    """
    Convert a SMILES string into atomic numbers (z) and 3D coordinates (pos) using RDKit.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  # Invalid SMILES string

    mol = Chem.AddHs(mol)  # Add hydrogens
    result = AllChem.EmbedMolecule(mol)
    if result != 0:
        # Embedding failed; try with random coordinates
        result = AllChem.EmbedMolecule(mol, useRandomCoords=True)
        if result != 0:
            print(f"Computation for {smiles} failed")
            return None  # Embedding failed again

    AllChem.UFFOptimizeMolecule(mol)  # Optimize geometry
    # Extract atomic numbers
    z = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long)
    
    # Extract 3D positions
    conf = mol.GetConformer()
    pos = torch.tensor([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], dtype=torch.float)
    
    return z, pos


def lorentzian_func(x, center, amplitude, gamma=0.05):
    """
    Basic Lorentzian function:
      amplitude / [1 + ((x - center) / gamma)^2]
    """
    return amplitude / (1.0 + ((x - center) / gamma)**2)

def gaussian_func(x, center, amplitude, sigma=0.5):
    """
    Basic Gaussian function:
      amplitude * exp( -((x - center)^2 / (2 * sigma^2)) )
    """
    exponent = -((x - center)**2) / (2.0 * sigma**2)
    return amplitude * np.exp(exponent)

def load_spectra(positions, osc_strength, fun_type="l", width_param=0.05):

    # Convert them to floats (the intensities)
    numeric_pos = [float(v) for v in positions]
    numeric_osc_strength = [float(v) for v in osc_strength]

    # Create line-shape callables for each frequency/intensity pair
    # partial(...) => returns a function f(x)
    if fun_type == "l":
        # Lorentzian
        line_shapes = [
            partial(lorentzian_func, center=f, amplitude=a, gamma=width_param)
            for f, a in zip(numeric_pos, numeric_osc_strength)
        ]
    else:
        # Gaussian
        line_shapes = [
            partial(gaussian_func, center=f, amplitude=a, sigma=width_param)
            for f, a in zip(numeric_pos, numeric_osc_strength)
        ]

    return line_shapes

def inverse_std(x_std, scaler, device=None, dtype=None):
    μ = torch.tensor(scaler.mean_,  device=device, dtype=dtype or x_std.dtype)
    σ = torch.tensor(scaler.scale_, device=device, dtype=dtype or x_std.dtype)
    return x_std * σ + μ



def undo_spectrum_scaling_full(flat: torch.Tensor,
                               scaler) -> torch.Tensor:
    """
    Invert a StandardScaler over the full [124,3,3] spectrum.

    Args:
      flat: Tensor of shape [N, 3, 3] (N = B*124), flattened & scaled.
      scaler: sklearn StandardScaler fitted on shape [num_samples, 124*9].

    Returns:
      phys: Tensor of shape [N, 3, 3] in original physical units.
    """
    N = flat.shape[0]
    if N % 124 != 0:
        raise ValueError(f"First dim must be a multiple of 124, got {N}")
    B = N // 124
    
    # 1) Recover the [B, 124, 3, 3] structure
    ts = flat.view(B, 124, 3, 3)

    # 2) Flatten the last two dims → [B, 124*9]
    ts9 = ts.view(B, 124 * 9)

    # 3) Inverse-standardise:  r = z*σ + μ
    mu    = torch.as_tensor(scaler.mean_,  device=ts9.device, dtype=ts9.dtype)  # [124*9]
    sigma = torch.as_tensor(scaler.scale_, device=ts9.device, dtype=ts9.dtype)  # [124*9]
    phys9 = ts9 * sigma + mu                                                       # [B, 124*9]

    # 4) Un-flatten back to [B, 124, 3, 3]
    phys = phys9.view(B, 124, 3, 3)

    # 5) Return to the original flat shape [N, 3, 3]
    return phys.view(N, 3, 3)
