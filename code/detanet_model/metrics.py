"""Loss functions and evaluation metrics.

`loss_emd_per_spectrum` is the one actually used for training/evaluating
the dynamic polarizability models (see trainer.py and
evaluate.py) and is the metric reported as "EMD" in the paper's
Table 1. The others are general-purpose regression metrics / losses used
for other DetaNet property heads (energies, forces, static tensors, etc.),
not specific to the polarizability task.
"""
import torch
import numpy as np
import ot                                # POT library (Python Optimal Transport)

def l2loss(out,target):
    '''Mean Square Error(MSE)'''
    diff = out-target
    return torch.mean(diff ** 2)

def l1loss(out,target):
    '''Mean Absolute Error(MAE)'''
    return torch.mean(torch.abs(out-target))

def rmse(out, target):
    '''Root Mean Square Eroor(rmse) (also known as RMSD)'''
    return torch.sqrt(torch.mean((out - target) ** 2))

def state_l2loss(out,target):
    '''    #It is used to get the loss of the excited state vector,
    but we have tested that it does not perform well in QM9spectra's transition dipole prediction.


    from J. Phys. Chem. Lett. 2020, 11, 3828−3834
    <Combining SchNet and SHARC: The SchNarc Machine Learning Approach for Excited-State Dynamics>'''
    diffa=torch.abs(out-target)**2
    diffb=torch.abs(out+target)**2
    diff=torch.min(diffa,diffb)
    return torch.mean(diff)

def R2(out,target):
    '''coefficient of determination,Square of Pearson's coefficient, used to assess regression accuracy'''
    mean=torch.mean(target)
    SSE=torch.sum((out-target)**2)
    SST=torch.sum((mean-target)**2)
    return 1-(SSE/SST)

def combine_lose(out_tuple,target_tuple,lamb=10):
    '''Combine loss of energy and force,Training for ab initio dynamics simulation'''
    return l2loss(out_tuple[0],target_tuple[0])+lamb*l2loss(out_tuple[1],target_tuple[1])



def rf_rmse(pred, target, eps=1e-12):
    '''Relative Frobenius RMSE: per-sample Frobenius-norm error, normalized
    by the target's own Frobenius norm, then averaged (as an RMS) over the
    tensor/frequency dimensions -- a scale-invariant alternative to plain
    RMSE for comparing tensors whose magnitude varies a lot across
    molecules.

    Args:
        pred, target: complex tensors of shape [batch, num_freq, 3, 3].
        eps: numerical safety term to avoid division by zero.

    Returns:
        Real tensor of shape [batch].
    '''
    diff2 = (pred - target).abs()**2        # |Δ|²
    num   = diff2.sum(dim=(-1, -2)).sum(dim=-1)          # Σ_f ||Δ||_F²
    den   = (target.abs()**2).sum(dim=(-1, -2)).sum(dim=-1)
    return torch.sqrt(num / (den + eps))    # [batch]

def cosine_similarity(pred: torch.Tensor,
                        target: torch.Tensor,
                        eps: float = 1e-12) -> torch.Tensor:
    """
    Complex cosine similarity  (one value per sample).

    cos_sim = |<x , y>| / (||x|| * ||y||)      in   [0 , 1]
              = 1 when x∥y   , 0 when orthogonal

    Parameters
    ----------
    pred, target : complex tensors
        Shape  [B, ...]  with the same remaining dimensions.
    eps : float
        Numerical safety term to avoid division by 0.

    Returns
    -------
    sims : real tensor  [B]
        cos-similarity for every item in the batch.
    """

    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {target.shape}")

    # flatten everything except the batch dimension  →  [B, N]
    x = pred.reshape(pred.shape[0], -1)
    y = target.reshape(target.shape[0], -1)
    inner = (x.conj() * y).sum(dim=-1) 
    nx = torch.linalg.norm(x, dim=-1)
    ny = torch.linalg.norm(y, dim=-1)
    cos_sim = inner.abs() / (nx * ny + eps)
    cos_loss = 1.0 - cos_sim      # still in [0,1]
    mean_loss = cos_loss.mean()   # scalar
    return mean_loss          # 1 → identical shape, 0 → orthogonal



def loss_emd_per_spectrum(pred: torch.Tensor,
                          target: torch.Tensor,
                          *,
                          S = 61,
                          num_iter: int = 2_000_000) -> torch.Tensor:
    """
    Exact 1-D Wasserstein-1 (Earth-Mover) distance/loss between predicted
    and target dynamic polarizability spectra -- this is the "EMD" metric
    reported in the paper's Table 1 / Eq. for L_EMD, and the loss actually
    used to track EMD during training (see trainer.py).

    Conceptually treats each (molecule, real/imag part, tensor component)
    combination as its own 1D "pile of earth" distributed across S
    frequency points, and computes the minimal-cost transport plan to
    reshape the predicted distribution into the target one (via the exact
    network-simplex solver in the POT library), independently for:
    • every sample in the batch,
    • both the real and imaginary parts,
    • all 9 tensor components (3x3, flattened),
    then averages the result over batch x parts x components.

    Args:
        pred, target: tensors of shape [B, 2*S, 3, 3] (S frequency points,
            real part stacked before imaginary part along dim 1), must
            have identical shape.
        S: number of frequency points per spectrum (61 or 241 in the
            paper's QM9SPol/HOPV15 configurations, respectively).
        num_iter: max iterations for the underlying OT solver
            (`ot.emd2`).

    Returns:
        Scalar tensor on pred's device/dtype.
    """

    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape}, target {target.shape}")

    B, S2, H, W = pred.shape           # S2 = 2*S (real+imag stacked)
    C = H * W                           # 9 tensor components
    parts = 2                           # Re / Im

    # reshape → [B, parts, S, C]  where C = 9
    pred_r = pred.reshape(B, parts, S, C)
    targ_r = target.reshape(B, parts, S, C)

    # uniform weights for a single spectrum (S bins)
    a_np = np.full(S, 1.0 / S, dtype=np.float64)

    loss_sum = 0.0

    for b in range(B):
        for p in range(parts):          # 0 = Real, 1 = Imag
            # pull once to CPU / numpy for the whole spectrum to save transfers
            x_np = pred_r[b, p].detach().cpu().numpy()   # shape [61, 9]
            y_np = targ_r[b, p].detach().cpu().numpy()

            for c in range(C):          # loop over the 9 tensor entries
                # build 61×61 ground distance on *values*
                M_np = ot.dist(
                    x_np[:, c].reshape(-1, 1),
                    y_np[:, c].reshape(-1, 1),
                    metric="euclidean",
                )

                loss_sum += ot.emd2(a_np, a_np, M_np, numItermax=num_iter)

    # normalise: mean over (B × parts × C)
    normaliser = B * parts * C
    return pred.new_tensor(loss_sum / normaliser)
