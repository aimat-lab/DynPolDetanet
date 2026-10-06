"""Non-linear activation functions used throughout DetaNet.

All activations here operate only on invariant (scalar, l=0) features,
since applying an arbitrary point-wise non-linearity to an equivariant
(l>0) feature would break its rotational equivariance. Equivariant features
are instead only ever scaled by a learned invariant factor, which is
computed by applying these activations to some invariant channel first
(see e.g. `Equivariant_Multilayer` in multilayer_perceptron.py, which only
activates the l=0 component of its output irreps).
"""
from torch import nn
import torch
from torch.nn import functional as F


class Swish(nn.Module):
    '''Per-feature learnable Swish/SiLU activation (also known as
    "learnable swish"), following SpookyNet: https://doi.org/10.1038/s41467-021-27504-0

    Standard Swish is x*sigmoid(x). Here both the output scale (alpha) and
    the sigmoid's steepness (beta) are per-channel learnable parameters
    initialized to alpha=1.0, beta=1.702 (the value that makes Swish closely
    match GELU at initialization), so the network can adapt the activation
    shape per feature channel during training.
    '''
    def __init__(self,num_features,inital_alpha=1.00,inital_beta=1.702):
        super(Swish, self).__init__()
        self.initial_alpha = inital_alpha
        self.initial_beta = inital_beta
        self.register_parameter("alpha", nn.Parameter(torch.Tensor(num_features)))
        self.register_parameter("beta", nn.Parameter(torch.Tensor(num_features)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.alpha, self.initial_alpha)
        nn.init.constant_(self.beta, self.initial_beta)

    def forward(self,x):
        return x*self.alpha*torch.sigmoid(self.beta*x)




class HardSwish(nn.Module):
    '''Piecewise-linear approximation of Swish (x*sigmoid(x)), using
    relu6(x+3)/6 in place of the sigmoid. Cheaper to evaluate than Swish,
    at the cost of a fixed (non-learnable) shape.'''
    def __init__(self,inplace=False):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.

class ShiftedSoftplus(nn.Module):
    '''Softplus shifted so that ShiftedSoftplus(0) == 0, following SchNet:
    https://doi.org/10.1063/1.5019779 (softplus(x) - log(2)).'''
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()
    def forward(self, x):
        return F.softplus(x) - self.shift

def activations(type,num_features=128):
    '''Factory returning an activation module instance by name.

    `num_features` is only needed for `'swish'`, since Swish has one
    learnable (alpha, beta) pair per channel; it's ignored for the other,
    parameter-free activation types. `'softmax'` is included here for
    convenience even though it isn't a point-wise activation (it's used by
    the attention modules, e.g. Edge_Attention and Tensorproduct_Attention,
    to normalize attention weights along the last dimension).

    Args:
        type: one of 'swish', 'hardswish', 'shiftedsoftplus', 'softmax',
            'silu', 'relu', 'sigmoid', 'tanh'.
        num_features: channel count, used only by 'swish'.

    Returns:
        An nn.Module implementing the requested activation.
    '''
    act = {'swish': Swish(num_features=num_features),
       'hardswish': HardSwish(inplace=False),
       'shiftedsoftplus': ShiftedSoftplus(),
        'softmax':nn.Softmax(dim=-1),
        'silu':nn.SiLU(),
        'relu':nn.ReLU(),
        'sigmoid':nn.Sigmoid(),
        'tanh':nn.Tanh()
        }
    return act[type]