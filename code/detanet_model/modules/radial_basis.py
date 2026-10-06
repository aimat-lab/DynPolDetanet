"""Radial basis functions for encoding interatomic distances.

Once the molecular graph's edges are built (atom pairs within the model's
cutoff radius), each edge's scalar distance r needs to be expanded into a
higher-dimensional feature vector before it's useful to the attention
mechanism in Edge_Attention. This module provides that expansion, plus an
optional smooth cosine cutoff envelope that can be multiplied in on top.
"""
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

def cosine_cutoff(r,rc):
    '''Smooth cosine envelope that goes from 1 at r=0 to 0 at r=rc, used to
    make the radial encoding (and hence the whole message) vanish smoothly
    as an atom pair approaches the edge of the cutoff radius, avoiding a
    discontinuity when an atom pair crosses in or out of the neighbor list
    as positions change.'''
    zeros = torch.zeros_like(r)
    rc = torch.where(r < rc, r, zeros)
    out=torch.cos(r/rc)
    return torch.where(r<rc,out,zeros)

def softplus_inverse(x):
    '''Inverse of `F.softplus`: given a target softplus *output* x, returns
    the input that would produce it. Used to initialize a raw parameter so
    that after passing through softplus (see Exp_Gaussian_function below,
    which applies softplus to its `_alpha` parameter to keep the effective
    alpha positive), it starts at a chosen positive value.'''
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    return x + torch.log(-torch.expm1(-x))

class Bessel_Function(nn.Module):
    '''Bessel radial basis, first proposed by DimeNet: arXiv:2003.03123v1 [cs.LG] 6 Mar 2020.
    Modified here to add optional learnable weighting of the basis terms.

    Expands a scalar distance r into `num_radial` channels via
    sin(n*pi*r/rc)/r for n=1..num_radial, i.e. the radial part of the
    spherical Bessel functions of the first kind (j_0), which form a
    complete, orthogonal basis on [0, rc]. With `weighting=True`, the
    integer mode index n and the fixed prefactor beta both become learnable
    per-channel parameters (`alpha`, `beta`), letting the network reshape
    the basis away from the analytic Bessel form during training.

    Args:
        num_radial: number of basis channels (output feature dimension).
        rc: length scale of the basis expansion.
        weighting: if True, `alpha` (mode index) and `beta` (denominator
            scale) are learnable parameters instead of a fixed integer
            sequence and 1.0, respectively.
        inital_beta: initial value for the learnable `beta`, when
            `weighting=True`.
    '''
    def __init__(self,num_radial,rc,weighting=True,inital_beta=2.0):
        super(Bessel_Function,self).__init__()
        assert rc !=0
        if weighting:
            self.register_parameter('alpha',
                                    nn.Parameter(torch.arange(0,num_radial,dtype=torch.float32).view(1,num_radial)))
            self.register_parameter('beta',nn.Parameter(torch.Tensor(1,num_radial)))
            nn.init.constant_(self.beta,inital_beta)

        else:
            self.register_buffer("alpha", torch.arange(1,num_radial+1,dtype=torch.float32).view(1,num_radial))
            self.beta = 1.0
        self.rc=rc
        self.prefactor=(2 /rc) ** 0.5

    def forward(self,r,cutoff=None):
        '''Args:
            r: distances, tensor of shape [num_edges].
            cutoff: optional multiplicative envelope (e.g. from
                `cosine_cutoff`), same shape as r.
        Returns:
            Radial basis features, shape [num_edges, num_radial].
        '''
        r = r.view(-1, 1)
        rbf = self.prefactor * torch.sin(self.alpha*torch.pi * r / self.rc) /(self.beta*r)
        if cutoff is not None:
            rbf=rbf*cutoff.view(-1,1)
        return rbf

class Exp_Gaussian_function(nn.Module):
    '''Learnable exponential-Gaussian radial basis, following PhysNet:
    DOI: 10.1021/acs.jctc.9b00181.

    Maps r to exp(-alpha*r), then expands that value with `num_radial`
    Gaussians of fixed, evenly-spaced centers on [0, 1] (in units of
    exp(-alpha*r), which is why they're spaced in "exponential distance"
    rather than linear distance from the atom). Only the decay rate
    `alpha` is learnable (via a softplus-transformed raw parameter, so the
    effective alpha stays positive); the Gaussian centers/width are fixed.

    Args:
        num_radial: number of basis channels.
        init_alpha: initial value of the (softplus-transformed) decay rate.
        no_inf: if True, drops the Gaussian centered at 0 (which would
            correspond to r=infinity) to avoid it saturating at large r.
        exp_weighting: if True, additionally multiplies the output by
            exp(-alpha*r), damping the basis magnitude at large r.
    '''
    def __init__(self,num_radial,init_alpha=0.95,no_inf=False,exp_weighting=False):
        super(Exp_Gaussian_function,self).__init__()
        self.init_alpha=init_alpha
        self.exp_weighting=exp_weighting
        if no_inf:
            self.register_buffer(
                "center",
                torch.linspace(1, 0, num_radial + 1, dtype=torch.float32)[:-1],
            )
            self.register_buffer(
                "width",
                torch.tensor(1.0 * (num_radial + 1), dtype=torch.float32),
            )
        else:
            self.register_buffer(
                "center", torch.linspace(1, 0, num_radial, dtype=torch.float32)
            )
            self.register_buffer(
                "width", torch.tensor(1.0 * num_radial, dtype=torch.float32)
            )
        self.register_parameter(
            "_alpha", nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self._alpha, softplus_inverse(self.init_alpha))

    def forward(self, r, cutoff=None):
        expalphar = torch.exp(-F.softplus(self._alpha) * r.view(-1, 1))
        rbf =torch.exp(-self.width * (expalphar - self.center) ** 2)
        if cutoff is not None:
            rbf=cutoff.view(-1,1)*rbf
        if self.exp_weighting:
            return rbf * expalphar
        else:
            return rbf

class Radial_Basis(nn.Module):
    '''Thin dispatcher over the four available radial basis types
    ('gaussian', 'exp_gaussian', 'bessel', 'trainable_bessel'), with an
    optional cosine cutoff envelope applied on top.

    Args:
        radial_type: one of 'gaussian', 'exp_gaussian', 'bessel',
            'trainable_bessel'; selects which basis class (above) is used.
        cut_function: the cutoff envelope function, applied only if
            `use_cutoff` is True. Defaults to `cosine_cutoff`.
        num_radial: number of basis channels (output feature dimension).
        rc: length scale used both by the Bessel basis types and by the
            cutoff envelope (when enabled).
        use_cutoff: whether to multiply the raw basis output by
            `cut_function(r, rc)`. The Bessel bases already decay smoothly
            within their length scale, so this is generally most useful
            with the Gaussian bases, which don't have a built-in cutoff.
    '''
    def __init__(self,radial_type='bessel',cut_function=cosine_cutoff,num_radial=32,rc=5.0,use_cutoff=False):
        super(Radial_Basis,self).__init__()
        self.rc=rc
        self.radial_type=radial_type
        self.cut=cut_function
        self.use_cutoff = use_cutoff
        radials={'gaussian':Exp_Gaussian_function(num_radial=num_radial,exp_weighting=False),
                 'exp_gaussian':Exp_Gaussian_function(num_radial=num_radial,exp_weighting=True),
                 'bessel':Bessel_Function(num_radial=num_radial,rc=rc,weighting=False),
                 'trainable_bessel':Bessel_Function(num_radial=num_radial,rc=rc,weighting=True),
        }
        self.radial=radials[radial_type]

    def forward(self,r):
        if self.use_cutoff:
            cutoff=self.cut(r=r,rc=self.rc)
        else:
            cutoff=None
        return self.radial(r,cutoff)



