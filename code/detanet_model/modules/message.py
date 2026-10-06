"""Message construction: combines radial (Edge_Attention) attention output
with the geometric spherical-harmonic encoding of each edge direction to
produce both an invariant and an equivariant per-edge message, ready to be
aggregated (summed over neighbors) in update.py.
"""
from e3nn import o3
import torch
from torch import nn
from .edge_attention import Edge_Attention

class Message(nn.Module):
    '''Wraps Edge_Attention with an equivariant tensor product against the
    edge's spherical-harmonic direction encoding, producing the two message
    components consumed by Update: an invariant part `mijs` and an
    equivariant (irreps) part `mijt`.

    Args:
        head: number of attention heads, passed through to Edge_Attention.
        num_radial: radial basis dimensionality, passed through to
            Edge_Attention.
        num_features: scalar feature dimension.
        irreps_sh: e3nn.o3.Irreps describing the spherical-harmonic degrees
            used to encode edge directions (i.e. `DetaNet.irreps_sh`,
            degrees 1..maxl).
        act: activation name, passed through to Edge_Attention.
    '''
    def __init__(self,head,num_radial,num_features,irreps_sh,act):
        super(Message,self).__init__()
        self.feature=num_features
        self.Attention=Edge_Attention(head=head,num_radial=num_radial,num_features=num_features,act=act)
        irreps_mout = []
        instructions = []
        # Tensor product of irrep tensor features generated from scalar features and irrep tensor
        # Taking a scalar of 128 dimensions and a tensor of l=1,2,3,
        # each feature in the scalar is multiplied by the irrep tensor of '1o+2e+3o'.
        # The final result is '128x1o+128x2e+128x3o'
        # It is worth mentioning that each tensor product here is given a learnable weight.
        for i, (_, ir_sh) in enumerate(irreps_sh):
            for ir_out in o3.Irrep('0e') * ir_sh:
                k = len(irreps_mout)
                irreps_mout.append((num_features, ir_out))
                instructions.append((0, i, k, "uvu", True))
        self.tp = o3.TensorProduct(irreps_in1=o3.Irreps([(num_features, (0, 1))]), irreps_in2=irreps_sh,
                                   irreps_out=irreps_mout, instructions=instructions, shared_weights=True,
                                   internal_weights=True)

    def forward(self,S,rbf,sh,index):
        '''
        Args:
            S: current scalar atomic features, shape [num_atoms, num_features].
            rbf: radial basis features per edge, shape [num_edges, num_radial].
            sh: spherical-harmonic encoding of each edge's direction vector,
                shape [num_edges, irreps_sh.dim].
            index: edge_index, shape [2, num_edges], as (i, j) atom index pairs.

        Returns:
            mijt: equivariant per-edge message, irreps `irreps_mout`
                (a tensor product of the invariant half of eij with `sh`).
            mijs: invariant per-edge message, shape [num_edges, num_features]
                (the other half of eij, passed straight through).
        '''
        # First invariant features and radial features generate edge features(eij) by attention mechanism
        eij=self.Attention(S=S,rbf=rbf,index=index)
        # eij is split into two num_features-sized halves: mijs2 feeds the
        # tensor product below to build the equivariant message, while mijs
        # is used as-is as the invariant message.
        mijs2,mijs=torch.split(eij,split_size_or_sections=[self.feature, self.feature],dim=-1)
        mijt=self.tp(mijs2,sh)
        return mijt,mijs


