"""Top-level interaction layer: one full message-passing step, combining
Message (edge-level, distance-dependent) and Update (aggregation +
atomwise refinement). DetaNet stacks `num_block` of these in sequence
(see DetaNet.blocks in detanet.py).
"""
from torch import nn
from .update import Update
from .message import Message
class Interaction_Block(nn.Module):
    '''One interaction (message-passing) layer: builds per-edge messages
    with Message, then aggregates and refines per-atom features with
    Update. Stateless wiring only — see Message/Update for the actual
    computation.'''
    def __init__(self,
                 num_features,
                 act,
                 head,
                 num_radial,
                 irreps_sh,
                 irreps_T,
                 dropout
                 ):
        super(Interaction_Block,self).__init__()
        self.message=Message(head=head,num_radial=num_radial,act=act,
                             num_features=num_features,irreps_sh=irreps_sh)
        self.update=Update(num_features=num_features,act=act,irreps_mout=self.message.tp.irreps_out,
                           irreps_T=irreps_T,dropout=dropout)

    def forward(self,S,T,rbf,sh,index):
        '''
        Args:
            S: scalar atomic features, shape [num_atoms, num_features].
            T: tensor (irreps) atomic features, irreps `irreps_T`.
            rbf: radial basis features per edge, shape [num_edges, num_radial].
            sh: spherical-harmonic edge-direction encoding, shape [num_edges, irreps_sh.dim].
            index: edge_index, shape [2, num_edges].

        Returns:
            (S, T): updated scalar and tensor atomic features.
        '''
        mijt,mijs=self.message(S=S,rbf=rbf,sh=sh,index=index)
        T,S=self.update(T=T,S=S,mijt=mijt,mijs=mijs,index=index)
        return S,T





