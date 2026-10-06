"""Update step: aggregates per-edge messages into per-atom features, then
refines them with an atomwise (no neighbor/edge dependence) self-attention
that mixes across an atom's own tensor-degree channels.
"""
from e3nn import o3
from .acts import activations
from torch_scatter import scatter
from torch import nn
import torch



class Tensorproduct_Attention(nn.Module):
    '''Atomwise self-attention over an atom's own irreps tensor feature T
    and scalar feature S — no neighbor or edge information enters here.

    Query and key are both linear projections of the same T; a
    feature-wise ("uuu") tensor product of query and key produces an
    invariant attention score per channel, which is turned into a gate via
    softmax and used to scale a value projection of T (equivariant output)
    and of S (invariant output).

    Args:
        num_features: scalar feature dimension.
        irreps_T: e3nn.o3.Irreps of the tensor feature T.
        act: activation name for the invariant branch.
    '''
    def __init__(self,num_features,irreps_T,act,):
        super(Tensorproduct_Attention, self).__init__()
        self.feature=num_features
        self.lq=o3.Linear(irreps_T,irreps_T, internal_weights=True,shared_weights=True)
        self.lk=o3.Linear(irreps_T,irreps_T, internal_weights=True,shared_weights=True)
        self.lv=o3.Linear(irreps_T,irreps_T, internal_weights=True,shared_weights=True)
        self.ls=nn.Linear(num_features,2*num_features)
        self.lvs=nn.Linear(num_features,num_features)
        irreps_scalar = o3.Irreps([(num_features, (0, 1))])
        intp1=[]
        intp2=[]
        # 'uuu' is the feature-wise tensor product
        # That is, a tensor product is performed for each irrep tensor feature in each set of irrep tensor features.
        # For example, two 128-dimensional '1o' features are multiplied 'uuuu'
        # each of the 128 features are multiplied to produce a set of '0e+1o+2e', resulting in '128x0e+128x1o+128x2e'
        for i, (_, _) in enumerate(o3.Irreps(irreps_T)):
                intp1.append((i,i,0,'uuu',False))
                intp2.append((0,i,i,'uuu',True))
        self.tp1=o3.TensorProduct(irreps_T,irreps_T,irreps_scalar,instructions=intp1)
        self.tp2=o3.TensorProduct(irreps_scalar,irreps_T,irreps_T,instructions=intp2)
        self.softmax=activations('softmax')
        self.actlvs=activations(act, num_features=num_features)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.ls.weight)
        self.ls.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lvs.weight)
        self.lvs.bias.data.fill_(0)

    def forward(self,T,S):
        # First, the feature-wise tensor product is performed on the query and key of the irrep tensor feature,
        # and the attention feature is generated
        s=self.tp1(self.lq(T),self.lk(T))
        # The generated attention feature is passed through a linear layer and a softmax function
        # and is divided into 2 parts
        su,sd=torch.split(self.softmax(self.ls(s)),split_size_or_sections=[self.feature,self.feature],dim=-1)
        #The value of final scalar and irrep tensor features are multiplied by attention feature to generate the result
        tu=self.tp2(sd,self.lv(T))
        return tu,su*self.actlvs(self.lvs(S))

class Update(nn.Module):
    '''Aggregates per-edge messages (from Message) into per-atom features
    via a scatter-sum, adds them residually to T/S, then applies
    Tensorproduct_Attention as a second residual refinement.

    Args:
        num_features: scalar feature dimension.
        act: activation name.
        irreps_mout: e3nn.o3.Irreps of the incoming equivariant message
            `mijt` (i.e. `Message.tp.irreps_out`).
        irreps_T: e3nn.o3.Irreps of the tensor feature T being updated.
        dropout: dropout probability applied to the (invariant) scalar
            update terms.
    '''
    def __init__(self,num_features,act,irreps_mout,irreps_T,dropout=0.0):
        super(Update, self).__init__()
        self.actu=activations(act,num_features=num_features)
        self.drop=nn.Dropout(dropout)
        self.outt = o3.Linear(irreps_in=irreps_mout, irreps_out=irreps_T, internal_weights=True,
                                shared_weights=True)
        self.outs=nn.Linear(num_features,num_features)

        self.uattn=Tensorproduct_Attention(num_features=num_features,irreps_T=irreps_T,act=act)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.outs.weight)
        self.outs.bias.data.fill_(0)
    def forward(self,T,S,mijt,mijs,index):
        '''
        Args:
            T: current tensor (irreps) atomic features, irreps `irreps_T`.
            S: current scalar atomic features, shape [num_atoms, num_features].
            mijt, mijs: per-edge equivariant/invariant messages from Message.forward.
            index: edge_index, shape [2, num_edges].

        Returns:
            Updated (T, S), same shapes as the inputs.
        '''
        # Residual update: aggregate per-edge messages onto atoms (scatter-
        # sum grouped by index[1]), add them to T/S, then apply a second
        # residual refinement via atomwise Tensorproduct_Attention.
        j=index[1]
        ut=self.outt(scatter(src=mijt,index=j,dim=0))
        us=self.actu(self.outs(scatter(src=mijs,index=j,dim=0)))
        T=T+ut
        S=S+self.drop(us)
        ut2,us2=self.uattn(T=T,S=S)
        T=T+ut2
        S=S+self.drop(us2)
        return T,S



