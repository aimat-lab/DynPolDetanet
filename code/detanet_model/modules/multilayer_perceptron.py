"""Output-head MLPs: turn accumulated per-atom scalar/tensor features into
final property predictions, after all interaction layers have run.
"""
from torch import nn
from e3nn import o3
from e3nn.nn import Activation
from .acts import activations
class MLP(nn.Module):
    '''Standard invariant multi-layer perceptron (scalar features only).
    Used for the final scalar output head (`DetaNet.sout`).

    Args:
        size: sequence of layer widths, e.g. (in, hidden, ..., out);
            must have at least 2 entries.
        act: activation name, applied between all but the last layer
            (unless `last_act=True`).
        bias: whether Linear layers have a bias term.
        last_act: if True, also apply activation (+dropout) after the
            final layer; if False (default), the final layer's output is
            returned without activation.
        dropout: dropout probability applied after each activation.
    '''
    def __init__(self,size,act,bias=True,last_act=False,dropout=0.0):
        super(MLP,self).__init__()
        assert len(size)>1,'Multilayer perceptrons must be larger than one layer'

        mlp=[]
        for si in range(0,len(size)-1):

            l=nn.Linear(size[si], size[si + 1], bias=bias)
            nn.init.xavier_uniform_(l.weight)
            l.bias.data.fill_(0)
            mlp.append(l)
            activation = activations(act, num_features=size[si + 1])
            if last_act is False:
                if (si!=len(size)-2):

                    mlp.append(activation)
                    mlp.append(nn.Dropout(p=dropout))
            else:
                mlp.append(activation)
                mlp.append(nn.Dropout(p=dropout))
        self.mlp=nn.Sequential(*mlp)

    def forward(self,x):
        for f in self.mlp:
            x=f(x)
        return x

class Equivariant_Multilayer(nn.Module):
    '''Equivariant MLP over irreps features (mixed scalar + tensor). Used
    for the final tensor output head (`DetaNet.tout`).

    Each layer is an `o3.Linear` (equivariance-preserving by construction),
    followed by a non-linearity applied *only* to the l=0 (scalar)
    sub-channels of the output irreps (activations can't be applied to
    l>0 channels without breaking equivariance) — l>0 channels pass
    through the Activation module unchanged (acts=None for them).

    Args:
        irreps_list: sequence of e3nn.o3.Irreps, e.g. (irreps_in,
            irreps_hidden, ..., irreps_out); one o3.Linear is created
            between each consecutive pair.
        act: activation name applied to l=0 sub-channels.
        last_act: if True, also activate after the final layer.
    '''
    def __init__(self,irreps_list,act,last_act=False):
        super(Equivariant_Multilayer, self).__init__()
        e_mlp=[]
        for ei in range(0,len(irreps_list)-1):
            acts = []
            irreps_in=irreps_list[ei]
            irreps_out=irreps_list[ei+1]
            l=o3.Linear(irreps_in=irreps_in,irreps_out=irreps_out)
            for irr in o3.Irreps(irreps_out):
                if o3.Irrep('0e') in irr:
                    activation = activations(act, num_features=irr.dim)
                    acts.append(activation)
                else:
                    acts.append(None)
            e_act=Activation(irreps_in=irreps_out,acts=acts)
            e_mlp.append(l)
            if last_act is False:
                if (ei != len(irreps_list) - 2):
                    e_mlp.append(e_act)
            else:
                e_mlp.append(e_act)
        self.e_mlp = nn.Sequential(*e_mlp)

    def forward(self,x):
        for f in self.e_mlp:
            x=f(x)
        return x