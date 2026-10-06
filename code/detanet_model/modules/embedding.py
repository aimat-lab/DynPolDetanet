"""Initial per-atom scalar embedding: turns a bare atomic number into the
model's first invariant (l=0) feature vector S^0, combining a learned
nuclear (one-hot) embedding with a handcrafted electronic-structure
descriptor. This is the very first step of the forward pass, before any
message passing between atoms happens.
"""
from torch import nn,device,tensor,float32
from .acts import activations
def get_elec_feature(max_atomic_number,device):
    '''
    Handcrafted per-element electronic-structure descriptor, covering
    elements up to Kr (Z=36): the first 4 periods of the periodic table.

    For each element, encodes how many paired ("P") and unpaired/single
    ("S") electrons occupy each of the 8 orbital groups relevant across
    these periods (1s, 2s, 2p, 3s, 3p, 4s, 3d, 4p), giving a 16-dimensional
    vector per element (2 counts x 8 orbital groups). This implicitly
    encodes chemically meaningful information the model can't easily infer
    from the atomic number alone: valence electron count, whether shells
    are filled/half-filled, etc. Row 0 is a dummy padding entry (Z=0,
    "None"), matching the same indexing convention as `atom_masses` in
    constant.py, so this table can be indexed directly by atomic number.

    Args:
        max_atomic_number: largest atomic number the model will see;
            entries above this are dropped from the returned table.
        device: torch device to place the returned tensor on.

    Returns:
        Tensor of shape [max_atomic_number+1, 16], normalized so the
        overall maximum entry is 1 (simple scale normalization, not
        per-row/per-column).
      '''
    elec_feature = tensor(
        # P:Pair electron
        # S:Single electron
        #|  1s |  2s |  2p |  3s |  3p |  4s |  3d |  4p |
        # P  S  P  S  P  S  P  S  P  S  P  S  P  S  P  S
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],  # 0 None
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],  # 1 H
         [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],  # 2 He
         [2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],  # 3 Li
         [2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],  # 4 Be
         [2, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],  # 5 B
         [2, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],  # 6 C
         [2, 0, 2, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],  # 7 N
         [2, 0, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],  # 8 O
         [2, 0, 2, 0, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],  # 9 F
         [2, 0, 2, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],  # 10 Ne
         [2, 0, 2, 0, 6, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, ],  # 11 Na
         [2, 0, 2, 0, 6, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],  # 12 Me
         [2, 0, 2, 0, 6, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, ],  # 13 Al
         [2, 0, 2, 0, 6, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, ],  # 14 Si
         [2, 0, 2, 0, 6, 0, 2, 0, 0, 3, 0, 0, 0, 0, 0, 0, ],  # 15 P
         [2, 0, 2, 0, 6, 0, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, ],  # 16 S
         [2, 0, 2, 0, 6, 0, 2, 0, 4, 1, 0, 0, 0, 0, 0, 0, ],  # 17 Cl
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 0, 0, 0, 0, 0, 0, ],  # 18 Ar
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 0, 1, 0, 0, 0, 0, ],  # 19 K
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 2, 0, 0, 0, 0, 0, ],  # 20 Ca
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 2, 0, 0, 1, 0, 0, ],  # 21 Sc
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 2, 0, 0, 2, 0, 0, ],  # 22 Ti
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 2, 0, 0, 3, 0, 0, ],  # 23 V
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 0, 1, 0, 5, 0, 0, ],  # 24 Cr
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 2, 0, 0, 5, 0, 0, ],  # 25 Mn
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 2, 0, 2, 4, 0, 0, ],  # 26 Fe
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 2, 0, 4, 3, 0, 0, ],  # 27 Co
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 2, 0, 6, 2, 0, 0, ],  # 28 Ni
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 0, 1,10, 0, 0, 0, ],  # 29 Cu
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 2, 0,10, 0, 0, 0, ],  # 30 Zn
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 2, 0,10, 0, 0, 1, ],  # 31 Ga
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 2, 0,10, 0, 0, 2, ],  # 32 Ge
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 2, 0,10, 0, 0, 3, ],  # 33 As
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 2, 0,10, 0, 2, 2, ],  # 34 Se
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 2, 0,10, 0, 4, 1, ],  # 35 Br
         [2, 0, 2, 0, 6, 0, 2, 0, 6, 0, 2, 0,10, 0, 6, 0, ],  # 36 Kr
         ], dtype=float32)
    elec=elec_feature[0:max_atomic_number+1]
    return (elec/elec.max()).to(device)

class Embedding(nn.Module):
    '''Builds the initial invariant (scalar) atomic feature S^0 from two
    parallel branches, each a distinct learnable layer:

    - a nuclear branch: `nuclare_emb`, an `nn.Embedding` lookup table over
      atomic number Z (functionally a learned one-hot encoding + linear
      layer combined into a single embedding table);
    - an electronic branch: `elec_emb`, an `nn.Linear` applied to the fixed
      16-dim handcrafted descriptor from `get_elec_feature`.

    The two branches' outputs are summed and passed through a shared
    integrating linear layer (`ls`) plus a non-linearity, giving
    S^0 = act(ls(nuclare_emb(Z) + elec_emb(elec_feature(Z)))).
    Note that UV-vis spectral features, when used, are concatenated onto
    this embedding *outside* this module (see `x_features` handling in
    `DetaNet.forward`), not here.

    Args:
        num_features: output feature dimension (must match the rest of the
            model's `num_features`).
        act: activation name, passed to `activations()`.
        max_atomic_number: largest atomic number to support; determines
            the size of the nuclear embedding table and the electronic
            descriptor table.
        device: torch device for the (fixed, non-learnable) electronic
            descriptor table.
    '''
    def __init__(self,num_features,act,max_atomic_number=9,device=device('cuda')):
        super(Embedding,self).__init__()
        self.elec=get_elec_feature(max_atomic_number=max_atomic_number,device=device)
        self.act=activations(type=act,num_features=num_features)
        self.elec_emb = nn.Linear(16, num_features, bias=False)
        self.nuclare_emb = nn.Embedding(max_atomic_number+1, num_features)
        self.ls = nn.Linear(num_features, num_features)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.ls.weight)
        self.ls.bias.data.fill_(0)
        self.nuclare_emb.reset_parameters()
        nn.init.xavier_uniform_(self.elec_emb.weight)

    def forward(self,z):
        '''
        The initial invariant feature of an atom consists of a mixture of nuclear one-hot and electronic features
        S0=ls(ln(O(z))+lq(Q(z)))
        '''
        return self.act(self.ls(self.nuclare_emb(z)+self.elec_emb(self.elec[z,:])))
