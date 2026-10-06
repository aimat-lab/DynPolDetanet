"""Radial (edge-level) self-attention: the first stage of message passing.

For each directed edge i<-j in the molecular graph, computes an edge
message by attending over the neighbor j's scalar features, with the
attention's key/value additionally modulated by a radial-basis embedding
of the i-j distance (from Radial_Basis). This is what makes the message
distance-dependent, and is distinct from the atomwise self-attention in
update.py, which instead attends across an atom's own tensor-feature
channels (no neighbor/edge dependence, no distance dependence).
"""
from torch import nn
from .acts import activations

class Edge_Attention(nn.Module):
    '''Radial (distance-weighted, edge-indexed) self-attention module.

    Query comes from the central atom i's scalar feature S_i; key and
    value come from the neighbor atom j's scalar feature S_j, each
    additionally scaled by its own linear projection of the radial basis
    features for the i-j distance. The resulting attention output is
    split downstream (in Message) into an invariant part and a part that
    gets tensor-producted with spherical harmonics to build the
    equivariant edge message.

    Args:
        num_radial: dimensionality of the radial basis input.
        num_features: scalar feature dimension (must match the rest of the
            model's `num_features`).
        act: activation name, passed to `activations()`.
        head: number of attention heads; `num_features*2` must be
            divisible by `head` (the output is split into query/key/value
            of size `2*num_features`, then reshaped per-head).
    '''
    def __init__(self,num_radial,num_features,act,head=8):
        super(Edge_Attention,self).__init__()
        self.head=head
        self.actq = activations(act,num_features=num_features)
        self.actk = activations(act, num_features=num_features)
        self.actv = activations(act, num_features=2*num_features)
        self.acta = activations(act, num_features=2*num_features)
        self.softmax=activations('softmax')
        self.lq=nn.Linear(num_features,num_features)
        self.lk=nn.Linear(num_features,num_features)
        self.lv=nn.Linear(num_features,2*num_features)
        self.la=nn.Linear(2*num_features,2*num_features)
        self.lrbf=nn.Linear(num_radial,num_features,bias=False)
        self.lkrbf=nn.Linear(num_features,num_features,bias=False)
        self.lvrbf=nn.Linear(num_features,2*num_features,bias=False)
        self.feature=num_features
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lq.weight)
        self.lq.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lk.weight)
        self.lk.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lv.weight)
        self.lv.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lrbf.weight)
        nn.init.xavier_uniform_(self.lkrbf.weight)
        nn.init.xavier_uniform_(self.lvrbf.weight)
        nn.init.xavier_uniform_(self.la.weight)
        self.la.bias.data.fill_(0)


    def resize(self,x):
        '''Reshape [num_edges, dim] -> [num_edges, head, dim/head] for
        multi-head attention.'''
        return x.reshape(x.shape[0],self.head,-1)

    def attention(self, Q, K, V):
        '''Standard scaled dot-product attention (per edge, per head),
        followed by flattening heads back into a single feature dimension
        of size 2*num_features.'''
        d = Q.shape[-1]
        dot = Q @ K.permute(0, 2, 1)
        A = self.softmax(dot / (d**0.5))
        return (A @ V).reshape(-1, 2 * self.feature)

    def forward(self,S,rbf,index):
        i,j=index
        #The invariant feature S through 3 different linear layers to obtain query,key and value node feature
        sq=self.actq(self.lq(S))
        sk=self.actk(self.lk(S))
        sv=self.actv(self.lv(S))
        #The radial baseline is passed through a weight matrix
        #and then through two different weight matrices to obtain the radial weights of key and value
        rbf=self.lrbf(rbf)
        rk=self.lkrbf(rbf)
        rv=self.lvrbf(rbf)
        q=sq[i]
        k=sk[j]*rk
        v=sv[j]*rv
        #Finally an attention mechanism is performed and then a linear layer is output.
        #q is the feature of atom i,and k and v are the products of the features of atom j and the radial weights.
        return self.acta(self.la(self.attention(Q=self.resize(q),K=self.resize(k),V=self.resize(v))))



