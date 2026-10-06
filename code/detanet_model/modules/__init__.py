"""Building blocks of the DetaNet architecture. See each submodule's own
docstring for details; roughly in forward-pass order: Embedding (initial
per-atom features) -> Radial_Basis (distance encoding) -> Edge_Attention +
Message (per-edge messages) -> Update (aggregation + atomwise refinement),
wired together per interaction layer by Interaction_Block. MLP/
Equivariant_Multilayer are the final output heads, activations() is the
shared non-linearity factory used throughout."""
from .edge_attention import Edge_Attention
from .embedding import Embedding
from .radial_basis import Radial_Basis
from .message import Message
from .update import Update
from .block import Interaction_Block
from .multilayer_perceptron import MLP,Equivariant_Multilayer
from .acts import activations