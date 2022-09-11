import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch_geometric.data import Data
from torch_geometric.nn import GINConv
from torch_geometric.nn import JumpingKnowledge

from federatedscope.core.mlp import MLP
from federatedscope.gfl.model.gtricks import random_feature
"""
Model param names of GIN:
[
    'convs.0.eps',
    'convs.0.nn.linears.0.weight',
    'convs.0.nn.linears.0.bias',
    'convs.0.nn.linears.1.weight',
    'convs.0.nn.linears.1.bias',
    'convs.0.nn.norms.0.weight',
    'convs.0.nn.norms.0.bias',
    'convs.0.nn.norms.0.running_mean',
    'convs.0.nn.norms.0.running_var',
    'convs.0.nn.norms.0.num_batches_tracked',
    'convs.0.nn.norms.1.weight',
    'convs.0.nn.norms.1.bias',
    'convs.0.nn.norms.1.running_mean',
    'convs.0.nn.norms.1.running_var',
    'convs.0.nn.norms.1.num_batches_tracked',
    'convs.1.eps',
    'convs.1.nn.linears.0.weight',
    'convs.1.nn.linears.0.bias',
    'convs.1.nn.linears.1.weight',
    'convs.1.nn.linears.1.bias',
    'convs.1.nn.norms.0.weight',
    'convs.1.nn.norms.0.bias',
    'convs.1.nn.norms.0.running_mean',
    'convs.1.nn.norms.0.running_var',
    'convs.1.nn.norms.0.num_batches_tracked',
    'convs.1.nn.norms.1.weight',
    'convs.1.nn.norms.1.bias',
    'convs.1.nn.norms.1.running_mean',
    'convs.1.nn.norms.1.running_var',
    'convs.1.nn.norms.1.num_batches_tracked',
]
"""


class GIN_Net(torch.nn.Module):
    r"""Graph Isomorphism Network model from the "How Powerful are Graph
    Neural Networks?" paper, in ICLR'19

    Arguments:
        in_channels (int): dimension of input.
        out_channels (int): dimension of output.
        hidden (int): dimension of hidden units, default=64.
        max_depth (int): layers of GNN, default=2.
        dropout (float): dropout ratio, default=.0.

    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden=64,
                 max_depth=2,
                 dropout=.0,
                 use_random_feature=False,
                 jk_mode='last'):
        super(GIN_Net, self).__init__()
        
        self.use_rf = use_random_feature
        if self.use_rf:
            hidden += 1 # the dim of hidden state plus 1

        self.convs = ModuleList()
        for i in range(max_depth):
            if i == 0:
                self.convs.append(
                    GINConv(MLP([in_channels, hidden, hidden],
                                batch_norm=True)))
            elif (i + 1) == max_depth:
                self.convs.append(
                    GINConv(
                        MLP([hidden, hidden, out_channels], batch_norm=True)))
            else:
                self.convs.append(
                    GINConv(MLP([hidden, hidden, hidden], batch_norm=True)))
        self.dropout = dropout
        
        self.jk_mode = jk_mode
        # JK连接类型，默认'last'即不连接 'cat', 'max', or 'lstm'
        if jk_mode == 'max':
            self.jk = JumpingKnowledge(mode=jk_mode)
            self.jk_out = torch.nn.Linear(hidden, out_channels)
        elif jk_mode == 'cat':
            self.jk = JumpingKnowledge(mode=jk_mode)
            self.jk_out = torch.nn.Linear(max_depth * hidden, out_channels)
        elif jk_mode == 'last':
            self.jk_out = lambda layer_outs: layer_outs[-1]
        else:
            assert NotImplementedError

    def forward(self, data):
        if isinstance(data, Data):
            x, edge_index = data.x, data.edge_index
        elif isinstance(data, tuple):
            x, edge_index = data
        else:
            raise TypeError('Unsupported data type!')
            
        if self.use_rf:
            x = random_feature(x)

        layer_outs = [x]
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if (i + 1) == len(self.convs):
                break
            x = F.relu(F.dropout(x, p=self.dropout, training=self.training))
            layer_outs.append(x)
            
        if self.jk_mode != 'last':
            x = self.jk(layer_outs)  # JK层
        x = self.jk_out(x)
        x = F.log_softmax(x, dim=1)
        return x
