import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.norm import GraphNorm, PairNorm, MessageNorm, DiffGroupNorm, BatchNorm

from order_parameters import ActivationProbe, ConvolutionProbe

class GCN(torch.nn.Module):
    def __init__(
            self,
            hidden_channels,
            num_input_features,
            num_layers=3,
            task='classification',
            norm=None,
            norm_args=None):
        # Normalization layers need specific kwargs:
        # BatchNorm: in_channels (int)
        # GraphNorm: in_channels (int)
        # PairNorm: scale (int)
        # DiffGroupNorm: in_channels (int), groups (int)
        super(GCN, self).__init__()
        self.num_layers = num_layers

        norm_map = {'BatchNorm': BatchNorm, 'GraphNorm': GraphNorm, 'PairNorm': PairNorm, 'DiffGroupNorm': DiffGroupNorm}
        self.norm = norm

        self.conv_layers = nn.ModuleList()
        self.conv_probes = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.activ_probes = nn.ModuleList()

        for l in range(num_layers):
            if l == 0:
                self.conv_layers.append(GCNConv(num_input_features, hidden_channels))
            else:
                self.conv_layers.append(GCNConv(hidden_channels, hidden_channels))

            self.conv_probes.append(ConvolutionProbe())
            if norm:
                if norm_args:
                    self.norm_layers.append(norm_map[norm](**norm_args))
                else:
                    self.norm_layers.append(norm_map[norm]())

            self.activations.append(nn.ReLU(inplace=True))
            self.activ_probes.append(ActivationProbe())

        self.lin = torch.nn.Linear(hidden_channels, 2) if task == 'classification' else torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        for l in range(self.num_layers):
            x = self.conv_layers[l](x, edge_index)
            x = self.conv_probes[l](x, batch)
            if self.norm:
                x = self.norm_layers[l](x)
            x = self.activations[l](x)
            x = self.activ_probes[l](x, batch)

        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x
