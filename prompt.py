import torch
import torch.nn as nn
import torch.nn.functional as F

class AttributePromptGenerator(nn.Module):
    def __init__(self, feature_dim, hidden_dim=64, output_dim=None):
        super().__init__()
        output_dim = feature_dim if output_dim is None else output_dim
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.mlp(x)


class StructPromptGenerator(nn.Module):
    def __init__(self, node_dim, hidden_dim=64, output_dim=None):
        super().__init__()
        self.output_dim = node_dim if output_dim is None else output_dim
        self.mlp = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, self.output_dim),
        )

    def forward(self, x, edge_index):
        src, dst = edge_index
        edge_features = torch.cat([x[src], x[dst]], dim=-1)
        return self.mlp(edge_features)
