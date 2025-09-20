import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data

class model(nn.Module):
    def __init__(self, pretrained_model: nn.Module, input_dim: int, hidden_dim: int,
                 out_dim: int, num_sensitive_classes: int = 2, adv_alpha: float = 1.0):
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, 2)
        self.encoder = pretrained_model

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.attribute_prompt_generator = AttributePromptGenerator(
            feature_dim=input_dim, hidden_dim=64, output_dim=input_dim
        )
        self.struct_prompt_generator1 = StructPromptGenerator(
            node_dim=input_dim, hidden_dim=64, output_dim=out_dim
        )
        self.struct_prompt_generator2 = StructPromptGenerator(
            node_dim=out_dim, hidden_dim=64, output_dim=hidden_dim
        )

        self.conv1 = StructPromptGCNConv(input_dim, out_dim)
        self.conv2 = StructPromptGCNConv(out_dim, hidden_dim)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, input_dim)
        )

        self.grl = AdversarialLayer(alpha=adv_alpha)
        self.adv_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, num_sensitive_classes)
        )

        self.counterfactual_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, data: Data):
        attribute_prompt_mask = self.attribute_prompt_generator(data.x)
        x_prompted = data.x * attribute_prompt_mask

        struct_prompt1 = self.struct_prompt_generator1(x_prompted, data.edge_index)

        h1 = self.conv1(x_prompted, data.edge_index, struct_prompt1)
        h1 = F.LeakyReLU(h1)
        h1 = F.dropout(h1, p=0.2, training=self.training)

        struct_prompt2 = self.struct_prompt_generator2(h1, data.edge_index)

        h2 = self.conv2(h1, data.edge_index, struct_prompt2)
        h2 = F.LeakyReLU(h2)
        h2 = F.dropout(h2, p=0.2, training=self.training)

        x_recon = self.decoder(h2)

        logits = self.classifier(h2)

        adv_features = self.grl(h2)
        adv_logits = self.adv_classifier(adv_features)

        cf_representations = self.counterfactual_generator(h2)

        return x_recon, h2, attribute_prompt_mask, (struct_prompt1, struct_prompt2), logits, adv_logits, cf_representations
