import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        backbone = models.resnet50(weights=None)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        self.embedding_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward_once(self, x):
        feat = self.feature_extractor(x)
        feat = torch.flatten(feat, 1)
        emb = self.embedding_head(feat)
        return F.normalize(emb, p=2, dim=1)

    def forward(self, x1, x2=None, x3=None):
        e1 = self.forward_once(x1)
        if x2 is None:
            return e1
        e2 = self.forward_once(x2)
        if x3 is None:
            return e1, e2
        e3 = self.forward_once(x3)
        return e1, e2, e3

