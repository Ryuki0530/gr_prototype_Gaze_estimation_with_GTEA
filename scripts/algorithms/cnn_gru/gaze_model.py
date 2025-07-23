import torch
import torch.nn as nn
from .cnn_encoder import CNNEncoder
from .temporal_model import TemporalModel

class GazeEstimationModel(nn.Module):
    def __init__(self, cnn_out_dim=1280, gru_hidden_dim=128):
        super(GazeEstimationModel, self).__init__()
        self.encoder = CNNEncoder()
        self.temporal = TemporalModel(input_dim=cnn_out_dim, hidden_dim=gru_hidden_dim)
        self.fc = nn.Linear(gru_hidden_dim, 2)  # 視線位置 (x, y)

    def forward(self, x):
        """
        x: Tensor (B, T=6, C=3, H=224, W=224)
        return: Tensor (B, 2)
        """
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)         # → (B*T, 3, 224, 224)
        feats = self.encoder(x)           # → (B*T, 1280)
        feats = feats.view(B, T, -1)      # → (B, T, 1280)
        temp = self.temporal(feats)       # → (B, 128)
        out = self.fc(temp)               # → (B, 2)
        return out
