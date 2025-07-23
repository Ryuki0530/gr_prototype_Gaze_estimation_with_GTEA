import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

class GazeMobileNetV3(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        # MobileNetV3 Small をロード
        self.base = mobilenet_v3_small(pretrained=pretrained)
        
        # 入力チャンネル数を 6 に変更（3+3）
        # ConvBNActivation -> Conv2d(6, 16, ...)
        self.base.features[0][0] = nn.Conv2d(
            in_channels=6, 
            out_channels=16, 
            kernel_size=3, 
            stride=2, 
            padding=1, 
            bias=False
        )
        
        # 出力層を視線座標 (x, y) に変更
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.regressor = nn.Linear(576, 2)  # 576 = V3 Smallの最終特徴数

    def forward(self, x):
        x = self.base.features(x)   # (B, 576, 7, 7)
        x = self.pool(x)            # (B, 576, 1, 1)
        x = x.view(x.size(0), -1)   # (B, 576)
        out = self.regressor(x)     # (B, 2)
        return out
