import torch
import torch.nn as nn
import torchvision.models as models

class GazeCNN(nn.Module):
    def __init__(self, pretrained=True):
        super(GazeCNN, self).__init__()
        
        # MobileNetV2 をベースにする（出力 1280次元）
        mobilenet = models.mobilenet_v2(pretrained=pretrained)
        self.features = mobilenet.features  # 特徴抽出部分（畳み込み層）

        # グローバルプーリングで flatten
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # 視線位置を回帰する全結合層
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2),
            nn.Sigmoid()  # 出力は [0, 1] 範囲の相対座標
        )

    def forward(self, x):
        x = self.features(x)  # 畳み込み特徴抽出
        x = self.pool(x)      # グローバルプーリングで 1x1 に
        x = self.regressor(x) # 2次元座標に回帰
        return x              # [batch_size, 2]
