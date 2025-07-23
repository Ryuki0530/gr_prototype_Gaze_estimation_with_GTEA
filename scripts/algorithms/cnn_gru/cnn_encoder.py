import torch
import torch.nn as nn
import torchvision.models as models

class CNNEncoder(nn.Module):
    def __init__(self, output_dim=1280, freeze=True):
        """
        MobileNetV2ベースのCNNエンコーダー

        parameters:
            output_dim : mobile_net_v2の出力次元数 1280 
            freeze : true モデルの重みを固定するかどうか
        """
        super(CNNEncoder,self).__init__()

        #事前学習済みのMobileNetV2のfeatureをロード
        base_model = models.mobilenet_v2(pretrained = True)
        self.features = base_model.features

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_dim = output_dim

        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        """
        x: Tensor, shape = (B, C, H, W) = (バッチ, チャンネル数3, 高さ224, 幅224)
        return: Tensor, shape = (B, output_dim) = (バッチ, 1280)
        """
        x = self.features(x)  # x.shape = (B, C, H, W)
        x = self.pool(x)  # x.shape = (B, C, 1, 1)
        x = x.view(x.size(0), -1)  # x.shape = (B, C)
        return x

