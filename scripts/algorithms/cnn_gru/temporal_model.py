import torch
import torch.nn as nn

class TemporalModel(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=128, num_layers=1, bidirectional=False):
        """
        GRUを用いた時系列モデル
        Parameters:
            input_dim     : 入力特徴ベクトルの次元（例：1280）
            hidden_dim    : 隠れ状態の次元（例：128）
            num_layers    : GRU層の数
            bidirectional : 双方向GRUを使用するか
        """
        super(TemporalModel, self).__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.output_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, x):
        """
        x: Tensor, shape = (B, T, F) = (バッチ, 時系列長6, 特徴次元1280)
        return: Tensor, shape = (B, output_dim)
        """
        out, _ = self.gru(x)  # out.shape = (B, T, H)
        last_output = out[:, -1, :]  # 最後のステップの出力を使用
        return last_output
