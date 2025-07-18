import torch
import torch.nn.functional as F
import numpy as np
import cv2
from collections import deque
import os
import sys

# 相対インポート
from .gaze_model import GazeEstimationModel
from .cnn_encoder import CNNEncoder
from .temporal_model import TemporalModel
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from base import GazeEstimator

class CnnGruHybridEstimator(GazeEstimator):
    def __init__(self, model_path=None, device=None, frame_size=(224, 224)):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.frame_size = frame_size
        self.buffer = deque(maxlen=6)  # 過去6フレームを保持

        # モデル読み込み
        self.model = None
        self._load_model(model_path)

    def _load_model(self, model_path):
        # デフォルトモデルパス
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "gaze_estimation_model.pth")

        try:
            model = GazeEstimationModel().to(self.device)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            self.model = model
            print(f"✅ モデルを読み込みました: {model_path}")
        except Exception as e:
            print(f"❌ モデルの読み込みに失敗しました: {e}")
            print(f"確認パス: {model_path}")
            self.model = None

    def preprocess(self, frame):
        resized = cv2.resize(frame, self.frame_size)
        img = resized.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # HWC → CHW
        tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # (1, 3, H, W)
        return tensor

    def estimate_gaze(self, frame):
        """
        推論を行い、視線のピクセル座標を返す。
        推論可能なフレームが6個未満の場合は中央を返す。
        """
        if self.model is None:
            return self._default_center(frame)

        self.buffer.append(self.preprocess(frame))
        if len(self.buffer) < 6:
            return self._default_center(frame)

        sequence = torch.cat(list(self.buffer), dim=0).unsqueeze(0).to(self.device)  # (1, 6, 3, 224, 224)

        with torch.no_grad():
            pred = self.model(sequence)  # (1, 2)
        gaze = pred.squeeze().cpu().numpy()

        h, w = frame.shape[:2]
        x = np.clip(int(gaze[0] * w), 0, w - 1)
        y = np.clip(int(gaze[1] * h), 0, h - 1)
        return x, y

    def _default_center(self, frame):
        h, w = frame.shape[:2]
        return w // 2, h // 2

    def draw(self, frame):
        """
        デバッグ・可視化用の情報をフレームに描画する。
        """
        buffer_status = f"Buffer: {len(self.buffer)}/6"
        model_status = "Model: OK" if self.model is not None else "Model: ERROR"

        cv2.putText(frame, buffer_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, model_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if self.model else (0, 0, 255), 2)
