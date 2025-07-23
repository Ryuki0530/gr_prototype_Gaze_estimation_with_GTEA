import os
import sys
import cv2
import torch
import numpy as np
from collections import deque
from .gaze_mobilenetv3 import GazeMobileNetV3
# 継承元を読み込むためのパス設定
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from base import GazeEstimator

class GazeEstimatorMobileNet(GazeEstimator):
    def __init__(self, model_path="scripts/algorithms/cnn_doubleframe/model_epoch750.pth", img_size=224, frame_history=90):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        self.frame_history = frame_history
        self.model = GazeMobileNetV3().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.frame_queue = deque(maxlen=frame_history)

    def preprocess_frame(self, frame):
        resized = cv2.resize(frame, (self.img_size, self.img_size))
        normalized = resized.astype(np.float32) / 255.0
        tensor = torch.tensor(normalized).permute(2, 0, 1)  # (C, H, W)
        return tensor

    def estimate_gaze(self, frame):
        h, w = frame.shape[:2]
        current = self.preprocess_frame(frame)
        self.frame_queue.append(current)

        if len(self.frame_queue) < self.frame_history:
            # まだ3秒分のフレームが揃っていない間は中央下を仮想注視点とする
            return (w // 2, int(h * 2 / 3))

        past = self.frame_queue[0] * 0.3
        stacked = torch.cat([current, past], dim=0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            gaze = self.model(stacked)[0].cpu().numpy()  # 正規化された (x, y)

        gaze_x = int(gaze[0] * w)
        gaze_y = int(gaze[1] * h)
        return (gaze_x, gaze_y)

