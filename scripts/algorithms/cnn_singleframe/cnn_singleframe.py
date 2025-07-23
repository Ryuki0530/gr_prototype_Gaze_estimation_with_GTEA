import numpy as np
import torch
import os
import sys
from torchvision import transforms
from .gaze_cnn import GazeCNN
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from base import GazeEstimator


class CnnSingleFrameEstimator(GazeEstimator):
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "gaze_model_epoch55.pth")
    IMG_SIZE = (224, 224)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self):
        """
        初期化時にモデルをロード
        """
        self.model = GazeCNN()
        self.model.load_state_dict(torch.load(self.MODEL_PATH, map_location=self.DEVICE))
        self.model = self.model.to(self.DEVICE)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.IMG_SIZE),
            transforms.ToTensor()
        ])

    def estimate_gaze(self, frame):
        """
        画像フレームからCNNで視線座標を推定
        """
        with torch.no_grad():
            input_tensor = self.transform(frame).unsqueeze(0).to(self.DEVICE)
            output = self.model(input_tensor)
            x_norm, y_norm = output[0].cpu().numpy()
        h, w = frame.shape[:2]
        x = int(x_norm * w)
        y = int(y_norm * h)
        return (x, y)
