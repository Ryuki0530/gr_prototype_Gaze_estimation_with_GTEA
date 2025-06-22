import numpy as np
import cv2
from .base import GazeEstimator

class OpticalFlowMovingRefrectsCenter(GazeEstimator):


    def estimate_gaze(self, frame):
        """
        画像フレームの中心座標を視線位置として返す

        Args:
            frame (numpy.ndarray): 入力画像フレーム

        Returns:
            tuple: (x, y) 画面中心のピクセル座標
        """
        height, width = frame.shape[:2]
        center_x = width // 2
        center_y = height // 2
        return (center_x, center_y)