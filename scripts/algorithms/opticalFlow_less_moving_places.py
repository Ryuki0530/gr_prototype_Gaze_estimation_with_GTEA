import numpy as np
import cv2
from .base import GazeEstimator

class OpticalFlowLessMovingPlacesEstimator(GazeEstimator):
    def __init__(self, grid_size=(4, 4)):
        self.prev_gray = None
        self.grid_size = grid_size

    def estimate_gaze(self, frame):
        """
        画面をグリッド分割し、最も光フロー移動量が小さいグリッドの中心を返す

        Args:
            frame (numpy.ndarray): 入力画像フレーム

        Returns:
            tuple: (x, y) 視線座標（ピクセル座標）
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            # 最初のフレームは画面中心を返す
            h, w = gray.shape
            return (w // 2, h // 2)

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        h, w = gray.shape
        gh, gw = self.grid_size
        grid_h = h // gh
        grid_w = w // gw

        min_motion = None
        min_idx = (0, 0)

        for i in range(gh):
            for j in range(gw):
                y0 = i * grid_h
                x0 = j * grid_w
                y1 = y0 + grid_h
                x1 = x0 + grid_w
                grid_flow = flow[y0:y1, x0:x1]
                # 各グリッドの移動量（ノルムの平均）
                motion = np.mean(np.linalg.norm(grid_flow, axis=2))
                if (min_motion is None) or (motion < min_motion):
                    min_motion = motion
                    min_idx = (i, j)

        # 最小移動グリッドの中心座標を返す
        center_x = min_idx[1] * grid_w + grid_w // 2
        center_y = min_idx[0] * grid_h + grid_h // 2

        self.prev_gray = gray.copy()
        return (center_x, center_y)