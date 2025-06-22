import numpy as np
import cv2
from .base import GazeEstimator

class OpticalFlowLessMovingPlacesEstimator(GazeEstimator):
    def __init__(self, grid_size=(4, 4), resize_scale=0.5, flow_interval=2):
        self.prev_gray = None
        self.grid_size = grid_size
        self.resize_scale = resize_scale  # 画像縮小率
        self.flow_interval = flow_interval  # 何フレームごとに計算するか
        self.frame_count = 0
        self.last_gaze = None

    def estimate_gaze(self, frame):
        """
        画面をグリッド分割し、最も光フロー移動量が小さいグリッドの中心を返す

        Args:
            frame (numpy.ndarray): 入力画像フレーム

        Returns:
            tuple: (x, y) 視線座標（ピクセル座標）
        """
        self.frame_count += 1

        # 画像を縮小
        small_frame = cv2.resize(frame, (0, 0), fx=self.resize_scale, fy=self.resize_scale)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        h, w = gray.shape
        gh, gw = self.grid_size
        grid_h = h // gh
        grid_w = w // gw

        # 最初のフレームは中央
        if self.prev_gray is None:
            self.prev_gray = gray
            center_x = w // 2
            center_y = h // 2
            # 元画像座標に戻す
            orig_x = int(center_x / self.resize_scale)
            orig_y = int(center_y / self.resize_scale)
            self.last_gaze = (orig_x, orig_y)
            return self.last_gaze

        # flow_intervalごとにのみオプティカルフロー計算
        if self.frame_count % self.flow_interval == 0:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None,
                pyr_scale=0.5, levels=2, winsize=9,
                iterations=2, poly_n=5, poly_sigma=1.1, flags=0
            )

            min_motion = None
            min_idx = (0, 0)

            for i in range(gh):
                for j in range(gw):
                    y0 = i * grid_h
                    x0 = j * grid_w
                    y1 = y0 + grid_h
                    x1 = x0 + grid_w
                    grid_flow = flow[y0:y1, x0:x1]
                    motion = np.mean(np.linalg.norm(grid_flow, axis=2))
                    if (min_motion is None) or (motion < min_motion):
                        min_motion = motion
                        min_idx = (i, j)

            # 最小移動グリッドの中心座標（縮小画像上）
            center_x = min_idx[1] * grid_w + grid_w // 2
            center_y = min_idx[0] * grid_h + grid_h // 2

            # 元画像座標に変換
            orig_x = int(center_x / self.resize_scale)
            orig_y = int(center_y / self.resize_scale)
            self.last_gaze = (orig_x, orig_y)

            self.prev_gray = gray.copy()
        # 計算しないフレームは前回の結果を返す

        return self.last_gaze

    def draw(self, frame):
        """
        グリッド線をframeに描画する
        """
        h, w = frame.shape[:2]
        gh, gw = self.grid_size
        # 横線
        for i in range(1, gh):
            y = int(h * i / gh)
            cv2.line(frame, (0, y), (w, y), (255, 255, 0), 1)
        # 縦線
        for j in range(1, gw):
            x = int(w * j / gw)
            cv2.line(frame, (x, 0), (x, h), (255, 255, 0), 1)