import numpy as np
import cv2
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from base import GazeEstimator

class WeightedFlowSmoothGazeEstimator(GazeEstimator):
    RESIZE_SCALE = 0.25
    FLOW_INTERVAL = 2
    CENTER_DAMPING = 0.12
    EMA_ALPHA = 0.6  # 平滑化を弱めて追従性向上
    FLOW_EXP = 1.5   # フロー強調の指数

    def __init__(self, resize_scale=None, flow_interval=None):
        self.prev_gray = None
        self.gaze_point = None
        self.raw_point = None
        self.resize_scale = resize_scale or self.RESIZE_SCALE
        self.flow_interval = flow_interval or self.FLOW_INTERVAL
        self.frame_count = 0
        self.debug_flow = None

    def estimate_gaze(self, frame):
        self.frame_count += 1

        small_frame = cv2.resize(frame, (0, 0), fx=self.resize_scale, fy=self.resize_scale)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        center = np.array([w // 2, h * 2 // 3], dtype=np.float32)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.gaze_point = center.copy()
            self.raw_point = center.copy()
            return self._to_orig_coords(self.gaze_point)

        if self.frame_count % self.flow_interval == 0:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None,
                pyr_scale=0.5,
                levels=3,         # 強化: 層を深くして大きな動きを検出
                winsize=15,       # 強化: ウィンドウサイズ拡大
                iterations=3,     # 強化: 反復回数増加
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )

            # --- ガウス重み付き平均 ---
            Y, X = np.mgrid[0:h, 0:w]
            sigma = min(h, w) * 0.3
            weight = np.exp(-(((X - center[0]) ** 2 + (Y - center[1]) ** 2) / (2 * sigma ** 2)))
            flow_x = (flow[..., 0] * weight).sum() / weight.sum()
            flow_y = (flow[..., 1] * weight).sum() / weight.sum()
            move = np.array([flow_x, flow_y], dtype=np.float32)

            # --- フロー強調（指数的スケーリング） ---
            move_norm = np.linalg.norm(move)
            if move_norm > 0:
                move_dir = move / move_norm
                move = move_dir * (move_norm ** self.FLOW_EXP)
            else:
                move = np.zeros_like(move)

            # --- 未平滑位置更新 ---
            self.raw_point -= move
            self.raw_point += (center - self.raw_point) * self.CENTER_DAMPING

            # --- EMAで平滑化更新 ---
            self.gaze_point = (
                self.EMA_ALPHA * self.gaze_point + (1 - self.EMA_ALPHA) * self.raw_point
            )

            self.gaze_point = np.clip(self.gaze_point, [0, 0], [w - 1, h - 1])
            self.prev_gray = gray.copy()
            self.debug_flow = flow

        return self._to_orig_coords(self.gaze_point)

    def draw(self, frame):
        if self.debug_flow is not None:
            h, w = self.debug_flow.shape[:2]
            step = 16
            for y in range(0, h, step):
                for x in range(0, w, step):
                    fx, fy = self.debug_flow[y, x]
                    x0 = int(x / self.resize_scale)
                    y0 = int(y / self.resize_scale)
                    x1 = int((x + fx * 3) / self.resize_scale)
                    y1 = int((y + fy * 3) / self.resize_scale)
                    cv2.arrowedLine(frame, (x0, y0), (x1, y1), (255, 255, 0), 1, tipLength=0.3)

    def _to_orig_coords(self, pt):
        return int(pt[0] / self.resize_scale), int(pt[1] / self.resize_scale)
