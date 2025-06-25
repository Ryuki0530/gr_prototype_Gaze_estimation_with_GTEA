import numpy as np
import cv2
from .base import GazeEstimator

class OpticalFlowMovingRefrectsCenter(GazeEstimator):
    # --- 定数宣言 ---
    RESIZE_SCALE = 0.25  # 画像縮小率
    FLOW_INTERVAL = 2    # 何フレームごとに計算するか
    MOVE_GAIN = 1.0      # 光フロー移動量のスケール
    MOVE_EXP = 6.0       # 光フロー移動量にかける指数
    CENTER_DAMPING = 0.06  # 中央に戻る減衰係数

    def __init__(self, resize_scale=None, flow_interval=None):
        self.prev_gray = None
        self.gaze_point = None  # 現在の視線座標
        self.resize_scale = resize_scale if resize_scale is not None else self.RESIZE_SCALE
        self.flow_interval = flow_interval if flow_interval is not None else self.FLOW_INTERVAL
        self.frame_count = 0

    def estimate_gaze(self, frame):
        """
        光フローを用いてカメラの動きから視線位置を推定する

        Args:
            frame (numpy.ndarray): 入力画像フレーム

        Returns:
            tuple: (x, y) 視線のピクセル座標
        """
        self.frame_count += 1

        # さらに小さくリサイズ
        small_frame = cv2.resize(frame, (0, 0), fx=self.resize_scale, fy=self.resize_scale)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        center = np.array([w // 2, (h // 3)*2], dtype=np.float32)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.gaze_point = center.copy()
            orig_x = int(self.gaze_point[0] / self.resize_scale)
            orig_y = int(self.gaze_point[1] / self.resize_scale)
            return (orig_x, orig_y)

        if self.frame_count % self.flow_interval == 0:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None,
                pyr_scale=0.5, levels=1, winsize=5,
                iterations=1, poly_n=3, poly_sigma=1.1, flags=0
            )
            mean_flow = flow.mean(axis=(0, 1))
            dx, dy = mean_flow

            # 移動量を指数的に増加させる
            move = np.array([dx, dy], dtype=np.float32)
            move_norm = np.linalg.norm(move)
            if move_norm > 0:
                move_dir = move / move_norm
                move_exp = (move_norm ** self.MOVE_EXP) * self.MOVE_GAIN
                move = move_dir * move_exp
            else:
                move = np.zeros_like(move)

            self.gaze_point -= move
            self.gaze_point += (center - self.gaze_point) * self.CENTER_DAMPING
            self.gaze_point[0] = np.clip(self.gaze_point[0], 0, w - 1)
            self.gaze_point[1] = np.clip(self.gaze_point[1], 0, h - 1)
            self.prev_gray = gray.copy()

        orig_x = int(self.gaze_point[0] / self.resize_scale)
        orig_y = int(self.gaze_point[1] / self.resize_scale)
        return (orig_x, orig_y)
