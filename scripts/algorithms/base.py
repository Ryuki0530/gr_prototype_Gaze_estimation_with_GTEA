from abc import ABC, abstractmethod

class GazeEstimator(ABC):
    @abstractmethod
    def estimate_gaze(self, frame):
        """
        1フレームの画像データから視線座標を推定するメソッド

        Args:
            frame (numpy.ndarray): 入力画像フレーム

        Returns:
            tuple: (x, y) 視線座標（正規化座標またはピクセル座標）
        """
        pass