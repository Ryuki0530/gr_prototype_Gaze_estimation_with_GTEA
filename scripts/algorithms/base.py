from abc import ABC, abstractmethod

class GazeEstimator(ABC):
    
    def set_feedback(self, x, y):
        """
        視線座標のフィードバックを設定するメソッド
        （必要な場合のみ継承先でオーバーライド）
        """
        pass

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
    
    def draw(self, frame):
        """
        処理可視化用の描画メソッド
        Args:
            frame (numpy.ndarray): 入力画像フレーム
        """
        pass