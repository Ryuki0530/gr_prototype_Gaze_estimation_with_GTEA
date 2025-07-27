import numpy as np

class GazeEvaluator:
    """
    視線推定アルゴリズムの評価を行うクラス
    - 推定値と正解値を蓄積し、平均ユークリッド距離などを計算
    - 将来的なグラフ化やリアルタイム表示にも拡張しやすい設計
    """
    def __init__(self):
        self.pred_list = []
        self.gt_list = []
        self.dist_list = []
        self.squared_dist_list = []

    def add(self, pred, gt):
        """
        推定値と正解値を1組追加し、距離も記録
        Args:
            pred: (x, y) 推定座標
            gt: (x, y) 正解座標
        """
        self.pred_list.append(pred)
        self.gt_list.append(gt)
        self.dist_list.append(self.euclidean_distance(pred, gt))
        self.squared_dist_list.append(self.euclidean_distance(pred, gt) ** 2)
        return self.dist_list[-1], self.squared_dist_list[-1]

    @staticmethod
    def euclidean_distance(p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def mean_distance(self):
        """平均ユークリッド距離を返す"""
        if not self.dist_list:
            return None
        return float(np.mean(self.dist_list))

    def mean_squared_distance(self):
        """平均ユークリッド距離の二乗を返す"""
        if not self.squared_dist_list:
            return None
        return float(np.mean(self.squared_dist_list))

    def all_distances(self):
        """全フレームのユークリッド距離リストを返す"""
        return self.dist_list

    def all_squared_distances(self):
        """全フレームのユークリッド距離の二乗リストを返す"""
        return self.squared_dist_list

    def print_eval_results(self):
        """評価結果をコンソールに出力"""
        print("\n=== 評価結果 ===")
        mean_dist = self.mean_distance()
        if mean_dist is not None:
            print(f"ユークリッド距離（正規化座標）の平均: {mean_dist:.4f}")
        else:
            print("ユークリッド距離（正規化座標）の平均の評価データがありませんでした。")

        mean_squared_dist = self.mean_squared_distance()
        if mean_squared_dist is not None:
            print(f"ユークリッド距離（正規化座標）の二乗の平均: {mean_squared_dist:.4f}")
        else:
            print("ユークリッド距離（正規化座標）の二乗の平均の評価データがありませんでした。")

    def clear(self):
        """内部データをリセット"""
        self.pred_list.clear()
        self.gt_list.clear()
        self.dist_list.clear()
