import cv2
import numpy as np
import os
import sys
import argparse
from algorithms.center import CenterGazeEstimator
from algorithms.opticalFlow_less_moving_places import OpticalFlowLessMovingPlacesEstimator
# 追加: 他のアルゴリズムが必要な場合はここにインポート
from evaluation import GazeEvaluator
from algorithms.cnn_singleframe.cnn_singleframe import CnnSingleFrameEstimator
from algorithms.cnn_gru.cnn_gru import CnnGruHybridEstimator

# === キャリブレーション（画面サイズに基づく正規化用）
CALIBRATION_WIDTH = 1280
CALIBRATION_HEIGHT = 960

# === 描画設定 ===
CIRCLE_RADIUS = 10
CIRCLE_COLOR = (0, 0, 255)  # 赤
CIRCLE_THICKNESS = 2

ALGORITHM_DICT = {
    "center": CenterGazeEstimator,
    "less_moving_places": OpticalFlowLessMovingPlacesEstimator,
    "cnn_single":  CnnSingleFrameEstimator,
    "cnn_gru": CnnGruHybridEstimator,  # 追加: CNN-GRUアルゴリズムを使用する場合はコメントアウトを外す
    # "other": OtherGazeEstimator,
}

def load_begaze_gaze_data(path):
    gaze_dict = {}

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue

            parts = line.strip().split()
            if len(parts) < 6:
                continue

            try:
                frame_id = int(parts[5])
                x = float(parts[3]) / CALIBRATION_WIDTH
                y = float(parts[4]) / CALIBRATION_HEIGHT

                # 同じフレームが複数回出現する場合は平均化する
                if frame_id not in gaze_dict:
                    gaze_dict[frame_id] = []
                gaze_dict[frame_id].append((x, y))
            except ValueError:
                continue

    # フレームごとに平均をとる（視線が複数あれば）
    gaze_avg = {}
    for fid, points in gaze_dict.items():
        xs, ys = zip(*points)
        gaze_avg[fid] = (sum(xs) / len(xs), sum(ys) / len(ys))

    return gaze_avg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", help="データセット名（拡張子不要）")
    parser.add_argument("--algorithm", default="center", choices=ALGORITHM_DICT.keys(), help="使用するアルゴリズム")
    parser.add_argument("--no-eval", action="store_true", help="評価を無効化する")
    args = parser.parse_args()

    dataset_name = os.path.splitext(args.dataset_name)[0]

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    VIDEO_PATH = os.path.join(BASE_DIR, "data", f"{dataset_name}.avi")
    GAZE_PATH = os.path.join(BASE_DIR, "data", f"{dataset_name}.txt")
    OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", f"annotated_{dataset_name}.avi")

    gaze_data = load_begaze_gaze_data(GAZE_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("動画を開けませんでした")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    # アルゴリズムのインスタンス化
    estimator = ALGORITHM_DICT[args.algorithm]()

    # 評価機構のインスタンス化（デフォルト有効、--no-evalで無効）
    evaluator = GazeEvaluator() if not args.no_eval else None

    frame_idx = 1  # BeGazeのフレーム番号は1始まり

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 推定アルゴリズムの利用
        pred_x, pred_y = estimator.estimate_gaze(frame)
        estimator.draw(frame)
        cv2.circle(frame, (int(pred_x), int(pred_y)), CIRCLE_RADIUS, (0,255,0), CIRCLE_THICKNESS)  # 予測視線（緑）

        if frame_idx in gaze_data:
            gx, gy = gaze_data[frame_idx]
            cx = int(gx * width)
            cy = int(gy * height)
            cv2.circle(frame, (cx, cy), CIRCLE_RADIUS, CIRCLE_COLOR, CIRCLE_THICKNESS)  # 正解視線（赤）

            # 評価用データ追加（正規化座標→ピクセル座標で比較）
            if evaluator is not None:
                evaluator.add((pred_x / width, pred_y / height), (gx, gy))

        cv2.imshow("Gaze Overlay", frame)
        out.write(frame)

        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"出力完了：{OUTPUT_PATH}")

    # 評価結果の表示
    if evaluator is not None:
        mean_dist = evaluator.mean_distance()
        if mean_dist is not None:
            print(f"平均ユークリッド距離（正規化座標）: {mean_dist:.4f}")
        else:
            print("評価データがありませんでした。")

if __name__ == "__main__":
    main()
