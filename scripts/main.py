import cv2
import numpy as np
import os
import sys

# === キャリブレーション（画面サイズに基づく正規化用）
CALIBRATION_WIDTH = 1280
CALIBRATION_HEIGHT = 960

# === 描画設定 ===
CIRCLE_RADIUS = 10
CIRCLE_COLOR = (0, 0, 255)  # 赤
CIRCLE_THICKNESS = 2

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
    if len(sys.argv) < 2:
        print("Usage:",sys.argv[0]," <dataset_name>")
        sys.exit(1)

    dataset_name = sys.argv[1]
    # 拡張子が含まれていたら除去
    dataset_name = os.path.splitext(dataset_name)[0]

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    VIDEO_PATH = os.path.join(BASE_DIR, "data", f"{dataset_name}.avi")
    GAZE_PATH = os.path.join(BASE_DIR, "data", f"{dataset_name}.txt")
    OUTPUT_PATH = os.path.join(BASE_DIR, "output", f"annotated_{dataset_name}.avi")

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

    frame_idx = 1  # BeGazeのフレーム番号は1始まり

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in gaze_data:
            gx, gy = gaze_data[frame_idx]
            cx = int(gx * width)
            cy = int(gy * height)
            cv2.circle(frame, (cx, cy), CIRCLE_RADIUS, CIRCLE_COLOR, CIRCLE_THICKNESS)

        cv2.imshow("Gaze Overlay", frame)
        out.write(frame)

        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"出力完了：{OUTPUT_PATH}")

if __name__ == "__main__":
    main()
