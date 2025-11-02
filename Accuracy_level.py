import cv2
import numpy as np
import os
import argparse
from ultralytics import YOLO

VIDEO_FOLDER = "videos"
OUTPUT_FOLDER = "out"
MODEL_PATH = "yolov8n.pt"
CONF_THRESHOLD = 0.5


def draw_line(frame, p1, p2, color, label):
    cv2.line(frame, p1, p2, color, 3)
    cv2.putText(frame, label, (p1[0], p1[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def put_text(frame, text, pos, color=(255, 255, 255)):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


class CentroidTracker:

    def __init__(self, max_disappeared=40):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        if not rects:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects

        centroids = np.array(
            [[(x1 + x2) // 2, (y1 + y2) // 2] for x1, y1, x2, y2 in rects]
        )

        if not self.objects:
            for c in centroids:
                self.register(c)
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = np.array(list(self.objects.values()))
        distances = np.linalg.norm(object_centroids[:, None] - centroids, axis=2)

        rows = distances.min(axis=1).argsort()
        cols = distances.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()
        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols or distances[row, col] > 50:
                continue
            obj_id = object_ids[row]
            self.objects[obj_id] = centroids[col]
            self.disappeared[obj_id] = 0
            used_rows.add(row)
            used_cols.add(col)

        for row in set(range(distances.shape[0])) - used_rows:
            obj_id = object_ids[row]
            self.disappeared[obj_id] += 1
            if self.disappeared[obj_id] > self.max_disappeared:
                self.deregister(obj_id)

        for col in set(range(distances.shape[1])) - used_cols:
            self.register(centroids[col])

        return self.objects


def process_video(video_path):
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {video_path}")
        return

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    line_gap = int(height * 0.1)
    entry_line = [(int(width * 0.2), int(height * 0.5) - line_gap),
                  (int(width * 0.8), int(height * 0.5) - line_gap)]
    exit_line = [(int(width * 0.2), int(height * 0.5) + line_gap),
                 (int(width * 0.8), int(height * 0.5) + line_gap)]

    tracker = CentroidTracker()
    tracks, colors = {}, {}
    count_in = count_out = 0
    frame_confidences = []

    out_path = os.path.join(OUTPUT_FOLDER, os.path.basename(video_path))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    print(f"[INFO] Processing {video_path} ...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = model(frame, verbose=False)
        rects = []
        confidences = []

        for result in detections:
            for box in result.boxes:
                conf = float(box.conf)
                if conf < CONF_THRESHOLD or model.names[int(box.cls)] != "person":
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                rects.append((x1, y1, x2, y2))
                confidences.append(conf)

        if confidences:
            avg_conf = sum(confidences) / len(confidences)
            frame_confidences.append(avg_conf)
        else:
            avg_conf = 0.0

        objects = tracker.update(rects)

        for obj_id, centroid in objects.items():
            tracks.setdefault(obj_id, []).append(centroid)
            track = tracks[obj_id][-30:]
            color = colors.get(obj_id, (255, 255, 255))

            if len(track) > 2:
                cy, prev_y = centroid[1], track[-2][1]
                if prev_y > entry_line[0][1] and cy <= entry_line[0][1]:
                    count_in += 1
                    color = (0, 255, 0)
                elif prev_y < exit_line[0][1] and cy >= exit_line[0][1]:
                    count_out += 1
                    color = (0, 0, 255)
                colors[obj_id] = color

            for x1, y1, x2, y2 in rects:
                if abs((x1 + x2) // 2 - centroid[0]) < 5:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    conf_text = f"{avg_conf*100:.1f}%"
                    cv2.putText(frame, conf_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    break

            cv2.circle(frame, tuple(centroid), 4, color, -1)
            put_text(frame, f"ID {obj_id}", (centroid[0] - 10, centroid[1] - 10), color)

        draw_line(frame, *entry_line, (0, 255, 255), "ENTRY LINE")
        draw_line(frame, *exit_line, (0, 255, 255), "EXIT LINE")

        put_text(frame, f"IN: {count_in}", (10, 40), (0, 255, 0))
        put_text(frame, f"OUT: {count_out}", (10, 80), (0, 0, 255))
        put_text(frame, f"Avg Accuracy: {avg_conf*100:.1f}%", (10, 120), (255, 255, 0))

        writer.write(frame)
        cv2.imshow("Footfall Counter", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    avg_model_conf = np.mean(frame_confidences) * 100 if frame_confidences else 0
    results_file = os.path.join(OUTPUT_FOLDER, "results.txt")
    with open(results_file, "a") as f:
        f.write(f"{os.path.basename(video_path)} - IN: {count_in}, OUT: {count_out}, Avg Accuracy: {avg_model_conf:.2f}%\n")

    print(f"[INFO] Done. Saved to {out_path}")
    print(f"[RESULT] Entered: {count_in}, Exited: {count_out}, Avg Detection Accuracy: {avg_model_conf:.2f}%")
    print(f"[INFO] Results saved to {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="videos/Football.mp4",
                        help="Path to input video file (default: videos/Football.mp4)")
    args = parser.parse_args()

    os.makedirs(VIDEO_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    process_video(args.source)
