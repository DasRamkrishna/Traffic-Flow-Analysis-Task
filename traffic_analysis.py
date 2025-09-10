import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import os

# ---------- CONFIG ----------
INPUT_VIDEO = "traffic.mp4"
OUTPUT_VIDEO = "output/demo_output.mp4"
CSV_FILE = "output/counts.csv"

# Create output folder if not exists
os.makedirs("output", exist_ok=True)

# Define 3 lane boundaries (x1,y1,x2,y2 style lines)
lanes = [
    [(100, 720), (600, 300)],   # Lane 1
    [(600, 720), (1100, 300)],  # Lane 2
    [(1100, 720), (1600, 300)]  # Lane 3
]

# ---------- VEHICLE CLASSES (COCO IDs) ----------
VEHICLE_CLASSES = [2, 3, 5, 7]  # Car, Motorcycle, Bus, Truck

# ---------- INIT MODELS ----------
model = YOLO("yolov8n.pt")  # lightweight YOLOv8
tracker = DeepSort(max_age=30)

vehicle_data = []
lane_counts = {1: 0, 2: 0, 3: 0}
seen = set()

# ---------- VIDEO CAPTURE ----------
cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps is None:
    fps = 30  # fallback if FPS not detected

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_num = 0

# VideoWriter for saving output
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

if not out.isOpened():
    print("Error: Could not open VideoWriter.")
    exit()

# ---------- PROCESSING LOOP ----------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1
    timestamp = frame_num / fps

    # Detect vehicles
    results = model(frame, stream=True)

    detections = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls in VEHICLE_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(([x1, y1, x2 - x1, y2 - y1], box.conf[0], cls))

    # Track vehicles
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Check lane assignment
        for i, (p1, p2) in enumerate(lanes, start=1):
            dist = np.cross(np.array(p2) - np.array(p1),
                            np.array([cx, cy]) - np.array(p1)) / np.linalg.norm(np.array(p2) - np.array(p1))
            if abs(dist) < 30:  # threshold distance to lane line
                key = (track_id, i)
                if key not in seen:
                    lane_counts[i] += 1
                    seen.add(key)
                    vehicle_data.append([track_id, i, frame_num, timestamp])

        # Draw box + ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw lane lines + counts
    for i, (p1, p2) in enumerate(lanes, start=1):
        cv2.line(frame, tuple(p1), tuple(p2), (0, 0, 255), 3)
        cv2.putText(frame, f"Lane {i}: {lane_counts[i]}", (50, 50 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show video
    cv2.imshow("Traffic Analysis", frame)

    # Write frame to output video
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# ---------- EXPORT CSV ----------
df = pd.DataFrame(vehicle_data, columns=["VehicleID", "Lane", "Frame", "Timestamp"])
df.to_csv(CSV_FILE, index=False)

print("Final Counts:", lane_counts)
print(f"Video saved to {OUTPUT_VIDEO}")
print(f"CSV saved to {CSV_FILE}")
