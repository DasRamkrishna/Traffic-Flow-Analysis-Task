import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import os

INPUT_VIDEO = "traffic.mp4"   # my local video file
OUTPUT_VIDEO = "output/demo_output.mp4"
CSV_FILE = "output/counts.csv"

os.makedirs("output", exist_ok=True)

# VEHICLE CLASSES (COCO IDs)
VEHICLE_CLASSES = [2, 3, 5, 7]

# INIT MODELS
model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=30)

vehicle_data = []
lane_counts = {1: 0, 2: 0, 3: 0}
seen = set()

# VIDEO CAPTURE
cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS) or 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_num = 0

# VideoWriter for saving output
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

if not out.isOpened():
    print("Error: Could not open VideoWriter.")
    exit()

# DEFINE LANES
lane1_x = width // 3
lane2_x = 2 * width // 3

# Horizontal counting line (near bottom of screen)
count_line_y = int(height * 0.85)

# PROCESSING LOOP
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

        # Only count when crossing horizontal line
        if cy > count_line_y:
            # Determine lane by x-position
            if cx < lane1_x:
                lane = 1
            elif cx < lane2_x:
                lane = 2
            else:
                lane = 3

            key = (track_id, lane)
            if key not in seen:
                lane_counts[lane] += 1
                seen.add(key)
                vehicle_data.append([track_id, lane, frame_num, timestamp])

        # Draw box + ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # DRAW LANE LINES
    # Vertical lane dividers
    cv2.line(frame, (lane1_x, 0), (lane1_x, height), (0, 0, 255), 2)
    cv2.line(frame, (lane2_x, 0), (lane2_x, height), (0, 0, 255), 2)

    # Horizontal counting line
    cv2.line(frame, (0, count_line_y), (width, count_line_y), (255, 0, 0), 2)

    # Lane counts display
    for i in range(1, 4):
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

#EXPORT CSV
df = pd.DataFrame(vehicle_data, columns=["VehicleID", "Lane", "Frame", "Timestamp"])
df.to_csv(CSV_FILE, index=False)

print("Final Counts:", lane_counts)
print(f"Video saved to {OUTPUT_VIDEO}")
print(f"CSV saved to {CSV_FILE}")
