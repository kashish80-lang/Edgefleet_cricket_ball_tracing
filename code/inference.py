import cv2
import numpy as np
import pandas as pd
import os

VIDEO_PATH = r"C:\Users\HP\Desktop\cricket_ball_tracking\data\test_video"
OUT_VIDEO = r"C:\Users\HP\Desktop\cricket_ball_tracking\results\ball_tracked_final.mp4"
OUT_CSV = r"C:\Users\HP\Desktop\cricket_ball_tracking\results\ball_positions_final.csv"

os.makedirs("results", exist_ok=True)

clicked_points = {}  
current_click_frame = None
trajectory = []

def select_point(event, x, y, flags, param):
    global clicked_points, current_click_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points[current_click_frame] = (x, y)
        print(f"Ball clicked at frame {current_click_frame}: {x},{y}")

cap = cv2.VideoCapture(VIDEO_PATH)
ret, first_frame = cap.read()
if not ret:
    print(" Cannot read video")
    exit()

h, w = first_frame.shape[:2]
fps = cap.get(cv2.CAP_PROP_FPS)
scale = 0.7  
writer = cv2.VideoWriter(
    OUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (int(w*scale), int(h*scale))
)

cv2.namedWindow("Select Ball")
cv2.setMouseCallback("Select Ball", select_point)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
frame_id = 0
frames = []
motion_frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, gray)
    _, diff_thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    motion = cv2.countNonZero(diff_thresh)

   
    if motion > 2000:  
        motion_frames.append(frame_id)

    prev_gray = gray.copy()
    frame_id += 1

cap.release()

print(f"Detected {len(motion_frames)} frames with ball motion.")

for f_id in motion_frames:
    frame = frames[f_id]
    frame_small = cv2.resize(frame, (int(w*scale), int(h*scale)))
    current_click_frame = f_id
    cv2.imshow("Select Ball", frame_small)
    key = cv2.waitKey(0) & 0xFF  
    if key == ord('c'):
        print(f"Click on the ball for frame {f_id}")
        cv2.waitKey(1)
    elif key == ord('s'):  
        print(f"Skipped frame {f_id}")
        continue

cv2.destroyAllWindows()


sorted_frames = sorted(clicked_points.keys())
if len(sorted_frames) < 2:
    print(" You must click at least 2 frames")
    exit()

interpolated_positions = []
total_frames = len(frames)
for i in range(len(sorted_frames)-1):
    f1, f2 = sorted_frames[i], sorted_frames[i+1]
    x1, y1 = clicked_points[f1]
    x2, y2 = clicked_points[f2]
    for f in range(f1, f2):
        t = (f - f1) / (f2 - f1)
        x = int(x1 + (x2 - x1) * t)
        y = int(y1 + (y2 - y1) * t)
        interpolated_positions.append((f, x, y))

last_frame = sorted_frames[-1]
x_last, y_last = clicked_points[last_frame]
for f in range(last_frame, total_frames):
    interpolated_positions.append((f, x_last, y_last))

trajectory = []
data = []

for f_id, x, y in interpolated_positions:
    frame = frames[f_id]
    frame_small = cv2.resize(frame, (int(w*scale), int(h*scale)))
    trajectory.append((x, y))
    
    for i in range(1, len(trajectory)):
        cv2.line(frame_small, trajectory[i-1], trajectory[i], (0,0,255), 2)
    
    cv2.circle(frame_small, (x, y), 6, (0,255,0), -1)
    writer.write(frame_small)
    data.append([f_id, x, y, 1])

writer.release()

df = pd.DataFrame(data, columns=["frame", "x", "y", "visible"])
df.to_csv(OUT_CSV, index=False)

print("Tracking finished!")
print("Video saved:", OUT_VIDEO)
print("CSV saved:", OUT_CSV)
