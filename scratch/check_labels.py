import sys
sys.path.insert(0, "src")

import csv
from collections import Counter
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from hand_slots import hands_to_slots

options = vision.HandLandmarkerOptions(
  base_options=mp_python.BaseOptions(model_asset_path="models/hand_landmarker.task"),
  running_mode=vision.RunningMode.VIDEO,
  num_hands=2,
)
detector = vision.HandLandmarker.create_from_options(options)

# take the first repetition from the table
with open("data/meta/segments.csv") as f:
  rows = list(csv.DictReader(f))

sign_name = "how_are_you"  # sign to explore
row = next(r for r in rows if r["sign"] == sign_name)
start, end = int(row["start"]), int(row["end"])

cap = cv2.VideoCapture(row["video"])
fps = cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_POS_FRAMES, start)

labels = Counter()          # how many times each label appeared
hands_per_frame = Counter() # how many frames had 0, 1 or 2 hands

for frame_index in range(start, end + 1):
  ok, frame = cap.read()
  if not ok:
    break
  rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
  timestamp_ms = int(frame_index * 1000 / fps)
  result = detector.detect_for_video(image, timestamp_ms)

  hands_per_frame[len(result.hand_landmarks)] += 1
  for handed in result.handedness:
    labels[handed[0].category_name] += 1

  hands_per_frame[len(result.hand_landmarks)] += 1
  for handed in result.handedness:
    labels[handed[0].category_name] += 1

  slots = hands_to_slots(result)
  print(frame_index, slots["Right"] is not None, slots["Left"] is not None)

cap.release()
print(row["video"], "rep", row["rep"])
print("labels:", labels)
print("hands per frame:", hands_per_frame)