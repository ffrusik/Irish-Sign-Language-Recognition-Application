import sys
sys.path.insert(0, "src")

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from features import body_frame

options = vision.PoseLandmarkerOptions(
  base_options=mp_python.BaseOptions(model_asset_path="models/pose_landmarker_lite.task"),
  running_mode=vision.RunningMode.IMAGE,
  num_poses=1,
)
detector = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture("data/raw/videos/how_are_you/P00_HowAreYou_S1.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 40)
ok, frame = cap.read()
cap.release()

rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
result = detector.detect(image)

print("people found:", len(result.pose_landmarks))
if result.pose_landmarks:
  pose = result.pose_landmarks[0]          # first (and only) person: 33 points
  print("nose:", round(pose[0].x, 2), round(pose[0].y, 2))
  print("left shoulder:", round(pose[11].x, 2), round(pose[11].y, 2))
  print("right shoulder:", round(pose[12].x, 2), round(pose[12].y, 2))

print(body_frame(pose))