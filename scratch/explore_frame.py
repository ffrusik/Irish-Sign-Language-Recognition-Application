import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

options = vision.HandLandmarkerOptions(
  base_options=mp_python.BaseOptions(model_asset_path="models/hand_landmarker.task"),
  running_mode=vision.RunningMode.IMAGE,   # one single image, no timestamps needed
  num_hands=2,
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture("data/raw/videos/hello/P00_Hello_S1.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 40)       # jump to frame 40
ok, frame = cap.read()
cap.release()

rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
result = detector.detect(image)

print("hands found:", len(result.hand_landmarks))
for hand, handed in zip(result.hand_landmarks, result.handedness):
  print(handed[0].category_name, handed[0].score, len(hand))
  print(hand[0])   # the wrist: x, y, z