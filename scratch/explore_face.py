import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# detector for single images, with blendshapes switched on
options = vision.FaceLandmarkerOptions(
  base_options=mp_python.BaseOptions(model_asset_path="models/face_landmarker.task"),
  running_mode=vision.RunningMode.IMAGE,
  output_face_blendshapes=True,
  num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# take one frame from a video
cap = cv2.VideoCapture("data/raw/videos/how_are_you/P00_HowAreYou_S1.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
ok, frame = cap.read()
cap.release()

rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
result = detector.detect(image)

print(len(result.face_blendshapes))  # number of faces found (expected: 1)
for item in result.face_blendshapes[0]:
  print(item.category_name, round(item.score, 2))
detector.close()