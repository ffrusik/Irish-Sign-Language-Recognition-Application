import sys
sys.path.insert(0, "src")

import time
import cv2
from video_utils import FrameProcessor

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
processor = FrameProcessor()
start = time.time()
last_ts = -1

while True:
  ok, frame = cap.read()
  if not ok:
    break
  timestamp_ms = max(int((time.time() - start) * 1000), last_ts + 1)  # must grow strictly
  last_ts = timestamp_ms
  slots, pose, face = processor.process(frame, timestamp_ms)  # raw (not mirrored) frame
  h, w = frame.shape[:2]
  for hand in slots.values():
    if hand is not None:
      for x, y, _ in hand:
        cv2.circle(frame, (int(x * w), int(y * h)), 4, (0, 255, 0), -1)

  # text = f"R:{slots['Right'] is not None} L:{slots['Left'] is not None} body:{pose is not None} face:{face is not None}"
  text = f"R:{slots['Right'] is not None} L:{slots['Left'] is not None} {w}x{h}"
  shown = cv2.flip(frame, 1)  # mirror only the picture we show, like a mirror
  cv2.putText(shown, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
  cv2.imshow("live", shown)
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

cap.release()
processor.close()
cv2.destroyAllWindows()