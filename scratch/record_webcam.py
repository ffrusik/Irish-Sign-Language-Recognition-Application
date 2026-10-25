import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30
print("webcam size:", w, h, "fps:", fps)

writer = cv2.VideoWriter("data/raw/webcam_test.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
while True:
  ok, frame = cap.read()
  if not ok:
    break
  writer.write(frame)                     # raw frame, not mirrored
  cv2.imshow("rec", cv2.flip(frame, 1))   # mirrored only for display
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

cap.release()
writer.release()
cv2.destroyAllWindows()