# Irish Sign Language Recognition

A real-time Irish Sign Language (ISL) recognition application built with Python, PyTorch and MediaPipe. It extracts hand, body and facial landmarks from video and classifies isolated signs with a GRU sequence model, showing the predicted sign and its confidence in a live webcam demo.

This is my final-year project for the BSc in Computer Science at ATU Donegal. It is a work in progress: the current vocabulary is three signs (HELLO, HOW ARE YOU, SORRY), and the aim is to extend it with more signs and more signers.

<img width="426" height="240" alt="live_demo" src="https://github.com/user-attachments/assets/fa5a1b76-686c-4ffc-a398-58b34281f07a" />

## How it works

```mermaid
flowchart LR
  A[Video or webcam] --> B[MediaPipe landmarkers<br/>hands, pose, face]
  B --> C[Per-frame features<br/>191 values]
  C --> D[Sign segmentation<br/>and gap filling]
  D --> E[Resample to<br/>30 frames]
  E --> F[GRU classifier]
  F --> G[Sign + confidence]
```

1. **Landmarks.** MediaPipe HandLandmarker (two hands), PoseLandmarker and FaceLandmarker (with blendshapes) run on every frame.
2. **Features.** Each frame becomes a 191-value vector (see below). Missing parts are zero-filled and flagged, and short hand dropouts (up to 3 frames) are filled by linear interpolation.
3. **Segmentation.** Runs of frames where a hand is visible are found, nearby runs are merged, very short runs are dropped, and a few frames of padding are added. The same logic runs offline on recorded videos and online in the live demo.
4. **Sequence.** Each segment is resampled to a fixed 30 frames.
5. **Classification.** A GRU (hidden size 64, about 49.5k parameters) predicts the sign. Features are standardised with training-set statistics and clipped to ±5.

## Features (191 per frame)

| Columns | Content |
| --- | --- |
| 0–64 | Right hand: 63 shape values (21 points relative to the wrist, scaled by hand size) + 2 values for wrist position relative to the body |
| 65–129 | Left hand, same layout |
| 130–135 | Body: nose and both elbows (x, y), relative to the shoulder centre and scaled by shoulder width |
| 136–137 | Visibility flags for the right and left hand |
| 138–190 | Face: 52 MediaPipe blendshapes + 1 visibility flag |

Normalising by wrist position, hand size and shoulder width makes the features largely independent of where the signer stands and how far they are from the camera.

For training, features can be selected in named groups (`hands`, `hands_body`, `hands_body_mouth`, `all`, and eye variants). The default model uses **hands + body + eyebrows + mouth/jaw**.

## Evaluation approach

- Models are evaluated on a **held-out recording session**, never on a random split of repetitions, so the test data is not near-duplicates of the training data.
- A separate webcam recording (different camera from the training videos) is used as a cross-device check.
- Feature-group ablations showed that the hands carry most of the signal. Body-only and face-only features are weaker on their own. Eye and gaze features were not robust across cameras and are currently disabled.
- Cross-camera robustness required clipping standardised features (rare blendshapes with tiny training variance otherwise explode on a new camera) and regularisation (dropout, weight decay, and random dropping of the face features during training).

## Project structure

```
src/
  hand_slots.py       assigns detected hands to right/left slots
  video_utils.py      MediaPipe detectors, per-frame extraction, segmentation helpers
  features.py         landmarks -> 191-value feature vectors, gap filling, resampling
  make_segments.py    finds sign segments in raw videos -> data/meta/segments.csv
  build_dataset.py    segments -> fixed-length sequences (.npy)
  dataset.py          loads sequences, labels and session ids
  model.py            GRU classifier
  train.py            training and held-out evaluation
  demo.py             classify a saved sequence
  demo_video.py       full pipeline on a video file
  live_demo.py        real-time webcam demo
scratch/              exploration, checks and experiment scripts
models/               MediaPipe .task files and trained weights (not tracked)
data/                 recordings and extracted features (not tracked)
```

## Getting started

### Requirements

- Python 3.10+
- `mediapipe`, `opencv-python`, `numpy`, `torch`, `scikit-learn`

```bash
pip install -r requirements.txt
```

Download the MediaPipe Hand, Pose and Face Landmarker `.task` model files and place them in `models/`.

### Data layout

```
data/raw/videos/<sign>/P<id>_<label>_S<session>.mp4
```

Recordings are not included in this repository, since they contain participants' faces.

### Run

```bash
# 1. Find sign segments in the raw videos
python src/make_segments.py

# 2. Extract features and build fixed-length sequences
python src/build_dataset.py

# 3. Train (the last session is held out for testing)
python src/train.py hands_body_mouth

# 4. Train on all sessions for the final model
python src/train.py hands_body_mouth final

# 5. Classify a video file
python src/demo_video.py path/to/video.mp4

# 6. Live webcam demo (press "d" to toggle landmark drawing)
python src/live_demo.py
```

The live demo works best with good, even lighting. In low light the hand detector can lose the hand, especially a closed fist.

## Current status and limitations

- Three signs, recorded by a single signer so far. Results on a larger vocabulary and on new signers will be lower, and measuring that honestly is part of the project.
- The signs have not yet been validated by ISL users. User testing will be done with ethics approval from ATU.
- Isolated signs only, not continuous signing or sentences.
- The model currently relies on hands, body position and parts of the face. Head movement (e.g. tilt) is not used yet.

## Roadmap

- [ ] Record more signs with more signers (home recordings, with ethics approval)
- [ ] Evaluate on signers not seen in training
- [ ] Add head-movement features
- [ ] Compare the GRU with a 1D CNN and the logistic-regression baseline on the larger dataset
- [ ] Reduce run-to-run variance (fixed seeds, model ensembles)
- [ ] Test with ISL users

## Author

Ruslan Fefelov - BSc Computer Science, ATU Donegal
