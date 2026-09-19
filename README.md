# CyberHUD

A real-time cyberpunk-inspired computer vision HUD with local face detection and multi-target tracking using Python and OpenCV.

## Overview

CyberHUD turns a webcam feed into an animated AR-style targeting interface. YuNet detects faces locally, geometric tracking maintains temporary targets, and OpenCV draws a cyan/orange HUD over mirrored video.

**CyberHUD performs face detection and tracking only.** It does not recognize people, identify individuals biometrically, detect emotions, or infer age or gender. The V2 application has been personally tested with a real webcam, including detection, tracking states, animations, numbering and safe shutdown.

## Demo

> Demo coming soon: a real webcam screenshot, GIF, or short video will be added here.

No demo media is included yet. This placeholder does not represent a recorded demonstration.

## Features

- Mirrored live webcam view with multiple face targets.
- Session IDs such as `TARGET-01`, `TARGET-02`, and `TARGET-03`.
- Smoothed tracking with acquisition, lock, loss and reacquisition states.
- Animated brackets, scan line, segmented reticle and lock animation.
- Measured FPS, target count/state, coordinates and local timestamp.
- Latest detector score displayed as `LAST DET`.
- Local CPU inference with a bundled, small YuNet model.
- Camera error handling, bounded dropped-frame retries and safe cleanup.

## How It Works

1. Capture a webcam frame and mirror it horizontally.
2. Resize selected frames to a maximum width of 320 pixels for YuNet detection.
3. Map detected boxes back to the camera frame.
4. Associate boxes using overlap, distance, size checks and bounded motion prediction.
5. Smooth target geometry and update time-based tracking states.
6. Draw the HUD and handle keyboard/window events.

Coordinates use the mirrored frame: origin at the top left, X increases rightward, Y downward. `LAST DET` is the latest actual detector score, not a calibrated probability, identity confidence or fresh measurement on skipped frames.

## Tracking States

**ACQUIRING → LOCKED → TRACKING**

Temporary loss displays **LOST / REACQUIRING**.

| State | Meaning |
| --- | --- |
| ACQUIRING | New or reacquired face; at least two detections and 0.15 seconds are needed for lock. |
| LOCKED | Confirmation animation lasts 0.5 seconds. |
| TRACKING | Normal tracking; misses up to 0.45 seconds do not trigger LOST. |
| LOST / REACQUIRING | More than 0.45 seconds since detection; retain the target for possible reacquisition. |
| Removed | More than 1.6 seconds since detection; remove the indicator. |

IDs start at `TARGET-01` each application session. Matching before removal preserves the ID. A face detected after removal receives the next unused ID. These labels track rectangles, not identities; crossing faces can swap IDs.

The global HUD uses the same retained targets as the face indicators. It shows `TARGET TRACKING // ACTIVE` when targets remain, with state and lost count in the details. ACTIVE includes targets awaiting reacquisition. With no retained targets, it shows `SCANNING / NO TARGET`.

## Architecture

| Module | Responsibility |
| --- | --- |
| `main.py` | Entry point and readable dependency errors |
| `app.py` | CLI, processing loop, FPS and shutdown |
| `camera.py` | Camera ownership, backend fallback and cleanup |
| `detection.py` | Local YuNet inference and coordinate mapping |
| `tracking.py` | Association, temporary IDs, smoothing and states |
| `geometry.py` | Boxes, overlap and bracket calculations |
| `hud.py` | OpenCV interface rendering |
| `effects.py` | Time-based scan and pulse animations |

## Project Structure

```text
CyberHUD/
├── main.py
├── cyberhud/
│   ├── __init__.py
│   ├── app.py
│   ├── camera.py
│   ├── detection.py
│   ├── effects.py
│   ├── geometry.py
│   ├── hud.py
│   └── tracking.py
├── models/
│   ├── face_detection_yunet_2023mar.onnx
│   ├── LICENSE-YuNet.txt
│   └── SOURCE.txt
├── tests/
│   ├── test_cyberhud.py
│   └── test_v2.py
├── .gitignore
├── requirements.txt
├── LICENSE
└── README.md
```

`.venv/` is local only and excluded from publication. CyberHUD is [MIT licensed](LICENSE).

## Installation

Tested on Windows with standard CPython 3.13.3, OpenCV 4.12.0.88 and NumPy 2.2.6. Use a desktop session with a webcam and camera permission. Other platforms have not been manually verified.

Download/extract the project, open PowerShell in its root, then run:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Dependencies install only inside `.venv`. Activation is unnecessary. Use `opencv-python`, not the headless distribution, because CyberHUD needs an OpenCV window.

Keep the included `models/` folder beside `main.py`. The application loads the model locally and never downloads it automatically. Missing or invalid model files produce an initialization error before camera access. Model provenance, immutable upstream URLs and SHA-256 are in [models/SOURCE.txt](models/SOURCE.txt).

## Usage

```powershell
.\.venv\Scripts\python.exe main.py
```

Examples:

```powershell
.\.venv\Scripts\python.exe main.py --camera 1
.\.venv\Scripts\python.exe main.py --detect-width 480 --detect-every 2
.\.venv\Scripts\python.exe main.py --threads 1
.\.venv\Scripts\python.exe main.py --help
```

| Option | Default | Purpose |
| --- | --- | --- |
| `--camera` | `0` | Webcam index |
| `--width`, `--height` | `960`, `540` | Requested capture resolution |
| `--fps` | `30` | Requested camera FPS and loop cap |
| `--detect-width` | `320` | Maximum detector input width |
| `--detect-every` | `2` | Detect every N valid frames |
| `--threads` | `2` | OpenCV CPU worker threads |

If the camera is unavailable, check OS permissions, the device index and other apps using it. Thirty consecutive dropped frames end the session with an error.

## Controls

| Control | Action |
| --- | --- |
| Q or Escape, with the video window focused | Quit and release camera |
| Close video window | Quit and release camera |
| Ctrl+C in terminal | Interrupt and release camera |

## Testing

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
```

The 31-test suite covers geometry, session numbering, stable IDs, multiple faces, state transitions, temporary loss, removal, global HUD consistency, detector initialization, coordinate mapping, mirroring and cleanup. Camera and GUI lifecycle calls are mocked; tests do not activate a physical webcam. One simulated detector failure intentionally prints an error message.

Publication audit validation: **31/31 tests passed**, syntax checks passed for all 11 Python files, application imports passed, installed dependencies satisfied their constraints, `pip check` passed, and the model SHA-256 matched its recorded upstream value.

## Performance

The default target is approximately 30 FPS where hardware permits. V1 measured about 21 FPS on the development laptop. V2 has been manually verified as working; a real-world V2 FPS benchmark has not been recorded.

V2 reduces detection input to 320 pixels wide, detects every second valid frame and uses two CPU threads. Animations update every rendered frame without full-frame blur or bloom. FPS is calculated from measured, smoothed loop timing including capture and display waits, not simulated telemetry.

An earlier synthetic V2 run measured median detection at 4.81 ms and single-target HUD rendering plus frame copy at 2.71 ms (100 samples after 10 warmups, constant 960×540 frame, two threads). These exclude camera/display waits and are not end-to-end webcam FPS.

Larger detection inputs may help distant faces at a performance cost. Less frequent detection saves work but can cause lag or timeout if updates are too far apart. Camera drivers may choose a different resolution or frame rate.

## Privacy

Application code processes webcam frames locally in memory. It contains no networking, cloud upload, automatic recording or frame-saving operations. It stores no persistent target identities. Package installation and initial model provisioning involve downloads.

This is a source-code audit of CyberHUD, not a network audit of every third-party library or OS camera component. No blanket claim of zero third-party network activity is made.

## Technologies

- Python: application logic and standard-library unittest testing.
- OpenCV: capture, YuNet DNN inference and drawing primitives.
- NumPy: image arrays.
- YuNet ONNX: local face detection model from OpenCV Zoo.

Only OpenCV and NumPy are direct runtime dependencies. Version ranges stay within the validated release families; no unrelated transitive packages are pinned.

## Limitations

- Low light, occlusion, extreme poses and small faces can cause misses or false detections.
- Geometric tracking cannot guarantee identities through crossings or prolonged absence.
- Overlapping targets may overlap labels; compact frames omit secondary telemetry.
- Positions during brief misses are held/smoothed toward the last measurement.
- FPS depends on hardware and camera drivers; a blocking camera read may delay shutdown.
- This is an AR-style overlay, not calibrated 3D spatial registration.

## Future Improvements

Potential work, not current capabilities:

- More robust association during face crossings.
- Adaptive inference cadence and reproducible hardware benchmarks.
- Better label placement for crowded scenes.
- Additional platform verification and configurable visual themes.

## Model / Third-Party Notices

The bundled `face_detection_yunet_2023mar.onnx` comes from [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet). The upstream model README explicitly licenses all files in that directory under MIT. Redistribution is permitted with its copyright and permission notice retained.

The unmodified notice is included in [models/LICENSE-YuNet.txt](models/LICENSE-YuNet.txt), alongside [pinned provenance and checksum](models/SOURCE.txt). The binary matches the upstream Git LFS SHA-256. These files must stay together when redistributing the model. The 2023mar model is retained for the OpenCV 4.x backend.

Model reference: Wei Wu, Hanyang Peng and Shiqi Yu, *YuNet: A Tiny Millisecond-Level Face Detector*, Machine Intelligence Research 20, 656–665 (2023).

OpenCV and NumPy retain their respective upstream licenses; their distributions are installed separately and are not vendored in this repository.

## License

CyberHUD's original source code is licensed under the [MIT License](LICENSE).
Copyright (c) 2026 Anbu Udhaya Selvan.

YuNet is a third-party model with its own separate [MIT license and attribution](models/LICENSE-YuNet.txt). The project license does not claim authorship or ownership of YuNet; see Model / Third-Party Notices above.
