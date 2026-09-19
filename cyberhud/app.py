"""Capture, detect, track and render orchestration."""
import argparse
import sys
import time
from datetime import datetime
import cv2
import numpy as np
from .camera import Camera
from .detection import FaceDetector
from .tracking import Tracker
from .hud import HUD

WINDOW = 'CyberHUD // Local Vision'


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be greater than zero')
    return number


def run(camera, detector, hud=None, tracker=None, detect_every=2, fps_limit=30):
    hud, tracker = hud or HUD(), tracker or Tracker()
    frames = failures = 0
    fps = 0.0
    average_interval = None
    previous = None
    last_frame = None
    try:
        with camera:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            while True:
                started = time.perf_counter()
                frame = camera.read()
                if previous is not None:
                    interval = max(started-previous,1e-6)
                    average_interval = interval if average_interval is None else 0.9*average_interval+0.1*interval
                    fps = 1/average_interval
                previous = started
                if frame is None:
                    failures += 1
                    if failures >= 30:
                        raise RuntimeError('Webcam stopped delivering frames after 30 retries.')
                    # Do not leave a stale face picture appearing to be live.
                    shape = last_frame.shape if last_frame is not None else (540,960,3)
                    frame = np.zeros(shape, dtype=np.uint8)
                    targets = tracker.update([], started)
                    shown = hud.draw(frame, targets, started, fps,
                                     datetime.now().strftime('%Y-%m-%d %H:%M:%S'), True)
                else:
                    failures = 0
                    frame = cv2.flip(frame, 1)
                    last_frame = frame
                    detections = detector.detect(frame) if frames % detect_every == 0 else None
                    targets = tracker.update(detections, started)
                    shown = hud.draw(frame, targets, started, fps,
                                     datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    frames += 1
                cv2.imshow(WINDOW, shown)
                elapsed = time.perf_counter()-started
                delay = max(1, round((1/fps_limit-elapsed)*1000))
                key = cv2.waitKey(delay) & 0xff
                if key in (ord('q'), ord('Q'), 27):
                    break
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        cv2.destroyAllWindows()


def main(argv=None):
    parser = argparse.ArgumentParser(description='CyberHUD: private local webcam face HUD')
    parser.add_argument('--camera', type=int, default=0, help='webcam index (default: 0)')
    parser.add_argument('--width', type=positive, default=960)
    parser.add_argument('--height', type=positive, default=540)
    parser.add_argument('--fps', type=positive, default=30, help='requested FPS and render cap')
    parser.add_argument('--detect-every', type=positive, default=2)
    parser.add_argument('--detect-width', type=positive, default=320)
    parser.add_argument('--threads', type=positive, default=2, help='OpenCV CPU worker threads')
    args = parser.parse_args(argv)
    try:
        # Initialize detector before requesting camera access.
        cv2.setNumThreads(args.threads)
        detector = FaceDetector(args.detect_width)
        camera = Camera(args.camera,args.width,args.height,args.fps)
        run(camera,detector,detect_every=args.detect_every,fps_limit=args.fps)
    except KeyboardInterrupt:
        return 0
    except (RuntimeError, cv2.error, OSError) as exc:
        print(f'CyberHUD: {exc}', file=sys.stderr)
        return 1
    return 0
