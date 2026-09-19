"""Local YuNet inference; model loading never triggers network access."""
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
from .geometry import Box


@dataclass(frozen=True)
class Detection:
    box: Box
    confidence: float


class FaceDetector:
    def __init__(self, width=320, model_path=None, threshold=0.75):
        self.width = width
        path = Path(model_path) if model_path else (
            Path(__file__).resolve().parent.parent / 'models/face_detection_yunet_2023mar.onnx')
        try:
            if not path.is_file():
                raise RuntimeError(f'local YuNet model missing: {path}; restore the models folder')
            self.classifier = cv2.FaceDetectorYN.create(
                str(path), '', (320,320), threshold, 0.3, 5000,
                cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU)
            self.classifier.detect(np.zeros((320,320,3),np.uint8))
        except (cv2.error, RuntimeError) as exc:
            raise RuntimeError(f'Face detector initialization failed: {exc}') from exc

    def detect(self, frame):
        height, width = frame.shape[:2]
        small_width = min(width, self.width)
        small_height = max(1, round(height*small_width/width))
        small = cv2.resize(frame,(small_width,small_height),interpolation=cv2.INTER_AREA)
        self.classifier.setInputSize((small_width,small_height))
        _, faces = self.classifier.detect(small)
        if faces is None:
            return []
        sx, sy = width/small_width, height/small_height
        result = []
        for face in faces:
            if not np.isfinite(face).all():
                continue
            x,y,w,h = (float(v) for v in face[:4])
            x1,y1 = max(0,x*sx),max(0,y*sy)
            x2,y2 = min(width,(x+w)*sx),min(height,(y+h)*sy)
            if x2 > x1 and y2 > y1:
                result.append(Detection(Box(x1,y1,x2-x1,y2-y1),float(face[-1])))
        return result
