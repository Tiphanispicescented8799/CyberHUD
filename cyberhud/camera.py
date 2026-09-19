"""Camera ownership and deterministic cleanup."""
import os
import cv2


class Camera:
    def __init__(self, index=0, width=960, height=540, fps=30, factory=None):
        self.index, self.width, self.height, self.fps = index, width, height, fps
        self.factory = factory or cv2.VideoCapture
        self.capture = None

    def open(self):
        try:
            backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if os.name == 'nt' else [cv2.CAP_ANY]
            for backend in backends:
                self.capture = self.factory(self.index, backend)
                if self.capture.isOpened():
                    break
                self.close()
            if self.capture is None:
                raise RuntimeError('Webcam unavailable. Check camera permissions, index, and other apps.')
            for key, value in ((cv2.CAP_PROP_FRAME_WIDTH,self.width),
                               (cv2.CAP_PROP_FRAME_HEIGHT,self.height),
                               (cv2.CAP_PROP_FPS,self.fps), (cv2.CAP_PROP_BUFFERSIZE,1)):
                self.capture.set(key, value)
            return self
        except BaseException:
            self.close()
            raise

    def read(self):
        ok, frame = self.capture.read()
        return frame if ok and frame is not None and frame.size else None

    def close(self):
        if self.capture is not None:
            capture, self.capture = self.capture, None
            capture.release()

    def __enter__(self):
        return self.open()

    def __exit__(self, *_):
        self.close()
