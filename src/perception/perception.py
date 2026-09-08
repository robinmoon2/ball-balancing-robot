from src.perception.detector import OrangeDetector, Detection
from src.perception.camera import Camera

class Perception:
    def __init__(self, camera_source: int | str = 0, width: int = 640, height: int = 480):
        self.camera = Camera(source=camera_source, width=width, height=height)
        self.detector = OrangeDetector()

    def process_frame(self) -> Detection:
        frame = self.camera.read()
        if frame is None:
            return Detection(found=False)
        return self.detector.detect(frame)

    def release(self) -> None:
        self.camera.release()