"""Camera abstraction: hides the source (webcam, Pi cam, video file)."""

from __future__ import annotations
import shutil
import subprocess

import cv2
import numpy as np


class Camera:
    def __init__(self, source: int | str = 0, width: int = 640, height: int = 480):
        self._process = None
        self._buffer = b""

        # Raspberry Pi CSI cameras are managed by libcamera, not as ordinary
        # V4L2 streams. rpicam-vid provides a dependency-free MJPEG stream.
        if source == 0 and shutil.which("rpicam-vid"):
            self._process = subprocess.Popen(
                [
                    "rpicam-vid",
                    "--codec",
                    "mjpeg",
                    "--timeout",
                    "0",
                    "--nopreview",
                    "--width",
                    str(width),
                    "--height",
                    str(height),
                    "--framerate",
                    "30",
                    "--output",
                    "-",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            self._cap = None
            return

        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera source: {source}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self) -> np.ndarray | None:
        if self._process is not None:
            while True:
                start = self._buffer.find(b"\xff\xd8")
                end = self._buffer.find(b"\xff\xd9", start + 2)
                if start >= 0 and end >= 0:
                    jpeg = self._buffer[start : end + 2]
                    self._buffer = self._buffer[end + 2 :]
                    frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        return frame

                chunk = self._process.stdout.read(4096)
                if not chunk:
                    return None
                self._buffer += chunk

        ok, frame = self._cap.read()
        return frame if ok else None

    def release(self) -> None:
        if self._process is not None:
            self._process.terminate()
            self._process.wait()
            self._process = None
        elif self._cap is not None:
            self._cap.release()

    # Context manager support: `with Camera() as cam:`
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
