"""Camera abstraction: hides the source (Pi cam, webcam, video file).

The Pi CSI path streams RAW I420 (planar YUV 4:2:0), not MJPEG. Three
reasons, all of which matter for a control loop:

  - No decode. MJPEG cost ~8.9 ms per frame in cv2.imdecode before the
    detector had even started. I420 is already pixels.
  - Fixed frame size. An I420 frame is exactly width*height*3//2 bytes, so
    framing is `len(buffer) // frame_bytes` instead of scanning for JPEG
    \xff\xd8/\xff\xd9 markers. That is what makes "skip to the newest
    frame" (below) a two-line operation.
  - The chroma planes are already half-resolution. A colour-blob detector
    wants U/V at 320x240, not an RGB image at 640x480 - see
    OrangeDetector.detect_yuv.

LATENCY, which is why read_frame() exists:
    rpicam-vid free-runs at --framerate into a pipe whether or not we are
    reading. If a loop tick is slower than the frame period, frames queue
    up, and the OLD reader returned the OLDEST queued frame - so the
    control loop acted on a stale ball position. read_frame() instead
    drains everything available and keeps only the NEWEST complete frame,
    reporting the rest via `dropped_frames`.

    It also stamps each frame with the time it arrived (`t_capture`), so
    the estimator can fuse the measurement at the instant it belongs to
    rather than at "whenever the loop got around to it". Note this is the
    arrival time in this process: the sensor exposure and the trip through
    rpicam-vid happened slightly earlier. That residual is roughly
    constant, so measure it once and pass it as `latency_offset_s`.

Dimensions must be multiples of 32 (width) and 16 (height), otherwise
libcamera pads the planes and the fixed-size framing above silently
desynchronises. 640x480 and 320x240 both qualify.
"""

from __future__ import annotations

import os
import select
import shutil
import subprocess
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Frame:
    """One captured frame, plus when it arrived.

    Holds whichever representation its source produced natively and
    converts on demand - so the I420 path never pays for a BGR image it
    doesn't use, and the VideoCapture fallback never pays to go to I420
    and back.
    """

    t_capture: float  # time.monotonic() when this frame reached us
    width: int
    height: int
    _i420: np.ndarray | None = None  # (H*3//2, W) uint8, planar
    _bgr: np.ndarray | None = None  # (H, W, 3) uint8

    @property
    def i420(self) -> np.ndarray:
        if self._i420 is None:
            self._i420 = cv2.cvtColor(self._bgr, cv2.COLOR_BGR2YUV_I420)
        return self._i420

    @property
    def bgr(self) -> np.ndarray:
        if self._bgr is None:
            self._bgr = cv2.cvtColor(self._i420, cv2.COLOR_YUV2BGR_I420)
        return self._bgr

    @property
    def y(self) -> np.ndarray:
        """Luma plane, full resolution (H, W)."""
        return self.i420[: self.height]

    @property
    def u(self) -> np.ndarray:
        """Blue-difference chroma, half resolution (H//2, W//2)."""
        h, w = self.height, self.width
        return self.i420[h : h + h // 4].reshape(h // 2, w // 2)

    @property
    def v(self) -> np.ndarray:
        """Red-difference chroma, half resolution (H//2, W//2)."""
        h, w = self.height, self.width
        return self.i420[h + h // 4 : h + h // 2].reshape(h // 2, w // 2)


class Camera:
    def __init__(
        self,
        source: int | str = 0,
        width: int = 640,
        height: int = 480,
        framerate: int = 30,
        latency_offset_s: float = 0.0,
        shutter_us: int | None = None,
        gain: float | None = None,
        awb_gains: tuple[float, float] | None = None,
        denoise: str | None = None,
    ):
        """shutter_us/gain/awb_gains lock the sensor's auto algorithms.

        Leave them None and libcamera runs auto-exposure and auto-white-
        balance, which is fine for looking at pictures and wrong for a
        control loop: the first ~0.5 s of a run is an exposure transient,
        and AE/AWB keep re-converging whenever tilting the plate changes
        how much light reaches the sensor - which moves the U/V values
        OrangeDetector.detect_yuv thresholds against.

        Set shutter_us and gain TOGETHER; fixing only one leaves the other
        free and auto-exposure still chases. Measure all three with
        `python -m src.perception.calibrate_exposure`.
        """
        self._process = None
        self._fd = -1
        self._buffer = b""
        self.width = width
        self.height = height
        self.framerate = framerate
        self.latency_offset_s = latency_offset_s
        self.frame_bytes = width * height * 3 // 2
        self.dropped_frames = 0  # frames skipped as stale, cumulative

        # Raspberry Pi CSI cameras are managed by libcamera, not as ordinary
        # V4L2 streams. rpicam-vid provides a dependency-free raw stream.
        if source == 0 and shutil.which("rpicam-vid"):
            if width % 32 or height % 16:
                raise ValueError(
                    f"{width}x{height}: width must be a multiple of 32 and height "
                    "a multiple of 16, otherwise libcamera pads the I420 planes "
                    "and fixed-size framing desynchronises"
                )
            if (shutter_us is None) != (gain is None):
                raise ValueError(
                    "set shutter_us and gain together: fixing only one leaves "
                    "auto-exposure free to chase with the other"
                )
            argv = [
                "rpicam-vid",
                "--codec", "yuv420",
                "--timeout", "0",
                "--nopreview",
                "--flush",  # hand each frame over immediately, don't batch
                "--width", str(width),
                "--height", str(height),
                "--framerate", str(framerate),
                "--output", "-",
            ]
            if shutter_us is not None:
                argv += ["--shutter", str(int(shutter_us)), "--gain", str(gain)]
            if awb_gains is not None:
                argv += ["--awbgains", f"{awb_gains[0]},{awb_gains[1]}"]
            if denoise is not None:
                argv += ["--denoise", denoise]
            self.argv = argv
            self._process = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,  # raw fd: we do our own buffering and draining
            )
            self._fd = self._process.stdout.fileno()
            os.set_blocking(self._fd, False)
            self._cap = None
            return

        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera source: {source}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read_frame(self, timeout_s: float = 1.0) -> Frame | None:
        """Newest available frame, or None on timeout/end of stream.

        Any frames that queued up behind it are discarded, not returned
        later: in a control loop a stale position is worse than no
        position. `dropped_frames` counts them, and a steadily climbing
        count means the loop is slower than --framerate.
        """
        if self._process is None:
            ok, bgr = self._cap.read()
            if not ok:
                return None
            return Frame(
                t_capture=time.monotonic() - self.latency_offset_s,
                width=self.width,
                height=self.height,
                _bgr=bgr,
            )

        deadline = time.monotonic() + timeout_s
        while True:
            # Pull everything the pipe currently holds.
            while True:
                try:
                    chunk = os.read(self._fd, 1 << 20)
                except BlockingIOError:
                    break
                if not chunk:  # stream ended
                    return None
                self._buffer += chunk

            n_complete = len(self._buffer) // self.frame_bytes
            if n_complete:
                t_capture = time.monotonic() - self.latency_offset_s
                # Keep the LAST complete frame; everything before it is stale.
                end = n_complete * self.frame_bytes
                start = end - self.frame_bytes
                i420 = np.frombuffer(
                    self._buffer[start:end], dtype=np.uint8
                ).reshape(self.height * 3 // 2, self.width)
                self._buffer = self._buffer[end:]
                self.dropped_frames += n_complete - 1
                return Frame(
                    t_capture=t_capture,
                    width=self.width,
                    height=self.height,
                    _i420=i420,
                )

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            select.select([self._fd], [], [], remaining)

    def read(self) -> np.ndarray | None:
        """Newest frame as a BGR image, or None.

        Compatibility shim for callers that want an ordinary OpenCV image
        (calibrate.py, test_perception.py). It pays a YUV->BGR conversion
        that the detector's chroma path does not need - prefer
        read_frame() in the control loop.
        """
        frame = self.read_frame()
        return None if frame is None else frame.bgr

    def release(self) -> None:
        if self._process is not None:
            self._process.terminate()
            self._process.wait()
            self._process = None
            self._fd = -1
        elif self._cap is not None:
            self._cap.release()

    # Context manager support: `with Camera() as cam:`
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
