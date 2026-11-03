"""OpenCV capture ownership and a bounded, latest-frame live reader."""

import math
import sys
import threading
from dataclasses import dataclass
from time import monotonic

import cv2
import numpy as np

from inference_tester.config import Source, SourceKind

RTSP_OPEN_TIMEOUT_MS = 5000
RTSP_READ_TIMEOUT_MS = 2000


@dataclass(frozen=True)
class VideoInfo:
    fps: float | None = None
    frame_count: int | None = None

    @property
    def seekable(self) -> bool:
        return self.frame_count is not None and self.frame_count > 1 and self.fps is not None


@dataclass(frozen=True)
class CapturedFrame:
    image: np.ndarray
    index: int
    timestamp: float | None


class OpenCVCapture:
    def __init__(self, source: Source) -> None:
        self.source = source
        self._next_index = 0
        self.cap = None
        try:
            if source.kind == SourceKind.RTSP:
                self.cap = cv2.VideoCapture(
                    str(source.value),
                    cv2.CAP_FFMPEG,
                    [
                        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                        RTSP_OPEN_TIMEOUT_MS,
                        cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                        RTSP_READ_TIMEOUT_MS,
                    ],
                )
            elif source.kind == SourceKind.CAMERA:
                backend = cv2.CAP_DSHOW if sys.platform == 'win32' else cv2.CAP_ANY
                self.cap = cv2.VideoCapture(int(source.value), backend)
            else:
                self.cap = cv2.VideoCapture(str(source.value), cv2.CAP_FFMPEG)
            if not self.cap.isOpened():
                raise OSError(self._open_error())
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.info = VideoInfo(
                fps=fps if math.isfinite(fps) and fps > 0 else None,
                frame_count=(
                    int(count)
                    if source.kind == SourceKind.VIDEO and math.isfinite(count) and count > 0
                    else None
                ),
            )
        except Exception:
            self.close()
            # Backend exceptions may include RTSP credentials; do not echo them.
            raise OSError(self._open_error()) from None

    def _open_error(self) -> str:
        if self.source.kind == SourceKind.CAMERA:
            return 'Cannot open camera. Check the device index and close other apps using it.'
        if self.source.kind == SourceKind.RTSP:
            return 'Cannot open RTSP stream. Check the URL, credentials, and network connection.'
        return 'Cannot open video. Check that the file is readable and uses a supported codec.'

    def read(self) -> CapturedFrame | None:
        success, image = self.cap.read()
        if not success or image is None:
            return None
        index = self._next_index
        self._next_index += 1
        timestamp = (
            index / self.info.fps
            if self.source.kind == SourceKind.VIDEO and self.info.fps is not None
            else None
        )
        return CapturedFrame(image, index, timestamp)

    def seek(self, index: int) -> None:
        if not self.info.seekable:
            raise OSError('This video does not provide the metadata needed for seeking.')
        index = max(0, min(index, self.info.frame_count - 1))
        if not self.cap.set(cv2.CAP_PROP_POS_FRAMES, index):
            raise OSError(
                'The video decoder could not seek to that frame. Try another video codec.'
            )
        position = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        self._next_index = int(position) if math.isfinite(position) and position >= 0 else index

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()


class LiveCapture(threading.Thread):
    """This thread alone opens, reads, and releases its capture object."""

    def __init__(self, source: Source, *, capture_factory=OpenCVCapture) -> None:
        super().__init__(name='live-capture', daemon=True)
        self.source = source
        self._factory = capture_factory
        self._condition = threading.Condition()
        self._stop_requested = threading.Event()
        self._latest: CapturedFrame | None = None
        self.info: VideoInfo | None = None
        self.error: str | None = None
        self.done = False

    def run(self) -> None:
        capture = None
        try:
            if self._stop_requested.is_set():
                return
            capture = self._factory(self.source)
            with self._condition:
                self.info = capture.info
                self._condition.notify_all()
            while not self._stop_requested.is_set():
                frame = capture.read()
                if self._stop_requested.is_set():
                    break
                if frame is None:
                    raise OSError(
                        'Live source stopped delivering frames. Check the connection and restart.'
                    )
                with self._condition:
                    self._latest = frame
                    self._condition.notify_all()
        except Exception as exc:
            with self._condition:
                self.error = (
                    'RTSP stream failed. Check the URL, credentials, and network connection.'
                    if self.source.kind == SourceKind.RTSP
                    else str(exc)
                )
        finally:
            if capture is not None:
                try:
                    capture.close()
                except Exception:
                    with self._condition:
                        self.error = 'Could not release the live source. Close other camera apps.'
            with self._condition:
                self.done = True
                self._condition.notify_all()

    def take(self, timeout: float = 0.1) -> CapturedFrame | None:
        deadline = monotonic() + timeout
        with self._condition:
            while self._latest is None and not self.done and not self._stop_requested.is_set():
                remaining = deadline - monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
            if self.error:
                raise OSError(self.error)
            frame, self._latest = self._latest, None
            return frame

    def stop(self) -> None:
        self._stop_requested.set()
        with self._condition:
            self._condition.notify_all()
