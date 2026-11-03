import threading

import cv2
import numpy as np
import pytest

from inference_tester.capture import (
    RTSP_OPEN_TIMEOUT_MS,
    RTSP_READ_TIMEOUT_MS,
    CapturedFrame,
    LiveCapture,
    OpenCVCapture,
    VideoInfo,
)
from inference_tester.config import Source, SourceKind


def test_generated_video_reads_seeks_and_releases(video_path):
    capture = OpenCVCapture(Source(SourceKind.VIDEO, video_path))
    assert capture.info == VideoInfo(10, 8)
    assert capture.read().index == 0
    capture.seek(5)
    frame = capture.read()
    assert frame.index == 5
    assert frame.timestamp == 0.5
    assert frame.image.shape == (64, 96, 3)
    capture.seek(7)
    assert capture.read().index == 7
    assert capture.read() is None
    capture.close()
    assert not capture.cap.isOpened()


def test_rtsp_timeouts_are_open_parameters_and_failure_releases(monkeypatch):
    calls = []

    class FailedCapture:
        released = False

        def __init__(self, *args):
            calls.append(args)

        def isOpened(self):
            return False

        def release(self):
            self.released = True

    instance = FailedCapture()
    monkeypatch.setattr(cv2, 'VideoCapture', lambda *args: (calls.append(args), instance)[1])
    with pytest.raises(OSError, match='Check the URL') as error:
        OpenCVCapture(Source(SourceKind.RTSP, 'rtsp://user:secret@example.test/live'))
    assert 'secret' not in str(error.value)
    assert calls[-1][1] == cv2.CAP_FFMPEG
    assert calls[-1][2] == [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
        RTSP_OPEN_TIMEOUT_MS,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC,
        RTSP_READ_TIMEOUT_MS,
    ]
    assert instance.released


def test_live_capture_retains_only_latest_frame_and_owns_cleanup():
    burst_done = threading.Event()
    release_read = threading.Event()
    closed_by = []

    class BurstCapture:
        info = VideoInfo(30)

        def __init__(self, source):
            self.index = 0

        def read(self):
            if self.index == 10:
                burst_done.set()
                assert release_read.wait(3)
                return None
            index = self.index
            self.index += 1
            return CapturedFrame(np.zeros((8, 8, 3), dtype=np.uint8), index, None)

        def close(self):
            closed_by.append(threading.current_thread().name)

    live = LiveCapture(Source(SourceKind.CAMERA, 0), capture_factory=BurstCapture)
    live.start()
    try:
        assert burst_done.wait(3)
        assert live.take().index == 9
        assert live.take(timeout=0.01) is None
    finally:
        live.stop()
        release_read.set()
        live.join(3)
    assert not live.is_alive()
    assert closed_by == ['live-capture']
    assert live.error is None


def test_live_disconnect_is_reported_and_released():
    closed = []

    class DisconnectedCapture:
        info = VideoInfo(30)

        def __init__(self, source):
            pass

        def read(self):
            return None

        def close(self):
            closed.append(True)

    live = LiveCapture(Source(SourceKind.CAMERA, 0), capture_factory=DisconnectedCapture)
    live.start()
    live.join(3)
    with pytest.raises(OSError, match='stopped delivering'):
        live.take()
    assert closed == [True]


@pytest.mark.parametrize('url', ['http://camera/live', 'rtsp://', 'rtsp://camera:bad/live'])
def test_invalid_rtsp_url(url):
    with pytest.raises(ValueError, match='RTSP URL'):
        Source(SourceKind.RTSP, url).validate()
