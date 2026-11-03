import threading
from contextlib import contextmanager
from time import monotonic, sleep

import numpy as np
import pytest

from inference_tester.capture import CapturedFrame, OpenCVCapture, VideoInfo
from inference_tester.config import Options, SessionConfig, Source, SourceKind
from inference_tester.engine import Prediction
from inference_tester.session import Session, State


def wait_until(predicate, timeout=4):
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        if value := predicate():
            return value
        sleep(0.005)
    raise AssertionError('Timed out waiting for worker')


def wait_frame(session, index=None):
    def check():
        frame = session.poll().frame
        return frame if frame is not None and (index is None or frame.index == index) else None

    return wait_until(check)


def wait_state(session, state):
    def check():
        snapshot = session.poll()
        return snapshot if snapshot.state == state else None

    return wait_until(check)


class FakeEngine:
    def __init__(self, *, block_first=False, fail=False):
        self.calls = []
        self.resets = 0
        self.closed = False
        self.entered = threading.Event()
        self.release = threading.Event()
        self.block_first = block_first
        self.fail = fail

    def process(self, frame, options):
        self.calls.append((frame.copy(), options))
        if self.block_first and len(self.calls) == 1:
            self.entered.set()
            assert self.release.wait(4)
        if self.fail:
            raise RuntimeError('Controlled inference failure')
        return Prediction(frame.copy(), 2, 3, 1)

    def reset(self):
        self.resets += 1

    def close(self):
        self.closed = True


class FakeCapture:
    def __init__(self, source, *, fps=20, count=6):
        self.info = VideoInfo(fps, count)
        self.index = 0
        self.closed = False
        self.seeks = []

    def read(self):
        if self.index >= self.info.frame_count:
            return None
        frame = CapturedFrame(
            np.full((8, 8, 3), self.index, dtype=np.uint8),
            self.index,
            self.index / self.info.fps,
        )
        self.index += 1
        return frame

    def seek(self, index):
        self.seeks.append(index)
        self.index = index

    def close(self):
        self.closed = True


@contextmanager
def running(model_path, video_path, *, engine=None, capture=None, paused=False):
    engine = engine or FakeEngine()
    capture = capture or FakeCapture(None)
    config = SessionConfig(model_path, Source(SourceKind.VIDEO, video_path))
    session = Session(config, engine_factory=lambda _: engine, capture_factory=lambda _: capture)
    session.set_paused(paused)
    session.start()
    try:
        yield session, engine, capture
    finally:
        session.stop()
        engine.release.set()
        session.join(4)
        assert not session.is_alive()


def test_pause_step_seek_resume_and_eof(model_path, video_path):
    with running(model_path, video_path, paused=True) as (session, engine, capture):
        wait_state(session, State.PAUSED)
        assert engine.calls == []
        session.step()
        assert wait_frame(session).index == 0
        sleep(0.08)
        assert len(engine.calls) == 1
        assert session.poll().paused
        session.step()
        assert wait_frame(session).index == 1
        session.seek(4)
        assert wait_frame(session).index == 4
        assert engine.resets == 1
        assert capture.seeks == [4]
        assert session.poll().paused
        session.set_paused(False)
        assert wait_frame(session, 5).index == 5
        assert wait_state(session, State.COMPLETED).paused
        session.set_paused(False)
        assert wait_frame(session, 0).index == 0
        assert capture.seeks == [4, 0]
        assert engine.resets == 2
    assert capture.closed and engine.closed


def test_seek_discards_inflight_and_coalesces_requests(model_path, video_path):
    engine = FakeEngine(block_first=True)
    with running(model_path, video_path, engine=engine, paused=True) as (session, _, capture):
        wait_state(session, State.PAUSED)
        session.step()
        assert engine.entered.wait(3)
        session.seek(2)
        session.seek(4)
        assert session.poll().frame is None
        engine.release.set()
        frame = wait_frame(session)
        assert frame.index == 4
        assert frame.generation == session.poll().generation
        assert capture.seeks == [4]
        assert session.poll().paused


def test_running_seek_keeps_playing(model_path, video_path):
    engine = FakeEngine(block_first=True)
    with running(model_path, video_path, engine=engine) as (session, _, _):
        assert engine.entered.wait(3)
        session.seek(3)
        engine.release.set()
        assert wait_frame(session).index == 3
        assert not session.poll().paused
        wait_state(session, State.COMPLETED)
        assert len(engine.calls) == 4  # stale frame 0, then 3, 4, 5


def test_options_change_during_inference_and_stop_suppresses_frame(model_path, video_path):
    engine = FakeEngine(block_first=True)
    with running(model_path, video_path, engine=engine, paused=True) as (session, _, _):
        wait_state(session, State.PAUSED)
        session.step()
        assert engine.entered.wait(3)
        session.set_options(Options(0.67, 'ByteTrack'))
        engine.release.set()
        wait_until(lambda: len(engine.calls) == 1 and session.poll().state == State.PAUSED)
        assert session.poll().frame is None
        session.step()
        frame = wait_frame(session)
        assert frame.options == Options(0.67, 'ByteTrack')
        assert frame.index == 1
        session.stop()
        assert session.poll().frame is None


def test_video_processes_every_frame_and_respects_pacing(model_path, video_path):
    with running(model_path, video_path) as (session, engine, _):
        started = monotonic()
        wait_state(session, State.COMPLETED)
        elapsed = monotonic() - started
        assert [int(frame[0, 0, 0]) for frame, _ in engine.calls] == list(range(6))
        assert elapsed >= 0.25


def test_generated_video_through_worker(model_path, video_path):
    engine = FakeEngine()
    config = SessionConfig(model_path, Source(SourceKind.VIDEO, video_path))
    session = Session(config, engine_factory=lambda _: engine, capture_factory=OpenCVCapture)
    session.start()
    try:
        wait_state(session, State.COMPLETED)
        assert len(engine.calls) == 8
    finally:
        session.stop()
        session.join(4)
    assert not session.is_alive()
    assert engine.closed


def test_inference_failure_releases_capture_and_engine(model_path, video_path):
    with running(model_path, video_path, engine=FakeEngine(fail=True)) as (
        session,
        engine,
        capture,
    ):
        snapshot = wait_state(session, State.ERROR)
        assert snapshot.error == 'Controlled inference failure'
        session.join(3)
        assert capture.closed and engine.closed


def test_stop_during_model_loading_does_not_open_source(model_path, video_path):
    entered = threading.Event()
    release = threading.Event()
    engine = FakeEngine()

    def load(_):
        entered.set()
        assert release.wait(3)
        return engine

    def unexpected_open(_):
        pytest.fail('Source opened after Stop was requested')

    config = SessionConfig(model_path, Source(SourceKind.VIDEO, video_path))
    session = Session(config, engine_factory=load, capture_factory=unexpected_open)
    session.start()
    try:
        assert entered.wait(3)
        session.stop()
    finally:
        release.set()
        session.join(4)
    assert not session.is_alive()
    assert engine.closed
    assert session.poll().state == State.STOPPED


def test_empty_video_reports_error_and_cleans_up(model_path, video_path):
    with running(model_path, video_path, capture=FakeCapture(None, count=0)) as (
        session,
        engine,
        capture,
    ):
        assert 'no decodable frames' in wait_state(session, State.ERROR).error
        session.join(3)
        assert engine.closed and capture.closed


def test_live_failure_finishes_both_workers(model_path):
    engine = FakeEngine()
    capture = FakeCapture(None, count=0)
    config = SessionConfig(model_path, Source(SourceKind.RTSP, 'rtsp://camera.test/live'))
    session = Session(config, engine_factory=lambda _: engine, capture_factory=lambda _: capture)
    session.start()
    session.join(3)
    assert not session.is_alive()
    assert 'RTSP stream failed' in session.poll().error
    assert engine.closed and capture.closed


def test_cleanup_attempts_engine_even_if_capture_close_fails(model_path, video_path):
    class BadRelease(FakeCapture):
        def close(self):
            super().close()
            raise OSError('Controlled release failure')

    with running(model_path, video_path, capture=BadRelease(None), paused=True) as (
        session,
        engine,
        capture,
    ):
        wait_state(session, State.PAUSED)
        session.stop()
        session.join(3)
        assert capture.closed and engine.closed
        assert 'Controlled release failure' in session.poll().error


def test_stop_during_live_open_waits_for_owned_release(model_path):
    entered = threading.Event()
    release = threading.Event()
    capture = FakeCapture(None)
    engine = FakeEngine()

    def open_capture(_):
        entered.set()
        assert release.wait(3)
        return capture

    config = SessionConfig(model_path, Source(SourceKind.CAMERA, 0))
    session = Session(config, engine_factory=lambda _: engine, capture_factory=open_capture)
    session.start()
    try:
        assert entered.wait(3)
        session.stop()
    finally:
        release.set()
        session.join(4)
    assert not session.is_alive()
    assert capture.closed and engine.closed
    assert session.poll().state == State.STOPPED
