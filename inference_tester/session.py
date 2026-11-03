"""Playback and inference, with a command mailbox and one pending display frame.

The GUI only polls snapshots and posts commands. No Qt objects cross threads and
slow inference cannot accumulate queued image signals or block the GUI event loop.
"""

import threading
from dataclasses import dataclass
from enum import StrEnum
from time import monotonic

import numpy as np

from inference_tester.capture import LiveCapture, OpenCVCapture, VideoInfo
from inference_tester.config import InferenceDevice, Options, SessionConfig, SourceKind
from inference_tester.engine import YoloEngine


class State(StrEnum):
    LOADING = 'Loading model'
    CONVERTING = 'Preparing Intel iGPU model (conversion may take a few minutes)'
    OPENING = 'Opening source'
    RUNNING = 'Running'
    PAUSED = 'Paused'
    SEEKING = 'Seeking'
    COMPLETED = 'Completed'
    STOPPING = 'Stopping'
    STOPPED = 'Stopped'
    ERROR = 'Error'


@dataclass(frozen=True)
class DisplayFrame:
    image: np.ndarray
    index: int
    timestamp: float | None
    fps: float
    inference_ms: float
    detections: int
    options: Options
    generation: int


@dataclass(frozen=True)
class Snapshot:
    state: State
    info: VideoInfo
    paused: bool
    error: str | None
    generation: int
    frame: DisplayFrame | None


@dataclass(frozen=True)
class Action:
    options: Options
    generation: int
    seek: int | None
    paused: bool
    rate_revision: int


class Session(threading.Thread):
    def __init__(
        self,
        config: SessionConfig,
        options: Options | None = None,
        *,
        engine_factory=YoloEngine,
        capture_factory=OpenCVCapture,
    ) -> None:
        super().__init__(name='inference-playback', daemon=True)
        self.config = config
        self._engine_factory = engine_factory
        self._capture_factory = capture_factory
        self._condition = threading.Condition()
        self._options = options or Options()
        self._paused = False
        self._steps = 0
        self._seek: int | None = None
        self._stop_requested = False
        self._generation = 0
        self._rate_revision = 0
        self._state = State.LOADING
        self._info = VideoInfo()
        self._error: str | None = None
        self._frame: DisplayFrame | None = None

    @property
    def is_video(self) -> bool:
        return self.config.source.kind == SourceKind.VIDEO

    def poll(self) -> Snapshot:
        with self._condition:
            frame, self._frame = self._frame, None
            return Snapshot(
                self._state, self._info, self._paused, self._error, self._generation, frame
            )

    def set_options(self, options: Options) -> None:
        with self._condition:
            if options.tracker != self._options.tracker:
                self._invalidate()
            self._options = options
            self._condition.notify_all()

    def set_paused(self, paused: bool) -> None:
        if not self.is_video:
            return
        with self._condition:
            if self._state == State.COMPLETED and not paused:
                self._seek = 0
                self._invalidate()
            self._paused = paused
            self._steps = 0
            self._rate_revision += 1
            self._condition.notify_all()

    def step(self) -> None:
        if not self.is_video:
            return
        with self._condition:
            if self._state == State.COMPLETED:
                return
            self._paused = True
            self._steps += 1
            self._rate_revision += 1
            self._condition.notify_all()

    def seek(self, index: int) -> None:
        with self._condition:
            if not self.is_video or not self._info.seekable or self._stop_requested:
                return
            self._seek = max(0, min(index, self._info.frame_count - 1))
            self._steps = 0
            self._invalidate()
            self._state = State.SEEKING
            self._condition.notify_all()

    def stop(self) -> None:
        with self._condition:
            self._stop_requested = True
            self._invalidate()
            if self.is_alive():
                self._state = State.STOPPING
            self._condition.notify_all()

    def _invalidate(self) -> None:
        # Called with the mailbox lock held. In-flight work from older generations
        # can finish, but it must never overwrite the result of a seek or tracker change.
        self._generation += 1
        self._rate_revision += 1
        self._frame = None

    def _set_state(self, state: State) -> None:
        with self._condition:
            if not self._stop_requested:
                self._state = state

    def _next_action(self) -> Action | None:
        with self._condition:
            while not self._stop_requested:
                if self._seek is not None or not self._paused or self._steps:
                    seek, self._seek = self._seek, None
                    if seek is None and self._steps:
                        self._steps -= 1
                    return Action(
                        self._options,
                        self._generation,
                        seek,
                        self._paused,
                        self._rate_revision,
                    )
                if self._state != State.COMPLETED:
                    self._state = State.PAUSED
                self._condition.wait()
            return None

    def _current(self, action: Action) -> bool:
        return not self._stop_requested and action.generation == self._generation

    def _pace(self, action: Action, deadline: float) -> None:
        with self._condition:
            while self._current(action) and not self._paused and self._seek is None:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)

    def run(self) -> None:
        engine = None
        capture = None
        live = None
        try:
            self.config.validate()
            with self._condition:
                if self._stop_requested:
                    return
            if self.config.device == InferenceDevice.INTEL_IGPU:
                self._set_state(State.CONVERTING)
                engine = self._engine_factory(self.config.model_path, device=self.config.device)
            else:
                engine = self._engine_factory(self.config.model_path)
            with self._condition:
                if self._stop_requested:
                    return
            self._set_state(State.OPENING)
            if self.is_video:
                capture = self._capture_factory(self.config.source)
                with self._condition:
                    self._info = capture.info
            else:
                live = LiveCapture(self.config.source, capture_factory=self._capture_factory)
                live.start()
            last_published = None
            rate_revision = -1
            had_frame = False
            while (action := self._next_action()) is not None:
                started = monotonic()
                if action.seek is not None:
                    capture.seek(action.seek)
                    engine.reset()
                if self.is_video:
                    frame = capture.read()
                    if frame is None:
                        if not had_frame:
                            raise OSError(
                                'The video contains no decodable frames. Try another file.'
                            )
                        with self._condition:
                            if self._current(action):
                                self._paused = True
                                self._steps = 0
                                self._state = State.COMPLETED
                        continue
                else:
                    frame = live.take()
                    if live.info is not None:
                        with self._condition:
                            self._info = live.info
                    if frame is None:
                        continue
                with self._condition:
                    if not self._current(action):
                        continue
                prediction = engine.process(frame.image, action.options)
                now = monotonic()
                if action.rate_revision != rate_revision:
                    last_published = None
                    rate_revision = action.rate_revision
                elapsed = now - (last_published if last_published is not None else started)
                with self._condition:
                    if self._current(action):
                        self._frame = DisplayFrame(
                            prediction.image,
                            frame.index,
                            frame.timestamp,
                            1 / max(elapsed, 1e-6),
                            prediction.inference_ms,
                            prediction.detections,
                            action.options,
                            action.generation,
                        )
                        self._state = State.PAUSED if self._paused else State.RUNNING
                        last_published = now
                        had_frame = True
                # Pace from the start of this frame, so CPU delays never trigger
                # a catch-up burst or skipping frames in a video file.
                if self.is_video and not action.paused:
                    self._pace(action, started + 1 / (self._info.fps or 30))
        except Exception as exc:
            with self._condition:
                if not self._stop_requested:
                    self._error = str(exc) or type(exc).__name__
                    self._state = State.ERROR
        finally:
            if live is not None:
                live.stop()
                # RTSP reads/opening have finite backend timeouts. Keep the
                # GUI responsive while this worker waits for capture ownership.
                live.join()
            for resource in (capture, engine):
                if resource is None:
                    continue
                try:
                    resource.close()
                except Exception as exc:
                    with self._condition:
                        message = f'Could not finish cleanup: {exc}'
                        self._error = f'{self._error}\n{message}' if self._error else message
                        self._state = State.ERROR
            with self._condition:
                if self._state != State.ERROR:
                    self._state = State.STOPPED
