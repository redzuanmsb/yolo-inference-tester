from functools import partial
from time import monotonic, sleep

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog

from inference_tester.config import InferenceDevice, SourceKind
from inference_tester.engine import Prediction
from inference_tester.session import Session, State
from inference_tester.ui import SEEK_STEPS, MainWindow, format_time


class PreviewEngine:
    def __init__(self, path, *, device=InferenceDevice.CPU):
        self.closed = False

    def process(self, image, options):
        return Prediction(np.ascontiguousarray(image), 2.5, 3, 0)

    def reset(self):
        pass

    def close(self):
        self.closed = True


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def wait_ui(app, predicate, timeout=4):
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        app.processEvents()
        if value := predicate():
            return value
        sleep(0.005)
    raise AssertionError('Timed out waiting for UI')


def test_desktop_controls_file_pickers_playback_seek_and_close(
    app, monkeypatch, model_path, video_path
):
    window = MainWindow(session_factory=partial(Session, engine_factory=PreviewEngine))
    window.show()
    try:
        monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *args: (model_path, ''))
        window.model_button.click()
        assert window.model_path.text() == model_path
        window.source_type.setCurrentIndex(window.source_type.findData(SourceKind.VIDEO))
        monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *args: (video_path, ''))
        window._choose_video()
        assert window.video_path.text() == video_path
        assert window.device.currentData() == InferenceDevice.CPU
        window.device.setCurrentIndex(window.device.findData(InferenceDevice.INTEL_IGPU))
        window.start_button.click()
        wait_ui(app, lambda: window.pause_button.isEnabled())
        assert window.session.config.device == InferenceDevice.INTEL_IGPU
        assert not window.device.isEnabled()
        assert not window.input_group.isEnabled()
        assert window.confidence.isEnabled() and window.tracker.isEnabled()
        window.pause_button.click()
        wait_ui(app, lambda: window._snapshot.paused)
        assert window.step_button.isEnabled()
        window.seek_slider.setValue(round(5 / 7 * SEEK_STEPS))
        wait_ui(app, lambda: window._last_index == 5)
        assert window._snapshot.paused
        window.step_button.click()
        wait_ui(app, lambda: window._last_index == 6)
        window.confidence.setValue(0.65)
        assert window.confidence_slider.value() == 65
        window.tracker.setCurrentText('ByteTrack')
        window.pause_button.click()
        wait_ui(app, lambda: window._snapshot.state == State.COMPLETED)
        assert window.start_button.text() == 'Restart'
        assert not window.preview._image.isNull()
        assert window.seek_slider.isEnabled()
        window.start_button.click()
        wait_ui(app, lambda: window._last_index == 0)
        session = window.session
        window.close()
        wait_ui(app, lambda: not window.isVisible())
        assert not session.is_alive()
        assert window.session is None
    finally:
        if window.session is not None:
            window.session.stop()
            window.session.join(4)
        window.close()


def test_drag_seeks_only_on_release_and_stop_unlocks_inputs(app, model_path, video_path):
    window = MainWindow(session_factory=partial(Session, engine_factory=PreviewEngine))
    window.model_path.setText(model_path)
    window.source_type.setCurrentIndex(window.source_type.findData(SourceKind.VIDEO))
    window.video_path.setText(video_path)
    window.show()
    try:
        window.start_button.click()
        wait_ui(app, lambda: window.seek_slider.isEnabled())
        window.pause_button.click()
        wait_ui(app, lambda: window._snapshot.paused)
        generation = window._snapshot.generation
        window.seek_slider.setSliderDown(True)
        window.seek_slider.setValue(SEEK_STEPS)
        window._poll()
        assert window._snapshot.generation == generation
        window.seek_slider.setSliderDown(False)  # emits sliderReleased
        wait_ui(app, lambda: window._last_index == 7)
        assert window._snapshot.generation == generation + 1
        QTest.keyClick(window.seek_slider, Qt.Key.Key_Left)
        wait_ui(app, lambda: window._snapshot.generation == generation + 2)
        window.stop_button.click()
        wait_ui(app, lambda: window.session is None)
        assert window.input_group.isEnabled()
        assert not window.step_button.isEnabled()
    finally:
        if window.session is not None:
            window.session.stop()
            window.session.join(4)
        window.close()


def test_invalid_model_and_live_controls(app):
    window = MainWindow()
    try:
        assert not window.video_controls.isVisible()
        window.start_button.click()
        assert 'existing Ultralytics' in window.status.text()
        assert window.session is None
        assert not window.pause_button.isEnabled()
        assert not window.step_button.isEnabled()
    finally:
        window.close()


def test_format_time():
    assert format_time(None) == '--:--'
    assert format_time(0.125) == '00:00.125'
    assert format_time(3661.999) == '01:01:01.999'
