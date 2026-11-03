import sys

from PySide6.QtCore import QRect, Qt, QTimer
from PySide6.QtGui import QCloseEvent, QImage, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from inference_tester.config import (
    TRACKERS,
    InferenceDevice,
    Options,
    SessionConfig,
    Source,
    SourceKind,
)
from inference_tester.session import Session, State

SEEK_STEPS = 10000


def format_time(seconds: float | None) -> str:
    if seconds is None:
        return '--:--'
    milliseconds = max(0, round(seconds * 1000))
    seconds, ms = divmod(milliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    prefix = f'{hours:02}:' if hours else ''
    return f'{prefix}{minutes:02}:{seconds:02}.{ms:03}'


class Preview(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._image = QImage()
        self.setMinimumSize(480, 270)

    def set_frame(self, image) -> None:
        height, width = image.shape[:2]
        self._image = QImage(
            image.data, width, height, image.strides[0], QImage.Format.Format_BGR888
        ).copy()
        self.update()

    def clear(self) -> None:
        self._image = QImage()
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        if self._image.isNull():
            painter.setPen(Qt.GlobalColor.lightGray)
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                'Select a YOLO .pt model and a source, then click Start.',
            )
            return
        size = self._image.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        rect = QRect(
            (self.width() - size.width()) // 2,
            (self.height() - size.height()) // 2,
            size.width(),
            size.height(),
        )
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(rect, self._image)


class MainWindow(QMainWindow):
    def __init__(self, *, session_factory=Session) -> None:
        super().__init__()
        self.setWindowTitle('YOLO Inference Tester')
        self.resize(1120, 820)
        self._session_factory = session_factory
        self.session: Session | None = None
        self._snapshot = None
        self._closing = False
        self._updating_seek = False
        self._last_index = 0
        self._build_ui()
        self._timer = QTimer(self)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._poll)
        self._timer.start()
        self._source_changed()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(10)

        self.input_group = QGroupBox('Model and source')
        inputs = QGridLayout(self.input_group)
        self.model_path = QLineEdit()
        self.model_path.setReadOnly(True)
        self.model_path.setPlaceholderText('Select an Ultralytics object-detection model (.pt)')
        self.model_button = QPushButton('Select .pt…')
        self.model_button.clicked.connect(self._choose_model)
        inputs.addWidget(QLabel('Model'), 0, 0)
        inputs.addWidget(self.model_path, 0, 1)
        inputs.addWidget(self.model_button, 0, 2)

        self.source_type = QComboBox()
        for kind in SourceKind:
            self.source_type.addItem(kind.value, kind)
        self.source_type.currentIndexChanged.connect(self._source_changed)
        inputs.addWidget(self.source_type, 1, 0)
        self.source_stack = QStackedWidget()
        self.camera_index = QSpinBox()
        self.camera_index.setRange(0, 99)
        self.camera_index.setPrefix('Camera index: ')
        self.source_stack.addWidget(self.camera_index)

        video_widget = QWidget()
        video_layout = QHBoxLayout(video_widget)
        video_layout.setContentsMargins(0, 0, 0, 0)
        self.video_path = QLineEdit()
        self.video_path.setReadOnly(True)
        self.video_path.setPlaceholderText('Select a video file')
        video_button = QPushButton('Select video…')
        video_button.clicked.connect(self._choose_video)
        video_layout.addWidget(self.video_path, 1)
        video_layout.addWidget(video_button)
        self.source_stack.addWidget(video_widget)
        self.rtsp_url = QLineEdit()
        self.rtsp_url.setPlaceholderText('rtsp://[user:password@]camera:554/stream')
        self.rtsp_url.setClearButtonEnabled(True)
        self.source_stack.addWidget(self.rtsp_url)
        inputs.addWidget(self.source_stack, 1, 1, 1, 2)
        self.device = QComboBox()
        for device in InferenceDevice:
            self.device.addItem(device.value, device)
        self.device.setToolTip(
            'Intel iGPU converts a temporary copy of the .pt model to OpenVINO FP16. '
            'First startup can take longer. Stop before changing devices.'
        )
        inputs.addWidget(QLabel('Inference device'), 2, 0)
        inputs.addWidget(self.device, 2, 1, 1, 2)
        inputs.setColumnStretch(1, 1)
        layout.addWidget(self.input_group)

        inference_group = QGroupBox('Inference')
        inference = QHBoxLayout(inference_group)
        inference.addWidget(QLabel('Confidence'))
        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setRange(1, 100)
        self.confidence_slider.setValue(25)
        self.confidence_slider.setMaximumWidth(240)
        self.confidence_slider.setAccessibleName('Confidence threshold')
        self.confidence = QDoubleSpinBox()
        self.confidence.setRange(0.01, 1.00)
        self.confidence.setDecimals(2)
        self.confidence.setSingleStep(0.01)
        self.confidence.setValue(0.25)
        self.confidence_slider.valueChanged.connect(
            lambda value: self.confidence.setValue(value / 100)
        )
        self.confidence.valueChanged.connect(self._confidence_changed)
        inference.addWidget(self.confidence_slider, 1)
        inference.addWidget(self.confidence)
        inference.addSpacing(20)
        inference.addWidget(QLabel('Tracker'))
        self.tracker = QComboBox()
        self.tracker.addItems(list(TRACKERS))
        self.tracker.currentTextChanged.connect(self._options_changed)
        self.tracker.setToolTip('Changing trackers starts fresh track IDs on the next frame.')
        inference.addWidget(self.tracker)
        inference.addStretch(1)
        inference.addWidget(QLabel('Image size 640  •  All classes'))
        layout.addWidget(inference_group)

        controls = QHBoxLayout()
        self.start_button = QPushButton('Start')
        self.start_button.clicked.connect(self._start)
        self.stop_button = QPushButton('Stop')
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop)
        self.pause_button = QPushButton('Pause')
        self.pause_button.setEnabled(False)
        self.pause_button.clicked.connect(self._pause)
        self.step_button = QPushButton('Next frame')
        self.step_button.setEnabled(False)
        self.step_button.clicked.connect(self._step)
        for button in (
            self.start_button,
            self.stop_button,
            self.pause_button,
            self.step_button,
        ):
            controls.addWidget(button)
        controls.addStretch(1)
        self.metrics = QLabel('Processed FPS: —    Inference: —    Detections: —')
        controls.addWidget(self.metrics)
        layout.addLayout(controls)

        self.preview = Preview()
        layout.addWidget(self.preview, 1)

        self.video_controls = QWidget()
        timeline = QVBoxLayout(self.video_controls)
        timeline.setContentsMargins(0, 0, 0, 0)
        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setRange(0, SEEK_STEPS)
        self.seek_slider.setEnabled(False)
        self.seek_slider.setAccessibleName('Video position')
        self.seek_slider.valueChanged.connect(self._seek_changed)
        self.seek_slider.sliderReleased.connect(self._commit_seek)
        timeline.addWidget(self.seek_slider)
        self.position = QLabel('Position: --:-- / --:--')
        timeline.addWidget(self.position)
        layout.addWidget(self.video_controls)

        self.status = QLabel('Ready. Select a model, source, and inference device.')
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.status)

    def _source_changed(self, *_args) -> None:
        self.source_stack.setCurrentIndex(self.source_type.currentIndex())
        self.video_controls.setVisible(self.source_type.currentData() == SourceKind.VIDEO)

    def _choose_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, 'Select YOLO model', self.model_path.text(), 'YOLO weights (*.pt)'
        )
        if path:
            self.model_path.setText(path)

    def _choose_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            'Select video',
            self.video_path.text(),
            'Video files (*.mp4 *.avi *.mkv *.mov *.m4v *.wmv *.webm *.ts);;All files (*)',
        )
        if path:
            self.video_path.setText(path)

    def _options(self) -> Options:
        return Options(self.confidence.value(), self.tracker.currentText())

    def _confidence_changed(self, value: float) -> None:
        self.confidence_slider.setValue(round(value * 100))
        self._options_changed()

    def _options_changed(self, *_args) -> None:
        if self.session is not None:
            self.session.set_options(self._options())

    def _start(self) -> None:
        if self.session is not None and self.session.is_alive():
            if self._snapshot is not None and self._snapshot.state == State.COMPLETED:
                self.session.set_paused(False)
                self._poll()
            return
        kind = self.source_type.currentData()
        value = {
            SourceKind.CAMERA: self.camera_index.value(),
            SourceKind.VIDEO: self.video_path.text(),
            SourceKind.RTSP: self.rtsp_url.text().strip(),
        }[kind]
        config = SessionConfig(
            self.model_path.text(), Source(kind, value), InferenceDevice(self.device.currentData())
        )
        try:
            config.validate()
        except ValueError as exc:
            self._show_status(str(exc), error=True)
            return
        self.preview.clear()
        self.metrics.setText('Processed FPS: —    Inference: —    Detections: —')
        self.position.setText('Position: --:-- / --:--')
        self._last_index = 0
        self.session = self._session_factory(config, self._options())
        self.session.start()
        self._poll()

    def _stop(self) -> None:
        if self.session is not None:
            self.session.stop()
            self._poll()

    def _pause(self) -> None:
        if self.session is not None and self._snapshot is not None:
            self.session.set_paused(not self._snapshot.paused)
            self._poll()

    def _step(self) -> None:
        if self.session is not None:
            self.session.step()
            self._poll()

    def _seek_changed(self, value: int) -> None:
        if self._updating_seek or self._snapshot is None:
            return
        info = self._snapshot.info
        if not info.seekable:
            return
        index = round(value / SEEK_STEPS * (info.frame_count - 1))
        self.position.setText(f'Seek to {format_time(index / info.fps)}')
        if not self.seek_slider.isSliderDown():
            self._commit_seek()

    def _commit_seek(self) -> None:
        if self.session is None or self._snapshot is None:
            return
        info = self._snapshot.info
        if info.seekable:
            index = round(self.seek_slider.value() / SEEK_STEPS * (info.frame_count - 1))
            self.session.seek(index)
            self._poll()

    def _show_status(self, message: str, *, error: bool = False) -> None:
        self.status.setText(message)
        self.status.setStyleSheet('color: #c63838;' if error else '')

    def _poll(self) -> None:
        if self.session is None:
            return
        snapshot = self.session.poll()
        self._snapshot = snapshot
        frame = snapshot.frame
        if frame is not None and frame.generation == snapshot.generation:
            self.preview.set_frame(frame.image)
            self._last_index = frame.index
            self.metrics.setText(
                f'Processed FPS: {frame.fps:.1f}    '
                f'Inference: {frame.inference_ms:.1f} ms    '
                f'Detections: {frame.detections}'
            )
            if self.session.is_video and not self.seek_slider.isSliderDown():
                info = snapshot.info
                count = f' / {info.frame_count}' if info.frame_count is not None else ''
                duration = info.frame_count / info.fps if info.seekable else None
                self.position.setText(
                    f'{format_time(frame.timestamp)} / {format_time(duration)}'
                    f'    •    Frame {frame.index + 1}{count}'
                )
                if info.seekable:
                    self._updating_seek = True
                    self.seek_slider.setValue(
                        round(frame.index / (info.frame_count - 1) * SEEK_STEPS)
                    )
                    self._updating_seek = False
        alive = self.session.is_alive()
        ready = snapshot.state in {
            State.RUNNING,
            State.PAUSED,
            State.COMPLETED,
            State.SEEKING,
        }
        completed = snapshot.state == State.COMPLETED
        self.input_group.setEnabled(not alive)
        self.start_button.setEnabled(not alive or completed)
        self.start_button.setText('Restart' if completed else 'Start')
        self.stop_button.setEnabled(alive and snapshot.state != State.STOPPING)
        self.pause_button.setEnabled(alive and ready and self.session.is_video and not completed)
        self.pause_button.setText('Resume' if snapshot.paused else 'Pause')
        self.step_button.setEnabled(
            alive and ready and self.session.is_video and snapshot.paused and not completed
        )
        self.seek_slider.setEnabled(
            alive and ready and self.session.is_video and snapshot.info.seekable
        )
        if snapshot.error:
            self._show_status(snapshot.error, error=True)
        elif snapshot.state == State.COMPLETED:
            self._show_status(
                'Video completed. Seek to review, click Restart, or Stop to change inputs.'
            )
        elif snapshot.state == State.STOPPING:
            self._show_status('Stopping… waiting for the current inference or capture to finish.')
        else:
            self._show_status(snapshot.state.value)
        if not alive:
            self.session.join()
            self.session = None
            if self._closing:
                QTimer.singleShot(0, self.close)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.session is not None and self.session.is_alive():
            self._closing = True
            self.session.stop()
            self.centralWidget().setEnabled(False)
            self._show_status('Closing… waiting for the current inference or capture to finish.')
            event.ignore()
        else:
            self._timer.stop()
            event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName('YOLO Inference Tester')
    app.setOrganizationName('YOLO Inference Tester')
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    return app.exec()
