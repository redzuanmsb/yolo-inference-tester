import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from inference_tester.config import InferenceDevice, Options
from inference_tester.engine import YoloEngine, intel_igpu_device


class ModelSpy:
    task = 'detect'

    def __init__(self, path):
        self.path = path
        self.predictor = None
        self.calls = []
        self.resets = 0
        self.devices = []

    def to(self, device):
        self.devices.append(device)

    def reset_callbacks(self):
        self.resets += 1

    def _result(self, image, method, kwargs):
        self.calls.append((method, kwargs))
        self.predictor = SimpleNamespace()
        return [
            SimpleNamespace(
                speed={'inference': 12.5},
                boxes=[1, 2],
                plot=lambda **_: image.copy(),
            )
        ]

    def predict(self, image, **kwargs):
        return self._result(image, 'predict', kwargs)

    def track(self, image, **kwargs):
        return self._result(image, 'track', kwargs)


def test_detection_and_tracking_forward_cpu_confidence_and_reset(monkeypatch):
    engine = YoloEngine('test.pt', model_factory=ModelSpy)
    monkeypatch.setattr(engine, '_tracker_path', lambda name: name + '.yaml')
    image = np.zeros((64, 96, 3), dtype=np.uint8)
    result = engine.process(image, Options(0.42))
    assert result.inference_ms == 12.5
    assert result.detections == 2
    method, kwargs = engine.model.calls[-1]
    assert method == 'predict'
    assert kwargs['device'] == 'cpu'
    assert kwargs['conf'] == 0.42
    assert kwargs['imgsz'] == 640
    assert kwargs['quantize'] is None
    assert 'half' not in kwargs
    assert engine.model.devices == ['cpu']

    engine.process(image, Options(0.73, 'ByteTrack'))
    assert engine.model.resets == 2
    method, kwargs = engine.model.calls[-1]
    assert method == 'track'
    assert kwargs['persist'] is True
    assert kwargs['tracker'] == 'ByteTrack.yaml'
    assert kwargs['conf'] == 0.73
    assert kwargs['device'] == 'cpu'
    engine.process(image, Options(0.8, 'ByteTrack'))
    assert engine.model.resets == 2  # confidence changes preserve tracks
    engine.process(image, Options(0.8, 'BoT-SORT'))
    assert engine.model.resets == 3
    engine.process(image, Options())
    assert engine.model.calls[-1][0] == 'predict'
    assert engine.model.resets == 4
    engine.reset()
    assert engine.model.predictor is None


def test_reset_removes_feature_hook():
    engine = YoloEngine('test.pt', model_factory=ModelSpy)
    removed = []
    engine.model.predictor = SimpleNamespace(
        _hook=SimpleNamespace(remove=lambda: removed.append(1))
    )
    engine.reset()
    assert removed == [1]


def test_wrong_model_task_is_actionable():
    class Classifier(ModelSpy):
        task = 'classify'

    with pytest.raises(ValueError, match='object-detection'):
        YoloEngine('test.pt', model_factory=Classifier)


def test_invalid_checkpoint_error_has_guidance():
    def broken_model(_):
        raise RuntimeError('invalid checkpoint')

    with pytest.raises(ValueError, match='Select an Ultralytics YOLO'):
        YoloEngine('test.pt', model_factory=broken_model)


@pytest.mark.parametrize('value', [0, -0.5, 1.1, float('nan')])
def test_invalid_confidence_is_rejected(value):
    with pytest.raises(ValueError, match='Confidence'):
        Options(value)


def test_igpu_selection_ignores_discrete_gpu_and_never_falls_back(monkeypatch):
    devices = {'GPU.0': 'DISCRETE', 'GPU.1': 'INTEGRATED'}
    core = SimpleNamespace(
        available_devices=['CPU', *devices],
        get_property=lambda name, prop: devices[name],
    )
    monkeypatch.setitem(sys.modules, 'openvino', SimpleNamespace(Core=lambda: core))
    assert intel_igpu_device() == 'intel:gpu.1'
    devices['GPU.1'] = 'DISCRETE'
    with pytest.raises(RuntimeError, match='integrated GPU'):
        intel_igpu_device()


def test_igpu_converts_copy_and_retains_export_until_close(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'original.pt'
    checkpoint.write_bytes(b'weights')
    exports = []

    class ExportSpy(ModelSpy):
        def __init__(self, path, **kwargs):
            super().__init__(path)

        def export(self, **kwargs):
            exports.append((self.path, kwargs))
            output = Path(self.path).parent / 'model_openvino_model'
            output.mkdir()
            return output

    monkeypatch.setattr('inference_tester.engine.intel_igpu_device', lambda: 'intel:gpu.1')
    engine = YoloEngine(str(checkpoint), device=InferenceDevice.INTEL_IGPU, model_factory=ExportSpy)
    output = Path(engine.model.path)
    try:
        copied, kwargs = exports[0]
        assert Path(copied).parent != tmp_path
        assert Path(copied).read_bytes() == b'weights'
        assert kwargs['quantize'] == 16
        assert kwargs['dynamic'] is False
        assert kwargs['device'] == 'cpu'
        assert engine.model.devices == []  # exported models cannot use .to()
        engine.process(np.zeros((64, 96, 3), dtype=np.uint8), Options())
        assert engine.model.calls[-1][1]['device'] == 'intel:gpu.1'
        engine.reset()
        assert output.is_dir()
        assert list(tmp_path.iterdir()) == [checkpoint]
    finally:
        engine.close()
    assert not output.parent.exists()


def test_failed_conversion_removes_temporary_files(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'model.pt'
    checkpoint.write_bytes(b'weights')
    copies = []

    class BrokenExport(ModelSpy):
        def export(self, **kwargs):
            copies.append(Path(self.path))
            raise RuntimeError('unsupported operator')

    monkeypatch.setattr('inference_tester.engine.intel_igpu_device', lambda: 'intel:gpu')
    with pytest.raises(ValueError, match='unsupported operator'):
        YoloEngine(str(checkpoint), device=InferenceDevice.INTEL_IGPU, model_factory=BrokenExport)
    assert not copies[0].parent.exists()
    assert checkpoint.read_bytes() == b'weights'
