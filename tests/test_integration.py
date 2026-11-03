import numpy as np
import pytest

from inference_tester.config import TRACKERS, InferenceDevice, Options
from inference_tester.engine import YoloEngine


@pytest.mark.integration
def test_real_cpu_checkpoint_inference_and_every_tracker(tmp_path):
    import torch
    from ultralytics import YOLO
    from ultralytics.engine.results import Boxes

    assert torch.version.cuda is None
    # Exercise the actual .pt loading path without downloading pretrained weights.
    path = tmp_path / 'random-model.pt'
    YOLO('yolo11n.yaml').save(path)
    engine = YoloEngine(str(path))
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    try:
        for name, filename in TRACKERS.items():
            result = engine.process(image, Options(0.25, name))
            assert result.image.shape == image.shape
            assert result.inference_ms > 0
            assert next(engine.model.model.parameters()).device.type == 'cpu'
            assert next(engine.model.model.parameters()).dtype == torch.float32
            if filename is not None:
                tracker = engine.model.predictor.trackers[0]
                assert tracker.args.tracker_type == filename.removesuffix('.yaml')
                assert not getattr(tracker.args, 'with_reid', False)
                # Feed real trackers confident boxes, as random weights normally
                # produce no detections. This also exercises their association code.
                boxes = Boxes(np.array([[40, 40, 90, 90, 0.95, 0]], dtype=np.float32), (256, 256))
                for _ in range(3):
                    tracker.update(boxes, image)
                assert tracker.frame_id == 4
                engine.reset()
        engine.process(image, Options())
        assert not getattr(engine.model.predictor, 'trackers', [])
    finally:
        engine.close()


@pytest.mark.integration
@pytest.mark.parametrize('target', ['cpu', 'igpu'])
def test_real_openvino_conversion_and_tracking(tmp_path, monkeypatch, target):
    from ultralytics import YOLO

    from inference_tester.engine import intel_igpu_device

    path = tmp_path / 'random-model.pt'
    YOLO('yolo11n.yaml').save(path)
    # Also exercise conversion on CPU-only hosts, without claiming GPU coverage.
    if target == 'cpu':
        monkeypatch.setattr('inference_tester.engine.intel_igpu_device', lambda: 'intel:cpu')
    else:
        try:
            intel_igpu_device()
        except RuntimeError as exc:
            pytest.skip(str(exc))
    engine = YoloEngine(str(path), device=InferenceDevice.INTEL_IGPU)
    try:
        image = np.zeros((256, 384, 3), dtype=np.uint8)
        for options in (Options(), Options(tracker='ByteTrack'), Options()):
            result = engine.process(image, options)
            assert result.image.shape == image.shape
            assert result.inference_ms > 0
            if target == 'igpu':
                backend = engine.model.predictor.model
                devices = backend.ov_compiled_model.get_property('EXECUTION_DEVICES')
                assert devices and all(str(d).startswith('GPU') for d in devices)
        assert not list(tmp_path.glob('*openvino*'))
    finally:
        engine.close()
