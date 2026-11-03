"""The only module that owns Ultralytics model and tracker state."""

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

import numpy as np

from inference_tester.config import TRACKERS, InferenceDevice, Options


def intel_igpu_device() -> str:
    """Select an integrated GPU explicitly; never use OpenVINO's AUTO fallback."""
    try:
        import openvino as ov

        core = ov.Core()
        for name in core.available_devices:
            if name.startswith('GPU'):
                kind = str(core.get_property(name, 'DEVICE_TYPE')).upper()
                if kind.endswith('INTEGRATED'):
                    return f'intel:{name.lower()}'
    except Exception as exc:
        raise RuntimeError(
            'Cannot initialize Intel iGPU inference. Run uv sync and install the '
            f'Intel graphics driver with OpenVINO support. Details: {exc}'
        ) from exc
    raise RuntimeError(
        'OpenVINO did not find a supported Intel integrated GPU. Install the Intel '
        'graphics driver, enable the iGPU, or select CPU.'
    )


@dataclass(frozen=True)
class Prediction:
    image: np.ndarray
    inference_ms: float
    processing_ms: float
    detections: int


class YoloEngine:
    def __init__(self, model_path: str, *, device=InferenceDevice.CPU, model_factory=None) -> None:
        # Dependencies are resolved by uv, never by a tracker at runtime.
        os.environ['YOLO_AUTOINSTALL'] = 'false'
        self._temporary: TemporaryDirectory | None = None
        self._device = intel_igpu_device() if device == InferenceDevice.INTEL_IGPU else 'cpu'
        if model_factory is None:
            from ultralytics import YOLO

            model_factory = YOLO
        try:
            self.model = model_factory(model_path)
        except Exception as exc:
            raise ValueError(
                'Cannot load this .pt model. Select an Ultralytics YOLO object-detection '
                f'checkpoint supported by this version. Details: {str(exc)[:800]}'
            ) from exc
        if self.model.task != 'detect':
            raise ValueError(
                'Choose a YOLO object-detection model (.pt), not a different model task.'
            )
        if device == InferenceDevice.INTEL_IGPU:
            self._temporary = TemporaryDirectory(prefix='yolo-inference-')
            try:
                # Ultralytics exports beside the checkpoint. Load a copy so neither
                # conversion nor export metadata writes into the user's model folder.
                copied = Path(self._temporary.name) / 'model.pt'
                shutil.copy2(model_path, copied)
                export_model = model_factory(str(copied))
                exported = export_model.export(
                    format='openvino',
                    imgsz=640,
                    batch=1,
                    dynamic=False,
                    quantize=16,
                    device='cpu',
                    nms=False,
                )
                self.model = model_factory(str(exported), task='detect')
            except Exception as exc:
                self._temporary.cleanup()
                self._temporary = None
                raise ValueError(
                    'Cannot convert this .pt model for Intel iGPU inference. '
                    f'Try CPU or a supported YOLO checkpoint. Details: {str(exc)[:800]}'
                ) from exc
        else:
            self.model.to('cpu')
        self._tracker: str | None = None
        self._tracker_paths: dict[str, str] = {}

    def reset(self) -> None:
        """Forget predictor callbacks, hooks, and tracks, retaining the loaded model."""
        predictor = self.model.predictor
        if predictor is not None:
            hook = getattr(predictor, '_hook', None)
            if hook is not None:
                hook.remove()
        self.model.reset_callbacks()
        self.model.predictor = None
        self._tracker = None

    def _tracker_path(self, name: str) -> str:
        if name not in self._tracker_paths:
            from ultralytics.utils import ROOT, YAML

            # Absolute bundled paths avoid accidentally loading a same-named local YAML.
            path = Path(ROOT) / 'cfg' / 'trackers' / TRACKERS[name]
            config = YAML.load(path)
            if config.get('with_reid', False):
                raise ValueError(f'{name} enables ReID; this tester requires ReID disabled.')
            self._tracker_paths[name] = str(path)
        return self._tracker_paths[name]

    def process(self, frame: np.ndarray, options: Options) -> Prediction:
        if self._tracker != options.tracker:
            self.reset()
            self._tracker = options.tracker
        kwargs: dict[str, Any] = {
            'device': self._device,
            'conf': options.confidence,
            'imgsz': 640,
            'quantize': None,
            'verbose': False,
            'save': False,
            'show': False,
        }
        started = perf_counter()
        if options.tracker == 'None':
            results = self.model.predict(frame, **kwargs)
        else:
            results = self.model.track(
                frame, tracker=self._tracker_path(options.tracker), persist=True, **kwargs
            )
        result = results[0]
        image = result.plot(labels=True, conf=True)
        return Prediction(
            image=image,
            inference_ms=float(result.speed.get('inference') or 0),
            processing_ms=(perf_counter() - started) * 1000,
            detections=len(result.boxes) if result.boxes is not None else 0,
        )

    def close(self) -> None:
        try:
            self.reset()
        finally:
            if self._temporary is not None:
                self._temporary.cleanup()
                self._temporary = None
