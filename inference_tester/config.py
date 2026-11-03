from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from urllib.parse import urlsplit


class SourceKind(StrEnum):
    CAMERA = 'Camera'
    VIDEO = 'Video file'
    RTSP = 'RTSP stream'


class InferenceDevice(StrEnum):
    CPU = 'CPU'
    INTEL_IGPU = 'Intel iGPU (OpenVINO)'


TRACKERS: dict[str, str | None] = {
    'None': None,
    'ByteTrack': 'bytetrack.yaml',
    'BoT-SORT': 'botsort.yaml',
    'OC-SORT': 'ocsort.yaml',
    'Deep OC-SORT': 'deepocsort.yaml',
    'FastTracker': 'fasttrack.yaml',
    'TrackTrack': 'tracktrack.yaml',
}


@dataclass(frozen=True)
class Source:
    kind: SourceKind
    value: str | int

    def validate(self) -> None:
        if self.kind == SourceKind.CAMERA:
            if not isinstance(self.value, int) or self.value < 0:
                raise ValueError('Choose a camera index of 0 or greater.')
        elif self.kind == SourceKind.VIDEO:
            if not isinstance(self.value, str) or not Path(self.value).is_file():
                raise ValueError('Select an existing video file.')
        else:
            try:
                url = urlsplit(str(self.value))
                valid = url.scheme.lower() == 'rtsp' and bool(url.hostname)
                _ = url.port
            except ValueError:
                valid = False
            if not valid:
                raise ValueError('Enter an RTSP URL, for example rtsp://camera:554/stream.')


@dataclass(frozen=True)
class Options:
    confidence: float = 0.25
    tracker: str = 'None'

    def __post_init__(self) -> None:
        if not 0.01 <= self.confidence <= 1.0:
            raise ValueError('Confidence must be between 0.01 and 1.00.')
        if self.tracker not in TRACKERS:
            raise ValueError(f'Unknown tracker: {self.tracker}')


@dataclass(frozen=True)
class SessionConfig:
    model_path: str
    source: Source
    device: InferenceDevice = InferenceDevice.CPU

    def validate(self) -> None:
        if not isinstance(self.device, InferenceDevice):
            raise ValueError('Choose a supported inference device.')
        path = Path(self.model_path)
        if not path.is_file() or path.suffix.lower() != '.pt':
            raise ValueError('Select an existing Ultralytics YOLO .pt model file.')
        self.source.validate()
