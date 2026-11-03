"""YOLO inference tester for CPU and Intel integrated GPUs."""

import os
from pathlib import Path

# Keep library settings/cache local to this checkout instead of changing a user's
# shared Ultralytics settings. Explicit environment overrides remain supported.
os.environ.setdefault(
    'YOLO_CONFIG_DIR', str(Path(__file__).resolve().parents[1] / '.cache' / 'ultralytics')
)
os.environ['YOLO_AUTOINSTALL'] = 'false'
