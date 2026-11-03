import os
from pathlib import Path

import cv2
import numpy as np
import pytest

# UI tests run without opening windows. A separate manual check uses the desktop.
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ['YOLO_AUTOINSTALL'] = 'false'


@pytest.fixture
def model_path(tmp_path: Path) -> str:
    path = tmp_path / 'test.pt'
    path.write_bytes(b'controlled test model')
    return str(path)


@pytest.fixture
def video_path(tmp_path: Path) -> str:
    path = tmp_path / 'clip.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (96, 64))
    assert writer.isOpened()
    try:
        for index in range(8):
            image = np.full((64, 96, 3), index * 25, dtype=np.uint8)
            cv2.rectangle(image, (5 + index * 4, 15), (25 + index * 4, 35), (0, 180, 255), -1)
            writer.write(image)
    finally:
        writer.release()
    return str(path)
