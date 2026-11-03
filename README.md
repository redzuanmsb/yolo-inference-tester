# YOLO Inference Tester

A general-purpose Windows desktop app for testing Ultralytics YOLO
object-detection `.pt` models and tracking across their trained classes.
Inference runs on **CPU or Intel iGPU**, using PySide6 for the interface and
OpenCV for video capture. No CUDA installation is required.

## Run

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run
these commands in this repository:

```powershell
uv sync
uv run python main.py
```

uv selects Python 3.11 and installs the locked dependencies. PyTorch and
torchvision come from the explicit CPU wheel index. When using Codex, run uv
and environment commands through host-approved execution; do not rebuild an
existing environment or change its permissions.

## Use

1. Click **Select .pt…** and choose your local Ultralytics detection model.
2. Choose **Camera** (device index, initially `0`), **Video file**, or **RTSP stream**.
3. Choose **Inference device**: **CPU** (default) or **Intel iGPU (OpenVINO)**.
   Set **Confidence** (`0.01–1.00`, initially `0.25`) and a **Tracker**, then click **Start**.
4. Adjust confidence or switch trackers while running. Changes apply on the next
   processed frame. Switching trackers starts fresh track IDs without reopening
   the source. While paused, settings apply when another frame is processed.
5. Click **Stop** before choosing another model, source, or inference device.

**Intel iGPU:** Keep selecting your original `.pt` file. On Start, the worker
copies it into a unique `yolo-inference-*` directory in the system temporary
folder and exports a static 640 × 640, batch-1 OpenVINO model with FP16 weights.
The source checkpoint and its folder are untouched. Conversion happens once per
session; seeking and switching trackers reuse the exported files. Normal Stop,
close, or conversion failure removes the temporary directory. Starting another
session converts again. Initial conversion and GPU compilation can take a few
minutes; the interface stays responsive, but Stop waits for conversion to finish.

Install a compatible Intel graphics driver and enable the integrated GPU.
The app selects the integrated device explicitly, including on systems with a
discrete GPU. Missing iGPU support produces an error; choose CPU to continue.
There is no automatic CPU fallback. See the
[Ultralytics OpenVINO guide](https://docs.ultralytics.com/integrations/openvino/).

The preview preserves the source aspect ratio and shows class labels,
confidence, and track IDs when available. The status row reports processed FPS,
model inference time, and the current result count. FPS includes capture and
playback pacing; inference time is the model's inference stage only.

**Video files:** Pause/Resume, advance one frame with Next frame while paused,
or drag and release the timeline to seek. Keyboard timeline changes also seek.
Seeking resets tracking and preserves the playing/paused state. Every frame is
processed in order during normal playback; inference may make playback
slower than the original video. At the end, the last frame stays visible: seek
to review it or click Restart. Seeking requires valid frame-count/FPS metadata;
it is disabled when the decoder cannot provide it. There is no audio playback.

**Live sources:** Camera and RTSP capture keep only the newest pending frame
so slow inference does not build an application-side backlog. They have
Start/Stop controls; pause, step, and seek are for video files. RTSP uses the
OpenCV FFmpeg backend with a 5-second open timeout and a 2-second read timeout.
A disconnected stream stops with an error; click Start to reconnect.

Trackers: None (detection only), ByteTrack, BoT-SORT, OC-SORT, Deep OC-SORT,
FastTracker, and TrackTrack. They use the locked Ultralytics package's bundled
configurations with ReID disabled, without downloading another model. Tracker
association thresholds also affect which detections become tracks; a tracking
result can differ from detection-only output at the same confidence setting.
See the [Ultralytics tracking guide](https://docs.ultralytics.com/modes/track).

The input model must be a supported Ultralytics **object-detection** checkpoint;
classification, pose, segmentation, and other tasks are not supported by this
tester. No model is bundled or downloaded automatically. Inference uses image
size `640` and all model classes. CPU uses full precision; Intel iGPU uses
an OpenVINO export with FP16 weights. Settings and RTSP URLs are
session-only. `.pt` files are ignored by Git, including in subdirectories.

This app does not train models, record/export results, or build an installer.

## Development checks

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

The tests use generated videos and controlled captures for playback and failure
cases. CPU integration tests initialize a small YOLO model from its bundled
architecture (random weights, no download) and exercise every tracker.
OpenVINO integration tests also export and run a random-weight model on CPU and,
when available, the Intel iGPU, checking the GPU execution device. These
checks validate execution and lifecycle behavior, not detection accuracy or
real camera/RTSP compatibility. Test your actual model and sources in the app.
